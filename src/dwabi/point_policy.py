"""Fit a guarded WAPE point forecast from completed, originally issued cases."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from operator import index

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


_FLOOR = 1e-12
_TOL = 1e-11


@dataclass(frozen=True)
class PointComparison:
    """One mature comparison of actions fit at the original issue."""

    origin: int | str
    issued_at: int
    available_at: int
    reference_wape: float
    peer_wape: float


@dataclass(frozen=True)
class PointDecision:
    """Frozen point actions and their evidence, in the original member order."""

    models: tuple[str, ...]
    issue_at: int
    ordinal_weights: tuple[int, ...] | None
    ordinal_origins: tuple[int | str, ...]
    accept: bool
    operator: str
    members: tuple[str, ...]
    peer_count: int
    comparisons: tuple[PointComparison, ...]

    def __post_init__(self):
        # A JSON round trip preserves the ordering and nested comparison records.
        for name in ("models", "ordinal_origins", "members"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.ordinal_weights is not None:
            object.__setattr__(self, "ordinal_weights", tuple(self.ordinal_weights))
        object.__setattr__(
            self,
            "comparisons",
            tuple(
                PointComparison(**row) if isinstance(row, dict) else row
                for row in self.comparisons
            ),
        )


@dataclass(frozen=True)
class _Issue:
    origin: int | str
    issued_at: int
    first_target_at: int
    available_at: int
    cost: np.ndarray | None
    constant: float | None
    menu_wape: np.ndarray | None
    reference_wape: float | None


def _mature(rows, boundary):
    eligible = sorted(
        (
            row
            for row in rows
            if row.issued_at < boundary and row.available_at < boundary
        ),
        key=lambda row: (
            row.available_at,
            row.issued_at,
            isinstance(row.origin, str),
            row.origin,
        ),
    )
    selected = []
    for row in eligible:
        if not selected or row.first_target_at > selected[-1].available_at:
            selected.append(row)
    return selected


@lru_cache(maxsize=1)
def _median_patterns(size):
    shares = 3 * size
    cut_space = range(shares + size - 1)
    cuts = np.array(list(combinations(cut_space, size - 1)), dtype=np.int16)
    borders = np.column_stack(
        (np.full(len(cuts), -1), cuts, np.full(len(cuts), len(cut_space)))
    )
    allocations = np.diff(borders, axis=1).astype(np.int16) - 1
    order = np.argsort(np.abs(allocations - 3).sum(axis=1), kind="stable")
    allocations = allocations[order]
    subsets = ((np.arange(1 << size)[:, None] >> np.arange(size)) & 1).astype(np.int16)
    majority = (allocations @ subsets.T) >= (shares + 1) // 2
    _, canonical = np.unique(np.packbits(majority, axis=1), axis=0, return_index=True)
    keep = np.sort(canonical)
    return allocations[keep], majority[keep].astype(float)


def _subset_lookup(menu, size):
    count = 1 << size
    slots = np.full(count, -1, dtype=np.int64)
    for slot, (mask, method) in enumerate(menu):
        if method == "median":
            slots[mask] = slot
    low = np.zeros(count, dtype=np.int64)
    high = np.zeros(count, dtype=np.int64)
    bit = np.zeros(count, dtype=np.int64)
    depth = np.zeros(count, dtype=np.int64)
    for mask in range(1, count):
        positions = [j for j in range(size) if mask & (1 << j)]
        depth[mask] = len(positions)
        bit[mask] = positions[0]
        low[mask] = positions[(len(positions) - 1) // 2]
        high[mask] = positions[len(positions) // 2]
    return slots, depth, bit, low, high


def _subset_risks(points, actual, mass, order, slots, depth, bit, low, high):
    size = points.shape[1]
    totals = np.zeros(2 * (slots >= 0).sum(), dtype=np.float64)
    ranked_bits = np.empty(size, dtype=np.int64)
    sums = np.zeros(len(slots), dtype=np.float64)
    ranks = np.zeros(len(slots), dtype=np.int64)
    for row in range(len(actual)):
        value, weight = actual[row], mass[row]
        p, sorted_by = points[row], order[row]
        for j in range(size):
            ranked_bits[sorted_by[j]] = 1 << j
        for mask in range(1, len(slots)):
            parent = mask & (mask - 1)
            model = bit[mask]
            sums[mask] = sums[parent] + p[model]
            ranks[mask] = ranks[parent] | ranked_bits[model]
            slot = slots[mask]
            if slot >= 0:
                rank_mask = ranks[mask]
                median = (
                    p[sorted_by[low[rank_mask]]] + p[sorted_by[high[rank_mask]]]
                ) * 0.5
                totals[slot] += weight * abs(value - median)
                totals[slot + 1] += weight * abs(value - sums[mask] / depth[mask])
    return totals


_compiled_subset_risks = None if njit is None else njit(_subset_risks)


def _numpy_subset_risks(points, actual, mass, menu):
    totals = np.empty(len(menu))
    for slot in range(0, len(menu), 2):
        mask, _ = menu[slot]
        members = [j for j in range(points.shape[-1]) if mask & (1 << j)]
        selected = points[:, members]
        totals[slot] = mass @ np.abs(actual - np.median(selected, axis=1))
        totals[slot + 1] = mass @ np.abs(actual - np.mean(selected, axis=1))
    return totals


def _scores(values, width, name, *, signed=False):
    if values is None:
        return None
    vector = np.asarray(values, dtype=float).copy()
    if (
        vector.shape != (width,)
        or not np.isfinite(vector).all()
        or (not signed and (vector < 0.0).any())
    ):
        raise ValueError(f"{name} has invalid values or member geometry.")
    return vector


class PointHistory:
    """Learn the WAPE point action for a small ordered roster of forecasters.

    Each registration records its issue, target start, and full availability clocks.
    Task and family keys identify related histories. The fit uses those keys for
    chronology and source exclusion. Arrays carry the roster on the final axis.
    """

    def __init__(self, models):
        self.models = tuple(models)
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Provide a nonempty roster of distinct model names.")
        size = len(self.models)
        masks = sorted(
            (mask for mask in range(1, 1 << size) if mask.bit_count() >= 3),
            key=lambda mask: (-mask.bit_count(), mask),
        )
        self.menu = tuple(
            (mask, method) for mask in masks for method in ("median", "mean")
        )
        self._lookup = (
            _subset_lookup(self.menu, size)
            if self.menu and _compiled_subset_risks
            else None
        )
        self._records = {}
        self._families = {}

    def observe(
        self,
        task,
        family,
        *,
        origin,
        issued_at,
        first_target_at,
        available_at,
        points,
        actual,
        case_weights,
        include_menu=True,
    ):
        """Register issued points and target values with the caller's native loss mass.

        The loss mass sets the scale of the absolute-error WAPE objective. A caller
        may omit subset scores when its full target support criterion is unmet.
        """
        actual = np.asarray(actual, dtype=float)
        points = np.asarray(points, dtype=float)
        if points.shape != actual.shape + (len(self.models),):
            raise ValueError("Point forecasts must match the target axes and roster.")
        mass = np.broadcast_to(np.asarray(case_weights, dtype=float), actual.shape)
        if not np.isfinite(mass).all() or (mass < 0.0).any():
            raise ValueError("Case weights must be finite and nonnegative.")
        active = np.isfinite(actual) & (mass > 0.0)
        cost = constant = menu_wape = None
        if active.any():
            values, truth, weight = points[active], actual[active], mass[active]
            if not np.isfinite(values).all():
                raise ValueError("Point forecasts must be finite on scored cells.")
            order = np.argsort(values, axis=-1, kind="stable")
            absolute_errors = np.abs(
                truth[:, None] - np.take_along_axis(values, order, axis=-1)
            )
            prefixes = (1 << order).cumsum(axis=-1)[..., :-1]
            weighted_change = weight[:, None] * (
                absolute_errors[:, :-1] - absolute_errors[:, 1:]
            )
            cost = np.bincount(
                prefixes.ravel(),
                weights=weighted_change.ravel(),
                minlength=1 << len(self.models),
            )
            constant = float(weight @ absolute_errors[:, -1])
            if include_menu and self.menu:
                if self._lookup is None:
                    menu_wape = _numpy_subset_risks(values, truth, weight, self.menu)
                else:
                    menu_wape = _compiled_subset_risks(
                        values, truth, weight, order, *self._lookup
                    )
        self.observe_scores(
            task,
            family,
            origin=origin,
            issued_at=issued_at,
            first_target_at=first_target_at,
            available_at=available_at,
            ordinal_cost=cost,
            ordinal_constant=constant,
            menu_wape=menu_wape,
        )

    def observe_scores(
        self,
        task,
        family,
        *,
        origin,
        issued_at,
        first_target_at,
        available_at,
        ordinal_cost,
        ordinal_constant,
        menu_wape,
        reference_wape=None,
    ):
        """Register compact absolute-loss coefficients and per-menu losses.

        `ordinal_cost` indexes prefix masks in member order. A supplied
        `reference_wape` is the actual WAPE of the ordinal action issued at
        `issued_at`. Without it, the fit reconstructs that loss from the
        compact coefficients and the original issue's mature history.
        """
        issued_at, first_target_at, available_at = map(
            index, (issued_at, first_target_at, available_at)
        )
        if not issued_at < first_target_at <= available_at:
            raise ValueError(
                "A target window must start after issue and end at or after its start."
            )
        if not isinstance(origin, str):
            origin = index(origin)
        if task in self._families and self._families[task] != family:
            raise ValueError("A task's physical family must stay consistent.")
        cost = _scores(
            ordinal_cost, 1 << len(self.models), "Ordinal coefficients", signed=True
        )
        if (cost is None) != (ordinal_constant is None):
            raise ValueError(
                "Ordinal coefficients and constant must be supplied together."
            )
        if ordinal_constant is not None and (
            not np.isfinite(ordinal_constant) or ordinal_constant < 0.0
        ):
            raise ValueError("Ordinal constant must be finite and nonnegative.")
        menu_wape = _scores(menu_wape, len(self.menu), "Subset WAPE")
        if reference_wape is not None and (
            not np.isfinite(reference_wape) or reference_wape < 0.0
        ):
            raise ValueError("Issued reference WAPE must be finite and nonnegative.")
        if menu_wape is not None and cost is None and reference_wape is None:
            raise ValueError(
                "Supply issued reference WAPE or ordinal coefficients with subset WAPE."
            )
        if origin in self._records.get(task, {}):
            raise ValueError("A task origin can be registered once.")
        self._records.setdefault(task, {})[origin] = _Issue(
            origin,
            issued_at,
            first_target_at,
            available_at,
            cost,
            ordinal_constant,
            menu_wape,
            reference_wape,
        )
        self._families[task] = family

    def _ordinal(self, rows, issue):
        allocations, profiles = _median_patterns(len(self.models))
        donors = _mature((row for row in rows if row.cost is not None), issue)
        candidate = 0
        if len(donors) > 3:
            costs = np.column_stack([row.cost for row in donors])
            excess = profiles @ costs - profiles[0] @ costs
            risk = (excess.sum(axis=1) - excess.min(axis=1)) / (len(donors) - 1)
        elif len(donors) == 3:
            risk = profiles @ np.mean([row.cost for row in donors], axis=0) + np.mean(
                [row.constant for row in donors]
            )
        if len(donors) >= 3:
            tolerance = 16.0 * np.finfo(float).eps * max(1.0, np.abs(risk).max())
            candidate = int(np.flatnonzero(risk <= risk.min() + tolerance)[0])
        return (
            allocations[candidate],
            profiles[candidate],
            tuple(row.origin for row in donors),
        )

    def _peer(self, family, issue):
        if not self.menu:
            return 0, 0
        families = {}
        for donor, rows in self._records.items():
            source = self._families[donor]
            if source == family:
                continue
            ready = _mature(
                (row for row in rows.values() if row.menu_wape is not None), issue
            )
            if len(ready) >= 2:
                log_means = np.log(
                    np.maximum(
                        np.mean([row.menu_wape for row in ready], axis=0), _FLOOR
                    )
                )
                families.setdefault(source, []).append(log_means - log_means[0])
        if len(families) < 3:
            return 0, len(families)
        by_family = np.stack(
            [np.mean(members, axis=0) for members in families.values()]
        )
        risk = (by_family.sum(axis=0) - by_family.min(axis=0)) / (len(families) - 1)
        return int(np.flatnonzero(risk <= risk.min() + _TOL)[0]), len(families)

    def freeze(self, task, family, *, issue_at):
        """Fit the ordinal action and gate the peer roster at one as-of issue."""
        issue_at = index(issue_at)
        if task in self._families and self._families[task] != family:
            raise ValueError(
                "The requested physical family differs from this task's history."
            )
        rows = tuple(self._records.get(task, {}).values())
        weights, _, origins = self._ordinal(rows, issue_at)
        if not any(row.issued_at < issue_at for row in rows):
            weights = None
        else:
            weights = tuple(map(int, weights))
        selected, sources = self._peer(family, issue_at)
        comparisons = []
        if self.menu and sources >= 3:
            for row in _mature(
                (row for row in rows if row.menu_wape is not None), issue_at
            ):
                earlier, peer_sources = self._peer(family, row.issued_at)
                if peer_sources < 3:
                    continue
                reference = row.reference_wape
                if reference is None:
                    if any(previous.issued_at < row.issued_at for previous in rows):
                        _, profile, _ = self._ordinal(rows, row.issued_at)
                        reference = max(0.0, float(profile @ row.cost + row.constant))
                    else:
                        reference = float(row.menu_wape[0])
                comparisons.append(
                    PointComparison(
                        row.origin,
                        row.issued_at,
                        row.available_at,
                        float(reference),
                        float(row.menu_wape[earlier]),
                    )
                )
        accepted = False
        if len(comparisons) >= 3:
            differences = np.array(
                [row.peer_wape - row.reference_wape for row in comparisons]
            )
            accepted = bool(
                ((differences.sum() - differences) / (len(differences) - 1)).max()
                < -_TOL
            )
        mask, method = (
            self.menu[selected]
            if self.menu
            else ((1 << len(self.models)) - 1, "median")
        )
        members = tuple(name for j, name in enumerate(self.models) if mask & (1 << j))
        return PointDecision(
            self.models,
            issue_at,
            weights,
            origins,
            accepted,
            method,
            members,
            sources,
            tuple(comparisons),
        )


def center(points, decision):
    """Apply a frozen point decision to the ordered model axis of issued points."""
    points = np.asarray(points, dtype=float)
    if (
        not points.ndim
        or points.shape[-1] != len(decision.models)
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "Issued points must have a finite last axis matching the frozen roster."
        )
    if decision.accept:
        selected = points[
            ..., [decision.models.index(name) for name in decision.members]
        ]
        return {"median": np.median, "mean": np.mean}[decision.operator](
            selected, axis=-1
        )
    if decision.ordinal_weights is None:
        return np.median(points, axis=-1)
    weights = np.asarray(decision.ordinal_weights, dtype=float)
    if (
        weights.shape != (len(decision.models),)
        or not np.isfinite(weights).all()
        or (
            (weights < 0.0).any()
            or not np.isfinite(weights.sum())
            or weights.sum() <= 0.0
        )
    ):
        raise ValueError(
            "Ordinal weights must have positive finite mass in the frozen roster order."
        )
    order = np.argsort(points, kind="stable", axis=-1)
    stop = np.argmax(weights[order].cumsum(axis=-1) >= weights.sum() / 2.0, axis=-1)
    chosen = np.take_along_axis(order, stop[..., None], axis=-1)
    return np.take_along_axis(points, chosen, axis=-1)[..., 0]
