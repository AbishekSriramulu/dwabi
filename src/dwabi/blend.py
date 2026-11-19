"""Choose and combine forecasters using their previously issued predictions."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral
from types import MappingProxyType

import numpy as np


_DECILES = tuple(i / 10 for i in range(1, 10))
_GRID = np.asarray([(i, j, 6 - i - j) for i in range(7) for j in range(7 - i)], dtype=float) / 6
_EQUAL_GRID = int(np.flatnonzero((_GRID == 1 / 3).all(axis=1))[0])
_TOLERANCE = 1e-12
_MIN_WINDOWS = 3


@dataclass(frozen=True)
class BlendDecision:
    """A forecast's member order, issued weights and supporting origins."""

    members: tuple[str, ...]
    weights: tuple[float, ...]
    selection_origins: tuple[Hashable, ...]
    weight_origins: tuple[Hashable, ...]
    reason: str


@dataclass(frozen=True)
class BlendForecast:
    """A combined forecast and the member arrays prepared at this issue."""

    quantiles: np.ndarray
    levels: tuple[float, ...]
    members: Mapping[str, np.ndarray]
    issued_at: int
    decision: BlendDecision

    @property
    def point(self) -> np.ndarray:
        """Return the combined median."""
        return self.quantiles[..., self.levels.index(0.5)]


@dataclass
class _Window:
    issued_at: int
    first_target_at: int
    available_at: int
    values: np.ndarray
    outcomes: np.ndarray
    magnitude: np.ndarray
    choice_loss: np.ndarray
    grid_loss: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)


def _clock(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Supply issuance and outcome clocks as integers on one time axis.")
    return int(value)


def _quantiles(values, levels: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim < 2 or not result.size or result.shape[-1] != len(levels):
        raise ValueError("Each member must provide case axes followed by the configured quantile levels.")
    if not np.isfinite(result).all() or (np.diff(result, axis=-1) < 0.).any():
        raise ValueError("Member quantiles must be finite and ordered.")
    return result


def _loss(values: np.ndarray, outcomes: np.ndarray, magnitude: np.ndarray,
          levels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum weighted pinball loss for each candidate on one matched window."""
    scores = np.zeros(len(weights))
    for start in range(0, len(outcomes), 4096):
        rows = slice(start, start + 4096)
        for offset in range(0, len(weights), 64):
            candidates = slice(offset, offset + 64)
            error = outcomes[rows, None, None] - values[rows] @ weights[candidates].T
            pinball = np.maximum(levels * error, (levels - 1) * error)
            scores[candidates] += (magnitude[rows, None, None] * pinball).sum(axis=(0, 1))
    if not np.isfinite(scores).all():
        raise ValueError("The weighted losses must be finite.")
    return scores


class WalkForwardBlend:
    """Choose three forecasters and estimate their weights at each issue.

    Forecasters are a mapping of names to callables. Every callable receives
    the same issue-time context and returns aligned quantile arrays.
    Cacheable forecasts can also be registered directly using observe.
    """

    def __init__(self, models: Mapping[str, Callable], *,
                 levels: Sequence[float] = _DECILES, objective: str = "pinball",
                 selection: str = "joint", lookback: int | None = None, weighted: bool = True):
        self._models = dict(models)
        if len(self._models) < 3 or any(
            not isinstance(name, str) or not name or not callable(model)
            for name, model in self._models.items()
        ):
            raise ValueError("Provide at least three named callable forecasters.")
        self._names = tuple(self._models)
        self.levels = tuple(float(tau) for tau in levels)
        if (0.5 not in self.levels or any(not 0 < tau < 1 for tau in self.levels)
                or any(a >= b for a, b in zip(self.levels, self.levels[1:]))):
            raise ValueError("Quantile levels must be strictly increasing and include 0.5.")
        if objective not in ("pinball", "absolute"):
            raise ValueError("objective must be 'pinball' or 'absolute'.")
        if selection not in ("joint", "individual"):
            raise ValueError("selection must be 'joint' or 'individual'.")
        if lookback is not None and (_clock(lookback) < _MIN_WINDOWS):
            raise ValueError("lookback must allow at least three mature windows.")
        self._point_only = objective == "absolute"
        self._selection = selection
        self._lookback = lookback
        self._weighted = weighted
        self._loss_levels = np.asarray((0.5,) if self._point_only else self.levels)[None, :, None]
        self._triples = tuple(combinations(range(len(self._names)), 3))
        choices = np.zeros((len(self._triples), len(self._names)))
        choices[np.arange(len(self._triples))[:, None], np.asarray(self._triples)] = 1 / 3
        self._choices = np.vstack((np.eye(len(self._names)), choices,
                                  np.full((1, len(self._names)), 1 / len(self._names))))
        self._windows: dict[Hashable, _Window | None] = {}

    def observe(self, forecasts: Mapping[str, np.ndarray], outcomes, *, origin: Hashable,
                issued_at: int, first_target_at: int, available_at: int, case_weights=None) -> None:
        """Register all members' original predictions after an outcome window matures.

        The caller aligns case keys and supplies the intended evaluation weights.
        Missing outcomes and zero case weights are omitted from fitting.
        """
        issue, first, available = map(_clock, (issued_at, first_target_at, available_at))
        if not issue < first <= available:
            raise ValueError("Require issued_at < first_target_at <= available_at.")
        if origin is None or origin in self._windows:
            raise ValueError("Each observation requires a unique origin.")
        if set(forecasts) != set(self._names):
            raise ValueError("Learning membership requires an originally issued forecast from every member.")
        actual = np.asarray(outcomes, dtype=float)
        arrays = [_quantiles(forecasts[name], self.levels) for name in self._names]
        if any(array.shape[:-1] != actual.shape for array in arrays):
            raise ValueError("Member forecasts and outcomes must have the same case axes.")
        magnitude = np.ones(actual.shape) if case_weights is None else np.asarray(case_weights, dtype=float)
        if (magnitude.shape != actual.shape or not np.isfinite(magnitude).all()
                or (magnitude < 0.).any() or np.isinf(actual).any()):
            raise ValueError("Case weights must be finite and nonnegative and outcomes cannot be infinite.")
        valid = np.isfinite(actual) & (magnitude > 0.)
        window = None
        if valid.any():
            if self._point_only:
                k = self.levels.index(0.5)
                arrays = [array[..., k:k + 1] for array in arrays]
            q = np.stack([array[valid] for array in arrays], axis=-1)
            y, mass = actual[valid], magnitude[valid]
            losses = _loss(q, y, mass, self._loss_levels, self._choices)
            if losses[-1] > 0:
                relative = losses / losses[-1]
                if not np.isfinite(relative).all():
                    raise ValueError("The relative losses must be finite.")
                window = _Window(issue, first, available, q, y, mass, relative)
        self._windows[origin] = window

    def _support(self, issue: int):
        eligible = [(origin, row) for origin, row in self._windows.items()
                    if row is not None and row.available_at < issue]
        blocks = []
        for origin, row in sorted(eligible, key=lambda item: (item[1].available_at, item[1].issued_at)):
            if not blocks or blocks[-1][1].available_at < row.first_target_at:
                blocks.append((origin, row))
        return blocks if self._lookback is None else blocks[-self._lookback:]

    def _fit_weights(self, past, triple):
        risk, origins = [], []
        if self._weighted:
            for origin, row in past:
                if triple not in row.grid_loss:
                    row.grid_loss[triple] = _loss(
                        row.values[..., list(triple)], row.outcomes, row.magnitude, self._loss_levels, _GRID,
                    )
                losses = row.grid_loss[triple]
                if losses[_EQUAL_GRID] <= 0:
                    continue
                ratios = losses / losses[_EQUAL_GRID]
                if not np.allclose(ratios, 1., rtol=_TOLERANCE, atol=_TOLERANCE):
                    risk.append(ratios)
                    origins.append(origin)
        n = len(risk)
        if n < _MIN_WINDOWS:
            return (1 / 3,) * 3, tuple(origins)
        average = np.mean(risk, axis=0)
        ties = np.isclose(average, average.min(), rtol=_TOLERANCE, atol=_TOLERANCE)
        learned = _GRID[ties].mean(axis=0)
        shrunk = (n * learned + 1.) / (n + _MIN_WINDOWS)
        return tuple(shrunk.tolist()), tuple(origins)

    def explain(self, issue_at: int) -> BlendDecision:
        """Return the decision and its evidence without running a forecaster."""
        past = self._support(_clock(issue_at))
        origins = tuple(origin for origin, _ in past)
        full = (1 / len(self._names),) * len(self._names)
        if len(past) < _MIN_WINDOWS:
            return BlendDecision(self._names, full, origins, (), "insufficient_history")
        losses = np.mean([row.choice_loss for _, row in past], axis=0)
        n = len(self._names)
        if self._selection == "individual":
            triple = tuple(sorted(np.argsort(losses[:n], kind="stable")[:3].tolist()))
        else:
            candidates = losses[n:-1]
            tied = np.flatnonzero(np.isclose(candidates, candidates.min(), rtol=_TOLERANCE, atol=_TOLERANCE))
            if len(tied) == len(candidates) and len(candidates) > 1:
                return BlendDecision(self._names, full, origins, (), "indistinguishable_triples")
            triple = self._triples[int(tied[0])]
        weights, fitting = self._fit_weights(past, triple)
        return BlendDecision(tuple(self._names[i] for i in triple), weights, origins, fitting, self._selection)

    def predict(self, context, *, issue_at: int, full_roster: bool = True) -> BlendForecast:
        """Run the configured forecasters and issue their combined quantiles.

        Full-roster predictions can update selection after their outcomes arrive.
        A selected-only call saves inference when a current decision is in use.
        """
        issue = _clock(issue_at)
        decision = self.explain(issue)
        names = self._names if full_roster else decision.members
        members = {name: _quantiles(self._models[name](context), self.levels).copy() for name in names}
        shape = members[names[0]].shape
        if any(q.shape != shape for q in members.values()):
            raise ValueError("Model forecast case shapes differ.")
        values = np.zeros(shape)
        for name, weight in zip(decision.members, decision.weights, strict=True):
            values += weight * members[name]
        for array in (*members.values(), values):
            array.setflags(write=False)
        return BlendForecast(values, self.levels, MappingProxyType(members), issue, decision)
