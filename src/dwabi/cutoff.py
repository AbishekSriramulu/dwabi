"""Issue a point and probability forecast for each actual cutoff cohort."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .blend import _clock
from .point_policy import PointDecision, center
from .quantile_policy import QuantileDecision, _roster
from .wape import _ordered_deciles, observed_support, wape_deciles, wape_point


def _times(dates):
    dates = np.asarray(dates)
    if (dates.ndim != 1 or not dates.size or dates.dtype.kind not in "iu"
            or (dates.dtype.kind == "u" and int(dates.max()) > np.iinfo(np.int64).max)
            or (dates[1:] <= dates[:-1]).any()):
        raise ValueError("Supply nonempty strictly increasing integer clock arrays on one time axis.")
    return dates.astype(np.int64, copy=False)


def _covariates(values, length, role):
    if not isinstance(values, Mapping):
        raise ValueError(f"Supply a mapping for {role} covariates.")
    arrays = {name: np.asarray(value) for name, value in values.items()}
    if any(not isinstance(name, str) or not name or value.shape != (length,)
           for name, value in arrays.items()):
        raise ValueError(f"Each {role} covariate must cover its entire declared clock.")
    return arrays


@dataclass(frozen=True)
class ForecastSeries:
    """Inputs available for one recipient at an actual issuance time.

    History has shape (time, target). Known covariates cover the history and
    future clocks, and past covariates cover the history clock. Times and
    issue_at are integers measured on a common timeline.
    """

    issue_at: int
    history_times: np.ndarray
    history: np.ndarray
    future_times: np.ndarray
    known: Mapping = field(default_factory=dict)
    past: Mapping = field(default_factory=dict)
    static: Mapping = field(default_factory=dict)

    def __post_init__(self):
        stamp = _clock(self.issue_at)
        past_times, future_times = _times(self.history_times), _times(self.future_times)
        targets = np.asarray(self.history, dtype=float)
        if past_times[-1] > stamp or future_times[0] <= stamp:
            raise ValueError("Each history must end by its cutoff and its future must start after that cutoff.")
        if targets.ndim != 2 or targets.shape[0] != len(past_times) or not targets.shape[1]:
            raise ValueError("Target histories must have matching time and nonempty target axes.")
        known = _covariates(self.known, len(past_times) + len(future_times), "known")
        past = _covariates(self.past, len(past_times), "past")
        if set(known) & set(past):
            raise ValueError("A covariate must have just one dynamic availability role.")
        if not isinstance(self.static, Mapping):
            raise ValueError("Supply a mapping for static covariates.")
        for name, value in (("issue_at", stamp), ("history_times", past_times), ("history", targets),
                            ("future_times", future_times), ("known", known), ("past", past),
                            ("static", dict(self.static))):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ForecastCohort:
    """The recipients issued at one cutoff, in original input order."""

    issue_at: int
    rows: tuple[int, ...]
    series: tuple[ForecastSeries, ...]


@dataclass(frozen=True)
class ForecastBudget:
    """A supplemental model's effective history capacity and numeric input budget."""

    context_limit: int
    max_cells: int

    def __post_init__(self):
        for name in ("context_limit", "max_cells"):
            value = _clock(getattr(self, name))
            if value <= 0:
                raise ValueError("Model context capacity and the input cell budget must be positive.")
            object.__setattr__(self, name, value)

    def cells(self, cohort):
        """Count target/known history and forecast cells for an issued cohort."""
        first = cohort.series[0]
        slots = sum(min(len(s.history_times), self.context_limit) + len(s.future_times)
                    for s in cohort.series)
        return slots * first.history.shape[1] * (1 + len(first.known))


@dataclass(frozen=True)
class CutoffDecision:
    """A reusable pair of issue-time point and quantile decisions."""

    point: PointDecision
    quantile: QuantileDecision

    def __post_init__(self):
        for name, cls in (("point", PointDecision), ("quantile", QuantileDecision)):
            value = getattr(self, name)
            if isinstance(value, dict):
                object.__setattr__(self, name, cls(**value))
            elif not isinstance(value, cls):
                raise TypeError(f"Supply a frozen {name} decision.")


@dataclass(frozen=True)
class ForecastRoute:
    """The pool actually used for a cutoff cohort."""

    issue_at: int
    rows: tuple[int, ...]
    cells: int | None
    members: tuple[str, ...]


@dataclass(frozen=True)
class CutoffForecast:
    """The aligned point and nine deciles with the actual cohort routes."""

    values: np.ndarray
    routes: tuple[ForecastRoute, ...]

    @property
    def point(self):
        return self.values[..., 0]

    @property
    def quantiles(self):
        return self.values[..., 1:]


class CutoffEnsemble:
    """Use the original roster for the point and an optional budgeted median pool.

    Forecasters receive a ForecastCohort with its issued histories and plans.
    Return (series, lead, target, 10) with point then deciles, or nine deciles
    when the member's median is also its point.
    """

    def __init__(self, models, *, epsilon=0., supplemental=None, budget=None):
        if not isinstance(models, Mapping) or any(not callable(model) for model in models.values()):
            raise ValueError("Supply a mapping of named callable forecasters.")
        if (supplemental is None) != (budget is None) or (
            supplemental is not None and (supplemental not in models or not isinstance(budget, ForecastBudget))
        ):
            raise ValueError("Configure a supplemental model and its input budget together.")
        self.models = _roster((name for name in models if name != supplemental), supplemental)
        self.supplemental = supplemental
        self.budget = budget
        self.epsilon = float(epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon < 0.:
            raise ValueError("WAPE epsilon must be finite and nonnegative.")
        self._forecasters = dict(models)

    def freeze(self, point_history, quantile_history, *, task, family, issue_at):
        """Freeze both policies from registered evidence available by issue_at."""
        issue_at = _clock(issue_at)
        if (point_history.models != self.models or quantile_history.models != self.models
                or quantile_history.supplemental != self.supplemental):
            raise ValueError("Both histories must use the ensemble's original and supplemental rosters.")
        return CutoffDecision(point_history.freeze(task, family, issue_at=issue_at),
                              quantile_history.freeze(issue_at=issue_at))

    def predict(self, series, *, decision):
        """Issue with the supplied fixed state on every cutoff cohort."""
        cohorts, shape = self._cohorts(series)
        return self._predict(cohorts, shape, (decision,) * len(cohorts))

    def predict_online(self, series, *, point_history, quantile_history, task, family):
        """Issue using a fresh as-of decision for each cutoff from registered history."""
        cohorts, shape = self._cohorts(series)
        decisions = tuple(self.freeze(point_history, quantile_history, task=task, family=family,
                                      issue_at=cohort.issue_at) for cohort in cohorts)
        return self._predict(cohorts, shape, decisions)

    @staticmethod
    def _cohorts(series):
        series = tuple(series)
        if not series or any(not isinstance(row, ForecastSeries) for row in series):
            raise ValueError("Supply one or more ForecastSeries inputs.")
        horizon, targets = len(series[0].future_times), series[0].history.shape[1]
        groups = {}
        for row, item in enumerate(series):
            if len(item.future_times) != horizon or item.history.shape[1] != targets:
                raise ValueError("All recipients must share the lead and target axes.")
            groups.setdefault(item.issue_at, []).append(row)
        cohorts = []
        for cutoff, rows in sorted(groups.items()):
            own = tuple(series[i] for i in rows)
            if any(row.known.keys() != own[0].known.keys() or row.past.keys() != own[0].past.keys()
                   for row in own):
                raise ValueError("A cutoff cohort requires common covariate roles.")
            cohorts.append(ForecastCohort(cutoff, tuple(rows), own))
        return cohorts, (len(series), horizon, targets)

    def _predict(self, cohorts, shape, decisions):
        for cohort, decision in zip(cohorts, decisions, strict=True):
            p, q = decision.point, decision.quantile
            if _clock(p.issue_at) > cohort.issue_at or q.issue_at > cohort.issue_at:
                raise ValueError("Forecast decisions must be frozen by the recipient cutoff.")
            if p.models != self.models or q.models != self.models or q.supplemental != self.supplemental:
                raise ValueError("Forecast decisions must match the configured roster in order.")
            if p.accept and (p.operator not in ("median", "mean") or not p.members
                             or len(p.members) != len(set(p.members)) or not set(p.members) <= set(self.models)):
                raise ValueError("The accepted point action must use a valid original-roster pool.")

        values = np.empty((*shape, 10), dtype=float)
        routes = []
        for cohort, decision in zip(cohorts, decisions, strict=True):
            cells = None if self.budget is None else self.budget.cells(cohort)
            quantile = decision.quantile
            members = quantile.admitted if cells is not None and cells <= self.budget.max_cells else ()
            members = members or quantile.fallback
            expected = (len(cohort.rows), *shape[1:])
            laws = {name: self._law(name, cohort, expected) for name in self.models}
            base = center(np.stack([law[..., 0] for law in laws.values()], axis=-1), decision.point)
            deciles, _ = wape_deciles(np.stack([law[..., 1:] for law in laws.values()]))
            history = (np.concatenate([series.history[:, j] for series in cohort.series]) for j in range(shape[2]))
            nonnegative, integer = observed_support(history)
            point = wape_point(base, deciles, epsilon=self.epsilon, nonnegative=nonnegative, integer=integer)
            if self.supplemental in members:
                extra = self._law(self.supplemental, cohort, expected)
                if (np.diff(extra[..., 1:], axis=-1) < 0.).any():
                    raise ValueError("The supplemental model must return ordered deciles.")
                laws[self.supplemental] = extra
            pool = [laws[name][..., 1:] if name == self.supplemental
                    else _ordered_deciles(laws[name][..., 1:]) for name in members]
            q = np.median(np.stack(pool, axis=-1), axis=-1)
            rows = list(cohort.rows)
            values[rows, ..., 0] = point
            values[rows, ..., 1:] = q
            routes.append(ForecastRoute(cohort.issue_at, cohort.rows, cells, members))
        return CutoffForecast(values, tuple(routes))

    def _law(self, name, cohort, shape):
        values = np.asarray(self._forecasters[name](cohort), dtype=float)
        if values.shape == (*shape, 9):
            values = np.concatenate((values[..., 4:5], values), axis=-1)
        if values.shape != (*shape, 10) or not np.isfinite(values).all():
            raise ValueError(f"{name}: supply finite forecasts on the recipient, lead, target and decile axes.")
        return values
