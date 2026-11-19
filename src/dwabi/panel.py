"""A keyed panel interface for automatic, chronological ensemble fitting."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import polars as pl
from utilsforecast.processing import make_future_dataframe

from .blend import WalkForwardBlend
from .point_policy import PointHistory, center
from .wape import observed_support, wape_deciles, wape_point


@dataclass(frozen=True)
class ForecastRequest:
    """History and permitted future rows for one model call."""

    history: pl.DataFrame
    future: pl.DataFrame
    h: int
    freq: int | str | None
    quantiles: tuple[float, ...]

    @property
    def quantile_cols(self) -> tuple[str, ...]:
        return tuple(f"q{q}" for q in self.quantiles)


def _positive(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


class DWabi:
    """Fit one ensemble on held forecasts made from earlier panel prefixes.

    Each model takes a ForecastRequest and returns a Polars DataFrame with
    the request's series/date keys and named quantile columns. Models receive
    all observed prefix rows and explicitly declared known-future columns.
    """

    def __init__(
        self,
        models: Mapping[str, Callable[[ForecastRequest], pl.DataFrame]],
        *,
        h: int,
        freq: str | int | None = None,
        n_windows: int = 12,
        objective: str = "pinball",
        selection: str = "individual",
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        known_future: Sequence[str] = (),
        quantiles: Sequence[float] = tuple(i / 10 for i in range(1, 10)),
    ):
        self.h = _positive(h, "h")
        self.n_windows = _positive(n_windows, "n_windows")
        self.freq = freq
        self.keys = (id_col, time_col)
        self.target_col = target_col
        self.known_future = tuple(known_future)
        self.quantiles = tuple(quantiles)
        self._objective = objective
        self._selection = selection
        if len(models) < 3 or any(not isinstance(name, str) or not name for name in models):
            raise ValueError("Provide at least three named callable forecasters.")
        if objective == "wape" and self.quantiles != tuple(i / 10 for i in range(1, 10)):
            raise ValueError("The WAPE point action uses the nine deciles.")
        if len(set((*self.keys, target_col, *self.known_future))) != 3 + len(self.known_future):
            raise ValueError("Identifier, time, target and known-future roles must use distinct columns.")
        self._models = {name: self._adapter(model) for name, model in models.items()}
        self._learner()

    def _adapter(self, model):
        if not callable(model):
            raise ValueError("Each member must be a callable accepting a ForecastRequest.")

        def predict(request):
            answer = self._panel(model(request), (*self.keys, *request.quantile_cols))
            columns = request.quantile_cols
            if self._objective == "wape":
                point_column = "point" if "point" in answer.columns else "q0.5"
                answer = answer.select(*self.keys, pl.col(point_column).alias("_model_point"), *columns)
                columns = ("_model_point", *columns)
            if answer.height != request.future.height:
                raise ValueError("A forecaster changed the requested panel size.")
            keyed = request.future.select(self.keys).join(
                answer.select((*self.keys, *columns)),
                on=list(self.keys), how="left", validate="1:1", maintain_order="left",
            )
            values = keyed.select(pl.col(columns).cast(pl.Float64)).to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("A forecaster omitted requested keys or supplied nonfinite quantiles.")
            return values

        return predict

    def _learner(self):
        if self._objective == "wape":
            return PointHistory(tuple(self._models))
        return WalkForwardBlend(self._models, levels=self.quantiles, objective=self._objective, selection=self._selection)

    def _point_laws(self, request):
        return tuple(model(request).reshape(-1, self.h, 1, 10) for model in self._models.values())

    def _panel(self, frame, required):
        if not isinstance(frame, pl.DataFrame) or not set(required) <= set(frame.columns):
            raise ValueError(f"Expected a Polars DataFrame with columns {tuple(required)}.")
        if frame.is_empty() or any(frame[k].null_count() for k in self.keys):
            raise ValueError("Forecast panels require nonempty data and nonmissing keys.")
        if frame.select(pl.struct(self.keys).n_unique()).item() != frame.height:
            raise ValueError("Series and timestamp keys must be unique.")
        return frame.sort(self.keys)

    def _windows(self, history, clock, first_validation):
        id_col, time_col = self.keys
        n_items = history[id_col].n_unique()
        timestamp = pl.col(time_col).cast(pl.Int64)
        first_complete = history.group_by(id_col).agg(timestamp.min())[time_col].max()
        counts = history.group_by(time_col).len().sort(time_col)["len"].to_numpy()
        found = []
        # A shared issuance clock must follow the previous final target.
        for end in range(len(clock) - 2, self.h - 1, -(self.h + 1)):
            first = end - self.h + 1
            if first < first_validation or clock[first - 1] < first_complete:
                break
            if (counts[first:end + 1] == n_items).all():
                found.append((first - 1, first, end))
            if len(found) == self.n_windows:
                break
        for issue, first, end in reversed(found):
            context = history.filter(timestamp <= int(clock[issue]))
            actual = history.filter(timestamp.is_between(int(clock[first]), int(clock[end])))
            yield issue, first, end, context, actual

    def _future(self, history, supplied):
        id_col, time_col = self.keys
        last = history.group_by(id_col).agg(pl.col(time_col).max()).sort(id_col)
        future_columns = (*self.keys, *self.known_future)
        if supplied is None:
            if self.known_future:
                raise ValueError("Supply values for the declared known-future variables.")
            frequency = self.freq
            if frequency is None:
                clock = history[time_col].unique().sort()
                deltas = clock.cast(pl.Int64).diff().drop_nulls().unique()
                if len(deltas) != 1:
                    raise ValueError("Supply freq or future rows for a grid with an unknown cadence.")
                delta = int(deltas[0])
                dtype = clock.dtype
                frequency = delta if dtype.is_integer() else f"{delta}{'d' if dtype == pl.Date else dtype.time_unit}"
            return make_future_dataframe(last[id_col], last[time_col], frequency, self.h, id_col=id_col, time_col=time_col)

        future = self._panel(supplied, future_columns).select(future_columns)
        if future[time_col].dtype != last[time_col].dtype:
            raise ValueError("Supply future timestamps with the same type as the fitted history.")
        if (future[id_col].unique().sort().to_list() != last[id_col].to_list()
                or not future[id_col].value_counts()["count"].eq(self.h).all()
                or future[time_col].cast(pl.Int64).min() <= last[time_col].cast(pl.Int64).max()):
            raise ValueError("Supply exactly h future rows after the cutoff for every fitted series.")
        return future

    def fit(self, train: pl.DataFrame, validation: pl.DataFrame | None = None):
        """Fit chronological blend weights using training and optional validation."""
        required = (*self.keys, self.target_col, *self.known_future)
        history = self._panel(train, required)
        id_col, time_col = self.keys
        time_type = history[time_col].dtype
        if not (time_type.is_integer() or time_type == pl.Date or isinstance(time_type, pl.Datetime)):
            raise ValueError("Use an integer, date or datetime column for the shared clock.")
        first_validation = 0
        if validation is not None:
            supplied = self._panel(validation, required)
            if supplied[id_col].unique().sort().to_list() != history[id_col].unique().sort().to_list():
                raise ValueError("Validation must contain the same series as training.")
            if supplied[time_col].dtype != time_type:
                raise ValueError("Supply validation timestamps with the same type as training.")
            if supplied[time_col].cast(pl.Int64).min() <= history[time_col].cast(pl.Int64).max():
                raise ValueError("Validation must start strictly after the training span.")
            first_validation = history[time_col].n_unique()
            history = pl.concat((history, supplied.select(history.columns)), how="vertical_relaxed").sort(self.keys)
        if history.group_by(id_col).agg(pl.col(time_col).max())[time_col].n_unique() != 1:
            raise ValueError("The panel interface requires a shared final cutoff.")
        dates = history[time_col].unique().sort()
        clock = dates.cast(pl.Int64).to_numpy()
        learner = self._learner()
        summary = []
        for issue, first, last, context, observed in self._windows(history, clock, first_validation):
            request = ForecastRequest(context, observed.select((*self.keys, *self.known_future)),
                                      self.h, self.freq, self.quantiles)
            actual = observed[self.target_col].cast(pl.Float64).to_numpy()
            clocks = dict(origin=int(clock[issue]), issued_at=issue, first_target_at=first, available_at=last)
            if self._objective == "wape":
                laws = self._point_laws(request)
                decision = learner.freeze(0, 0, issue_at=issue)
                weights = np.asarray(decision.ordinal_weights if decision.ordinal_weights is not None
                                     else np.ones(len(self._models)), dtype=float)
                members, weights = list(self._models), (weights / weights.sum()).tolist()
                volume = np.abs(actual[np.isfinite(actual)]).sum()
                if volume > 0.:
                    actual = actual.reshape(-1, self.h, 1)
                    learner.observe(0, 0, **clocks, points=np.stack([law[..., 0] for law in laws], axis=-1),
                                    actual=actual, case_weights=np.full(actual.shape, 1. / volume),
                                    include_menu=False)
            else:
                forecasts = learner.predict(request, issue_at=issue)
                members, weights = list(forecasts.decision.members), list(forecasts.decision.weights)
                learner.observe(forecasts.members, actual, **clocks)
            summary.append({"cutoff": issue, "valid_from": first, "valid_through": last,
                            "members": members, "weights": weights})
        self._history = history
        self._blend = learner
        self._final_issue = len(dates) - 1
        self.decision_ = (learner.freeze(0, 0, issue_at=self._final_issue) if self._objective == "wape"
                          else learner.explain(self._final_issue))
        windows = pl.from_dicts(summary, schema={"cutoff": pl.UInt32, "valid_from": pl.UInt32,
                                               "valid_through": pl.UInt32, "members": pl.List(pl.String),
                                               "weights": pl.List(pl.Float64)})
        self.validation_windows_ = windows.with_columns(
            dates.gather(windows[column]).alias(column) for column in ("cutoff", "valid_from", "valid_through")
        )
        return self

    def predict(self, future: pl.DataFrame | None = None) -> pl.DataFrame:
        """Forecast using the fit's pre-test state and optional known future inputs."""
        if not hasattr(self, "_blend"):
            raise ValueError("Fit the ensemble before making a forecast.")
        dates = self._future(self._history, future)
        request = ForecastRequest(self._history, dates, self.h, self.freq, self.quantiles)
        if self._objective == "wape":
            laws = self._point_laws(request)
            points = np.stack([law[..., 0] for law in laws], axis=-1)
            internal, quantiles = wape_deciles(tuple(law[..., 1:] for law in laws))
            nonnegative, integer = observed_support([self._history[self.target_col].cast(pl.Float64).to_numpy()])
            point = wape_point(center(points, self.decision_), internal, epsilon=0.,
                               nonnegative=nonnegative, integer=integer).ravel()
            quantiles = quantiles.reshape(-1, len(self.quantiles))
        else:
            forecast = self._blend.predict(request, issue_at=self._final_issue, full_roster=False)
            point, quantiles = forecast.point, forecast.quantiles
        return dates.select(self.keys).with_columns(
            pl.Series("point", point),
            *(pl.Series(col, quantiles[:, i]) for i, col in enumerate(request.quantile_cols)),
        )

    def evaluate(self, test: pl.DataFrame) -> dict[str, float | int | None]:
        """Score a held horizon without using its outcomes to fit weights."""
        test = self._panel(test, (*self.keys, self.target_col, *self.known_future))
        forecasts = self.predict(test)
        actual = test[self.target_col].cast(pl.Float64).to_numpy()
        valid = np.isfinite(actual)
        error = np.abs(actual[valid] - forecasts["point"].to_numpy()[valid])
        volume = np.abs(actual[valid]).sum()
        levels = np.asarray(self.quantiles)
        quantiles = forecasts.select([f"q{q}" for q in self.quantiles]).to_numpy()[valid]
        residual = actual[valid, None] - quantiles
        pinball = np.maximum(levels * residual, (levels - 1.) * residual)
        return {"observed": int(valid.sum()), "mae": float(error.mean()) if valid.any() else None,
                "wape": float(error.sum() / volume) if volume else None,
                "pinball": float(pinball.mean()) if valid.any() else None}
