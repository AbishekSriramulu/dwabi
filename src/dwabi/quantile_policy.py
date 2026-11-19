"""Frozen median pools from earlier matched probabilistic losses."""

from dataclasses import dataclass
from statistics import fmean

import numpy as np

from .blend import _clock
from .point_policy import _mature


def _roster(models, supplemental):
    models = tuple(models)
    if (len(models) < 3 or any(not isinstance(name, str) or not name for name in models)
            or len(set(models)) != len(models)):
        raise ValueError("Supply a roster of at least three distinct model names.")
    if supplemental is not None and (
        not isinstance(supplemental, str) or not supplemental or supplemental in models
    ):
        raise ValueError("A supplemental model needs its own name outside the original roster.")
    return models


@dataclass(frozen=True)
class QuantileDecision:
    """One frozen quantile policy in the original model order."""

    models: tuple[str, ...]
    supplemental: str | None
    issue_at: int
    fallback: tuple[str, ...]
    admitted: tuple[str, ...] = ()
    fallback_origins: tuple[int | str, ...] = ()
    admitted_origins: tuple[int | str, ...] = ()

    def __post_init__(self):
        models = _roster(self.models, self.supplemental)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "issue_at", _clock(self.issue_at))
        for field in ("fallback", "admitted", "fallback_origins", "admitted_origins"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if len(self.fallback) not in (3, len(models)) or len(set(self.fallback)) != len(self.fallback):
            raise ValueError("Use a three-model fallback or the complete original roster.")
        if not set(self.fallback) <= set(models):
            raise ValueError("The quantile fallback must use the original roster.")
        if self.admitted and (
            self.supplemental is None or len(set(self.admitted)) != 3 or len(self.admitted) != 3
            or not set(self.admitted) <= {*models, self.supplemental}
            or len(set(self.admitted_origins)) < 3
        ):
            raise ValueError("An admitted triple requires three common earlier comparisons and a supplemental roster.")


@dataclass(frozen=True)
class _LossIssue:
    origin: int | str
    issued_at: int
    first_target_at: int
    available_at: int
    losses: tuple[float | None, ...]


class QuantileHistory:
    """Register matched loss rows for one task and select medians from mature evidence.

    Losses can be any common probabilistic metric for which smaller is better.
    Supply the caller's native scores on the same outcome mask for each issue.
    A supplemental loss is present only when its original prediction qualified.
    """

    def __init__(self, models, *, supplemental=None):
        self.models = _roster(models, supplemental)
        self.supplemental = supplemental
        self._records = {}

    def observe(self, losses, *, origin, issued_at, first_target_at, available_at):
        issued_at, first_target_at, available_at = map(_clock, (issued_at, first_target_at, available_at))
        if not issued_at < first_target_at <= available_at:
            raise ValueError("A target window must start after issue and finish before it can supply evidence.")
        if not isinstance(origin, str):
            origin = _clock(origin)
        if origin in self._records:
            raise ValueError("A completed quantile origin has already been registered.")
        base = set(self.models)
        permitted = base if self.supplemental is None else base | {self.supplemental}
        if not base <= set(losses) <= permitted:
            raise ValueError("Every comparison must cover the complete original roster.")
        values = tuple(losses[name] for name in self.models)
        if self.supplemental is not None:
            values += (losses.get(self.supplemental),)
        values = tuple(None if x is None else float(x) for x in values)
        if any(x is not None and (not np.isfinite(x) or x < 0.) for x in values):
            raise ValueError("Matched quantile losses must be nonnegative, finite or undefined.")
        self._records[origin] = _LossIssue(origin, issued_at, first_target_at, available_at, values)

    def freeze(self, *, issue_at):
        """Freeze decisions using strictly earlier disjoint outcomes."""
        issue_at = _clock(issue_at)
        length = len(self.models)
        rows = tuple(self._records.values())
        ordinary = tuple(r for r in _mature(rows, issue_at) if all(x is not None for x in r.losses[:length]))
        fallback = self.models
        if len(ordinary) >= 3:
            rank = sorted(range(length), key=lambda i: np.mean([r.losses[i] for r in ordinary]))
            fallback = tuple(self.models[i] for i in rank[:3])
        choice, common = (), ()
        if self.supplemental is not None:
            common = _mature((r for r in rows if all(x is not None for x in r.losses)), issue_at)
            if len(common) >= 3:
                roster = (*self.models, self.supplemental)
                rank = sorted(range(len(roster)), key=lambda i: fmean(r.losses[i] for r in common))
                choice = tuple(roster[i] for i in rank[:3])
        return QuantileDecision(self.models, self.supplemental, issue_at, fallback, choice,
                                tuple(r.origin for r in ordinary), tuple(r.origin for r in common))
