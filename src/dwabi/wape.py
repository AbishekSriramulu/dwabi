"""Numerical point decisions for weighted absolute percentage error."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

try:
    from numba import njit as _compile
except ImportError:
    def _compile(function):
        return function


@_compile
def _segment_mass(a, b, floor):
    if b < a:
        a, b = b, a
    if a < floor and b < floor:
        return 1. / floor
    if a < floor:
        crossing = (floor - a) / (b - a)
        return crossing / floor + (np.log(b) - np.log(floor)) / (b - a)
    ratio = (b - a) / a
    if not np.isfinite(ratio):
        return (np.log(b) - np.log(a)) / (b - a)
    return np.log1p(ratio) / (a * ratio) if abs(ratio) > 1e-12 else 1. / a


@_compile
def _absolute_mass(start, stop, other, floor):
    if start * stop < 0.:
        crossing = abs(start) / (abs(start) + abs(stop))
        return crossing * _segment_mass(other + abs(start), other, floor) + (
            1. - crossing
        ) * _segment_mass(other, other + abs(stop), floor)
    return _segment_mass(other + abs(start), other + abs(stop), floor)


@_compile
def _tilted_level(law, other, floor):
    # Extend the first and last deciles as constant tails.
    knots = np.empty(11)
    knots[0] = law[0]
    knots[1:10] = law
    knots[10] = law[8]
    mass = np.zeros(10)
    for index in range(10):
        mass[index] = _absolute_mass(knots[index], knots[index + 1], other, floor)
    half = mass.sum() * .5
    if abs(half - mass[:5].sum()) <= half * 1e-14:
        return .5
    accumulated = 0.
    for index in range(10):
        if accumulated + mass[index] >= half:
            begin, end = 0., 1.
            a, b = knots[index], knots[index + 1]
            for _ in range(35):
                midpoint = .5 * (begin + end)
                value = a + midpoint * (b - a)
                partial = _absolute_mass(a, value, other, floor)
                if accumulated + midpoint * partial < half:
                    begin = midpoint
                else:
                    end = midpoint
            return (index + .5 * (begin + end)) / 10.
        accumulated += mass[index]
    return 1.


@_compile
def _quantile(law, level):
    if level <= .1:
        return law[0]
    if level >= .9:
        return law[8]
    position = level * 10.
    lower = int(position) - 1
    return law[lower] + (position - int(position)) * (law[lower + 1] - law[lower])


def _expected_absolute(law):
    left, right = law[..., :-1], law[..., 1:]
    a, b = abs(left), abs(right)
    mixed = np.sign(left) * np.sign(right) < 0.
    average = (a + b) * .5
    corners = np.divide(.5 * (a * a + b * b), a + b, out=np.zeros_like(a), where=(a + b) > 0.)
    mean = np.where(mixed, corners, average)
    return .1 * (mean.sum(axis=-1) + abs(law[..., 0]) + abs(law[..., 8]))


@_compile
def _cell_point(law, mean, base, epsilon):
    series_count, horizon, target_count = base.shape
    result = base.copy()
    floor = max(epsilon * series_count * horizon, np.finfo(np.float64).tiny)
    for target in range(target_count):
        volume = mean[:, :, target].sum()
        if volume <= 0.:
            continue
        for series in range(series_count):
            for lead in range(horizon):
                row = law[series, lead, target]
                other = max(0., volume - mean[series, lead, target])
                level = _tilted_level(row, other, floor)
                if level != .5:
                    result[series, lead, target] = _quantile(row, level)
    return result


def _forecast_array(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError("Forecast values must be real numbers.")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError("Forecast values must be finite.")
    return array


def _support_flags(values: ArrayLike, target_count: int, argument: str) -> np.ndarray:
    flags = np.asarray(values)
    if flags.shape != (target_count,) or flags.dtype != np.bool_:
        raise ValueError(f"{argument} must contain one boolean per target.")
    return flags


def wape_point(
    point: ArrayLike,
    median_deciles: ArrayLike,
    *,
    epsilon: float,
    nonnegative: ArrayLike | None = None,
    integer: ArrayLike | None = None,
) -> np.ndarray:
    """Choose each point under a reciprocal expected volume approximation.

    Points have shape (series, lead, target). Supply sorted deciles 0.1 through
    0.9 on the last axis of median_deciles with the same first three axes.
    Epsilon is the floor for mean observed absolute volume in WAPE.
    Optional per-target support flags apply clipping before rounding.
    """
    base = _forecast_array(point)
    deciles = _forecast_array(median_deciles)
    if base.ndim != 3 or deciles.shape != (*base.shape, 9):
        raise ValueError("Supply (series, lead, target) points and nine matching deciles.")
    if (deciles[..., :-1] > deciles[..., 1:]).any():
        raise ValueError("Median deciles must be ordered.")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0.:
        raise ValueError("epsilon must be finite and nonnegative.")

    target_count = base.shape[-1]
    lower = None if nonnegative is None else _support_flags(nonnegative, target_count, "nonnegative")
    lattice = None if integer is None else _support_flags(integer, target_count, "integer")
    positive_law = (deciles[..., 0] >= 0.) & (base >= 0.)
    with np.errstate(over="ignore", invalid="ignore"):
        law = base[..., None] + (deciles - deciles[..., 4:5])
        law = np.where(positive_law[..., None], np.maximum(law, 0.), law)
        mean = _expected_absolute(law)
    if not np.isfinite(mean).all():
        raise ValueError("Forecast volumes must be finite.")

    result = _cell_point(law, mean, base, epsilon)
    if not np.isfinite(result).all() or (result[positive_law] < 0.).any():
        raise ValueError("Volume adjustment produced an invalid point forecast.")
    if lower is not None:
        result = np.where(lower[None, None, :], np.maximum(result, 0.), result)
    if lattice is not None:
        result = np.where(lattice[None, None, :], np.rint(result), result)
    return result


def _ordered_deciles(raw: np.ndarray) -> np.ndarray:
    center = raw[..., 4:5]
    lower = np.sort(np.minimum(raw[..., :4], center), axis=-1)
    upper = np.sort(np.maximum(raw[..., 5:], center), axis=-1)
    return np.concatenate((lower, center, upper), axis=-1)


def wape_deciles(member_deciles: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Pool member deciles for the point calculation and the served forecast.

    Input has shape (member, series, lead, target, 9). The calculus uses the
    median of fully sorted member deciles. The served law pools member deciles
    after repairing each side of the original member median.
    """
    raw = _forecast_array(member_deciles)
    if raw.ndim != 5 or raw.shape[0] == 0 or raw.shape[-1] != 9:
        raise ValueError("Supply member forecasts with nine deciles on their last axis.")

    sorted_median = np.median(np.sort(raw, axis=-1), axis=0)
    served = np.median(_ordered_deciles(raw), axis=0)
    return sorted_median, served


def observed_support(target_histories: Iterable[ArrayLike]) -> tuple[np.ndarray, np.ndarray]:
    """Find nonnegative and integer support in each flat target history.

    Nonfinite entries, including missing observations, supply no evidence.
    Both support flags are false for an empty or entirely missing history.
    """
    nonnegative, integer = [], []
    for values in target_histories:
        history = np.asarray(values, dtype=float)
        if history.ndim != 1:
            raise ValueError("Supply one flat history per target.")
        finite = history[np.isfinite(history)]
        observed = finite.size > 0
        nonnegative.append(observed and np.min(finite) >= 0.)
        integer.append(observed and np.equal(finite, np.trunc(finite)).all())
    return np.asarray(nonnegative, dtype=bool), np.asarray(integer, dtype=bool)

