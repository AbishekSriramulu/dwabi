import json
from dataclasses import asdict

import numpy as np
import pytest

from dwabi import (
    CutoffDecision,
    CutoffEnsemble,
    ForecastBudget,
    ForecastSeries,
    PointDecision,
    PointHistory,
    QuantileDecision,
    QuantileHistory,
)


def series(cutoff, history, *, known=False):
    values = np.asarray(history, dtype=float).reshape(-1, 1)
    ticks = np.arange(cutoff - len(values) + 1, cutoff + 1)
    dates = np.asarray([cutoff + 1])
    plan = {"driver": np.arange(len(values) + 1)} if known else {}
    return ForecastSeries(cutoff, ticks, values, dates, known=plan)


def forecaster(name, output, called):
    def predict(cohort):
        values = [output[row] for row in cohort.rows]
        assert all(s.issue_at == cohort.issue_at for s in cohort.series)
        called.append((name, cohort.issue_at, cohort.rows))
        quantiles = np.broadcast_to(np.asarray(values)[:, None, None, None], (len(values), 1, 1, 9))
        return np.concatenate((quantiles[..., :1], quantiles), axis=-1)
    return predict


def point(models, clock=9):
    return PointDecision(tuple(models), clock, None, (), False, "median", tuple(models), 0, ())


def full_decision(models, clock=9):
    return CutoffDecision(point(models, clock), QuantileDecision(tuple(models), None, clock, tuple(models)))


def test_frozen_cutoff_api_contains_only_local_support_and_issues_in_native_order():
    traces = []
    arsenal = {"a": forecaster("a", [8.4, -2.6], traces),
               "b": forecaster("b", [8.6, -2.8], traces),
               "c": forecaster("c", [8.8, -3.0], traces)}
    ensemble = CutoffEnsemble(arsenal)
    frozen = full_decision(arsenal)
    frozen = CutoffDecision(**json.loads(json.dumps(asdict(frozen))))
    original = [series(20, [0., -0.51]), series(10, [1., 2.])]
    result = ensemble.predict(original, decision=frozen)
    np.testing.assert_array_equal(result.point[:, 0, 0], [8.6, 0.])
    np.testing.assert_array_equal(result.quantiles[:, 0, 0, 4], [8.6, -2.8])
    assert [r.rows for r in result.routes] == [(1,), (0,)]
    assert len(traces) == 6
    with pytest.raises(ValueError, match="cutoff"):
        ensemble.predict(original, decision=full_decision(arsenal, 11))
    assert len(traces) == 6


def test_budget_interaction_uses_declared_capability_and_keeps_original_point():
    trace = []
    members = {
        name: forecaster(name, values, trace)
        for name, values in {"a": [3.2, 3.2], "b": [6.5, 6.5], "c": [9., 9.], "extra": [0., 0.]}.items()
    }
    models = ("a", "b", "c")
    ensemble = CutoffEnsemble(members, supplemental="extra", budget=ForecastBudget(context_limit=8, max_cells=6))
    decision = CutoffDecision(point(models), QuantileDecision(models, "extra", 9, models,
        admitted=("extra", "a", "b"), admitted_origins=(1, 2, 3)))
    data = [series(20, [0.4, 1.2, 1.7], known=True), series(10, [0.7, 1.1], known=True)]
    result = ensemble.predict(data, decision=decision)
    np.testing.assert_array_equal(result.point[:, 0, 0], [6.5, 6.5])
    np.testing.assert_array_equal(result.quantiles[:, 0, 0, 4], [6.5, 3.2])
    assert [r.cells for r in result.routes] == [6, 8]
    assert [r.members for r in result.routes] == [("extra", "a", "b"), models]
    assert [row[1] for row in trace if row[0] == "extra"] == [10]


def test_online_selection_waits_for_disjoint_comparable_losses_and_freeze_stays_fixed():
    names = ("a", "b", "c", "d")
    trace = []
    members = {k: forecaster(k, [float(i), float(i)], trace) for i, k in enumerate((*names, "extra"))}
    ensemble = CutoffEnsemble(members, supplemental="extra", budget=ForecastBudget(8, 8))
    point_history = PointHistory(names)
    history = QuantileHistory(names, supplemental="extra")

    def observe(origin, start, end, include_extra=True):
        losses = dict(a=3., b=4., c=2., d=1.)
        if include_extra:
            losses["extra"] = 0.
        history.observe(losses, origin=origin, issued_at=start - 1, first_target_at=start, available_at=end)

    observe(1, 1, 2)
    observe(2, 2, 3)
    observe(3, 4, 5)
    observe(4, 7, 9)
    observe(5, 11, 13, include_extra=False)
    frozen = ensemble.freeze(point_history, history, task="panel", family="source", issue_at=9)
    assert frozen.quantile.admitted == ()
    assert frozen.quantile.fallback_origins == (1, 3)
    clocked = [series(9, [0., 1.]), series(10, [0., 1.])]
    fixed = ensemble.predict(clocked, decision=frozen)
    online = ensemble.predict_online(clocked, point_history=point_history, quantile_history=history,
                                     task="panel", family="source")
    assert [r.members for r in fixed.routes] == [names, names]
    assert [r.members for r in online.routes] == [names, ("extra", "d", "c")]
    assert history.freeze(issue_at=10).admitted_origins == (1, 3, 4)
    assert [row[1] for row in trace if row[0] == "extra"] == [10]


def test_panel_inputs_and_member_arrays_fail_at_the_boundary():
    with pytest.raises(ValueError, match="future"):
        ForecastSeries(10, [8, 9], np.ones((2, 1)), [9])
    with pytest.raises(ValueError, match="known"):
        ForecastSeries(10, [8, 9], np.ones((2, 1)), [11], known={"x": [1, 2]})
    with pytest.raises(ValueError, match="clock"):
        ForecastSeries(10, [8., 9.], np.ones((2, 1)), [11])
    members = {k: lambda cohort: np.zeros((1, 9)) for k in ("a", "b", "c")}
    with pytest.raises(ValueError, match="axes"):
        CutoffEnsemble(members).predict([series(10, [1., 2.])], decision=full_decision(members))


def test_common_cutoff_returns_one_route_and_skips_unselected_supplement():
    trace = []
    models = ("a", "b", "c", "d")
    callbacks = {name: forecaster(name, [float(i), float(i + 2)], trace)
                 for i, name in enumerate((*models, "extra"))}
    ensemble = CutoffEnsemble(callbacks, supplemental="extra", budget=ForecastBudget(8, 8))
    state = CutoffDecision(point(models), QuantileDecision(models, "extra", 9, models,
                           admitted=("d", "c", "b"), admitted_origins=(1, 2, 3)))
    result = ensemble.predict([series(10, [1.1, 2.]), series(10, [0.3, 1.])], decision=state)
    np.testing.assert_array_equal(result.point[:, 0, 0], [1.5, 3.5])
    np.testing.assert_array_equal(result.quantiles[:, 0, 0, 4], [2., 4.])
    assert len(result.routes) == 1 and result.routes[0].rows == (0, 1)
    assert [k for k, _, _ in trace] == list(models)
