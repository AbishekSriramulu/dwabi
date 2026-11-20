# D-wabi

D-wabi combines a user-supplied roster of at least three forecasters. It estimates forecast combinations from earlier issues whose outcomes are available and shares the fitted decision with the caller. Each member supplies quantiles on the same forecast cases.

Install with `pip install dwabi`. On FEV-Bench’s 100 tasks, D-wabi ranks first on SQL win rate and skill with systems enabled and closed APIs excluded. Including closed APIs gives first on overall win rate and second on skill. D-wabi ranks first on both SQL measures on all 20 Retail tasks in either field. The scores use FEV’s original covariates and forecast windows. [Verified results and reproduction instructions](https://github.com/AbishekSriramulu/dwabi/tree/main/results/fev/native) · [Python package](https://pypi.org/project/dwabi/0.1.1.post2/) · [Upstream FEV pull request](https://github.com/autogluon/fev/pull/214)

Uses ci-proto, released by Continuous Intelligence. Continuous Intelligence does not endorse or approve this product or service. The D-wabi Python package is MIT-licensed and ships without model weights. The published FEV configuration uses noncommercial checkpoints. For commercial forecasting, supply models and data licensed for your intended use. [Package license](https://github.com/AbishekSriramulu/dwabi/blob/v0.1.1.post2/LICENSE) · [Benchmark licenses and attribution](https://github.com/AbishekSriramulu/dwabi/tree/main/results/fev/native#licensing)

## Fit a panel

```python
from dwabi import DWabi

ensemble = DWabi(models, h=7).fit(train)
forecast = ensemble.predict()
print(ensemble.decision_)
print(ensemble.validation_windows_)
print(ensemble.evaluate(test))
```

`train` and `test` are Polars DataFrames with `unique_id`, `ds` and `y` columns. The current panel interface supports a shared cutoff and a regular time grid. Supply an explicit frequency such as `"1d"` when the cadence cannot be inferred. `models` maps names to callables. Each receives a `ForecastRequest` and returns its forecasts keyed by series and timestamp, with quantile columns named `q0.1` through `q0.9` by default.

Pass `validation=validation` to use a designated later segment. With either input, fitting issues historical forecasts from growing prefixes. D-wabi uses only earlier complete windows for each selection decision. `n_windows` controls the backtest budget. Twelve windows is a development default. Each member still receives the full observed prefix at each issue. Forecasts use the full equal pool until three usable windows support a shortlist.

Declare genuinely known-future predictors in `known_future` and provide their issue-time values to `predict(future=...)`. The panel interface uses timestamps as the label-availability clock. Use the issued-forecast interface below for publication delays or separately recorded feature vintages. `evaluate(test)` reports held error with the fitted blend and preserves its decision.

### WAPE point forecasts

Use `DWabi(models, h=7, objective="wape")` to fit a point forecast for weighted absolute percentage error. Each member supplies the nine deciles and can include a separate `point` column. When that column is absent, its median supplies the member point. D-wabi learns ordinal voting weights from completed, disjoint earlier windows. With at least four fitting windows, the optimizer minimizes the largest average increase over the full-roster median after deleting one fitting window. With three it minimizes mean WAPE, and with less evidence it starts from equal influence.

The WAPE point action treats the outcome in one forecast cell as uncertain, approximates the remaining absolute volume in its target channel by its predicted expectation, and takes the median under inverse-volume weighting. It projects the result to nonnegative or integer support when the visible target history supplies that evidence. The returned nine quantiles are the full-roster quantile medians. WAPE and quantile forecasts have separately defined actions.

The `fast` extra enables Numba for large-panel WAPE computations. For several panels, use `PointHistory` to register matched point arrays and native loss weights with explicit issue, first-target and availability clocks. Assign the same physical family key to panels derived from the same source. `freeze(task, family, issue_at=...)` returns the ordinal weights, a proposed peer subset and its comparison history. A peer subset can replace the ordinal point only after at least three mature, disjoint own-task comparisons favor it even after deleting any one comparison. Each peer fit excludes the evaluated family and uses earlier available outcomes. A single panel uses the ordinal action.

## Issue at staggered cutoffs

`CutoffEnsemble` serves the WAPE point and a separate nine-decile median pool. Supply one `ForecastSeries(issue_at, history_times, history, future_times)` per recipient. The history array is shaped `(time, target)`. All clocks use the same integer time axis. Pass historical-only covariates through `past` and planned covariates through `known`. Historical-only values cover the history clock and known values cover both clocks. The ensemble sends each model only the recipients sharing the same actual cutoff. A model returns an array shaped `(recipient, lead, target, 10)` with the point followed by the nine deciles, or nine deciles to use its median as the point.

```python
from dwabi import CutoffEnsemble, ForecastBudget, PointHistory, QuantileHistory

ensemble = CutoffEnsemble(models, supplemental="extra",
                         budget=ForecastBudget(context_limit=16_384, max_cells=2_000_000))
point_history = PointHistory(ensemble.models)
quantile_history = QuantileHistory(ensemble.models, supplemental=ensemble.supplemental)

# Register matched earlier point evidence and probabilistic losses on these histories.
decision = ensemble.freeze(point_history, quantile_history,
                           task="panel", family="source", issue_at=earliest_cutoff)
forecast = ensemble.predict(series, decision=decision)
print(forecast.point, forecast.quantiles, forecast.routes)
```

`QuantileHistory.observe(losses, origin=..., issued_at=..., first_target_at=..., available_at=...)` takes one loss per original model on the same cases and metric, plus the supplemental loss when its earlier forecast was issued. After three strictly mature disjoint comparisons, the fallback pools the three best original models by mean loss. It uses the full original median before that. With a configured supplemental model, three common comparisons allow a shortlist across both rosters. A cohort qualifies when the declared input capacity covers its history and lead cells. Only selected supplemental models run at prediction. The point always uses the original roster. Omit both `supplemental` and `budget` for an original-only system.

Use `predict(series, decision=...)` to replay the same fixed decision at each cutoff. Use `predict_online(series, point_history=..., quantile_history=..., task=..., family=...)` to freeze from each cohort's available history. Register completed evidence separately. Both methods return forecasts in the original recipient order and expose the cohort routes.

## Register issued forecasts

```python
from dwabi import WalkForwardBlend

ensemble = WalkForwardBlend({
    "model-a": model_a,
    "model-b": model_b,
    "model-c": model_c,
    "model-d": model_d,
})

# Each callable receives the same context as it stood at issuance.
issued = ensemble.predict(context, issue_at=10)
print(issued.point, issued.decision)

# When the complete target window has arrived.
ensemble.observe(
    issued.members, actual, origin="window-1",
    issued_at=10, first_target_at=11, available_at=17,
)
next_forecast = ensemble.predict(next_context, issue_at=18)
```

Use integer clocks on a common timeline. Use one instance per task. Align recipient, target and lead coordinates before passing a forecast to the ensemble. The last axis is the configured quantile grid. The combined point is the combined 0.5 quantile.

For warmup, provide a few earlier issued backcasts to `observe` using their original forecasts and outcome availability. The first three admitted windows must have nonoverlapping target windows. D-wabi uses the equal full-roster average before enough evidence is available. When earlier three-model averages have different scores, the joint selector chooses the one with the best average relative loss. An individual-risk selector is also available. Weight fitting uses the one-sixth simplex grid with three windows of shrinkage toward equal weights within the chosen set. Use `weighted=False` for equal weights and `lookback=3` to restrict both fits to the latest three admitted windows.

The default objective is pinball loss across all supplied levels. `objective="absolute"` fits the median with absolute loss. Optional `case_weights` let callers match the scoring scale and aggregation. Each admitted window contributes loss relative to the equal full-roster forecast for member selection. The weight fit normalizes against the selected triple's own equal average.

Keep `full_roster=True` while collecting forecasts for future updates. Use `full_roster=False` to run the current selected members only. Existing backcast forecasts can be supplied directly to `observe`. `explain(issue_at)` returns the subset, weights and supporting origins without model calls.

The benchmark adapter fits the point policy from earlier FEV data and evaluates it on the complete later benchmark. The library also accepts independent, user-supplied forecasts. Model weights and evaluation data are downloaded separately for benchmark reproduction under their original terms. [FEV setup, fit and runtime measurements](docs/FEV.md)
