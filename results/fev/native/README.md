# Reproduce D-wabi on FEV-Bench

The archived task scores and ranks use the pinned FEV snapshot below. Enable the Systems category in the FEV dashboard to include system submissions. [Task scores](FEV_EXPORT.csv) · [Archived ranks](ranks.csv) · [Upstream submission and timing](https://github.com/autogluon/fev/pull/214)

## Evaluation and reproduction

The results cover 1,308 forecast windows and 8,926,250 observed target cells. The adapter uses FEV's declared historical, static and known-future inputs, recipient cutoffs, horizon and row order. The 400 task-level metric reductions match the stock scorer. [Score verification](VERIFIED.json) · [Per-window results](WINDOWS.json) · [Issued-window receipts](ISSUED.json)

Follow the [pinned adapter instructions](../../../docs/FEV.md#run-the-benchmark) to recreate the forecasts. Each task fits its policy using earlier eligible outcomes. The [frozen policy](policy.json) records the configuration used for the scores. [Pinned checkpoint revisions](../checkpoints.json)

To reproduce the ranking, install FEV at commit 844509e5211292620fbd9d5f24695f1d27da767a and run the following from the D-wabi checkout.

~~~bash
python results/fev/rank_native.py /path/to/fev results/fev/native/FEV_EXPORT.csv
~~~

The score export covers accuracy. Full cold-run timing is reported separately.

## Licensing

The D-wabi Python package is MIT-licensed and ships without checkpoint weights. The benchmark system uses separately licensed checkpoints, including TimesFM-3 and TimeH2O-v1, and is marked noncommercial. [Package license](../../../LICENSE) · [Checkpoint revisions](../checkpoints.json)

Uses ci-proto, released by Continuous Intelligence. Continuous Intelligence does not endorse or approve this product or service. [Checkpoint terms](../CI_PROTO_LICENSE.txt)
