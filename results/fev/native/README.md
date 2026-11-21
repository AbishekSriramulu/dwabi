# D-wabi on the 100-task FEV-Bench

On the pinned FEV comparison snapshot, D-wabi ranks first on scaled quantile loss (SQL) by win rate across all 100 tasks. It ranks first on overall SQL skill when closed APIs are excluded and second when they are included. It ranks first on both SQL measures in the 20-task Retail panel in either field. [FEV submission](https://github.com/autogluon/fev/pull/214) · [Task scores](FEV_EXPORT.csv) · [All 16 comparisons](ranks.csv)

## Headline SQL

The final two columns give the score and rank. Win rate averages pairwise task comparisons, giving half credit to ties. Skill is one minus the geometric mean of task error ratios to Seasonal Naive, after FEV’s imputation and clipping. The dashboard hides systems by default. Enable systems to include D-wabi. Both fields include noncommercial models. [FEV evaluator](https://github.com/autogluon/fev/blob/844509e5211292620fbd9d5f24695f1d27da767a/src/fev/analysis.py) · [Retail membership](../retail_tasks.json)

| Panel | Tasks | Closed APIs | Systems and models | Win rate · rank | Skill · rank |
| --- | ---: | --- | ---: | ---: | ---: |
| Overall | 100 | Off | 36 | 92.17% · 1 | 51.59% · 1 |
| Overall | 100 | On | 41 | 89.975% · 1 | 51.59% · 2 |
| Retail | 20 | Off | 36 | 94.57% · 1 | 46.93% · 1 |
| Retail | 20 | On | 41 | 92.25% · 1 | 46.93% · 1 |

## Ranks on every published metric

Each cell gives win-rate rank / skill rank. The evaluator uses its published rules for missing or pretraining-flagged competitor scores. D-wabi has a finite score on every task for all four metrics. [Machine-readable ranks, rates and field leaders](ranks.csv)

| Panel | Closed APIs | SQL | WQL | WAPE | MASE |
| --- | --- | ---: | ---: | ---: | ---: |
| Overall | Off | 1 / 1 | 1 / 1 | 1 / 3 | 1 / 3 |
| Overall | On | 1 / 2 | 1 / 3 | 2 / 5 | 1 / 5 |
| Retail | Off | 1 / 1 | 2 / 2 | 2 / 2 | 2 / 3 |
| Retail | On | 1 / 1 | 1 / 3 | 3 / 5 | 3 / 6 |

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
