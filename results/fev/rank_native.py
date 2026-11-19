"""Rank a complete stock FEV export on the full panel and the leaderboard's Retail domain."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fev import Benchmark
from fev.analysis import leaderboard, pivot_table

SYSTEMS_ON = {"pretrained", "statistical", "task-specific", "system"}
METRICS = ("SQL", "WQL", "WAPE", "MASE")


def read_fields(benchmark, candidate):
    metadata = yaml.safe_load((benchmark / "models.yaml").read_text(encoding="utf-8"))
    fields = {"default_insertion": [], "expanded": []}
    for key, entry in metadata.items():
        if entry.get("hidden", False) or key == "dwabi":
            continue
        data = pd.read_csv(benchmark / "results" / f"{key}.csv")
        fields["expanded"].append(data)
        if entry["model_type"] in SYSTEMS_ON:
            fields["default_insertion"].append(data)
    return {name: pd.concat([*members, candidate], ignore_index=True) for name, members in fields.items()}


def ranks(checkout, scores):
    benchmark = checkout / "benchmarks" / "fev_bench"
    definitions = Benchmark.from_yaml(benchmark / "tasks.yaml").tasks
    panel = {task.task_name for task in definitions}
    retail = set(json.loads(Path(__file__).with_name("retail_tasks.json").read_bytes()))
    candidate = pd.read_csv(scores, float_precision="round_trip")
    if (
        len(panel) != len(definitions)
        or set(candidate.task_name) != panel
        or len(candidate) != len(panel)
        or candidate.model_name.isna().any()
        or candidate.model_name.nunique() != 1
        or not retail <= panel
    ):
        raise ValueError("Expected one model on exactly the published tasks.")
    name = candidate.model_name.iloc[0]
    records = []
    for field, frame in read_fields(benchmark, candidate).items():
        for group, tasks in (("all", panel), ("retail", retail)):
            selected = frame.loc[frame.task_name.isin(tasks)]
            for metric in METRICS:
                raw = pivot_table(selected, metric_column=metric)
                if len(raw) != len(tasks):
                    raise ValueError(f"Task definitions are incompatible in {field}/{group}/{metric}.")
                board = leaderboard(
                    selected,
                    metric_column=metric,
                    missing_strategy="impute",
                    baseline_model="Seasonal Naive",
                    leakage_imputation_model="Chronos-Bolt",
                    normalize_time_per_n_forecasts=100,
                )
                win, skill = board.loc[name, ["win_rate", "skill_score"]]
                records.append({
                    "group": group,
                    "field": field,
                    "model": name,
                    "metric": metric,
                    "tasks": len(tasks),
                    "candidate_raw_finite": int(np.isfinite(raw[name]).sum()),
                    "field_size": len(board),
                    "rank": int((board.win_rate > win).sum()) + 1,
                    "skill_rank": int((board.skill_score > skill).sum()) + 1,
                    "win_rate": 100 * win,
                    "skill": 100 * skill,
                    "win_leaders": " | ".join(sorted(board.index[board.win_rate == board.win_rate.max()])),
                    "skill_leaders": " | ".join(sorted(board.index[board.skill_score == board.skill_score.max()])),
                })
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fev_checkout", type=Path)
    parser.add_argument("stock_export", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = ranks(args.fev_checkout, args.stock_export)
    if args.output:
        result.to_csv(args.output, index=False, lineterminator="\n")
    print(result.to_string(index=False))
