"""Fit and issue D-wabi on the native FEV input contract."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fev import Benchmark
from dwabi import CutoffDecision

from .cases import predict_at_cutoffs, predictions
from .cutoff import METHOD, SUPPLEMENTAL, issue, read_policy
from .guarded import ROOT
from .native_calendar import build_calendars, official_episode
from .providers import digest, make_provider


PRETRAINED_ON = tuple(json.loads((ROOT / "pretraining_overlap.json").read_bytes()))


@dataclass(frozen=True)
class _FittedTask:
    entry: dict
    decision: CutoffDecision
    definition: dict


class NativeFEV:
    """Use an FEV checkout for its benchmark suite and pinned upstream provider adapters."""

    def __init__(self, *, fev_root, method=METHOD, checkpoint_root=None, device="cuda",
                 policy_path=None, benchmark=None, families=None):
        if method != METHOD:
            raise ValueError(f"The FEV adapter implements {METHOD!r}.")
        fev_root = Path(fev_root).resolve()
        self.models_root = fev_root / "models"
        self.checkpoint_root = checkpoint_root
        self.device = device
        self.provider_pins = json.loads((ROOT / "provider_pins.json").read_bytes())
        self.pins = {name: spec for name, spec in self.provider_pins.items() if name != SUPPLEMENTAL}
        self.providers = {}
        self._dataset_files = json.loads((ROOT / "dataset_files.json").read_bytes())["files"]
        self._verified = set()
        self.schedule, self.decisions, self.task_definitions = {}, {}, {}
        self.benchmark = Path(benchmark) if benchmark else fev_root / "benchmarks" / "fev_bench" / "tasks.yaml"
        self.families = Path(families) if families else ROOT / "physical_families.json"
        self._calendar = None
        policy_path = policy_path or os.environ.get("DWABI_FEV_POLICY")
        self._prepared = bool(policy_path)
        if policy_path:
            state = json.loads(Path(policy_path).read_bytes())
            self.schedule, self.decisions = read_policy(state, self.pins)
            self.task_definitions = state["task_definitions"]

    def check_data(self, task):
        # File pins identify the benchmark revision. Roles and cutoffs come from the FEV task itself.
        prefix = os.environ.get("FEV_DATASETS_PREFIX")
        if not prefix:
            return
        config = task.dataset_config
        key = (prefix, config)
        if key in self._verified or config not in self._dataset_files:
            return
        root = Path(prefix) / config
        actual = {path.name: digest(path) for path in root.glob("*.parquet")}
        expected = {Path(name).name: sha for name, sha in self._dataset_files[config].items()}
        if actual != expected:
            raise ValueError(f"{config}: mirror files differ from the pinned benchmark data.")
        self._verified.add(key)

    def provider(self, name):
        if name not in self.providers:
            self.providers[name] = make_provider(
                name, self.provider_pins[name], self.checkpoint_root, self.device, models_root=self.models_root,
            )
        return self.providers[name]

    def _forecast(self, task, case, ends, members):
        expected = (*case[4].shape, len(task.target_columns), len(task.quantile_levels) + 1)
        laws = {}
        for name in members:
            law = predict_at_cutoffs(self.provider(name), task, case, ends)
            if law.shape != expected or not np.isfinite(law).all():
                raise ValueError(f"{name}: predictions do not cover the native forecast axes.")
            laws[name] = law
        return laws

    def fit(self, task):
        """Load the task and fit on forecasts issued inside its pre-benchmark history."""
        self.check_data(task)
        name = task.task_name
        if self._prepared:
            if name not in self.decisions:
                raise ValueError("The prepared policy does not contain this FEV task.")
            fitted = _FittedTask(self.schedule[name], self.decisions[name], self.task_definitions[name])
        else:
            fitted = self._fit_one(task)
        if task.to_dict() != fitted.definition:
            raise ValueError("The FEV task definition differs from the one used to fit this policy.")
        task.load_full_dataset(num_proc=1)
        return fitted

    def predict(self, task, fitted):
        """Issue every official window with the frozen task-level decision."""
        if task.to_dict() != fitted.definition:
            raise ValueError("The FEV task definition differs from the one used to fit this policy.")
        entry = fitted.entry
        roles = {"known": list(task.known_dynamic_columns), "past_only": list(task.past_dynamic_columns),
                 "static": list(task.static_columns)}
        if (roles != entry["roles"] or task.num_windows != len(entry["official"])
                or task.freq != entry["frequency"]):
            raise ValueError("The FEV task differs from the issued calibration configuration.")
        results = []
        for row in entry["official"]:
            episode = official_episode(task, row)
            forecast = issue(task, episode.case, episode.ends, self.pins, self.provider, fitted.decision)
            results.append(predictions(task, forecast.values))
        return results

    def _fit_one(self, task):
        from .prepare_cutoff import fit

        if self._calendar is None:
            suite = Benchmark.from_yaml(str(self.benchmark)).tasks
            lineage = json.loads(self.families.read_bytes())
            schedule, local = build_calendars(suite, {t.task_name: lineage[t.task_name] for t in suite})
            self._calendar = schedule, local, {t.task_name: t.to_dict() for t in suite}
        schedule, local, definitions = self._calendar
        name = task.task_name
        if name not in schedule:
            raise ValueError("Supply a benchmark and source-lineage file containing the requested task.")
        if task.to_dict() != definitions[name]:
            raise ValueError("The task definition differs from its declared calibration calendar.")
        state = fit(self, [task], {name: schedule[name]}, {name: local[name]} if name in local else {})
        entries, decisions = read_policy(state, self.pins)
        return _FittedTask(entries[name], decisions[name], state["task_definitions"][name])
