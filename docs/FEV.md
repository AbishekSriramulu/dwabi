# D-wabi on FEV-Bench

D-wabi combines forecasts from fixed pretrained models. Each FEV task learns its combination from its own earlier forecast errors. The same procedure runs on every task. Its outputs are one point and nine ordered quantiles for every target, recipient, lead and official window. The combination is task-trained and the component checkpoints are frozen.

## What each model sees

FEV constructs the forecast windows. Each provider receives the history and future inputs returned by `EvaluationWindow.get_input_data()`. Declared known-future values span the forecast horizon. Past-only covariates stop at the recipient's cutoff. Static covariates retain their declared role. Providers can use the fields their architectures support. For example, the stock Toto-2 adapter consumes target history. Where recipient cutoffs differ within one FEV window, D-wabi groups recipients by their actual cutoff for inference and restores FEV's original row order. FEV determines the cadence, missing observations, minimum and maximum context, horizon, quantile levels and scoring settings. [FEV's forecast-time data contract](https://github.com/autogluon/fev/blob/844509e5211292620fbd9d5f24695f1d27da767a/src/fev/task.py#L60-L111)

Chronos-2 target-encodes categorical covariates when forecasting a single target. If that recipient's entire target history is missing, its encoder has no response to fit. For this provider only, D-wabi marks the unencodable categories as missing and passes the numeric drivers through. The fallback applied to one recipient during calibration and zero recipients in the official forecast windows. [Chronos-2 input handling](../src/dwabi/fev/chronos_missing.py)

## How it fits the combination

The base roster comprises t0-gamma, ci-proto, Timestral-209M, TiRex-2, TimesFM-3, Chronos-2 and Toto-2.0-1B. TimeH2O-v1 is eligible for the quantile pool on cohorts that fit its published input budget. Model identifiers, checkpoint revisions, file digests and package versions appear in [provider_pins.json](../src/dwabi/fev/provider_pins.json).

1. The calibration calendar uses timestamps, recipient membership and source lineage. It places earlier forecasting episodes inside the history FEV supplies at each recipient's first official cutoff. A task's `max_context_length` therefore limits both the evidence used for fitting and the history a provider sees. The first set ends before the earliest official issue in its physical source family. If that provides fewer than three disjoint comparisons, the scheduler also searches the same task's later eligible history within this prefix, protecting sibling tasks' official target intervals. Family identifiers only set these time exclusions. Target values from other tasks do not enter a task's fit.
2. For the point, seven member forecasts are sorted at each recipient, lead and target. The package learns ordinal weights on the original members by minimizing the task's WAPE on mature, disjoint calibration forecasts. With more than three observations it minimizes worst leave-one-origin-out excess loss versus the equal-weight median. With insufficient evidence it keeps equal weights. The seven members' deciles estimate how much a cell contributes to the WAPE denominator. The package uses that estimate and the target's observed support to compute its final point. [Point solver](https://github.com/AbishekSriramulu/dwabi/blob/v0.1.2/src/dwabi/point_policy.py) · [WAPE point adjustment](https://github.com/AbishekSriramulu/dwabi/blob/v0.1.2/src/dwabi/wape.py)
3. For the quantiles, the fitter ranks individual models by FEV's native scaled quantile loss (SQL) on common, mature, disjoint earlier windows of this task. Three complete comparisons admit the three best members. Before that point it uses all seven base members. It orders each selected member's deciles and takes a coordinatewise median. The optional eight-member comparison uses the same qualification rule and a budget of two million input cells. Cell counts use up to 16,384 history steps, the full requested horizon, the number of targets and the number of declared known-future covariates. The selected three can include TimeH2O.
4. When only task-local windows provide enough support, the seven-member fallback shortlist must beat the full-seven median on four earlier, actually issued validation forecasts. It must improve average native SQL and maintain its pairwise win rate against the individual members. Each held forecast is selected using outcomes strictly before that forecast's issue. All four outcomes must have arrived before using their validation decision. Any missing required evidence leaves the ordinary fallback in place.

All 100 submitted tasks leave `max_context_length` unset. Their full pre-cutoff histories are available for selecting earlier calibration windows, and each provider uses its supported context length. The fitter freezes a task's learned choices before its first official issue. On later issues it receives the advancing native history, recomputes input-size eligibility, and issues forecasts with those frozen choices. This benchmark configuration estimates weights and qualifies shortlists independently for each task. [FEV protocol, section 7.1](https://arxiv.org/html/2509.26468v1#S7.SS1)

## Run the benchmark

Use Linux, Python 3.12.13 and a CUDA-capable NVIDIA GPU. Check out FEV at `19bfb6b5ca57a242378b487c725480cbee9dca3c` from the D-wabi fork. Its adapter delegates calibration, fitting and issuance to `dwabi==0.1.2`. D-wabi checks the source digests of FEV's TimesFM-3, Toto-2 and TimeH2O adapters. Authenticate with Hugging Face after H2O grants access to the gated TimeH2O checkpoint. [Pinned FEV checkout](https://github.com/AbishekSriramulu/fev/tree/19bfb6b5ca57a242378b487c725480cbee9dca3c)

Run these commands from that checkout. The pinned providers have conflicting upstream dependency declarations, so install the locked requirements with `--no-deps`. The CUDA PyTorch wheel is installed separately. `PYTHONHASHSEED=0` preserves the pinned TimesFM-3 adapter's categorical encoding. [TimesFM-3 categorical encoding](https://github.com/autogluon/fev/blob/844509e5211292620fbd9d5f24695f1d27da767a/models/timesfm-3/model.py)

```bash
uv venv --python 3.12.13 .venv
source .venv/bin/activate
export PYTHONHASHSEED=0
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128'
grep -Ev '^(dwabi|torch)==' models/dwabi/requirements.txt > .venv/runtime-requirements.txt
uv pip install --no-deps -r .venv/runtime-requirements.txt
uv pip install --no-deps --only-binary dwabi 'dwabi==0.1.2'
uv pip install --no-deps -e .
export CUDA_HOME="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Download and verify inputs before measuring task runtimes. The checkpoint download uses the revisions and files in the installed wheel.

```bash
python -m dwabi.fev.prepare_data ./fev-data
python - <<'PYTHON'
import json
from importlib.resources import files
from pathlib import Path

from huggingface_hub import snapshot_download

from dwabi.fev.providers import checkpoint

root = Path("checkpoints")
pins = json.loads(files("dwabi.fev").joinpath("provider_pins.json").read_text(encoding="utf-8"))
for member, pin in pins.items():
    snapshot_download(
        pin["repo"],
        revision=pin["revision"],
        allow_patterns=list(pin["weights"]),
        local_dir=root / member,
    )
    checkpoint(pin, root, member)
PYTHON
```

Run FEV's stock evaluator with the verified local inputs and an empty in-process forecast state.

```bash
export FEV_DATASETS_PREFIX="$PWD/fev-data"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.72
ulimit -Sn 65536
unset DWABI_FEV_POLICY
python -B -u models/evaluate.py -m dwabi -d -b benchmarks/fev_bench/tasks.yaml -n dwabi \
    -k '{"method":"native_cutoff","checkpoint_root":"checkpoints"}'
```

The normal evaluator call builds the input calendar and fits each task on demand. FEV's `training_time_s` measures data checks, the earlier member forecasts and selection. The first task includes shared calendar setup and any provider initialization performed during its fit. `inference_time_s` measures all official-window member calls and combination through conversion to FEV forecasts. Initialization first needed at inference is charged to that phase. The process can keep pretrained weights resident between tasks. Download benchmark files and checkpoints before measuring. No LLM or agent is involved in task fitting or prediction.

Preparation can also run separately to resume earlier forecasts or allocate one provider per GPU job. The optional forecast bank binds each forecast to its provider, task definition, recipient order, cutoff and the values and missingness in FEV's past and known-future inputs. The final preparation call verifies the files and writes the decisions to a fresh policy file. Start a fresh bank and rebuild prepared policies when upgrading from the earlier FEV adapter. Each receipt binds the installed package version and the provider source.

```bash
export PYTHONHASHSEED=0
for member in t0 ci timestral tirex timesfm3 chronos2 toto2 timeh2o; do
    python -m dwabi.fev.prepare_cutoff benchmarks/fev_bench/tasks.yaml \
        --fev-root . --forecast-bank ./dwabi-earlier --checkpoint-root checkpoints --issue-member "$member"
done
python -m dwabi.fev.prepare_cutoff benchmarks/fev_bench/tasks.yaml ./dwabi-policy.json \
    --fev-root . --forecast-bank ./dwabi-earlier --checkpoint-root checkpoints
export DWABI_FEV_POLICY="$PWD/dwabi-policy.json"
python models/evaluate.py -m dwabi -d -b benchmarks/fev_bench/tasks.yaml -n D-wabi \
    -k '{"method":"native_cutoff","checkpoint_root":"checkpoints"}'
```

Prepared policies support replay. Use the stock evaluator command above with `DWABI_FEV_POLICY` unset to measure cold timing, including fresh calibration forecasts. Stock `Task.evaluation_summary()` scores the issued forecasts.

The package's FEV tests run against the same checkout and installed environment. Set `DWABI_FEV_CHECKOUT` to the FEV checkout and run `python -m pytest test/fev` from the D-wabi source tree.

## Checkpoint attribution

Uses ci-proto, released by Continuous Intelligence. Continuous Intelligence does not endorse or approve this product or service. Checkpoint use is subject to its [original license](../src/dwabi/fev/CI_PROTO_LICENSE.txt). Each other provider retains its published terms, including the [Timestral license](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/LICENSE) and the [TimeH2O noncommercial license](https://huggingface.co/h2oai/timeh2o-v1/tree/1310be782553943a4c520d2810a98ce8f56b2709). The pretraining-overlap metadata follows the declarations for the pinned checkpoints in the FEV checkout. This repository distributes the adapter code and checkpoint references.
