#!/usr/bin/env bash
# =========================================================
# run_all.sh — run the complete ARS-Stack project end to end, on any machine.
#
#   bash scripts/run_all.sh --quick    # smoke test of every step first (~20-30 min)
#   bash scripts/run_all.sh            # full run on all 28 datasets (see README for time; resumable)
#
# What it does, in order:
#   0. Checks Python >= 3.11, creates venv/ and installs requirements.txt if
#      needed, and checks the raw data (see "Data" below).
#   1. Unit tests
#   2. dvc repro: ingest + clean + dedup + split + scale; downloads mammography,
#      yeast and the 25 benchmark datasets automatically (needs internet)
#   3. Literature baseline reproduction
#   4. Single-split ARS-Stack runs + saved models for serving
#   5. Repeated-CV ablation study + all statistics (the paper's results)
#   Everything is logged to reports/run_all[_quick].log.
#
# Data: data/raw/creditcard/creditcard.csv is the one file that cannot be
# downloaded automatically (Kaggle login required). Copy it from the original
# machine, or download it from
#   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# and place it at data/raw/creditcard/creditcard.csv (150 MB) before running.
#
# If interrupted (sleep, power loss, Ctrl-C), run the same command again:
# finished steps are fast to redo and the CV step resumes from the last fold.
#
# Windows: run from Git Bash or WSL. Set PYTHON=/path/to/python3.11 if the
# right interpreter isn't found automatically.
# =========================================================
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=""
[[ "${1:-}" == "--quick" ]] && QUICK="--quick"
DATASETS=(creditcard mammography yeast_multiclass)

mkdir -p reports
LOG="reports/run_all${QUICK:+_quick}.log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "================ $* ================  ($(date '+%Y-%m-%d %H:%M'))"; }
die() { echo; echo "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------------------
step "0/5  Environment + data checks"
# ---------------------------------------------------------------------------

# Venv interpreter path differs between macOS/Linux and Windows.
venv_python() {
  if [[ -x venv/bin/python ]]; then echo venv/bin/python
  elif [[ -x venv/Scripts/python.exe ]]; then echo venv/Scripts/python.exe
  fi
}

python_ok() {  # $1 = interpreter; true if it runs and is >= 3.11
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

if [[ -z "$(venv_python)" ]]; then
  BOOT="${PYTHON:-}"
  if [[ -z "$BOOT" ]]; then
    for cand in python3.11 python3.12 python3 python py; do
      if command -v "$cand" >/dev/null 2>&1 && python_ok "$cand"; then BOOT="$cand"; break; fi
    done
  fi
  [[ -n "$BOOT" ]] && python_ok "$BOOT" || die "Python >= 3.11 not found. Install Python 3.11 (python.org) or set PYTHON=/path/to/python3.11."
  echo "Creating venv/ with $("$BOOT" --version) ..."
  "$BOOT" -m venv venv
  PY="$(venv_python)"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
else
  PY="$(venv_python)"
  python_ok "$PY" || die "venv/ uses Python < 3.11. Delete venv/ and re-run so it is recreated."
  echo "Using existing venv/ ($("$PY" --version)); checking requirements are installed ..."
  "$PY" -m pip install -q -r requirements.txt
fi

# dvc.yaml stages call plain `python`; make that the venv's interpreter.
export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"

[[ -f data/raw/creditcard/creditcard.csv ]] || die "data/raw/creditcard/creditcard.csv is missing.
  It can't be downloaded automatically (Kaggle login needed). Copy it from the original
  machine, or download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  and place it at data/raw/creditcard/creditcard.csv, then re-run this script."

if command -v git >/dev/null 2>&1 && [[ -n "$(git status --porcelain -- src configs 2>/dev/null)" ]]; then
  echo "WARNING: src/ or configs/ has uncommitted changes — results you publish should come from a committed, tagged version."
fi
echo "Machine: $(uname -sm), $("$PY" -c 'import os; print(os.cpu_count())') CPUs"

# GPU: XGBoost uses it when configs/models/ensemble.yaml -> ensemble.device is
# "auto"/"cuda". If an NVIDIA GPU is present but XGBoost can't use it, stop
# instead of silently spending the whole run on CPU.
DEVICE="$("$PY" -c 'from src.ensemble.base_learners import resolve_device as r; import yaml; print(r(yaml.safe_load(open("configs/models/ensemble.yaml"))["ensemble"].get("device", "cpu")))')"
echo "XGBoost device: $DEVICE (Random Forest, LightGBM and resampling always run on CPU)"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  [[ "$DEVICE" == "cuda" ]] || [[ "${ALLOW_CPU:-}" == "1" ]] || die "An NVIDIA GPU is present but XGBoost can't use it.
  Check the NVIDIA driver/CUDA install (xgboost from PyPI on Linux/Windows includes CUDA),
  or re-run with ALLOW_CPU=1 to deliberately run everything on CPU."
fi

# Keep the machine awake for the long steps (the OS sleeping pauses the run).
KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then KEEP_AWAKE=(caffeinate -i)            # macOS
elif command -v systemd-inhibit >/dev/null 2>&1; then KEEP_AWAKE=(systemd-inhibit --what=sleep:idle --why=ars-stack)  # Linux
else echo "NOTE: can't prevent sleep automatically here — disable sleep in the power settings for the run."
fi
run() { ${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} "$PY" "$@"; }  # form works with an empty array on bash 3.2

# ---------------------------------------------------------------------------
step "1/5  Unit tests"
run -m pytest tests/ -q

step "2/5  Data: ingest (downloads) + clean + dedup + split + scale (DVC pipeline)"
run -m dvc repro

step "3/5  Literature baseline reproduction (plain SMOTE + RF / XGB / simple stacking)"
run -m src.experiments.baseline_reproduction --dataset creditcard
run -m src.experiments.baseline_reproduction --dataset mammography

step "4/5  ARS-Stack on the fixed train/val/test split + saved models for serving"
for ds in "${DATASETS[@]}"; do
  run -m src.pipeline --dataset "$ds" --save-model
  run -m src.pipeline --dataset "$ds" --resampler smote     # fixed-SMOTE reference on the same split
done

step "5/5  Repeated-CV ablation study + statistics (all 28 datasets)"
run -m src.experiments.run_ablation $QUICK

step "Done — results"
echo "  reports/baseline_results_*.csv         literature baseline reproduction"
echo "  reports/pipeline_results/*.json        single-split runs (metrics, threshold, weights, resampler choices)"
echo "  models/*.joblib                        model bundles served by src/serving/app.py"
if [[ -n "$QUICK" ]]; then
  echo "  reports/ablation_results_quick/        QUICK smoke-test CV results (not for the paper)"
else
  echo "  reports/ablation_results/summary.md    the paper's results: cross-dataset ranks + tests, per-dataset tests"
  echo "  reports/ablation_results/environment.json   code version + packages + machine, for the paper"
fi
echo "  full log: $LOG"
