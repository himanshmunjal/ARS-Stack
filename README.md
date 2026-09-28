# ARS-Stack: Adaptive Resampler-Selection Stacking Ensemble

Per-cluster adaptive resampler selection + confidence-weighted stacking for
imbalanced classification, across binary-extreme, binary-moderate, and
multiclass imbalance. See `ARS-Stack_Project_Plan.pdf` for the full research
plan and literature review this project implements.

**Core idea:** instead of applying one resampling technique (SMOTE, ADASYN,
SMOTE-Tomek, SMOTE-ENN) uniformly across the whole minority class — what
every reviewed paper does — ARS-Stack clusters the minority class into local
neighborhoods, scores each cluster's density/class-overlap, and picks the
best-suited resampler *per cluster*. Every variant oversamples to the same
overall class balance (`target_balance_ratio`), so comparisons isolate *which*
resampler is used where, not *how much* data is added. Base learners (Random
Forest, XGBoost, LightGBM) are then combined by a confidence-weighted average,
each learner weighted by its own validation-set AUC-PR, instead of trusting
every learner equally; the decision threshold is tuned on the validation set.

## Architecture

```
ingest -> preprocess -> minority clustering -> density/overlap scoring
       -> adaptive resampler selection (per cluster) -> merge
       -> confidence-weighted ensemble (+ val-tuned threshold) -> evaluation -> serving
```

Wrapped in an MLOps layer: DVC (data versioning), MLflow (experiment
tracking), Prefect (workflow orchestration), Docker (reproducible env),
GitHub Actions (CI), FastAPI (model serving).

## Datasets

| Dataset | Type | Imbalance | Source |
|---|---|---|---|
| Credit Card Fraud (ULB) | Binary, extreme | ≈1:578 | Kaggle |
| Mammography | Binary, moderate-high | ≈1:42 | imbalanced-learn / OpenML |
| Yeast (UCI) | Multiclass, 10 classes | up to ≈1:92 | UCI ML Repository |

Plus **25 binary benchmark datasets** from imbalanced-learn's collection
(`configs/datasets/benchmarks.yaml`: imbalance ratios 8.6–130, 330–145k rows,
7–617 features; every dataset in the collection except the two that overlap the core datasets), used in the repeated-CV study so conclusions rest on a
cross-dataset comparison (Friedman test + post-hoc; Demšar 2006), not only on
three case studies. Exact duplicate rows are removed from every dataset before
splitting (`configs/base.yaml -> preprocessing.drop_duplicates`); this removes
30% of mammography's rows, so its scores are not directly comparable to papers
that used the raw file.

All three core datasets run through the full adaptive pipeline end to end,
including the multiclass path (each non-majority class is clustered and
resampled independently; see `src/resampling/apply_resampling.py`'s
`apply_adaptive_resampling_multiclass`).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # activating puts `python` on PATH (dvc.yaml needs it)
pip install -r requirements.txt
dvc pull            # only if data/raw/ is missing (restores it from the DVC remote)
```

## Running the complete project

One command does everything — environment, data, tests, training, results
(every step saves its results; see the table below):

```bash
bash scripts/run_all.sh --quick    # smoke test of every step first (~20-30 min)
bash scripts/run_all.sh            # full run (~18-24 h on 8 cores; re-run the same command to resume)
```

On a new machine you only need: Python >= 3.11, internet, and one file.
The script creates `venv/`, installs `requirements.txt`, downloads mammography,
yeast and the 25 benchmark datasets, keeps the machine awake (macOS/Linux)
and logs to `reports/run_all.log`. The one file it cannot download is
`data/raw/creditcard/creditcard.csv` (Kaggle login required) — copy it from
the original machine or download it from
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> first; the script
stops with instructions if it's missing. On Windows, run it from Git Bash or WSL.

Or step by step, in this order:

```bash
# 0. (optional, separate terminal) MLflow tracking server — runs are logged if it's up
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 1. Tests
pytest tests/ -q

# 2. Data: ingest -> clean -> dedup -> stratified split -> scale, + the 25 benchmarks (DVC pipeline)
dvc repro

# 3. Literature baseline reproduction (plain SMOTE + RF / XGBoost / simple stacking)
python -m src.experiments.baseline_reproduction --dataset creditcard
python -m src.experiments.baseline_reproduction --dataset mammography

# 4. ARS-Stack on the fixed split, saving the model for serving (+ a fixed-SMOTE reference)
python -m src.pipeline --dataset creditcard --save-model
python -m src.pipeline --dataset mammography --save-model
python -m src.pipeline --dataset yeast_multiclass --save-model
python -m src.pipeline --dataset creditcard --resampler smote      # also: adasyn, smote_tomek, smote_enn, none

# 5. Repeated-CV ablation study + significance tests (the numbers for the paper)
python -m src.experiments.run_ablation --quick     # smoke test -> reports/ablation_results_quick/
python -m src.experiments.run_ablation             # full: 28 datasets x 9 variants x 5 folds x 3 repeats

# 6. Serve the saved models
uvicorn src.serving.app:app --port 8000
```

Useful options for step 5: `--datasets core` / `benchmarks` / dataset names,
`--variants ars_stack plain_smote equal_average`, `--summarize-only` (rebuild
tables/plot from saved folds), `--fresh` (discard saved folds after a config change).

Orchestrated alternative to step 4 (Prefect flow, independently retryable tasks):

```bash
python -m dags.ars_stack_dag --all
```

### Where results are saved

| Path | Contents |
|---|---|
| `reports/baseline_results_<dataset>.csv` | literature baseline reproduction |
| `reports/pipeline_results/<dataset>[_<resampler>]_results.json` | single-split metrics, tuned threshold, weights, per-cluster resampler choices |
| `models/<dataset>.joblib` | model bundles served by FastAPI |
| `reports/ablation_results/summary.md` | everything below as readable tables |
| `reports/ablation_results/cross_dataset_ranks.csv` | average rank of every variant per metric (across datasets) |
| `reports/ablation_results/cross_dataset_tests.csv/.json` | Friedman / Iman-Davenport + Holm-corrected Wilcoxon of ARS-Stack vs each variant across datasets, win/tie/loss |
| `reports/ablation_results/significance_results.csv/.json` | per dataset: corrected resampled t-test (primary) and Wilcoxon, both Holm-corrected per dataset, rank-biserial effect size |
| `reports/ablation_results/aggregated_summary.csv`, `summary_long.csv` | mean ± std per dataset × variant × metric |
| `reports/ablation_results/resampler_choices.csv` | every cluster's scores and chosen/used resampler |
| `reports/ablation_results/average_ranks.png`, `ablation_comparison.png` | rank chart (all datasets), mean ± std chart (core datasets) |
| `reports/ablation_results/environment.json` | git commit, package versions, machine — cite in the paper |
| `reports/ablation_results/folds/*.json` | raw per-fold results (resume state) |
| `mlflow.db` | MLflow runs (when a tracking server was running) |

### Ablation variants (each compared against `ars_stack`)

| Variant | What the comparison isolates |
|---|---|
| `equal_average` | confidence weighting vs equal-weight averaging |
| `meta_stacking` | weighted average vs logistic-regression meta-learner stacking |
| `ars_dbscan` | KMeans vs DBSCAN clustering |
| `plain_smote`, `plain_adasyn`, `plain_smote_tomek`, `plain_smote_enn` | adaptive per-cluster selection vs one global resampler |
| `no_resampling` | whether resampling helps at all |

### Statistical protocol

- **Per dataset:** 5-fold stratified CV × 3 repeats (15 paired folds). Primary
  test: Nadeau–Bengio corrected resampled t-test (Bouckaert & Frank 2004 form),
  which accounts for the overlapping training sets of CV folds; Wilcoxon
  signed-rank reported alongside. Holm correction over every
  (comparison × metric) test of a dataset. Effect size: matched-pairs rank-biserial r.
- **Across datasets:** average ranks, Friedman / Iman–Davenport test, then
  ARS-Stack vs each variant with a Wilcoxon signed-rank test over per-dataset
  means (Benavoli et al. 2016), Holm-corrected within each metric.
- **No leakage:** per fold, the scaler, resampler, confidence weights and
  decision threshold are fitted only on that fold's training portion (weights
  and threshold on an inner 15% validation split).
- **Fixed a priori:** all hyperparameters and the resampler-selection
  thresholds (`configs/models/resamplers.yaml`) were set before any results
  were produced and are not tuned on test folds.

### Before the run you'll publish

```bash
pytest tests/ -q && git status              # tests pass, nothing uncommitted in src/ or configs/
git tag -a paper-run-v1 -m "Code used for the published experiments"
dvc push                                     # data snapshot matching dvc.lock
caffeinate -i python -m src.experiments.run_ablation     # macOS: keep the machine awake
```

`environment.json` then records that commit; the run warns if `src/` or
`configs/` have uncommitted changes. Saved folds are tied to a fingerprint of
the configs and the result-affecting code, so a code change can never mix
with earlier folds (use `--fresh` to restart).

Approximate run time of the full study on an 8-core Apple M1 / 8 GB
(similar on an 8 vCPU VM; roughly halves on a 16-core CPU): creditcard
~12–15 h, protein_homo ~10 h, isolet + webpage ~4–5 h, the other 23 datasets
~4–6 h — about 30–36 h in total (measured on full-size folds of creditcard and
protein_homo, extrapolated from the `--quick` run for the rest). Random Forest
dominates and is CPU-only, so an NVIDIA GPU (used by XGBoost) saves only
~5–10%; CPU cores are what matter. Peak memory stays under ~2 GB.

## Serving

```bash
uvicorn src.serving.app:app --reload

curl http://localhost:8000/models/creditcard
curl -X POST http://localhost:8000/predict/creditcard \
  -H "Content-Type: application/json" \
  -d '{"features": {"Time": 0, "V1": -1.36, "...": "..."}}'
```

## Docker

```bash
docker compose up --build                 # FastAPI + MLflow UI
docker compose run --rm pipeline python -m src.pipeline --dataset creditcard --save-model
```

## Results

Final results are produced by step 5 above into `reports/ablation_results/`
(see `summary.md` there). Results from before the 2026-09-22 methodology fixes
are kept in `archive/results_pre_fix_2026-09-22/`, with `ARCHIVE_NOTES.md`
explaining why they should not be reported as final (unequal oversampling
between variants, confidence weights cancelled by the meta-learner,
rank-forced resampler selection, fixed 0.5 threshold, duplicate rows).

## Repository structure

```
src/
  data/           # ingest + preprocess (clean, dedup, split, scale)
  clustering/     # minority clustering + density/overlap scoring
  resampling/     # adaptive resampler selection + application, global baselines
  ensemble/       # base learners + confidence-weighted ensemble (binary + multiclass)
  evaluation/     # metrics + Wilcoxon / Holm / effect sizes
  experiments/    # repeated-CV ablation study, literature baseline reproduction
  serving/        # FastAPI inference endpoint
  pipeline.py     # end-to-end single-split run, single entry point
dags/             # Prefect flow wrapping the same pipeline as retryable tasks
configs/          # YAML configs (base, datasets, models)
scripts/          # run_all.sh — the whole project in order
tests/            # pytest unit tests
notebook/         # EDA (01); 02-04 are superseded by src/experiments/ (kept for reference)
reports/          # generated results (see "Where results are saved")
archive/          # results produced before the 2026-09-22 fixes
```
