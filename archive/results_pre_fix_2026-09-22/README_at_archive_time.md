# ARS-Stack: Adaptive Resampler-Selection Stacking Ensemble

Per-cluster adaptive resampler selection + confidence-weighted stacking for
imbalanced classification, across binary-extreme, binary-moderate, and
multiclass imbalance. See `ARS-Stack_Project_Plan.pdf` for the full research
plan and literature review this project implements.

**Core idea:** instead of applying one resampling technique (SMOTE, ADASYN,
SMOTE-Tomek, SMOTE-ENN) uniformly across the whole minority class — what
every reviewed paper does — ARS-Stack clusters the minority class into local
neighborhoods, scores each cluster's density/class-overlap, and picks the
best-suited resampler *per cluster*. Base learners (Random Forest, XGBoost,
LightGBM) are then combined by a meta-learner using confidence weights
derived from each learner's own validation-set AUC-PR, instead of trusting
every learner equally.

## Architecture

```
ingest -> preprocess -> minority clustering -> density/overlap scoring
       -> adaptive resampler selection (per cluster) -> merge
       -> confidence-weighted stacking ensemble -> evaluation -> serving
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

All three datasets run through the full adaptive pipeline end to end,
including the multiclass path (each non-majority class is clustered and
resampled independently; see `src/resampling/apply_resampling.py`'s
`apply_adaptive_resampling_multiclass`).

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Full ARS-Stack run for one dataset
python -m src.pipeline --dataset creditcard
python -m src.pipeline --dataset mammography
python -m src.pipeline --dataset yeast_multiclass

# Ablation: fixed global SMOTE instead of adaptive per-cluster selection
python -m src.pipeline --dataset creditcard --no-adaptive-resampling

# Skip MLflow logging (e.g. no tracking server running)
python -m src.pipeline --dataset creditcard --no-mlflow

# Persist the fitted model for FastAPI serving
python -m src.pipeline --dataset creditcard --save-model
```

Or via the orchestrated Prefect DAG (independently retryable, re-runnable
tasks — see `dags/ars_stack_dag.py`):

```bash
python -m dags.ars_stack_dag --dataset creditcard
python -m dags.ars_stack_dag --all
```

Or the full DVC-tracked pipeline (`dvc.yaml`):

```bash
dvc repro
```

## Serving

```bash
python -m src.pipeline --dataset creditcard --save-model
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

## Tests

```bash
pytest tests/ -v
```

## Results (repeated stratified 5-fold CV, 3 repeats — see `reports/ablation_results/`)

| Dataset | Variant | F1 | AUC-PR | G-mean | MCC |
|---|---|---|---|---|---|
| creditcard | ARS-Stack | 0.863 ± 0.029 | 0.857 ± 0.034 | 0.887 ± 0.020 | 0.867 ± 0.028 |
| creditcard | plain SMOTE | 0.842 ± 0.026 | 0.856 ± 0.035 | 0.914 ± 0.018 | 0.842 ± 0.026 |
| creditcard | unweighted stacking | 0.850 ± 0.038 | 0.845 ± 0.040 | 0.873 ± 0.029 | 0.856 ± 0.036 |
| mammography | ARS-Stack | 0.689 ± 0.046 | 0.746 ± 0.039 | 0.782 ± 0.043 | 0.690 ± 0.045 |
| mammography | plain SMOTE | 0.693 ± 0.039 | 0.744 ± 0.051 | 0.862 ± 0.027 | 0.688 ± 0.039 |

Wilcoxon signed-rank tests (`reports/ablation_results/significance_results.json`)
show adaptive resampling significantly improves F1 and MCC on creditcard
(p < 0.01) but *not* AUC-PR, and significantly *reduces* G-mean relative to
plain SMOTE on both datasets — reported honestly per the project's risk
mitigation plan (negative/neutral results are still a valid empirical
finding, not a failure to hide). Confidence-weighted stacking significantly
improves AUC-PR, F1, and MCC over unweighted stacking on creditcard.

On the multiclass yeast dataset, a single run (`reports/pipeline_results/`)
shows plain SMOTE slightly ahead of ARS-Stack on F1/MCC at this dataset's
small per-class sample sizes (some classes have only 3–24 samples) — a
useful counter-example worth including in the write-up, not a bug: very
small minority classes give the clustering step little signal to act on
before the `min_cluster_size_for_smote` fallback (`configs/models/resamplers.yaml`)
kicks in.

## Repository structure

```
src/
  data/           # ingest + preprocess
  clustering/     # minority clustering + density/overlap scoring
  resampling/     # adaptive resampler selection + application (binary + multiclass)
  ensemble/       # base learners + confidence-weighted stacking (binary + multiclass)
  evaluation/     # metrics + Wilcoxon significance testing
  serving/        # FastAPI inference endpoint
  pipeline.py     # end-to-end orchestration, single entry point
dags/             # Prefect flow wrapping the same pipeline as retryable tasks
configs/          # YAML configs (base, datasets, models)
tests/            # pytest unit tests
notebook/         # EDA, baseline reproduction, results analysis, effect sizes
reports/          # metrics, ablation results, significance tests, figures
```
