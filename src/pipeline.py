"""
src/pipeline.py

The end-to-end ARS-Stack pipeline, chaining every module built so far:

    ingest -> preprocess -> adaptive resampling -> stacking ensemble -> evaluation

This is the file dags/ars_stack_dag.py (Week 10) will wrap in an
Airflow/Prefect DAG, and the file Week 9's ablation-study notebook will call
repeatedly (once per dataset, once per config variant) to produce the
paired fold results src.evaluation.significance_test compares.

Supports both binary datasets (creditcard, mammography) and the multiclass
dataset (yeast_multiclass): src.resampling.apply_resampling.resample_training_set
and src.ensemble.stacking's ConfidenceWeightedStackingEnsemble both branch
internally on class count.

This is ONE train/val/test run on the DVC-tracked split — the model you save
and serve. The statistically tested comparison (repeated CV + Wilcoxon) is
src/experiments/run_ablation.py.

Usage:
    python -m src.pipeline --dataset creditcard --save-model
    python -m src.pipeline --dataset mammography --resampler smote          # fixed global SMOTE baseline
    python -m src.pipeline --dataset creditcard --no-mlflow                  # skip MLflow logging
    python -m src.pipeline --dataset yeast_multiclass                        # multiclass path
"""

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np

from src.data.ingest import DATASET_REGISTRY, ensure_parent_dir, ingest, load_config, load_dataset_config
from src.data.preprocess import preprocess
from src.ensemble.stacking import ConfidenceWeightedStackingEnsemble
from src.evaluation.metrics import compute_metrics
from src.resampling.apply_resampling import RESAMPLING_METHODS, resample_training_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# MLflow helper — never let logging failures break the pipeline run itself
# ---------------------------------------------------------------------------

def _mlflow_server_reachable(tracking_uri: str, timeout_seconds: float = 1.5) -> bool:
    """
    Fast reachability check before handing off to MLflow's own client.
    MLflow's default HTTP retry/backoff takes several MINUTES to give up
    when the tracking server is down (confirmed during testing) — far too
    slow for a pipeline that should just skip logging and move on. A quick
    raw socket check avoids that entirely.
    """
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(tracking_uri)
    if parsed.scheme not in ("http", "https"):
        return True  # e.g. a local "file:" or "sqlite:" URI — no network check needed

    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _safe_mlflow_log(base_cfg: dict, run_name: str, params: dict, metrics: dict, use_mlflow: bool):
    if not use_mlflow:
        return

    tracking_uri = base_cfg["project"]["mlflow_tracking_uri"]
    if not _mlflow_server_reachable(tracking_uri):
        logger.warning(
            "MLflow tracking server at %s is not reachable — skipping logging "
            "for this run. Run `mlflow ui` or `mlflow server` to enable it, or "
            "pass use_mlflow=False to silence this check.", tracking_uri,
        )
        return

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(base_cfg["project"]["mlflow_experiment_name"])
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception as exc:
        logger.warning(
            "MLflow logging failed (%s) — continuing without it. Run "
            "`mlflow ui` or `mlflow server` on the configured tracking URI "
            "to enable logging, or pass use_mlflow=False to skip it.", exc,
        )


# ---------------------------------------------------------------------------
# The full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset_name: str,
    configs_dir: str = "configs",
    resampler: str = "adaptive",
    use_mlflow: bool = True,
    run_name_suffix: str = "",
    save_model: bool = False,
) -> dict:
    """
    Runs the complete ARS-Stack pipeline for one dataset and returns a
    results dict: {"metrics", "resampling_report", "weights", "threshold",
    "config", "model"}. Also writes a JSON summary to reports/pipeline_results/.

    Args:
        resampler: "adaptive" = the real ARS-Stack path; "none", "smote",
            "adasyn", "smote_tomek", "smote_enn" = fixed global baselines.
        use_mlflow: whether to log this run to MLflow (safe to leave True —
            logging failures are caught and logged as warnings, never crash the run).
    """
    if resampler not in RESAMPLING_METHODS:
        raise ValueError(f"Unknown resampler '{resampler}' (expected one of {RESAMPLING_METHODS}).")
    base_cfg = load_config(Path(configs_dir) / "base.yaml")
    ds_full_cfg = load_dataset_config(dataset_name, configs_dir)
    ds_cfg = ds_full_cfg["dataset"]
    task_type = "multiclass" if ds_cfg.get("type", "binary") == "multiclass" else "binary"

    logger.info("=" * 70)
    logger.info(
        "Running ARS-Stack pipeline for '%s' (task_type=%s, resampler=%s)",
        dataset_name, task_type, resampler,
    )
    logger.info("=" * 70)

    # --- Step 1: ingest (no-op if raw file already exists) ---
    ingest(dataset_name, configs_dir)

    # --- Step 2: preprocess (clean, encode, split, scale) ---
    processed = preprocess(dataset_name, configs_dir)
    train_df, val_df, test_df = processed["train"], processed["val"], processed["test"]

    target_column = ds_cfg["target_column"]

    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values
    X_val = val_df.drop(columns=[target_column]).values
    y_val = val_df[target_column].values
    X_test = test_df.drop(columns=[target_column]).values
    y_test = test_df[target_column].values

    # --- Step 3: resampling (adaptive, or a fixed global baseline) ---
    resamplers_cfg = load_config(Path(configs_dir) / "models" / "resamplers.yaml")["resamplers"]
    clustering_cfg = load_config(Path(configs_dir) / "models" / "clustering.yaml")["clustering"]

    X_train_res, y_train_res, resampling_report = resample_training_set(
        X_train, y_train, resampler, task_type, clustering_cfg, resamplers_cfg,
        positive_label=1,  # preprocess.py encodes the positive class to 1
    )

    # --- Step 4: train the confidence-weighted stacking ensemble ---
    ensemble_full_cfg = load_config(Path(configs_dir) / "models" / "ensemble.yaml")
    ensemble_cfg = ensemble_full_cfg["ensemble"]

    model = ConfidenceWeightedStackingEnsemble(ensemble_cfg)
    model.fit(X_train_res, y_train_res, X_val, y_val)

    # --- Step 5: evaluate on the held-out test set ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if task_type == "multiclass" else model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba, task_type=task_type)

    logger.info("Test metrics for '%s': %s", dataset_name, {k: round(v, 4) for k, v in metrics.items()})
    logger.info("Base learner confidence weights: %s", {k: round(v, 4) for k, v in model.weights_.items()})
    if task_type == "binary":
        logger.info("Decision threshold (tuned on validation set): %.4f", model.threshold_)

    # --- Step 6: log to MLflow (best-effort, never crashes the run) ---
    suffix = "" if resampler == "adaptive" else f"_{resampler}"
    suffix += f"_{run_name_suffix}" if run_name_suffix else ""
    run_name = f"ars_stack_{dataset_name}{suffix}"
    log_params = {
        "dataset": dataset_name,
        "task_type": task_type,
        "resampler": resampler,
        "target_balance_ratio": resamplers_cfg.get("target_balance_ratio", 1.0),
        "selection_mode": resamplers_cfg["selection"].get("mode"),
        "combiner": model.combiner,
        "confidence_weighting_enabled": ensemble_cfg["confidence_weighting"]["enabled"],
        "threshold_tuning": ensemble_cfg.get("threshold_tuning", {}).get("enabled", False),
    }
    log_metrics = dict(metrics)
    if task_type == "binary":
        log_metrics["decision_threshold"] = model.threshold_
    _safe_mlflow_log(base_cfg, run_name, log_params, log_metrics, use_mlflow)

    # --- Step 7: save a JSON summary ---
    results = {
        "dataset": dataset_name,
        "resampler": resampler,
        "metrics": metrics,
        "decision_threshold": model.threshold_ if task_type == "binary" else None,
        "base_learner_weights": model.weights_,
        "params": log_params,
        "resampling_report": resampling_report,
    }
    results_path = Path("reports") / "pipeline_results" / f"{dataset_name}{suffix}_results.json"
    ensure_parent_dir(results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Saved pipeline results to %s", results_path)

    # --- Step 8 (optional): persist the fitted model + scaler for serving ---
    model_path = None
    if save_model:
        import joblib

        scaler_path = Path(ds_cfg["processed_file"]).parent / "scaler.joblib"
        model_path = Path("models") / f"{dataset_name}.joblib"
        ensure_parent_dir(model_path)
        joblib.dump(
            {
                "model": model,
                "task_type": task_type,
                "dataset": dataset_name,
                "feature_columns": [c for c in train_df.columns if c != target_column],
                "scaler_path": str(scaler_path) if scaler_path.exists() else None,
            },
            model_path,
        )
        logger.info("Saved fitted model bundle to %s (for FastAPI serving)", model_path)

    return {
        "metrics": metrics,
        "resampling_report": resampling_report,
        "weights": model.weights_,
        "threshold": model.threshold_,
        "config": {"dataset": ds_cfg, "ensemble": ensemble_cfg, "resamplers": resamplers_cfg},
        "model": model,
        "model_path": str(model_path) if model_path else None,
    }


def _json_default(obj):
    """Handles numpy types that json.dump can't serialize natively."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end ARS-Stack pipeline.")
    parser.add_argument(
        "--dataset", type=str, default="creditcard",
        choices=list(DATASET_REGISTRY.keys()),
        help="Which dataset to run.",
    )
    parser.add_argument("--configs-dir", type=str, default="configs")
    parser.add_argument(
        "--resampler", type=str, default="adaptive", choices=list(RESAMPLING_METHODS),
        help="adaptive = ARS-Stack; none/smote/adasyn/smote_tomek/smote_enn = fixed global baselines.",
    )
    parser.add_argument(
        "--no-adaptive-resampling", action="store_true",
        help="Shortcut for --resampler smote (kept for backwards compatibility).",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging for this run.")
    parser.add_argument("--run-name-suffix", type=str, default="", help="Optional suffix for the run/results file name.")
    parser.add_argument(
        "--save-model", action="store_true",
        help="Persist the fitted model bundle to models/<dataset>.joblib for FastAPI serving.",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            dataset_name=args.dataset,
            configs_dir=args.configs_dir,
            save_model=args.save_model,
            resampler="smote" if args.no_adaptive_resampling else args.resampler,
            use_mlflow=not args.no_mlflow,
            run_name_suffix=args.run_name_suffix,
        )
    except Exception as exc:
        logger.error("Pipeline run failed for '%s': %s", args.dataset, exc)
        raise


if __name__ == "__main__":
    main()