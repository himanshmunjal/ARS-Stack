"""
dags/ars_stack_dag.py

Orchestrates the ARS-Stack pipeline as a Prefect flow of discrete tasks:

    ingest -> preprocess -> adaptive resampling -> train ensemble -> evaluate -> log

This wraps the SAME functions used by src/pipeline.py's run_pipeline() —
it does not duplicate their logic — but breaks them into separate Prefect
tasks so each stage is independently retryable, individually re-runnable,
and observable in the Prefect UI, which src.pipeline.run_pipeline() (a
single Python function call) doesn't give you on its own.

Why Prefect over Airflow: pure Python, no separate scheduler/webserver
processes to stand up for local development, and the @flow/@task decorator
API is far faster to get working solo — matches the call made earlier in
this project's planning (see reports/architecture.md).

Usage:
    python -m dags.ars_stack_dag --dataset creditcard
    python -m dags.ars_stack_dag --dataset creditcard --resampler smote
    python -m dags.ars_stack_dag --all              # runs the 3 core datasets

    # Or import and call directly:
    from dags.ars_stack_dag import ars_stack_flow
    result = ars_stack_flow(dataset_name="creditcard")

    # To view the DAG graph and task run history in Prefect's UI:
    prefect server start
    # (in another terminal) run the flow, then open http://127.0.0.1:4200

Note: in local/ephemeral mode (no `prefect server start` running), Prefect's
background telemetry heartbeat can log a harmless "database is locked"
SQLite warning on some systems — it doesn't affect the flow result. Silence
it by setting PREFECT_SERVER_ANALYTICS_ENABLED=false in your environment.
"""

import argparse
import json
from pathlib import Path

from prefect import flow, get_run_logger, task

from src.data.ingest import CORE_DATASETS, DATASET_REGISTRY, load_config, load_dataset_config
from src.data.ingest import ingest as ingest_dataset
from src.data.preprocess import preprocess as preprocess_dataset
from src.ensemble.stacking import ConfidenceWeightedStackingEnsemble
from src.evaluation.metrics import compute_metrics
from src.pipeline import _json_default, _safe_mlflow_log, ensure_parent_dir
from src.resampling.apply_resampling import RESAMPLING_METHODS, resample_training_set


# ---------------------------------------------------------------------------
# Task 1: Ingest
# ---------------------------------------------------------------------------

@task(name="ingest", retries=3, retry_delay_seconds=15, log_prints=True)
def ingest_task(dataset_name: str, configs_dir: str) -> None:
    """
    Retries matter here specifically because of real network flakiness seen
    with the mammography dataset's download (see src/data/ingest.py's
    retry+OpenML-fallback logic) — this adds a Prefect-level retry on top
    for the whole task, not just the code has-its-own internal retries.
    """
    logger = get_run_logger()
    logger.info("Ingesting dataset '%s'...", dataset_name)
    ingest_dataset(dataset_name, configs_dir)
    logger.info("Ingest complete for '%s'.", dataset_name)


# ---------------------------------------------------------------------------
# Task 2: Preprocess
# ---------------------------------------------------------------------------

@task(name="preprocess", retries=1, log_prints=True)
def preprocess_task(dataset_name: str, configs_dir: str) -> dict:
    logger = get_run_logger()
    logger.info("Preprocessing dataset '%s'...", dataset_name)
    processed = preprocess_dataset(dataset_name, configs_dir)
    logger.info(
        "Preprocess complete: train=%d, val=%d, test=%d rows",
        len(processed["train"]), len(processed["val"]), len(processed["test"]),
    )
    return processed


# ---------------------------------------------------------------------------
# Task 3: Resampling (adaptive, or a fixed global baseline)
# ---------------------------------------------------------------------------

@task(name="resample", log_prints=True)
def resample_task(
    dataset_name: str,
    configs_dir: str,
    processed: dict,
    resampler: str,
) -> dict:
    logger = get_run_logger()
    ds_cfg = load_dataset_config(dataset_name, configs_dir)["dataset"]
    target_column = ds_cfg["target_column"]
    task_type = "multiclass" if ds_cfg.get("type", "binary") == "multiclass" else "binary"

    train_df = processed["train"]
    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values

    resamplers_cfg = load_config(Path(configs_dir) / "models" / "resamplers.yaml")["resamplers"]
    clustering_cfg = load_config(Path(configs_dir) / "models" / "clustering.yaml")["clustering"]

    logger.info("Running '%s' resampling for '%s' (%s)...", resampler, dataset_name, task_type)
    X_res, y_res, report = resample_training_set(
        X_train, y_train, resampler, task_type, clustering_cfg, resamplers_cfg, positive_label=1,
    )
    logger.info("Resampling complete: %s", report.get("final_class_counts") or {
        "majority": report.get("final_majority_count"), "minority": report.get("final_minority_count"),
    })
    return {"X_train_res": X_res, "y_train_res": y_res, "report": report}


# ---------------------------------------------------------------------------
# Task 4: Train the confidence-weighted stacking ensemble
# ---------------------------------------------------------------------------

@task(name="train_ensemble", log_prints=True)
def train_task(configs_dir: str, resample_result: dict, processed: dict, target_column: str):
    logger = get_run_logger()
    ensemble_cfg = load_config(Path(configs_dir) / "models" / "ensemble.yaml")["ensemble"]

    val_df = processed["val"]
    X_val = val_df.drop(columns=[target_column]).values
    y_val = val_df[target_column].values

    logger.info("Training confidence-weighted ensemble...")
    model = ConfidenceWeightedStackingEnsemble(ensemble_cfg)
    model.fit(resample_result["X_train_res"], resample_result["y_train_res"], X_val, y_val)
    logger.info("Training complete. Base learner weights: %s", model.weights_)
    return model


# ---------------------------------------------------------------------------
# Task 5: Evaluate on the held-out test set
# ---------------------------------------------------------------------------

@task(name="evaluate", log_prints=True)
def evaluate_task(model, processed: dict, target_column: str, task_type: str = "binary") -> dict:
    logger = get_run_logger()
    test_df = processed["test"]
    X_test = test_df.drop(columns=[target_column]).values
    y_test = test_df[target_column].values

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if task_type == "multiclass" else model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba, task_type=task_type)
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


# ---------------------------------------------------------------------------
# Task 6: Log results (MLflow + JSON report)
# ---------------------------------------------------------------------------

@task(name="log_results", log_prints=True)
def log_results_task(
    dataset_name: str,
    configs_dir: str,
    resampler: str,
    model,
    metrics: dict,
    resample_result: dict,
    use_mlflow: bool,
    run_name_suffix: str,
) -> dict:
    logger = get_run_logger()
    base_cfg = load_config(Path(configs_dir) / "base.yaml")
    ensemble_cfg = load_config(Path(configs_dir) / "models" / "ensemble.yaml")["ensemble"]

    suffix = "" if resampler == "adaptive" else f"_{resampler}"
    suffix += f"_{run_name_suffix}" if run_name_suffix else ""
    run_name = f"ars_stack_{dataset_name}{suffix}_prefect"
    log_params = {
        "dataset": dataset_name,
        "resampler": resampler,
        "combiner": model.combiner,
        "confidence_weighting_enabled": ensemble_cfg["confidence_weighting"]["enabled"],
        "orchestrated_via": "prefect",
    }
    _safe_mlflow_log(base_cfg, run_name, log_params, metrics, use_mlflow)

    results = {
        "dataset": dataset_name,
        "resampler": resampler,
        "metrics": metrics,
        "decision_threshold": model.threshold_ if model.n_classes_ == 2 else None,
        "base_learner_weights": model.weights_,
        "resampling_report": resample_result["report"],
    }
    results_path = Path("reports") / "pipeline_results" / f"{dataset_name}{suffix}_prefect_results.json"
    ensure_parent_dir(results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Saved results to %s", results_path)

    return results


# ---------------------------------------------------------------------------
# The flow: wires all 6 tasks together for ONE dataset
# ---------------------------------------------------------------------------

@flow(name="ars-stack-pipeline", log_prints=True)
def ars_stack_flow(
    dataset_name: str = "creditcard",
    configs_dir: str = "configs",
    resampler: str = "adaptive",
    use_mlflow: bool = True,
    run_name_suffix: str = "",
) -> dict:
    """
    The full DAG: ingest -> preprocess -> resample -> train -> evaluate -> log.
    Supports both binary datasets (creditcard, mammography) and the
    multiclass dataset (yeast_multiclass) — mirrors src.pipeline.run_pipeline().
    """
    ds_cfg = load_dataset_config(dataset_name, configs_dir)["dataset"]
    task_type = "multiclass" if ds_cfg.get("type", "binary") == "multiclass" else "binary"
    target_column = ds_cfg["target_column"]

    ingest_task(dataset_name, configs_dir)
    processed = preprocess_task(dataset_name, configs_dir)
    resample_result = resample_task(dataset_name, configs_dir, processed, resampler)
    model = train_task(configs_dir, resample_result, processed, target_column)
    metrics = evaluate_task(model, processed, target_column, task_type)
    results = log_results_task(
        dataset_name, configs_dir, resampler,
        model, metrics, resample_result, use_mlflow, run_name_suffix,
    )

    return {
        "metrics": metrics,
        "weights": model.weights_,
        "resampling_report": resample_result["report"],
        "model": model,
        "results_path": results,
    }


# ---------------------------------------------------------------------------
# A second flow: runs the above for multiple datasets in one go
# ---------------------------------------------------------------------------

@flow(name="ars-stack-multi-dataset", log_prints=True)
def ars_stack_flow_all(
    dataset_names: list = None,
    configs_dir: str = "configs",
    resampler: str = "adaptive",
    use_mlflow: bool = True,
) -> dict:
    """Runs ars_stack_flow for the core datasets (binary and multiclass) by default."""
    if dataset_names is None:
        dataset_names = list(CORE_DATASETS)

    all_results = {}
    for name in dataset_names:
        all_results[name] = ars_stack_flow(
            dataset_name=name, configs_dir=configs_dir,
            resampler=resampler, use_mlflow=use_mlflow,
        )
    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the ARS-Stack pipeline as a Prefect flow.")
    parser.add_argument("--dataset", type=str, default="creditcard",
                         choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--all", action="store_true", help="Run the 3 core datasets.")
    parser.add_argument("--configs-dir", type=str, default="configs")
    parser.add_argument("--resampler", type=str, default="adaptive", choices=list(RESAMPLING_METHODS))
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--run-name-suffix", type=str, default="")
    args = parser.parse_args()

    if args.all:
        ars_stack_flow_all(
            configs_dir=args.configs_dir,
            resampler=args.resampler,
            use_mlflow=not args.no_mlflow,
        )
    else:
        ars_stack_flow(
            dataset_name=args.dataset,
            configs_dir=args.configs_dir,
            resampler=args.resampler,
            use_mlflow=not args.no_mlflow,
            run_name_suffix=args.run_name_suffix,
        )


if __name__ == "__main__":
    main()