"""
src/experiments/baseline_reproduction.py

Week 2-3 baseline reproduction (script version of
notebook/02_baseline_reproduction.ipynb): plain SMOTE + Random Forest,
plain SMOTE + XGBoost, and simple (unweighted) RF+XGB stacking with a
logistic-regression meta-learner, trained on the DVC-tracked train split and
scored on the test split — the setups used by Papers 7, 9 and 10. Settings
come from configs/models/ensemble.yaml -> baseline. Uses a fixed 0.5
threshold, as those papers do.

Output: reports/baseline_results_<dataset>.csv (our numbers, plus the
papers' reference numbers for creditcard). Run `python -m src.data.preprocess --dataset creditcard`
(or `dvc repro`) first.

Usage:
    python -m src.experiments.baseline_reproduction                  # creditcard
    python -m src.experiments.baseline_reproduction --dataset mammography
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.ingest import load_config, load_dataset_config
from src.ensemble.base_learners import resolve_device
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger("baseline_reproduction")


def main():
    parser = argparse.ArgumentParser(description="Reproduce the literature's fixed-SMOTE baselines.")
    parser.add_argument("--dataset", default="creditcard", choices=["creditcard", "mammography"])
    parser.add_argument("--configs-dir", default="configs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ds_cfg = load_dataset_config(args.dataset, args.configs_dir)["dataset"]
    ensemble_yaml = load_config(Path(args.configs_dir) / "models" / "ensemble.yaml")
    baseline_cfg = ensemble_yaml["baseline"]
    device = resolve_device(ensemble_yaml["ensemble"].get("device", "cpu"))
    logger.info("XGBoost device: %s", device)
    target = ds_cfg["target_column"]
    processed_dir = Path(ds_cfg["processed_file"]).parent

    splits = {}
    for split in ("train", "test"):
        path = processed_dir / f"{args.dataset}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.data.preprocess --dataset {args.dataset}` first.")
        df = pd.read_csv(path)
        splits[split] = (df.drop(columns=[target]).values, df[target].values)
    (X_train, y_train), (X_test, y_test) = splits["train"], splits["test"]

    smote_cfg = baseline_cfg["smote"]
    X_res, y_res = SMOTE(k_neighbors=smote_cfg["k_neighbors"], random_state=smote_cfg["random_state"]).fit_resample(X_train, y_train)
    logger.info("SMOTE: %d -> %d training rows", len(y_train), len(y_res))

    rf_cfg = baseline_cfg["models"]["random_forest"]
    xgb_cfg = {**baseline_cfg["models"]["xgboost"], "device": device}
    stack_cfg = baseline_cfg["simple_stacking"]
    models = {
        "RandomForest + SMOTE": RandomForestClassifier(**rf_cfg),
        "XGBoost + SMOTE": XGBClassifier(**xgb_cfg),
        "Simple Stacking (RF+XGB, LR meta) + SMOTE": StackingClassifier(
            estimators=[("random_forest", RandomForestClassifier(**rf_cfg)), ("xgboost", XGBClassifier(**xgb_cfg))],
            final_estimator=LogisticRegression(**baseline_cfg["models"]["logistic_regression"]),
            cv=stack_cfg["cv_folds"], passthrough=stack_cfg["passthrough"],
        ),
    }

    rows = []
    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_res, y_res)
        metrics = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])
        logger.info("%s: %s", name, {k: round(v, 4) for k, v in metrics.items()})
        rows.append({"dataset": args.dataset, "model": name, "source": "this project", **metrics})

    if args.dataset == "creditcard":
        for paper, ref in baseline_cfg["reference_numbers"].items():
            rows.append({"dataset": args.dataset, "model": paper, "source": "literature", **ref})
    for row in rows:
        if row["source"] == "this project":
            row["xgboost_device"] = device

    out_path = Path("reports") / f"baseline_results_{args.dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
