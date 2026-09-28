"""
src/data/preprocess.py

Takes the raw CSV produced by src/data/ingest.py and turns it into clean,
split, scaled train/val/test CSVs under data/processed/<dataset_name>/.

Steps applied (generic across all 3 datasets, driven entirely by config):
    1. Load raw CSV
    2. Drop id columns (e.g. sequence_name in yeast)
    3. Drop rows with missing values in feature/target columns, then exact
       duplicate rows (configs/base.yaml -> preprocessing.drop_duplicates)
       so no duplicate can land in both train and test
    4. Encode the target:
         - binary datasets  -> {0, 1}, 1 = positive/minority class
         - multiclass datasets -> integer-encoded, mapping saved to JSON
    5. Stratified train / val / test split (ratios from configs/base.yaml)
    6. Fit scaler on TRAIN ONLY, apply to val/test (no data leakage)
    7. Save processed CSVs + scaler + label mapping artifacts

Usage:
    python -m src.data.preprocess --dataset creditcard
    python -m src.data.preprocess --dataset mammography
    python -m src.data.preprocess --dataset yeast_multiclass
    python -m src.data.preprocess --dataset all
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.ingest import (
    DATASET_GROUPS,
    DATASET_REGISTRY,
    expand_dataset_names,
    ensure_parent_dir,
    load_config,
    load_dataset_config,
    log_class_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def drop_id_columns(df: pd.DataFrame, ds_cfg: dict) -> pd.DataFrame:
    id_columns = ds_cfg.get("id_columns", []) or []
    present = [c for c in id_columns if c in df.columns]
    if present:
        logger.info("Dropping id columns: %s", present)
        df = df.drop(columns=present)
    return df


def drop_missing_rows(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(axis=0, how="any")
    n_after = len(df)
    if n_after < n_before:
        logger.info("Dropped %d rows with missing values (%d -> %d)", n_before - n_after, n_before, n_after)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found after cleaning.")
    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        logger.info("Dropped %d exact duplicate rows (%d -> %d)", n_before - len(df), n_before, len(df))
    return df


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def encode_target(df: pd.DataFrame, ds_cfg: dict, artifacts_dir: Path) -> pd.DataFrame:
    """
    Binary datasets: map positive_label -> 1, everything else -> 0.
    Multiclass datasets: integer-encode with sklearn's LabelEncoder and
    save the class -> integer mapping to JSON for later interpretation.
    """
    target_column = ds_cfg["target_column"]
    dataset_type = ds_cfg.get("type", "binary")

    if dataset_type == "binary":
        positive_label = ds_cfg["positive_label"]
        df[target_column] = (df[target_column] == positive_label).astype(int)
        logger.info("Encoded binary target: positive_label=%s -> 1, else -> 0", positive_label)
    elif dataset_type == "multiclass":
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column].astype(str))
        mapping = {cls: int(idx) for idx, cls in enumerate(encoder.classes_)}
        ensure_parent_dir(artifacts_dir / "label_mapping.json")
        with open(artifacts_dir / "label_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)
        logger.info("Encoded multiclass target (%d classes). Mapping saved to %s",
                    len(mapping), artifacts_dir / "label_mapping.json")
    else:
        raise ValueError(f"Unknown dataset type '{dataset_type}' (expected 'binary' or 'multiclass').")

    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, target_column: str, base_cfg: dict):
    split_cfg = base_cfg["split"]
    seed = base_cfg["project"]["seed"]
    stratify_col = df[target_column] if split_cfg.get("stratify", True) else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=seed,
        shuffle=split_cfg.get("shuffle", True),
        stratify=stratify_col,
    )

    # val_size in config is a fraction of the ORIGINAL dataset; convert it to
    # a fraction of the remaining train_val_df before splitting again.
    relative_val_size = split_cfg["val_size"] / (1 - split_cfg["test_size"])
    stratify_col_2 = train_val_df[target_column] if split_cfg.get("stratify", True) else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=split_cfg.get("shuffle", True),
        stratify=stratify_col_2,
    )

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Scaling (fit on train only — avoids leakage from val/test into scaling stats)
# ---------------------------------------------------------------------------

def scaling_columns(ds_cfg: dict, all_columns) -> list:
    """Which columns the dataset config says to standard-scale."""
    columns_spec = ds_cfg.get("scaling", {}).get("columns", [])
    target_column = ds_cfg["target_column"]
    if columns_spec == "all_numeric":
        return [c for c in all_columns if c != target_column]
    return [c for c in columns_spec if c in all_columns]


def scale_features(train_df, val_df, test_df, ds_cfg: dict, artifacts_dir: Path):
    method = ds_cfg.get("scaling", {}).get("method", "standard")
    columns = scaling_columns(ds_cfg, train_df.columns)

    if not columns:
        logger.info("No columns specified for scaling; skipping.")
        return train_df, val_df, test_df

    if method != "standard":
        raise NotImplementedError(f"Scaling method '{method}' not implemented (only 'standard' supported).")

    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[columns] = scaler.fit_transform(train_df[columns])
    val_df[columns] = scaler.transform(val_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])

    scaler_path = artifacts_dir / "scaler.joblib"
    ensure_parent_dir(scaler_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaled columns %s using StandardScaler (fit on train only). Saved to %s", columns, scaler_path)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_clean_dataset(dataset_name: str, configs_dir: str = "configs"):
    """
    Raw CSV -> cleaned, target-encoded, UNSCALED, UNSPLIT DataFrame (steps 1-4).
    Used by preprocess() below and by src/experiments/run_ablation.py, which
    splits into CV folds itself and fits the scaler inside each fold.

    Returns:
        (df, ds_cfg, base_cfg)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_REGISTRY.keys())}")

    base_cfg = load_config(Path(configs_dir) / "base.yaml")
    ds_cfg = load_dataset_config(dataset_name, configs_dir)["dataset"]

    raw_file = Path(ds_cfg["raw_file"])
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw file not found at {raw_file}. Run "
            f"`python -m src.data.ingest --dataset {dataset_name}` first."
        )

    logger.info("Loading + cleaning dataset: %s", dataset_name)
    df = pd.read_csv(raw_file)
    target_column = ds_cfg["target_column"]
    artifacts_dir = Path(ds_cfg["processed_file"]).parent  # scaler + label mapping live alongside processed CSVs

    df = drop_id_columns(df, ds_cfg)
    df = drop_missing_rows(df, target_column)
    if base_cfg.get("preprocessing", {}).get("drop_duplicates", True):
        df = drop_duplicate_rows(df)
    df = encode_target(df, ds_cfg, artifacts_dir)
    log_class_distribution(df, target_column)
    return df, ds_cfg, base_cfg


def preprocess(dataset_name: str, configs_dir: str = "configs") -> dict:
    df, ds_cfg, base_cfg = load_clean_dataset(dataset_name, configs_dir)
    target_column = ds_cfg["target_column"]
    processed_dir = Path(ds_cfg["processed_file"]).parent

    train_df, val_df, test_df = split_data(df, target_column, base_cfg)
    train_df, val_df, test_df = scale_features(train_df, val_df, test_df, ds_cfg, processed_dir)

    ensure_parent_dir(processed_dir / "placeholder")  # ensures processed_dir exists
    paths = {
        "train": processed_dir / f"{dataset_name}_train.csv",
        "val": processed_dir / f"{dataset_name}_val.csv",
        "test": processed_dir / f"{dataset_name}_test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    logger.info("Saved processed files: %s", {k: str(v) for k, v in paths.items()})

    return {"train": train_df, "val": val_df, "test": test_df, "paths": paths}


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw datasets for ARS-Stack.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASET_REGISTRY.keys()) + list(DATASET_GROUPS.keys()),
        help="Dataset or group (core / benchmarks / all) to preprocess (default: all).",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Path to the configs directory (default: configs).",
    )
    args = parser.parse_args()

    targets = expand_dataset_names([args.dataset])

    failed = []
    for name in targets:
        try:
            preprocess(name, args.configs_dir)
        except Exception as exc:
            logger.error("Failed to preprocess '%s': %s", name, exc)
            failed.append(name)

    logger.info("Preprocessing complete for: %s", [t for t in targets if t not in failed])
    if failed:
        raise SystemExit(f"Preprocessing failed for: {failed}")


if __name__ == "__main__":
    main()