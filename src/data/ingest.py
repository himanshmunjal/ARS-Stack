"""
src/data/ingest.py

Downloads/loads the raw dataset for a given config and saves it as a CSV
under the path specified in configs/datasets/<name>.yaml -> dataset.raw_file.

Each core dataset has its own fetch function because each comes from a
different source (Kaggle, imbalanced-learn's bundled fetcher, UCI). The
benchmark collection (configs/datasets/benchmarks.yaml) shares one fetcher.
All of them write out a plain CSV so that everything downstream
(src/data/preprocess.py) can work with one consistent format regardless of source.

Usage:
    python -m src.data.ingest --dataset creditcard
    python -m src.data.ingest --dataset mammography
    python -m src.data.ingest --dataset yeast_multiclass
    python -m src.data.ingest --dataset benchmarks     # the 20 benchmark datasets
    python -m src.data.ingest --dataset all            # core + benchmarks
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a single YAML config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_benchmark_list(configs_dir: str = "configs") -> dict:
    """The `benchmarks` section of configs/datasets/benchmarks.yaml ({} if absent)."""
    path = Path(configs_dir) / "datasets" / "benchmarks.yaml"
    return load_config(path)["benchmarks"] if path.exists() else {}


def load_dataset_config(dataset_name: str, configs_dir: str = "configs") -> dict:
    """
    Load configs/datasets/<dataset_name>.yaml, or — for a name listed in
    configs/datasets/benchmarks.yaml — build the same structure from that
    file's shared defaults.
    """
    cfg_path = Path(configs_dir) / "datasets" / f"{dataset_name}.yaml"
    if cfg_path.exists():
        return load_config(cfg_path)

    bench = load_benchmark_list(configs_dir)
    if dataset_name in bench.get("datasets", []):
        return {"dataset": {
            **bench["defaults"],
            "name": dataset_name,
            "source": bench["source"],
            "fetch_key": dataset_name,
            "raw_file": f"data/raw/benchmarks/{dataset_name}.csv",
            "processed_file": f"data/processed/benchmarks/{dataset_name}/{dataset_name}_processed.csv",
        }}
    raise FileNotFoundError(f"Config not found: {cfg_path} (and '{dataset_name}' is not in benchmarks.yaml)")


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def log_class_distribution(df: pd.DataFrame, target_column: str) -> None:
    counts = df[target_column].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())
    if len(counts) > 1:
        ratio = counts.max() / counts.min()
        logger.info("Approx. majority:minority ratio = %.1f : 1", ratio)


# ---------------------------------------------------------------------------
# Dataset-specific fetchers
# ---------------------------------------------------------------------------

def ingest_creditcard(cfg: dict) -> pd.DataFrame:
    """
    Kaggle ULB Credit Card Fraud Detection dataset.

    Requires Kaggle API credentials (~/.kaggle/kaggle.json) with the
    `kaggle` package installed. If credentials aren't set up, download
    manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    and place creditcard.csv at the path in dataset.raw_file.
    """
    ds = cfg["dataset"]
    raw_file = Path(ds["raw_file"])

    if raw_file.exists():
        logger.info("Found existing file at %s, loading directly.", raw_file)
        df = pd.read_csv(raw_file)
        log_class_distribution(df, ds["target_column"])
        return df

    logger.info("Raw file not found, attempting Kaggle API download...")
    ensure_parent_dir(raw_file)
    try:
        import kaggle  # imported lazily; requires `pip install kaggle` + API token

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path=str(raw_file.parent),
            unzip=True,
        )
        downloaded = raw_file.parent / "creditcard.csv"
        if downloaded != raw_file:
            downloaded.rename(raw_file)
        logger.info("Downloaded and saved to %s", raw_file)
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Set up ~/.kaggle/kaggle.json with your "
            "API token, or manually download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place it at {raw_file}. Original error: {exc}"
        ) from exc

    df = pd.read_csv(raw_file)
    log_class_distribution(df, ds["target_column"])
    return df


def _fetch_mammography_via_imblearn(retries: int = 3, delay_seconds: int = 5):
    """
    Attempt to download via imbalanced-learn's fetch_datasets(), which pulls
    one large archive (~40MB, all bundled KEEL/UCI imbalance datasets) with
    no resume support. Flaky connections can trigger IncompleteRead mid-
    transfer, so retry a few times with a longer socket timeout before
    giving up and falling back to OpenML.
    """
    import socket
    import time
    from imblearn.datasets import fetch_datasets

    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(180)  # generous timeout for a large single-file download
    try:
        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    "Fetching 'mammography' via imbalanced-learn (attempt %d/%d)...",
                    attempt, retries,
                )
                return fetch_datasets()["mammography"]
            except Exception as exc:
                logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
                if attempt < retries:
                    time.sleep(delay_seconds)
        return None
    finally:
        socket.setdefaulttimeout(original_timeout)


def _fetch_mammography_via_openml():
    """
    Fallback source: OpenML dataset id 310 ('mammography'), fetched via
    scikit-learn's fetch_openml with its own caching/retry handling.
    Target values are mapped to the same {-1, 1} convention imblearn uses.
    """
    from sklearn.datasets import fetch_openml

    logger.info("Falling back to OpenML for the mammography dataset...")
    bunch = fetch_openml(name="mammography", version=1, as_frame=True, parser="auto")

    data = bunch.data.to_numpy()
    raw_target = bunch.target.to_numpy()
    # OpenML encodes classes as strings/ints that vary by version; normalize
    # so the minority (positive/calcification) class maps to 1, majority to -1,
    # matching the convention imblearn's fetch_datasets() uses.
    minority_val = pd.Series(raw_target).value_counts().idxmin()
    target = pd.Series(raw_target).apply(lambda v: 1 if v == minority_val else -1).to_numpy()

    class _Bunch:
        pass

    result = _Bunch()
    result.data = data
    result.target = target
    return result


def ingest_mammography(cfg: dict) -> pd.DataFrame:
    """
    Mammography imbalanced dataset. Tries imbalanced-learn's fetch_datasets()
    first (with retries), falls back to OpenML if that keeps failing (e.g.
    due to network instability causing IncompleteRead on the large archive).
    """
    ds = cfg["dataset"]
    raw_file = Path(ds["raw_file"])

    if raw_file.exists():
        logger.info("Found existing file at %s, loading directly.", raw_file)
        df = pd.read_csv(raw_file)
        log_class_distribution(df, ds["target_column"])
        return df

    bunch = _fetch_mammography_via_imblearn()
    if bunch is None:
        logger.warning("imbalanced-learn download did not succeed after retries.")
        try:
            bunch = _fetch_mammography_via_openml()
        except Exception as exc:
            raise RuntimeError(
                "Failed to fetch the mammography dataset from both "
                "imbalanced-learn and OpenML. This is likely a network/proxy "
                "issue rather than a code issue. Try: (1) re-running on a more "
                "stable connection, (2) running "
                "`python -c \"from imblearn.datasets import fetch_datasets; "
                "fetch_datasets()\"` alone a few times until it completes, or "
                "(3) manually downloading dataset id 310 from "
                "https://www.openml.org/d/310 and placing it as a CSV at "
                f"{raw_file} with columns feature_0..feature_5 + target. "
                f"Original error: {exc}"
            ) from exc

    n_features = bunch.data.shape[1]
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(bunch.data, columns=columns)
    df[ds["target_column"]] = bunch.target  # typically -1 / 1

    ensure_parent_dir(raw_file)
    df.to_csv(raw_file, index=False)
    logger.info("Saved to %s (%d rows, %d columns)", raw_file, *df.shape)
    log_class_distribution(df, ds["target_column"])
    return df


def ingest_yeast(cfg: dict) -> pd.DataFrame:
    """
    UCI Yeast dataset. Whitespace-delimited, no header in the raw file.
    Fetched directly from the UCI repository over HTTP.
    """
    ds = cfg["dataset"]
    raw_file = Path(ds["raw_file"])

    if raw_file.exists():
        logger.info("Found existing file at %s, loading directly.", raw_file)
        df = pd.read_csv(raw_file)
        log_class_distribution(df, ds["target_column"])
        return df

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    columns = [
        "sequence_name", "mcg", "gvh", "alm", "mit",
        "erl", "pox", "vac", "nuc", "class",
    ]

    logger.info("Fetching Yeast dataset from UCI: %s", url)
    try:
        df = pd.read_csv(url, sep=r"\s+", header=None, names=columns)
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch Yeast dataset from UCI. Check your internet "
            "connection, or manually download yeast.data from "
            "https://archive.ics.uci.edu/dataset/110/yeast and place it at "
            f"{raw_file}. Original error: {exc}"
        ) from exc

    ensure_parent_dir(raw_file)
    df.to_csv(raw_file, index=False)
    logger.info("Saved to %s (%d rows, %d columns)", raw_file, *df.shape)
    log_class_distribution(df, ds["target_column"])
    return df


def ingest_imblearn_benchmark(cfg: dict) -> pd.DataFrame:
    """
    One dataset from imbalanced-learn's benchmark collection
    (configs/datasets/benchmarks.yaml). Downloaded once into
    ~/scikit_learn_data by imblearn, then saved as a CSV with columns
    feature_0..feature_{d-1} + target (-1 / 1).
    """
    ds = cfg["dataset"]
    raw_file = Path(ds["raw_file"])

    if raw_file.exists():
        df = pd.read_csv(raw_file)
        log_class_distribution(df, ds["target_column"])
        return df

    from imblearn.datasets import fetch_datasets

    logger.info("Fetching '%s' via imbalanced-learn...", ds["fetch_key"])
    bunch = fetch_datasets(filter_data=(ds["fetch_key"],))[ds["fetch_key"]]
    df = pd.DataFrame(bunch.data, columns=[f"feature_{i}" for i in range(bunch.data.shape[1])])
    df[ds["target_column"]] = bunch.target

    ensure_parent_dir(raw_file)
    df.to_csv(raw_file, index=False)
    logger.info("Saved to %s (%d rows, %d columns)", raw_file, *df.shape)
    log_class_distribution(df, ds["target_column"])
    return df


# ---------------------------------------------------------------------------
# Registry + CLI entry point
# ---------------------------------------------------------------------------

CORE_DATASETS = ["creditcard", "mammography", "yeast_multiclass"]
BENCHMARK_DATASETS = list(load_benchmark_list().get("datasets", []))

DATASET_REGISTRY = {
    "creditcard": ingest_creditcard,
    "mammography": ingest_mammography,
    "yeast_multiclass": ingest_yeast,
    **{name: ingest_imblearn_benchmark for name in BENCHMARK_DATASETS},
}

DATASET_GROUPS = {
    "core": CORE_DATASETS,
    "benchmarks": BENCHMARK_DATASETS,
    "all": CORE_DATASETS + BENCHMARK_DATASETS,
}


def expand_dataset_names(names: list) -> list:
    """Expand group names ("core", "benchmarks", "all") into dataset names,
    keeping order and dropping repeats."""
    out = []
    for name in names:
        for ds in DATASET_GROUPS.get(name, [name]):
            if ds not in out:
                out.append(ds)
    return out


def ingest(dataset_name: str, configs_dir: str = "configs") -> pd.DataFrame:
    """Load config for `dataset_name` and run its fetch function."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_REGISTRY.keys())}"
        )
    cfg = load_dataset_config(dataset_name, configs_dir)
    logger.info("Ingesting dataset: %s", dataset_name)
    return DATASET_REGISTRY[dataset_name](cfg)


def main():
    parser = argparse.ArgumentParser(description="Ingest raw datasets for ARS-Stack.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASET_REGISTRY.keys()) + list(DATASET_GROUPS.keys()),
        help="Dataset or group (core / benchmarks / all) to ingest (default: all).",
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
            ingest(name, args.configs_dir)
        except Exception as exc:
            logger.error("Failed to ingest '%s': %s", name, exc)
            failed.append(name)

    logger.info("Ingestion complete for: %s", [t for t in targets if t not in failed])
    if failed:
        # Non-zero exit so `dvc repro` doesn't record a half-finished stage as done.
        raise SystemExit(f"Ingestion failed for: {failed}")


if __name__ == "__main__":
    main()