"""
src/evaluation/metrics.py

Computes the metrics used throughout ARS-Stack's evaluation: F1, AUC-PR,
G-mean, and MCC — never raw accuracy, since accuracy is misleading under
imbalance (e.g. predicting "not fraud" on every row already scores >99%
accuracy on the credit card dataset).

Supports both:
    - binary datasets (creditcard, mammography) -> standard binary metrics
    - multiclass datasets (yeast) -> macro-averaged versions, since no single
      class is "the" positive class (per configs/datasets/yeast_multiclass.yaml's
      note that this dataset needs macro-averaged metrics, not binary-only ones)
"""

import logging

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("metrics")

SUPPORTED_METRICS = ["f1", "auc_pr", "g_mean", "mcc"]


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Args:
        y_true: shape (n_samples,), values in {0, 1}
        y_pred: shape (n_samples,), predicted class labels
        y_proba: shape (n_samples,), predicted probability of the POSITIVE class (1)
    """
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_proba),
        "g_mean": float(geometric_mean_score(y_true, y_pred, average="binary")),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Multiclass metrics
# ---------------------------------------------------------------------------

def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Args:
        y_true: shape (n_samples,), integer-encoded class labels 0..K-1
        y_pred: shape (n_samples,), predicted class labels
        y_proba: shape (n_samples, K), predicted probability per class, column
                 k = class k (the model's training classes). A small test fold
                 may not contain every class; AUC-PR is then macro-averaged
                 over the classes present in y_true.
    """
    n_cols = y_proba.shape[1]
    classes = np.arange(n_cols)
    if not set(np.unique(y_true)).issubset(set(classes)):
        raise ValueError(
            f"y_true contains labels outside 0..{n_cols - 1} — y_proba columns must "
            f"cover every class (labels are integer-encoded by preprocess.py)."
        )

    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        # label_binarize collapses to 1 column when there are only 2 classes;
        # AUC-PR macro-average needs the full one-hot form to work correctly.
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    present = y_true_bin.sum(axis=0) > 0

    return {
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_pr": average_precision_score(y_true_bin[:, present], y_proba[:, present], average="macro"),
        "g_mean": float(geometric_mean_score(y_true, y_pred, average="macro")),
        "mcc": matthews_corrcoef(y_true, y_pred),  # MCC generalizes natively to multiclass
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    task_type: str = "binary",
) -> dict:
    """
    Args:
        task_type: "binary" (y_proba is 1D, positive-class probability) or
                   "multiclass" (y_proba is 2D, one column per class)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    if task_type == "binary":
        if y_proba.ndim == 2:
            # Accept a full (n_samples, 2) predict_proba output too, and just
            # take the positive-class column, so callers don't have to remember
            # to slice it themselves every time.
            y_proba = y_proba[:, 1]
        return compute_binary_metrics(y_true, y_pred, y_proba)
    elif task_type == "multiclass":
        if y_proba.ndim != 2:
            raise ValueError("Multiclass metrics require a 2D y_proba (n_samples, n_classes).")
        return compute_multiclass_metrics(y_true, y_pred, y_proba)
    else:
        raise ValueError(f"Unknown task_type '{task_type}' (expected 'binary' or 'multiclass').")


# ---------------------------------------------------------------------------
# Aggregation helpers (for repeated CV / multi-run comparisons)
# ---------------------------------------------------------------------------

def aggregate_metrics(metric_dicts: list) -> dict:
    """
    Given a list of per-fold/per-run metric dicts (all with the same keys),
    returns {metric_name: {"mean": ..., "std": ...}} — used to summarize
    repeated cross-validation results (configs/base.yaml -> cross_validation.repeats).
    """
    if not metric_dicts:
        raise ValueError("metric_dicts is empty — nothing to aggregate.")

    keys = metric_dicts[0].keys()
    aggregated = {}
    for key in keys:
        values = [d[key] for d in metric_dicts]
        aggregated[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return aggregated


def metrics_to_dataframe(results: dict):
    """
    Convenience formatter matching the style used in
    notebooks/02_baseline_reproduction.ipynb — turns {model_name: metrics_dict}
    into a tidy pandas DataFrame for display/CSV export.
    """
    import pandas as pd

    return pd.DataFrame(results).T.round(4)