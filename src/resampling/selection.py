"""
src/resampling/selection.py

Validated resampling selection: instead of choosing a resampler with fixed
rules, try every candidate (resampling method x oversampling amount, plus
"no resampling") in an inner stratified cross-validation of the TRAINING data
only, score each by out-of-fold AUC-PR, and return the best.

Scoring uses a single proxy learner (LightGBM with the ensemble's own
hyperparameters) rather than the full RF + XGBoost + LightGBM ensemble:
the candidates are compared relative to each other, and the proxy keeps the
search affordable (candidates x inner folds fits per outer fold). The chosen
configuration is then used to train the full ensemble as usual.

Ties go to the earlier candidate in the search order — "none" first, then
smaller oversampling amounts — i.e. to the simpler choice.
"""

import logging

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("selection")


def _auc_pr(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Binary: AP of the positive class. Multiclass: macro AP over classes present in y."""
    if len(classes) == 2:
        return float(average_precision_score(y == classes[1], proba[:, 1]))
    y_bin = label_binarize(y, classes=classes)
    present = y_bin.sum(axis=0) > 0
    return float(average_precision_score(y_bin[:, present], proba[:, present], average="macro"))


def candidate_configs(methods: list, ratios: list) -> list:
    """[("none", None), (method, ratio) for each method and ratio], in tie-break order."""
    return [("none", None)] + [(m, float(r)) for r in sorted(ratios) for m in methods]


def select_resampling(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    methods: list,
    clustering_cfg: dict,
    resamplers_cfg: dict,
    proxy_params: dict,
    ratios: list = (0.1, 0.25, 0.5, 1.0),
    inner_folds: int = 3,
    seed: int = 42,
) -> dict:
    """
    Args:
        methods: resampling methods to search besides "none", e.g.
                 ["adaptive", "smote"] (see resample_training_set).
        ratios: oversampling amounts to try (target_balance_ratio values).
        proxy_params: LightGBM hyperparameters for the scoring model.
    Returns:
        {"method": str, "ratio": float | None, "scores": {"method@ratio": auc_pr}}
    """
    from src.resampling.apply_resampling import resample_training_set  # avoid import cycle

    classes = np.unique(y)
    min_count = int(np.bincount(np.searchsorted(classes, y)).min())
    n_splits = max(2, min(inner_folds, min_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    scores, best = {}, None
    for method, ratio in candidate_configs(methods, ratios):
        key = method if ratio is None else f"{method}@{ratio:g}"
        cfg = resamplers_cfg if ratio is None else {**resamplers_cfg, "target_balance_ratio": ratio}
        oof = np.zeros((len(y), len(classes)))
        try:
            for tr, va in splits:
                X_res, y_res, _ = resample_training_set(
                    X[tr], y[tr], method, task_type, clustering_cfg, cfg, positive_label=classes[-1],
                )
                model = LGBMClassifier(**proxy_params).fit(X_res, y_res)
                cols = np.searchsorted(classes, model.classes_)
                oof[np.ix_(va, cols)] = model.predict_proba(X[va])
            score = _auc_pr(y, oof, classes)
        except Exception as exc:  # e.g. a class too small for SMOTE in an inner fold
            logger.warning("Candidate %s failed (%s) — skipped.", key, exc)
            score = float("-inf")
        scores[key] = score
        if best is None or score > best[2]:
            best = (method, ratio, score)

    logger.info("Validated selection: %s@%s (AUC-PR %.4f)", best[0], best[1], best[2])
    return {"method": best[0], "ratio": best[1], "scores": scores}
