"""
src/ensemble/stacking.py

The confidence-weighted ensemble — the second half of ARS-Stack's own
contribution (alongside the adaptive resampler-selector). Each base learner
(RF / XGBoost / LightGBM) gets a confidence weight = its own validation-set
AUC-PR (normalized to sum to 1, floored at min_weight), and the ensemble's
probability is the confidence-WEIGHTED AVERAGE of the base learners'
probabilities, instead of every learner being trusted equally.

Two combiners (configs/models/ensemble.yaml -> ensemble.combiner):
    - "weighted_average" (ARS-Stack default): p = sum_i w_i * p_i. With
      confidence_weighting.enabled=false this is plain equal-weight soft
      voting — the "simple averaging" ablation from the project plan.
    - "meta_learner": classic stacking. A meta-learner (logistic regression
      or XGBoost) is trained on the base learners' OUT-OF-FOLD probabilities.
      Confidence weights are deliberately NOT applied here: multiplying a
      meta-feature by a positive constant is cancelled by the meta-learner
      (LR rescales its coefficient; tree splits are scale-invariant), so it
      would be a no-op — which is exactly what the pre-fix version did.

Binary tasks also tune the decision threshold on the validation set
(threshold_tuning), since the best cut-off depends on how much the training
set was resampled; multiclass predicts the arg-max class.

Usage:
    from src.ensemble.stacking import ConfidenceWeightedStackingEnsemble

    ensemble = ConfidenceWeightedStackingEnsemble(config["ensemble"])
    ensemble.fit(X_train, y_train, X_val, y_val)
    proba = ensemble.predict_proba(X_test)
    labels = ensemble.predict(X_test)      # uses the tuned threshold
"""

import logging

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from src.ensemble.base_learners import (
    build_base_learners,
    predict_base_learners,
    predict_base_learners_multiclass,
    train_base_learners,
)

logger = logging.getLogger("stacking")


# ---------------------------------------------------------------------------
# Meta-learner factory
# ---------------------------------------------------------------------------

def build_meta_learner(config: dict, n_classes: int = 2):
    """
    Args:
        config: the "ensemble" section of configs/models/ensemble.yaml
                (reads config["meta_learner"])
        n_classes: see build_base_learners() — "aucpr" is binary-only, swapped
                   for "mlogloss" when n_classes > 2.
    """
    meta_cfg = config["meta_learner"]
    meta_type = meta_cfg.get("type", "logistic_regression")

    if meta_type == "logistic_regression":
        return LogisticRegression(
            max_iter=meta_cfg.get("max_iter", 1000),
            random_state=meta_cfg.get("random_state", 42),
        )
    elif meta_type == "xgboost":
        return XGBClassifier(
            random_state=meta_cfg.get("random_state", 42),
            eval_metric="aucpr" if n_classes == 2 else "mlogloss",
        )
    else:
        raise ValueError(f"Unknown meta_learner type '{meta_type}' (expected 'logistic_regression' or 'xgboost').")


# ---------------------------------------------------------------------------
# Confidence weighting
# ---------------------------------------------------------------------------

def compute_confidence_weights(learners: dict, X_val: np.ndarray, y_val: np.ndarray, config: dict) -> dict:
    """
    Weight each base learner by its own validation-set AUC-PR, normalized to
    sum to 1 and floored at min_weight so no learner is ever fully zeroed out.

    Args:
        learners: dict of FITTED base learners {name: model}
        config: the "ensemble" section of ensemble.yaml (reads config["confidence_weighting"])
    """
    cw_cfg = config.get("confidence_weighting", {})
    if not cw_cfg.get("enabled", True):
        n = len(learners)
        logger.info("Confidence weighting disabled — using equal weights (1/%d each).", n)
        return {name: 1.0 / n for name in learners}

    min_weight = cw_cfg.get("min_weight", 0.05)
    raw_scores = {}
    for name, model in learners.items():
        proba = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, proba)
        raw_scores[name] = score
        logger.info("Base learner '%s' validation AUC-PR: %.4f", name, score)

    return _normalize_weights(raw_scores, min_weight)


def compute_confidence_weights_multiclass(
    learners: dict, X_val: np.ndarray, y_val: np.ndarray, config: dict, classes: np.ndarray = None,
) -> dict:
    """
    Multiclass counterpart to compute_confidence_weights(): weights each base
    learner by its own validation-set macro-averaged (one-vs-rest) AUC-PR,
    matching src.evaluation.metrics.compute_multiclass_metrics's convention.

    Args:
        classes: the classes the learners were trained on (= predict_proba
            column order). A tiny class can be absent from a small validation
            set; AUC-PR is then averaged over the classes that ARE present.
    """
    cw_cfg = config.get("confidence_weighting", {})
    if not cw_cfg.get("enabled", True):
        n = len(learners)
        logger.info("Confidence weighting disabled — using equal weights (1/%d each).", n)
        return {name: 1.0 / n for name in learners}

    min_weight = cw_cfg.get("min_weight", 0.05)
    if classes is None:
        classes = np.unique(y_val)
    classes = np.asarray(classes)
    y_val_bin = label_binarize(y_val, classes=classes)
    if y_val_bin.shape[1] == 1:
        y_val_bin = np.hstack([1 - y_val_bin, y_val_bin])
    present = y_val_bin.sum(axis=0) > 0

    raw_scores = {}
    for name, model in learners.items():
        proba = model.predict_proba(X_val)
        score = average_precision_score(y_val_bin[:, present], proba[:, present], average="macro")
        raw_scores[name] = score
        logger.info("Base learner '%s' validation macro AUC-PR: %.4f", name, score)

    return _normalize_weights(raw_scores, min_weight)


def _normalize_weights(raw_scores: dict, min_weight: float) -> dict:
    """Floor every score at min_weight, then normalize so weights sum to 1."""
    floored = {name: max(score, min_weight) for name, score in raw_scores.items()}
    total = sum(floored.values())
    if total == 0:
        n = len(floored)
        return {name: 1.0 / n for name in floored}
    normalized = {name: score / total for name, score in floored.items()}
    logger.info("Normalized confidence weights: %s", {k: round(v, 4) for k, v in normalized.items()})
    return normalized


# ---------------------------------------------------------------------------
# Meta-feature construction
# ---------------------------------------------------------------------------

def build_meta_features(probas: dict, weights: dict, learner_names: list, passthrough: bool, X_original: np.ndarray = None) -> np.ndarray:
    """
    Args:
        probas: {learner_name: np.ndarray of shape (n_samples,)}
        weights: {learner_name: float confidence weight}
        learner_names: fixed, sorted order of learner names (so train/inference
                       time always build columns in the same order)
        passthrough: whether to concatenate original features alongside
        X_original: required if passthrough=True
    """
    columns = [probas[name] * weights[name] for name in learner_names]
    meta_X = np.column_stack(columns)

    if passthrough:
        if X_original is None:
            raise ValueError("passthrough=True requires X_original to be provided.")
        meta_X = np.hstack([meta_X, X_original])

    return meta_X


def build_meta_features_multiclass(probas: dict, weights: dict, learner_names: list, passthrough: bool, X_original: np.ndarray = None) -> np.ndarray:
    """
    Multiclass counterpart to build_meta_features(): each learner contributes
    its FULL weighted per-class probability matrix (n_samples, n_classes),
    horizontally stacked across learners — instead of a single weighted
    positive-class column per learner in the binary case.

    Args:
        probas: {learner_name: np.ndarray of shape (n_samples, n_classes)}
        weights: {learner_name: float confidence weight}
        learner_names: fixed, sorted order of learner names
        passthrough: whether to concatenate original features alongside
        X_original: required if passthrough=True
    """
    columns = [probas[name] * weights[name] for name in learner_names]
    meta_X = np.hstack(columns)

    if passthrough:
        if X_original is None:
            raise ValueError("passthrough=True requires X_original to be provided.")
        meta_X = np.hstack([meta_X, X_original])

    return meta_X


# ---------------------------------------------------------------------------
# Threshold tuning (binary)
# ---------------------------------------------------------------------------

def tune_threshold(y_true: np.ndarray, proba: np.ndarray, metric: str = "f1") -> float:
    """
    Decision threshold on the positive-class probability that maximizes
    `metric` on (validation) data. Falls back to 0.5 if y_true has no
    positives. Only "f1" is supported (the minority-class F1 the project
    reports; MCC/G-mean follow the same precision/recall trade-off).
    """
    if metric != "f1":
        raise ValueError(f"Unsupported threshold_tuning metric '{metric}' (only 'f1').")
    if y_true.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # last P/R point has no threshold
    return float(thresholds[best])


# ---------------------------------------------------------------------------
# The ensemble itself
# ---------------------------------------------------------------------------

class ConfidenceWeightedStackingEnsemble:
    """
    RF + XGBoost + LightGBM ensemble whose outputs are combined either by a
    confidence-weighted average ("weighted_average", ARS-Stack) or by a
    stacking meta-learner ("meta_learner"). See the module docstring.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: the "ensemble" section of configs/models/ensemble.yaml
        """
        self.config = config
        self.combiner = config.get("combiner", "weighted_average")
        if self.combiner not in ("weighted_average", "meta_learner"):
            raise ValueError(
                f"Unknown combiner '{self.combiner}' (expected 'weighted_average' or 'meta_learner')."
            )
        self.cv_folds = config.get("cv_folds", 5)
        self.passthrough = config.get("passthrough", False)
        self.random_state = config.get("meta_learner", {}).get("random_state", 42)

        self.base_learners_ = {}
        self.weights_ = {}
        self.meta_learner_ = None
        self.learner_names_ = None
        self.classes_ = None
        self.n_classes_ = 2
        self.threshold_ = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        # 0. Detect binary vs multiclass once, up front — everything below
        #    branches on this (multiclass uses full per-class probability
        #    matrices instead of a single positive-class column per learner).
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        is_multiclass = self.n_classes_ > 2

        # 1. Build fresh, unfitted base learners.
        learner_defs = build_base_learners(self.config, n_classes=self.n_classes_)
        self.learner_names_ = sorted(learner_defs.keys())

        # 2. meta_learner combiner only: out-of-fold predictions on X_train, so
        #    the meta-learner never sees a prediction from a model that was
        #    trained on that same row.
        oof_probas = {}
        if self.combiner == "meta_learner":
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            for name, model in learner_defs.items():
                logger.info("Generating out-of-fold predictions for '%s' (%d-fold CV)...", name, self.cv_folds)
                oof = cross_val_predict(
                    clone(model), X_train, y_train, cv=skf, method="predict_proba", n_jobs=None,
                )
                oof_probas[name] = oof if is_multiclass else oof[:, 1]

        # 3. Fit each base learner on the FULL training set (used at inference time).
        self.base_learners_ = train_base_learners(learner_defs, X_train, y_train)

        # 4. Combiner.
        if self.combiner == "weighted_average":
            # Confidence weights from validation-set AUC-PR (macro one-vs-rest
            # for multiclass); equal weights if confidence_weighting is disabled.
            if is_multiclass:
                self.weights_ = compute_confidence_weights_multiclass(
                    self.base_learners_, X_val, y_val, self.config, classes=self.classes_,
                )
            else:
                self.weights_ = compute_confidence_weights(self.base_learners_, X_val, y_val, self.config)
        else:
            # Weights would be cancelled by the meta-learner (see module
            # docstring), so it's trained on the raw, unweighted OOF probabilities.
            self.weights_ = {name: 1.0 for name in self.learner_names_}
            build = build_meta_features_multiclass if is_multiclass else build_meta_features
            meta_X_train = build(oof_probas, self.weights_, self.learner_names_, self.passthrough, X_train)
            self.meta_learner_ = build_meta_learner(self.config, n_classes=self.n_classes_)
            logger.info("Training meta-learner on %d meta-features...", meta_X_train.shape[1])
            self.meta_learner_.fit(meta_X_train, y_train)

        # 5. Binary: tune the decision threshold on the validation set.
        tt_cfg = self.config.get("threshold_tuning", {})
        if not is_multiclass and tt_cfg.get("enabled", False):
            val_proba = self.predict_proba(X_val)[:, 1]
            self.threshold_ = tune_threshold(np.asarray(y_val) == self.classes_[1], val_proba, tt_cfg.get("metric", "f1"))
            logger.info("Tuned decision threshold on validation set: %.4f", self.threshold_)
        else:
            self.threshold_ = 0.5

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n_samples, n_classes) like sklearn's predict_proba."""
        is_multiclass = self.n_classes_ > 2
        if self.combiner == "weighted_average":
            if is_multiclass:
                probas = predict_base_learners_multiclass(self.base_learners_, X)
                return sum(self.weights_[name] * probas[name] for name in self.learner_names_)
            probas = predict_base_learners(self.base_learners_, X)
            p1 = sum(self.weights_[name] * probas[name] for name in self.learner_names_)
            return np.column_stack([1 - p1, p1])

        if is_multiclass:
            probas = predict_base_learners_multiclass(self.base_learners_, X)
            meta_X = build_meta_features_multiclass(probas, self.weights_, self.learner_names_, self.passthrough, X)
        else:
            probas = predict_base_learners(self.base_learners_, X)
            meta_X = build_meta_features(probas, self.weights_, self.learner_names_, self.passthrough, X)
        return self.meta_learner_.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.n_classes_ > 2:
            return self.classes_[np.argmax(proba, axis=1)]
        return np.where(proba[:, 1] >= self.threshold_, self.classes_[1], self.classes_[0])


def build_stacking_ensemble(config: dict) -> ConfidenceWeightedStackingEnsemble:
    """Convenience factory, mirrors the style of the other build_* functions."""
    return ConfidenceWeightedStackingEnsemble(config)