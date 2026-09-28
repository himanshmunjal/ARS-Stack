"""
src/ensemble/base_learners.py

Builds and trains the base learners for ARS-Stack's stacking ensemble:
Random Forest, XGBoost, and LightGBM. Hyperparameters come from
configs/models/ensemble.yaml -> ensemble.base_learners.

This module deliberately does NOT combine the learners' outputs — that's
src/ensemble/stacking.py's job (confidence weighting + meta-learner). This
module only knows how to build, train, and get predictions from each
individual learner.
"""

import logging

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger("base_learners")

LEARNER_CLASSES = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}


def build_base_learners(config: dict, n_classes: int = 2) -> dict:
    """
    Instantiate all configured base learners (unfitted).

    Args:
        config: the "ensemble" section of configs/models/ensemble.yaml
                (i.e. config["base_learners"] holds each learner's hyperparams)
        n_classes: number of target classes. Defaults to 2 (binary), matching
                   every config file in this project. XGBoost's "aucpr"
                   eval_metric (used throughout configs/models/ensemble.yaml)
                   is binary-only and raises at fit time for >2 classes, so
                   when n_classes > 2 it's swapped for "mlogloss" here —
                   the only place multiclass needs a different hyperparameter.

    Returns:
        dict: {learner_name: unfitted estimator instance}
    """
    learners_cfg = config["base_learners"]
    learners = {}

    for name, params in learners_cfg.items():
        if name not in LEARNER_CLASSES:
            raise ValueError(
                f"Unknown base learner '{name}' (expected one of {list(LEARNER_CLASSES.keys())})."
            )
        cls = LEARNER_CLASSES[name]
        # XGBoost/LightGBM don't accept a bare `None` the same way sklearn's
        # RandomForest does for max_depth in older versions, but both
        # libraries handle None/-1 fine in current versions — pass through as-is.
        clean_params = dict(params)  # copy so we never mutate the config dict
        if name == "xgboost" and n_classes > 2 and clean_params.get("eval_metric") == "aucpr":
            clean_params["eval_metric"] = "mlogloss"
        learners[name] = cls(**clean_params)
        logger.info("Built base learner '%s' with params: %s", name, clean_params)

    return learners


def train_base_learners(learners: dict, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Fit every base learner on the full training set. Returns the SAME dict
    with each estimator now fitted (mutated in place, and also returned for
    convenience).
    """
    for name, model in learners.items():
        logger.info("Training base learner '%s' on %d samples...", name, len(X_train))
        model.fit(X_train, y_train)
    return learners


def predict_base_learners(learners: dict, X: np.ndarray) -> dict:
    """
    Get positive-class probabilities from every (fitted) base learner.
    Binary-only — see predict_base_learners_multiclass for >2 classes.

    Returns:
        dict: {learner_name: np.ndarray of shape (n_samples,)}
    """
    predictions = {}
    for name, model in learners.items():
        proba = model.predict_proba(X)[:, 1]
        predictions[name] = proba
    return predictions


def predict_base_learners_multiclass(learners: dict, X: np.ndarray) -> dict:
    """
    Get full per-class probability matrices from every (fitted) base learner.

    Returns:
        dict: {learner_name: np.ndarray of shape (n_samples, n_classes)}
    """
    return {name: model.predict_proba(X) for name, model in learners.items()}