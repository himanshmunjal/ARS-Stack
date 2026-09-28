"""
tests/test_ensemble.py

Unit tests for src/ensemble/base_learners.py and src/ensemble/stacking.py.
"""

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.ensemble.base_learners import build_base_learners, predict_base_learners, train_base_learners
from src.ensemble.stacking import (
    ConfidenceWeightedStackingEnsemble,
    build_meta_features,
    build_meta_learner,
    build_stacking_ensemble,
    compute_confidence_weights,
    compute_confidence_weights_multiclass,
    tune_threshold,
    _normalize_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_learners_config():
    return {
        "base_learners": {
            "random_forest": {"n_estimators": 20, "max_depth": None, "random_state": 42, "n_jobs": -1},
            "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "random_state": 42, "eval_metric": "aucpr", "n_jobs": -1},
            "lightgbm": {"n_estimators": 20, "max_depth": -1, "learning_rate": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1},
        }
    }


@pytest.fixture
def ensemble_config(base_learners_config):
    cfg = dict(base_learners_config)
    cfg.update({
        "meta_learner": {"type": "logistic_regression", "max_iter": 500, "random_state": 42},
        "combiner": "weighted_average",
        "confidence_weighting": {"enabled": True, "method": "validation_auc_pr", "min_weight": 0.05},
        "threshold_tuning": {"enabled": True, "metric": "f1"},
        "cv_folds": 3,
        "passthrough": False,
    })
    return cfg


@pytest.fixture
def imbalanced_data():
    """Small synthetic imbalanced binary dataset, split into train/val/test."""
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=4, n_redundant=1,
        weights=[0.85, 0.15], flip_y=0.01, random_state=0,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=0)
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture
def fitted_base_learners(base_learners_config, imbalanced_data):
    X_train, y_train, *_ = imbalanced_data
    learners = build_base_learners(base_learners_config)
    return train_base_learners(learners, X_train, y_train)


# ---------------------------------------------------------------------------
# build_base_learners
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("random_forest", RandomForestClassifier),
        ("xgboost", XGBClassifier),
        ("lightgbm", LGBMClassifier),
    ],
)
def test_build_base_learners_returns_correct_types(name, expected_type, base_learners_config):
    learners = build_base_learners(base_learners_config)
    assert isinstance(learners[name], expected_type)


def test_build_base_learners_returns_all_configured_learners(base_learners_config):
    learners = build_base_learners(base_learners_config)
    assert set(learners.keys()) == {"random_forest", "xgboost", "lightgbm"}


def test_build_base_learners_unknown_name_raises():
    with pytest.raises(ValueError):
        build_base_learners({"base_learners": {"not_a_real_model": {"random_state": 42}}})


def test_build_base_learners_does_not_mutate_config(base_learners_config):
    original = {k: dict(v) for k, v in base_learners_config["base_learners"].items()}
    build_base_learners(base_learners_config)
    assert base_learners_config["base_learners"] == original


def test_build_base_learners_returns_unfitted_models(base_learners_config):
    learners = build_base_learners(base_learners_config)
    for model in learners.values():
        assert not hasattr(model, "classes_")  # sklearn/xgboost/lightgbm set this only after fit()


# ---------------------------------------------------------------------------
# train_base_learners / predict_base_learners
# ---------------------------------------------------------------------------

def test_train_base_learners_fits_all_models(base_learners_config, imbalanced_data):
    X_train, y_train, *_ = imbalanced_data
    learners = build_base_learners(base_learners_config)
    trained = train_base_learners(learners, X_train, y_train)
    for model in trained.values():
        assert hasattr(model, "classes_")  # now fitted


def test_predict_base_learners_returns_correct_structure(fitted_base_learners, imbalanced_data):
    *_, X_test, y_test = imbalanced_data
    preds = predict_base_learners(fitted_base_learners, X_test)

    assert set(preds.keys()) == set(fitted_base_learners.keys())
    for name, proba in preds.items():
        assert proba.shape == (len(X_test),)
        assert np.all((proba >= 0) & (proba <= 1))


# ---------------------------------------------------------------------------
# build_meta_learner
# ---------------------------------------------------------------------------

def test_build_meta_learner_logistic_regression():
    cfg = {"meta_learner": {"type": "logistic_regression", "max_iter": 500, "random_state": 42}}
    meta = build_meta_learner(cfg)
    assert isinstance(meta, LogisticRegression)
    assert meta.max_iter == 500


def test_build_meta_learner_xgboost():
    cfg = {"meta_learner": {"type": "xgboost", "random_state": 42}}
    meta = build_meta_learner(cfg)
    assert isinstance(meta, XGBClassifier)


def test_build_meta_learner_unknown_type_raises():
    cfg = {"meta_learner": {"type": "not_a_real_meta_learner"}}
    with pytest.raises(ValueError):
        build_meta_learner(cfg)


# ---------------------------------------------------------------------------
# _normalize_weights
# ---------------------------------------------------------------------------

def test_normalize_weights_sums_to_one():
    raw = {"a": 0.8, "b": 0.6, "c": 0.2}
    normalized = _normalize_weights(raw, min_weight=0.05)
    assert pytest.approx(sum(normalized.values()), abs=1e-9) == 1.0


def test_normalize_weights_floors_low_scores():
    raw = {"a": 0.9, "b": 0.0}  # b would be zeroed out without the floor
    normalized = _normalize_weights(raw, min_weight=0.1)
    # b's raw score (0.0) should have been floored to 0.1 before normalizing,
    # so it must end up with nonzero weight in the output.
    assert normalized["b"] > 0.0


def test_normalize_weights_preserves_relative_order():
    raw = {"a": 0.9, "b": 0.5, "c": 0.3}
    normalized = _normalize_weights(raw, min_weight=0.05)
    assert normalized["a"] > normalized["b"] > normalized["c"]


def test_normalize_weights_all_zero_falls_back_to_equal():
    raw = {"a": 0.0, "b": 0.0}
    normalized = _normalize_weights(raw, min_weight=0.0)
    assert normalized["a"] == normalized["b"] == 0.5


# ---------------------------------------------------------------------------
# compute_confidence_weights
# ---------------------------------------------------------------------------

def test_compute_confidence_weights_enabled_sums_to_one(fitted_base_learners, imbalanced_data):
    _, _, X_val, y_val, _, _ = imbalanced_data
    cfg = {"confidence_weighting": {"enabled": True, "min_weight": 0.05}}
    weights = compute_confidence_weights(fitted_base_learners, X_val, y_val, cfg)
    assert set(weights.keys()) == set(fitted_base_learners.keys())
    assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0


def test_compute_confidence_weights_disabled_returns_equal_weights(fitted_base_learners, imbalanced_data):
    _, _, X_val, y_val, _, _ = imbalanced_data
    cfg = {"confidence_weighting": {"enabled": False}}
    weights = compute_confidence_weights(fitted_base_learners, X_val, y_val, cfg)
    n = len(fitted_base_learners)
    assert all(pytest.approx(w, abs=1e-9) == 1.0 / n for w in weights.values())


# ---------------------------------------------------------------------------
# build_meta_features
# ---------------------------------------------------------------------------

def test_build_meta_features_shape_without_passthrough():
    probas = {"a": np.array([0.1, 0.5, 0.9]), "b": np.array([0.2, 0.4, 0.8])}
    weights = {"a": 0.6, "b": 0.4}
    meta_X = build_meta_features(probas, weights, learner_names=["a", "b"], passthrough=False)
    assert meta_X.shape == (3, 2)


def test_build_meta_features_shape_with_passthrough():
    probas = {"a": np.array([0.1, 0.5, 0.9]), "b": np.array([0.2, 0.4, 0.8])}
    weights = {"a": 0.6, "b": 0.4}
    X_original = np.random.default_rng(0).normal(size=(3, 4))
    meta_X = build_meta_features(probas, weights, learner_names=["a", "b"], passthrough=True, X_original=X_original)
    assert meta_X.shape == (3, 2 + 4)


def test_build_meta_features_passthrough_without_x_original_raises():
    probas = {"a": np.array([0.1, 0.5])}
    weights = {"a": 1.0}
    with pytest.raises(ValueError):
        build_meta_features(probas, weights, learner_names=["a"], passthrough=True, X_original=None)


def test_build_meta_features_applies_weight_scaling_correctly():
    probas = {"a": np.array([1.0, 1.0])}
    weights = {"a": 0.7}
    meta_X = build_meta_features(probas, weights, learner_names=["a"], passthrough=False)
    assert np.allclose(meta_X[:, 0], 0.7)


def test_build_meta_features_column_order_matches_learner_names():
    probas = {"a": np.array([1.0]), "b": np.array([2.0])}
    weights = {"a": 1.0, "b": 1.0}
    meta_X = build_meta_features(probas, weights, learner_names=["b", "a"], passthrough=False)
    # learner_names order is ["b", "a"], so column 0 should be b's value, column 1 a's
    assert meta_X[0, 0] == 2.0
    assert meta_X[0, 1] == 1.0


# ---------------------------------------------------------------------------
# ConfidenceWeightedStackingEnsemble — full integration
# ---------------------------------------------------------------------------

def test_ensemble_fit_predict_end_to_end(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, X_test, y_test = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config)
    ensemble.fit(X_train, y_train, X_val, y_val)

    proba = ensemble.predict_proba(X_test)
    pred = ensemble.predict(X_test)

    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert set(np.unique(pred)).issubset({0, 1})


def test_ensemble_weights_sum_to_one_after_fit(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, *_ = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config)
    ensemble.fit(X_train, y_train, X_val, y_val)
    assert pytest.approx(sum(ensemble.weights_.values()), abs=1e-9) == 1.0


def test_ensemble_base_learners_and_meta_learner_are_fitted(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, *_ = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble({**ensemble_config, "combiner": "meta_learner"})
    ensemble.fit(X_train, y_train, X_val, y_val)

    for model in ensemble.base_learners_.values():
        assert hasattr(model, "classes_")
    assert hasattr(ensemble.meta_learner_, "classes_")


def test_ensemble_with_passthrough(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, X_test, y_test = imbalanced_data
    cfg = dict(ensemble_config)
    cfg["combiner"] = "meta_learner"
    cfg["passthrough"] = True
    ensemble = ConfidenceWeightedStackingEnsemble(cfg)
    ensemble.fit(X_train, y_train, X_val, y_val)
    proba = ensemble.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_ensemble_with_confidence_weighting_disabled(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, X_test, y_test = imbalanced_data
    cfg = dict(ensemble_config)
    cfg["confidence_weighting"] = {"enabled": False}
    ensemble = ConfidenceWeightedStackingEnsemble(cfg)
    ensemble.fit(X_train, y_train, X_val, y_val)

    n = len(ensemble.base_learners_)
    assert all(pytest.approx(w, abs=1e-9) == 1.0 / n for w in ensemble.weights_.values())
    proba = ensemble.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_ensemble_with_xgboost_meta_learner(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, X_test, y_test = imbalanced_data
    cfg = dict(ensemble_config)
    cfg["combiner"] = "meta_learner"
    cfg["meta_learner"] = {"type": "xgboost", "random_state": 42}
    ensemble = ConfidenceWeightedStackingEnsemble(cfg)
    ensemble.fit(X_train, y_train, X_val, y_val)
    proba = ensemble.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert isinstance(ensemble.meta_learner_, XGBClassifier)


def test_ensemble_learner_names_sorted_and_consistent(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, *_ = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config)
    ensemble.fit(X_train, y_train, X_val, y_val)
    assert ensemble.learner_names_ == sorted(ensemble.learner_names_)
    assert set(ensemble.learner_names_) == set(ensemble.base_learners_.keys())


def test_build_stacking_ensemble_factory_returns_ensemble_instance(ensemble_config):
    ensemble = build_stacking_ensemble(ensemble_config)
    assert isinstance(ensemble, ConfidenceWeightedStackingEnsemble)


def test_ensemble_predictions_better_than_random_on_separable_data(ensemble_config):
    """Sanity check: on data with real signal, the ensemble should clearly
    beat random guessing (AUC-PR well above the positive class base rate)."""
    from sklearn.metrics import average_precision_score

    X, y = make_classification(
        n_samples=500, n_features=6, n_informative=5, n_redundant=0,
        weights=[0.8, 0.2], flip_y=0.0, class_sep=2.0, random_state=1,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=1)

    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config)
    ensemble.fit(X_train, y_train, X_val, y_val)
    proba = ensemble.predict_proba(X_test)[:, 1]

    base_rate = y_test.mean()
    auc_pr = average_precision_score(y_test, proba)
    assert auc_pr > base_rate

# ---------------------------------------------------------------------------
# Combiners, confidence weighting, threshold tuning
# ---------------------------------------------------------------------------

def test_weighted_average_proba_is_weighted_mean_of_base_learners(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, X_test, _ = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config).fit(X_train, y_train, X_val, y_val)
    base = predict_base_learners(ensemble.base_learners_, X_test)
    expected = sum(ensemble.weights_[n] * base[n] for n in ensemble.learner_names_)
    assert np.allclose(ensemble.predict_proba(X_test)[:, 1], expected)
    assert ensemble.meta_learner_ is None


def test_confidence_weights_change_the_prediction(ensemble_config, imbalanced_data):
    """The pre-fix bug: weights were cancelled by the meta-learner. With the
    weighted-average combiner, different weights must give different output."""
    X_train, y_train, X_val, y_val, X_test, _ = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config).fit(X_train, y_train, X_val, y_val)
    before = ensemble.predict_proba(X_test)[:, 1]
    ensemble.weights_ = {"lightgbm": 0.8, "random_forest": 0.1, "xgboost": 0.1}
    after = ensemble.predict_proba(X_test)[:, 1]
    assert not np.allclose(before, after)


def test_meta_learner_combiner_uses_unweighted_features(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, *_ = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble({**ensemble_config, "combiner": "meta_learner"})
    ensemble.fit(X_train, y_train, X_val, y_val)
    assert set(ensemble.weights_.values()) == {1.0}


def test_unknown_combiner_raises(ensemble_config):
    with pytest.raises(ValueError):
        ConfidenceWeightedStackingEnsemble({**ensemble_config, "combiner": "not_a_combiner"})


def test_tune_threshold_maximizes_f1():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.9])
    # every positive is above 0.35 and every negative at/below it -> perfect F1 at 0.4
    assert tune_threshold(y, proba) == pytest.approx(0.4)


def test_tune_threshold_without_positives_falls_back_to_half():
    assert tune_threshold(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4])) == 0.5


def test_predict_uses_tuned_threshold(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, X_test, _ = imbalanced_data
    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config).fit(X_train, y_train, X_val, y_val)
    ensemble.threshold_ = 0.9
    proba = ensemble.predict_proba(X_test)[:, 1]
    assert np.array_equal(ensemble.predict(X_test), (proba >= 0.9).astype(int))


def test_threshold_is_half_when_tuning_disabled(ensemble_config, imbalanced_data):
    X_train, y_train, X_val, y_val, *_ = imbalanced_data
    cfg = {**ensemble_config, "threshold_tuning": {"enabled": False}}
    ensemble = ConfidenceWeightedStackingEnsemble(cfg).fit(X_train, y_train, X_val, y_val)
    assert ensemble.threshold_ == 0.5


def test_multiclass_weighted_average_end_to_end(ensemble_config):
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4, n_classes=3,
                               weights=[0.7, 0.2, 0.1], random_state=3)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=3)
    ensemble = ConfidenceWeightedStackingEnsemble(ensemble_config).fit(X_train, y_train, X_val, y_val)
    proba = ensemble.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(ensemble.predict(X_test))).issubset({0, 1, 2})


def test_multiclass_weights_handle_class_missing_from_validation(base_learners_config):
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, n_classes=3,
                               weights=[0.6, 0.3, 0.1], random_state=4)
    learners = train_base_learners(build_base_learners(base_learners_config, n_classes=3), X, y)
    X_val, y_val = X[y != 2][:60], y[y != 2][:60]  # class 2 absent from validation
    cfg = {"confidence_weighting": {"enabled": True, "min_weight": 0.05}}
    weights = compute_confidence_weights_multiclass(learners, X_val, y_val, cfg, classes=np.array([0, 1, 2]))
    assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0
