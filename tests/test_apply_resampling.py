"""
tests/test_apply_resampling.py

Unit tests for src/resampling/apply_resampling.py — the module that ties
clustering + scoring + selection into the actual per-cluster resample/merge
step, for both binary (apply_adaptive_resampling) and multiclass
(apply_adaptive_resampling_multiclass) datasets.
"""

import numpy as np
import pytest

from src.resampling.apply_resampling import (
    allocate_synthetic_counts,
    apply_adaptive_resampling,
    apply_adaptive_resampling_multiclass,
    compute_n_synthetic_needed,
    resample_globally,
    resample_training_set,
)


@pytest.fixture
def clustering_cfg():
    return {
        "method": "kmeans",
        "random_state": 42,
        "min_cluster_size": 5,
        "kmeans": {"auto_k": True, "k_min": 2, "k_max": 3, "n_clusters": 2},
        "dbscan": {"eps": 0.5, "min_samples": 5},
        "density_scoring": {"overlap_k": 5, "density_k": 3, "positive_label": 1},
    }


@pytest.fixture
def resamplers_cfg():
    return {
        "candidates": ["smote", "adasyn", "smote_tomek", "smote_enn"],
        "default_random_state": 42,
        "smote": {"k_neighbors": 3, "random_state": 42},
        "adasyn": {"n_neighbors": 3, "random_state": 42},
        "smote_tomek": {"smote_k_neighbors": 3, "random_state": 42},
        "smote_enn": {"smote_k_neighbors": 3, "random_state": 42},
        "target_balance_ratio": 1.0,
        "fallback_resampler": "smote",
        "selection": {
            "mode": "relative",
            "relative": {
                "high_overlap_percentile": 66,
                "very_high_overlap_percentile": 90,
                "low_density_percentile": 33,
            },
            "absolute": {
                "high_overlap_threshold": 0.5,
                "very_high_overlap_threshold": 0.75,
                "low_density_threshold": 1.0,
            },
            "min_cluster_size_for_smote": 6,
        },
    }


# ---------------------------------------------------------------------------
# Binary path
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    X_majority = rng.normal(loc=0, scale=1, size=(150, 3))
    X_minority = rng.normal(loc=3, scale=1, size=(30, 3))
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * 150 + [1] * 30)
    return X, y


def test_apply_adaptive_resampling_balances_classes(binary_data, clustering_cfg, resamplers_cfg):
    X, y = binary_data
    X_res, y_res, report = apply_adaptive_resampling(X, y, positive_label=1, clustering_cfg=clustering_cfg, resamplers_cfg=resamplers_cfg)

    assert len(X_res) == len(y_res)
    assert X_res.shape[1] == X.shape[1]
    # minority count should have grown toward the majority count
    assert (y_res == 1).sum() > (y == 1).sum()
    assert report["final_minority_count"] == (y_res == 1).sum()
    assert report["n_clusters"] >= 1


def test_apply_adaptive_resampling_report_is_json_safe_types(binary_data, clustering_cfg, resamplers_cfg):
    X, y = binary_data
    _, _, report = apply_adaptive_resampling(X, y, positive_label=1, clustering_cfg=clustering_cfg, resamplers_cfg=resamplers_cfg)
    for cluster in report["clusters"]:
        assert isinstance(cluster["cluster_id"], int)
        assert isinstance(cluster["resampler_chosen"], str)


# ---------------------------------------------------------------------------
# Multiclass path
# ---------------------------------------------------------------------------

@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(1)
    X_majority = rng.normal(loc=0, scale=1, size=(150, 3))
    X_class_a = rng.normal(loc=4, scale=1, size=(25, 3))
    X_class_b = rng.normal(loc=-4, scale=1, size=(15, 3))
    X = np.vstack([X_majority, X_class_a, X_class_b])
    y = np.array([0] * 150 + [1] * 25 + [2] * 15)
    return X, y


def test_apply_adaptive_resampling_multiclass_preserves_original_labels(multiclass_data, clustering_cfg, resamplers_cfg):
    X, y = multiclass_data
    X_res, y_res, report = apply_adaptive_resampling_multiclass(X, y, clustering_cfg, resamplers_cfg)

    assert len(X_res) == len(y_res)
    assert set(np.unique(y_res)).issubset({0, 1, 2})
    assert report["majority_class"] == 0


def test_apply_adaptive_resampling_multiclass_grows_minority_classes(multiclass_data, clustering_cfg, resamplers_cfg):
    X, y = multiclass_data
    _, y_res, report = apply_adaptive_resampling_multiclass(X, y, clustering_cfg, resamplers_cfg)

    original_counts = {c: int((y == c).sum()) for c in np.unique(y)}
    final_counts = {c: int((y_res == c).sum()) for c in np.unique(y_res)}

    for c in (1, 2):
        assert final_counts[c] >= original_counts[c]
    assert report["final_class_counts"]["1"] == final_counts[1] if "1" in report["final_class_counts"] else True


def test_apply_adaptive_resampling_multiclass_three_plus_classes_all_resampled(clustering_cfg, resamplers_cfg):
    rng = np.random.default_rng(2)
    X = np.vstack([
        rng.normal(loc=0, scale=1, size=(100, 3)),
        rng.normal(loc=5, scale=1, size=(20, 3)),
        rng.normal(loc=-5, scale=1, size=(15, 3)),
        rng.normal(loc=10, scale=1, size=(12, 3)),
    ])
    y = np.array([0] * 100 + [1] * 20 + [2] * 15 + [3] * 12)

    X_res, y_res, report = apply_adaptive_resampling_multiclass(X, y, clustering_cfg, resamplers_cfg)

    assert set(np.unique(y_res)) == {0, 1, 2, 3}
    assert len(report["classes"]) == 3  # every non-majority class gets its own report entry


# ---------------------------------------------------------------------------
# Global synthetic budget (the adaptive-vs-plain-SMOTE confound fix)
# ---------------------------------------------------------------------------

def test_allocate_synthetic_counts_sums_exactly_and_is_proportional():
    alloc = allocate_synthetic_counts({0: 70, 1: 20, 2: 10}, 1001)
    assert sum(alloc.values()) == 1001
    assert alloc[0] > alloc[1] > alloc[2]
    assert abs(alloc[0] - 700.7) < 1


def test_allocate_synthetic_counts_zero_budget():
    assert allocate_synthetic_counts({0: 5, 1: 5}, 0) == {0: 0, 1: 0}


def test_compute_n_synthetic_needed():
    assert compute_n_synthetic_needed(30, 150, 1.0) == 120
    assert compute_n_synthetic_needed(30, 150, 0.5) == 45
    assert compute_n_synthetic_needed(200, 150, 1.0) == 0


def test_adaptive_resampling_reaches_the_global_target_ratio(binary_data, clustering_cfg, resamplers_cfg):
    """Pre-fix bug: each cluster was only balanced against ~3x its size of local
    majority context, so the result stayed heavily imbalanced."""
    X, y = binary_data
    _, y_res, report = apply_adaptive_resampling(X, y, 1, clustering_cfg, resamplers_cfg)
    n_min, n_maj = (y_res == 1).sum(), (y_res == 0).sum()
    # cleaning resamplers may trim a few points either side, so allow 10%
    assert abs(n_min / n_maj - 1.0) < 0.1
    assert sum(c["n_synthetic_allocated"] for c in report["clusters"]) == 150 - 30


def test_adaptive_and_global_smote_add_similar_amounts(binary_data, clustering_cfg, resamplers_cfg):
    X, y = binary_data
    _, y_adaptive, _ = apply_adaptive_resampling(X, y, 1, clustering_cfg, resamplers_cfg)
    _, y_global, _ = resample_globally(X, y, "smote", resamplers_cfg)
    assert abs(int((y_adaptive == 1).sum()) - int((y_global == 1).sum())) <= 0.1 * (y_global == 1).sum()


def test_target_balance_ratio_is_respected(binary_data, clustering_cfg, resamplers_cfg):
    X, y = binary_data
    cfg = {**resamplers_cfg, "target_balance_ratio": 0.5}
    _, y_adaptive, _ = apply_adaptive_resampling(X, y, 1, clustering_cfg, cfg)
    _, y_global, _ = resample_globally(X, y, "smote", cfg)
    assert (y_global == 1).sum() == 75
    assert abs(int((y_adaptive == 1).sum()) - 75) <= 8


def test_multiclass_adaptive_brings_every_class_near_majority(multiclass_data, clustering_cfg, resamplers_cfg):
    X, y = multiclass_data
    _, y_res, _ = apply_adaptive_resampling_multiclass(X, y, clustering_cfg, resamplers_cfg)
    n_major = (y_res == 0).sum()
    for c in (1, 2):
        assert abs((y_res == c).sum() / n_major - 1.0) < 0.15


@pytest.mark.parametrize("method", ["smote", "adasyn", "smote_tomek", "smote_enn"])
def test_resample_globally_binary_and_multiclass(method, binary_data, multiclass_data, resamplers_cfg):
    for X, y in (binary_data, multiclass_data):
        X_res, y_res, report = resample_globally(X, y, method, resamplers_cfg)
        assert len(X_res) == len(y_res)
        assert set(np.unique(y_res)) == set(np.unique(y))
        assert report["mode"] == f"global_{method}"


def test_global_adasyn_falls_back_to_smote_on_separable_data(resamplers_cfg):
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(0, 0.5, size=(100, 2)), rng.normal(50, 0.5, size=(20, 2))])
    y = np.array([0] * 100 + [1] * 20)
    _, y_res, report = resample_globally(X, y, "adasyn", resamplers_cfg)
    assert report["resampler_used"] == {1: "smote"}
    assert (y_res == 1).sum() == 100


def test_global_adasyn_multiclass_falls_back_per_class_only(resamplers_cfg):
    rng = np.random.default_rng(7)
    X = np.vstack([
        rng.normal(0, 1, size=(100, 2)),     # majority
        rng.normal(0.5, 1, size=(20, 2)),    # overlaps majority -> ADASYN works
        rng.normal(80, 0.5, size=(10, 2)),   # isolated -> ADASYN raises
    ])
    y = np.array([0] * 100 + [1] * 20 + [2] * 10)
    _, y_res, report = resample_globally(X, y, "adasyn", resamplers_cfg)
    assert report["resampler_used"] == {1: "adasyn", 2: "smote"}
    assert (y_res == 2).sum() == 100
    assert abs((y_res == 1).sum() - 100) <= 5


def test_resample_training_set_none_returns_input(binary_data, clustering_cfg, resamplers_cfg):
    X, y = binary_data
    X_res, y_res, report = resample_training_set(X, y, "none", "binary", clustering_cfg, resamplers_cfg)
    assert X_res is X and y_res is y and report["mode"] == "none"


def test_failed_resampler_falls_back_and_still_generates(clustering_cfg, resamplers_cfg):
    """ADASYN raises when no minority point has a majority neighbour (a
    far-away cluster); the SMOTE fallback must still deliver the allocation."""
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 0.5, size=(120, 2)), rng.normal(50, 0.5, size=(24, 2))])
    y = np.array([0] * 120 + [1] * 24)
    cfg = {**resamplers_cfg, "selection": {**resamplers_cfg["selection"], "mode": "absolute",
           "absolute": {"high_overlap_threshold": 0.4, "very_high_overlap_threshold": 0.7,
                        "low_density_ratio_threshold": 5.0}}}  # force adasyn for every cluster
    _, y_res, report = apply_adaptive_resampling(X, y, 1, clustering_cfg, cfg)
    assert all(c["resampler_chosen"] == "adasyn" for c in report["clusters"])
    assert (y_res == 1).sum() == 120
