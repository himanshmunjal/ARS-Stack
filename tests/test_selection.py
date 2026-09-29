"""
tests/test_selection.py

Unit tests for src/resampling/selection.py (validated resampling selection)
and the single-cluster option in src/clustering/minority_clustering.py.
"""

import numpy as np
import pytest

from src.clustering.minority_clustering import cluster_minority_class
from src.resampling.selection import candidate_configs, select_resampling


@pytest.fixture
def cfgs():
    clustering = {
        "method": "kmeans", "random_state": 42, "min_cluster_size": 5,
        "kmeans": {"auto_k": True, "k_min": 2, "k_max": 3},
        "density_scoring": {"overlap_k": 5, "density_k": 3, "positive_label": 1},
    }
    resamplers = {
        "default_random_state": 42, "target_balance_ratio": 1.0, "fallback_resampler": "smote",
        "smote": {"k_neighbors": 3, "random_state": 42}, "adasyn": {"n_neighbors": 3, "random_state": 42},
        "smote_tomek": {"smote_k_neighbors": 3, "random_state": 42},
        "smote_enn": {"smote_k_neighbors": 3, "random_state": 42},
        "selection": {"mode": "absolute", "min_clusters_for_relative": 3, "min_cluster_size_for_smote": 6,
                      "absolute": {"high_overlap_threshold": 0.4, "very_high_overlap_threshold": 0.7,
                                   "low_density_ratio_threshold": 0.75}},
    }
    proxy = {"n_estimators": 20, "random_state": 0, "verbose": -1, "n_jobs": 1}
    return clustering, resamplers, proxy


def test_candidate_order_puts_none_and_small_ratios_first():
    assert candidate_configs(["adaptive", "smote"], [1.0, 0.25]) == [
        ("none", None), ("adaptive", 0.25), ("smote", 0.25), ("adaptive", 1.0), ("smote", 1.0),
    ]


def test_select_returns_a_scored_candidate(cfgs):
    clustering, resamplers, proxy = cfgs
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(200, 4)), rng.normal(1.2, 1, size=(30, 4))])
    y = np.array([0] * 200 + [1] * 30)
    out = select_resampling(X, y, "binary", ["adaptive", "smote"], clustering, resamplers, proxy,
                            ratios=[0.25, 1.0], inner_folds=3, seed=0)
    assert set(out["scores"]) == {"none", "adaptive@0.25", "smote@0.25", "adaptive@1", "smote@1"}
    key = out["method"] if out["ratio"] is None else f"{out['method']}@{out['ratio']:g}"
    assert out["scores"][key] == max(out["scores"].values())


def test_select_works_for_multiclass_with_tiny_class(cfgs):
    clustering, resamplers, proxy = cfgs
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (150, 3)), rng.normal(3, 1, (25, 3)), rng.normal(-3, 1, (4, 3))])
    y = np.array([0] * 150 + [1] * 25 + [2] * 4)
    out = select_resampling(X, y, "multiclass", ["adaptive", "smote"], clustering, resamplers, proxy,
                            ratios=[0.5], inner_folds=3, seed=0)
    assert out["method"] in {"none", "adaptive", "smote"}
    assert any(np.isfinite(v) for v in out["scores"].values())


def test_min_silhouette_keeps_unstructured_minority_as_one_cluster(cfgs):
    clustering, _, _ = cfgs
    blob = np.random.default_rng(2).normal(0, 1, size=(60, 3))        # no cluster structure
    assert len(np.unique(cluster_minority_class(blob, clustering))) >= 2  # default: split is forced
    cfg = {**clustering, "kmeans": {**clustering["kmeans"], "min_silhouette": 0.5}}
    assert len(np.unique(cluster_minority_class(blob, cfg))) == 1


def test_min_silhouette_still_splits_clear_clusters(cfgs):
    clustering, _, _ = cfgs
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(10, 0.1, (30, 2))])
    cfg = {**clustering, "kmeans": {**clustering["kmeans"], "min_silhouette": 0.25}}
    assert len(np.unique(cluster_minority_class(X, cfg))) == 2
