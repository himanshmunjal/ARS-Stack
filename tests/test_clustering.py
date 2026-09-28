"""
tests/test_clustering.py

Unit tests for src/clustering/minority_clustering.py and
src/clustering/density_scoring.py.
"""

import numpy as np
import pytest

from src.clustering.density_scoring import (
    compute_density_score,
    compute_overlap_score,
    score_clusters,
)
from src.clustering.minority_clustering import cluster_minority_class


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_clustering_config():
    return {
        "method": "kmeans",
        "random_state": 42,
        "min_cluster_size": 5,
        "kmeans": {"auto_k": True, "k_min": 2, "k_max": 4, "n_clusters": 3},
        "dbscan": {"eps": 0.5, "min_samples": 5},
        "density_scoring": {"overlap_k": 5, "density_k": 3, "positive_label": 1},
    }


@pytest.fixture
def two_blob_minority():
    """Two well-separated, well-formed blobs of 'minority class' points."""
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=[0, 0], scale=0.1, size=(30, 2))
    blob_b = rng.normal(loc=[10, 10], scale=0.1, size=(30, 2))
    return np.vstack([blob_a, blob_b])


# ---------------------------------------------------------------------------
# cluster_minority_class
# ---------------------------------------------------------------------------

def test_cluster_minority_class_returns_correct_length(two_blob_minority, base_clustering_config):
    labels = cluster_minority_class(two_blob_minority, base_clustering_config)
    assert len(labels) == len(two_blob_minority)


def test_cluster_minority_class_finds_two_blobs(two_blob_minority, base_clustering_config):
    labels = cluster_minority_class(two_blob_minority, base_clustering_config)
    # Two well-separated blobs should be found as (at least) 2 distinct clusters
    assert len(np.unique(labels)) >= 2


def test_small_minority_falls_back_to_single_cluster(base_clustering_config):
    tiny_X = np.random.default_rng(1).normal(size=(4, 2))  # below min_cluster_size * 2 = 10
    labels = cluster_minority_class(tiny_X, base_clustering_config)
    assert len(labels) == 4
    assert len(np.unique(labels)) == 1


def test_dbscan_method_runs_and_relabels_noise(two_blob_minority, base_clustering_config):
    cfg = dict(base_clustering_config, method="dbscan")
    cfg["dbscan"] = {"eps": 1.0, "min_samples": 3}
    labels = cluster_minority_class(two_blob_minority, cfg)
    assert len(labels) == len(two_blob_minority)
    assert (labels == -1).sum() == 0  # no raw DBSCAN noise labels should remain


def test_min_cluster_size_merging_leaves_no_undersized_clusters(base_clustering_config):
    rng = np.random.default_rng(2)
    # One large blob + a handful of scattered outliers that would otherwise
    # form their own tiny clusters
    large_blob = rng.normal(loc=[0, 0], scale=0.2, size=(40, 2))
    outliers = rng.normal(loc=[20, 20], scale=0.01, size=(2, 2))
    X = np.vstack([large_blob, outliers])

    cfg = dict(base_clustering_config)
    cfg["min_cluster_size"] = 5
    cfg["kmeans"] = {"auto_k": False, "n_clusters": 5}

    labels = cluster_minority_class(X, cfg)
    _, counts = np.unique(labels, return_counts=True)
    assert all(c >= cfg["min_cluster_size"] for c in counts)


def test_unknown_method_raises(two_blob_minority, base_clustering_config):
    cfg = dict(base_clustering_config, method="not_a_real_method")
    with pytest.raises(ValueError):
        cluster_minority_class(two_blob_minority, cfg)


# ---------------------------------------------------------------------------
# compute_overlap_score
# ---------------------------------------------------------------------------

def test_overlap_score_is_zero_when_isolated_from_majority():
    rng = np.random.default_rng(3)
    X_cluster = rng.normal(loc=[0, 0], scale=0.1, size=(10, 2))
    # Majority class points far away, minority points also far from cluster
    X_full = rng.normal(loc=[50, 50], scale=0.1, size=(50, 2))
    y_full = np.zeros(50)  # all majority class

    # Mix in the cluster's own points as minority class so neighbors aren't purely majority
    X_full_with_cluster = np.vstack([X_full, X_cluster])
    y_full_with_cluster = np.concatenate([y_full, np.ones(len(X_cluster))])

    score = compute_overlap_score(X_cluster, X_full_with_cluster, y_full_with_cluster, k=5, positive_label=1)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_overlap_score_is_high_when_surrounded_by_majority():
    rng = np.random.default_rng(4)
    X_cluster = rng.normal(loc=[0, 0], scale=0.1, size=(5, 2))
    # Majority points densely packed right around the cluster location
    X_majority_nearby = rng.normal(loc=[0, 0], scale=0.1, size=(100, 2))
    y_majority_nearby = np.zeros(100)

    score = compute_overlap_score(X_cluster, X_majority_nearby, y_majority_nearby, k=5, positive_label=1)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_overlap_score_in_valid_range():
    rng = np.random.default_rng(5)
    X_cluster = rng.normal(size=(10, 2))
    X_full = rng.normal(size=(100, 2))
    y_full = rng.choice([0, 1], size=100)

    score = compute_overlap_score(X_cluster, X_full, y_full, k=10, positive_label=1)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_density_score
# ---------------------------------------------------------------------------

def test_density_score_higher_for_tight_cluster_than_sparse_cluster():
    rng = np.random.default_rng(6)
    tight_cluster = rng.normal(loc=[0, 0], scale=0.01, size=(20, 2))
    sparse_cluster = rng.normal(loc=[0, 0], scale=5.0, size=(20, 2))

    tight_score = compute_density_score(tight_cluster, k=5)
    sparse_score = compute_density_score(sparse_cluster, k=5)

    assert tight_score > sparse_score


def test_density_score_single_point_returns_zero():
    single_point = np.array([[1.0, 2.0]])
    assert compute_density_score(single_point) == 0.0


# ---------------------------------------------------------------------------
# score_clusters (integration of clustering + both scores)
# ---------------------------------------------------------------------------

def test_score_clusters_returns_expected_structure(two_blob_minority, base_clustering_config):
    labels = cluster_minority_class(two_blob_minority, base_clustering_config)

    rng = np.random.default_rng(7)
    X_full = np.vstack([two_blob_minority, rng.normal(loc=[5, 5], scale=1.0, size=(60, 2))])
    y_full = np.concatenate([np.ones(len(two_blob_minority)), np.zeros(60)])

    scores = score_clusters(two_blob_minority, X_full, y_full, labels, base_clustering_config)

    assert set(scores.keys()) == set(np.unique(labels).tolist())
    for cluster_id, cluster_scores in scores.items():
        assert "size" in cluster_scores
        assert "overlap_score" in cluster_scores
        assert "density_score" in cluster_scores
        assert 0.0 <= cluster_scores["overlap_score"] <= 1.0
        assert cluster_scores["size"] > 0