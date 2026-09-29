"""
src/clustering/minority_clustering.py

Clusters the minority class into local neighborhoods so that
src/resampling/resampler_selector.py (Week 5) can pick a different
resampling strategy per neighborhood, instead of applying one resampler
globally across the whole dataset — this is the core novelty of ARS-Stack.

Two clustering backends supported (choice + hyperparams come from
configs/models/clustering.yaml):
    - "kmeans" (default): fast, simple, k auto-selected via silhouette score
    - "dbscan": density-based, no need to pre-specify k, but sensitive to eps

Safety fallbacks:
    - If the minority class is too small to cluster meaningfully
      (< min_cluster_size * 2 samples), everything is treated as one cluster.
    - DBSCAN noise points (label -1) are relabeled into their own cluster
      rather than being dropped from the pipeline.
    - Any cluster smaller than min_cluster_size after fitting is merged into
      the largest cluster, so the resampler-selector never has to handle a
      near-empty cluster.
"""

import logging

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("clustering")


def _auto_select_k(X: np.ndarray, k_range, random_state: int, min_silhouette: float = None) -> int:
    """
    Pick k via best silhouette score over the given range of candidate k values.
    If min_silhouette is set and even the best split scores below it, the
    minority class has no clear cluster structure: return k=1 instead of
    forcing a split (k_range starts at 2, so otherwise one is always forced).
    """
    best_k, best_score = k_range[0], -1.0
    for k in k_range:
        if k >= len(X):
            continue
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    if min_silhouette is not None and best_score < min_silhouette:
        logger.info("Best silhouette %.3f < min_silhouette %.2f -> keeping a single cluster", best_score, min_silhouette)
        return 1
    logger.info("Auto-selected k=%d (silhouette=%.3f)", best_k, best_score)
    return best_k


def _relabel_dbscan_noise(labels: np.ndarray) -> np.ndarray:
    """DBSCAN marks outliers as -1; give them their own cluster id instead of dropping them."""
    labels = labels.copy()
    noise_mask = labels == -1
    if noise_mask.any():
        new_id = labels.max() + 1 if (labels != -1).any() else 0
        labels[noise_mask] = new_id
        logger.info("Relabeled %d DBSCAN noise point(s) into cluster %d", noise_mask.sum(), new_id)
    return labels


def _enforce_min_cluster_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    """
    Merge any cluster smaller than min_size into the single largest cluster.
    A simplification (merging by size rather than nearest centroid) chosen
    deliberately to keep the module easy to reason about and test; documented
    here as a known limitation rather than a hidden implicit choice.
    """
    unique, counts = np.unique(labels, return_counts=True)
    undersized = unique[counts < min_size]
    if len(undersized) == 0:
        return labels

    largest_cluster = unique[np.argmax(counts)]
    labels = labels.copy()
    for cid in undersized:
        if cid == largest_cluster:
            continue
        labels[labels == cid] = largest_cluster

    logger.info(
        "Merged %d undersized cluster(s) (< %d samples) into cluster %d",
        len(undersized), min_size, largest_cluster,
    )
    return labels


def cluster_minority_class(X_minority: np.ndarray, config: dict) -> np.ndarray:
    """
    Cluster the minority-class feature matrix into local neighborhoods.

    Args:
        X_minority: feature matrix (n_minority_samples, n_features),
                    minority-class rows only, numeric/scaled.
        config: the "clustering" section of configs/models/clustering.yaml.

    Returns:
        cluster_labels: np.ndarray of shape (n_minority_samples,) with
                         0-indexed integer cluster ids. Always returns at
                         least one valid cluster (id 0), even in edge cases.
    """
    X_minority = np.asarray(X_minority)
    n_samples = len(X_minority)
    min_cluster_size = config.get("min_cluster_size", 10)

    if n_samples < min_cluster_size * 2:
        logger.warning(
            "Only %d minority samples (< %d) — too few to cluster meaningfully. "
            "Falling back to a single cluster.",
            n_samples, min_cluster_size * 2,
        )
        return np.zeros(n_samples, dtype=int)

    method = config.get("method", "kmeans")
    random_state = config.get("random_state", 42)

    if method == "kmeans":
        kmeans_cfg = config.get("kmeans", {})
        if kmeans_cfg.get("auto_k", True):
            k_range = range(kmeans_cfg.get("k_min", 2), kmeans_cfg.get("k_max", 6) + 1)
            k = _auto_select_k(X_minority, k_range, random_state, kmeans_cfg.get("min_silhouette"))
        else:
            k = kmeans_cfg.get("n_clusters", 3)
        k = max(1, min(k, n_samples - 1))  # can't have more clusters than samples-1
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X_minority)

    elif method == "dbscan":
        dbscan_cfg = config.get("dbscan", {})
        labels = DBSCAN(
            eps=dbscan_cfg.get("eps", 0.5),
            min_samples=dbscan_cfg.get("min_samples", 5),
        ).fit_predict(X_minority)
        labels = _relabel_dbscan_noise(labels)

    else:
        raise ValueError(f"Unknown clustering method '{method}' (expected 'kmeans' or 'dbscan').")

    labels = _enforce_min_cluster_size(labels, min_cluster_size)
    return labels