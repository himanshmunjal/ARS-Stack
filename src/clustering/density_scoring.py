"""
src/clustering/density_scoring.py

For each minority-class cluster (produced by minority_clustering.py),
computes two scores that src/resampling/resampler_selector.py (Week 5)
uses to decide which resampling technique fits that cluster best:

    - overlap_score:  fraction of a cluster's k-nearest neighbors (drawn
                       from the FULL training set, both classes) that
                       belong to the majority class. High overlap means the
                       minority cluster sits in/near majority territory ->
                       favor a resampler that also cleans noisy points
                       (SMOTE-Tomek, SMOTE-ENN).
    - density_score:   inverse of the average distance from each cluster
                       point to its k nearest neighbors WITHIN the same
                       cluster. Low density (sparse, spread-out points) ->
                       favor a resampler that emphasizes harder points
                       (ADASYN); high density (tight cluster) -> plain
                       SMOTE is enough.
    - density_ratio:   density_score divided by the density of the whole
                       class (all its clusters pooled, same k). Scale-free:
                       1.0 = as dense as the class overall, < 1 = sparser.
                       This is what the "absolute" selection mode compares
                       against a fixed cutoff, since raw density_score
                       depends on each dataset's feature scale.

Both are deliberately simplified, tractable versions of ideas from the
DCSHS (clustering + overlap) and SDUS (density-aware sampling) papers in
the project's literature review — not a reimplementation of either paper's
full method.
"""

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("density_scoring")


def compute_overlap_score(
    X_cluster: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    k: int = 10,
    positive_label: int = 1,
) -> float:
    """
    Mean fraction of each cluster point's k nearest neighbors (searched
    across the full dataset, both classes) that belong to the majority
    class. Returns a value in [0, 1] — higher means more class overlap.
    """
    X_cluster = np.asarray(X_cluster)
    X_full = np.asarray(X_full)
    y_full = np.asarray(y_full)

    k_eff = min(k, len(X_full))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(X_full)
    _, indices = nn.kneighbors(X_cluster)

    neighbor_labels = y_full[indices]  # shape (n_cluster_points, k_eff)
    majority_fraction_per_point = (neighbor_labels != positive_label).mean(axis=1)
    return float(majority_fraction_per_point.mean())


def compute_density_score(X_cluster: np.ndarray, k: int = 5) -> float:
    """
    Inverse of the average distance from each cluster point to its k
    nearest neighbors within the same cluster. Higher score = denser/
    tighter cluster. Unbounded above — compare relatively across clusters
    from the same run, not against a fixed absolute threshold.
    """
    X_cluster = np.asarray(X_cluster)
    if len(X_cluster) <= 1:
        return 0.0  # a single point has no internal structure; treat as sparse

    k_eff = min(k, len(X_cluster) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1)  # +1: a point is its own nearest neighbor
    nn.fit(X_cluster)
    distances, _ = nn.kneighbors(X_cluster)

    avg_distance = distances[:, 1:].mean()  # drop the self-distance column (always 0)
    if avg_distance == 0:
        return float("inf")
    return float(1.0 / avg_distance)


def score_clusters(
    X_minority: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    cluster_labels: np.ndarray,
    config: dict,
    positive_label=None,
) -> dict:
    """
    Compute overlap_score and density_score for every cluster.

    Args:
        X_minority: feature matrix of minority-class samples only, in the
                    same row order as cluster_labels.
        X_full, y_full: the FULL training set (both classes), used as the
                    neighbor pool for the overlap score.
        cluster_labels: output of cluster_minority_class(), same length
                    as X_minority.
        config: the "clustering" section of configs/models/clustering.yaml
                    (reads config["density_scoring"]).
        positive_label: which label in y_full counts as "this cluster's class"
                    for the overlap score. Defaults to
                    config["density_scoring"]["positive_label"] (binary use
                    case); pass explicitly to score one class at a time in
                    the multiclass adaptive-resampling path (each non-majority
                    class scored against "everything else" in turn).

    Returns:
        dict: {cluster_id: {"size": int, "overlap_score": float,
                            "density_score": float, "density_ratio": float}}
    """
    X_minority = np.asarray(X_minority)
    cluster_labels = np.asarray(cluster_labels)

    scoring_cfg = config.get("density_scoring", {})
    overlap_k = scoring_cfg.get("overlap_k", 10)
    density_k = scoring_cfg.get("density_k", 5)
    if positive_label is None:
        positive_label = scoring_cfg.get("positive_label", 1)

    class_density = compute_density_score(X_minority, k=density_k)

    scores = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        X_cluster = X_minority[mask]

        overlap = compute_overlap_score(X_cluster, X_full, y_full, k=overlap_k, positive_label=positive_label)
        density = compute_density_score(X_cluster, k=density_k)
        density_ratio = _safe_ratio(density, class_density)

        scores[int(cluster_id)] = {
            "size": int(mask.sum()),
            "overlap_score": overlap,
            "density_score": density,
            "density_ratio": density_ratio,
        }
        logger.info(
            "Cluster %d: size=%d, overlap_score=%.3f, density_score=%.3f, density_ratio=%.3f",
            cluster_id, mask.sum(), overlap, density, density_ratio,
        )

    return scores

def _safe_ratio(density: float, class_density: float) -> float:
    """density / class_density, treating inf (all-duplicate points) and a
    zero class density sensibly instead of producing nan."""
    if np.isinf(density) and np.isinf(class_density):
        return 1.0
    if np.isinf(density):
        return float("inf")
    if class_density == 0 or np.isinf(class_density):
        return 0.0 if np.isinf(class_density) else float("inf")
    return float(density / class_density)
