"""
src/resampling/resampler_selector.py

THE CORE NOVEL MODULE of ARS-Stack.

Given per-cluster overlap/density scores (from
src/clustering/density_scoring.py), decides which resampling technique to
apply to EACH minority-class cluster — instead of applying one resampler
globally across the whole dataset, which is what every paper in the
project's literature review does.

Decision logic (rule-based, deliberately simple and explainable):
    - overlap_score very high                      -> SMOTE-ENN
      (cluster sits deep in majority territory; needs strong noise cleaning)
    - overlap_score high                           -> SMOTE-Tomek
      (moderate class overlap; needs some boundary cleaning)
    - overlap_score low AND density_score low        -> ADASYN
      (sparse/isolated minority points; emphasize the harder ones)
    - otherwise                                      -> SMOTE
      (well-behaved, reasonably dense cluster; the safe default)

Thresholds come from one of two modes (configs/models/resamplers.yaml):
    - "absolute" (default): fixed cutoffs on overlap_score (already a [0, 1]
      fraction) and density_ratio (cluster density / whole-class density,
      scale-free). A cluster's choice depends only on its own scores.
    - "relative": PERCENTILES across the clusters found in the current run.
      With only 1-2 clusters percentiles decide by rank alone (1 cluster is
      always "very high overlap" -> SMOTE-ENN; 2 clusters always split into
      SMOTE-ENN + ADASYN), so below min_clusters_for_relative clusters this
      mode falls back to "absolute".

Safety fallback: if a cluster is too small for SMOTE/ADASYN's k-neighbors
requirement, RandomOverSampler is used instead regardless of the rule-based
choice, since SMOTE-family methods require n_samples > n_neighbors.
"""

import logging

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

logger = logging.getLogger("resampler_selector")

VALID_RESAMPLERS = {"smote", "adasyn", "smote_tomek", "smote_enn", "random_oversample"}


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def _effective_mode(cluster_scores: dict, config: dict) -> str:
    selection_cfg = config["selection"]
    mode = selection_cfg.get("mode", "absolute")
    if mode not in ("relative", "absolute"):
        raise ValueError(f"Unknown selection mode '{mode}' (expected 'relative' or 'absolute').")

    min_clusters = selection_cfg.get("min_clusters_for_relative", 3)
    if mode == "relative" and len(cluster_scores) < min_clusters:
        logger.info(
            "Only %d cluster(s) (< min_clusters_for_relative=%d): percentile "
            "thresholds would pick by rank alone -> using absolute thresholds.",
            len(cluster_scores), min_clusters,
        )
        return "absolute"
    return mode


def _density_value(scores: dict, mode: str) -> float:
    """Absolute mode compares the scale-free density_ratio; relative mode
    ranks raw density_score (equivalent within one run). Falls back to
    density_score for callers that don't provide density_ratio."""
    if mode == "absolute" and "density_ratio" in scores:
        return scores["density_ratio"]
    return scores["density_score"]


def _compute_thresholds(cluster_scores: dict, config: dict, mode: str) -> dict:
    """
    Compute overlap/density thresholds, either from percentiles across the
    current run's clusters ("relative" mode) or fixed values ("absolute" mode).
    """
    selection_cfg = config["selection"]

    if mode == "relative":
        rel_cfg = selection_cfg["relative"]
        overlap_scores = [c["overlap_score"] for c in cluster_scores.values()]
        density_scores = [_density_value(c, mode) for c in cluster_scores.values()]
        thresholds = {
            "high_overlap": float(np.percentile(overlap_scores, rel_cfg["high_overlap_percentile"])),
            "very_high_overlap": float(np.percentile(overlap_scores, rel_cfg["very_high_overlap_percentile"])),
            "low_density": float(np.percentile(density_scores, rel_cfg["low_density_percentile"])),
        }
    else:
        abs_cfg = selection_cfg["absolute"]
        thresholds = {
            "high_overlap": abs_cfg["high_overlap_threshold"],
            "very_high_overlap": abs_cfg["very_high_overlap_threshold"],
            "low_density": abs_cfg.get("low_density_ratio_threshold", abs_cfg.get("low_density_threshold")),
        }

    logger.info("Selection thresholds (mode=%s): %s", mode, thresholds)
    return thresholds


# ---------------------------------------------------------------------------
# Per-cluster decision
# ---------------------------------------------------------------------------

def select_resampler_for_cluster(overlap_score: float, density_score: float, thresholds: dict) -> str:
    """Rule-based decision for a single cluster's resampler choice."""
    if overlap_score >= thresholds["very_high_overlap"]:
        return "smote_enn"
    elif overlap_score >= thresholds["high_overlap"]:
        return "smote_tomek"
    elif density_score <= thresholds["low_density"]:
        return "adasyn"
    else:
        return "smote"


def select_resamplers_for_clusters(cluster_scores: dict, config: dict) -> dict:
    """
    Decide a resampler name for every cluster.

    Args:
        cluster_scores: output of density_scoring.score_clusters(), i.e.
            {cluster_id: {"size":.., "overlap_score":.., "density_score":..}}
        config: the "resamplers" section of configs/models/resamplers.yaml

    Returns:
        dict: {cluster_id: resampler_name}
    """
    mode = _effective_mode(cluster_scores, config)
    thresholds = _compute_thresholds(cluster_scores, config, mode)
    min_size_for_smote = config["selection"].get("min_cluster_size_for_smote", 6)

    selection = {}
    for cluster_id, scores in cluster_scores.items():
        if scores["size"] < min_size_for_smote:
            chosen = "random_oversample"
            logger.info(
                "Cluster %s: size=%d < min_cluster_size_for_smote=%d -> forcing random_oversample",
                cluster_id, scores["size"], min_size_for_smote,
            )
        else:
            chosen = select_resampler_for_cluster(scores["overlap_score"], _density_value(scores, mode), thresholds)
        selection[cluster_id] = chosen
        logger.info("Cluster %s -> resampler: %s", cluster_id, chosen)

    return selection


# ---------------------------------------------------------------------------
# Resampler instantiation
# ---------------------------------------------------------------------------

def build_resampler_instance(name: str, cluster_size: int, config: dict, n_target: int = None):
    """
    Instantiate a configured resampler object for a given cluster, capping
    k_neighbors/n_neighbors so it never exceeds cluster_size - 1 — the hard
    requirement SMOTE-family methods impose (n_samples > n_neighbors).

    Args:
        name: one of VALID_RESAMPLERS
        cluster_size: number of minority samples in this cluster
        config: the "resamplers" section of configs/models/resamplers.yaml
        n_target: how many minority samples (label 1) the resampler should
            produce in its local fit, i.e. cluster_size + this cluster's share
            of synthetic samples. None = imblearn's "auto" (balance the local
            neighbourhood only).

    Returns:
        An unfitted imbalanced-learn resampler instance.
    """
    if name not in VALID_RESAMPLERS:
        raise ValueError(f"Unknown resampler '{name}' (expected one of {sorted(VALID_RESAMPLERS)}).")

    max_k = max(1, cluster_size - 1)
    default_seed = config.get("default_random_state", 42)
    strategy = {1: int(n_target)} if n_target is not None else "auto"

    if name == "smote":
        cfg = config["smote"]
        k = min(cfg.get("k_neighbors", 5), max_k)
        return SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=cfg.get("random_state", default_seed))

    if name == "adasyn":
        cfg = config["adasyn"]
        k = min(cfg.get("n_neighbors", 5), max_k)
        return ADASYN(sampling_strategy=strategy, n_neighbors=k, random_state=cfg.get("random_state", default_seed))

    if name == "smote_tomek":
        cfg = config["smote_tomek"]
        k = min(cfg.get("smote_k_neighbors", 5), max_k)
        inner_smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=cfg.get("random_state", default_seed))
        return SMOTETomek(smote=inner_smote, random_state=cfg.get("random_state", default_seed))

    if name == "smote_enn":
        cfg = config["smote_enn"]
        k = min(cfg.get("smote_k_neighbors", 5), max_k)
        inner_smote = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=cfg.get("random_state", default_seed))
        return SMOTEENN(smote=inner_smote, random_state=cfg.get("random_state", default_seed))

    # name == "random_oversample"
    return RandomOverSampler(sampling_strategy=strategy, random_state=default_seed)


def build_resampler_instances_for_clusters(cluster_scores: dict, config: dict, n_targets: dict = None) -> dict:
    """
    Convenience wrapper: runs select_resamplers_for_clusters() and then
    instantiates the corresponding resampler object for every cluster.
    Used directly by src/resampling/apply_resampling.py.

    Args:
        n_targets: optional {cluster_id: target minority count} passed to
            build_resampler_instance(); None = local "auto" balancing.

    Returns:
        dict: {cluster_id: (resampler_name, resampler_instance)}
    """
    selection = select_resamplers_for_clusters(cluster_scores, config)

    instances = {}
    for cluster_id, resampler_name in selection.items():
        cluster_size = cluster_scores[cluster_id]["size"]
        n_target = n_targets.get(cluster_id) if n_targets is not None else None
        instance = build_resampler_instance(resampler_name, cluster_size, config, n_target=n_target)
        instances[cluster_id] = (resampler_name, instance)

    return instances