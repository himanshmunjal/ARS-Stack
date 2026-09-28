"""
src/resampling/apply_resampling.py

Ties together Weeks 4-5 into the actual adaptive resampling step:

    1. src.clustering.minority_clustering.cluster_minority_class
       -> clusters the minority class into local neighborhoods
    2. src.clustering.density_scoring.score_clusters
       -> computes overlap_score/density_score per cluster
    3. src.resampling.resampler_selector.build_resampler_instances_for_clusters
       -> picks + instantiates a resampler (SMOTE/ADASYN/SMOTE-Tomek/
          SMOTE-ENN/RandomOverSampler) for EACH cluster individually

This module's own job is the part none of the above three do: actually
RUN each cluster's chosen resampler on that cluster's local neighborhood
(its own minority points + nearby majority context points) and MERGE every
cluster's contribution back into one balanced training set — handling
synthetic-sample bookkeeping and majority-point removal (from SMOTE-Tomek/
SMOTE-ENN cleaning) correctly across overlapping cluster contexts.

How much to oversample: the total number of synthetic samples is set
GLOBALLY — target_balance_ratio * majority_count - minority_count (see
configs/models/resamplers.yaml) — and split across clusters in proportion to
cluster size. The global baselines (resample_globally) use the same ratio, so
adaptive vs. fixed-resampler comparisons differ only in WHICH resampler runs
where, not in how much synthetic data is added. (Previously each cluster was
only balanced against its ~3x-sized local majority context, leaving the
adaptive training set far more imbalanced than plain SMOTE's.)

Both binary (apply_adaptive_resampling) and multiclass
(apply_adaptive_resampling_multiclass: each non-majority class in turn) are
supported; resample_training_set() is the single entry point used by
src/pipeline.py, dags/ and src/experiments/.

Usage:
    python -m src.resampling.apply_resampling --dataset creditcard
    python -m src.resampling.apply_resampling --dataset mammography
    python -m src.resampling.apply_resampling --dataset all
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.neighbors import NearestNeighbors

from src.clustering.density_scoring import score_clusters
from src.clustering.minority_clustering import cluster_minority_class
from src.data.ingest import CORE_DATASETS, DATASET_REGISTRY, ensure_parent_dir, load_config, load_dataset_config
from src.resampling.resampler_selector import build_resampler_instance, build_resampler_instances_for_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("apply_resampling")


# ---------------------------------------------------------------------------
# Row-identity helpers
#
# Used to figure out, after a resampler runs on a cluster's local
# neighborhood, which rows are original-and-retained, original-but-removed
# (SMOTE-Tomek/SMOTE-ENN cleaning), or newly-synthesized — WITHOUT adding an
# id column that would distort the resampler's own distance calculations.
# Safe because untouched original rows come back bit-for-bit identical.
#
# IMPORTANT: matching is done with COUNTS (a multiset), not plain set
# membership/a plain dict. A plain dict of {byte_key: global_idx} silently
# collapses duplicate rows to a single global index (last one wins), which
# then misattributes genuine duplicates as "removed" even for resamplers
# (SMOTE/ADASYN) that mathematically never touch the majority class at all.
# This matters concretely here: the credit card fraud dataset is documented
# to contain ~1,081 exact-duplicate transactions, so duplicate rows inside a
# cluster's majority context are a realistic, not just theoretical, case.
# ---------------------------------------------------------------------------

def _match_retained_global_idx(input_arr: np.ndarray, input_global_idx: np.ndarray, output_arr: np.ndarray) -> np.ndarray:
    """
    Given an input array (with known global indices) and a resampler's
    output array (containing some retained subset of those input rows, byte-
    for-byte, plus possibly other rows), returns which global indices are
    still present — correctly handling duplicate (bit-identical) input rows
    by matching counts per unique byte-key rather than set membership.
    """
    from collections import Counter, defaultdict

    key_to_global_ids = defaultdict(list)
    for row, gid in zip(input_arr, input_global_idx):
        key_to_global_ids[row.tobytes()].append(gid)

    output_key_counts = Counter(row.tobytes() for row in output_arr)

    retained_global_idx = []
    for key, gids in key_to_global_ids.items():
        n_retained = min(len(gids), output_key_counts.get(key, 0))
        # Duplicates are bit-identical, so which specific one we call
        # "retained" vs "removed" is arbitrary and functionally irrelevant —
        # only the COUNT of removals matters for correctness.
        retained_global_idx.extend(gids[:n_retained])

    return np.array(retained_global_idx, dtype=int)


def _find_synthetic_mask(input_arr: np.ndarray, output_arr: np.ndarray) -> np.ndarray:
    """
    Boolean mask over output_arr: True = this row does NOT correspond to any
    input row (i.e. it's newly synthesized), correctly accounting for
    duplicate input rows via count-based matching rather than set membership.
    """
    from collections import Counter

    remaining = Counter(row.tobytes() for row in input_arr)
    is_synthetic = np.zeros(len(output_arr), dtype=bool)
    for i, row in enumerate(output_arr):
        key = row.tobytes()
        if remaining.get(key, 0) > 0:
            remaining[key] -= 1
            is_synthetic[i] = False
        else:
            is_synthetic[i] = True
    return is_synthetic


# ---------------------------------------------------------------------------
# Majority context for a cluster
# ---------------------------------------------------------------------------

def get_majority_context(
    X_majority_all: np.ndarray,
    majority_global_idx: np.ndarray,
    cluster_points: np.ndarray,
    k_majority: int,
) -> np.ndarray:
    """
    Finds the k majority points nearest to a cluster's centroid, giving
    cleaning-based resamplers (SMOTE-Tomek/SMOTE-ENN) enough local majority
    context to identify boundary noise. Returns GLOBAL indices (into the
    original training set).
    """
    k_majority = min(k_majority, len(X_majority_all))
    if k_majority <= 0:
        return np.array([], dtype=int)

    centroid = cluster_points.mean(axis=0, keepdims=True)
    nn = NearestNeighbors(n_neighbors=k_majority).fit(X_majority_all)
    _, local_idx = nn.kneighbors(centroid)
    return majority_global_idx[local_idx[0]]


# ---------------------------------------------------------------------------
# Synthetic-sample budget
# ---------------------------------------------------------------------------

def compute_n_synthetic_needed(n_minority: int, n_majority: int, target_ratio: float) -> int:
    """Synthetic samples needed so that minority / majority == target_ratio
    (never negative: an already-large-enough class gets 0)."""
    return max(0, int(round(target_ratio * n_majority)) - int(n_minority))


def allocate_synthetic_counts(cluster_sizes: dict, n_total: int) -> dict:
    """
    Split n_total synthetic samples across clusters in proportion to cluster
    size (largest-remainder rounding, so the parts sum exactly to n_total).

    Args:
        cluster_sizes: {cluster_id: n_minority_points}
    Returns:
        {cluster_id: n_synthetic_to_generate}
    """
    ids = list(cluster_sizes.keys())
    sizes = np.array([cluster_sizes[c] for c in ids], dtype=float)
    if n_total <= 0 or sizes.sum() == 0:
        return {c: 0 for c in ids}

    exact = sizes / sizes.sum() * n_total
    alloc = np.floor(exact).astype(int)
    remainder = n_total - alloc.sum()
    for i in np.argsort(-(exact - alloc))[:remainder]:
        alloc[i] += 1
    return {c: int(a) for c, a in zip(ids, alloc)}


# ---------------------------------------------------------------------------
# Per-cluster local resampling
# ---------------------------------------------------------------------------

def resample_single_cluster(
    X: np.ndarray,
    minority_global_idx_cluster: np.ndarray,
    majority_context_global_idx: np.ndarray,
    resampler_name: str,
    resampler_instance,
    fallback=None,
):
    """
    Runs one (already-built, correctly k-capped) resampler instance on one
    cluster's local neighborhood.

    Args:
        fallback: optional (name, instance) tried if resampler_instance raises
            (e.g. ADASYN when no minority point has a majority neighbour), so
            the cluster still receives its synthetic-sample allocation.

    Returns:
        synthetic_minority_rows: newly generated minority feature rows
        retained_majority_global_idx: majority context indices that survived
        retained_minority_global_idx: this cluster's minority indices that survived
        resampler_used: name of the resampler that actually ran ("none" if all failed)
    """
    X_min_local = X[minority_global_idx_cluster]
    X_maj_local = X[majority_context_global_idx]

    if len(X_min_local) == 0:
        return np.empty((0, X.shape[1])), majority_context_global_idx, minority_global_idx_cluster, "none"

    X_local = np.vstack([X_maj_local, X_min_local])
    y_local = np.array([0] * len(X_maj_local) + [1] * len(X_min_local))

    attempts = [(resampler_name, resampler_instance)]
    if fallback is not None:
        attempts.append(fallback)

    X_res = y_res = None
    resampler_used = "none"
    for name, instance in attempts:
        try:
            X_res, y_res = instance.fit_resample(X_local, y_local)
            resampler_used = name
            break
        except Exception as exc:
            logger.warning("Resampler '%s' failed on this cluster (%s).", name, exc)

    if X_res is None:
        logger.warning("All resamplers failed on this cluster — keeping original points only.")
        return np.empty((0, X.shape[1])), majority_context_global_idx, minority_global_idx_cluster, "none"

    res_majority_rows = X_res[y_res == 0]
    res_minority_rows = X_res[y_res == 1]

    retained_majority_global_idx = _match_retained_global_idx(
        X_maj_local, majority_context_global_idx, res_majority_rows,
    )
    retained_minority_global_idx = _match_retained_global_idx(
        X_min_local, minority_global_idx_cluster, res_minority_rows,
    )
    synthetic_mask = _find_synthetic_mask(X_min_local, res_minority_rows)
    synthetic_minority_rows = res_minority_rows[synthetic_mask]

    return synthetic_minority_rows, retained_majority_global_idx, retained_minority_global_idx, resampler_used


def _resample_one_class(
    X: np.ndarray,
    y: np.ndarray,
    class_label,
    X_majority_all: np.ndarray,
    majority_global_idx: np.ndarray,
    n_synthetic_needed: int,
    clustering_cfg: dict,
    resamplers_cfg: dict,
    majority_context_multiplier: float,
) -> dict:
    """
    cluster -> score -> select -> allocate -> resample for ONE minority class.
    Shared by the binary and multiclass paths.
    """
    minority_global_idx = np.where(y == class_label)[0]
    X_minority = X[minority_global_idx]

    cluster_labels = cluster_minority_class(X_minority, clustering_cfg)
    cluster_scores = score_clusters(X_minority, X, y, cluster_labels, clustering_cfg, positive_label=class_label)

    allocation = allocate_synthetic_counts(
        {cid: sc["size"] for cid, sc in cluster_scores.items()}, n_synthetic_needed,
    )
    n_targets = {cid: cluster_scores[cid]["size"] + allocation[cid] for cid in cluster_scores}
    resampler_instances = build_resampler_instances_for_clusters(cluster_scores, resamplers_cfg, n_targets=n_targets)
    fallback_name = resamplers_cfg.get("fallback_resampler", "smote")

    synthetic, retained_min, retained_maj, removed_maj = [], set(), set(), set()
    cluster_report = []

    for cluster_id, (resampler_name, resampler_instance) in resampler_instances.items():
        cluster_minority_global_idx = minority_global_idx[cluster_labels == cluster_id]
        cluster_size = cluster_scores[cluster_id]["size"]

        if allocation[cluster_id] == 0:
            # Nothing to generate (class already at the target ratio) — keep as-is.
            retained_min.update(cluster_minority_global_idx.tolist())
            resampler_used, synthetic_rows, removed_from_this_cluster = "none", np.empty((0, X.shape[1])), set()
        else:
            cluster_points = X[cluster_minority_global_idx]
            k_majority = max(int(cluster_size * majority_context_multiplier), clustering_cfg.get("min_cluster_size", 10))
            majority_context_idx = get_majority_context(X_majority_all, majority_global_idx, cluster_points, k_majority)

            fallback = None
            if fallback_name and fallback_name != resampler_name:
                fallback = (fallback_name, build_resampler_instance(
                    fallback_name, cluster_size, resamplers_cfg, n_target=n_targets[cluster_id],
                ))

            synthetic_rows, retained_maj_idx, retained_min_idx, resampler_used = resample_single_cluster(
                X, cluster_minority_global_idx, majority_context_idx,
                resampler_name, resampler_instance, fallback=fallback,
            )
            retained_min.update(retained_min_idx.tolist())
            retained_maj.update(retained_maj_idx.tolist())
            removed_from_this_cluster = set(majority_context_idx.tolist()) - set(retained_maj_idx.tolist())
            removed_maj.update(removed_from_this_cluster)

        synthetic.append(synthetic_rows)
        scores = cluster_scores[cluster_id]
        cluster_report.append({
            "cluster_id": int(cluster_id),
            "cluster_size": int(cluster_size),
            "overlap_score": round(float(scores["overlap_score"]), 4),
            "density_score": _json_float(scores["density_score"]),
            "density_ratio": _json_float(scores.get("density_ratio", float("nan"))),
            "resampler_chosen": resampler_name,
            "resampler_used": resampler_used,
            "n_synthetic_allocated": int(allocation[cluster_id]),
            "n_synthetic_generated": int(len(synthetic_rows)),
            "n_majority_removed": int(len(removed_from_this_cluster)),
        })

    synthetic_arr = np.vstack(synthetic) if any(len(a) for a in synthetic) else np.empty((0, X.shape[1]))
    return {
        "original_count": int(len(minority_global_idx)),
        "kept_minority_idx": np.array(sorted(retained_min), dtype=int),
        "synthetic": synthetic_arr,
        "retained_majority": retained_maj,
        "removed_majority": removed_maj,
        "clusters": cluster_report,
    }


def _json_float(value: float):
    return round(float(value), 4) if np.isfinite(value) else str(value)


# ---------------------------------------------------------------------------
# Orchestration: adaptive resampling across ALL clusters (binary)
# ---------------------------------------------------------------------------

def apply_adaptive_resampling(
    X: np.ndarray,
    y: np.ndarray,
    positive_label,
    clustering_cfg: dict,
    resamplers_cfg: dict,
    majority_context_multiplier: float = 3.0,
):
    """
    Full adaptive resampling pipeline: cluster -> score -> select -> allocate
    -> resample -> merge. Returns a balanced (X, y) plus a JSON-serializable report.

    Args:
        X, y: full training set (both classes)
        positive_label: which value in y is the minority/positive class
        clustering_cfg: the "clustering" section of configs/models/clustering.yaml
        resamplers_cfg: the "resamplers" section of configs/models/resamplers.yaml
    """
    minority_mask = y == positive_label
    majority_global_idx = np.where(~minority_mask)[0]
    target_ratio = resamplers_cfg.get("target_balance_ratio", 1.0)
    n_needed = compute_n_synthetic_needed(int(minority_mask.sum()), len(majority_global_idx), target_ratio)

    logger.info(
        "Starting adaptive resampling: %d minority, %d majority samples, %d synthetic to generate (target ratio %.2f)",
        int(minority_mask.sum()), len(majority_global_idx), n_needed, target_ratio,
    )

    result = _resample_one_class(
        X, y, positive_label, X[majority_global_idx], majority_global_idx,
        n_needed, clustering_cfg, resamplers_cfg, majority_context_multiplier,
    )

    # A majority point can be "context" for multiple clusters; only treat it
    # as removed if it was never retained in ANY cluster it appeared in.
    final_removed_majority = result["removed_majority"] - result["retained_majority"]
    kept_majority_idx = np.array(sorted(set(majority_global_idx.tolist()) - final_removed_majority), dtype=int)
    kept_minority_idx = result["kept_minority_idx"]
    synthetic_minority_arr = result["synthetic"]

    X_final = np.vstack([X[kept_majority_idx], X[kept_minority_idx], synthetic_minority_arr])
    y_final = np.array(
        [0] * len(kept_majority_idx) + [1] * len(kept_minority_idx) + [1] * len(synthetic_minority_arr)
    )

    report = {
        "mode": "adaptive",
        "target_balance_ratio": target_ratio,
        "n_clusters": len(result["clusters"]),
        "original_minority_count": result["original_count"],
        "original_majority_count": int(len(majority_global_idx)),
        "final_minority_count": int(len(kept_minority_idx) + len(synthetic_minority_arr)),
        "final_majority_count": int(len(kept_majority_idx)),
        "total_synthetic_generated": int(len(synthetic_minority_arr)),
        "total_majority_removed": int(len(final_removed_majority)),
        "clusters": result["clusters"],
    }

    logger.info(
        "Adaptive resampling done. Final class balance: majority=%d, minority=%d",
        report["final_majority_count"], report["final_minority_count"],
    )

    return X_final, y_final, report


# ---------------------------------------------------------------------------
# Multiclass adaptive resampling
#
# Generalizes apply_adaptive_resampling() above to >2 classes by treating
# every non-majority class as its own "minority class" and running it
# through the SAME cluster -> score -> select -> resample pipeline in turn,
# one class at a time. Context points for cleaning-based resamplers
# (SMOTE-Tomek/SMOTE-ENN) are drawn from the single MAJORITY class only
# (not from other minority classes) — this keeps the "overlap with the
# dominant class" semantics that density_scoring.py's docstring describes,
# and avoids the ambiguity of a majority-context point simultaneously being
# "kept" via its own class's processing and "removed" via another class's
# cleaning pass.
# ---------------------------------------------------------------------------

def apply_adaptive_resampling_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    clustering_cfg: dict,
    resamplers_cfg: dict,
    majority_context_multiplier: float = 3.0,
):
    """
    Multiclass counterpart to apply_adaptive_resampling(). Every class except
    the single largest ("majority") class is clustered and adaptively
    resampled independently, each up to target_balance_ratio * majority
    count; the majority class itself is kept as-is except for points removed
    by another class's cleaning-based resampler (SMOTE-Tomek/SMOTE-ENN).

    Returns:
        X_final, y_final (y_final keeps the ORIGINAL class labels, not 0/1),
        plus a JSON-serializable report.
    """
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    minority_classes = [c for c in classes if c != majority_class]
    target_ratio = resamplers_cfg.get("target_balance_ratio", 1.0)

    majority_global_idx_all = np.where(y == majority_class)[0]
    X_majority_all = X[majority_global_idx_all]

    logger.info(
        "Starting multiclass adaptive resampling: majority_class=%s (%d samples), "
        "%d minority classes to resample",
        majority_class, len(majority_global_idx_all), len(minority_classes),
    )

    retained_majority_idx_all, removed_majority_idx_all = set(), set()
    per_class, class_reports = {}, []

    for c in minority_classes:
        n_needed = compute_n_synthetic_needed(int((y == c).sum()), len(majority_global_idx_all), target_ratio)
        result = _resample_one_class(
            X, y, c, X_majority_all, majority_global_idx_all,
            n_needed, clustering_cfg, resamplers_cfg, majority_context_multiplier,
        )
        per_class[c] = result
        retained_majority_idx_all.update(result["retained_majority"])
        removed_majority_idx_all.update(result["removed_majority"])
        class_reports.append({
            "class": int(c),
            "original_count": result["original_count"],
            "final_count": int(len(result["kept_minority_idx"]) + len(result["synthetic"])),
            "n_clusters": len(result["clusters"]),
            "clusters": result["clusters"],
        })

    final_removed_majority = removed_majority_idx_all - retained_majority_idx_all
    kept_majority_idx = np.array(sorted(set(majority_global_idx_all.tolist()) - final_removed_majority), dtype=int)

    X_parts = [X[kept_majority_idx]]
    y_parts = [np.full(len(kept_majority_idx), majority_class)]
    for c in minority_classes:
        kept_idx, synth = per_class[c]["kept_minority_idx"], per_class[c]["synthetic"]
        if len(kept_idx):
            X_parts.append(X[kept_idx])
            y_parts.append(np.full(len(kept_idx), c))
        if len(synth):
            X_parts.append(synth)
            y_parts.append(np.full(len(synth), c))

    X_final = np.vstack(X_parts)
    y_final = np.concatenate(y_parts)

    final_classes, final_counts = np.unique(y_final, return_counts=True)
    report = {
        "mode": "adaptive_multiclass",
        "target_balance_ratio": target_ratio,
        "majority_class": int(majority_class),
        "original_class_counts": {int(cl): int(n) for cl, n in zip(classes, counts)},
        "final_class_counts": {int(cl): int(n) for cl, n in zip(final_classes, final_counts)},
        "total_majority_removed": int(len(final_removed_majority)),
        "classes": class_reports,
    }

    logger.info(
        "Multiclass adaptive resampling done. Final class counts: %s",
        report["final_class_counts"],
    )

    return X_final, y_final, report


# ---------------------------------------------------------------------------
# Global (fixed, single-resampler) baselines
# ---------------------------------------------------------------------------

GLOBAL_RESAMPLERS = ("smote", "adasyn", "smote_tomek", "smote_enn")
RESAMPLING_METHODS = ("adaptive", "none") + GLOBAL_RESAMPLERS


def resample_globally(X: np.ndarray, y: np.ndarray, method: str, resamplers_cfg: dict):
    """
    One resampler applied to the whole training set — the fixed-resampler
    baselines ARS-Stack is compared against (Papers 7/9/10). Uses the same
    per-method hyperparameters and the same target_balance_ratio as the
    adaptive path. Works for binary and multiclass y.
    """
    if method not in GLOBAL_RESAMPLERS:
        raise ValueError(f"Unknown global resampler '{method}' (expected one of {GLOBAL_RESAMPLERS}).")

    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    target_ratio = resamplers_cfg.get("target_balance_ratio", 1.0)
    target_count = int(round(target_ratio * counts.max()))
    strategy = {
        c: max(int(n), target_count) for c, n in zip(classes, counts) if c != majority_class
    }

    seed = resamplers_cfg.get("default_random_state", 42)
    min_class_count = int(counts.min())
    max_k = max(1, min_class_count - 1)

    def smote(k_key):
        cfg = resamplers_cfg.get(method, {})
        return SMOTE(
            sampling_strategy=strategy, k_neighbors=min(cfg.get(k_key, 5), max_k),
            random_state=cfg.get("random_state", seed),
        )

    if method == "adasyn":
        X_res, y_res, used = _adasyn_per_class(X, y, strategy, resamplers_cfg, max_k, seed)
    else:
        if method == "smote":
            sampler = smote("k_neighbors")
        elif method == "smote_tomek":
            sampler = SMOTETomek(smote=smote("smote_k_neighbors"), random_state=seed)
        else:
            sampler = SMOTEENN(smote=smote("smote_k_neighbors"), random_state=seed)
        X_res, y_res = sampler.fit_resample(X, y)
        used = method

    final_classes, final_counts = np.unique(y_res, return_counts=True)
    report = {
        "mode": f"global_{method}",
        "resampler_used": used,  # "adasyn", or {class: "adasyn"|"smote"} when ADASYN fell back
        "target_balance_ratio": target_ratio,
        "original_class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        "final_class_counts": {int(c): int(n) for c, n in zip(final_classes, final_counts)},
    }
    return X_res, y_res, report


def _adasyn_per_class(X, y, strategy: dict, resamplers_cfg: dict, max_k: int, seed: int):
    """
    Global ADASYN, one minority class at a time. ADASYN raises when a class
    has no majority-class neighbours (RuntimeError) or when its per-point
    rounding yields zero new samples (ValueError); that class alone then falls
    back to SMOTE with the same target count, instead of the whole run failing
    or every class silently switching to SMOTE. imblearn appends new samples
    after the original rows, so each call's synthetic rows are X_res[len(X):].
    """
    cfg = resamplers_cfg.get("adasyn", {})
    k = min(cfg.get("n_neighbors", 5), max_k)
    rs = cfg.get("random_state", seed)
    X_parts, y_parts, used = [X], [y], {}
    for c, target in strategy.items():
        n_new = target - int((y == c).sum())
        if n_new <= 0:
            continue
        try:
            X_res, y_res = ADASYN(sampling_strategy={c: target}, n_neighbors=k, random_state=rs).fit_resample(X, y)
            used[int(c)] = "adasyn"
        except (RuntimeError, ValueError) as exc:
            logger.warning("Global ADASYN failed for class %s (%s) — using SMOTE for that class.", c, exc)
            X_res, y_res = SMOTE(sampling_strategy={c: target}, k_neighbors=k, random_state=rs).fit_resample(X, y)
            used[int(c)] = "smote"
        X_parts.append(X_res[len(X):])
        y_parts.append(y_res[len(y):])
    if all(u == "adasyn" for u in used.values()):
        used = "adasyn"
    return np.vstack(X_parts), np.concatenate(y_parts), used


def resample_training_set(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task_type: str,
    clustering_cfg: dict,
    resamplers_cfg: dict,
    positive_label=1,
):
    """
    Single entry point for every resampling variant, used by src/pipeline.py,
    dags/ars_stack_dag.py and src/experiments/run_ablation.py.

    Args:
        method: "adaptive" (ARS-Stack), "none" (no resampling), or a global
                baseline: "smote", "adasyn", "smote_tomek", "smote_enn".
        task_type: "binary" or "multiclass".
    """
    if method == "adaptive":
        if task_type == "multiclass":
            return apply_adaptive_resampling_multiclass(X, y, clustering_cfg, resamplers_cfg)
        return apply_adaptive_resampling(X, y, positive_label, clustering_cfg, resamplers_cfg)
    if method == "none":
        classes, counts = np.unique(y, return_counts=True)
        counts_dict = {int(c): int(n) for c, n in zip(classes, counts)}
        return X, y, {"mode": "none", "original_class_counts": counts_dict, "final_class_counts": counts_dict}
    return resample_globally(X, y, method, resamplers_cfg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_for_dataset(dataset_name: str, configs_dir: str = "configs"):
    ds_full_cfg = load_dataset_config(dataset_name, configs_dir)
    ds_cfg = ds_full_cfg["dataset"]

    clustering_full_cfg = load_config(Path(configs_dir) / "models" / "clustering.yaml")
    resamplers_full_cfg = load_config(Path(configs_dir) / "models" / "resamplers.yaml")
    clustering_cfg = clustering_full_cfg["clustering"]
    resamplers_cfg = resamplers_full_cfg["resamplers"]

    processed_dir = Path(ds_cfg["processed_file"]).parent
    train_path = processed_dir / f"{dataset_name}_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run preprocess.py first.")

    train_df = pd.read_csv(train_path)
    target_column = ds_cfg["target_column"]
    X = train_df.drop(columns=[target_column]).values
    y = train_df[target_column].values

    task_type = "multiclass" if ds_cfg.get("type", "binary") == "multiclass" else "binary"
    X_res, y_res, report = resample_training_set(
        X, y, "adaptive", task_type, clustering_cfg, resamplers_cfg,
        positive_label=1,  # preprocess.py encodes the positive class to 1
    )

    feature_cols = [c for c in train_df.columns if c != target_column]
    out_df = pd.DataFrame(X_res, columns=feature_cols)
    out_df[target_column] = y_res

    # Written under reports/ (not data/processed/, which is a DVC-tracked
    # output that `dvc status` would then report as modified).
    out_dir = Path("reports") / "resampling"
    out_path = out_dir / f"{dataset_name}_train_resampled.csv"
    ensure_parent_dir(out_path)
    out_df.to_csv(out_path, index=False)

    report_path = out_dir / f"{dataset_name}_resampling_report.json"
    ensure_parent_dir(report_path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved resampled train set to %s", out_path)
    logger.info("Saved resampling report to %s", report_path)
    return out_df, report


def main():
    parser = argparse.ArgumentParser(description="Apply adaptive per-cluster resampling.")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
    )
    parser.add_argument("--configs-dir", type=str, default="configs")
    args = parser.parse_args()

    targets = list(CORE_DATASETS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        try:
            run_for_dataset(name, args.configs_dir)
        except Exception as exc:
            logger.error("Failed to resample '%s': %s", name, exc)


if __name__ == "__main__":
    main()