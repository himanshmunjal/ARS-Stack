"""
tests/test_resampler_selector.py

Unit tests for src/resampling/resampler_selector.py — the core novel
module of ARS-Stack.
"""

import numpy as np
import pytest
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

from src.resampling.resampler_selector import (
    VALID_RESAMPLERS,
    build_resampler_instance,
    build_resampler_instances_for_clusters,
    select_resampler_for_cluster,
    select_resamplers_for_clusters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_resampler_config():
    return {
        "candidates": ["smote", "adasyn", "smote_tomek", "smote_enn"],
        "default_random_state": 42,
        "smote": {"k_neighbors": 5, "random_state": 42},
        "adasyn": {"n_neighbors": 5, "random_state": 42},
        "smote_tomek": {"smote_k_neighbors": 5, "random_state": 42},
        "smote_enn": {"smote_k_neighbors": 5, "random_state": 42},
        "selection": {
            "mode": "relative",
            "relative": {
                "high_overlap_percentile": 66,
                "very_high_overlap_percentile": 90,
                "low_density_percentile": 33,
            },
            "absolute": {
                "high_overlap_threshold": 0.4,
                "very_high_overlap_threshold": 0.7,
                "low_density_ratio_threshold": 0.75,
            },
            "min_cluster_size_for_smote": 6,
        },
    }


@pytest.fixture
def sample_cluster_scores():
    """Four synthetic clusters spanning the full decision space."""
    return {
        0: {"size": 20, "overlap_score": 0.05, "density_score": 10.0},  # low overlap, dense -> smote
        1: {"size": 20, "overlap_score": 0.10, "density_score": 0.1},   # low overlap, sparse -> adasyn
        2: {"size": 20, "overlap_score": 0.60, "density_score": 5.0},   # high overlap -> smote_tomek
        3: {"size": 20, "overlap_score": 0.95, "density_score": 5.0},   # very high overlap -> smote_enn
    }


# ---------------------------------------------------------------------------
# select_resampler_for_cluster — direct threshold logic
# ---------------------------------------------------------------------------

def test_very_high_overlap_selects_smote_enn():
    thresholds = {"high_overlap": 0.5, "very_high_overlap": 0.75, "low_density": 1.0}
    assert select_resampler_for_cluster(0.9, 5.0, thresholds) == "smote_enn"


def test_high_overlap_selects_smote_tomek():
    thresholds = {"high_overlap": 0.5, "very_high_overlap": 0.75, "low_density": 1.0}
    assert select_resampler_for_cluster(0.6, 5.0, thresholds) == "smote_tomek"


def test_low_overlap_low_density_selects_adasyn():
    thresholds = {"high_overlap": 0.5, "very_high_overlap": 0.75, "low_density": 1.0}
    assert select_resampler_for_cluster(0.1, 0.5, thresholds) == "adasyn"


def test_low_overlap_high_density_selects_smote():
    thresholds = {"high_overlap": 0.5, "very_high_overlap": 0.75, "low_density": 1.0}
    assert select_resampler_for_cluster(0.1, 5.0, thresholds) == "smote"


def test_boundary_at_exact_threshold_goes_to_cleaning_branch():
    """At exactly the threshold, >= means it should take the 'high overlap' branch."""
    thresholds = {"high_overlap": 0.5, "very_high_overlap": 0.75, "low_density": 1.0}
    assert select_resampler_for_cluster(0.5, 5.0, thresholds) == "smote_tomek"


# ---------------------------------------------------------------------------
# select_resamplers_for_clusters — full pipeline with percentile thresholds
# ---------------------------------------------------------------------------

def test_select_resamplers_covers_all_clusters(sample_cluster_scores, base_resampler_config):
    selection = select_resamplers_for_clusters(sample_cluster_scores, base_resampler_config)
    assert set(selection.keys()) == set(sample_cluster_scores.keys())
    assert all(v in VALID_RESAMPLERS for v in selection.values())


def test_undersized_cluster_forces_random_oversample(base_resampler_config):
    scores = {0: {"size": 3, "overlap_score": 0.9, "density_score": 5.0}}  # would be smote_enn if not undersized
    selection = select_resamplers_for_clusters(scores, base_resampler_config)
    assert selection[0] == "random_oversample"


def test_absolute_mode_uses_fixed_thresholds(sample_cluster_scores, base_resampler_config):
    cfg = dict(base_resampler_config)
    cfg["selection"] = dict(base_resampler_config["selection"])
    cfg["selection"]["mode"] = "absolute"
    selection = select_resamplers_for_clusters(sample_cluster_scores, cfg)
    # cluster 3 (overlap=0.95) should still land on smote_enn under fixed 0.75 threshold
    assert selection[3] == "smote_enn"
    # cluster 0 (overlap=0.05, dense) should still land on smote
    assert selection[0] == "smote"


def test_unknown_mode_raises(sample_cluster_scores, base_resampler_config):
    cfg = dict(base_resampler_config)
    cfg["selection"] = dict(base_resampler_config["selection"])
    cfg["selection"]["mode"] = "not_a_real_mode"
    with pytest.raises(ValueError):
        select_resamplers_for_clusters(sample_cluster_scores, cfg)


# ---------------------------------------------------------------------------
# build_resampler_instance — correct types + k_neighbors capping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("smote", SMOTE),
        ("adasyn", ADASYN),
        ("smote_tomek", SMOTETomek),
        ("smote_enn", SMOTEENN),
        ("random_oversample", RandomOverSampler),
    ],
)
def test_build_resampler_instance_returns_correct_type(name, expected_type, base_resampler_config):
    instance = build_resampler_instance(name, cluster_size=20, config=base_resampler_config)
    assert isinstance(instance, expected_type)


def test_build_resampler_instance_caps_k_neighbors_for_small_cluster(base_resampler_config):
    # cluster_size=4 -> max_k = 3, well below the configured k_neighbors=5
    instance = build_resampler_instance("smote", cluster_size=4, config=base_resampler_config)
    assert instance.k_neighbors <= 3


def test_build_resampler_instance_unknown_name_raises(base_resampler_config):
    with pytest.raises(ValueError):
        build_resampler_instance("not_a_real_resampler", cluster_size=20, config=base_resampler_config)


# ---------------------------------------------------------------------------
# build_resampler_instances_for_clusters — integration + actual fit_resample
# ---------------------------------------------------------------------------

def test_build_resampler_instances_for_clusters_structure(sample_cluster_scores, base_resampler_config):
    instances = build_resampler_instances_for_clusters(sample_cluster_scores, base_resampler_config)
    assert set(instances.keys()) == set(sample_cluster_scores.keys())
    for cluster_id, (name, instance) in instances.items():
        assert name in VALID_RESAMPLERS
        assert instance is not None


def test_every_candidate_resampler_actually_runs_on_synthetic_data(base_resampler_config):
    """
    End-to-end smoke test: every resampler type the selector can choose
    must successfully fit_resample on a small, realistic imbalanced dataset
    without raising — this is what apply_resampling.py will rely on.
    """
    rng = np.random.default_rng(0)
    X_majority = rng.normal(loc=0, scale=1, size=(100, 4))
    X_minority = rng.normal(loc=3, scale=1, size=(15, 4))
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * 100 + [1] * 15)

    for name in ["smote", "adasyn", "smote_tomek", "smote_enn", "random_oversample"]:
        instance = build_resampler_instance(name, cluster_size=15, config=base_resampler_config)
        X_res, y_res = instance.fit_resample(X, y)
        assert len(X_res) >= len(X)
        assert set(np.unique(y_res)) == {0, 1}

# ---------------------------------------------------------------------------
# Relative-mode fallback + density_ratio + target counts
# ---------------------------------------------------------------------------

def test_single_cluster_is_not_forced_to_smote_enn(base_resampler_config):
    """Pre-fix bug: with one cluster, relative percentiles made it 'very high
    overlap' by definition. It must now be judged on its own scores."""
    scores = {0: {"size": 50, "overlap_score": 0.01, "density_score": 9.0, "density_ratio": 1.0}}
    assert select_resamplers_for_clusters(scores, base_resampler_config)[0] == "smote"


def test_two_clusters_fall_back_to_absolute_thresholds(base_resampler_config):
    # Both clusters barely overlap the majority class and are about as dense as
    # the class -> both plain SMOTE, not the rank-forced SMOTE-ENN + ADASYN pair.
    scores = {
        0: {"size": 40, "overlap_score": 0.10, "density_score": 1.0, "density_ratio": 1.05},
        1: {"size": 30, "overlap_score": 0.05, "density_score": 0.9, "density_ratio": 0.95},
    }
    assert set(select_resamplers_for_clusters(scores, base_resampler_config).values()) == {"smote"}


def test_absolute_mode_uses_density_ratio_when_available(base_resampler_config):
    cfg = dict(base_resampler_config)
    cfg["selection"] = {**base_resampler_config["selection"], "mode": "absolute"}
    cfg["selection"]["absolute"] = {
        "high_overlap_threshold": 0.4, "very_high_overlap_threshold": 0.7, "low_density_ratio_threshold": 0.75,
    }
    # raw density_score is large (would look dense), but the ratio says sparse
    scores = {0: {"size": 40, "overlap_score": 0.1, "density_score": 50.0, "density_ratio": 0.5}}
    assert select_resamplers_for_clusters(scores, cfg)[0] == "adasyn"


@pytest.mark.parametrize("name", ["smote", "adasyn", "smote_tomek", "smote_enn", "random_oversample"])
def test_n_target_sets_minority_output_count(name, base_resampler_config):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(60, 3)), rng.normal(1.5, 1, size=(20, 3))])
    y = np.array([0] * 60 + [1] * 20)
    instance = build_resampler_instance(name, cluster_size=20, config=base_resampler_config, n_target=200)
    _, y_res = instance.fit_resample(X, y)
    n_min = int((y_res == 1).sum())
    if name in ("smote", "random_oversample"):
        assert n_min == 200
    else:
        # ADASYN rounds per point; cleaning resamplers may remove a few
        assert 150 <= n_min <= 210
