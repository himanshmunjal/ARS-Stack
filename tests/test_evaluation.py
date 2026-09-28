"""
tests/test_evaluation.py

Unit tests for src/evaluation/metrics.py and src/evaluation/significance_test.py.
"""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics
from src.evaluation.significance_test import holm_correction, rank_biserial, wilcoxon_test


def test_binary_metrics_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m == pytest.approx({"f1": 1.0, "auc_pr": 1.0, "g_mean": 1.0, "mcc": 1.0})


def test_multiclass_metrics_when_a_class_is_missing_from_the_fold():
    """A tiny yeast class can be absent from a CV test fold; the model still
    outputs a probability column for it."""
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
    m = compute_metrics(y_true, proba.argmax(axis=1), proba, task_type="multiclass")
    assert m["auc_pr"] == pytest.approx(1.0)
    assert m["mcc"] == pytest.approx(1.0)


def test_multiclass_metrics_reject_labels_beyond_proba_columns():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 3]), np.array([0, 1]), np.full((2, 2), 0.5), task_type="multiclass")


def test_holm_correction_matches_manual_computation():
    out = holm_correction({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    # sorted: a(0.01*4=0.04), c(0.03*3=0.09), b(0.04*2=0.08 -> max 0.09), d(0.5)
    assert out["a"]["p_holm"] == pytest.approx(0.04)
    assert out["c"]["p_holm"] == pytest.approx(0.09)
    assert out["b"]["p_holm"] == pytest.approx(0.09)
    assert out["d"]["p_holm"] == pytest.approx(0.5)
    assert out["a"]["significant"] and not out["b"]["significant"]


def test_rank_biserial_extremes_and_ties():
    a = [0.1, 0.2, 0.3, 0.4]
    assert rank_biserial(a, [x + 0.1 for x in a]) == pytest.approx(1.0)
    assert rank_biserial(a, [x - 0.1 for x in a]) == pytest.approx(-1.0)
    assert rank_biserial(a, a) == 0.0


def test_wilcoxon_identical_scores_not_significant():
    r = wilcoxon_test([0.5] * 6, [0.5] * 6)
    assert r["p_value"] == 1.0 and not r["significant"]


# ---------------------------------------------------------------------------
# Corrected resampled t-test + cross-dataset tests
# ---------------------------------------------------------------------------

import pandas as pd

from src.evaluation.significance_test import average_ranks, corrected_resampled_ttest, friedman_test


def test_corrected_ttest_is_more_conservative_than_naive():
    from scipy.stats import ttest_rel

    rng = np.random.default_rng(0)
    a = rng.normal(0.80, 0.02, size=15)
    b = a + rng.normal(0.01, 0.01, size=15)
    corrected = corrected_resampled_ttest(a, b, test_train_ratio=1 / 4)
    naive_p = ttest_rel(b, a).pvalue
    assert corrected["p_value"] > naive_p
    assert corrected["mean_diff"] == pytest.approx(np.mean(b - a))


def test_corrected_ttest_matches_formula():
    a = np.array([0.70, 0.72, 0.71, 0.69, 0.73])
    b = np.array([0.72, 0.75, 0.71, 0.72, 0.74])
    d = b - a
    t_expected = d.mean() / np.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
    assert corrected_resampled_ttest(a, b, 0.25)["t"] == pytest.approx(t_expected)


def test_corrected_ttest_identical_scores():
    r = corrected_resampled_ttest([0.5] * 5, [0.5] * 5, 0.25)
    assert r["p_value"] == 1.0


def test_average_ranks_and_ties():
    table = pd.DataFrame({"a": [0.9, 0.8, 0.7], "b": [0.8, 0.8, 0.6], "c": [0.1, 0.2, 0.3]})
    ranks = average_ranks(table)
    # dataset 2: a and b tie for first -> 1.5 each
    assert ranks["a"] == pytest.approx((1 + 1.5 + 1) / 3)
    assert ranks["b"] == pytest.approx((2 + 1.5 + 2) / 3)
    assert ranks["c"] == pytest.approx(3.0)
    assert list(ranks.index) == ["a", "b", "c"]


def test_friedman_detects_consistent_ordering():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.5, 0.9, size=12)
    table = pd.DataFrame({"good": base + 0.05, "mid": base, "bad": base - 0.05})
    res = friedman_test(table)
    assert res["n_datasets"] == 12 and res["n_methods"] == 3
    assert res["p_value"] < 0.001 and res["p_value_iman_davenport"] < 0.001


def test_friedman_needs_three_methods():
    assert friedman_test(pd.DataFrame({"a": [1, 2], "b": [2, 1]}))["p_value"] is None
