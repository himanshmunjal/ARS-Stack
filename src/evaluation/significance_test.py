"""
src/evaluation/significance_test.py

Statistical significance testing for comparing ARS-Stack against baselines
across repeated cross-validation folds (configs/base.yaml ->
cross_validation.repeats), using the Wilcoxon signed-rank test — the same
test used in Paper 7 from the literature review.

Wilcoxon is used (rather than a paired t-test) because it doesn't assume the
per-fold score differences are normally distributed, which is safer with a
small number of folds (5-15 paired scores is typical here).

Usage:
    from src.evaluation.significance_test import wilcoxon_test, compare_metrics_across_folds

    result = wilcoxon_test(baseline_f1_scores, ars_stack_f1_scores)
    if result["significant"]:
        print(f"ARS-Stack significantly {result['direction']} baseline (p={result['p_value']:.4f})")
"""

import logging

import numpy as np
from scipy.stats import wilcoxon

logger = logging.getLogger("significance_test")

DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Single-metric Wilcoxon test
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: list,
    scores_b: list,
    alpha: float = DEFAULT_ALPHA,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> dict:
    """
    Paired Wilcoxon signed-rank test between two sets of per-fold scores for
    the SAME metric (e.g. F1 from 15 repeated-CV folds for baseline vs
    ARS-Stack, fold-for-fold paired).

    Args:
        scores_a: e.g. baseline's per-fold scores
        scores_b: e.g. ARS-Stack's per-fold scores (same folds, same order)
        alpha: significance threshold (default 0.05, standard for this kind of comparison)
        alternative: "two-sided", "greater" (b > a), or "less" (b < a)
        zero_method: how ties/zero-differences are handled (scipy default "wilcox"
                     drops zero-difference pairs, which is standard practice)

    Returns:
        dict with statistic, p_value, significant (bool), direction, mean_diff,
        median_diff, and n_pairs. If scores_a == scores_b exactly (identical
        arrays), scipy raises an error since there's nothing to rank — this is
        handled gracefully by returning p_value=1.0, significant=False.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must be paired (same length) — got "
            f"{len(scores_a)} vs {len(scores_b)}. Wilcoxon requires fold-for-fold pairing."
        )
    if len(scores_a) < 5:
        logger.warning(
            "Only %d paired samples — Wilcoxon's test statistic is unreliable "
            "below ~5-6 pairs. Consider more CV repeats (configs/base.yaml -> "
            "cross_validation.repeats) before trusting this result.",
            len(scores_a),
        )

    diffs = scores_b - scores_a
    if np.all(diffs == 0):
        logger.info("scores_a and scores_b are identical — no difference to test.")
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "direction": "no difference",
            "mean_diff": 0.0,
            "median_diff": 0.0,
            "n_pairs": len(scores_a),
        }

    try:
        statistic, p_value = wilcoxon(scores_a, scores_b, alternative=alternative, zero_method=zero_method)
    except ValueError as exc:
        # scipy raises if, after dropping zero-diff pairs, nothing is left to test
        logger.warning("Wilcoxon test could not be computed (%s) — treating as not significant.", exc)
        return {
            "statistic": None,
            "p_value": 1.0,
            "significant": False,
            "direction": "insufficient variation",
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "n_pairs": len(scores_a),
        }

    significant = bool(p_value < alpha)
    mean_diff = float(np.mean(diffs))

    if not significant:
        direction = "no significant difference"
    else:
        direction = "b > a" if mean_diff > 0 else "b < a"

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": significant,
        "direction": direction,
        "mean_diff": mean_diff,
        "median_diff": float(np.median(diffs)),
        "n_pairs": len(scores_a),
    }


# ---------------------------------------------------------------------------
# Multi-metric convenience wrapper
# ---------------------------------------------------------------------------

def compare_metrics_across_folds(
    baseline_fold_metrics: list,
    candidate_fold_metrics: list,
    alpha: float = DEFAULT_ALPHA,
    metrics: list = None,
) -> dict:
    """
    Runs a Wilcoxon test independently for EACH metric (F1, AUC-PR, G-mean,
    MCC), given two lists of per-fold metric dicts (same folds, same order)
    — e.g. the output of running src.evaluation.metrics.compute_metrics once
    per CV fold for the baseline pipeline and once for ARS-Stack.

    Args:
        baseline_fold_metrics: list of dicts, one per fold, e.g.
            [{"f1": 0.81, "auc_pr": 0.77, ...}, {"f1": 0.79, ...}, ...]
        candidate_fold_metrics: same structure, same fold order, for ARS-Stack
        metrics: which metric keys to test (default: all keys present in both)

    Returns:
        dict {metric_name: wilcoxon_test() result}
    """
    if len(baseline_fold_metrics) != len(candidate_fold_metrics):
        raise ValueError(
            f"Fold counts must match — got {len(baseline_fold_metrics)} baseline "
            f"folds vs {len(candidate_fold_metrics)} candidate folds."
        )

    if metrics is None:
        metrics = sorted(set(baseline_fold_metrics[0].keys()) & set(candidate_fold_metrics[0].keys()))

    results = {}
    for metric_name in metrics:
        scores_a = [fold[metric_name] for fold in baseline_fold_metrics]
        scores_b = [fold[metric_name] for fold in candidate_fold_metrics]
        results[metric_name] = wilcoxon_test(scores_a, scores_b, alpha=alpha)

    return results


def summarize_comparison(comparison_results: dict, baseline_name: str = "baseline", candidate_name: str = "candidate") -> str:
    """
    Turns compare_metrics_across_folds()'s output into a short, readable
    summary string — useful for printing directly in a notebook or logging
    to the final report.
    """
    lines = [f"Comparison: {candidate_name} vs {baseline_name}"]
    for metric_name, result in comparison_results.items():
        if result["significant"]:
            verdict = f"SIGNIFICANT (p={result['p_value']:.4f}), {candidate_name} {'better' if result['mean_diff'] > 0 else 'worse'}"
        else:
            verdict = f"not significant (p={result['p_value']:.4f})"
        lines.append(f"  {metric_name:10s}: mean diff = {result['mean_diff']:+.4f}  ->  {verdict}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Multiple-comparison correction + effect size
# ---------------------------------------------------------------------------

def holm_correction(p_values: dict, alpha: float = DEFAULT_ALPHA) -> dict:
    """
    Holm-Bonferroni step-down correction over one family of tests (e.g. the
    4 metrics of one baseline-vs-ARS-Stack comparison).

    Args:
        p_values: {name: raw p-value}
    Returns:
        {name: {"p_holm": adjusted p-value, "significant": bool}}
    """
    names = sorted(p_values, key=lambda k: p_values[k])
    m = len(names)
    adjusted, running_max = {}, 0.0
    for rank, name in enumerate(names):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running_max
    return {name: {"p_holm": adjusted[name], "significant": bool(adjusted[name] < alpha)} for name in p_values}


def rank_biserial(scores_a: list, scores_b: list) -> float:
    """
    Matched-pairs rank-biserial correlation — the effect size that goes with
    the Wilcoxon signed-rank test. In [-1, 1]; positive = b tends to beat a.
    Returns 0.0 when every pair is tied.
    """
    diffs = np.asarray(scores_b, dtype=float) - np.asarray(scores_a, dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    from scipy.stats import rankdata

    ranks = rankdata(np.abs(diffs))
    w_pos, w_neg = ranks[diffs > 0].sum(), ranks[diffs < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


# ---------------------------------------------------------------------------
# Corrected resampled t-test (per dataset, repeated k-fold CV)
# ---------------------------------------------------------------------------

def corrected_resampled_ttest(scores_a: list, scores_b: list, test_train_ratio: float) -> dict:
    """
    Nadeau & Bengio (2003) corrected resampled t-test, in the repeated k-fold
    form of Bouckaert & Frank (2004). Folds of repeated CV share most of their
    training data, so their scores are correlated; a plain paired test (t or
    Wilcoxon) then understates the variance and overstates significance. The
    correction inflates the variance by n_test / n_train:

        t = mean(d) / sqrt((1/n + n_test/n_train) * var(d)),   df = n - 1

    with d = scores_b - scores_a over the n = k * r folds.

    Args:
        test_train_ratio: n_test / n_train of one fold = 1 / (k - 1) for k-fold.
    Returns:
        {"t": float, "p_value": float (two-sided), "mean_diff": float, "n_pairs": int}
    """
    from scipy.stats import t as t_dist

    d = np.asarray(scores_b, dtype=float) - np.asarray(scores_a, dtype=float)
    n = len(d)
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1)) if n > 1 else 0.0
    if var_d == 0.0:
        p = 1.0 if mean_d == 0.0 else 0.0
        return {"t": float("inf") if mean_d else 0.0, "p_value": p, "mean_diff": mean_d, "n_pairs": n}
    t_stat = mean_d / np.sqrt((1.0 / n + test_train_ratio) * var_d)
    p = float(2 * t_dist.sf(abs(t_stat), df=n - 1))
    return {"t": float(t_stat), "p_value": p, "mean_diff": mean_d, "n_pairs": n}


# ---------------------------------------------------------------------------
# Cross-dataset comparison (Demšar 2006; Benavoli et al. 2016)
# ---------------------------------------------------------------------------

def average_ranks(score_table) -> "pd.Series":
    """
    score_table: DataFrame, rows = datasets, columns = methods, values = mean
    score (higher is better). Rank 1 = best within each dataset, ties share
    the average rank. Returns the mean rank per method (lower is better).
    """
    ranks = score_table.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()


def friedman_test(score_table) -> dict:
    """
    Friedman test that all methods perform equally across datasets, plus the
    Iman-Davenport F correction (less conservative; Demšar 2006, sec. 3.2.2).
    Rows with a missing method are dropped.
    """
    from scipy.stats import f as f_dist
    from scipy.stats import friedmanchisquare

    table = score_table.dropna(axis=0, how="any")
    n, k = table.shape
    if n < 2 or k < 3:
        return {"n_datasets": n, "n_methods": k, "chi2": None, "p_value": None,
                "iman_davenport_F": None, "p_value_iman_davenport": None}
    chi2, p = friedmanchisquare(*[table[c].to_numpy() for c in table.columns])
    denom = n * (k - 1) - chi2
    if denom <= 0:
        f_stat, p_f = float("inf"), 0.0
    else:
        f_stat = (n - 1) * chi2 / denom
        p_f = float(f_dist.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    return {"n_datasets": n, "n_methods": k, "chi2": float(chi2), "p_value": float(p),
            "iman_davenport_F": float(f_stat), "p_value_iman_davenport": p_f}
