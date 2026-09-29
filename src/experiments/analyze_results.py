"""
src/experiments/analyze_results.py

Further analysis of a finished repeated-CV study (src/experiments/run_ablation.py),
computed from its saved folds only — no training:

1. Bayesian tests with a region of practical equivalence (ROPE, default ±0.01;
   Benavoli et al., JMLR 2017), turning "no significant difference" into
   posterior probabilities of "practically equivalent":
     - per dataset: Bayesian correlated t-test over the paired CV folds;
     - across datasets: Bayesian signed-rank test over per-dataset means.
2. When does resampling help? The gain of resampling over no resampling on
   each dataset, related to dataset characteristics (minority-class size,
   imbalance ratio, number of features, size) with Spearman correlations,
   Holm-corrected.
3. Critical-difference diagrams (Demšar 2006): average ranks with the Nemenyi
   critical difference, one per metric.

Outputs (in <results-dir>/analysis/):
    bayes_per_dataset.csv, bayes_cross_dataset.csv
    dataset_characteristics.csv, resampling_benefit.csv, resampling_benefit_correlations.csv
    resampling_benefit.png, critical_difference_<metric>.png
    analysis.md                              all of the above as readable tables

Usage:
    python -m src.experiments.analyze_results
    python -m src.experiments.analyze_results --results-dir reports/ablation_results --rope 0.01
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.metrics import SUPPORTED_METRICS
from src.evaluation.significance_test import (
    average_ranks,
    bayesian_correlated_ttest,
    bayesian_signed_rank,
    bayesian_verdict,
    holm_correction,
    nemenyi_critical_difference,
)

logger = logging.getLogger("analyze_results")

COLORS = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
          "accent": "#2a78d6", "series2": "#eb6834", "context": "#8f8e88"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_folds(results_dir: Path) -> tuple:
    folds, settings = {}, {}
    for path in sorted((results_dir / "folds").glob("*__*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        s = saved["settings"]
        folds[(s["dataset"], s["variant"])] = saved["folds"]
        settings[(s["dataset"], s["variant"])] = s
    if not folds:
        raise FileNotFoundError(f"No fold results in {results_dir / 'folds'} — run src.experiments.run_ablation first.")
    return folds, settings


def per_dataset_means(folds: dict) -> pd.DataFrame:
    rows = [{"dataset": d, "variant": v, "metric": m, "mean": float(np.mean([f["metrics"][m] for f in fl]))}
            for (d, v), fl in folds.items() for m in SUPPORTED_METRICS]
    return pd.DataFrame(rows)


def dataset_characteristics(datasets: list, cache: Path, configs_dir: str) -> pd.DataFrame:
    """Size, minority count, imbalance ratio and feature count of each dataset
    after cleaning/deduplication. Cached so re-analysis doesn't need the raw data."""
    cached = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["dataset"])
    missing = [d for d in datasets if d not in set(cached["dataset"])]
    if missing:
        from src.data.preprocess import load_clean_dataset

        rows = []
        for ds in missing:
            df, cfg, _ = load_clean_dataset(ds, configs_dir)
            counts = df[cfg["target_column"]].value_counts()
            rows.append({"dataset": ds, "n_samples": len(df), "n_features": df.shape[1] - 1,
                         "n_classes": len(counts), "minority_count": int(counts.min()),
                         "imbalance_ratio": float(counts.max() / counts.min())})
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache, index=False)
    return cached[cached.dataset.isin(datasets)].set_index("dataset")


# ---------------------------------------------------------------------------
# 1. Bayesian analysis
# ---------------------------------------------------------------------------

def bayes_per_dataset(folds, settings, reference, variants, rope) -> pd.DataFrame:
    rows = []
    for ds in sorted({d for d, _ in folds}):
        ref = folds.get((ds, reference))
        if ref is None:
            continue
        rho = 1.0 / settings[(ds, reference)]["n_splits"]
        for var in variants:
            other = folds.get((ds, var))
            if var == reference or other is None:
                continue
            n = min(len(ref), len(other))
            for m in SUPPORTED_METRICS:
                r = bayesian_correlated_ttest([f["metrics"][m] for f in other[:n]],
                                              [f["metrics"][m] for f in ref[:n]], rho=rho, rope=rope)
                rows.append({"dataset": ds, "baseline": var, "candidate": reference, "metric": m, **r,
                             "verdict": bayesian_verdict(r)})
    return pd.DataFrame(rows)


def bayes_cross_dataset(means, reference, variants, rope) -> pd.DataFrame:
    rows = []
    for m in SUPPORTED_METRICS:
        table = means[means.metric == m].pivot(index="dataset", columns="variant", values="mean").dropna()
        for var in variants:
            if var == reference or var not in table:
                continue
            r = bayesian_signed_rank(table[var].to_numpy(), table[reference].to_numpy(), rope=rope)
            rows.append({"metric": m, "baseline": var, "candidate": reference, "n_datasets": len(table), **r,
                         "verdict": bayesian_verdict(r)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. When does resampling help?
# ---------------------------------------------------------------------------

BENEFIT_PAIRS = [("plain_smote", "global SMOTE"), ("ars_stack", "ARS-Stack")]
CHARACTERISTICS = [("minority_count", "minority-class size"), ("imbalance_ratio", "imbalance ratio"),
                   ("n_features", "number of features"), ("n_samples", "dataset size")]
BENEFIT_METRICS = ["auc_pr", "f1", "mcc"]


def resampling_benefit(means: pd.DataFrame, chars: pd.DataFrame) -> tuple:
    rows = []
    for m in BENEFIT_METRICS:
        t = means[means.metric == m].pivot(index="dataset", columns="variant", values="mean")
        for var, _ in BENEFIT_PAIRS:
            if var in t and "no_resampling" in t:
                for ds, gain in (t[var] - t["no_resampling"]).dropna().items():
                    rows.append({"dataset": ds, "metric": m, "resampler": var, "gain_over_none": float(gain)})
    benefit = pd.DataFrame(rows).join(chars, on="dataset")

    tests = []
    for m in BENEFIT_METRICS:
        for var, _ in BENEFIT_PAIRS:
            sub = benefit[(benefit.metric == m) & (benefit.resampler == var)]
            if len(sub) < 5:
                continue
            for col, _ in CHARACTERISTICS:
                rho, p = spearmanr(sub[col], sub["gain_over_none"])
                tests.append({"metric": m, "resampler": var, "characteristic": col,
                              "spearman_rho": float(rho), "p_raw": float(p), "n_datasets": len(sub)})
    corr = pd.DataFrame(tests)
    if not corr.empty:
        holm = holm_correction({i: p for i, p in corr["p_raw"].items()})
        corr["p_holm"] = [holm[i]["p_holm"] for i in corr.index]
        corr["significant"] = [holm[i]["significant"] for i in corr.index]
    return benefit, corr


def plot_resampling_benefit(benefit: pd.DataFrame, path: Path) -> None:
    """Gain over no resampling vs minority-class size (log scale), one panel
    per metric, one point per dataset; two series with a legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = COLORS
    metrics = [m for m in BENEFIT_METRICS if m in set(benefit.metric)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.4), squeeze=False, facecolor=c["surface"])
    styles = {"plain_smote": (c["series2"], "s"), "ars_stack": (c["accent"], "o")}
    for ax, m in zip(axes[0], metrics):
        ax.set_facecolor(c["surface"])
        ax.axhline(0, color=c["ink2"], lw=1)
        for var, label in BENEFIT_PAIRS:
            sub = benefit[(benefit.metric == m) & (benefit.resampler == var)]
            color, marker = styles[var]
            ax.scatter(sub["minority_count"], sub["gain_over_none"], s=64, marker=marker, color=color,
                       edgecolor=c["surface"], linewidth=1.5, label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("minority-class size (log scale)", fontsize=9, color=c["ink2"])
        ax.set_ylabel(f"{m}: resampling − no resampling", fontsize=9, color=c["ink2"])
        ax.set_title(m, fontsize=11, color=c["ink"], loc="left")
        ax.grid(color=c["grid"], lw=0.8)
        ax.tick_params(labelsize=8, colors=c["ink2"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["grid"])
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Gain from resampling vs minority-class size (one point per dataset; above 0 = resampling helps)",
                 fontsize=11, color=c["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=c["surface"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Critical-difference diagrams
# ---------------------------------------------------------------------------

def _cliques(ranks: pd.Series, cd: float) -> list:
    """Maximal groups of methods whose average ranks lie within cd of each other."""
    names, vals = list(ranks.index), ranks.to_numpy()
    groups = []
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(g[0] <= i and j <= g[1] for g in groups):
            groups.append((i, j))
    return [(names[a], names[b], a, b) for a, b in groups]


def plot_critical_difference(ranks: pd.Series, cd: float, n_datasets: int, reference: str, metric: str,
                             path: Path) -> None:
    """Demšar-style CD diagram: rank axis on top, better methods on the left,
    thick bars join methods not significantly different (Nemenyi, alpha 0.05)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = COLORS
    ranks = ranks.sort_values()
    k = len(ranks)
    lo, hi = 1, k
    half = int(np.ceil(k / 2))
    fig_h = 1.6 + 0.32 * half
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor=c["surface"])
    ax.set_facecolor(c["surface"])
    ax.set_xlim(lo - 3.0, hi + 3.0)
    ax.set_ylim(-(half + 1.2), 1.6)
    ax.axis("off")

    # rank axis
    ax.plot([lo, hi], [0, 0], color=c["ink2"], lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0, 0.12], color=c["ink2"], lw=1)
        ax.text(r, 0.22, str(r), ha="center", va="bottom", fontsize=9, color=c["ink2"])
    # CD bar
    ax.plot([lo, lo + cd], [1.15, 1.15], color=c["ink"], lw=1.5)
    for x in (lo, lo + cd):
        ax.plot([x, x], [1.05, 1.25], color=c["ink"], lw=1.5)
    ax.text(lo + cd / 2, 1.3, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9, color=c["ink"])

    # labels: best half on the left, rest on the right
    for i, (name, r) in enumerate(ranks.items()):
        left = i < half
        row = -(i + 1) if left else -(k - i)
        color = c["accent"] if name == reference else c["ink"]
        weight = "bold" if name == reference else "normal"
        x_end = lo - 0.3 if left else hi + 0.3        # leader line stops here; label sits beyond it
        ax.plot([r, r], [0, row], color=color, lw=1)
        ax.plot([r, x_end], [row, row], color=color, lw=1)
        ax.text(x_end + (-0.08 if left else 0.08), row, f"{name}  {r:.2f}" if left else f"{r:.2f}  {name}",
                ha="right" if left else "left", va="center", fontsize=9, color=color, fontweight=weight)

    # cliques
    for n, (_, _, a, b) in enumerate(_cliques(ranks, cd)):
        y = -0.35 - 0.22 * n
        ax.plot([ranks.iloc[a] - 0.03, ranks.iloc[b] + 0.03], [y, y], color=c["ink"], lw=3.5, solid_capstyle="round")

    ax.set_title(f"{metric} — average rank over {n_datasets} datasets (1 = best); "
                 f"bars join methods not significantly different (Nemenyi, α = 0.05)",
                 fontsize=10, color=c["ink"], loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=c["surface"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(df, spec):
    df = df.copy()
    for col, f in spec.items():
        if col in df:
            df[col] = df[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "")
    return df


def write_report(out: Path, reference, rope, bayes_ds, bayes_x, chars, benefit, corr, cd_rows) -> None:
    L = [f"# Further analysis of the ablation study (reference: {reference})", "",
         "Computed from the saved CV folds by `python -m src.experiments.analyze_results`.", ""]

    L += ["## 1. Bayesian analysis (region of practical equivalence ±%g)" % rope, "",
          "Posterior probabilities that the baseline is practically better, the two are practically "
          f"equivalent (|difference| < {rope}), or {reference} is practically better (Benavoli et al., JMLR 2017). "
          "A verdict is given when one region reaches 0.95.", "",
          "### Across datasets (Bayesian signed-rank test on per-dataset means)", ""]
    if not bayes_x.empty:
        show = _fmt(bayes_x[["metric", "baseline", "n_datasets", "mean_diff", "p_baseline_better", "p_equivalent",
                             "p_candidate_better", "verdict"]],
                    {"mean_diff": "{:+.4f}", "p_baseline_better": "{:.3f}", "p_equivalent": "{:.3f}",
                     "p_candidate_better": "{:.3f}"})
        L += [show.to_markdown(index=False), ""]
    if not bayes_ds.empty:
        counts = (bayes_ds.groupby(["baseline", "verdict"]).size().unstack(fill_value=0)
                  .reindex(columns=["candidate better", "practically equivalent", "baseline better", "undecided"],
                           fill_value=0))
        L += ["### Per dataset (Bayesian correlated t-test over CV folds)", "",
              "Number of (dataset × metric) comparisons with each verdict; full table in bayes_per_dataset.csv.", "",
              counts.reset_index().to_markdown(index=False), ""]

    L += ["## 2. When does resampling help?", "",
          "Gain = mean score with resampling − mean score without, per dataset. Spearman correlation "
          "between the gain and each dataset characteristic, Holm-corrected over all tests below.", ""]
    if not corr.empty:
        show = _fmt(corr[["metric", "resampler", "characteristic", "spearman_rho", "p_raw", "p_holm", "significant",
                          "n_datasets"]], {"spearman_rho": "{:+.3f}", "p_raw": "{:.4f}", "p_holm": "{:.4f}"})
        L += [show.to_markdown(index=False), ""]
    if not benefit.empty:
        g = (benefit[benefit.metric == "auc_pr"].pivot(index="dataset", columns="resampler", values="gain_over_none")
             .join(chars[["minority_count", "imbalance_ratio", "n_features", "n_samples"]])
             .sort_values("minority_count"))
        L += ["AUC-PR gain over no resampling per dataset, ordered by minority-class size:", "",
              g.round(4).reset_index().to_markdown(index=False), ""]

    L += ["## 3. Critical-difference diagrams", "",
          "Average ranks and Nemenyi critical difference (α = 0.05); the diagrams are "
          "critical_difference_<metric>.png. Methods whose ranks differ by less than CD are not "
          "significantly different.", "", pd.DataFrame(cd_rows).to_markdown(index=False), ""]
    (out / "analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bayesian, meta-analysis and CD-diagram analysis of saved CV folds.")
    parser.add_argument("--results-dir", default="reports/ablation_results")
    parser.add_argument("--reference", default="ars_stack")
    parser.add_argument("--rope", type=float, default=0.01, help="Half-width of the region of practical equivalence.")
    parser.add_argument("--configs-dir", default="configs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for name in ("preprocess", "ingest"):
        logging.getLogger(name).setLevel(logging.WARNING)

    results_dir = Path(args.results_dir)
    out = results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    folds, settings = load_folds(results_dir)
    means = per_dataset_means(folds)
    variants = sorted({v for _, v in folds})
    datasets = sorted({d for d, _ in folds})
    logger.info("Loaded %d datasets x %d variants", len(datasets), len(variants))

    # 1. Bayesian
    bayes_ds = bayes_per_dataset(folds, settings, args.reference, variants, args.rope)
    bayes_ds.to_csv(out / "bayes_per_dataset.csv", index=False)
    bayes_x = bayes_cross_dataset(means, args.reference, variants, args.rope)
    bayes_x.to_csv(out / "bayes_cross_dataset.csv", index=False)

    # 2. Resampling benefit
    chars = dataset_characteristics(datasets, out / "dataset_characteristics.csv", args.configs_dir)
    benefit, corr = resampling_benefit(means, chars)
    benefit.to_csv(out / "resampling_benefit.csv", index=False)
    corr.to_csv(out / "resampling_benefit_correlations.csv", index=False)
    plot_resampling_benefit(benefit, out / "resampling_benefit.png")

    # 3. CD diagrams
    cd_rows = []
    for m in SUPPORTED_METRICS:
        table = means[means.metric == m].pivot(index="dataset", columns="variant", values="mean").dropna()
        ranks = average_ranks(table)
        cd = nemenyi_critical_difference(len(ranks), len(table))
        plot_critical_difference(ranks, cd, len(table), args.reference, m, out / f"critical_difference_{m}.png")
        cd_rows.append({"metric": m, "n_datasets": len(table), "n_methods": len(ranks), "critical_difference": round(cd, 3),
                        "best": ranks.index[0], "best_rank": round(ranks.iloc[0], 2),
                        f"{args.reference}_rank": round(ranks[args.reference], 2),
                        "worst": ranks.index[-1], "worst_rank": round(ranks.iloc[-1], 2)})

    write_report(out, args.reference, args.rope, bayes_ds, bayes_x, chars, benefit, corr, cd_rows)
    logger.info("Analysis written to %s", out)


if __name__ == "__main__":
    main()
