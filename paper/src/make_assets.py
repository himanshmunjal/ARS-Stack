"""
paper/src/make_assets.py

Generates every results table (tables/*.tex) and copies/creates every results
figure (figures/) used by the paper drafts, directly from the experiment
outputs — so no number in the paper is transcribed by hand.

Inputs:
    reports/ablation_results/            full repeated-CV study (folds, summaries, analysis/)
    reports/baseline_results_*.csv       literature baseline reproduction
    paper/src/data/pilot_means.csv       validated-selection pilot (fold data on the
                                         pilot-validated-selection branch)

Usage (from the repository root):
    python paper/src/make_assets.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "reports" / "ablation_results"
SRC = Path(__file__).resolve().parent
TAB, FIG = SRC / "tables", SRC / "figures"

METRICS = ["f1", "auc_pr", "g_mean", "mcc"]
MNAME = {"f1": "F1", "auc_pr": "AUC-PR", "g_mean": "G-mean", "mcc": "MCC"}
VNAME = {
    "ars_stack": "ARS-Stack", "equal_average": "Equal-weight avg.", "meta_stacking": "Meta-learner stacking",
    "ars_dbscan": "ARS (DBSCAN)", "plain_smote": "SMOTE", "plain_adasyn": "ADASYN",
    "plain_smote_tomek": "SMOTE-Tomek", "plain_smote_enn": "SMOTE-ENN", "no_resampling": "No resampling",
    "ars_select": "ARS-Select (validated)", "smote_select": "SMOTE-Select (validated)",
}
DNAME = {"yeast_multiclass": "yeast (10-class)", "creditcard": "creditcard"}


def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def dname(d):
    return tex(DNAME.get(d, d))


def write(name, body):
    (TAB / name).write_text(body, encoding="utf-8")


def load_folds():
    F = {}
    for p in (RES / "folds").glob("*__*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        F[(d["settings"]["dataset"], d["settings"]["variant"])] = d["folds"]
    return F


# ---------------------------------------------------------------------------
def table_datasets(chars):
    core = ["creditcard", "mammography", "yeast_multiclass"]
    chars = chars.copy()
    chars["group"] = np.where(chars.index.isin(core), "core", "benchmark")
    chars = chars.sort_values(["group", "minority_count"], ascending=[False, True])
    rows = []
    for d, r in chars.iterrows():
        rows.append(f"{dname(d)} & {r.group} & {int(r.n_samples):,} & {int(r.n_features)} & {int(r.n_classes)} & "
                    f"{int(r.minority_count):,} & {r.imbalance_ratio:.1f} \\\\")
    write("datasets.tex", "\\begin{tabular}{llrrrrr}\n\\toprule\n"
          "Dataset & Group & Samples & Features & Classes & Minority & IR \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_ranks():
    ranks = pd.read_csv(RES / "cross_dataset_ranks.csv").pivot(index="variant", columns="metric", values="average_rank")
    tests = pd.read_csv(RES / "cross_dataset_tests.csv")
    fried = tests.groupby("metric")["friedman_p"].first()
    ranks = ranks[METRICS].sort_values("auc_pr")
    best = ranks.min()
    rows = []
    for v, r in ranks.iterrows():
        cells = [(f"\\textbf{{{r[m]:.2f}}}" if np.isclose(r[m], best[m]) else f"{r[m]:.2f}") for m in METRICS]
        name = f"\\textbf{{{VNAME[v]}}}" if v == "ars_stack" else VNAME[v]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    frow = "Friedman $p$ & " + " & ".join(f"{fried[m]:.2g}" for m in METRICS) + " \\\\"
    write("ranks.tex", "\\begin{tabular}{lrrrr}\n\\toprule\nVariant & " + " & ".join(MNAME[m] for m in METRICS)
          + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\midrule\n" + frow + "\n\\bottomrule\n\\end{tabular}\n")


def table_crossdataset():
    t = pd.read_csv(RES / "cross_dataset_tests.csv")
    order = ["equal_average", "meta_stacking", "ars_dbscan", "plain_smote", "plain_smote_tomek", "plain_adasyn",
             "plain_smote_enn", "no_resampling"]
    rows = []
    for v in order:
        cells = []
        for m in ["f1", "auc_pr", "mcc"]:
            r = t[(t.metric == m) & (t.baseline == v)].iloc[0]
            p = f"{r.p_wilcoxon_holm:.3f}" if r.p_wilcoxon_holm < 1 else "1"
            if r.significant:
                p = f"\\textbf{{{p}}}"
            cells.append(f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)} & {r.mean_diff:+.4f} & {p}")
        rows.append(f"{VNAME[v]} & " + " & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{l" + "rrr" * 3 + "}\n\\toprule\n & \\multicolumn{3}{c}{F1} & \\multicolumn{3}{c}{AUC-PR}"
            " & \\multicolumn{3}{c}{MCC} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "ARS-Stack vs. & W/T/L & $\\Delta$ & $p_{\\mathrm{Holm}}$ & W/T/L & $\\Delta$ & $p_{\\mathrm{Holm}}$"
            " & W/T/L & $\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n")
    write("crossdataset.tex", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_bayes():
    b = pd.read_csv(RES / "analysis" / "bayes_cross_dataset.csv")
    order = ["equal_average", "meta_stacking", "ars_dbscan", "plain_smote", "plain_smote_tomek", "plain_adasyn",
             "plain_smote_enn", "no_resampling"]
    short = {"practically equivalent": "equiv.", "candidate better": "ARS better", "baseline better": "base better",
             "undecided": "undecided"}
    rows = []
    for v in order:
        cells = []
        for m in ["f1", "auc_pr", "mcc"]:
            r = b[(b.metric == m) & (b.baseline == v)].iloc[0]
            cells.append(f"{r.p_baseline_better:.2f}/{r.p_equivalent:.2f}/{r.p_candidate_better:.2f} & {short[r.verdict]}")
        rows.append(f"{VNAME[v]} & " + " & ".join(cells) + " \\\\")
    head = ("\\begin{tabular}{l" + "cl" * 3 + "}\n\\toprule\n & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{AUC-PR}"
            " & \\multicolumn{2}{c}{MCC} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "ARS-Stack vs. & $P(<)/P(=)/P(>)$ & Verdict & $P(<)/P(=)/P(>)$ & Verdict & $P(<)/P(=)/P(>)$ & Verdict"
            " \\\\\n\\midrule\n")
    write("bayes.tex", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    per = pd.read_csv(RES / "analysis" / "bayes_per_dataset.csv")
    cnt = (per.groupby(["baseline", "verdict"]).size().unstack(fill_value=0)
           .reindex(columns=["candidate better", "practically equivalent", "baseline better", "undecided"], fill_value=0)
           .reindex(order))
    rows = [f"{VNAME[v]} & " + " & ".join(str(int(x)) for x in r) + " \\\\" for v, r in cnt.iterrows()]
    write("bayes_perdataset.tex", "\\begin{tabular}{lrrrr}\n\\toprule\nARS-Stack vs. & ARS better & Equivalent & "
          "Baseline better & Undecided \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_perdataset(chars):
    L = pd.read_csv(RES / "summary_long.csv")
    t = L[L.metric == "auc_pr"].pivot(index="dataset", columns="variant", values="mean")
    cols = ["no_resampling", "plain_smote", "plain_smote_tomek", "plain_adasyn", "plain_smote_enn", "meta_stacking",
            "ars_stack"]
    t = t[cols].loc[chars.sort_values("minority_count").index]
    rows = []
    for d, r in t.iterrows():
        best = r.max()
        cells = [(f"\\textbf{{{r[c]:.3f}}}" if np.isclose(r[c], best) else f"{r[c]:.3f}") for c in cols]
        rows.append(f"{dname(d)} & {int(chars.loc[d, 'minority_count']):,} & " + " & ".join(cells) + " \\\\")
    mean_row = "Mean & -- & " + " & ".join(f"{t[c].mean():.3f}" for c in cols) + " \\\\"
    heads = ["None", "SMOTE", "SM-Tomek", "ADASYN", "SM-ENN", "Meta-stack", "ARS-Stack"]
    write("perdataset_aucpr.tex", "\\begin{tabular}{lr" + "r" * len(cols) + "}\n\\toprule\nDataset & Min. & "
          + " & ".join(heads) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\midrule\n" + mean_row
          + "\n\\bottomrule\n\\end{tabular}\n")


def table_datasets_results(chars):
    """Dataset characteristics + mean AUC-PR per variant, ordered by minority size."""
    L = pd.read_csv(RES / "summary_long.csv")
    t = L[L.metric == "auc_pr"].pivot(index="dataset", columns="variant", values="mean")
    cols = ["no_resampling", "plain_smote", "plain_smote_tomek", "plain_adasyn", "plain_smote_enn",
            "meta_stacking", "ars_stack"]
    core = {"creditcard", "mammography", "yeast_multiclass"}
    order = chars.sort_values("minority_count").index
    t = t.loc[order, cols]
    rows = []
    for d, r in t.iterrows():
        c = chars.loc[d]
        best = r.max()
        cells = [(f"\\textbf{{{r[k]:.3f}}}" if np.isclose(r[k], best) else f"{r[k]:.3f}") for k in cols]
        name = dname(d) + ("$^\\ast$" if d in core else "")
        rows.append(f"{name} & {int(c.n_samples):,} & {int(c.n_features)} & {int(c.minority_count):,} & "
                    f"{c.imbalance_ratio:.1f} & " + " & ".join(cells) + " \\\\")
    mean_row = "Mean & & & & & " + " & ".join(f"{t[k].mean():.3f}" for k in cols) + " \\\\"
    heads = ["None", "SMOTE", "SM-Tomek", "ADASYN", "SM-ENN", "Meta", "ARS"]
    write("datasets_results.tex",
          "\\begin{tabular}{lrrrr" + "r" * len(cols) + "}\n\\toprule\n"
          " & & & & & \\multicolumn{7}{c}{Mean AUC-PR} \\\\\n\\cmidrule(l){6-12}\n"
          "Dataset & $n$ & $d$ & Min. & IR & " + " & ".join(heads) + " \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\midrule\n" + mean_row + "\n\\bottomrule\n\\end{tabular}\n")


def table_core():
    L = pd.read_csv(RES / "summary_long.csv")
    vs = ["no_resampling", "plain_smote", "plain_smote_enn", "meta_stacking", "ars_stack"]
    out = []
    for d in ["creditcard", "mammography", "yeast_multiclass"]:
        out.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{dname(d)}}}}} \\\\")
        sub = L[L.dataset == d]
        best = sub.groupby("metric")["mean"].max()
        for v in vs:
            cells = []
            for m in METRICS:
                r = sub[(sub.variant == v) & (sub.metric == m)].iloc[0]
                s = f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}"
                cells.append(f"\\textbf{{{s}}}" if np.isclose(r["mean"], best[m]) else s)
            out.append(f"\\quad {VNAME[v]} & " + " & ".join(cells) + " \\\\")
    write("core.tex", "\\begin{tabular}{lrrrr}\n\\toprule\nVariant & " + " & ".join(MNAME[m] for m in METRICS)
          + " \\\\\n\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")


def table_literature():
    cc = pd.read_csv(ROOT / "reports" / "baseline_results_creditcard.csv")
    mm = pd.read_csv(ROOT / "reports" / "baseline_results_mammography.csv")
    rows = []
    fmt = lambda x: "--" if pd.isna(x) else f"{x:.3f}"
    for df in (cc, mm):
        for _, r in df.iterrows():
            src = "ours" if r["source"] == "this project" else "reported"
            name = (r["model"].replace("paper_7_", "Paper 7: ").replace("paper_9_", "Paper 9: ")
                    .replace("paper_10_", "Paper 10: ").replace("_", " "))
            rows.append(f"{r['dataset']} & {tex(name)} & {src} & {fmt(r['f1'])} & {fmt(r['auc_pr'])} & "
                        f"{fmt(r.get('mcc'))} \\\\")
    write("literature_baselines.tex", "\\begin{tabular}{lllrrr}\n\\toprule\nDataset & Model & Source & F1 & AUC-PR & "
          "MCC \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_pilot():
    p = pd.read_csv(SRC / "data" / "pilot_means.csv")
    agg = p.groupby(["variant", "metric"])["mean"].mean().unstack()[METRICS]
    ranks = {}
    for m in METRICS:
        t = p[p.metric == m].pivot(index="dataset", columns="variant", values="mean")
        ranks[m] = t.rank(axis=1, ascending=False).mean()
    order = ["no_resampling", "plain_smote", "ars_stack", "smote_select", "ars_select"]
    rows = []
    for v in order:
        rows.append(f"{VNAME[v]} & " + " & ".join(f"{agg.loc[v, m]:.4f} ({ranks[m][v]:.2f})" for m in METRICS) + " \\\\")
    write("pilot.tex", "\\begin{tabular}{lrrrr}\n\\toprule\nVariant & " + " & ".join(MNAME[m] for m in METRICS)
          + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def diagnostics(F):
    ds = sorted({d for d, _ in F})
    spread = [max(f["weights"].values()) - min(f["weights"].values()) for d in ds for f in F[(d, "ars_stack")]]
    n_clu, split = [], 0
    usage = {}
    for d in ds:
        for f in F[(d, "ars_stack")]:
            per_class = {}
            for c in f["resampling"].get("clusters", []):
                per_class[c["class"]] = per_class.get(c["class"], 0) + 1
                usage[(d, c["resampler_used"])] = usage.get((d, c["resampler_used"]), 0) + 1
            n_clu += list(per_class.values())
            split += sum(n > 1 for n in per_class.values())
    tomek_same = sum(a["metrics"] == b["metrics"] for d in ds
                     for a, b in zip(F[(d, "plain_smote")], F[(d, "plain_smote_tomek")]))
    eq_same = sum(a["metrics"] == b["metrics"] for d in ds for a, b in zip(F[(d, "ars_stack")], F[(d, "equal_average")]))
    hours = sum(f["seconds"]["resample"] + f["seconds"]["fit"] for fl in F.values() for f in fl) / 3600
    stats = {"weight_spread_median": float(np.median(spread)), "weight_spread_p90": float(np.percentile(spread, 90)),
             "clusters_mean": float(np.mean(n_clu)), "split_share": split / len(n_clu),
             "tomek_identical_folds": tomek_same, "n_folds_per_variant": 15 * len(ds),
             "equal_identical_folds": eq_same, "compute_hours": hours,
             "n_folds_total": sum(len(v) for v in F.values())}
    (SRC / "data" / "diagnostics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return usage


def figure_usage(usage):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Categorical slots 1-6 of the validated reference palette, fixed order.
    cats = [("smote", "SMOTE", "#2a78d6"), ("smote_tomek", "SMOTE-Tomek", "#eb6834"),
            ("smote_enn", "SMOTE-ENN", "#1baf7a"), ("adasyn", "ADASYN", "#eda100"),
            ("random_oversample", "Random oversampling", "#e87ba4"), ("none", "None (class already at target)", "#008300")]
    ds = sorted({d for d, _ in usage})
    tot = {d: sum(v for (dd, _), v in usage.items() if dd == d) for d in ds}
    share = pd.DataFrame({k: [usage.get((d, k), 0) / tot[d] for d in ds] for k, _, _ in cats}, index=ds)
    share = share.sort_values(["smote", "smote_tomek", "smote_enn"], ascending=False)
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor=surface)
    ax.set_facecolor(surface)
    left = np.zeros(len(share))
    for k, label, color in cats:
        ax.barh(range(len(share)), share[k], left=left, color=color, edgecolor=surface, linewidth=1.2,
                height=0.78, label=label)
        left += share[k].to_numpy()
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([DNAME.get(d, d) for d in share.index], fontsize=8, color=ink2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of minority clusters (all 15 folds)", fontsize=9, color=ink2)
    ax.tick_params(axis="x", labelsize=8, colors=ink2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#e4e3df")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.09))
    fig.tight_layout()
    fig.savefig(FIG / "resampler_usage.pdf", facecolor=surface)
    plt.close(fig)


def main():
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    chars = pd.read_csv(RES / "analysis" / "dataset_characteristics.csv").set_index("dataset")
    F = load_folds()
    table_datasets(chars)
    table_ranks()
    table_crossdataset()
    table_bayes()
    table_perdataset(chars)
    table_datasets_results(chars)
    table_core()
    table_literature()
    table_pilot()
    figure_usage(diagnostics(F))
    for src, dst in [(RES / "average_ranks.png", "average_ranks.png"),
                     (RES / "ablation_comparison.png", "core_comparison.png"),
                     (RES / "analysis" / "critical_difference_auc_pr.png", "cd_auc_pr.png"),
                     (RES / "analysis" / "critical_difference_f1.png", "cd_f1.png"),
                     (RES / "analysis" / "critical_difference_mcc.png", "cd_mcc.png"),
                     (RES / "analysis" / "resampling_benefit.png", "resampling_benefit.png")]:
        shutil.copy(src, FIG / dst)
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))
    print("figures:", sorted(p.name for p in FIG.iterdir()))
    print(json.loads((SRC / "data" / "diagnostics.json").read_text()))


if __name__ == "__main__":
    main()
