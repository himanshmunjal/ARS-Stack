"""
final_paper/scripts/make_tables.py

Writes the LaTeX tables of the final paper (final_paper/tables/*.tex) from the
saved results, so every number in the paper is traceable to a results file.

    python3 final_paper/scripts/make_tables.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / "reports_raw" / "ablation_results"
PIPE = ROOT / "reports_raw" / "pipeline_results"
FIG = ROOT / "final_paper" / "figures"
OUT = ROOT / "final_paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {
    "ars_stack": "ARS-Stack", "equal_average": "Equal-weight avg.", "meta_stacking": "LR meta-stacking",
    "ars_dbscan": "ARS (DBSCAN)", "plain_smote": "SMOTE", "plain_adasyn": "ADASYN",
    "plain_smote_tomek": "SMOTE-Tomek", "plain_smote_enn": "SMOTE-ENN", "no_resampling": "No resampling",
}
ISOLATES = {
    "equal_average": "confidence weighting", "meta_stacking": "weighted avg.\\ vs.\\ LR meta-learner",
    "ars_dbscan": "KMeans vs.\\ DBSCAN", "plain_smote": "adaptive vs.\\ global SMOTE",
    "plain_adasyn": "adaptive vs.\\ global ADASYN", "plain_smote_tomek": "adaptive vs.\\ global SMOTE-Tomek",
    "plain_smote_enn": "adaptive vs.\\ global SMOTE-ENN", "no_resampling": "resampling vs.\\ none",
}
BASELINES = list(ISOLATES)
VARIANTS = ["ars_stack"] + BASELINES
METRICS = {"f1": "F1", "auc_pr": "AUC-PR", "g_mean": "G-mean", "mcc": "MCC"}
CORE = ["creditcard", "mammography", "yeast_multiclass"]

long = pd.read_csv(ABL / "summary_long.csv")
ranks = pd.read_csv(ABL / "cross_dataset_ranks.csv")
xt = pd.read_csv(ABL / "cross_dataset_tests.csv")
sig = pd.read_csv(ABL / "significance_results.csv")
info = pd.read_csv(FIG / "dataset_info.csv").set_index("dataset")
mean = long.pivot_table(index=["dataset", "variant"], columns="metric", values="mean")
std = long.pivot_table(index=["dataset", "variant"], columns="metric", values="std")


def tex(s):
    return s.replace("_", "\\_")


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)


# Datasets ------------------------------------------------------------------
rows = []
for ds in info.sort_values("ir").index:
    r = info.loc[ds]
    group = "core" if ds in CORE else "benchmark"
    rows.append(f"{tex(ds)} & {group} & {int(r.n_after_dedup):,} & {int(r.n_features)} & {int(r.n_classes)} & {r.ir:.1f} \\\\")
write("tab_datasets", "\n".join([
    "\\begin{longtable}{llrrrr}",
    "\\caption{The 28 evaluation datasets, sorted by imbalance ratio (IR). $N$ is the approximate size after exact-duplicate removal (from the inner-training split size); IR is majority/minority count (for yeast, largest/smallest class) on the inner training split.}\\label{tab:datasets}\\\\",
    "\\toprule", "Dataset & Group & $N$ & $d$ & Classes & IR \\\\", "\\midrule", "\\endfirsthead",
    "\\toprule", "Dataset & Group & $N$ & $d$ & Classes & IR \\\\", "\\midrule", "\\endhead",
    *rows, "\\bottomrule", "\\end{longtable}"]))

# Average ranks ---------------------------------------------------------------
rk = ranks.pivot(index="variant", columns="metric", values="average_rank")[list(METRICS)]
rk["mean"] = rk.mean(axis=1)
rk = rk.sort_values("mean")
lines = []
for v, r in rk.iterrows():
    cells = []
    for m in list(METRICS) + ["mean"]:
        val = f"{r[m]:.2f}"
        if r[m] == rk[m].min():
            val = f"\\textbf{{{val}}}"
        cells.append(val)
    name = f"\\textbf{{{NAMES[v]}}}" if v == "ars_stack" else NAMES[v]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
fr = xt.groupby("metric")[["friedman_p", "iman_davenport_p"]].first()
write("tab_ranks", "\n".join([
    "\\begin{table}[t]\\centering",
    "\\caption{Average rank of each variant over the 28 datasets (1 = best; per-dataset mean over 15 folds). Best value per column in bold. Bottom rows: Friedman and Iman--Davenport $p$-values for the null hypothesis that all nine variants are equivalent.}",
    "\\label{tab:ranks}",
    "\\begin{tabular}{lrrrrr}", "\\toprule", "Variant & F1 & AUC-PR & G-mean & MCC & Mean \\\\", "\\midrule",
    *lines, "\\midrule",
    "Friedman $p$ & " + " & ".join(f"{fr.loc[m, 'friedman_p']:.1e}" for m in METRICS) + " & -- \\\\",
    "Iman--Davenport $p$ & " + " & ".join(f"{fr.loc[m, 'iman_davenport_p']:.1e}" for m in METRICS) + " & -- \\\\",
    "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

# Cross-dataset post-hoc tests -------------------------------------------------
lines = []
for b in BASELINES:
    cells = []
    for m in METRICS:
        r = xt[(xt.metric == m) & (xt.baseline == b)].iloc[0]
        p = f"{r.p_wilcoxon_holm:.3f}"
        if r.significant:
            p = f"\\textbf{{{p}}}"
        cells.append(f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)} & {p}")
    lines.append(f"{NAMES[b]} & " + " & ".join(cells) + " \\\\")
write("tab_xtests", "\n".join([
    "\\begin{table}[t]\\centering\\small",
    "\\caption{ARS-Stack versus each baseline across the 28 datasets: wins/ties/losses (W/T/L) of the per-dataset mean and Holm-corrected Wilcoxon signed-rank $p$-value (within each metric). Bold: $p<0.05$.}",
    "\\label{tab:xtests}",
    "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
    "Baseline & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{AUC-PR} & \\multicolumn{2}{c}{G-mean} & \\multicolumn{2}{c}{MCC} \\\\",
    " & W/T/L & $p$ & W/T/L & $p$ & W/T/L & $p$ & W/T/L & $p$ \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

# Per-dataset significant results -----------------------------------------------
lines = []
for b in BASELINES:
    s = sig[sig.baseline == b]
    w = int(((s.significant) & (s.mean_diff > 0)).sum())
    l = int(((s.significant) & (s.mean_diff < 0)).sum())
    where = sorted(set(s[s.significant].dataset))
    lines.append(f"{NAMES[b]} & {ISOLATES[b]} & {w} & {l} & {tex(', '.join(where)) if where else '--'} \\\\")
write("tab_sigcounts", "\n".join([
    "\\begin{table}[t]\\centering\\small",
    "\\caption{Per-dataset tests: number of (dataset, metric) pairs out of $28\\times4=112$ per baseline where the Holm-corrected Nadeau--Bengio corrected $t$-test is significant at $\\alpha=0.05$, in favour of ARS-Stack (Win) or the baseline (Loss).}",
    "\\label{tab:sigcounts}",
    "\\begin{tabular}{llrrl}", "\\toprule", "Baseline & Isolates & Win & Loss & Datasets \\\\", "\\midrule",
    *lines, "\\midrule",
    f"Total & & {int(((sig.significant) & (sig.mean_diff > 0)).sum())} & {int(((sig.significant) & (sig.mean_diff < 0)).sum())} & of {len(sig)} tests \\\\",
    "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

# Core datasets ------------------------------------------------------------------
lines = []
for v in VARIANTS:
    cells = []
    for ds in CORE:
        for m in ["auc_pr", "mcc"]:
            mu, sd = mean.loc[(ds, v), m], std.loc[(ds, v), m]
            best = max(mean.loc[(ds, u), m] for u in VARIANTS)
            val = f"{mu:.3f}$\\pm${sd:.3f}"
            cells.append(f"\\textbf{{{val}}}" if np.isclose(mu, best) else val)
    name = f"\\textbf{{{NAMES[v]}}}" if v == "ars_stack" else NAMES[v]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
write("tab_core", "\n".join([
    "\\begin{table}[t]\\centering\\small",
    "\\caption{Core datasets: mean $\\pm$ standard deviation over 15 folds (5-fold CV $\\times$ 3 repeats). Yeast metrics are macro-averaged. Best mean per column in bold.}",
    "\\label{tab:core}",
    "\\setlength{\\tabcolsep}{3pt}\\footnotesize",
    "\\begin{tabular}{lrrrrrr}", "\\toprule",
    "& \\multicolumn{2}{c}{Credit card (IR 600)} & \\multicolumn{2}{c}{Mammography (IR 30)} & \\multicolumn{2}{c}{Yeast, 10 classes} \\\\",
    "Variant & AUC-PR & MCC & AUC-PR & MCC & AUC-PR & MCC \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

# Single-split runs + literature-style baselines ---------------------------------
lines = []
for ds in CORE:
    for suffix, label in [("", "ARS-Stack (adaptive)"), ("_smote", "Our ensemble + global SMOTE")]:
        r = json.loads((PIPE / f"{ds}{suffix}_results.json").read_text())["metrics"]
        lines.append(f"{tex(ds)} & {label} & " + " & ".join(f"{r[m]:.3f}" for m in METRICS) + " \\\\")
    bfile = ROOT / "reports_raw" / f"baseline_results_{ds}.csv"
    if bfile.exists():
        b = pd.read_csv(bfile)
        for _, r in b[b.source == "this project"].iterrows():
            lines.append(f"{tex(ds)} & {r.model.replace('Simple Stacking (RF+XGB, LR meta)', 'LR stacking (RF+XGB)').replace('RandomForest', 'Random Forest')} & " + " & ".join(f"{r[m]:.3f}" for m in METRICS) + " \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
write("tab_single", "\n".join([
    "\\begin{table}[t]\\centering\\small",
    "\\caption{Single fixed stratified split (20\% test; 10\% of the remainder for validation; seed 42): ARS-Stack, the same ensemble with global SMOTE, and reproduced SMOTE-based baselines (RF, XGBoost, two-learner LR stacking). One split only; see Table~\\ref{tab:core} for the CV estimates.}",
    "\\label{tab:single}",
    "\\begin{tabular}{llrrrr}", "\\toprule", "Dataset & Model & F1 & AUC-PR & G-mean & MCC \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

# Full per-dataset AUC-PR and MCC (appendix) ------------------------------------
for m in ["auc_pr", "mcc"]:
    short = {"ars_stack": "ARS", "equal_average": "EqAvg", "meta_stacking": "Meta", "ars_dbscan": "DBSC",
             "plain_smote": "SMOTE", "plain_adasyn": "ADA", "plain_smote_tomek": "S-Tom", "plain_smote_enn": "S-ENN",
             "no_resampling": "None"}
    head = "Dataset & " + " & ".join(short[v] for v in VARIANTS) + " \\\\"
    lines = []
    for ds in info.sort_values("ir").index:
        vals = [mean.loc[(ds, v), m] for v in VARIANTS]
        best = max(vals)
        cells = [f"\\textbf{{{x:.3f}}}" if np.isclose(x, best) else f"{x:.3f}" for x in vals]
        lines.append(f"{tex(ds)} & " + " & ".join(cells) + " \\\\")
    write(f"tab_full_{m}", "\n".join([
        "{\\small\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{longtable}{lrrrrrrrrr}",
        f"\\caption{{Mean {METRICS[m]} over 15 folds for every dataset and variant (best in bold). ARS = ARS-Stack, EqAvg = equal-weight average, Meta = LR meta-stacking, DBSC = ARS with DBSCAN, ADA = ADASYN, S-Tom = SMOTE-Tomek, S-ENN = SMOTE-ENN, None = no resampling.}}\\label{{tab:full_{m}}}\\\\",
        "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead",
        *lines, "\\bottomrule", "\\end{longtable}}"]))

# Resampling behaviour -----------------------------------------------------------
st = json.loads((FIG / "stats.json").read_text())
removed = {}
for v in ["ars_stack", "plain_smote_tomek", "plain_smote_enn"]:
    fr_ = []
    for f in (ABL / "folds").glob(f"*__{v}.json"):
        for x in json.loads(f.read_text())["folds"]:
            r = x["resampling"]
            if v == "ars_stack":
                nmaj = max(r["original_class_counts"].values()) if "original_class_counts" in r else r["final_majority_count"] + r["total_majority_removed"]
                fr_.append(r["total_majority_removed"] / nmaj)
            else:
                o, fc = r["original_class_counts"], r["final_class_counts"]
                maj = max(o, key=o.get)
                fr_.append((o[maj] - fc[maj]) / o[maj])
    removed[v] = np.array(fr_)
st["majority_removed"] = {v: {"median": float(np.median(a)), "mean": float(a.mean()), "p90": float(np.percentile(a, 90)),
                              "max": float(a.max())} for v, a in removed.items()}
(FIG / "stats.json").write_text(json.dumps(st, indent=2))
lines = [f"{NAMES[v]} & {100 * np.median(a):.1f} & {100 * a.mean():.1f} & {100 * np.percentile(a, 90):.1f} & {100 * a.max():.1f} \\\\"
         for v, a in removed.items()]
write("tab_removed", "\n".join([
    "\\begin{table}[t]\\centering\\small",
    "\\caption{Percentage of majority-class training rows removed by the cleaning step, over all $28\\times15$ folds. ARS-Stack cleans only the local majority context of each cluster.}",
    "\\label{tab:removed}",
    "\\begin{tabular}{lrrrr}", "\\toprule", "Variant & Median & Mean & 90th pct. & Max \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

cost = st["creditcard_seconds"]
rel = st["median_relative_cost"]
lines = [f"{NAMES[v]} & {cost['resample'][v]:.1f} & {cost['total'][v]:.1f} & {rel[v]:.2f} \\\\" for v in VARIANTS]
write("tab_cost", "\n".join([
    "\\begin{table}[t]\\centering\\small",
    "\\caption{Computational cost. Columns 2--3: median seconds per fold on credit card (283{,}725 rows; resampling only, and resampling + ensemble fitting). Column 4: median over the 28 datasets of total fold time relative to global SMOTE.}",
    "\\label{tab:cost}",
    "\\begin{tabular}{lrrr}", "\\toprule", "Variant & Resample (s) & Total (s) & Rel.\\ to SMOTE \\\\", "\\midrule",
    *lines, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))
print(json.dumps(st["majority_removed"], indent=1))
