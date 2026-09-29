"""
Compare the validated-selection pilot variants on the SAME first 5 CV folds.

Usage: python scripts/compare_pilot.py reports/pilot_validated_selection
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PILOT = sys.argv[1] + "/folds"
FULL = "reports/ablation_results/folds"
METRICS = ["auc_pr", "f1", "mcc", "g_mean"]
N = 5


def load(folder):
    out = {}
    for f in glob.glob(f"{folder}/*.json"):
        d = json.load(open(f))
        s = d["settings"]
        out[(s["dataset"], s["variant"])] = d["folds"][:N]
    return out


pilot, full = load(PILOT), load(FULL)
datasets = sorted({d for d, v in pilot if v == "ars_select" and len(pilot[(d, v)]) == N})
print(f"datasets with complete pilot folds: {len(datasets)}\n")

# 1) device noise: CPU (pilot) vs GPU (full run) no_resampling on identical folds
if all((d, "no_resampling") in pilot for d in datasets):
    diffs = [abs(a["metrics"]["auc_pr"] - b["metrics"]["auc_pr"])
             for d in datasets for a, b in zip(pilot[(d, "no_resampling")], full[(d, "no_resampling")])]
    print(f"GPU vs CPU noise (no_resampling, same folds): mean |dAUC-PR| {np.mean(diffs):.4f}, max {np.max(diffs):.4f}\n")

rows = []
for d in datasets:
    for v in ["ars_select", "smote_select", "no_resampling", "ars_stack", "plain_smote"]:
        # same-machine results only: cross-machine noise is as large as the effects
        src = pilot
        if (d, v) in src and len(src[(d, v)]) == N:
            for m in METRICS:
                rows.append({"dataset": d, "variant": v, "metric": m,
                             "mean": np.mean([f["metrics"][m] for f in src[(d, v)]])})
df = pd.DataFrame(rows)

for m in METRICS:
    t = df[df.metric == m].pivot(index="dataset", columns="variant", values="mean").dropna()
    ranks = t.rank(axis=1, ascending=False).mean().sort_values()
    print(f"== {m}: mean over {len(t)} datasets / average rank (1 = best) ==")
    print(pd.DataFrame({"mean": t.mean().round(4), "rank": ranks.round(2)}).sort_values("rank").to_string())
    for base in ["ars_stack", "plain_smote", "no_resampling", "smote_select"]:
        diff = t["ars_select"] - t[base]
        p = wilcoxon(t["ars_select"], t[base]).pvalue if (diff != 0).any() else 1.0
        print(f"  ars_select vs {base:14s}: mean diff {diff.mean():+.4f}  wins/ties/losses "
              f"{(diff > 1e-9).sum()}/{(diff.abs() <= 1e-9).sum()}/{(diff < -1e-9).sum()}  Wilcoxon p={p:.3f}")
    print()

# what did the selector choose?
sel = pd.DataFrame([{"dataset": d, "variant": v, **f["resampling"]["selected"]}
                    for d in datasets for v in ["ars_select", "smote_select"]
                    for f in pilot[(d, v)]])
sel["choice"] = sel.method + "@" + sel.ratio.astype(str)
print("== choices made by the selectors (count over folds) ==")
print(sel.groupby(["variant", "choice"]).size().unstack(0, fill_value=0).to_string())
