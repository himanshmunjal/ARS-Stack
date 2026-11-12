"""
final_paper/scripts/make_figures.py

Builds every figure and the derived statistics used by the final paper from
the saved repeated-CV results (reports_raw/ablation_results) and the
single-split pipeline results (reports/pipeline_results).

    python3 final_paper/scripts/make_figures.py

Outputs: final_paper/figures/*.png and final_paper/figures/stats.json
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
ABL = ROOT / "reports_raw" / "ablation_results"
PIPE = ROOT / "reports" / "pipeline_results"
OUT = ROOT / "final_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#2b2b2a",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Reference categorical palette (light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY = "#a3a29c"
INK, INK2 = "#0b0b0b", "#52514e"

NAMES = {
    "ars_stack": "ARS-Stack", "equal_average": "Equal-weight avg.", "meta_stacking": "LR meta-stacking",
    "ars_dbscan": "ARS (DBSCAN)", "plain_smote": "SMOTE", "plain_adasyn": "ADASYN",
    "plain_smote_tomek": "SMOTE-Tomek", "plain_smote_enn": "SMOTE-ENN", "no_resampling": "No resampling",
}
METRICS = {"f1": "F1", "auc_pr": "AUC-PR", "g_mean": "G-mean", "mcc": "MCC"}
RES_COLORS = {"smote": BLUE, "smote_tomek": ORANGE, "smote_enn": AQUA, "adasyn": VIOLET, "random_oversample": GREY}
RES_NAMES = {"smote": "SMOTE", "smote_tomek": "SMOTE-Tomek", "smote_enn": "SMOTE-ENN",
             "adasyn": "ADASYN", "random_oversample": "Random oversample"}

# Feature counts of the imbalanced-learn benchmark collection (Ding 2011) and core datasets.
N_FEATURES = {
    "creditcard": 30, "mammography": 6, "yeast_multiclass": 8, "ecoli": 7, "optical_digits": 64,
    "satimage": 36, "pen_digits": 16, "abalone": 10, "sick_euthyroid": 42, "spectrometer": 93,
    "car_eval_34": 21, "us_crime": 100, "libras_move": 90, "thyroid_sick": 52, "solar_flare_m0": 32,
    "oil": 49, "car_eval_4": 21, "wine_quality": 11, "letter_img": 16, "ozone_level": 72,
    "abalone_19": 10, "coil_2000": 85, "yeast_ml8": 103, "isolet": 617, "scene": 294,
    "arrhythmia": 278, "webpage": 300, "protein_homo": 74,
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

long = pd.read_csv(ABL / "summary_long.csv")
ranks = pd.read_csv(ABL / "cross_dataset_ranks.csv")
xtests = pd.read_csv(ABL / "cross_dataset_tests.csv")
sig = pd.read_csv(ABL / "significance_results.csv")
choices = pd.read_csv(ABL / "resampler_choices.csv")

folds = {}
for f in sorted((ABL / "folds").glob("*.json")):
    ds, variant = f.stem.split("__")
    folds[(ds, variant)] = json.loads(f.read_text())["folds"]

datasets = sorted(long.dataset.unique())
mean = long.pivot_table(index=["dataset", "variant"], columns="metric", values="mean")
std = long.pivot_table(index=["dataset", "variant"], columns="metric", values="std")

# Dataset characteristics from the no-resampling folds (inner-train class counts, fold 0).
ds_rows = []
for ds in datasets:
    counts = folds[(ds, "no_resampling")][0]["resampling"]["original_class_counts"]
    c = np.array(sorted(counts.values()))
    n_inner = c.sum()
    ds_rows.append({
        "dataset": ds,
        "n_after_dedup": int(round(n_inner / (0.8 * 0.85))),
        "n_features": N_FEATURES[ds],
        "n_classes": len(c),
        "ir": float(c.max() / c.min()),
        "minority_inner_train": int(c.min()),
    })
ds_info = pd.DataFrame(ds_rows).set_index("dataset")
ds_info.to_csv(OUT / "dataset_info.csv")

stats = {}

# ---------------------------------------------------------------------------
# Fig 1 — System architecture
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, text, fc="#ffffff", ec="#52514e", fs=8, bold=False, lw=1.0, style="round,pad=0.02,rounding_size=0.04"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, color="#52514e", style="-|>", lw=1.1, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def group(ax, x, y, w, h, title, ec, fc, right=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2, ls="--"))
    tx = x + w - 0.12 if right else x + 0.12
    ax.text(tx, y + h - 0.08, title, ha="right" if right else "left", va="top", fontsize=9, fontweight="bold", color=ec)


def elbow(ax, pts, color="#52514e", lw=1.1):
    """Orthogonal connector through pts, arrow head on the last segment."""
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


fig, ax = plt.subplots(figsize=(11, 7.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 7.2); ax.axis("off")

# Layer 1: data
group(ax, 0.1, 5.55, 10.8, 1.5, "(A) Data layer", INK2, "#f6f6f3")
labels = ["Raw data\n(Kaggle, UCI, OpenML,\nimbalanced-learn)", "Ingest\n(registry of\n28 datasets)",
          "Clean + drop\nexact duplicates", "Stratified split\n(outer CV fold →\ninner train / val)", "StandardScaler\n(fit on inner\ntrain only)"]
xs = [0.3, 2.45, 4.6, 6.75, 8.9]
for x, t in zip(xs, labels):
    box(ax, x, 5.72, 1.85, 1.0, t, fs=7.5)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + 1.85, 6.22), (b, 6.22))

# Layer 2: adaptive resampling
group(ax, 0.1, 2.95, 10.8, 2.35, "(B) Adaptive per-cluster resampling (core contribution)", BLUE, "#eef4fc", right=True)
box(ax, 0.3, 3.65, 1.85, 1.05, "Minority clustering\nKMeans (auto-k by\nsilhouette, k∈[2,6])\nor DBSCAN", fs=7.2)
box(ax, 2.45, 3.65, 1.85, 1.05, "Cluster scoring\noverlap o_j (k=10,\nfull train set)\ndensity ratio ρ_j (k=5)", fs=7.2)
box(ax, 4.6, 3.65, 1.85, 1.05, "Rule-based selector\no≥0.7 → SMOTE-ENN\no≥0.4 → SMOTE-Tomek\nρ≤0.75 → ADASYN\nelse → SMOTE", fs=7.0)
box(ax, 6.75, 3.65, 1.85, 1.05, "Budget allocation\nB = β·N_maj − N_min\nsplit ∝ cluster size\n(largest remainder)", fs=7.2)
box(ax, 8.9, 3.65, 1.85, 1.05, "Local resampling\ncluster ∪ 3·|C_j| nearest\nmajority points,\nSMOTE fallback", fs=7.2)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + 1.85, 4.17), (b, 4.17))
box(ax, 4.6, 3.07, 6.15, 0.42, "Merge: kept minority ∪ synthetic rows ∪ (majority − cleaned)  →  balanced train set", fs=7.3, fc="#ffffff")
arrow(ax, (9.82, 3.65), (9.82, 3.49))
elbow(ax, [(9.82, 5.72), (9.82, 5.43), (1.22, 5.43), (1.22, 4.70)])  # scaled inner-train -> clustering
ax.text(1.32, 5.0, "scaled inner-train split", fontsize=7, color=INK2)

# Layer 3: ensemble
group(ax, 0.1, 0.95, 7.0, 1.8, "(C) Confidence-weighted ensemble", ORANGE, "#fdf1ec", right=True)
for i, t in enumerate(["LightGBM\n(300 rounds)", "XGBoost\n(300 rounds)", "Random Forest\n(300 trees)"]):
    box(ax, 0.3, 1.1 + i * 0.5, 1.6, 0.42, t, fs=7)
box(ax, 2.3, 1.15, 1.7, 1.2, "Inner-val\nAUC-PR a_i per learner\nw_i = max(a_i, 0.05)\n/ Σ (normalised)", fs=7)
box(ax, 4.3, 1.15, 1.25, 1.2, "Weighted\naverage\np̂ = Σ w_i p_i", fs=7.5)
box(ax, 5.8, 1.15, 1.15, 1.2, "Threshold τ*\nmaximising\nval-F1", fs=7.5)
for i in range(3):
    arrow(ax, (1.9, 1.31 + i * 0.5), (2.3, 1.75))
arrow(ax, (4.0, 1.75), (4.3, 1.75)); arrow(ax, (5.55, 1.75), (5.8, 1.75))
elbow(ax, [(7.67, 3.07), (7.67, 2.85), (1.1, 2.85), (1.1, 2.52)])
ax.text(1.2, 2.6, "train each learner", fontsize=7, color=INK2)

# Layer 4: evaluation + serving
group(ax, 7.3, 0.95, 3.6, 1.8, "(D) Evaluation & serving", AQUA, "#ebf7f2")
box(ax, 7.45, 1.15, 1.6, 1.2, "Outer test fold\nF1, AUC-PR,\nG-mean, MCC\n+ statistics", fs=7)
box(ax, 9.2, 1.15, 1.55, 1.2, "FastAPI\n/predict\nmodel bundle\n(joblib)", fs=7)
arrow(ax, (6.95, 1.75), (7.45, 1.75)); arrow(ax, (9.05, 1.75), (9.2, 1.75))

# MLOps strip
group(ax, 0.1, 0.05, 10.8, 0.72, "", GREEN, "#eef6ee")
ax.text(0.25, 0.41, "(E) MLOps", fontsize=9, fontweight="bold", color=GREEN, va="center")
for i, t in enumerate(["DVC: data\nversioning", "MLflow:\ntracking", "Prefect: DAG\norchestration", "Docker:\nenvironment",
                       "pytest:\n112 unit tests", "Fingerprints:\ncode/config hash"]):
    box(ax, 1.45 + i * 1.57, 0.14, 1.45, 0.54, t, fs=6.8)
save(fig, "fig1_architecture")

# ---------------------------------------------------------------------------
# Fig 2 — Resampler selection rule
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
box(ax, 0.1, 1.7, 1.7, 0.9, "Cluster C_j\n|C_j| ≥ 6 ?", fs=8)
box(ax, 0.1, 0.2, 1.7, 0.75, "Random\noversampling", fc="#f1f1ee", fs=8)
arrow(ax, (0.95, 1.7), (0.95, 0.95)); ax.text(1.02, 1.3, "no", fontsize=7.5, color=INK2)
box(ax, 2.4, 1.7, 1.7, 0.9, "o_j ≥ 0.70 ?", fs=8.5)
arrow(ax, (1.8, 2.15), (2.4, 2.15)); ax.text(1.95, 2.25, "yes", fontsize=7.5, color=INK2)
box(ax, 2.4, 3.2, 1.7, 0.75, "SMOTE-ENN\n(strong cleaning)", fc="#e7f6ef", ec=AQUA, fs=8)
arrow(ax, (3.25, 2.6), (3.25, 3.2)); ax.text(3.32, 2.85, "yes", fontsize=7.5, color=INK2)
box(ax, 4.7, 1.7, 1.7, 0.9, "o_j ≥ 0.40 ?", fs=8.5)
arrow(ax, (4.1, 2.15), (4.7, 2.15)); ax.text(4.25, 2.25, "no", fontsize=7.5, color=INK2)
box(ax, 4.7, 3.2, 1.7, 0.75, "SMOTE-Tomek\n(boundary cleaning)", fc="#fdeee7", ec=ORANGE, fs=8)
arrow(ax, (5.55, 2.6), (5.55, 3.2)); ax.text(5.62, 2.85, "yes", fontsize=7.5, color=INK2)
box(ax, 7.0, 1.7, 1.7, 0.9, "ρ_j ≤ 0.75 ?", fs=8.5)
arrow(ax, (6.4, 2.15), (7.0, 2.15)); ax.text(6.55, 2.25, "no", fontsize=7.5, color=INK2)
box(ax, 7.0, 3.2, 1.7, 0.75, "ADASYN\n(sparse cluster)", fc="#efedfb", ec=VIOLET, fs=8)
arrow(ax, (7.85, 2.6), (7.85, 3.2)); ax.text(7.92, 2.85, "yes", fontsize=7.5, color=INK2)
box(ax, 7.0, 0.2, 1.7, 0.75, "SMOTE\n(default)", fc="#eaf2fb", ec=BLUE, fs=8)
arrow(ax, (7.85, 1.7), (7.85, 0.95)); ax.text(7.92, 1.3, "no", fontsize=7.5, color=INK2)
ax.text(9.0, 2.15, "o_j: mean fraction of\nmajority points among\nk=10 nearest neighbours\n\nρ_j: cluster density /\nclass density (k=5)",
        fontsize=7.2, color=INK2, va="center")
save(fig, "fig2_selection_rule")

# ---------------------------------------------------------------------------
# Fig 3 — Leakage-free evaluation protocol
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.6, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
ax.text(0.05, 3.35, "Dataset D (deduplicated) — 5-fold stratified CV × 3 repeats = 15 paired folds per (dataset, variant)", fontsize=8.5, fontweight="bold")
for r in range(5):
    y = 2.55
    x0 = 0.05 + r * 1.98
    for k in range(5):
        ax.add_patch(plt.Rectangle((x0 + k * 0.36, y), 0.34, 0.4, fc=(ORANGE if k == r else "#d9e6f7"), ec="white", lw=1))
    ax.text(x0 + 0.9, y - 0.17, f"fold {r + 1}", ha="center", fontsize=7, color=INK2)
ax.add_patch(plt.Rectangle((0.05, 1.25), 6.2, 0.45, fc="#d9e6f7", ec="white"))
ax.add_patch(plt.Rectangle((6.25, 1.25), 1.1, 0.45, fc="#fbe0c8", ec="white"))
ax.add_patch(plt.Rectangle((7.45, 1.25), 2.4, 0.45, fc=ORANGE, ec="white"))
ax.text(3.15, 1.47, "inner train (85% of outer train): scaler, resampler, base learners", ha="center", va="center", fontsize=7.5)
ax.text(6.8, 1.47, "inner val (15%)", ha="center", va="center", fontsize=7)
ax.text(8.65, 1.47, "outer test fold (untouched)", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
arrow(ax, (1.0, 2.35), (2.0, 1.75))
ax.text(0.05, 0.85, "inner val → confidence weights w_i and threshold τ*;   outer test → F1, AUC-PR, G-mean, MCC", fontsize=8)
ax.text(0.05, 0.45, "Per dataset: Nadeau–Bengio corrected t-test (primary) + Wilcoxon, Holm over all 32 tests;   "
        "across datasets: ranks, Friedman/Iman–Davenport, Wilcoxon + Holm", fontsize=8, color=INK2)
save(fig, "fig3_protocol")

# ---------------------------------------------------------------------------
# Fig 4 — Average ranks (small multiples per metric)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(10, 3.4), sharey=True)
order = ranks[ranks.metric == "auc_pr"].sort_values("average_rank").variant.tolist()
for ax, (m, mname) in zip(axes, METRICS.items()):
    r = ranks[ranks.metric == m].set_index("variant").loc[order]
    y = np.arange(len(order))
    colors = [BLUE if v == "ars_stack" else GREY for v in order]
    ax.hlines(y, 1, r.average_rank, color="#e0dfda", lw=1.5)
    ax.scatter(r.average_rank, y, s=36, c=colors, zorder=3, edgecolor="white", linewidth=1.5)
    for yi, v in zip(y, r.average_rank):
        ax.text(v + 0.12, yi, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_title(mname, fontsize=9.5)
    ax.set_xlim(1, 8.2); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("average rank (1 = best)")
axes[0].set_yticks(np.arange(len(order)), [NAMES[v] for v in order])
save(fig, "fig4_average_ranks")

# ---------------------------------------------------------------------------
# Fig 5 — Wins / ties / losses of ARS-Stack across 28 datasets
# ---------------------------------------------------------------------------

baselines = ["equal_average", "meta_stacking", "ars_dbscan", "plain_smote", "plain_adasyn",
             "plain_smote_tomek", "plain_smote_enn", "no_resampling"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
for ax, m in zip(axes, ["auc_pr", "mcc"]):
    t = xtests[xtests.metric == m].set_index("baseline").loc[baselines]
    y = np.arange(len(baselines))
    ax.barh(y, t.wins, color=BLUE, height=0.6, label="ARS-Stack better", edgecolor="white", linewidth=1)
    ax.barh(y, t.ties, left=t.wins, color="#d6d5cf", height=0.6, label="tie", edgecolor="white", linewidth=1)
    ax.barh(y, t.losses, left=t.wins + t.ties, color=ORANGE, height=0.6, label="baseline better", edgecolor="white", linewidth=1)
    for yi, (_, row) in zip(y, t.iterrows()):
        star = " *" if row.significant else ""
        ax.text(28.4, yi, f"p={row.p_wilcoxon_holm:.3f}{star}", va="center", fontsize=7, color=INK2)
    ax.axvline(14, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 34); ax.set_title(f"{METRICS[m]} (Holm-corrected Wilcoxon p)", fontsize=9.5)
    ax.set_xlabel("number of datasets"); ax.grid(axis="y", visible=False)
axes[0].set_yticks(y, [f"vs {NAMES[b]}" for b in baselines]); axes[0].invert_yaxis()
axes[1].legend(loc="lower right", fontsize=7, frameon=False, bbox_to_anchor=(1.0, -0.42), ncol=3)
save(fig, "fig5_win_tie_loss")

# ---------------------------------------------------------------------------
# Fig 6 — Per-dataset AUC-PR difference vs global SMOTE and vs no resampling
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=False)
deltas = {}
for ax, base in zip(axes, ["plain_smote", "no_resampling"]):
    d = pd.Series({ds: mean.loc[(ds, "ars_stack"), "auc_pr"] - mean.loc[(ds, base), "auc_pr"] for ds in datasets}).sort_values()
    deltas[base] = d
    s = sig[(sig.baseline == base) & (sig.metric == "auc_pr")].set_index("dataset")
    colors = [BLUE if v > 0 else ORANGE for v in d.values]
    ax.barh(np.arange(len(d)), d.values, color=colors, height=0.7, edgecolor="white", linewidth=0.8)
    for i, ds in enumerate(d.index):
        if bool(s.loc[ds, "significant"]):
            ax.text(d[ds] + (0.002 if d[ds] > 0 else -0.002), i, "*", va="center", ha="left" if d[ds] > 0 else "right", fontsize=10)
    ax.set_yticks(np.arange(len(d)), d.index, fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(f"ARS-Stack − {NAMES[base]} (mean AUC-PR)", fontsize=9.5)
    ax.set_xlabel("difference in mean AUC-PR over 15 folds"); ax.grid(axis="y", visible=False)
fig.text(0.5, -0.02, "* significant after Holm correction (Nadeau–Bengio corrected t-test). Blue: ARS-Stack higher; orange: baseline higher.",
         ha="center", fontsize=7.5, color=INK2)
fig.tight_layout()
save(fig, "fig6_per_dataset_delta")

# ---------------------------------------------------------------------------
# Fig 7 — Cluster score space with selection thresholds
# ---------------------------------------------------------------------------

c = choices[choices.variant == "ars_stack"].copy()
c["density_ratio_c"] = c.density_ratio.clip(upper=2.5)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
markers = {"smote": "o", "smote_tomek": "s", "smote_enn": "^", "adasyn": "D", "random_oversample": "x"}
for r in ["smote", "adasyn", "smote_tomek", "smote_enn", "random_oversample"]:
    sub = c[c.resampler_chosen == r]
    if len(sub):
        ax.scatter(sub.overlap_score, sub.density_ratio_c, s=14, marker=markers[r], c=RES_COLORS[r],
                   alpha=0.65, linewidths=0.6 if r == "random_oversample" else 0.3,
                   edgecolors=None if r == "random_oversample" else "white",
                   label=f"{RES_NAMES[r]} (n={len(sub)})")
ax.axvline(0.4, color=INK2, ls="--", lw=0.9); ax.axvline(0.7, color=INK2, ls="--", lw=0.9)
ax.axhline(0.75, xmax=0.4 / 1.02, color=INK2, ls="--", lw=0.9)
ax.text(0.405, 2.42, "o = 0.40", fontsize=7, color=INK2); ax.text(0.705, 2.42, "o = 0.70", fontsize=7, color=INK2)
ax.text(0.01, 0.68, "ρ = 0.75", fontsize=7, color=INK2)
ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 2.6)
ax.set_xlabel("overlap score o_j (fraction of majority neighbours)")
ax.set_ylabel("density ratio ρ_j (clipped at 2.5)")
ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.38, 1.0))
save(fig, "fig7_cluster_space")

# ---------------------------------------------------------------------------
# Fig 8 — Share of minority points per chosen resampler, per dataset
# ---------------------------------------------------------------------------

share = c.groupby(["dataset", "resampler_chosen"]).cluster_size.sum().unstack(fill_value=0)
share = share.div(share.sum(axis=1), axis=0)
cols = [r for r in ["smote", "adasyn", "smote_tomek", "smote_enn", "random_oversample"] if r in share.columns]
share = share[cols].sort_values(cols[0])
fig, ax = plt.subplots(figsize=(10, 3.6))
bottom = np.zeros(len(share))
for r in cols:
    ax.bar(np.arange(len(share)), share[r].values, bottom=bottom, color=RES_COLORS[r], label=RES_NAMES[r],
           width=0.78, edgecolor="white", linewidth=0.8)
    bottom += share[r].values
ax.set_xticks(np.arange(len(share)), share.index, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("share of minority points"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
save(fig, "fig8_resampler_share")

overall_share = (c.groupby("resampler_chosen").cluster_size.sum() / c.cluster_size.sum()).to_dict()
stats["cluster_choice_counts"] = c.resampler_chosen.value_counts().to_dict()
stats["minority_point_share"] = overall_share
stats["fallback_rate"] = float((c.resampler_chosen != c.resampler_used).mean())
stats["n_cluster_records"] = int(len(c))
stats["clusters_per_class_fold"] = float(c.groupby(["dataset", "fold", "class"]).size().mean())
stats["single_resampler_datasets"] = int((share.max(axis=1) > 0.999).sum())

# ---------------------------------------------------------------------------
# Fig 9 — Confidence weights are nearly uniform
# ---------------------------------------------------------------------------

spreads, max_w = [], []
for ds in datasets:
    for f in folds[(ds, "ars_stack")]:
        w = np.array(list(f["weights"].values()))
        spreads.append(w.max() - w.min()); max_w.append(w.max())
spreads = np.array(spreads)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.hist(spreads, bins=30, color=BLUE, edgecolor="white", linewidth=0.8)
ax.axvline(np.median(spreads), color=ORANGE, lw=1.5)
ax.text(np.median(spreads) + 0.004, ax.get_ylim()[1] * 0.9, f"median = {np.median(spreads):.3f}", fontsize=7.5, color=INK)
ax.set_xlabel("max − min confidence weight across the three base learners (per fold)")
ax.set_ylabel("folds (28 datasets × 15)")
ax.grid(axis="x", visible=False)
save(fig, "fig9_weight_spread")
stats["weight_spread"] = {"median": float(np.median(spreads)), "p90": float(np.percentile(spreads, 90)),
                          "max": float(spreads.max()), "max_weight_max": float(max(max_w))}

# ---------------------------------------------------------------------------
# Fig 10 — Computational cost relative to global SMOTE
# ---------------------------------------------------------------------------

cost = []
for ds in datasets:
    base = np.median([f["seconds"]["resample"] + f["seconds"]["fit"] for f in folds[(ds, "plain_smote")]])
    for v in NAMES:
        t = np.median([f["seconds"]["resample"] + f["seconds"]["fit"] for f in folds[(ds, v)]])
        rs = np.median([f["seconds"]["resample"] for f in folds[(ds, v)]])
        cost.append({"dataset": ds, "variant": v, "total": t, "resample": rs, "rel": t / base if base > 0 else np.nan})
cost = pd.DataFrame(cost)
vorder = ["plain_smote", "no_resampling", "plain_adasyn", "plain_smote_tomek", "plain_smote_enn",
          "ars_stack", "equal_average", "ars_dbscan", "meta_stacking"]
fig, ax = plt.subplots(figsize=(8.2, 3.3))
data = [cost[cost.variant == v].rel.dropna().values for v in vorder]
bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=3, markerfacecolor=GREY, markeredgecolor="none"),
                medianprops=dict(color=INK, lw=1.2))
for patch, v in zip(bp["boxes"], vorder):
    patch.set_facecolor(BLUE if v == "ars_stack" else "#dcdbd5"); patch.set_edgecolor("white")
ax.set_yticks(np.arange(1, len(vorder) + 1), [NAMES[v] for v in vorder])
ax.axvline(1, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("median wall time per fold (resample + fit), relative to global SMOTE (log scale)")
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
save(fig, "fig10_cost")
med_rel = cost.groupby("variant").rel.median().to_dict()
stats["median_relative_cost"] = med_rel
stats["creditcard_seconds"] = cost[cost.dataset == "creditcard"].set_index("variant")[["total", "resample"]].round(1).to_dict()

# ---------------------------------------------------------------------------
# Fig 11 — Core datasets, mean ± std
# ---------------------------------------------------------------------------

core = ["creditcard", "mammography", "yeast_multiclass"]
vplot = ["ars_stack", "equal_average", "meta_stacking", "ars_dbscan", "plain_smote", "plain_adasyn",
         "plain_smote_tomek", "plain_smote_enn", "no_resampling"]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
for col, ds in enumerate(core):
    for row, m in enumerate(["auc_pr", "mcc"]):
        ax = axes[row, col]
        mu = [mean.loc[(ds, v), m] for v in vplot]; sd = [std.loc[(ds, v), m] for v in vplot]
        ax.bar(np.arange(len(vplot)), mu, yerr=sd, color=[BLUE if v == "ars_stack" else "#cfcec8" for v in vplot],
               width=0.7, edgecolor="white", linewidth=0.8, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2))
        lo = min(np.array(mu) - np.array(sd)); hi = max(np.array(mu) + np.array(sd))
        ax.set_ylim(max(0, lo - 0.05), min(1, hi + 0.03))
        ax.set_title(f"{ds} — {METRICS[m]}", fontsize=9)
        ax.set_xticks(np.arange(len(vplot)), [NAMES[v] for v in vplot] if row == 1 else [], rotation=60, ha="right", fontsize=7)
        ax.grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "fig11_core_datasets")

# ---------------------------------------------------------------------------
# Fig 12 — Does the gain depend on imbalance ratio or overlap?
# ---------------------------------------------------------------------------

mean_overlap = c.groupby("dataset").apply(lambda d: np.average(d.overlap_score, weights=d.cluster_size), include_groups=False)
dd = pd.DataFrame({"delta": deltas["plain_smote"], "ir": ds_info.ir, "overlap": mean_overlap})
rho_ir, p_ir = spearmanr(dd.ir, dd.delta)
rho_ov, p_ov = spearmanr(dd.overlap, dd.delta)
stats["spearman_delta_vs_ir"] = [float(rho_ir), float(p_ir)]
stats["spearman_delta_vs_overlap"] = [float(rho_ov), float(p_ov)]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), sharey=True)
for ax, col, lab, rho, p, logx in [(axes[0], "ir", "imbalance ratio (log scale)", rho_ir, p_ir, True),
                                   (axes[1], "overlap", "size-weighted mean overlap score of minority clusters", rho_ov, p_ov, False)]:
    ax.scatter(dd[col], dd.delta, s=28, c=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    if logx:
        ax.set_xscale("log")
    for ds in dd.delta.abs().sort_values().index[-3:]:
        ax.annotate(ds, (dd.loc[ds, col], dd.loc[ds, "delta"]), fontsize=6.5, color=INK2,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel(lab); ax.set_title(f"Spearman ρ = {rho:.2f} (p = {p:.2f})", fontsize=9)
axes[0].set_ylabel("ARS-Stack − SMOTE (AUC-PR)")
save(fig, "fig12_delta_vs_properties")

# ---------------------------------------------------------------------------
# Appendix A1 — MLOps / orchestration pipeline
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.8); ax.axis("off")
steps = ["dvc repro\ningest → preprocess\n(dvc.yaml / dvc.lock)", "pytest\n(7 test modules)", "baseline\nreproduction\n(SMOTE + RF/XGB)",
         "src.pipeline\nsingle split,\nsave model bundle", "run_ablation\n28 datasets × 9 variants\n× 15 folds (resumable)",
         "analysis\nranks, Friedman,\nHolm, figures"]
for i, t in enumerate(steps):
    box(ax, 0.1 + i * 1.65, 2.1, 1.5, 1.15, t, fs=7)
    if i:
        arrow(ax, (0.1 + i * 1.65 - 0.15, 2.67), (0.1 + i * 1.65, 2.67))
ax.text(0.1, 3.55, "scripts/run_all.sh  ≡  Prefect flow (dags/ars_stack_dag.py) with retryable tasks", fontsize=8.5, fontweight="bold")
for i, (t, col) in enumerate([("DVC remote\n(raw + processed data)", GREEN), ("MLflow server\n(params, metrics, artefacts)", ORANGE),
                              ("folds/*.json\n+ config/code fingerprint", BLUE), ("Docker image\n(FastAPI + MLflow UI)", AQUA)]):
    box(ax, 0.6 + i * 2.4, 0.3, 2.0, 0.85, t, ec=col, fs=7.2)
arrow(ax, (0.85, 2.1), (1.6, 1.15), ls="--"); arrow(ax, (4.9, 2.1), (3.6, 1.15), ls="--")
arrow(ax, (7.0, 2.1), (6.0, 1.15), ls="--"); arrow(ax, (5.0, 2.1), (8.2, 1.15), ls="--", rad=-0.15)
save(fig, "figA1_mlops")

# ---------------------------------------------------------------------------
# Appendix A2 — Serving sequence diagram
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
actors = ["Client", "FastAPI app", "Model bundle\n(scaler + ensemble)", "Base learners\n(RF, XGB, LGBM)"]
xs_a = [0.9, 3.5, 6.1, 8.7]
for x, a in zip(xs_a, actors):
    box(ax, x - 0.8, 3.45, 1.6, 0.6, a, fs=7.5)
    ax.plot([x, x], [0.2, 3.45], color="#c8c7c1", lw=1, ls="--")
msgs = [(0, 1, 3.1, "POST /predict/{dataset}  {features}"), (1, 2, 2.65, "validate columns, scale"),
        (2, 3, 2.2, "predict_proba(x) for each learner"), (3, 2, 1.75, "p_RF, p_XGB, p_LGBM"),
        (2, 1, 1.3, "p̂ = Σ w_i p_i;  label = [p̂ ≥ τ*]"), (1, 0, 0.85, "{label, probability, threshold}")]
for a, b, y, t in msgs:
    arrow(ax, (xs_a[a], y), (xs_a[b], y), ls="--" if b < a else "-")
    ax.text((xs_a[a] + xs_a[b]) / 2, y + 0.08, t, ha="center", fontsize=7.2)
save(fig, "figA2_serving_sequence")

# ---------------------------------------------------------------------------
# Appendix A3 — Multiclass extension
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 3.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
box(ax, 0.1, 1.15, 1.7, 0.9, "Training set\nK classes", fs=8)
box(ax, 2.2, 1.15, 1.8, 0.9, "majority class c*\nB_c = β·N_c* − N_c\nfor every c ≠ c*", fs=7.2)
for i, cl in enumerate(["class c₁: cluster → score\n→ select → resample", "class c₂: cluster → score\n→ select → resample", "class c_{K−1}: …"]):
    box(ax, 4.5, 0.15 + i * 1.0, 2.4, 0.8, cl, fs=7.2, ec=BLUE)
    arrow(ax, (4.0, 1.6), (4.5, 0.55 + i * 1.0))
    arrow(ax, (6.9, 0.55 + i * 1.0), (7.4, 1.6))
box(ax, 7.4, 1.15, 2.5, 0.9, "merge → balanced set\nensemble: per-learner\nmacro AUC-PR weights", fs=7.2)
arrow(ax, (1.8, 1.6), (2.2, 1.6))
ax.text(4.5, 3.05, "each minority class scored one-vs-rest", fontsize=7.5, color=INK2)
save(fig, "figA3_multiclass")

# ---------------------------------------------------------------------------
# Stats used in the text
# ---------------------------------------------------------------------------

stats["n_tests_per_dataset"] = int(len(sig) / len(datasets))
stats["n_tests_total"] = int(len(sig))
stats["n_significant"] = int(sig.significant.sum())
stats["verdicts"] = sig.verdict.value_counts().to_dict()
stats["friedman"] = xtests.groupby("metric")[["friedman_p", "iman_davenport_p"]].first().to_dict()
stats["mean_auc_pr_over_datasets"] = long[long.metric == "auc_pr"].groupby("variant")["mean"].mean().round(4).to_dict()
stats["mean_mcc_over_datasets"] = long[long.metric == "mcc"].groupby("variant")["mean"].mean().round(4).to_dict()
stats["delta_auc_pr_vs_smote"] = {"median": float(deltas["plain_smote"].median()), "min": float(deltas["plain_smote"].min()),
                                  "max": float(deltas["plain_smote"].max()), "n_pos": int((deltas["plain_smote"] > 0).sum())}
stats["delta_auc_pr_vs_none"] = {"median": float(deltas["no_resampling"].median()), "min": float(deltas["no_resampling"].min()),
                                 "max": float(deltas["no_resampling"].max()), "n_pos": int((deltas["no_resampling"] > 0).sum())}
(OUT / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
print(json.dumps(stats, indent=2, default=float))
print(ds_info.to_string())
