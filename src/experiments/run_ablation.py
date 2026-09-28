"""
src/experiments/run_ablation.py

The Week 8-9 evaluation (project plan, section 9) as one resumable script:
repeated stratified K-fold CV of ARS-Stack against every baseline/ablation
variant on every dataset — the 3 core datasets plus the 25 imbalanced-learn
benchmarks (configs/datasets/benchmarks.yaml) — followed by
    - per dataset: Nadeau-Bengio corrected resampled t-test (primary; accounts
      for overlapping CV training sets) and Wilcoxon signed-rank, each
      Holm-corrected over every (comparison x metric) test for that dataset,
      plus rank-biserial effect sizes;
    - across datasets: average ranks, Friedman / Iman-Davenport test, and
      Holm-corrected Wilcoxon signed-rank tests of ARS-Stack vs each variant
      over the per-dataset means (Demšar 2006; Benavoli et al. 2016).
Replaces the experiment loop in notebook/03_results_analysis.ipynb and the
statistics in notebook/04_effect_sizes.ipynb.

Per fold, nothing is fitted on data outside that fold's training portion:
    outer train/test split (RepeatedStratifiedKFold)
      -> inner stratified train/val split of the outer train portion
      -> StandardScaler fit on the inner train split only
      -> resample the inner train split (variant-specific)
      -> fit the ensemble (confidence weights + threshold from the val split)
      -> metrics on the untouched outer test fold

Variants (see VARIANTS below):
    ars_stack          adaptive per-cluster resampling + confidence-weighted average
    equal_average      same, but equal weights            -> value of confidence weighting
    meta_stacking      same resampling, LR meta-learner    -> weighted average vs classic stacking
    ars_dbscan         DBSCAN instead of KMeans clusters   -> clustering robustness
    plain_smote / plain_adasyn / plain_smote_tomek / plain_smote_enn
                       one global resampler, same ensemble -> value of adaptive selection
    no_resampling      no resampling at all                -> does resampling help at all?

Every resampling variant targets the same class balance
(configs/models/resamplers.yaml -> target_balance_ratio).

Outputs (default reports/ablation_results/, or reports/ablation_results_quick/):
    folds/<dataset>__<variant>.json   per-fold metrics, threshold, weights, resampler choices
    summary_long.csv                  mean/std per dataset x variant x metric
    aggregated_summary.csv            "mean ± std" table
    significance_results.csv/.json    per-dataset tests (corrected t, Wilcoxon, Holm, effect size)
    cross_dataset_ranks.csv           average rank of every variant per metric
    cross_dataset_tests.csv/.json     Friedman + Holm-corrected post-hoc vs ars_stack
    resampler_choices.csv             which resampler the adaptive path picked, per cluster
    ablation_comparison.png           mean ± std per variant on the core datasets
    average_ranks.png                 average ranks across all datasets
    environment.json                  git commit, package versions, machine — for the paper
    summary.md                        all of the above as readable tables

Folds are saved as they finish; re-running the same command resumes where it
stopped. Changing a config file, a setting, or any code that determines fold
results (RESULT_AFFECTING_CODE) invalidates saved folds (you're told to pass
--fresh), so one results folder always comes from one version of the code.

Usage:
    python -m src.experiments.run_ablation --quick                 # smoke test, ~minutes
    python -m src.experiments.run_ablation                         # full run: all 28 datasets, 9 variants
    python -m src.experiments.run_ablation --datasets core         # or: benchmarks, or dataset names
    python -m src.experiments.run_ablation --variants ars_stack plain_smote equal_average
    python -m src.experiments.run_ablation --summarize-only        # rebuild tables/plot from saved folds
"""

import argparse
import copy
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.ingest import CORE_DATASETS, DATASET_GROUPS, DATASET_REGISTRY, expand_dataset_names, load_config
from src.data.preprocess import load_clean_dataset, scaling_columns
from src.ensemble.base_learners import resolve_device
from src.ensemble.stacking import ConfidenceWeightedStackingEnsemble
from src.evaluation.metrics import SUPPORTED_METRICS, compute_metrics
from src.evaluation.significance_test import (
    average_ranks,
    corrected_resampled_ttest,
    friedman_test,
    holm_correction,
    rank_biserial,
    wilcoxon_test,
)
from src.resampling.apply_resampling import resample_training_set

logger = logging.getLogger("run_ablation")

REFERENCE_VARIANT = "ars_stack"

VARIANTS = {
    "ars_stack": {"resampler": "adaptive"},
    "equal_average": {"resampler": "adaptive", "ensemble": {"confidence_weighting": {"enabled": False}}},
    "meta_stacking": {"resampler": "adaptive", "ensemble": {"combiner": "meta_learner"}},
    "ars_dbscan": {"resampler": "adaptive", "clustering": {"method": "dbscan"}},
    "plain_smote": {"resampler": "smote"},
    "plain_adasyn": {"resampler": "adasyn"},
    "plain_smote_tomek": {"resampler": "smote_tomek"},
    "plain_smote_enn": {"resampler": "smote_enn"},
    "no_resampling": {"resampler": "none"},
}

# What each comparison against ars_stack isolates (for summary.md).
COMPARISON_MEANING = {
    "equal_average": "confidence weighting (vs equal-weight average)",
    "meta_stacking": "weighted average vs LR meta-learner stacking",
    "ars_dbscan": "KMeans vs DBSCAN clustering",
    "plain_smote": "adaptive selection vs global SMOTE",
    "plain_adasyn": "adaptive selection vs global ADASYN",
    "plain_smote_tomek": "adaptive selection vs global SMOTE-Tomek",
    "plain_smote_enn": "adaptive selection vs global SMOTE-ENN",
    "no_resampling": "any resampling vs none",
}

QUICK_SETTINGS = {"n_splits": 3, "n_repeats": 1, "n_estimators": 50, "max_rows": 20000}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_model_configs(configs_dir: str, n_estimators: int = None) -> dict:
    cfgs = {
        "clustering": load_config(Path(configs_dir) / "models" / "clustering.yaml")["clustering"],
        "resamplers": load_config(Path(configs_dir) / "models" / "resamplers.yaml")["resamplers"],
        "ensemble": load_config(Path(configs_dir) / "models" / "ensemble.yaml")["ensemble"],
    }
    if n_estimators is not None:
        for params in cfgs["ensemble"]["base_learners"].values():
            params["n_estimators"] = n_estimators
    return cfgs


def _fingerprint(root: str, pattern: str) -> str:
    """Hash of every matching file's path + contents, so saved folds are never
    mixed with folds produced under different configs or code."""
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob(pattern)):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


# Modules whose code determines a fold's numbers. Summary/plot/statistics code
# is deliberately excluded, so improving a table never forces a re-run.
RESULT_AFFECTING_CODE = ["src/data", "src/clustering", "src/resampling", "src/ensemble", "src/evaluation/metrics.py"]


def _code_fingerprint() -> str:
    h = hashlib.sha256()
    for entry in RESULT_AFFECTING_CODE:
        path = Path(entry)
        for f in (sorted(path.rglob("*.py")) if path.is_dir() else [path]):
            h.update(str(f).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _provenance() -> dict:
    """Git commit + environment, recorded next to the results (not part of the
    resume key: committing unchanged code must not invalidate saved folds)."""
    import platform
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError, version

    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    packages = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm"):
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = None
    status = git("status", "--porcelain", "--", "src", "configs")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        gpu = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_uncommitted_changes_in_src_or_configs": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "packages": packages,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "gpu": gpu,
    }


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

def _scale_in_fold(X_tr, X_val, X_test, feature_columns, ds_cfg):
    cols = scaling_columns(ds_cfg, feature_columns)
    if not cols:
        return X_tr, X_val, X_test
    idx = [feature_columns.index(c) for c in cols]
    scaler = StandardScaler().fit(X_tr[:, idx])
    out = []
    for X in (X_tr, X_val, X_test):
        X = X.copy()
        X[:, idx] = scaler.transform(X[:, idx])
        out.append(X)
    return tuple(out)


def _resampling_summary(report: dict) -> dict:
    """Compact, JSON-safe digest of a resampling report for the fold file."""
    summary = {k: report[k] for k in ("mode", "final_class_counts", "original_class_counts") if k in report}
    for k in ("final_minority_count", "final_majority_count", "total_majority_removed"):
        if k in report:
            summary[k] = report[k]
    clusters = []
    if "clusters" in report:
        clusters = [dict(c, **{"class": 1}) for c in report["clusters"]]
    for cls in report.get("classes", []):
        clusters += [dict(c, **{"class": cls["class"]}) for c in cls["clusters"]]
    if clusters:
        summary["clusters"] = [
            {k: c[k] for k in ("class", "cluster_size", "overlap_score", "density_ratio",
                               "resampler_chosen", "resampler_used", "n_synthetic_generated")}
            for c in clusters
        ]
    return summary


def run_fold(X_outer_train, y_outer_train, X_test, y_test, feature_columns, ds_cfg, task_type,
             variant: str, cfgs: dict, val_fraction: float, seed: int) -> dict:
    spec = VARIANTS[variant]
    clustering_cfg = _deep_update(cfgs["clustering"], spec.get("clustering", {}))
    ensemble_cfg = _deep_update(cfgs["ensemble"], spec.get("ensemble", {}))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_outer_train, y_outer_train, test_size=val_fraction, stratify=y_outer_train, random_state=seed,
    )
    X_tr, X_val, X_test = _scale_in_fold(X_tr, X_val, X_test, feature_columns, ds_cfg)

    t0 = time.time()
    X_res, y_res, report = resample_training_set(
        X_tr, y_tr, spec["resampler"], task_type, clustering_cfg, cfgs["resamplers"], positive_label=1,
    )
    t_resample = time.time() - t0

    model = ConfidenceWeightedStackingEnsemble(ensemble_cfg).fit(X_res, y_res, X_val, y_val)
    t_fit = time.time() - t0 - t_resample

    proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, model.predict(X_test), proba if task_type == "multiclass" else proba[:, 1],
                              task_type=task_type)

    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "threshold": float(model.threshold_) if task_type == "binary" else None,
        "weights": {k: float(v) for k, v in model.weights_.items()},
        "n_train_resampled": int(len(y_res)),
        "resampling": _resampling_summary(report),
        "seconds": {"resample": round(t_resample, 1), "fit": round(t_fit, 1)},
    }


# ---------------------------------------------------------------------------
# One dataset x variant, resumable
# ---------------------------------------------------------------------------

def _load_dataset(dataset_name: str, configs_dir: str, max_rows: int, seed: int):
    df, ds_cfg, _ = load_clean_dataset(dataset_name, configs_dir)
    target = ds_cfg["target_column"]
    if max_rows is not None and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, stratify=df[target], random_state=seed)
        logger.info("Subsampled %s to %d rows (quick mode)", dataset_name, len(df))
    feature_columns = [c for c in df.columns if c != target]
    task_type = "multiclass" if ds_cfg.get("type", "binary") == "multiclass" else "binary"
    return df[feature_columns].to_numpy(dtype=float), df[target].to_numpy(), feature_columns, ds_cfg, task_type


def run_experiment(dataset_name: str, variant: str, settings: dict, cfgs: dict, out_dir: Path,
                   configs_dir: str, fresh: bool, data_cache: dict, provenance: dict = None) -> list:
    fold_path = out_dir / "folds" / f"{dataset_name}__{variant}.json"
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    header = {"dataset": dataset_name, "variant": variant, "variant_spec": VARIANTS[variant], **settings}

    folds = []
    if fold_path.exists() and not fresh:
        saved = json.loads(fold_path.read_text())
        if saved["settings"] != header:
            raise RuntimeError(
                f"{fold_path} was produced with different settings/configs "
                f"({saved['settings']} vs {header}). Re-run with --fresh to discard it, "
                f"or point --output-dir somewhere else."
            )
        folds = saved["folds"]

    n_total = settings["n_splits"] * settings["n_repeats"]
    if len(folds) >= n_total:
        logger.info("[%s | %s] all %d folds already done — skipping.", dataset_name, variant, n_total)
        return folds

    if dataset_name not in data_cache:
        data_cache[dataset_name] = _load_dataset(dataset_name, configs_dir, settings["max_rows"], settings["seed"])
    X, y, feature_columns, ds_cfg, task_type = data_cache[dataset_name]

    rskf = RepeatedStratifiedKFold(n_splits=settings["n_splits"], n_repeats=settings["n_repeats"],
                                   random_state=settings["seed"])
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        if fold_idx < len(folds):
            continue
        t0 = time.time()
        result = run_fold(X[train_idx], y[train_idx], X[test_idx], y[test_idx], feature_columns, ds_cfg,
                          task_type, variant, cfgs, settings["val_fraction"], settings["seed"])
        result["fold"] = fold_idx
        folds.append(result)
        fold_path.write_text(json.dumps({"settings": header, "provenance": provenance, "folds": folds}, indent=2))
        m = result["metrics"]
        logger.info(
            "[%s | %s] fold %d/%d  f1=%.4f auc_pr=%.4f g_mean=%.4f mcc=%.4f  (%.0fs)",
            dataset_name, variant, fold_idx + 1, n_total, m["f1"], m["auc_pr"], m["g_mean"], m["mcc"],
            time.time() - t0,
        )
    return folds


# ---------------------------------------------------------------------------
# Summaries, significance, plot
# ---------------------------------------------------------------------------

def _load_all_folds(out_dir: Path) -> tuple:
    """Returns ({(dataset, variant): folds}, {(dataset, variant): settings})."""
    results, settings = {}, {}
    for path in sorted((out_dir / "folds").glob("*__*.json")):
        saved = json.loads(path.read_text())
        s = saved["settings"]
        n_expected = s["n_splits"] * s["n_repeats"]
        if len(saved["folds"]) < n_expected:
            logger.warning("%s has only %d/%d folds — included, but its CV is incomplete.",
                           path.name, len(saved["folds"]), n_expected)
        results[(s["dataset"], s["variant"])] = saved["folds"]
        settings[(s["dataset"], s["variant"])] = s
    return results, settings


def _per_dataset_tests(results: dict, settings: dict, datasets: list, variants: list) -> pd.DataFrame:
    """
    Every variant vs ars_stack on each dataset, on paired folds. Primary test:
    Nadeau-Bengio corrected resampled t-test; secondary: Wilcoxon signed-rank.
    Holm correction is applied per dataset over ALL (comparison x metric)
    tests of that dataset — the conservative family.
    """
    rows = []
    for ds in datasets:
        ref = results.get((ds, REFERENCE_VARIANT))
        if ref is None:
            continue
        n_splits = settings[(ds, REFERENCE_VARIANT)]["n_splits"]
        test_train_ratio = 1.0 / (n_splits - 1)
        ds_rows = []
        for var in variants:
            other = results.get((ds, var))
            if var == REFERENCE_VARIANT or other is None:
                continue
            n = min(len(ref), len(other))
            if n < max(len(ref), len(other)):
                logger.warning("[%s] %s vs %s: unequal fold counts, testing the first %d paired folds.",
                               ds, var, REFERENCE_VARIANT, n)
            for metric in SUPPORTED_METRICS:
                a = [f["metrics"][metric] for f in other[:n]]
                b = [f["metrics"][metric] for f in ref[:n]]
                ct = corrected_resampled_ttest(a, b, test_train_ratio)
                wx = wilcoxon_test(a, b)
                ds_rows.append({
                    "dataset": ds, "baseline": var, "candidate": REFERENCE_VARIANT,
                    "isolates": COMPARISON_MEANING.get(var, ""), "metric": metric,
                    "mean_baseline": float(np.mean(a)), "mean_candidate": float(np.mean(b)),
                    "mean_diff": ct["mean_diff"], "t_corrected": ct["t"], "p_corrected_t": ct["p_value"],
                    "p_wilcoxon": wx["p_value"], "effect_size_r": rank_biserial(a, b), "n_pairs": n,
                })
        if not ds_rows:
            continue
        key = lambda r: f"{r['baseline']}|{r['metric']}"
        holm_t = holm_correction({key(r): r["p_corrected_t"] for r in ds_rows})
        holm_w = holm_correction({key(r): r["p_wilcoxon"] for r in ds_rows})
        for r in ds_rows:
            r["p_corrected_t_holm"] = holm_t[key(r)]["p_holm"]
            r["significant"] = holm_t[key(r)]["significant"]
            r["p_wilcoxon_holm"] = holm_w[key(r)]["p_holm"]
            if r["significant"]:
                r["verdict"] = f"{REFERENCE_VARIANT} better" if r["mean_diff"] > 0 else f"{r['baseline']} better"
            else:
                r["verdict"] = "no significant difference"
        rows += ds_rows
    return pd.DataFrame(rows)


def _cross_dataset(long_df: pd.DataFrame, variants: list) -> tuple:
    """
    Average ranks + Friedman/Iman-Davenport per metric over the datasets where
    every variant finished, and ars_stack vs each variant with a Wilcoxon
    signed-rank test over the per-dataset means (Benavoli et al. 2016),
    Holm-corrected over the comparisons within each metric, plus win/tie/loss counts.
    """
    rank_rows, test_rows = [], []
    for metric in SUPPORTED_METRICS:
        table = (long_df[long_df.metric == metric]
                 .pivot_table(index="dataset", columns="variant", values="mean")
                 .reindex(columns=[v for v in variants]))
        table = table.dropna(axis=0, how="any")
        if len(table) < 2 or REFERENCE_VARIANT not in table.columns:
            continue
        ranks = average_ranks(table)
        for var, r in ranks.items():
            rank_rows.append({"metric": metric, "variant": var, "average_rank": float(r), "n_datasets": len(table)})
        fr = friedman_test(table)
        comps = {}
        for var in table.columns:
            if var == REFERENCE_VARIANT:
                continue
            a, b = table[var].to_numpy(), table[REFERENCE_VARIANT].to_numpy()
            diff = b - a
            comps[var] = {
                "wins": int((diff > 1e-12).sum()), "ties": int((np.abs(diff) <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()), "mean_diff": float(diff.mean()),
                "p_wilcoxon": wilcoxon_test(a, b)["p_value"], "effect_size_r": rank_biserial(a, b),
            }
        holm = holm_correction({v: c["p_wilcoxon"] for v, c in comps.items()})
        for var, c in comps.items():
            test_rows.append({
                "metric": metric, "baseline": var, "candidate": REFERENCE_VARIANT,
                "isolates": COMPARISON_MEANING.get(var, ""), "n_datasets": len(table),
                "friedman_p": fr["p_value"], "iman_davenport_p": fr["p_value_iman_davenport"],
                "rank_candidate": float(ranks[REFERENCE_VARIANT]), "rank_baseline": float(ranks[var]),
                **c, "p_wilcoxon_holm": holm[var]["p_holm"], "significant": holm[var]["significant"],
            })
    return pd.DataFrame(rank_rows), pd.DataFrame(test_rows)


def summarize(out_dir: Path) -> None:
    results, settings = _load_all_folds(out_dir)
    if not results:
        logger.warning("No fold results found in %s — nothing to summarize.", out_dir / "folds")
        return

    datasets = [d for d in DATASET_REGISTRY if any(k[0] == d for k in results)]
    variants = [v for v in VARIANTS if any(k[1] == v for k in results)]

    # --- mean/std tables ---
    rows = []
    for (ds, var), folds in results.items():
        for metric in SUPPORTED_METRICS:
            vals = np.array([f["metrics"][metric] for f in folds])
            rows.append({"dataset": ds, "variant": var, "metric": metric, "mean": vals.mean(),
                         "std": vals.std(ddof=1) if len(vals) > 1 else 0.0, "n_folds": len(vals)})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(out_dir / "summary_long.csv", index=False)

    wide = long_df.assign(cell=lambda d: d["mean"].map("{:.4f}".format) + " ± " + d["std"].map("{:.4f}".format))
    wide = wide.pivot_table(index=["dataset", "variant"], columns="metric", values="cell", aggfunc="first")
    wide = wide.reindex(columns=SUPPORTED_METRICS)
    wide = wide.reindex([(d, v) for d in datasets for v in variants if (d, v) in results])
    wide.to_csv(out_dir / "aggregated_summary.csv")

    # --- statistics ---
    sig_df = _per_dataset_tests(results, settings, datasets, variants)
    sig_df.to_csv(out_dir / "significance_results.csv", index=False)
    (out_dir / "significance_results.json").write_text(sig_df.to_json(orient="records", indent=2))

    ranks_df, cross_df = _cross_dataset(long_df, variants)
    ranks_df.to_csv(out_dir / "cross_dataset_ranks.csv", index=False)
    cross_df.to_csv(out_dir / "cross_dataset_tests.csv", index=False)
    (out_dir / "cross_dataset_tests.json").write_text(cross_df.to_json(orient="records", indent=2))

    # --- which resampler did the adaptive path pick? ---
    choice_rows = []
    for (ds, var), folds in results.items():
        for f in folds:
            for c in f["resampling"].get("clusters", []):
                choice_rows.append({"dataset": ds, "variant": var, "fold": f["fold"], **c})
    choices_df = pd.DataFrame(choice_rows)
    if not choices_df.empty:
        choices_df.to_csv(out_dir / "resampler_choices.csv", index=False)

    core = [d for d in datasets if d in CORE_DATASETS] or datasets[:3]
    _plot(long_df, core, variants, out_dir / "ablation_comparison.png")
    if not ranks_df.empty:
        _plot_ranks(ranks_df, variants, out_dir / "average_ranks.png")
    _write_markdown(out_dir, wide, sig_df, ranks_df, cross_df, choices_df, results)
    logger.info("Summaries written to %s", out_dir)


def _style_axis(ax, labels, surface, ink2, grid):
    ax.set_facecolor(surface)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color=ink2)
    ax.tick_params(axis="x", labelsize=8, colors=ink2)
    ax.grid(axis="x", color=grid, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)


PLOT_COLORS = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
               "accent": "#2a78d6", "context": "#8f8e88"}


def _plot(long_df: pd.DataFrame, datasets: list, variants: list, path: Path) -> None:
    """Small multiples (rows = datasets, cols = metrics): mean ± std per
    variant as a dot + whisker, ARS-Stack emphasized and every other variant
    gray, variants named on the shared y-axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = PLOT_COLORS
    order = list(reversed(variants))  # ars_stack at the top
    fig, axes = plt.subplots(len(datasets), len(SUPPORTED_METRICS),
                             figsize=(4.2 * len(SUPPORTED_METRICS), 0.45 * len(order) * len(datasets) + 1.4),
                             squeeze=False, sharey=True, facecolor=c["surface"])
    for r, ds in enumerate(datasets):
        for col, metric in enumerate(SUPPORTED_METRICS):
            ax = axes[r][col]
            sub = long_df[(long_df.dataset == ds) & (long_df.metric == metric)].set_index("variant")
            for y, var in enumerate(order):
                if var not in sub.index:
                    continue
                mean, std = sub.loc[var, "mean"], sub.loc[var, "std"]
                color = c["accent"] if var == REFERENCE_VARIANT else c["context"]
                ax.errorbar(mean, y, xerr=std, fmt="o", ms=8, color=color, ecolor=color, elinewidth=2, capsize=0)
                if var == REFERENCE_VARIANT:
                    ax.axvline(mean, color=c["accent"], lw=1, alpha=0.35, zorder=0)
                    ax.annotate(f"{mean:.3f}", (mean, y), xytext=(0, 9), textcoords="offset points",
                                ha="center", fontsize=8, color=c["ink"])
            _style_axis(ax, order, c["surface"], c["ink2"], c["grid"])
            ax.set_title(f"{ds} · {metric}", fontsize=10, color=c["ink"], loc="left")
    fig.suptitle("ARS-Stack (blue) vs ablations and baselines — mean ± std over CV folds",
                 fontsize=12, color=c["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=c["surface"])
    plt.close(fig)


def _plot_ranks(ranks_df: pd.DataFrame, variants: list, path: Path) -> None:
    """One panel per metric: average rank across datasets (1 = best),
    ARS-Stack emphasized, best variant at the top."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = PLOT_COLORS
    metrics = [m for m in SUPPORTED_METRICS if m in set(ranks_df.metric)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.45 * len(variants) + 1.6),
                             squeeze=False, facecolor=c["surface"])
    for col, metric in enumerate(metrics):
        ax = axes[0][col]
        sub = ranks_df[ranks_df.metric == metric].sort_values("average_rank", ascending=False)
        labels = list(sub.variant)
        for y, (var, rank) in enumerate(zip(sub.variant, sub.average_rank)):
            color = c["accent"] if var == REFERENCE_VARIANT else c["context"]
            ax.plot([1, rank], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(rank, y, "o", ms=8, color=color)
            ax.annotate(f"{rank:.2f}", (rank, y), xytext=(8, -3), textcoords="offset points",
                        fontsize=8, color=c["ink"])
        _style_axis(ax, labels, c["surface"], c["ink2"], c["grid"])
        ax.set_xlim(1, len(variants) + 0.5)
        n = int(sub.n_datasets.iloc[0])
        ax.set_title(f"{metric} · {n} datasets", fontsize=10, color=c["ink"], loc="left")
        ax.set_xlabel("average rank (1 = best)", fontsize=8, color=c["ink2"])
    fig.suptitle("Average rank across datasets — ARS-Stack in blue", fontsize=12, color=c["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=c["surface"])
    plt.close(fig)


def _fmt(df: pd.DataFrame, floats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, f in floats.items():
        if col in df:
            df[col] = df[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "")
    return df


def _write_markdown(out_dir, wide, sig_df, ranks_df, cross_df, choices_df, results) -> None:
    lines = ["# ARS-Stack ablation results", ""]
    lines += ["Repeated stratified K-fold CV; mean ± std (sample std) over folds. Generated by "
              "`python -m src.experiments.run_ablation`; code version and environment in `environment.json`.", ""]

    if not ranks_df.empty:
        lines += ["## Across datasets", "",
                  "Average rank of each variant (1 = best), computed on per-dataset mean scores over the "
                  "datasets where every variant finished.", ""]
        rank_table = ranks_df.pivot_table(index="variant", columns="metric", values="average_rank")
        rank_table = rank_table.reindex(columns=[m for m in SUPPORTED_METRICS if m in rank_table.columns])
        rank_table = rank_table.sort_values(rank_table.columns[0])
        lines += [rank_table.round(2).reset_index().to_markdown(index=False), ""]
    if not cross_df.empty:
        lines += ["ARS-Stack vs each variant across datasets: Wilcoxon signed-rank on per-dataset means, "
                  "Holm-corrected within each metric. `friedman_p` tests whether any variant differs. "
                  "Wins/ties/losses count datasets where ARS-Stack's mean is higher/equal/lower.", ""]
        show = _fmt(cross_df[["metric", "baseline", "isolates", "n_datasets", "friedman_p", "rank_candidate",
                              "rank_baseline", "wins", "ties", "losses", "mean_diff", "p_wilcoxon_holm",
                              "significant"]],
                    {"friedman_p": "{:.3g}", "rank_candidate": "{:.2f}", "rank_baseline": "{:.2f}",
                     "mean_diff": "{:+.4f}", "p_wilcoxon_holm": "{:.3g}"})
        lines += [show.to_markdown(index=False), ""]

    if not sig_df.empty:
        lines += ["## Per dataset (each variant vs ars_stack)", "",
                  "Primary test: Nadeau-Bengio corrected resampled t-test on paired folds; `p_holm` is "
                  "Holm-corrected over every comparison x metric of that dataset. Wilcoxon p-values (also "
                  "Holm-corrected) are in significance_results.csv. `mean_diff` > 0 means ars_stack is higher.", ""]
        show = _fmt(sig_df[["dataset", "baseline", "metric", "mean_baseline", "mean_candidate", "mean_diff",
                            "p_corrected_t_holm", "effect_size_r", "verdict"]]
                    .rename(columns={"p_corrected_t_holm": "p_holm"}),
                    {"mean_baseline": "{:.4f}", "mean_candidate": "{:.4f}", "mean_diff": "{:+.4f}",
                     "p_holm": "{:.3g}", "effect_size_r": "{:+.2f}"})
        lines += [show.to_markdown(index=False), ""]

    lines += ["## Metrics per dataset", "", wide.reset_index().to_markdown(index=False), ""]
    if not choices_df.empty:
        counts = (choices_df.groupby(["dataset", "variant", "resampler_used"]).size()
                  .unstack(fill_value=0).reset_index())
        lines += ["## Resamplers used by the adaptive path (cluster count over all folds)", "",
                  counts.to_markdown(index=False), ""]
    lines += ["## Fold counts", ""]
    lines += [f"- {ds} / {var}: {len(folds)} folds" for (ds, var), folds in sorted(results.items())]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Repeated-CV ablation study + significance tests for ARS-Stack.")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=list(DATASET_GROUPS.keys()) + list(DATASET_REGISTRY.keys()),
                        help="Dataset names and/or groups: core, benchmarks, all (default).")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), choices=list(VARIANTS.keys()))
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--output-dir", default=None,
                        help="Default: reports/ablation_results (reports/ablation_results_quick with --quick).")
    parser.add_argument("--n-splits", type=int, default=None, help="Default: configs/base.yaml cross_validation.n_splits")
    parser.add_argument("--n-repeats", type=int, default=None, help="Default: configs/base.yaml cross_validation.repeats")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Share of each outer training fold held out for confidence weights + threshold.")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 3 folds x 1 repeat, 50 trees, datasets capped at 20k rows.")
    parser.add_argument("--fresh", action="store_true", help="Discard previously saved folds instead of resuming.")
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild tables/plot from saved folds.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    # The per-cluster/per-learner INFO logs from the underlying modules are
    # far too chatty across hundreds of fits; keep this script's own progress.
    for name in ("clustering", "density_scoring", "resampler_selector", "apply_resampling",
                 "base_learners", "stacking", "preprocess", "ingest"):
        logging.getLogger(name).setLevel(logging.WARNING)

    out_dir = Path(args.output_dir or ("reports/ablation_results_quick" if args.quick else "reports/ablation_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        base_cfg = load_config(Path(args.configs_dir) / "base.yaml")
        cv_cfg = base_cfg["cross_validation"]
        settings = {
            "n_splits": args.n_splits or (QUICK_SETTINGS["n_splits"] if args.quick else cv_cfg["n_splits"]),
            "n_repeats": args.n_repeats or (QUICK_SETTINGS["n_repeats"] if args.quick else cv_cfg["repeats"]),
            "val_fraction": args.val_fraction,
            "seed": base_cfg["project"]["seed"],
            "quick": args.quick,
            "max_rows": QUICK_SETTINGS["max_rows"] if args.quick else None,
            "n_estimators_override": QUICK_SETTINGS["n_estimators"] if args.quick else None,
            "config_fingerprint": _fingerprint(args.configs_dir, "*.yaml"),
            "code_fingerprint": _code_fingerprint(),
        }
        # Part of the resume key: GPU and CPU XGBoost give slightly different
        # numbers, so folds from different devices must never be mixed.
        settings["xgboost_device"] = resolve_device(
            load_config(Path(args.configs_dir) / "models" / "ensemble.yaml")["ensemble"].get("device", "cpu")
        )
        logger.info("XGBoost device: %s (Random Forest, LightGBM and resampling run on CPU)", settings["xgboost_device"])
        cfgs = load_model_configs(args.configs_dir, settings["n_estimators_override"])

        provenance = _provenance()
        if provenance["git_uncommitted_changes_in_src_or_configs"]:
            logger.warning("src/ or configs/ has uncommitted changes — commit before a run you'll publish, "
                           "so environment.json points at the exact code.")
        (out_dir / "environment.json").write_text(json.dumps({**provenance, "settings": settings}, indent=2))

        started = time.time()
        data_cache = {}
        datasets = expand_dataset_names(args.datasets)
        for i, ds in enumerate(datasets, 1):
            for var in args.variants:
                logger.info("=== [%d/%d] %s | %s ===", i, len(datasets), ds, var)
                run_experiment(ds, var, settings, cfgs, out_dir, args.configs_dir, args.fresh, data_cache, provenance)
            data_cache.pop(ds, None)
        logger.info("All experiments finished in %.1f min.", (time.time() - started) / 60)

    summarize(out_dir)


if __name__ == "__main__":
    main()
