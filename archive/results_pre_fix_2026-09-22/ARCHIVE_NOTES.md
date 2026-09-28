# Archived results — before the 2026-09-22 methodology fixes

Everything in this folder was produced by the code as it stood before the
2026-09-22 fixes. It is kept for reference and for the write-up, but it
should NOT be reported as the final ARS-Stack results, because of the
issues below (all fixed in the current code).

| Issue | Effect on these numbers |
|---|---|
| Adaptive resampling only balanced each cluster against ~3x its size of local majority context, so the adaptive training set stayed ~1:200 (creditcard) / ~1:15 (mammography), while plain SMOTE balanced to 1:1. | ARS-Stack vs plain SMOTE compared different *amounts* of oversampling, not adaptive vs fixed *selection*. Explains ARS winning F1/MCC and losing G-mean. |
| Confidence weights scaled the meta-learner's input features; logistic regression cancels that scaling in its coefficients. | "Confidence weighting" vs "unweighted stacking" was close to a no-op (mammography rows are identical); any significance found there is not attributable to the weights. |
| Percentile ("relative") resampler selection with 1–2 clusters is fixed by rank (1 cluster -> always SMOTE-ENN; 2 clusters -> SMOTE-ENN + ADASYN). | Both datasets got 2 clusters, so the per-cluster choice was not data-driven. |
| Fixed 0.5 decision threshold. | F1/MCC/G-mean mostly reflected the resampling ratio. |
| creditcard exact-duplicate rows (~1,081) not removed before splitting. | Possible train/test duplicates, slightly optimistic scores. |
| Notebook 03 pooled splits already scaled with the train-fitted scaler. | Minor scaling leakage into CV folds. |
| Yeast only had a single train/test run. | No significance testing for multiclass. |

Contents:
- `reports/` — the old `reports/` folder (ablation folds, summary, significance, effect sizes, pipeline results, baseline reproduction)
- `models/` — saved model bundles
- `mlflow.db` — MLflow tracking database (open with `mlflow ui --backend-store-uri sqlite:///archive/results_pre_fix_2026-09-22/mlflow.db`)
- `data_processed/` — old adaptive resampling reports + resampled mammography train set
- `README_at_archive_time.md` — README with the old results table
