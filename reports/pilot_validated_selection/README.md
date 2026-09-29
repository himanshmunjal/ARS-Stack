# Pilot: validated resampling selection

A development pilot testing whether replacing ARS-Stack's rule-based
resampler choice with **validated selection** can beat the baselines, run
before deciding whether to spend a second full experimental run.

## What was tested

| Variant | Description |
|---|---|
| `ars_select` | Resampling method **and** amount chosen by 3-fold inner CV on the training data (scored by LightGBM out-of-fold AUC-PR). Candidates: no resampling; adaptive per-cluster resampling and global SMOTE at target ratios {0.1, 0.25, 0.5, 1.0}. The adaptive path keeps a single cluster unless the best silhouette is ≥ 0.25. |
| `smote_select` | Same selection, without the adaptive (clustering) option — isolates the value of clustering. |
| `ars_stack`, `plain_smote`, `no_resampling` | The original method and baselines, re-run on the same machine. |

Code: `src/resampling/selection.py`, variants in `src/experiments/run_ablation.py`,
settings under `validated_selection` in `configs/models/resamplers.yaml`.

## Protocol

- 12 of the 14 **development** datasets (the other 14 were held out, untouched):
  libras_move, abalone_19, ecoli, spectrometer, ozone_level, car_eval_34, scene,
  yeast_ml8, sick_euthyroid, abalone, coil_2000, satimage. The two largest
  development datasets (webpage, protein_homo) were skipped for time.
- 5-fold stratified CV (the first repeat of the main study's folds, seed 42),
  full-size models, all variants on one machine (Apple M1, XGBoost on CPU).
- Same-machine comparison matters: re-running `no_resampling` on identical folds
  on a different machine (Windows + CUDA XGBoost) changed AUC-PR by 0.007 on
  average — as large as the effects being measured.

## Result (see `comparison.txt`)

AUC-PR, mean over 12 datasets (average rank, 1 = best):
no_resampling 0.5301 (2.67) · smote_select 0.5307 (2.83) · ars_select 0.5307 (2.92)
· plain_smote 0.5290 (3.08) · ars_stack 0.5212 (3.50).

- `ars_select` vs `ars_stack`: +0.0095 AUC-PR (7/5 wins/losses, Wilcoxon p = 0.18)
  — validated selection repairs the rule-based selector's deficit…
- …but only up to the baselines: vs `no_resampling` +0.0006 (5/7), vs `plain_smote`
  +0.0017 (7/5); F1, MCC and G-mean differences are all < 0.01, no p < 0.4.
- `ars_select` vs `smote_select`: −0.0000 AUC-PR (5/2/5) — the clustering option
  adds nothing once selection is validated.
- The selectors' choices are spread across all candidates, i.e. inner validation
  cannot separate them because they perform alike.

**Conclusion:** validated selection does not make ARS-Stack beat the baselines;
the second full run was not spent on it. Kept as supporting evidence for the
empirical study on the main branch.

Reproduce the comparison: `python scripts/compare_pilot.py reports/pilot_validated_selection`
(the GPU-vs-CPU noise line additionally needs the main study's folds in
`reports/ablation_results/folds`).
