# ARS-Stack paper drafts

| Folder | Format | Pages | Contents |
|---|---|---|---|
| `single_column/` | `article`, 10pt, single column | 15 | `main.tex`, compiled `main.pdf`, `sections/`, `figures/`, `tables/`, `references.bib` |
| `ieee/` | `IEEEtran` conference, two columns | 10 | same layout |

Each folder is self-contained and can be uploaded to Overleaf as is (compiles with pdfLaTeX or XeLaTeX).

## How the drafts are built

```bash
bash paper/build.sh
```

1. `src/make_assets.py` regenerates every results table (`tables/*.tex`) and results figure from the
   experiment outputs in `reports/ablation_results/` — no number in the paper is typed by hand.
   The validated-selection pilot is summarised in `src/data/pilot_means.csv` (its fold files are on the
   `pilot-validated-selection` branch).
2. The shared sources in `src/` (sections, figures, tables, bibliography) are copied into both folders.
3. Both are compiled with `tectonic` (falls back to `latexmk`).

Edit text in `src/sections/`, never in the generated folders — they are overwritten on every build.

## Structure of the paper

Abstract · 1 Introduction (gaps, contributions) · 2 Related work and research gaps (Table I: literature
review) · 3 Method (architecture figure, scoring equations, selection rules, budgeted resampling,
confidence-weighted ensemble, Algorithm 1) · 4 Experimental setup (datasets, variants, leakage-free
protocol figure, statistics) · 5 Results (ranks, critical-difference diagrams, Wilcoxon/Holm, Bayesian
equivalence, per-dataset and core-dataset tables, literature baselines) · 6 Component analysis (resampler
usage, SMOTE-Tomek = SMOTE, weighting, when resampling helps, validated-selection pilot) · 7 Discussion
(implications, five evaluation pitfalls, threats to validity) · 8 Future work · 9 Conclusion · 57 references.

## Before submission

- Fill in author names and affiliations (placeholders in `src/main_single.tex` and `src/main_ieee.tex`).
- Complete the 8 bibliography entries marked `VERIFY` in `src/references.bib` — papers from the project
  plan whose authors/volumes were missing there.
- Re-check volume and page numbers of the remaining references against the publishers' records.
- Check the target venue's page limit and template (the IEEE draft uses the generic `conference` class).
