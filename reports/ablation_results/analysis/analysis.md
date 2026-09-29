# Further analysis of the ablation study (reference: ars_stack)

Computed from the saved CV folds by `python -m src.experiments.analyze_results`.

## 1. Bayesian analysis (region of practical equivalence ±0.01)

Posterior probabilities that the baseline is practically better, the two are practically equivalent (|difference| < 0.01), or ars_stack is practically better (Benavoli et al., JMLR 2017). A verdict is given when one region reaches 0.95.

### Across datasets (Bayesian signed-rank test on per-dataset means)

| metric   | baseline          |   n_datasets |   mean_diff |   p_baseline_better |   p_equivalent |   p_candidate_better | verdict                |
|:---------|:------------------|-------------:|------------:|--------------------:|---------------:|---------------------:|:-----------------------|
| f1       | ars_dbscan        |           28 |      0.0017 |               0     |          1     |                0     | practically equivalent |
| f1       | equal_average     |           28 |      0.0003 |               0     |          1     |                0     | practically equivalent |
| f1       | meta_stacking     |           28 |      0.0011 |               0     |          1     |                0     | practically equivalent |
| f1       | no_resampling     |           28 |      0.0066 |               0.032 |          0.792 |                0.177 | undecided              |
| f1       | plain_adasyn      |           28 |      0.0088 |               0     |          0.948 |                0.052 | undecided              |
| f1       | plain_smote       |           28 |      0.0011 |               0.001 |          0.996 |                0.003 | practically equivalent |
| f1       | plain_smote_enn   |           28 |      0.0216 |               0     |          0.038 |                0.962 | candidate better       |
| f1       | plain_smote_tomek |           28 |      0.0003 |               0.001 |          0.998 |                0.001 | practically equivalent |
| auc_pr   | ars_dbscan        |           28 |     -0.0011 |               0     |          1     |                0     | practically equivalent |
| auc_pr   | equal_average     |           28 |      0.0001 |               0     |          1     |                0     | practically equivalent |
| auc_pr   | meta_stacking     |           28 |      0.0019 |               0     |          1     |                0     | practically equivalent |
| auc_pr   | no_resampling     |           28 |     -0.0062 |               0.151 |          0.847 |                0.003 | undecided              |
| auc_pr   | plain_adasyn      |           28 |      0.0014 |               0     |          0.999 |                0.001 | practically equivalent |
| auc_pr   | plain_smote       |           28 |     -0.0045 |               0.006 |          0.994 |                0     | practically equivalent |
| auc_pr   | plain_smote_enn   |           28 |      0.0176 |               0.002 |          0.016 |                0.981 | candidate better       |
| auc_pr   | plain_smote_tomek |           28 |     -0.0041 |               0.004 |          0.996 |                0     | practically equivalent |
| g_mean   | ars_dbscan        |           28 |      0.0001 |               0     |          1     |                0     | practically equivalent |
| g_mean   | equal_average     |           28 |     -0.0018 |               0     |          1     |                0     | practically equivalent |
| g_mean   | meta_stacking     |           28 |     -0.0002 |               0.001 |          0.996 |                0.002 | practically equivalent |
| g_mean   | no_resampling     |           28 |      0.0014 |               0.133 |          0.657 |                0.21  | undecided              |
| g_mean   | plain_adasyn      |           28 |      0.005  |               0.008 |          0.835 |                0.158 | undecided              |
| g_mean   | plain_smote       |           28 |     -0.0013 |               0.024 |          0.953 |                0.023 | practically equivalent |
| g_mean   | plain_smote_enn   |           28 |      0.0039 |               0.084 |          0.555 |                0.362 | undecided              |
| g_mean   | plain_smote_tomek |           28 |     -0.0046 |               0.133 |          0.854 |                0.013 | undecided              |
| mcc      | ars_dbscan        |           28 |      0.0024 |               0     |          1     |                0     | practically equivalent |
| mcc      | equal_average     |           28 |      0.0003 |               0     |          1     |                0     | practically equivalent |
| mcc      | meta_stacking     |           28 |      0.0005 |               0     |          1     |                0     | practically equivalent |
| mcc      | no_resampling     |           28 |      0.0025 |               0.101 |          0.82  |                0.079 | undecided              |
| mcc      | plain_adasyn      |           28 |      0.0079 |               0     |          0.981 |                0.019 | practically equivalent |
| mcc      | plain_smote       |           28 |      0.0025 |               0.001 |          0.985 |                0.014 | practically equivalent |
| mcc      | plain_smote_enn   |           28 |      0.0222 |               0     |          0.006 |                0.994 | candidate better       |
| mcc      | plain_smote_tomek |           28 |      0.0012 |               0     |          0.995 |                0.005 | practically equivalent |

### Per dataset (Bayesian correlated t-test over CV folds)

Number of (dataset × metric) comparisons with each verdict; full table in bayes_per_dataset.csv.

| baseline          |   candidate better |   practically equivalent |   baseline better |   undecided |
|:------------------|-------------------:|-------------------------:|------------------:|------------:|
| ars_dbscan        |                  0 |                       42 |                 0 |          70 |
| equal_average     |                  0 |                       76 |                 0 |          36 |
| meta_stacking     |                  3 |                       40 |                 0 |          69 |
| no_resampling     |                  2 |                        9 |                 4 |          97 |
| plain_adasyn      |                  1 |                       10 |                 0 |         101 |
| plain_smote       |                  0 |                       16 |                 1 |          95 |
| plain_smote_enn   |                 12 |                        6 |                 0 |          94 |
| plain_smote_tomek |                  0 |                       16 |                 1 |          95 |

## 2. When does resampling help?

Gain = mean score with resampling − mean score without, per dataset. Spearman correlation between the gain and each dataset characteristic, Holm-corrected over all tests below.

| metric   | resampler   | characteristic   |   spearman_rho |   p_raw |   p_holm | significant   |   n_datasets |
|:---------|:------------|:-----------------|---------------:|--------:|---------:|:--------------|-------------:|
| auc_pr   | plain_smote | minority_count   |         -0.276 |  0.1544 |   1      | False         |           28 |
| auc_pr   | plain_smote | imbalance_ratio  |          0.228 |  0.2427 |   1      | False         |           28 |
| auc_pr   | plain_smote | n_features       |         -0.266 |  0.1706 |   1      | False         |           28 |
| auc_pr   | plain_smote | n_samples        |         -0.115 |  0.5611 |   1      | False         |           28 |
| auc_pr   | ars_stack   | minority_count   |         -0.225 |  0.2497 |   1      | False         |           28 |
| auc_pr   | ars_stack   | imbalance_ratio  |          0.249 |  0.2013 |   1      | False         |           28 |
| auc_pr   | ars_stack   | n_features       |         -0.059 |  0.765  |   1      | False         |           28 |
| auc_pr   | ars_stack   | n_samples        |         -0.114 |  0.5649 |   1      | False         |           28 |
| f1       | plain_smote | minority_count   |         -0.38  |  0.0461 |   0.9228 | False         |           28 |
| f1       | plain_smote | imbalance_ratio  |         -0.092 |  0.6417 |   1      | False         |           28 |
| f1       | plain_smote | n_features       |         -0.058 |  0.7682 |   1      | False         |           28 |
| f1       | plain_smote | n_samples        |         -0.348 |  0.0696 |   1      | False         |           28 |
| f1       | ars_stack   | minority_count   |         -0.479 |  0.0099 |   0.228  | False         |           28 |
| f1       | ars_stack   | imbalance_ratio  |         -0.162 |  0.4101 |   1      | False         |           28 |
| f1       | ars_stack   | n_features       |         -0.041 |  0.8378 |   1      | False         |           28 |
| f1       | ars_stack   | n_samples        |         -0.486 |  0.0088 |   0.2112 | False         |           28 |
| mcc      | plain_smote | minority_count   |         -0.307 |  0.1126 |   1      | False         |           28 |
| mcc      | plain_smote | imbalance_ratio  |          0.012 |  0.9515 |   1      | False         |           28 |
| mcc      | plain_smote | n_features       |         -0.018 |  0.9262 |   1      | False         |           28 |
| mcc      | plain_smote | n_samples        |         -0.248 |  0.2027 |   1      | False         |           28 |
| mcc      | ars_stack   | minority_count   |         -0.384 |  0.0436 |   0.9166 | False         |           28 |
| mcc      | ars_stack   | imbalance_ratio  |         -0.231 |  0.237  |   1      | False         |           28 |
| mcc      | ars_stack   | n_features       |         -0.037 |  0.8508 |   1      | False         |           28 |
| mcc      | ars_stack   | n_samples        |         -0.414 |  0.0286 |   0.6282 | False         |           28 |

AUC-PR gain over no resampling per dataset, ordered by minority-class size:

| dataset          |   ars_stack |   plain_smote |   minority_count |   imbalance_ratio |   n_features |   n_samples |
|:-----------------|------------:|--------------:|-----------------:|------------------:|-------------:|------------:|
| yeast_multiclass |      0.006  |        0.0087 |                5 |           87.6    |            8 |        1453 |
| libras_move      |      0.013  |        0.0141 |               24 |           12.75   |           90 |         330 |
| arrhythmia       |      0.0278 |        0.0178 |               25 |           17.08   |          278 |         452 |
| abalone_19       |      0.0054 |        0.0192 |               32 |          129.531  |           10 |        4177 |
| ecoli            |     -0.0195 |        0.0119 |               35 |            8.6    |            7 |         336 |
| oil              |      0.02   |        0.0417 |               41 |           21.8537 |           49 |         937 |
| spectrometer     |     -0.0484 |       -0.0351 |               45 |           10.8    |           93 |         531 |
| solar_flare_m0   |      0.0075 |       -0.0086 |               58 |            6.1034 |           32 |         412 |
| car_eval_4       |      0.0007 |        0.0003 |               65 |           25.5846 |           21 |        1728 |
| ozone_level      |      0.0351 |        0.0474 |               73 |           33.7397 |           72 |        2536 |
| car_eval_34      |     -0      |       -0.0005 |              134 |           11.8955 |           21 |        1728 |
| us_crime         |     -0.0434 |       -0.0472 |              150 |           12.2933 |          100 |        1994 |
| scene            |     -0.0448 |       -0.0288 |              173 |           12.8728 |          294 |        2400 |
| wine_quality     |     -0.0725 |       -0.0393 |              173 |           21.896  |           11 |        3961 |
| yeast_ml8        |     -0.0005 |       -0.0038 |              178 |           12.5787 |          103 |        2417 |
| thyroid_sick     |     -0.0085 |       -0.0082 |              231 |           15.3247 |           52 |        3771 |
| mammography      |     -0.0165 |       -0.0134 |              254 |           29.9016 |            6 |        7849 |
| sick_euthyroid   |     -0.0017 |       -0.0016 |              285 |            9.9088 |           42 |        3109 |
| abalone          |     -0.0073 |        0.0046 |              391 |            9.6829 |           10 |        4177 |
| creditcard       |      0.0032 |        0.0038 |              473 |          598.844  |           30 |      283726 |
| optical_digits   |     -0.0004 |       -0.0004 |              554 |            9.1444 |           64 |        5620 |
| coil_2000        |     -0.0016 |       -0.0013 |              568 |           13.7535 |           85 |        8380 |
| isolet           |     -0.0047 |       -0.0051 |              600 |           11.995  |          617 |        7797 |
| satimage         |      0.0018 |        0.002  |              626 |            9.2796 |           36 |        6435 |
| letter_img       |     -0.0002 |       -0.0002 |              640 |           28.1688 |           16 |       18668 |
| webpage          |     -0.0413 |       -0.0413 |              981 |           34.4536 |          300 |       34780 |
| pen_digits       |      0.0008 |        0.0006 |             1055 |            9.419  |           16 |       10992 |
| protein_homo     |      0.0169 |        0.0157 |             1251 |          114.882  |           74 |      144968 |

## 3. Critical-difference diagrams

Average ranks and Nemenyi critical difference (α = 0.05); the diagrams are critical_difference_<metric>.png. Methods whose ranks differ by less than CD are not significantly different.

| metric   |   n_datasets |   n_methods |   critical_difference | best              |   best_rank |   ars_stack_rank | worst           |   worst_rank |
|:---------|-------------:|------------:|----------------------:|:------------------|------------:|-----------------:|:----------------|-------------:|
| f1       |           28 |           9 |                  2.27 | ars_dbscan        |        4.2  |             4.79 | plain_smote_enn |         6.75 |
| auc_pr   |           28 |           9 |                  2.27 | no_resampling     |        4.21 |             4.7  | plain_smote_enn |         7.11 |
| g_mean   |           28 |           9 |                  2.27 | plain_smote_tomek |        4.39 |             4.61 | plain_adasyn    |         6.36 |
| mcc      |           28 |           9 |                  2.27 | no_resampling     |        4.29 |             4.82 | plain_smote_enn |         7    |

