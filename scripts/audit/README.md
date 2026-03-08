# Dataset Quality Audit (`scripts/audit/`)

Statistical tests to measure whether a synthetic fraud dataset is realistic or artificially easy.
Run these **after generating a dataset** and **before training** to catch data quality issues early.

## What It Tests

| # | Test | Method | What It Measures | Research Basis |
|---|------|--------|------------------|----------------|
| 1 | Silhouette Score | Cluster separation | How distinct fraud vs normal are in feature space | Rousseeuw (1987), J. Comp. & Appl. Math |
| 2 | Baseline Models | LR + Decision Stump | Can a trivial model solve it? If yes → too easy | Standard ML evaluation practice |
| 3 | Feature Leakage | Mutual Information + IV | Does any single feature perfectly predict the label? | Shannon (1948); Siddiqi (2006) Credit Risk Scorecards |
| 4 | Distribution Overlap | Bhattacharyya Coefficient | Do fraud/normal feature distributions overlap? | Bhattacharyya (1943), Bull. Calcutta Math Soc. |
| 5 | Borderline Ratio | N1 nearest-neighbor metric | What % of points are near the decision boundary? | Ho & Basu (2002), IEEE TPAMI |

## Usage

```bash
# Audit the parametric dataset
python3 scripts/audit/dataset_quality.py --input data/generated/parametric/pantau_dataset.csv

# Audit the GAN dataset
python3 scripts/audit/dataset_quality.py --input data/generated/gan/pantau_gan_ctgan.csv

# Save results to file
python3 scripts/audit/dataset_quality.py --input data/generated/parametric/pantau_dataset.csv --output logs/audit_parametric.txt
```

## Interpreting Results

**Realistic dataset targets:**
- Silhouette: < 0.3 at all levels (transaction, user, merchant)
- Baseline LR F1: 0.40–0.70 at transaction level, < 0.85 at user/merchant
- Stump F1: < 0.70 (no single feature should solve it)
- IV: < 0.50 for all features
- Distribution overlap: > 0.50 for all features
- Borderline ratio: 15–35%

**If combined F1 > 0.95 on the full ML pipeline but baseline LR F1 < 0.70, the model is adding real value. If baseline LR already gets F1 > 0.90, the dataset is too easy and needs more class overlap.**
