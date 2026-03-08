# Pantau — System Documentation

> AI-powered fraud detection for illegal online gambling (judol) in Indonesian QRIS payments.
> Built for **PIDI — DIGDAYA X Hackathon 2026** (Bank Indonesia).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Architecture Flow](#3-architecture-flow)
4. [Data Pipeline](#4-data-pipeline)
   - 4.1 [Parametric Dataset Generation](#41-parametric-dataset-generation)
   - 4.2 [GAN Augmentation](#42-gan-augmentation)
   - 4.3 [GAN Post-Processing](#43-gan-post-processing)
   - 4.4 [Dataset Quality Validation](#44-dataset-quality-validation)
5. [ML Detection Engine](#5-ml-detection-engine)
   - 5.1 [6-Layer Ensemble Architecture](#51-6-layer-ensemble-architecture)
   - 5.2 [Layer 1: User Behavior](#52-layer-1-user-behavior)
   - 5.3 [Layer 2: Merchant Behavior](#53-layer-2-merchant-behavior)
   - 5.4 [Layer 3: Network Clustering](#54-layer-3-network-clustering)
   - 5.5 [Layer 4: Temporal Pattern](#55-layer-4-temporal-pattern)
   - 5.6 [Layer 5: Velocity Delta](#56-layer-5-velocity-delta)
   - 5.7 [Layer 6: Money Flow](#57-layer-6-money-flow)
6. [Scoring & Risk Classification](#6-scoring--risk-classification)
7. [Training Pipeline](#7-training-pipeline)
   - 7.1 [K-Fold Cross-Validation](#71-k-fold-cross-validation)
   - 7.2 [Hyperparameter Grid](#72-hyperparameter-grid)
   - 7.3 [Dirichlet Weight Optimization](#73-dirichlet-weight-optimization)
   - 7.4 [Training Outputs](#74-training-outputs)
8. [Production Deployment Model](#8-production-deployment-model)
9. [Evaluation Metrics & Targets](#9-evaluation-metrics--targets)
10. [Domain Knowledge](#10-domain-knowledge)
11. [Technology Stack](#11-technology-stack)
12. [Regulatory Basis](#12-regulatory-basis)

---

## 1. Problem Statement

Indonesia faces a massive illegal online gambling (judol) crisis:

| Metric | Value | Source |
|--------|-------|--------|
| Total judol transaction value (2025) | **Rp 286.84 trillion** | PPATK |
| Total transactions | **422.1 million** | PPATK |
| % of suspicious transaction reports (LTKM) from judol | **47.49%** | PPATK |
| People involved | **12.3 million** | PPATK |

**The core challenge:** Judol operators use QRIS merchant accounts as payment gateways.
Deposits are instant, merchants are disposable (can be bought online for ~Rp 500K),
and operators rotate through hundreds of merchant accounts. Manual review and static
blacklists cannot keep up.

**Key insight:** *Operators can change identity, but they cannot change behavior.*
A judol merchant receiving deposits from thousands of users at 2 AM with uniform round
amounts has a fundamentally different behavioral signature than a legitimate merchant.

---

## 2. Solution Overview

**Pantau** is a dual-layer real-time detection platform that analyzes both **user behavior**
and **merchant behavior** simultaneously:

```
                    ┌─────────────────────────┐
                    │   Incoming Transaction   │
                    │   (QRIS Payment)         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Pantau API          │
                    │   (FastAPI Backend)      │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
     ┌────────▼───────┐  ┌──────▼──────┐  ┌────────▼───────┐
     │  User-Side      │  │  Merchant-  │  │  Cross-Entity  │
     │  Analysis       │  │  Side       │  │  Analysis      │
     │                 │  │  Analysis   │  │                │
     │  • User Behavior│  │  • Merchant │  │  • Network     │
     │  • Temporal     │  │    Behavior │  │    Clustering  │
     │    Pattern      │  │  • Velocity │  │  • Money Flow  │
     │                 │  │    Delta    │  │                │
     └────────┬───────┘  └──────┬──────┘  └────────┬───────┘
              │                  │                   │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Weighted Scoring       │
                    │   + Cross-Correlation    │
                    │   Bonus                  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Risk Level Output      │
                    │   Score: 0-100           │
                    │   Normal / Suspicious /  │
                    │   High Risk / Critical   │
                    └─────────────────────────┘
```

---

## 3. Architecture Flow

The complete system flow from data to production:

```
Phase 1: Data Generation
═══════════════════════════════════════════════════════════════
  Real-world patterns          Parametric Generator         GAN Augmentation
  (IBM AML, PaySim,     ──►   generate_dataset.py    ──►   train_gan.py
   Bustabit, PPATK)            500K transactions            CTGAN/TVAE
                                85% normal / 15% judol       300 epochs
                                                                  │
                                                                  ▼
                                                            fix_gan_output.py
                                                            (Fix IDs, amounts)
                                                                  │
                                                                  ▼
                                                            compare_datasets.py
                                                            (Validate quality)

Phase 2: Model Training
═══════════════════════════════════════════════════════════════
  Dataset (500K)    ──►   80/20 Split    ──►   K-Fold CV (5 folds)
                          (stratified)          9 param combos × 5 folds
                                                = 45 retrains
                                                         │
                                                         ▼
                                                50 Dirichlet weights
                                                × 4 thresholds
                                                = 9,000 scoring evals
                                                         │
                                                         ▼
                                                Best combo found
                                                         │
                                                         ▼
                                                Retrain on full 80%
                                                         │
                                                         ▼
                                                Evaluate on locked 20%
                                                         │
                                                         ▼
                                                7 .pkl model files

Phase 3: Production (Real-time Scoring)
═══════════════════════════════════════════════════════════════
  New transaction ──► Load .pkl models ──► 6-layer scoring ──► Risk level
                                                                  │
                                                                  ▼
                                                            Dashboard +
                                                            Alert System
```

---

## 4. Data Pipeline

### 4.1 Parametric Dataset Generation

**Script:** `scripts/generate_dataset.py`

Generates 500,000 synthetic QRIS transactions calibrated to real-world judol patterns
derived from IBM AML, PaySim, Bustabit gambling, and PPATK reports.

**Dataset Configuration:**

| Parameter | Value |
|-----------|-------|
| Total rows | 500,000 |
| Normal (label=0) | 425,000 (85%) |
| Judol (label=1) | 75,000 (15%) |
| Normal users | ~50,000 |
| Judol users | ~2,000 |
| Normal merchants | ~5,000 |
| Judol merchants | ~500 |
| Date range | 3 months |
| Geography | 38 provinces, 514 cities (real Indonesian data) |

**Transaction Schema:**

| Column | Format | Example |
|--------|--------|---------|
| `transaction_id` | TXN + 12 hex | `TXN1a2b3c4d5e6f` |
| `user_id` | USR + 12 hex | `USR1a2b3c4d5e6f` |
| `merchant_id` | 15-char NMID | `A1B2C3D4E5F6G7H` |
| `tx_type` | Always `"QRIS"` | `QRIS` |
| `amount` | Integer Rupiah | `150000` |
| `timestamp` | ISO 8601 | `2024-06-15 22:31:00` |
| `label` | 0 or 1 | `1` |
| `is_round_amount` | Boolean | `True` |
| `city`, `province` | Real names | `KOTA SURABAYA`, `JAWA TIMUR` |

**Calibrated Behavioral Patterns:**

| Behavior | Normal | Judol |
|----------|--------|-------|
| Transaction hours | Business hours (9-18) | Prime-time (20:00-02:00) |
| Round amount rate | ~12% | ~96% |
| Weekend concentration | ~25% | ~42% |
| Repeat merchant rate | ~62% | Higher (fewer merchants) |
| Amount distribution | Wide range | Fixed denominasi: 10K, 50K, 100K, 500K |
| Togel hour alignment | Random | Clusters at 13:00, 16:00, 19:00, 22:00 |
| Gajian day spike | Slight | Heavy (1st, 25th-28th) |

**Command:**
```bash
python3 scripts/generate_dataset.py
# Output: data/generated/parametric/pantau_dataset.csv (~76 MB)
```

---

### 4.2 GAN Augmentation

**Script:** `scripts/train_gan.py`

Uses **CTGAN** (Conditional Tabular GAN) to learn the statistical distribution of the
parametric dataset and generate new synthetic data with richer variation.

**Why GAN?**
- Parametric data follows hand-coded rules — patterns are too clean
- GAN introduces natural noise and learns implicit correlations
- Two different data sources (parametric + GAN) stress-test the ML model
- Demonstrates the pipeline can work with any data source

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | CTGANSynthesizer (SDV library) |
| Epochs | 300 |
| Batch size | 25,000 (tuned for H100 GPU) |
| GPU | NVIDIA H100 80GB HBM3 |
| Training time | ~5 hours |
| Output rows | 500,000 |

**What GAN Learns vs. What It Cannot:**

| ✅ GAN Learns Well | ❌ GAN Struggles With |
|--------------------|-----------------------|
| Amount distributions | Discrete round amounts (10K, 50K) |
| Hour/day-of-week patterns | Late-night concentration (softened) |
| Geographic distributions | ID relationships (user→merchant) |
| Label ratios (approximate) | Column dependencies |
| City/province variety | Same-city rate patterns |

**Command (requires GPU):**
```bash
python3 scripts/train_gan.py --model ctgan --rows 500000 --epochs 300 --batch-size 25000
# Output: data/generated/gan/pantau_gan_ctgan_500k.csv
# Model: models/pantau_ctgan.pkl (~42 MB)
```

---

### 4.3 GAN Post-Processing

**Script:** `scripts/fix_gan_output.py`

CTGAN generates one unique merchant per row (500K merchants instead of ~5,500). The fix
script repairs cardinality and recomputes derived features.

**Fixes Applied:**

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| 500K unique merchants | GAN treats IDs as independent | Reassign from realistic pools (5,000 normal + 500 judol) |
| 416K unique users | Same issue | Reassign from pools (~70K normal + ~11K judol) |
| Round amount drift | GAN decimal corruption | Recompute with 2% tolerance snapping |
| Transaction/device IDs | GAN-generated gibberish | Regenerate with proper format |

**Post-fix Statistics:**

| Metric | Raw GAN | After Fix |
|--------|---------|-----------|
| Unique merchants | 500,000 | 5,500 |
| Unique users | 416,798 | 80,736 |
| Avg tx/merchant | 1.0 | 90.9 |
| Avg tx/user | 1.2 | 6.2 |
| Judol round amount rate | GAN drift | 61.9% |
| Normal round amount rate | GAN drift | 21.4% |

**Command:**
```bash
python3 scripts/fix_gan_output.py
# Input:  data/generated/gan/pantau_gan_ctgan_500k.csv
# Output: data/generated/gan/pantau_gan_ctgan.csv (~72 MB)
```

---

### 4.4 Dataset Quality Validation

**Script:** `scripts/compare_datasets.py`

Compares parametric vs GAN datasets across 7 dimensions using statistical metrics
and KL divergence.

**Comparison Results (Actual):**

| Metric | Parametric | GAN | Status |
|--------|-----------|-----|--------|
| Total rows | 500,000 | 500,000 | ✅ |
| Label ratio (judol) | 15.0% | 21.6% | ⚠️ Acceptable drift |
| Amount mean | Rp 84,179 | Rp 102,103 | ❌ +21.3% drift |
| Late-night rate (judol) | 49.3% | 30.5% | ❌ Pattern softened |
| Round amount (judol) | 96.2% | 61.9% | ❌ Discrete spike lost |
| Weekend rate | 28.0% | 28.0% | ✅ Preserved |
| Hour KL divergence | — | 0.4431 | ❌ Above threshold |
| Province KL divergence | — | 0.0754 | ✅ Good |
| All 38 provinces | ✅ | ✅ | ✅ Preserved |
| All 514 cities | ✅ | ✅ | ✅ Preserved |

**Known GAN Limitations:** CTGAN smooths discrete distributions. The 96% round amount
rate for judol becomes 61.9% because the GAN cannot learn hard spikes. This is a known
limitation of neural-network-based tabular generators.

---

## 5. ML Detection Engine

### 5.1 6-Layer Ensemble Architecture

Pantau uses a **6-layer ensemble** where each layer detects fraud from a different angle.
Layers 1-3 use Isolation Forest (unsupervised ML). Layers 4-6 use rule-based and
statistical methods grounded in domain knowledge.

| Layer | Module | Technique | Granularity | What It Detects |
|-------|--------|-----------|-------------|-----------------|
| 1 | User Behavior | Isolation Forest | Per-user | Anomalous user spending patterns |
| 2 | Merchant Behavior | Isolation Forest | Per-merchant | Anomalous merchant receiving patterns |
| 3 | Network Clustering | PageRank + IF | Per-merchant | Merchant rings sharing user pools |
| 4 | Temporal Pattern | 7 rule-based scores | Per-user | Judol timing signatures |
| 5 | Velocity Delta | Z-score analysis | Per-merchant | Merchants deviating from population norms |
| 6 | Money Flow | Directed graph | Per-merchant | Fan-in concentration and flow anomalies |

**Design Philosophy:**
- **User-side layers (1, 4):** Detect gambling behavior from the bettor's perspective
- **Merchant-side layers (2, 5):** Detect payment gateway behavior from receiver's perspective
- **Cross-entity layers (3, 6):** Detect structural patterns in the transaction network
- **Ensemble:** A single layer can be fooled; 6 layers from different angles are robust

---

### 5.2 Layer 1: User Behavior

**Technique:** Isolation Forest (200 trees) + StandardScaler

**21 Features Engineered Per User:**

| Category | Features |
|----------|----------|
| Frequency | tx_count_1hr, tx_count_24hr, tx_count_7d |
| Amount | mean_amount, std_amount, amount_deviation |
| Temporal | peak_hour, late_night_rate, weekend_rate |
| Merchant diversity | unique_merchants, repeat_merchant_rate |
| Amount type | round_amount_rate |
| Geographic | geo_spread, same_city_rate |
| Profile | profile_age_days |

**How it works:** Aggregate all transactions per user → compute 21 features →
StandardScaler normalization → IsolationForest identifies users whose feature vectors
are "easy to isolate" (anomalous). Judol users have distinct patterns: high late-night
rate, high round amounts, low merchant diversity.

---

### 5.3 Layer 2: Merchant Behavior

**Technique:** Isolation Forest (200 trees) + StandardScaler

**21 Features Engineered Per Merchant:**

| Category | Features |
|----------|----------|
| Velocity | daily_tx_count, weekly_tx_count, monthly_tx_count |
| Amount | mean_amount, std_amount, coefficient_of_variation |
| Senders | unique_senders, repeat_sender_rate |
| Temporal | peak_hour, night_rate, hour_std |
| Geographic | unique_sender_cities, unique_sender_provinces, geo_spread |
| Amount type | round_amount_rate |
| Concentration | top_sender_concentration, weekend_rate |
| Profile | profile_age_days |

**How it works:** Judol merchants receive thousands of small round-amount deposits
from diverse users at unusual hours. Legitimate merchants have regular business
patterns with repeat customers.

---

### 5.4 Layer 3: Network Clustering

**Technique:** Bipartite graph → PageRank + community detection → Isolation Forest

**Graph Construction:**
1. Build **bipartite graph**: User ↔ Merchant (edges weighted by tx count + amount)
2. Project to **merchant graph**: Two merchants are connected if they share users
   (edge weight = number of shared users)
3. Compute graph features per merchant

**11 Features:**

| Feature | What It Captures |
|---------|-----------------|
| Degree centrality | How connected the merchant is |
| PageRank | Importance in the transaction network |
| Community ID | Which cluster the merchant belongs to |
| Community size | How large its cluster is |
| Shared merchant count | How many merchants share its users |
| Total shared users | Total users flowing between connected merchants |
| Clustering coefficient | How tightly connected its neighbors are |
| Hub score | Combination: unique senders × shared connections |
| Ring score | Combination: clustering × shared users |
| Sender dispersion | Geographic spread of senders |

**Why it works:** Judol operators run multiple merchant accounts simultaneously. Their
users (bettors) deposit across these merchants, creating a dense cluster in the graph.
Legitimate merchants share few users with other merchants.

---

### 5.5 Layer 4: Temporal Pattern

**Technique:** Pure rule-based scoring (7 rules, deterministic)

| Rule | Max Score | Signal |
|------|-----------|--------|
| Late-night rate (20:00-02:00) | 25 pts | Judol prime-time |
| Togel hour alignment (13, 16, 19, 22) | 15 pts | Pre-draw deposits |
| Gajian concentration (1st, 25-28th) | 15 pts | Salary day gambling spikes |
| Rapid burst sessions (<60 min gaps) | 20 pts | Continuous play pattern |
| Amount escalation (increasing within session) | 10 pts | Chasing losses behavior |
| Weekend concentration | 8 pts | Weekend gambling pattern |
| Time consistency (low hour_std) | 7 pts | Habitual player, same time daily |

**Total possible score: 100 points per user.**

**Why rule-based?** These patterns come directly from domain expertise and PPATK reports.
ML models may not learn these specific timing rules from data alone, especially when the
GAN softens temporal patterns.

---

### 5.6 Layer 5: Velocity Delta

**Technique:** Cross-merchant population z-scores

**How it works:**
1. Compute per-merchant statistics: total_tx, total_amount, avg_amount, unique_users,
   round_amount_rate, avg_daily_count, max_daily_count, burstiness
2. Compute **z-scores** for each metric vs. the entire merchant population
3. Merchants with extreme z-scores (far from population mean) are flagged

**Risk Formula:**
```
risk = weighted_z_abs_mean × 20
     + (burstiness > 3) × 10
     + (round_amount_rate > 0.5) × 10
```

**Why it works:** A legitimate kedai kopi receives ~10 transactions/day with varied
amounts. A judol merchant receives 200+ transactions/day with uniform round amounts.
The z-score quantifies "how different is this merchant from everyone else?"

---

### 5.7 Layer 6: Money Flow

**Technique:** Directed graph flow analysis

**How it works:**
1. Build directed graph: User → Merchant (edges = transactions)
2. Analyze each merchant's incoming flow characteristics

**Key Features:**

| Feature | Judol Signal |
|---------|-------------|
| Fan-in (unique senders) | High — thousands of users deposit to one merchant |
| Amount uniformity | High — same denomination repeated |
| Rapid inflow rate (<5 min) | High — continuous stream of deposits |
| Same-amount sequences | High — identical amounts back-to-back |
| Geographic dispersion | High — senders from many provinces |
| Top bucket concentration | High — one amount dominates (e.g., 80% are Rp 50K) |

**Why it works:** Money flow analysis captures the fundamental behavior of a payment
gateway — many-to-one fund concentration with uniform characteristics. This is
structurally different from legitimate merchant receiving patterns.

---

## 6. Scoring & Risk Classification

**Module:** `ml/scoring.py`

### Score Combination

Each layer produces a per-entity score (0-100). The scoring engine maps these back to
transactions and combines them:

```
Transaction T:
  ├── user_id = USR_X    → user_score from Layer 1 + Layer 4
  └── merchant_id = MRC_Y → merchant_score from Layer 2 + Layer 3 + Layer 5 + Layer 6

final_score = Σ(layer_score × weight)  [weighted sum, 0-100]

Cross-correlation bonus:
  If ≥3 layers flag the same entity (score ≥ 40): +10 bonus
  → Confirms suspicion from multiple independent signals
```

### Weight Vector

Weights are **not hand-picked**. They are optimized via Dirichlet random search during
training (see Section 7.3). Example optimized weights:

```
user_behavior:      0.08   (8%)
merchant_behavior:  0.27   (27%)
network_cluster:    0.22   (22%)
temporal_pattern:   0.05   (5%)
velocity_delta:     0.21   (21%)
money_flow:         0.17   (17%)
                    ─────
Total:              1.00   (100%)
```

### Risk Levels

| Score Range | Risk Level | Action |
|-------------|-----------|--------|
| 0 – 40 | **Normal** ✅ | No action |
| 40 – 60 | **Suspicious** ⚠️ | Flag for review |
| 60 – 80 | **High Risk** 🚩 | Escalate to compliance |
| 80 – 100 | **Critical** 🔴 | Immediate freeze + report to PPATK |

---

## 7. Training Pipeline

**Module:** `ml/train.py`

### 7.1 K-Fold Cross-Validation

The training pipeline uses **rigorous evaluation methodology** to prevent overfitting
and produce trustworthy metrics:

```
500,000 transactions
        │
        ▼
┌────────────────────────────────┐
│  Stratified 80/20 Split        │
│  (preserves label ratio)       │
├────────────────┬───────────────┤
│  Train Pool    │  Test Set     │
│  400,000 rows  │  100,000 rows │
│                │  🔒 LOCKED    │
│                │  (never used  │
│                │   for tuning) │
└───────┬────────┴───────────────┘
        │
        ▼
┌────────────────────────────────┐
│  5-Fold Cross-Validation       │
│  on Train Pool only            │
│                                │
│  Fold 1: Train 320K, Val 80K  │
│  Fold 2: Train 320K, Val 80K  │
│  Fold 3: Train 320K, Val 80K  │
│  Fold 4: Train 320K, Val 80K  │
│  Fold 5: Train 320K, Val 80K  │
│                                │
│  Average F1 across folds       │
│  = reliable performance        │
│    estimate                    │
└───────┬────────────────────────┘
        │
        ▼
  Best hyperparameters found
        │
        ▼
┌────────────────────────────────┐
│  Retrain on FULL Train Pool    │
│  (400,000 rows)                │
│  with best hyperparameters     │
└───────┬────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  Final Evaluation              │
│  on Locked Test Set 🔒         │
│  (100,000 rows)                │
│                                │
│  → F1, AUC-ROC, PR-AUC,       │
│    Precision, Recall           │
│  → This is the REAL score      │
└────────────────────────────────┘
```

**Why K-Fold?** A single train/test split can be lucky or unlucky. 5-Fold averages
performance across 5 different splits, giving a stable estimate. The locked test set
ensures the final reported metrics are unbiased.

---

### 7.2 Hyperparameter Grid

Training is split into two phases for efficiency:

**Phase 1 — Expensive (requires retraining):**

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Contamination | 0.10, 0.15, 0.20 | IsolationForest anomaly fraction (Layers 1-3) |
| Layer threshold | 30, 40, 50 | Rule-based score cutoff (Layers 4-6) |

3 × 3 = **9 train combos** × 5 folds = **45 retrains**

**Phase 2 — Cheap (scoring only, no retraining):**

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Weight vectors | 50 Dirichlet samples | Layer importance weights |
| Combined threshold | 30, 35, 40, 45 | Final flagging cutoff |

50 × 4 = **200 scoring combos** per retrain

**Total: 45 retrains + 9,000 scoring evaluations**

---

### 7.3 Dirichlet Weight Optimization

Layer weights are **not hand-picked**. We use the **Dirichlet distribution** to generate
random weight vectors:

```python
np.random.dirichlet(np.ones(6))
# Example outputs (always sum to 1.0):
# [0.08, 0.27, 0.22, 0.05, 0.21, 0.17]
# [0.31, 0.11, 0.04, 0.28, 0.15, 0.11]
# [0.02, 0.45, 0.18, 0.12, 0.03, 0.20]
```

50 random weight vectors are generated, each tested with 4 different combined thresholds.
The combination producing the highest average F1 across folds wins.

**Why Dirichlet?** It naturally generates N positive numbers that sum to 1.0 — perfect
for weights. It explores the entire weight space uniformly, unlike hand-picked presets
which might miss the optimal combination.

---

### 7.4 Training Outputs

| Output | Path | Purpose |
|--------|------|---------|
| User Behavior model | `models/{tag}/user_behavior.pkl` | IsolationForest + scaler |
| Merchant Behavior model | `models/{tag}/merchant_behavior.pkl` | IsolationForest + scaler |
| Network Cluster model | `models/{tag}/network_cluster.pkl` | IsolationForest + scaler |
| Temporal Pattern thresholds | `models/{tag}/temporal_pattern.pkl` | Rule thresholds |
| Velocity Delta thresholds | `models/{tag}/velocity_delta.pkl` | Z-score thresholds |
| Money Flow thresholds | `models/{tag}/money_flow.pkl` | Flow thresholds |
| Optimized weights | `models/{tag}/weights.pkl` | 6 weights + combined threshold |
| Scored transactions | `data/scored/{tag}/scored_transactions.csv` | Full test set with scores |
| Training report | `logs/training_report_{tag}.txt` | Human-readable summary |
| Metrics JSON | `logs/training_metrics_{tag}.json` | Machine-readable metrics |

**Command:**
```bash
# Train on parametric data
python3 -m ml.train --input data/generated/parametric/pantau_dataset.csv --tag parametric

# Train on GAN data
python3 -m ml.train --input data/generated/gan/pantau_gan_ctgan.csv --tag gan

# Quick test (10K sample, skip tuning)
python3 -m ml.train --sample 10000 --no-tune
```

---

## 8. Production Deployment Model

### Real-time Scoring

The API loads trained `.pkl` models and scores each incoming transaction in milliseconds:

```
Transaction arrives via API
      │
      ▼
Load models (cached in memory at startup)
      │
      ▼
Layer 1-6 each produce a score (0-100)
      │
      ▼
Weighted combination → final_score (0-100)
      │
      ▼
Return: { score: 82, risk_level: "Critical", layers_flagged: 5 }
```

### Periodic Retraining (Sliding Window)

Models are retrained periodically on a **sliding window** of recent data:

```
Week 1:  [Dec 1 ─────────────── June 1]    ← train on this window
Week 2:  [Dec 8 ─────────────── June 8]    ← window slides forward
Week 3:  [Dec 15 ────────────── June 15]   ← oldest week drops off
```

| Weekly Transactions | Retrain Frequency |
|--------------------|-------------------|
| 100,000+ | Weekly |
| 10,000 – 100,000 | Monthly |
| 1,000 – 10,000 | Quarterly |
| < 1,000 | When significant new data accumulates |

**Process:** Retrain runs in background → produces new `.pkl` files → API hot-swaps
to new models → no downtime. Scoring is always real-time; only learning is batch.

---

## 9. Evaluation Metrics & Targets

All metrics are measured on the **locked 20% test set** (data the model never saw during
training or hyperparameter tuning).

| Metric | Target | Why This Target |
|--------|--------|-----------------|
| **F1 Score** | ≥ 0.70 | Fraud detection baseline (IEEE S&P literature) |
| **AUC-ROC** | ≥ 0.80 | Discriminative ability threshold |
| **PR-AUC** | ≥ 0.65 | Imbalanced classification standard |
| **Precision** | ≥ 0.60 | Minimize false accusations of legitimate merchants |
| **Recall** | ≥ 0.70 | Catch at least 70% of actual judol transactions |

**Metric Definitions:**
- **Precision:** Of all transactions flagged as judol, what % actually are? (avoid false positives)
- **Recall:** Of all actual judol transactions, what % did we catch? (avoid false negatives)
- **F1:** Harmonic mean of precision and recall (balanced measure)
- **AUC-ROC:** How well the continuous score separates normal from judol (1.0 = perfect)
- **PR-AUC:** Area under precision-recall curve (robust under class imbalance)

---

## 10. Domain Knowledge

### Judol Behavioral Signatures

| Pattern | Description | Detection Layer |
|---------|-------------|-----------------|
| **Prime-time deposits** | 20:00 – 02:00 WIB peak | Temporal (Layer 4) |
| **Togel draw alignment** | Deposits cluster at 13:00, 16:00, 19:00, 22:00 | Temporal (Layer 4) |
| **Gajian spikes** | Surge on 1st and 25th-28th of month | Temporal (Layer 4) |
| **Round amounts** | Rp 5K, 10K, 25K, 50K, 100K, 200K, 500K, 1M, 2M | User + Merchant (L1, L2) |
| **Fan-in concentration** | Thousands of users → one merchant | Money Flow (Layer 6) |
| **Merchant rings** | Multiple judol merchants share the same user pool | Network (Layer 3) |
| **Velocity anomaly** | 200+ tx/day vs. normal merchant ~10/day | Velocity (Layer 5) |
| **Amount uniformity** | Same denomination repeated (80% = Rp 50K) | Money Flow (Layer 6) |
| **Geographic dispersion** | Senders from 20+ provinces to one merchant | Money Flow (Layer 6) |

### QRIS & NMID

- **QRIS:** Quick Response Code Indonesian Standard — national QR payment standard by Bank Indonesia
- **NMID:** National Merchant ID, 15-character alphanumeric — unique identifier for every QRIS merchant
- All transactions in Pantau are QRIS-only (the primary judol attack vector)

---

## 11. Technology Stack

| Component | Technology |
|-----------|-----------|
| **Data generation** | Python, Pandas, NumPy |
| **GAN training** | SDV (CTGANSynthesizer), PyTorch, CUDA (H100 GPU) |
| **ML models** | scikit-learn (IsolationForest, StandardScaler) |
| **Graph analysis** | NetworkX (PageRank, community detection, centrality) |
| **Training pipeline** | scikit-learn (StratifiedKFold, train_test_split, metrics) |
| **Weight optimization** | NumPy (Dirichlet distribution) |
| **API** | FastAPI (planned) |
| **Dashboard** | Next.js (planned) |
| **Explainability** | LLM-based natural language explanations (planned) |

---

## 12. Regulatory Basis

Pantau is designed to support compliance with Indonesian financial regulations:

| Regulation | Relevance |
|-----------|-----------|
| **POJK No. 12/2024** | Anti-fraud strategy for financial institutions |
| **PBI No. 2/2024** | Information system security for payment service providers |
| **Keppres No. 21/2024** | Presidential decree establishing Satgas Judol (gambling task force) |
| **UU No. 8/2010** | Anti-money laundering law — judol deposits are proceeds of crime |
| **PPATK Reports** | Suspicious Transaction Reports (LTKM) — 47.49% from judol |

Pantau automates the detection of suspicious transactions that would otherwise require
manual review, enabling financial institutions to comply with reporting obligations
at scale.

---

*Document generated for PIDI — DIGDAYA X Hackathon 2026. Bank Indonesia.*
