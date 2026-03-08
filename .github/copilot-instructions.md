# Pantau — Copilot Instructions

AI fraud detection system for illegal online gambling (judol) in Indonesian QRIS payments.
Built for **PIDI - DIGDAYA X Hackathon 2026** (Bank Indonesia). Deadline: March 27, 2026.

## Rules

0. **FIRST PRIORITY: Read `.claude/task/lessons.md` before every task.** This file contains
   accumulated lessons from past mistakes and user feedback. Load it before starting any work.
   If user feedback or corrections occur, immediately add a new Principle entry with: Rule, Why,
   How to verify, and Origin.
1. **Commit and push after every change**, even small ones. Never batch unrelated changes.
2. **Only gitignore `data/base/`** (Kaggle downloads). All generated data, models, logs, and
   scored outputs must be committed and pushed.
3. **Do not train GAN or generate GAN data on this server.** The server specs are insufficient.
   GAN training runs on Google Colab or a GPU server only.
4. **Be precise about data and methods before changing anything.** Before making a change, ask:
   "Would I, as a senior engineer, accept this? Is this defensible?" Do not introduce arbitrary
   numbers, hand-picked thresholds, or placeholder values without justification.
5. **All standards must be research-backed.** This is a scientific project. Every metric target,
   threshold, and method must reference established benchmarks — not just "other researchers do
   this" but concrete minimum values that define reliability:
   - F1 ≥ 0.70 (fraud detection baseline per IEEE S&P literature)
   - AUC-ROC ≥ 0.80 (discriminative ability threshold)
   - PR-AUC ≥ 0.65 (imbalanced classification standard)
   - Precision ≥ 0.60, Recall ≥ 0.70
   If a layer or method cannot meet these on held-out test data, it needs redesign, not
   threshold tweaking.

## Agent Behaviour

### 1. No Lazy Fixes
- Always find and fix root causes. Never apply temporary workarounds or band-aids.
- When fixing a file, check all other files that import from or depend on the changed code. Trace the full impact.
- Senior developer standards: would a staff engineer approve this change?

### 2. Skills and Lessons — Mandatory Pre-Task Reads
- **Before every task**, read `.claude/task/lessons.md` first. These are hard-won rules from
  past mistakes. Violating any lesson is a blocking issue.
- **Before every message and every to-do item**, scan `.claude/skills/` and load the skill
  files that match the current task. This is NOT optional — skills contain domain-specific
  procedures. Skipping skills leads to incorrect approaches.
  - At the start of each message: identify which skills apply → read their SKILL.md → follow their procedures.
  - At the start of each to-do item: re-evaluate if additional skills are needed.
  - Example: committing → load `git-commit` + `conventional-commit`. Writing PRD → load `prd`.
    Refactoring → load `refactor`. Creating issues → load `github-issues`.
- Lessons define principles and verification steps that override default assumptions.

### 3. Workflow Orchestration
- **Plan Mode**: Enter plan mode for any non-trivial task (3+ steps or architectural decisions).
  Write plan with checkable items. If something goes sideways, STOP and re-plan immediately.
- **Subagent Strategy**: Use subagents for research, exploration, and parallel analysis.
  One task per subagent. Keep main context window clean.
- **Self-Improvement Loop**: After ANY correction from the user, update `.claude/task/lessons.md`
  with the pattern and a rule to prevent recurrence. Review lessons at session start.
- **Verification Before Done**: Never mark a task complete without proving it works.
  Run the code, check for errors, demonstrate correctness.
- **Demand Elegance (Balanced)**: For non-trivial changes, pause and consider if there's
  a more elegant approach. Skip for simple, obvious fixes — don't over-engineer.
- **Autonomous Bug Fixing**: When given a bug report, just fix it. Point at logs/errors,
  then resolve. Zero hand-holding required from the user.

### 4. Task Management
1. Plan with checkable items for non-trivial tasks
2. Mark items complete as you go
3. High-level summary at each step
4. Update `.claude/task/lessons.md` after corrections

### 5. Git Discipline
- **Commit After Every Change**: After every change — even a single-line fix — immediately
  stage, commit, and push. No batching multiple unrelated changes.
- **Verify Push**: After every push, `git fetch origin` and verify `origin/main` matches HEAD.
- **Use Git Skills**: Before committing, follow relevant Git skills in `.claude/skills/`
  (e.g., `git-commit`, `conventional-commit`).
- **Remote**: `origin` → `https://github.com/gitdevaldo/pantau.git`, branch `main`.

### 6. Core Principles
- **Simplicity First**: Make every change as simple as possible. Minimal code impact.
- **No Laziness**: Find root causes. No temporary fixes.
- **Minimal Impact**: Changes touch only what's necessary. Avoid introducing bugs.
- **Full Traceability**: When changing shared code, verify all consumers still work.
- **Think Before Responding**: Re-read user context. Identify all assumptions. Get it right
  the first time. Maximum 1 correction per task.

## Commands

```bash
# Generate parametric dataset (500K transactions)
python3 scripts/generate_dataset.py

# Train GAN on parametric data (requires GPU / Colab)
python3 scripts/train_gan.py --model ctgan --rows 500000 --epochs 300 --batch-size 2000

# Post-process GAN output (fix IDs, recompute round amounts)
python3 scripts/fix_gan_output.py

# Compare parametric vs GAN datasets
python3 scripts/compare_datasets.py

# Train ML pipeline (auto-detects dataset, 70/30 split, auto-tunes weights)
python3 -m ml.train
python3 -m ml.train --input data/generated/parametric/pantau_dataset.csv --tag parametric
python3 -m ml.train --sample 10000 --no-tune   # quick test run

# pip requires this flag on the system Python
pip3 install <package> --break-system-packages
```

No test suite, linter, or CI exists yet.

## Architecture

### Data Pipeline

```
scripts/generate_dataset.py  →  data/generated/parametric/pantau_dataset.csv  (500K rows, 85/15 split)
        ↓
scripts/train_gan.py         →  data/generated/gan/pantau_gan_ctgan_500k.csv  (raw GAN output)
        ↓
scripts/fix_gan_output.py    →  data/generated/gan/pantau_gan_ctgan.csv       (cleaned for ML)
        ↓
ml/train.py                  →  models/{tag}/*.pkl + logs/ + data/scored/{tag}/
```

### 6-Layer ML Ensemble

All layers share the same API: `train(df) → dict` with keys `scores`, `metrics`, `model`/`scaler`.
Layers 1-3 use `IsolationForest + StandardScaler` (save/load model+scaler).
Layers 4-6 are rule/statistical-based (save/load thresholds only).

| Layer | Module | Technique | Granularity |
|-------|--------|-----------|-------------|
| 1 | `ml/models/user_behavior.py` | IsolationForest, 20 features | per-user |
| 2 | `ml/models/merchant_behavior.py` | IsolationForest, 20 features | per-merchant |
| 3 | `ml/models/network_cluster.py` | Fan-in + community detection + IF | per-merchant |
| 4 | `ml/models/temporal_pattern.py` | 7 rule-based scores (judol timing) | per-user |
| 5 | `ml/models/velocity_delta.py` | Cross-merchant z-scores | per-merchant |
| 6 | `ml/models/money_flow.py` | Directed graph fan-in analysis | per-merchant |

### Combined Scoring (`ml/scoring.py`)

Layers produce per-user or per-merchant scores (0-100). `combine_scores()` maps them back
to transactions via `user_id`/`merchant_id` joins, applies weighted sum + cross-correlation
bonus (+10 if ≥3 layers flag same entity), and assigns risk levels:

- 0-40: Normal, 40-60: Suspicious, 60-80: High Risk, 80-100: Critical

`WEIGHTS` dict at module level is mutable — `ml/train.py` auto-tunes it via Dirichlet
random search (200 trials × 5 thresholds) on the held-out test set.

### Training Pipeline (`ml/train.py`)

1. Load dataset → 2. Stratified 70/30 split → 3. Train 6 layers on train set →
4. Auto-tune weights on test set → 5. Final evaluation on test set →
6. Save models to `models/{tag}/`, scores to `data/scored/{tag}/`, report to `logs/`

Evaluation metrics: Precision, Recall, F1, AUC-ROC, PR-AUC (all on held-out test set).

## Key Conventions

### Transaction Data Schema

All transactions are **QRIS-only** (`tx_type = "QRIS"`). No e-wallet types.

| Column | Format | Example |
|--------|--------|---------|
| `user_id` | `USR` + 12 hex chars | `USR1a2b3c4d5e6f` |
| `merchant_id` | 15-char alphanumeric NMID | `A1B2C3D4E5F6G7H` |
| `tx_type` | Always `"QRIS"` | `QRIS` |
| `amount` | Integer Rupiah (no decimals) | `150000` |
| `timestamp` | ISO 8601 datetime | `2024-06-15 22:31:00` |
| `label` | 0 = normal, 1 = judol | `1` |
| `is_round_amount` | Boolean | `True` |
| `city`, `province` | Real Indonesian names | `KOTA SURABAYA`, `JAWA TIMUR` |

### Merchant Pools

The parametric generator pre-creates merchant/user pools (not per-transaction random IDs):
- 5,000 normal + 500 judol merchants (~5,500 total, avg 91 tx/merchant)
- 50,000 normal + 2,000 judol users

### GAN Limitations

CTGAN cannot learn: discrete amount spikes (round amounts), ID relationships, column
dependencies. The fix script (`fix_gan_output.py`) handles these post-hoc:
- Reassigns user/merchant IDs with proper cardinality
- Recomputes `is_round_amount` with 2% tolerance snapping
- Forces `tx_type` to QRIS
- Does NOT touch GAN-learned columns: amounts, timestamps, cities, labels

### Geolocation

Real data in `data/geolocation/`: 38 provinces (`provinsi.csv`), 514 cities
(`kabupaten_kota.csv`). Linked by ID prefix (city `"12.71"` → province `"12"`).

### Dependencies

Core: `pandas`, `numpy`, `scikit-learn` (IsolationForest, StandardScaler, train_test_split, metrics).
Graph: `networkx` (community detection, centrality, fan-in analysis).
GAN: `sdv` (CTGANSynthesizer, TVAESynthesizer) — only needed for `scripts/train_gan.py`.

### File Organization

- Models saved as pickle: `models/{tag}/{layer_name}.pkl`
- Scored CSV: `data/scored/{tag}/scored_transactions.csv`
- Reports: `logs/training_report_{tag}.txt` + `logs/training_metrics_{tag}.json`
- Tags: `"parametric"` or `"gan"` (auto-detected from input path)
- `data/base/` is gitignored (large Kaggle reference datasets)

### Domain Knowledge

- **Judol prime-time**: 20:00-02:00 WIB
- **Togel draw times**: 13:00, 16:00, 19:00, 22:00
- **Gajian (payday) spike**: 1st and 25th-28th of month
- **Round amounts**: Rp 5K, 10K, 25K, 50K, 100K, 200K, 500K, 1M, 2M — strong judol signal
- **NMID**: National Merchant ID, 15-char alphanumeric (Bank Indonesia standard for QRIS)
