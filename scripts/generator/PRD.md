# PRD: Parametric Dataset Generator v2 — Realistic Overlap & Edge Cases

> **Status**: Draft
> **Author**: AI + Human collaboration
> **Date**: 2026-03-08
> **Scope**: `scripts/generate_dataset.py` rewrite
> **Predecessor**: Dataset Generator v1 (current, produces 500K transactions with binary class separation)

---

## 1. Executive Summary

### Problem Statement

The current parametric dataset generator (v1) produces synthetic QRIS fraud data where judol
and normal transactions are **trivially separable**. A single decision stump splitting on
`round_ratio` achieves F1=1.000 at merchant level. The dataset audit (5 research-backed
statistical tests) reveals: merchant silhouette=0.66 (too clean), user stump F1=0.99 (one
feature solves it), `is_round` IV=4.43 (suspiciously predictive), and only 7.2% borderline
data points (real fraud datasets have 20-35%). This makes the trained ML model look perfect
(combined F1=0.9997) but scientifically indefensible — judges at PIDI Hackathon 2026 would
immediately question any model that achieves near-perfect accuracy on fraud detection.

### Proposed Solution

Redesign the generator to produce **realistic class overlap** by modeling real-world edge
cases: mixed merchants, cross-visiting users, 24-hour retail traffic, daytime gambling,
round-amount normal purchases, and diverse behavioral profiles. The signal should emerge
from **multi-feature patterns**, not from any single feature.

### Success Criteria (measured by `scripts/audit/dataset_quality.py`)

| Metric | v1 (current) | v2 Target |
|--------|:---:|:---:|
| Merchant silhouette | 0.66 | < 0.30 |
| User stump F1 (single feature) | 0.99 | < 0.70 |
| Merchant stump F1 (single feature) | 1.00 | < 0.70 |
| `is_round` Information Value | 4.43 | < 0.50 |
| Borderline ratio (N1) | 7.2% | 20-30% |
| Transaction-level LR F1 | 0.72 | 0.55-0.70 |
| Full ML pipeline combined F1 | 0.9997 | 0.75-0.90 |
| Generation time | ~3 min | < 10 min |

---

## 2. User Experience & Functionality

### User Personas

1. **ML Engineer (us)** — needs realistic data to train and validate the 6-layer ensemble.
   Requires overlap between classes so that the multi-layer architecture proves its value
   over a single-feature classifier.
2. **Hackathon Judge** — evaluates whether the model handles real-world complexity. Will ask
   about edge cases, false positives, and how the system handles ambiguous transactions.
3. **Future Data Scientist** — may use this generator to create custom datasets with different
   fraud rates, time periods, or regional distributions.

### User Stories

**US-1**: As an ML engineer, I want normal transactions to include round amounts (pulsa,
parking, fuel) so that `is_round_amount` alone cannot classify fraud.
- **AC**: Normal round amount rate ≥ 20%. `is_round` IV drops below 0.50.

**US-2**: As an ML engineer, I want judol users to also visit normal merchants (groceries,
fuel, food) so that user-merchant affiliation alone cannot classify fraud.
- **AC**: ≥ 40% of judol user transactions are at normal merchants. User stump F1 drops below 0.70.

**US-3**: As an ML engineer, I want hybrid merchants that receive both normal and judol
transactions so that merchant-level classification requires pattern analysis, not binary lookup.
- **AC**: ~10% of merchants are hybrid. Merchant stump F1 drops below 0.70.

**US-4**: As an ML engineer, I want normal nighttime transactions (24h stores, online shopping,
shift workers) so that time-of-day alone cannot classify fraud.
- **AC**: ≥ 25% of normal transactions occur between 20:00-06:00.

**US-5**: As an ML engineer, I want daytime judol transactions (lunch break gambling, afternoon
mobile sessions) so that the model cannot rely on nighttime as a judol signal.
- **AC**: ≥ 30% of judol transactions occur between 08:00-18:00.

**US-6**: As an ML engineer, I want diverse user behavioral profiles (casual, regular, heavy
gamblers; shift workers, drivers, students) so that no single user archetype dominates.
- **AC**: At least 5 distinct normal user profiles and 3 judol user profiles with different
  transaction patterns.

**US-7**: As an ML engineer, I want the borderline ratio to be 20-30% so that the dataset
has realistic decision boundary difficulty.
- **AC**: N1 metric between 0.20 and 0.30 when measured by `scripts/audit/dataset_quality.py`.

### Non-Goals

- **NOT changing the ML pipeline** — `ml/train.py`, layer modules, and scoring remain as-is.
- **NOT changing the output schema** — same CSV columns, same format.
- **NOT modeling concept drift** — all 90 days follow the same distribution (no evolving patterns).
- **NOT modeling adversarial evasion** — judol operators are not actively trying to fool the model (this is a v3 concern).
- **NOT touching GAN pipeline** — GAN retraining is a separate concern.

---

## 3. Real-World Edge Cases Catalog

> Every case below MUST be represented in the generated data. The generator should produce
> transactions that match these scenarios so the ML model encounters them during training.

### 3.1 Normal Transactions That Look Like Judol

#### Case N1: 24-Hour Minimarket at Night
- **Scenario**: A customer buys snacks at Indomaret at 1am.
- **Why it looks like judol**: Nighttime transaction, convenience store QRIS.
- **Distinguishing features**: Non-round amount (Rp 37,500), low frequency, diverse items.
- **Generator requirement**: 24h merchants generate 25-35% of their transactions between 20:00-06:00.

#### Case N2: Driver on Freeway Rest Area
- **Scenario**: Long-distance driver pays at rest-area merchant at midnight, visits the same
  rest-area merchant once a month.
- **Why it looks like judol**: Nighttime, many unique users at merchant, non-repeat pattern.
- **Distinguishing features**: Monthly recurrence, non-round amounts, geographic consistency
  (same route).
- **Generator requirement**: "Driver" user profile — high merchant diversity, cross-city,
  nighttime transactions, but low frequency per merchant and stable long-term pattern.

#### Case N3: Fuel Station Purchases
- **Scenario**: User fills fuel at SPBU — Rp 50,000, Rp 100,000, or Rp 200,000.
- **Why it looks like judol**: Round amounts matching exact judol deposit denominations.
- **Distinguishing features**: Low frequency (1-4x/month), daytime, specific merchant category.
- **Generator requirement**: Fuel merchants always produce round amounts from {50K, 100K, 150K, 200K}.

#### Case N4: Pulsa/Top-Up Purchases
- **Scenario**: User buys phone credit at counter — Rp 25,000, Rp 50,000, Rp 100,000.
- **Why it looks like judol**: Round amounts, can happen anytime.
- **Distinguishing features**: Low frequency (1-2x/month), very specific denominations.
- **Generator requirement**: 5-8% of normal transactions are pulsa with round amounts from
  {10K, 25K, 50K, 100K}.

#### Case N5: Parking/Toll Payments
- **Scenario**: User pays parking Rp 5,000 or toll Rp 15,000 via QRIS.
- **Why it looks like judol**: Round amounts.
- **Distinguishing features**: Very small amounts (< Rp 20,000), high frequency for commuters,
  daytime only.
- **Generator requirement**: "Commuter" profile generates 2-5 small round-amount transactions
  per week at parking/toll merchants.

#### Case N6: Shift Worker Late-Night Meals
- **Scenario**: Factory worker or nurse buys food at 3am after their shift.
- **Why it looks like judol**: Late night, potentially round amounts (Rp 25,000 nasi goreng set).
- **Distinguishing features**: Consistent schedule (same nights every week), same merchants near
  workplace.
- **Generator requirement**: "Shift worker" user profile — regular nighttime transactions at
  specific merchants, 3-5 nights per week.

#### Case N7: Online Midnight Shoppers
- **Scenario**: User makes QRIS payment for Shopee/Tokopedia order at midnight during flash sale.
- **Why it looks like judol**: Nighttime, potentially round amounts, burst during sales events.
- **Distinguishing features**: Irregular timing (only during sales), diverse amounts.
- **Generator requirement**: 3-5% of normal users have occasional midnight purchase bursts
  (1-3 transactions in one night, a few times per month).

#### Case N8: Payday Shopping Spree
- **Scenario**: Office worker spends heavily on 25th-28th — groceries, bills, dining.
- **Why it looks like judol**: Payday concentration, multiple transactions in short period.
- **Distinguishing features**: Diverse merchants (not concentrated), varied amounts, daytime.
- **Generator requirement**: Normal payday boost should be significant (2x+ normal volume on
  25th-28th) so it overlaps with judol payday pattern.

#### Case N9: Event/Concert Burst Merchants
- **Scenario**: A venue merchant gets 500 QRIS payments in 3 hours during a concert.
- **Why it looks like judol**: Sudden velocity spike, many unique users, bursty traffic.
- **Distinguishing features**: One-time burst (not recurring), diverse user base, varied amounts.
- **Generator requirement**: ~2% of normal merchants experience occasional traffic bursts
  (3-5x normal volume for 1-2 days).

#### Case N10: Subscription/Bill Payments
- **Scenario**: User pays Rp 100,000 internet bill on the 5th of every month via QRIS.
- **Why it looks like judol**: Round amount, same merchant, monthly pattern.
- **Distinguishing features**: Exact monthly recurrence, single transaction per cycle.
- **Generator requirement**: 10-15% of normal users have 1-3 recurring monthly payments
  at fixed round amounts.

#### Case N11: Small Warung with Low Traffic
- **Scenario**: A tiny warung gets only 5-15 QRIS transactions per month.
- **Why it looks like judol**: Low transaction count (similar to dormant judol merchant),
  potentially few unique users (neighborhood regulars).
- **Distinguishing features**: Consistent low volume (not dormant-then-spike), daytime only,
  non-round amounts.
- **Generator requirement**: 15-20% of normal merchants are "small warung" with 2-20
  transactions per month.

#### Case N12: Cash-Out Service Merchants
- **Scenario**: A merchant offers QRIS cash withdrawal — many unique users pay Rp 100K-500K.
- **Why it looks like judol**: Round amounts, many unique users, fan-in pattern.
- **Distinguishing features**: Known merchant category, daytime transactions, amounts capped
  by regulation.
- **Generator requirement**: 1-2% of normal merchants are cash-out services with high unique
  users, round amounts, and daytime-only traffic.

### 3.2 Judol Transactions That Look Normal

#### Case J1: Daytime Gambling (Lunch Break)
- **Scenario**: Office worker places bets during lunch break at 12:00-13:00.
- **Why it looks normal**: Daytime transaction, work hours.
- **Distinguishing features**: Still goes to judol merchant, round amount, may be togel-timed.
- **Generator requirement**: 30-40% of judol transactions occur between 08:00-18:00.

#### Case J2: Judol Player Also Buys Groceries
- **Scenario**: A judol user also shops at Alfamart, eats at restaurants, buys fuel.
- **Why it looks normal**: Mixed transaction history with legitimate purchases.
- **Distinguishing features**: Judol transactions are a subset of their overall activity —
  concentrated at specific merchants, with specific amounts/timing patterns.
- **Generator requirement**: 40-60% of a judol user's total transactions should be at normal
  merchants with normal patterns.

#### Case J3: Minimum Deposit Threshold (Rp 25,000)
- **Scenario**: All judol platforms enforce a minimum deposit of Rp 25,000. No gambling
  transaction exists below this amount.
- **Why it matters**: This creates a hard floor that separates judol from small normal
  purchases (parking Rp 5K, snacks Rp 8K, coffee Rp 15K). Normal transactions frequently
  fall below 25K; judol never does.
- **Distinguishing features**: The floor alone is not diagnostic (most normal transactions
  are also ≥ 25K), but the ABSENCE of sub-25K transactions for a user/merchant is a soft signal.
- **Generator requirement**: All judol transactions must have amount ≥ Rp 25,000. Deposits
  are free denomination (any amount ≥ 25K), though users naturally gravitate toward round
  numbers. Normal transactions should frequently include amounts below 25K (parking 5K,
  snacks 8K, coffee 15K) to create natural contrast.

#### Case J4: Smart Operator Using Non-Round Amounts
- **Scenario**: Judol operator accepts deposits of Rp 51,000 or Rp 99,500 instead of round numbers.
- **Why it looks normal**: Non-round amount breaks the round-amount signal.
- **Distinguishing features**: Amounts cluster near round denominations (±2%), timing and
  merchant network still anomalous.
- **Generator requirement**: 10-15% of judol transactions use "near-round" amounts
  (round ± random(500, 2000)).

#### Case J5: Casual Gambler (Low Frequency)
- **Scenario**: User places a Rp 50,000 bet once a week — total 4-5 judol transactions per month.
- **Why it looks normal**: Low frequency, small amounts, doesn't trigger velocity alerts.
- **Distinguishing features**: Still targets known judol merchants, round amounts.
- **Generator requirement**: "Casual" judol profile — 5-15 transactions total over 90 days,
  low amounts (Rp 10K-50K), no escalation.

#### Case J6: Judol Through Compromised/Hybrid Merchant
- **Scenario**: A legitimate warung's QRIS code is also used to process judol deposits.
  The same merchant_id receives both Rp 15,000 nasi goreng payments and Rp 100,000 judol
  deposits.
- **Why it looks normal**: The merchant is a real business with legitimate traffic.
- **Distinguishing features**: Bimodal amount distribution, some transactions at unusual hours,
  mix of regular customers and one-time users.
- **Generator requirement**: Hybrid merchants (see Section 4.2) receive both normal AND judol
  transactions through the same merchant_id.

#### Case J7: Structured Small Deposits (Smurfing)
- **Scenario**: Instead of one Rp 500,000 deposit, the user makes 10 deposits of Rp 50,000
  spread across 3 different merchants in 2 hours.
- **Why it looks normal**: Each individual transaction is small and unremarkable.
- **Distinguishing features**: Rapid succession across multiple merchants, total amount is
  significant, round amounts persist.
- **Generator requirement**: 5-10% of heavy judol users exhibit smurfing — multiple small
  deposits at different merchants within short windows (1-3 hours). Minimum per-transaction
  amount is Rp 25,000 (platform minimum).

### 3.3 Ambiguous/Gray-Zone Scenarios

#### Case G1: Seasonal Business Activation
- **Scenario**: A beach warung is closed for 2 months, then opens for holiday season with
  hundreds of transactions per day.
- **Why it's ambiguous**: Dormant-then-active pattern matches judol merchant lifecycle.
- **Generator requirement**: 3-5% of normal merchants are "seasonal" — inactive for 30-60
  days, then active with high volume for 2-4 weeks.

#### Case G2: New Business Opening
- **Scenario**: A new restaurant opens and gets a surge of transactions in the first 2 weeks.
- **Why it's ambiguous**: Sudden activation with high velocity.
- **Generator requirement**: 2-3% of normal merchants start mid-period with an initial
  burst (2-3x steady-state volume for the first 2 weeks).

#### Case G3: Student Allowance Pattern
- **Scenario**: A university student receives Rp 1,500,000 monthly allowance and spends
  heavily in the first week — small purchases at warung, round-amount pulsa top-ups.
- **Why it's ambiguous**: Monthly spending spike + round amounts + young demographic.
- **Generator requirement**: "Student" profile — spending concentrated in first 5 days after
  1st of month, small amounts, campus-area merchants, some round (pulsa).

#### Case G4: Couple/Family Shared QRIS Account
- **Scenario**: One QRIS account used by husband (daytime work lunches) and wife (evening
  grocery shopping). Mixed timing and merchant patterns.
- **Why it's ambiguous**: Irregular timing pattern, diverse merchants, looks like multiple
  behavioral profiles from one user.
- **Generator requirement**: 3-5% of normal users have "dual pattern" — two distinct
  time-of-day clusters and two distinct merchant clusters from one user_id.

#### Case G5: Cross-City Business Traveler
- **Scenario**: A sales rep visits Jakarta, Surabaya, and Bandung in the same month,
  transacting at different merchants in each city.
- **Why it's ambiguous**: High geographic spread, many unique merchants, looks like a
  distributed judol operation.
- **Generator requirement**: "Traveler" profile — 2-4 cities per month, 3-5 merchants per
  city, normal amounts and timing.

---

## 4. Technical Specifications

### 4.1 Dataset Parameters (unchanged)

| Parameter | Value |
|-----------|-------|
| Total transactions | 500,000 |
| Fraud rate | 1-5% (5,000-25,000 judol / 475,000-495,000 normal) |
| Date range | 90 days |
| Normal merchants | 5,000 |
| Judol merchants | ~150 (scales with fraud rate) |
| Hybrid merchants | ~550 (10% of total, new) |
| Normal users | 50,000 |
| Judol users | ~600 (scales with fraud rate) |
| Output schema | Same 15 columns as v1 |
| Seed | 42 (reproducible) |

### 4.2 Merchant Pool Redesign

#### Merchant Categories (Normal)

| Category | % of Normal Pool | Characteristics |
|----------|:---:|---|
| **Regular Retail** | 30% | Daytime only (08:00-20:00), diverse amounts, moderate traffic |
| **24h Minimarket** | 20% | All hours, 25-35% nighttime traffic, diverse amounts |
| **Fuel Station (SPBU)** | 8% | All hours bias daytime, round amounts (50K-200K), moderate unique users |
| **Food & Beverage** | 20% | Lunch + dinner peaks, non-round amounts (15K-75K), high repeat rate |
| **Small Warung** | 10% | Daytime, low traffic (2-20 tx/month), few regular customers |
| **Parking/Toll** | 4% | Daytime commute peaks, small round amounts (5K-15K), high frequency users |
| **Online/E-Commerce** | 5% | All hours (slight midnight peak), diverse amounts, low repeat |
| **Cash-Out Service** | 1% | Daytime, round amounts (100K-500K), many unique users |
| **Event/Seasonal** | 2% | Burst traffic on event days, otherwise low/zero volume |

#### Merchant Categories (Judol)

| Category | % of Judol Pool | Characteristics |
|----------|:---:|---|
| **Slot/Casino Operator** | 60% | Peak 20:00-02:00, round amounts, many unique users |
| **Togel Operator** | 30% | Deposits 10-30 min before draw times (13,16,19,22), smaller amounts |
| **Diversified Operator** | 10% | Mixed hours, some non-round amounts, trying to blend in |

#### Hybrid Merchants (NEW — Critical Feature)

- **Count**: ~550 merchants (~10% of total merchant pool)
- **Source**: A subset of normal merchants (from 24h minimarket, small warung, F&B categories)
  whose QRIS codes are ALSO used for judol deposits
- **Transaction mix**: 70-85% normal transactions, 15-30% judol transactions
- **Label assignment**: Transactions at hybrid merchants are labeled based on the transaction
  itself — normal purchases get label=0, judol deposits get label=1
- **Why this matters**: These are the hardest merchants to detect. The model must learn to
  distinguish transaction patterns, not just merchant identity.

### 4.3 User Profile Redesign

#### Normal User Profiles

| Profile | % of Normal Pool | Key Characteristics |
|---------|:---:|---|
| **Regular Worker** | 40% | Daytime transactions, 2-5 favorite merchants, payday spike, some round amounts (pulsa/fuel 1-2x/month) |
| **Shift Worker** | 8% | Nighttime transactions (22:00-06:00) 3-5 nights/week, same merchants near workplace, non-round food/drink amounts |
| **Driver/Commuter** | 10% | High merchant diversity (15+ unique), cross-city, nighttime rest-area stops, fuel + food, monthly recurring routes |
| **Student** | 12% | Low amounts (5K-50K), campus-area merchants, monthly allowance pattern (heavy first week), round pulsa purchases |
| **Online Shopper** | 10% | Occasional midnight bursts (flash sales), diverse amounts, low merchant repeat, some large purchases |
| **Payday Spender** | 8% | Heavy spending on 25th-28th, multiple merchants, diverse amounts including round (bills), otherwise low activity |
| **Family Account** | 5% | Dual time pattern (daytime + evening), two merchant clusters, mixed amounts |
| **Business Traveler** | 4% | Multiple cities per month, many unique merchants, hotels + food + transport |
| **Retiree/Low Activity** | 3% | 5-15 transactions per month, daytime only, few merchants, small amounts |

#### Judol User Profiles

| Profile | % of Judol Pool | Key Characteristics |
|---------|:---:|---|
| **Casual Gambler** | 30% | 5-15 judol tx over 90 days, small amounts (10K-50K), no escalation, also does 70-80% normal shopping. Hardest to detect. |
| **Regular Gambler** | 40% | 30-80 judol tx over 90 days, moderate amounts (50K-200K), mild escalation, payday spikes, 40-60% normal tx mixed in. |
| **Heavy/Addicted** | 20% | 80-200+ judol tx, high amounts (100K-500K+), strong escalation, late-night concentration, still 30-40% normal tx. Easiest to detect. |
| **Smurfer** | 10% | Splits deposits across 3-5 merchants within 1-3 hour windows, many small amounts (20K-50K), tries to stay under radar. |

**Critical change**: ALL judol user profiles MUST also generate normal transactions at
normal merchants. A judol user's transaction history should be a MIX of legitimate daily
life + judol activity. The judol signal hides within the noise of normal behavior.

### 4.4 Transaction Generation Rules

#### Amount Distribution Changes

| Parameter | v1 | v2 |
|-----------|-----|-----|
| Normal round amount rate | 12% | **22-28%** (includes fuel, pulsa, parking, bills, round-priced menu items) |
| Judol round amount rate (merchant-side) | 91% | **78-85%** |
| Judol round amount rate (user-side) | 88% | **75-82%** |
| Judol near-round rate | 0% | **10-15%** (amount = round ± random 500-2000) |
| Normal amount sources | Single lognormal | **Profile-based**: fuel={50K,100K,200K}, pulsa={10K,25K,50K,100K}, parking={5K,10K,15K}, food=lognormal(10.5, 0.6), retail=lognormal(11.0, 0.8). **Amount tail rounding**: most amounts round to nearest 100. Tail distribution weighted: ~50% end in 000, ~25% end in 500, ~15% other hundreds (100-400, 600-900), ~10% truly random tails (e.g., 688, 462) — these exist but are uncommon. |

#### Timing Distribution Changes

| Parameter | v1 | v2 |
|-----------|-----|-----|
| Normal nighttime (20:00-06:00) % | ~8% | **22-30%** (24h stores, shift workers, online, drivers) |
| Judol daytime (08:00-18:00) % | ~10% | **30-40%** (lunch break, afternoon, togel draws at 13:00 & 16:00) |
| Normal payday boost | +30% | **+80-100%** (normal people also spend heavily on payday) |
| Judol payday boost | +50% | **+60-80%** (keep slightly higher than normal, but not dramatically) |

#### Merchant Assignment Changes

| Parameter | v1 | v2 |
|-----------|-----|-----|
| Judol user visits normal merchants | 0% | **40-60%** of total judol user transactions |
| Normal user visits hybrid merchant | 0% | **5-8%** of normal users occasionally transact at hybrid merchants (labeled normal) |
| Judol user merchant diversity | Judol merchants only | Also visits 3-8 normal merchants for daily life |
| Repeat rate at judol merchants | 2% | **8-15%** (some regulars, especially casual gamblers) |

#### Velocity Pattern Changes

| Parameter | v1 | v2 |
|-----------|-----|-----|
| Normal merchant velocity | Steady | **Profile-based**: steady for most, burst for event merchants, weekly pattern for F&B, payday spikes for all |
| Judol merchant velocity | Always bursty | **60% bursty (payday-driven), 25% steady (established operations), 15% random** |
| Normal merchant seasonal burst | None | **3-5% of merchants have 1-2 burst days (events, promotions, holidays)** |

### 4.5 Label Assignment Logic

Labels should be assigned at the **transaction level**, not the user or merchant level:

- Transaction by a normal user at a normal merchant → **label = 0**
- Transaction by a normal user at a hybrid merchant for normal purchase → **label = 0**
- Transaction by a judol user at a normal merchant for normal purchase → **label = 0**
- Transaction by a judol user at a judol merchant for gambling → **label = 1**
- Transaction by a judol user at a hybrid merchant for gambling → **label = 1**

This means a judol user will have BOTH label=0 and label=1 transactions. The model must
learn to detect the judol TRANSACTIONS, not just the judol USERS.

### 4.6 Architecture Overview

```
scripts/generate_dataset.py
    │
    ├── load_wilayah_data()          # Same as v1
    ├── create_merchant_pool()       # NEW: categorized merchants + hybrid pool
    ├── create_user_pool()           # NEW: profiled users with behavioral params
    ├── generate_normal_transactions()  # NEW: profile-driven generation
    ├── generate_judol_transactions()   # NEW: mixed judol+normal per user
    ├── generate_hybrid_transactions()  # NEW: mixed merchant transactions
    ├── assign_labels()              # NEW: transaction-level labeling
    ├── shuffle_and_export()         # Same as v1
    │
    └── Output: data/generated/parametric/pantau_dataset.csv
```

### 4.7 Output Schema (unchanged)

Same 15 columns as v1 — no schema changes needed:
`transaction_id, timestamp, user_id, merchant_id, amount, user_city, user_province,
merchant_city, merchant_province, transaction_type, device_id, is_round_amount,
tx_hour, tx_day_of_week, label`

---

## 5. Risks & Roadmap

### Implementation Phases

#### Phase 1: Merchant & User Pool Redesign
- Implement merchant categories (9 normal types + 3 judol types)
- Create hybrid merchant pool from normal merchants
- Implement user profiles (9 normal types + 4 judol types)
- Assign behavioral parameters per profile

#### Phase 2: Transaction Generation Logic
- Profile-driven amount generation (fuel, pulsa, parking, food, retail distributions)
- Profile-driven timing (shift workers, drivers, online shoppers get nighttime)
- Judol user mixed transactions (normal merchants + judol merchants)
- Near-round amount logic for judol evasion
- Smurfing pattern generation

#### Phase 3: Overlap & Edge Cases
- Hybrid merchant transaction mixing (70-85% normal / 15-30% judol)
- Normal payday surge (comparable to judol payday pattern)
- Event/seasonal merchant bursts
- Cross-visiting normal users at hybrid merchants
- Seasonal/new business activation patterns

#### Phase 4: Validation
- Run `scripts/audit/dataset_quality.py` on generated data
- Verify all success criteria are met (see Section 1)
- If any metric is out of range, adjust profile distributions and re-generate
- Run `python3 -m ml.train --sample 10000 --no-tune` for quick sanity check
- Run full training and compare v1 vs v2 metrics

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Too much overlap → model can't learn at all (F1 < 0.60) | High | Start conservative with overlap percentages, validate incrementally |
| Generation time exceeds 10 min with complex profiles | Medium | Profile assignment is pool-level (O(users)), not per-transaction |
| Hybrid merchants dominate false positives | Medium | Keep hybrid at 10%, tune normal/judol transaction ratio |
| Smurfing pattern creates unrealistic network patterns | Low | Limit smurfing to 10% of heavy users, cap at 3-5 merchants |
| Label imbalance within judol users (most tx are label=0) | High | Track per-user judol transaction counts; ensure enough judol-labeled tx remain |
| Low fraud rate (1-5%) makes learning harder | Medium | Expected — this is realistic. Use class-weighted training in ML pipeline. |

### Post-Generation Checklist

After generating v2 dataset, verify:
1. [ ] `python3 scripts/audit/dataset_quality.py --input data/generated/parametric/pantau_dataset.csv` — all metrics in target range
2. [ ] Total row count = 500,000
3. [ ] Fraud rate = 1-5% (configurable, default 3%)
4. [ ] No NaN values in any column
5. [ ] All merchant_ids are valid 15-char NMID format
6. [ ] `is_round_amount` correctly computed
7. [ ] Timestamp range covers full 90-day window
8. [ ] Judol users have both label=0 and label=1 transactions
9. [ ] Hybrid merchants have both label=0 and label=1 transactions
10. [ ] Quick ML training (`--sample 10000 --no-tune`) produces combined F1 between 0.60-0.90

---

## Appendix A: v1 Parameters for Reference

These are the current v1 parameters that will be replaced:

| Parameter | v1 Value | Notes |
|-----------|----------|-------|
| Fraud rate | 15% | Real-world is 1-5%, 15% is unrealistically high |
| `NORMAL_PARAMS["round_amount_rate"]` | 0.12 | Too low, causes IV=4.43 |
| `JUDOL_MERCHANT_PARAMS["round_amount_rate"]` | 0.91 | Too high, trivially separable |
| `JUDOL_USER_PARAMS["round_amount_rate"]` | 0.88 | Too high |
| `JUDOL_MERCHANT_PARAMS["repeat_sender_rate"]` | 0.02 | Too low, creates distinct network |
| `NORMAL_24H_MERCHANT_RATIO` | 0.30 | OK but 24h merchants still only get normal traffic |
| `JUDOL_USER_PARAMS["late_night_rate"]` | 0.72 | Too concentrated in nighttime |
| Normal gajian boost | +30% | Too small compared to judol +50% |
| Judol user→normal merchant cross-visit | 0% | Critical missing feature |
| Hybrid merchants | 0 | Critical missing feature |

## Appendix B: Research References for Audit Metrics

| Metric | Target | Source |
|--------|--------|--------|
| Silhouette < 0.30 | Rousseeuw (1987), J. Comp. & Appl. Math |
| Stump F1 < 0.70 | Standard ML baseline evaluation |
| IV < 0.50 | Siddiqi (2006), Credit Risk Scorecards |
| Bhattacharyya overlap > 0.50 | Bhattacharyya (1943) |
| Borderline 20-30% | Ho & Basu (2002), IEEE TPAMI; Lorena et al. (2019) |
| Combined ML F1 0.75-0.90 | IEEE S&P fraud detection literature |
