"""
Pantau — GAN Output Post-Processing Fix
==========================================
Fixes known GAN output issues:
1. Unique ID explosion (500K unique merchants) — assign realistic cardinality
2. Preserves GAN-learned label ratio and all behavioral columns

Usage:
    python3 scripts/fix_gan_output.py [--input PATH] [--output PATH]
"""

import argparse
import os
import random
import string
import sys

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION — matches parametric dataset characteristics
# ============================================================

# User cardinality targets (from parametric: ~81K users, avg 6 tx/user)
TARGET_NORMAL_USERS = 70_000
TARGET_JUDOL_USERS = 11_000

# Merchant cardinality targets (from parametric: ~162K merchants, avg 3 tx/merchant)
TARGET_NORMAL_MERCHANTS = 140_000
TARGET_JUDOL_MERCHANTS = 22_000

# Max transactions per entity (cap to prevent super-entities)
MAX_TX_PER_USER = 30
MAX_TX_PER_MERCHANT = 15

# Seed for reproducibility
SEED = 42

# Phone prefixes for e-wallet merchant IDs
PHONE_PREFIXES = [
    "0852", "0853", "0811", "0812", "0813", "0821", "0822", "0851",
    "0857", "0856", "0896", "0895", "0897", "0898", "0899",
    "0817", "0818", "0819", "0859", "0877", "0878",
    "0832", "0833", "0838",
    "0881", "0882", "0883", "0884", "0885", "0886", "0887", "0888", "0889",
]


# ============================================================
# ID GENERATORS
# ============================================================

def generate_user_ids(n: int, rng: random.Random) -> list:
    """Generate n unique user_ids (10-15 digit bank account numbers)."""
    ids = set()
    while len(ids) < n:
        digits = rng.randint(10, 15)
        uid = str(rng.randint(10 ** (digits - 1), 10 ** digits - 1))
        ids.add(uid)
    return list(ids)


def generate_merchant_id_qris(rng: random.Random) -> str:
    acquirer = str(rng.randint(10, 99))
    suffix = "".join(rng.choices(string.ascii_uppercase + string.digits, k=11))
    return f"ID{acquirer}{suffix}"


def generate_merchant_id_ewallet(provider: str, rng: random.Random) -> str:
    prefix = rng.choice(PHONE_PREFIXES)
    phone = prefix + "".join([str(rng.randint(0, 9)) for _ in range(8)])
    return f"{provider}-{phone}"


def generate_merchant_ids(n: int, tx_types: pd.Series, rng: random.Random) -> list:
    """Generate n unique merchant_ids with appropriate format per transaction type."""
    # Determine tx_type distribution to generate correct mix
    type_counts = tx_types.value_counts(normalize=True)
    ids = set()
    id_list = []

    for _ in range(n * 2):  # over-generate to ensure uniqueness
        # Pick a random type based on distribution
        tx_type = rng.choices(list(type_counts.index), weights=list(type_counts.values))[0]
        if tx_type == "QRIS":
            mid = generate_merchant_id_qris(rng)
        elif tx_type.startswith("EWALLET_"):
            provider = tx_type.replace("EWALLET_", "")
            mid = generate_merchant_id_ewallet(provider, rng)
        else:
            mid = generate_merchant_id_qris(rng)

        if mid not in ids:
            ids.add(mid)
            id_list.append(mid)
            if len(id_list) >= n:
                break

    return id_list[:n]


# ============================================================
# ID REASSIGNMENT
# ============================================================

def assign_realistic_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reassign user_id and merchant_id with realistic cardinality.
    
    Strategy:
    - Normal users: ~70K unique, power-law distribution (some users more active)
    - Judol users: ~11K unique, higher tx frequency (gamblers are repeat users)
    - Normal merchants: ~140K unique, moderate repeat
    - Judol merchants: ~22K unique, higher repeat (fewer merchants, more txs each)
    """
    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)

    df = df.copy()
    df_normal = df[df["label"] == 0].copy()
    df_judol = df[df["label"] == 1].copy()

    # --- Generate user ID pools ---
    print("  Generating user ID pools...")
    normal_user_pool = generate_user_ids(TARGET_NORMAL_USERS, rng)
    judol_user_pool = generate_user_ids(TARGET_JUDOL_USERS, rng)

    # Assign with capped uniform-random distribution
    # Each entity gets at most MAX_TX txs, ensuring realistic spread
    def capped_assign(pool, n_rows, max_tx):
        """Assign IDs from pool ensuring no entity exceeds max_tx transactions."""
        assignments = []
        pool_size = len(pool)
        counts = {}

        for _ in range(n_rows):
            # Pick random entity, retry if at cap
            for _attempt in range(50):
                idx = np_rng.randint(0, pool_size)
                entity = pool[idx]
                if counts.get(entity, 0) < max_tx:
                    counts[entity] = counts.get(entity, 0) + 1
                    assignments.append(entity)
                    break
            else:
                # Fallback: find any entity under cap
                for eid in pool:
                    if counts.get(eid, 0) < max_tx:
                        counts[eid] = counts.get(eid, 0) + 1
                        assignments.append(eid)
                        break

        return assignments

    df_normal["user_id"] = capped_assign(normal_user_pool, len(df_normal), MAX_TX_PER_USER)
    df_judol["user_id"] = capped_assign(judol_user_pool, len(df_judol), MAX_TX_PER_USER)

    # --- Generate merchant ID pools PER TRANSACTION TYPE ---
    # This prevents format mismatch (QRIS merchant assigned to EWALLET tx)
    print("  Generating merchant ID pools per transaction type...")

    def generate_typed_merchant_pools(df_subset, target_total, rng):
        """Generate merchant pools split by transaction type."""
        type_dist = df_subset["transaction_type"].value_counts(normalize=True)
        pools = {}
        for tx_type, frac in type_dist.items():
            n = max(int(target_total * frac), 1)
            pool = set()
            while len(pool) < n:
                if tx_type == "QRIS":
                    pool.add(generate_merchant_id_qris(rng))
                elif tx_type.startswith("EWALLET_"):
                    provider = tx_type.replace("EWALLET_", "")
                    pool.add(generate_merchant_id_ewallet(provider, rng))
                else:
                    pool.add(generate_merchant_id_qris(rng))
            pools[tx_type] = list(pool)
        return pools

    normal_merchant_pools = generate_typed_merchant_pools(df_normal, TARGET_NORMAL_MERCHANTS, rng)
    judol_merchant_pools = generate_typed_merchant_pools(df_judol, TARGET_JUDOL_MERCHANTS, rng)

    # Assign merchants from correct pool per transaction type
    def capped_assign_by_type(df_subset, pools, max_tx):
        assignments = []
        counts = {}
        for _, row in df_subset.iterrows():
            tx_type = row["transaction_type"]
            pool = pools.get(tx_type, pools.get("QRIS", []))
            pool_size = len(pool)
            assigned = False
            for _attempt in range(50):
                idx = np_rng.randint(0, pool_size)
                entity = pool[idx]
                if counts.get(entity, 0) < max_tx:
                    counts[entity] = counts.get(entity, 0) + 1
                    assignments.append(entity)
                    assigned = True
                    break
            if not assigned:
                for eid in pool:
                    if counts.get(eid, 0) < max_tx:
                        counts[eid] = counts.get(eid, 0) + 1
                        assignments.append(eid)
                        break
        return assignments

    df_normal["merchant_id"] = capped_assign_by_type(df_normal, normal_merchant_pools, MAX_TX_PER_MERCHANT)
    df_judol["merchant_id"] = capped_assign_by_type(df_judol, judol_merchant_pools, MAX_TX_PER_MERCHANT)

    # --- Regenerate transaction_id and device_id ---
    df_out = pd.concat([df_normal, df_judol]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("  Regenerating transaction_id and device_id...")
    df_out["transaction_id"] = [f"GAN-{i:06d}" for i in range(len(df_out))]
    df_out["device_id"] = [f"DEV-{os.urandom(5).hex().upper()}" for _ in range(len(df_out))]

    return df_out


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fix GAN output post-processing issues")
    parser.add_argument("--input", "-i", type=str,
                        default="data/generated/gan/pantau_gan_ctgan_500k.csv",
                        help="Path to raw GAN output")
    parser.add_argument("--output", "-o", type=str,
                        default="data/generated/gan/pantau_gan_ctgan.csv",
                        help="Path to save fixed dataset")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: GAN output not found: {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("  PANTAU — GAN Output Post-Processing Fix")
    print("=" * 60)

    # Load
    print(f"\n[1/3] Loading GAN output: {args.input}")
    df = pd.read_csv(args.input)
    df["label"] = df["label"].astype(int)
    df["is_round_amount"] = df["is_round_amount"].astype(bool)
    n_judol = (df['label'] == 1).sum()
    n_normal = (df['label'] == 0).sum()
    print(f"  Loaded {len(df):,} rows — {n_normal:,} normal ({n_normal/len(df)*100:.1f}%), {n_judol:,} judol ({n_judol/len(df)*100:.1f}%)")
    print(f"  Keeping GAN's natural label ratio (no resampling)")

    # Reassign IDs
    print(f"\n[2/3] Assigning realistic user/merchant IDs...")
    df = assign_realistic_ids(df)

    # Reorder columns
    column_order = [
        "transaction_id", "timestamp", "user_id", "merchant_id", "amount",
        "user_city", "user_province", "merchant_city", "merchant_province",
        "transaction_type", "device_id", "is_round_amount",
        "tx_hour", "tx_day_of_week", "label",
    ]
    df = df[column_order]

    # Save
    print(f"\n[3/3] Saving fixed dataset...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Saved to {args.output} ({size_mb:.1f} MB)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  FIXED DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total rows:       {len(df):,}")
    print(f"  Normal (label=0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"  Judol (label=1):  {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"  Unique users:     {df['user_id'].nunique():,}")
    print(f"  Unique merchants: {df['merchant_id'].nunique():,}")
    print(f"  Amount range:     Rp{df['amount'].min():,} - Rp{df['amount'].max():,}")

    # Per-entity tx counts
    user_txs = df.groupby("user_id").size()
    merchant_txs = df.groupby("merchant_id").size()
    print(f"\n  Avg tx/user:      {user_txs.mean():.1f} (min={user_txs.min()}, max={user_txs.max()})")
    print(f"  Avg tx/merchant:  {merchant_txs.mean():.1f} (min={merchant_txs.min()}, max={merchant_txs.max()})")
    print(f"{'=' * 60}")
    print("  Done! ✓")


if __name__ == "__main__":
    main()
