"""
Pantau — GAN Output Post-Processing Fix
==========================================
Fixes known GAN output issues:
1. Label ratio drift (23% → 15% judol) — resample to restore 85/15
2. Unique ID explosion (500K unique merchants) — assign realistic cardinality
3. Preserves all GAN-learned behavioral columns (amount, temporal, geo, etc.)

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

TARGET_ROWS = 500_000
TARGET_JUDOL_RATE = 0.15  # 15% judol, 85% normal
TARGET_NORMAL = int(TARGET_ROWS * (1 - TARGET_JUDOL_RATE))  # 425,000
TARGET_JUDOL = TARGET_ROWS - TARGET_NORMAL                   # 75,000

# User cardinality targets (from parametric: ~81K users, avg 6 tx/user)
TARGET_NORMAL_USERS = 70_000
TARGET_JUDOL_USERS = 11_000

# Merchant cardinality targets (from parametric: ~162K merchants, avg 3 tx/merchant)
TARGET_NORMAL_MERCHANTS = 140_000
TARGET_JUDOL_MERCHANTS = 22_000

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
# LABEL RATIO FIX
# ============================================================

def fix_label_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to achieve target 85/15 label ratio."""
    df_normal = df[df["label"] == 0]
    df_judol = df[df["label"] == 1]

    print(f"  Before: {len(df_normal):,} normal, {len(df_judol):,} judol")

    # Downsample to targets
    if len(df_normal) > TARGET_NORMAL:
        df_normal = df_normal.sample(n=TARGET_NORMAL, random_state=SEED)
    elif len(df_normal) < TARGET_NORMAL:
        # Upsample if needed
        extra = TARGET_NORMAL - len(df_normal)
        df_normal = pd.concat([df_normal, df_normal.sample(n=extra, replace=True, random_state=SEED)])

    if len(df_judol) > TARGET_JUDOL:
        df_judol = df_judol.sample(n=TARGET_JUDOL, random_state=SEED)
    elif len(df_judol) < TARGET_JUDOL:
        extra = TARGET_JUDOL - len(df_judol)
        df_judol = pd.concat([df_judol, df_judol.sample(n=extra, replace=True, random_state=SEED)])

    df_fixed = pd.concat([df_normal, df_judol]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"  After:  {(df_fixed['label']==0).sum():,} normal, {(df_fixed['label']==1).sum():,} judol")

    return df_fixed


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

    # Assign with power-law distribution (some users have many txs)
    # Zipf distribution: index^(-alpha), alpha=1.2 gives realistic long tail
    def zipf_assign(pool, n_rows, alpha=1.2):
        weights = np.arange(1, len(pool) + 1, dtype=float) ** (-alpha)
        weights /= weights.sum()
        indices = np_rng.choice(len(pool), size=n_rows, p=weights)
        return [pool[i] for i in indices]

    df_normal["user_id"] = zipf_assign(normal_user_pool, len(df_normal), alpha=1.1)
    df_judol["user_id"] = zipf_assign(judol_user_pool, len(df_judol), alpha=0.9)

    # --- Generate merchant ID pools ---
    print("  Generating merchant ID pools...")
    normal_merchant_pool = generate_merchant_ids(
        TARGET_NORMAL_MERCHANTS, df_normal["transaction_type"], rng
    )
    judol_merchant_pool = generate_merchant_ids(
        TARGET_JUDOL_MERCHANTS, df_judol["transaction_type"], rng
    )

    # Merchants: judol merchants have more txs (fewer merchants, more concentrated)
    df_normal["merchant_id"] = zipf_assign(normal_merchant_pool, len(df_normal), alpha=1.3)
    df_judol["merchant_id"] = zipf_assign(judol_merchant_pool, len(df_judol), alpha=0.8)

    # --- Fix merchant_id format to match transaction_type ---
    # Ensure e-wallet txs have PROVIDER-phone format, QRIS has NMID format
    def fix_merchant_format(row):
        tx_type = row["transaction_type"]
        mid = row["merchant_id"]
        if tx_type == "QRIS" and not mid.startswith("ID"):
            return generate_merchant_id_qris(rng)
        elif tx_type.startswith("EWALLET_") and "-0" not in mid:
            provider = tx_type.replace("EWALLET_", "")
            return generate_merchant_id_ewallet(provider, rng)
        return mid

    df_normal["merchant_id"] = df_normal.apply(fix_merchant_format, axis=1)
    df_judol["merchant_id"] = df_judol.apply(fix_merchant_format, axis=1)

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
    print(f"\n[1/4] Loading GAN output: {args.input}")
    df = pd.read_csv(args.input)
    df["label"] = df["label"].astype(int)
    df["is_round_amount"] = df["is_round_amount"].astype(bool)
    print(f"  Loaded {len(df):,} rows ({(df['label']==1).mean()*100:.1f}% judol)")

    # Fix label ratio
    print(f"\n[2/4] Fixing label ratio → {TARGET_JUDOL_RATE*100:.0f}% judol...")
    df = fix_label_ratio(df)

    # Reassign IDs
    print(f"\n[3/4] Assigning realistic user/merchant IDs...")
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
    print(f"\n[4/4] Saving fixed dataset...")
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
