"""
Pantau — Stage 2: GAN Augmentation Pipeline
=============================================
Trains CTGAN on the parametric base dataset (500K rows)
and generates a new synthetic dataset with richer statistical properties.

Usage:
    python scripts/train_gan.py                     # default: CTGAN, 500K rows
    python scripts/train_gan.py --model tvae        # use TVAE (faster)
    python scripts/train_gan.py --rows 1000000      # generate 1M rows
    python scripts/train_gan.py --epochs 500        # more training epochs
"""

import argparse
import os
import time

import pandas as pd
from rdt.transformers import LabelEncoder
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMETRIC_PATH = os.path.join(BASE_DIR, "data", "generated", "parametric", "pantau_dataset.csv")
GAN_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "generated", "gan")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# COLUMN METADATA
# ============================================================

# Define which columns are categorical vs numerical vs datetime
# This helps CTGAN handle mixed types correctly
COLUMN_TYPES = {
    "transaction_id": "id",
    "timestamp": "datetime",
    "user_id": "categorical",
    "merchant_id": "categorical",
    "amount": "numerical",
    "user_city": "categorical",
    "user_province": "categorical",
    "merchant_city": "categorical",
    "merchant_province": "categorical",
    "transaction_type": "categorical",
    "device_id": "id",
    "is_round_amount": "boolean",
    "tx_hour": "numerical",
    "tx_day_of_week": "numerical",
    "label": "categorical",
}

# Columns excluded from GAN training (no behavioral signal, regenerated post-hoc)
EXCLUDE_COLUMNS = ["transaction_id", "device_id", "user_id", "merchant_id"]

# Columns the GAN should learn (behavioral features only)
TRAIN_COLUMNS = [c for c in COLUMN_TYPES if c not in EXCLUDE_COLUMNS]


# ============================================================
# METADATA BUILDER
# ============================================================

def build_metadata(df: pd.DataFrame) -> Metadata:
    """Build SDV Metadata object describing column types."""
    metadata = Metadata.detect_from_dataframes({"transactions": df})

    overrides = {
        "user_city": {"sdtype": "categorical"},
        "user_province": {"sdtype": "categorical"},
        "merchant_city": {"sdtype": "categorical"},
        "merchant_province": {"sdtype": "categorical"},
        "is_round_amount": {"sdtype": "boolean"},
        "label": {"sdtype": "categorical"},
    }

    for col, kwargs in overrides.items():
        metadata.update_column(column_name=col, table_name="transactions", **kwargs)

    return metadata


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for GAN training — drop ID columns, fix types."""
    df = df.copy()

    # Drop ID columns — regenerated post-hoc
    df = df.drop(columns=EXCLUDE_COLUMNS, errors="ignore")

    # Ensure correct types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["label"] = df["label"].astype(str)
    df["tx_hour"] = df["tx_hour"].astype(int)
    df["tx_day_of_week"] = df["tx_day_of_week"].astype(int)
    df["is_round_amount"] = df["is_round_amount"].astype(bool)

    # City/province as categorical
    for col in ["user_city", "user_province", "merchant_city", "merchant_province"]:
        df[col] = df[col].astype(str)

    return df


# ============================================================
# POST-PROCESSING
# ============================================================

def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up GAN output — regenerate IDs, fix types, match expected schema."""
    import random as _rand
    import string as _string

    df = df.copy()

    # Fix numerical types first
    df["amount"] = df["amount"].round(0).astype(int).clip(lower=1000)
    df["tx_hour"] = df["tx_hour"].round(0).astype(int).clip(0, 23)
    df["tx_day_of_week"] = df["tx_day_of_week"].round(0).astype(int).clip(0, 6)
    df["label"] = df["label"].astype(int)
    df["is_round_amount"] = df["amount"].apply(
        lambda x: x % 10000 == 0 or x % 50000 == 0
    )

    # Format timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # --- Regenerate all ID columns ---

    # transaction_id: GAN-NNNNNN
    df["transaction_id"] = [f"GAN-{i:06d}" for i in range(len(df))]

    # device_id: DEV-10 hex chars
    df["device_id"] = [f"DEV-{os.urandom(5).hex().upper()}" for _ in range(len(df))]

    # user_id: 10-15 digit bank account number
    df["user_id"] = [
        str(_rand.randint(10**9, 10**(_rand.randint(10, 15) - 1)))
        for _ in range(len(df))
    ]

    # merchant_id: depends on transaction_type
    phone_prefixes = [
        "0852", "0853", "0811", "0812", "0813", "0821", "0822", "0851",
        "0857", "0856", "0896", "0895", "0897", "0898", "0899",
        "0817", "0818", "0819", "0859", "0877", "0878",
        "0832", "0833", "0838",
        "0881", "0882", "0883", "0884", "0885", "0886", "0887", "0888", "0889",
    ]
    merchant_ids = []
    for _, row in df.iterrows():
        tx_type = row["transaction_type"]
        if tx_type == "QRIS":
            acquirer = str(_rand.randint(10, 99))
            suffix = "".join(_rand.choices(_string.ascii_uppercase + _string.digits, k=11))
            merchant_ids.append(f"ID{acquirer}{suffix}")
        elif tx_type.startswith("EWALLET_"):
            provider = tx_type.replace("EWALLET_", "")
            prefix = _rand.choice(phone_prefixes)
            phone = prefix + "".join([str(_rand.randint(0, 9)) for _ in range(8)])
            merchant_ids.append(f"{provider}-{phone}")
        else:
            merchant_ids.append(f"ID{_rand.randint(10, 99)}{''.join(_rand.choices(_string.ascii_uppercase + _string.digits, k=11))}")
    df["merchant_id"] = merchant_ids

    # Reorder columns to match original schema
    column_order = [
        "transaction_id", "timestamp", "user_id", "merchant_id", "amount",
        "user_city", "user_province", "merchant_city", "merchant_province",
        "transaction_type", "device_id", "is_round_amount",
        "tx_hour", "tx_day_of_week", "label",
    ]
    df = df[column_order]

    return df


# ============================================================
# SUMMARY STATS
# ============================================================

def print_summary(df: pd.DataFrame, label: str = "Dataset"):
    """Print dataset summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"{label} SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total rows:          {len(df):,}")
    print(f"Normal (label=0):    {(df['label'].astype(int)==0).sum():,} ({(df['label'].astype(int)==0).sum()/len(df)*100:.1f}%)")
    print(f"Judol  (label=1):    {(df['label'].astype(int)==1).sum():,} ({(df['label'].astype(int)==1).sum()/len(df)*100:.1f}%)")
    if "user_id" in df.columns:
        print(f"Unique users:        {df['user_id'].nunique():,}")
    if "merchant_id" in df.columns:
        print(f"Unique merchants:    {df['merchant_id'].nunique():,}")
    print(f"Amount range:        Rp{df['amount'].astype(float).min():,.0f} - Rp{df['amount'].astype(float).max():,.0f}")

    # Transaction type distribution
    print(f"\n--- Transaction Type Distribution ---")
    tx_counts = df["transaction_type"].value_counts()
    for tx_type, count in tx_counts.items():
        pct = count / len(df) * 100
        judol_count = ((df["transaction_type"] == tx_type) & (df["label"].astype(int) == 1)).sum()
        print(f"  {tx_type}: {count:,} ({pct:.1f}%) — {judol_count:,} judol")

    # Temporal
    ts = pd.to_datetime(df["timestamp"])
    print(f"\n--- Temporal Distribution ---")
    print(f"  Date range:        {ts.min()} → {ts.max()}")
    print(f"  Weekday/Weekend:   {(ts.dt.weekday < 5).sum():,} / {(ts.dt.weekday >= 5).sum():,}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pantau GAN Augmentation Pipeline")
    parser.add_argument("--model", choices=["ctgan", "tvae"], default="ctgan",
                        help="GAN model to use (default: ctgan)")
    parser.add_argument("--rows", type=int, default=500000,
                        help="Number of rows to generate (default: 500,000)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: 300)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Training batch size (default: 500)")
    parser.add_argument("--sample-input", type=int, default=None,
                        help="Use only N rows from base dataset for faster training (default: use all)")
    parser.add_argument("--save-model", action="store_true", default=True,
                        help="Save trained model to models/ directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Pantau — Stage 2: GAN Augmentation Pipeline")
    print("=" * 60)
    print(f"  Model:       {args.model.upper()}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Output rows: {args.rows:,}")
    print()

    # ---- Load base dataset ----
    print("[1/5] Loading parametric base dataset...")
    df_base = pd.read_csv(PARAMETRIC_PATH)
    print(f"  Loaded {len(df_base):,} rows, {len(df_base.columns)} columns")

    if args.sample_input:
        df_base = df_base.sample(n=args.sample_input, random_state=42).reset_index(drop=True)
        print(f"  Sampled down to {len(df_base):,} rows for faster training")

    print_summary(df_base, "BASE (PARAMETRIC)")

    # ---- Preprocess ----
    print("\n[2/5] Preprocessing data...")
    df_train = preprocess(df_base)
    print(f"  Training columns: {list(df_train.columns)}")
    print(f"  Excluded (regenerated post-hoc): {EXCLUDE_COLUMNS}")

    # ---- Build metadata ----
    print("\n[3/5] Building metadata & initializing model...")
    metadata = build_metadata(df_train)

    if args.model == "ctgan":
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True,
        )
    else:
        synthesizer = TVAESynthesizer(
            metadata,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    print(f"  Initialized {args.model.upper()} synthesizer")

    # Use LabelEncoder for high-cardinality city columns (514 unique each)
    # This avoids one-hot explosion: 1 column instead of 514 per city field
    synthesizer.auto_assign_transformers(df_train)
    synthesizer.update_transformers({
        "user_city": LabelEncoder(add_noise=True),
        "merchant_city": LabelEncoder(add_noise=True),
    })
    print("  Applied LabelEncoder for city columns (avoids one-hot OOM)")

    # ---- Train ----
    print(f"\n[4/5] Training {args.model.upper()} (this may take a while on CPU)...")
    start_time = time.time()
    synthesizer.fit(df_train)
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time/60:.1f} minutes")

    # Save model
    if args.save_model:
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"pantau_{args.model}.pkl")
        synthesizer.save(model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model saved: {model_path} ({model_size_mb:.1f} MB)")

    # ---- Generate ----
    print(f"\n[5/5] Generating {args.rows:,} synthetic rows...")
    start_time = time.time()
    df_gan = synthesizer.sample(num_rows=args.rows)
    gen_time = time.time() - start_time
    print(f"  Generation completed in {gen_time/60:.1f} minutes")

    # ---- Post-process ----
    print("\nPost-processing...")
    df_gan = postprocess(df_gan)

    print_summary(df_gan, "GAN-GENERATED")

    # ---- Save ----
    os.makedirs(GAN_OUTPUT_DIR, exist_ok=True)
    output_filename = f"pantau_gan_{args.model}_{args.rows // 1000}k.csv"
    output_path = os.path.join(GAN_OUTPUT_DIR, output_filename)
    df_gan.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({file_size_mb:.1f} MB)")

    # ---- Compare distributions ----
    print(f"\n{'=' * 60}")
    print("DISTRIBUTION COMPARISON (Base vs GAN)")
    print(f"{'=' * 60}")

    for col in ["amount", "tx_hour", "tx_day_of_week"]:
        base_mean = df_base[col].astype(float).mean()
        gan_mean = df_gan[col].astype(float).mean()
        base_std = df_base[col].astype(float).std()
        gan_std = df_gan[col].astype(float).std()
        print(f"  {col}:")
        print(f"    Base — mean: {base_mean:.1f}, std: {base_std:.1f}")
        print(f"    GAN  — mean: {gan_mean:.1f}, std: {gan_std:.1f}")

    base_label_pct = (df_base["label"] == 1).sum() / len(df_base) * 100
    gan_label_pct = (df_gan["label"].astype(int) == 1).sum() / len(df_gan) * 100
    print(f"  label (judol %):")
    print(f"    Base — {base_label_pct:.1f}%")
    print(f"    GAN  — {gan_label_pct:.1f}%")

    print(f"\n✅ GAN augmentation complete!")
    return df_gan


if __name__ == "__main__":
    main()
