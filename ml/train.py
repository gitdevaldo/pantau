"""
Pantau ML — Training Orchestrator
===================================
Trains all 6 detection layers and combines scores.

Usage:
    python3 -m ml.train [--input path/to/dataset.csv] [--sample N]
"""

import argparse
import os
import sys
import time

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models import (
    user_behavior,
    merchant_behavior,
    network_cluster,
    temporal_pattern,
    velocity_delta,
    money_flow,
)
from ml import scoring


def load_dataset(path: str, sample: int = None) -> pd.DataFrame:
    """Load and prepare the transaction dataset."""
    print(f"\n[1/8] Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} transactions")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # Ensure proper types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_round_amount"] = df["is_round_amount"].astype(bool)
    df["label"] = df["label"].astype(int)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(df):,} transactions")

    return df


def train_all(df: pd.DataFrame) -> dict:
    """Train all 6 layers and return combined results."""
    results = {}
    timings = {}

    # ---- Layer 1: User Behavior ----
    print("\n[2/8] Training Layer 1: User Behavior...")
    t0 = time.time()
    results["user"] = user_behavior.train(df)
    timings["user"] = time.time() - t0
    print(f"  ⏱ {timings['user']:.1f}s")

    # ---- Layer 2: Merchant Behavior ----
    print("\n[3/8] Training Layer 2: Merchant Behavior...")
    t0 = time.time()
    results["merchant"] = merchant_behavior.train(df)
    timings["merchant"] = time.time() - t0
    print(f"  ⏱ {timings['merchant']:.1f}s")

    # ---- Layer 3: Network Clustering ----
    print("\n[4/8] Training Layer 3: Network Clustering...")
    t0 = time.time()
    results["network"] = network_cluster.train(df)
    timings["network"] = time.time() - t0
    print(f"  ⏱ {timings['network']:.1f}s")

    # ---- Layer 4: Temporal Pattern ----
    print("\n[5/8] Training Layer 4: Temporal Pattern...")
    t0 = time.time()
    results["temporal"] = temporal_pattern.train(df)
    timings["temporal"] = time.time() - t0
    print(f"  ⏱ {timings['temporal']:.1f}s")

    # ---- Layer 5: Velocity Delta ----
    print("\n[6/8] Training Layer 5: Velocity Delta...")
    t0 = time.time()
    results["velocity"] = velocity_delta.train(df)
    timings["velocity"] = time.time() - t0
    print(f"  ⏱ {timings['velocity']:.1f}s")

    # ---- Layer 6: Money Flow ----
    print("\n[7/8] Training Layer 6: Money Flow...")
    t0 = time.time()
    results["flow"] = money_flow.train(df)
    timings["flow"] = time.time() - t0
    print(f"  ⏱ {timings['flow']:.1f}s")

    # ---- Combined Scoring ----
    print("\n[8/8] Computing combined risk scores...")
    scored_df = scoring.combine_scores(df, results)
    combined_metrics = scoring.evaluate(scored_df)
    scoring.print_report(combined_metrics)

    results["_scored"] = scored_df
    results["_combined_metrics"] = combined_metrics
    results["_timings"] = timings

    return results


def save_all(results: dict):
    """Save all trained models and artifacts."""
    print("\nSaving models...")

    # Layer 1 & 2: have sklearn models
    if "user" in results:
        user_behavior.save(results["user"]["model"], results["user"]["scaler"])
    if "merchant" in results:
        merchant_behavior.save(results["merchant"]["model"], results["merchant"]["scaler"])
    if "network" in results:
        network_cluster.save(results["network"]["model"], results["network"]["scaler"])

    # Layers 4-6: rule/stat-based, save configs
    temporal_pattern.save()
    velocity_delta.save()
    money_flow.save()

    # Save scored transactions
    if "_scored" in results:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scored")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "scored_transactions.csv")
        results["_scored"].to_csv(out_path, index=False)
        print(f"  Scored transactions saved to {out_path}")


def print_summary(results: dict):
    """Print per-layer performance summary."""
    print("\n" + "=" * 70)
    print("  PANTAU — Training Summary")
    print("=" * 70)
    print(f"  {'Layer':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>8}")
    print("  " + "-" * 65)

    layer_names = {
        "user": "1. User Behavior",
        "merchant": "2. Merchant Behavior",
        "network": "3. Network Clustering",
        "temporal": "4. Temporal Pattern",
        "velocity": "5. Velocity Delta",
        "flow": "6. Money Flow",
    }

    timings = results.get("_timings", {})

    for key, name in layer_names.items():
        if key in results:
            m = results[key]["metrics"]
            t = timings.get(key, 0)
            print(f"  {name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1_score']:>10.4f} {t:>7.1f}s")

    if "_combined_metrics" in results:
        cm = results["_combined_metrics"]
        total_time = sum(timings.values())
        print("  " + "-" * 65)
        print(f"  {'COMBINED'::<25} {cm['precision']:>10.4f} {cm['recall']:>10.4f} "
              f"{cm['f1_score']:>10.4f} {total_time:>7.1f}s")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pantau ML Training Pipeline")
    parser.add_argument("--input", "-i", type=str,
                        default="data/generated/gan/pantau_gan_ctgan.csv",
                        help="Path to training dataset CSV")
    parser.add_argument("--sample", "-s", type=int, default=None,
                        help="Sample N rows for faster testing")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving models")
    args = parser.parse_args()

    # Fallback to parametric dataset if GAN not available
    if not os.path.exists(args.input):
        fallback = "data/generated/parametric/pantau_dataset.csv"
        if os.path.exists(fallback):
            print(f"  GAN dataset not found, using parametric: {fallback}")
            args.input = fallback
        else:
            print(f"  ERROR: Dataset not found at {args.input}")
            sys.exit(1)

    start = time.time()
    df = load_dataset(args.input, sample=args.sample)
    results = train_all(df)

    if not args.no_save:
        save_all(results)

    print_summary(results)
    print(f"\n  Total pipeline time: {time.time() - start:.1f}s")
    print("  Done! ✓")


if __name__ == "__main__":
    main()
