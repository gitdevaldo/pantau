"""
Pantau ML — Training Orchestrator
===================================
Trains all 6 detection layers and combines scores.
Saves models, scored transactions, and training report per dataset source.

Usage:
    python3 -m ml.train [--input path/to/dataset.csv] [--sample N] [--tag name]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

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


def save_all(results: dict, tag: str):
    """Save all trained models and artifacts, organized by tag."""
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Models saved under models/{tag}/
    model_dir = os.path.join(project_root, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\nSaving models to models/{tag}/...")

    if "user" in results:
        user_behavior.save(results["user"]["model"], results["user"]["scaler"],
                           os.path.join(model_dir, "user_behavior.pkl"))
    if "merchant" in results:
        merchant_behavior.save(results["merchant"]["model"], results["merchant"]["scaler"],
                               os.path.join(model_dir, "merchant_behavior.pkl"))
    if "network" in results:
        network_cluster.save(results["network"]["model"], results["network"]["scaler"],
                             os.path.join(model_dir, "network_cluster.pkl"))

    temporal_pattern.save(path=os.path.join(model_dir, "temporal_pattern.pkl"))
    velocity_delta.save(path=os.path.join(model_dir, "velocity_delta.pkl"))
    money_flow.save(path=os.path.join(model_dir, "money_flow.pkl"))

    # Scored transactions saved under data/scored/{tag}/
    if "_scored" in results:
        scored_dir = os.path.join(project_root, "data", "scored", tag)
        os.makedirs(scored_dir, exist_ok=True)
        out_path = os.path.join(scored_dir, "scored_transactions.csv")
        results["_scored"].to_csv(out_path, index=False)
        print(f"  Scored transactions saved to {out_path}")


def build_report(results: dict, tag: str, input_path: str, elapsed: float) -> str:
    """Build a text report of training results."""
    lines = []
    lines.append("=" * 70)
    lines.append("  PANTAU — Training Report")
    lines.append(f"  Dataset: {tag} ({input_path})")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    layer_names = {
        "user": "1. User Behavior",
        "merchant": "2. Merchant Behavior",
        "network": "3. Network Clustering",
        "temporal": "4. Temporal Pattern",
        "velocity": "5. Velocity Delta",
        "flow": "6. Money Flow",
    }

    timings = results.get("_timings", {})

    lines.append(f"\n  {'Layer':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>8}")
    lines.append("  " + "-" * 65)

    for key, name in layer_names.items():
        if key in results:
            m = results[key]["metrics"]
            t = timings.get(key, 0)
            lines.append(f"  {name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                         f"{m['f1_score']:>10.4f} {t:>7.1f}s")

    if "_combined_metrics" in results:
        cm = results["_combined_metrics"]
        total_time = sum(timings.values())
        lines.append("  " + "-" * 65)
        lines.append(f"  {'COMBINED':<25} {cm['precision']:>10.4f} {cm['recall']:>10.4f} "
                     f"{cm['f1_score']:>10.4f} {total_time:>7.1f}s")

        lines.append(f"\n  Avg score (normal): {cm['avg_score_normal']:.1f}")
        lines.append(f"  Avg score (judol):  {cm['avg_score_judol']:.1f}")
        lines.append(f"\n  Risk distribution:")
        for level, count in sorted(cm.get("risk_distribution", {}).items()):
            lines.append(f"    {level}: {count:,}")

    lines.append(f"\n  Total time: {elapsed:.1f}s")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_report(report: str, results: dict, tag: str):
    """Save training report (text) and metrics (JSON) to logs/."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Text report
    report_path = os.path.join(log_dir, f"training_report_{tag}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    # JSON metrics (machine-readable, for comparison)
    metrics = {}
    for key in ["user", "merchant", "network", "temporal", "velocity", "flow"]:
        if key in results:
            metrics[key] = results[key]["metrics"]
    if "_combined_metrics" in results:
        metrics["combined"] = results["_combined_metrics"]
    metrics["timings"] = results.get("_timings", {})

    json_path = os.path.join(log_dir, f"training_metrics_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics saved to {json_path}")


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
        print(f"  {'COMBINED':<25} {cm['precision']:>10.4f} {cm['recall']:>10.4f} "
              f"{cm['f1_score']:>10.4f} {total_time:>7.1f}s")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pantau ML Training Pipeline")
    parser.add_argument("--input", "-i", type=str,
                        default="data/generated/gan/pantau_gan_ctgan.csv",
                        help="Path to training dataset CSV")
    parser.add_argument("--sample", "-s", type=int, default=None,
                        help="Sample N rows for faster testing")
    parser.add_argument("--tag", "-t", type=str, default=None,
                        help="Tag name for this run (default: auto-detect from input path)")
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

    # Auto-detect tag from input path
    if not args.tag:
        if "gan" in args.input.lower():
            args.tag = "gan"
        elif "parametric" in args.input.lower():
            args.tag = "parametric"
        else:
            args.tag = os.path.splitext(os.path.basename(args.input))[0]

    start = time.time()
    df = load_dataset(args.input, sample=args.sample)
    results = train_all(df)
    elapsed = time.time() - start

    if not args.no_save:
        save_all(results, tag=args.tag)

    print_summary(results)

    # Save report and metrics
    report = build_report(results, tag=args.tag, input_path=args.input, elapsed=elapsed)
    save_report(report, results, tag=args.tag)

    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print(f"  Tag: {args.tag}")
    print("  Done! ✓")


if __name__ == "__main__":
    main()
