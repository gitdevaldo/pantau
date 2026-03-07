"""
Pantau — Dataset Quality Comparison: Parametric vs GAN
========================================================
Compares statistical distributions between the base parametric dataset
and the GAN-generated dataset to validate that GAN output preserves
key patterns required for fraud detection training.

Usage:
    python3 scripts/compare_datasets.py [--parametric PATH] [--gan PATH]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ============================================================
# DISTRIBUTION HELPERS
# ============================================================

def pct(series):
    """Value counts as percentages."""
    return (series.value_counts(normalize=True) * 100).round(2)


def kl_divergence(p, q):
    """Compute KL divergence between two discrete distributions (aligned by index)."""
    all_keys = p.index.union(q.index)
    p = p.reindex(all_keys, fill_value=1e-10)
    q = q.reindex(all_keys, fill_value=1e-10)
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ============================================================
# COMPARISON SECTIONS
# ============================================================

def compare_basic(df_p, df_g):
    """Basic shape and label distribution."""
    print("\n" + "=" * 70)
    print("  1. BASIC SHAPE & LABEL DISTRIBUTION")
    print("=" * 70)

    print(f"  {'':30} {'Parametric':>12} {'GAN':>12} {'Diff':>10}")
    print("  " + "-" * 66)

    print(f"  {'Total rows':30} {len(df_p):>12,} {len(df_g):>12,} "
          f"{len(df_g) - len(df_p):>+10,}")
    print(f"  {'Unique users':30} {df_p['user_id'].nunique():>12,} "
          f"{df_g['user_id'].nunique():>12,}")
    print(f"  {'Unique merchants':30} {df_p['merchant_id'].nunique():>12,} "
          f"{df_g['merchant_id'].nunique():>12,}")

    for label_val in [0, 1]:
        name = "Normal (label=0)" if label_val == 0 else "Judol (label=1)"
        p_pct = (df_p["label"] == label_val).mean() * 100
        g_pct = (df_g["label"] == label_val).mean() * 100
        print(f"  {name:30} {p_pct:>11.1f}% {g_pct:>11.1f}% {g_pct - p_pct:>+9.1f}%")


def compare_amounts(df_p, df_g):
    """Amount distribution comparison."""
    print("\n" + "=" * 70)
    print("  2. AMOUNT DISTRIBUTION")
    print("=" * 70)

    print(f"  {'':30} {'Parametric':>12} {'GAN':>12} {'Diff %':>10}")
    print("  " + "-" * 66)

    for stat_name, stat_fn in [("Mean", "mean"), ("Median", "median"),
                                ("Std", "std"), ("Min", "min"), ("Max", "max")]:
        p_val = getattr(df_p["amount"], stat_fn)()
        g_val = getattr(df_g["amount"], stat_fn)()
        diff_pct = (g_val - p_val) / max(abs(p_val), 1) * 100
        print(f"  {stat_name:30} {p_val:>12,.0f} {g_val:>12,.0f} {diff_pct:>+9.1f}%")

    # Per label
    for label_val, label_name in [(0, "Normal"), (1, "Judol")]:
        p_mean = df_p[df_p["label"] == label_val]["amount"].mean()
        g_mean = df_g[df_g["label"] == label_val]["amount"].mean()
        diff_pct = (g_mean - p_mean) / max(abs(p_mean), 1) * 100
        print(f"  {'Mean (' + label_name + ')':30} {p_mean:>12,.0f} {g_mean:>12,.0f} {diff_pct:>+9.1f}%")

    # Round amount rate
    p_round = df_p["is_round_amount"].mean() * 100
    g_round = df_g["is_round_amount"].mean() * 100
    print(f"  {'Round amount rate':30} {p_round:>11.1f}% {g_round:>11.1f}% {g_round - p_round:>+9.1f}%")


def compare_temporal(df_p, df_g):
    """Temporal pattern comparison."""
    print("\n" + "=" * 70)
    print("  3. TEMPORAL PATTERNS")
    print("=" * 70)

    print(f"  {'':30} {'Parametric':>12} {'GAN':>12} {'Diff':>10}")
    print("  " + "-" * 66)

    # Late night rate (20:00-02:00)
    late_hours = {20, 21, 22, 23, 0, 1, 2}
    p_late = df_p["tx_hour"].isin(late_hours).mean() * 100
    g_late = df_g["tx_hour"].isin(late_hours).mean() * 100
    print(f"  {'Late night rate (20-02)':30} {p_late:>11.1f}% {g_late:>11.1f}% {g_late - p_late:>+9.1f}%")

    # Togel hours
    togel_hours = {13, 16, 19, 22}
    p_togel = df_p["tx_hour"].isin(togel_hours).mean() * 100
    g_togel = df_g["tx_hour"].isin(togel_hours).mean() * 100
    print(f"  {'Togel hour rate (13,16,19,22)':30} {p_togel:>11.1f}% {g_togel:>11.1f}% {g_togel - p_togel:>+9.1f}%")

    # Weekend rate
    p_wkd = (df_p["tx_day_of_week"] >= 5).mean() * 100
    g_wkd = (df_g["tx_day_of_week"] >= 5).mean() * 100
    print(f"  {'Weekend rate':30} {p_wkd:>11.1f}% {g_wkd:>11.1f}% {g_wkd - p_wkd:>+9.1f}%")

    # Peak hour
    p_peak = df_p["tx_hour"].mode().iloc[0]
    g_peak = df_g["tx_hour"].mode().iloc[0]
    print(f"  {'Peak hour':30} {p_peak:>12} {g_peak:>12}")

    # Hour distribution KL divergence
    p_hours = pct(df_p["tx_hour"])
    g_hours = pct(df_g["tx_hour"])
    kl = kl_divergence(p_hours, g_hours)
    print(f"  {'Hour KL divergence':30} {'':>12} {'':>12} {kl:>9.4f}")

    # Day of week KL divergence
    p_days = pct(df_p["tx_day_of_week"])
    g_days = pct(df_g["tx_day_of_week"])
    kl = kl_divergence(p_days, g_days)
    print(f"  {'Day-of-week KL divergence':30} {'':>12} {'':>12} {kl:>9.4f}")

    # Late night rate per label
    for label_val, name in [(0, "Normal"), (1, "Judol")]:
        p_val = df_p[df_p["label"] == label_val]["tx_hour"].isin(late_hours).mean() * 100
        g_val = df_g[df_g["label"] == label_val]["tx_hour"].isin(late_hours).mean() * 100
        print(f"  {'Late night (' + name + ')':30} {p_val:>11.1f}% {g_val:>11.1f}% {g_val - p_val:>+9.1f}%")


def compare_geo(df_p, df_g):
    """Geographic distribution comparison."""
    print("\n" + "=" * 70)
    print("  4. GEOGRAPHIC DISTRIBUTION")
    print("=" * 70)

    print(f"  {'':30} {'Parametric':>12} {'GAN':>12} {'Diff':>10}")
    print("  " + "-" * 66)

    print(f"  {'Unique user provinces':30} {df_p['user_province'].nunique():>12} "
          f"{df_g['user_province'].nunique():>12}")
    print(f"  {'Unique user cities':30} {df_p['user_city'].nunique():>12} "
          f"{df_g['user_city'].nunique():>12}")
    print(f"  {'Unique merchant provinces':30} {df_p['merchant_province'].nunique():>12} "
          f"{df_g['merchant_province'].nunique():>12}")
    print(f"  {'Unique merchant cities':30} {df_p['merchant_city'].nunique():>12} "
          f"{df_g['merchant_city'].nunique():>12}")

    # Same city rate (user_city == merchant_city)
    p_same = (df_p["user_city"] == df_p["merchant_city"]).mean() * 100
    g_same = (df_g["user_city"] == df_g["merchant_city"]).mean() * 100
    print(f"  {'Same city rate':30} {p_same:>11.1f}% {g_same:>11.1f}% {g_same - p_same:>+9.1f}%")

    # Province KL divergence
    p_prov = pct(df_p["user_province"])
    g_prov = pct(df_g["user_province"])
    kl = kl_divergence(p_prov, g_prov)
    print(f"  {'User province KL divergence':30} {'':>12} {'':>12} {kl:>9.4f}")

    # Top 5 provinces
    print(f"\n  Top 5 user provinces:")
    p_top5 = pct(df_p["user_province"]).head(5)
    g_top5 = pct(df_g["user_province"]).head(5)
    all_top = p_top5.index.union(g_top5.index)[:7]
    for prov in all_top:
        p_v = p_top5.get(prov, 0)
        g_v = g_top5.get(prov, 0)
        print(f"    {prov:28} {p_v:>8.1f}%   {g_v:>8.1f}%")


def compare_transaction_type(df_p, df_g):
    """Transaction type distribution comparison."""
    print("\n" + "=" * 70)
    print("  5. TRANSACTION TYPE DISTRIBUTION")
    print("=" * 70)

    print(f"  {'':30} {'Parametric':>12} {'GAN':>12} {'Diff':>10}")
    print("  " + "-" * 66)

    p_types = pct(df_p["transaction_type"])
    g_types = pct(df_g["transaction_type"])
    all_types = p_types.index.union(g_types.index)

    for tx_type in sorted(all_types):
        p_v = p_types.get(tx_type, 0)
        g_v = g_types.get(tx_type, 0)
        print(f"  {tx_type:30} {p_v:>11.1f}% {g_v:>11.1f}% {g_v - p_v:>+9.1f}%")

    kl = kl_divergence(p_types, g_types)
    print(f"  {'Transaction type KL div':30} {'':>12} {'':>12} {kl:>9.4f}")

    # E-wallet rate overall
    p_ew = df_p["transaction_type"].str.startswith("EWALLET").mean() * 100
    g_ew = df_g["transaction_type"].str.startswith("EWALLET").mean() * 100
    print(f"  {'E-wallet rate (total)':30} {p_ew:>11.1f}% {g_ew:>11.1f}% {g_ew - p_ew:>+9.1f}%")


def compare_judol_patterns(df_p, df_g):
    """Judol-specific pattern comparison."""
    print("\n" + "=" * 70)
    print("  6. JUDOL-SPECIFIC PATTERNS")
    print("=" * 70)

    print(f"  {'':30} {'Parametric':>12} {'GAN':>12} {'Diff':>10}")
    print("  " + "-" * 66)

    for label_val, name in [(0, "Normal"), (1, "Judol")]:
        subset_p = df_p[df_p["label"] == label_val]
        subset_g = df_g[df_g["label"] == label_val]

        print(f"\n  --- {name} ---")

        # Count
        print(f"  {'Count':30} {len(subset_p):>12,} {len(subset_g):>12,}")

        # Avg amount
        p_v = subset_p["amount"].mean()
        g_v = subset_g["amount"].mean()
        diff = (g_v - p_v) / max(abs(p_v), 1) * 100
        print(f"  {'Avg amount':30} {p_v:>12,.0f} {g_v:>12,.0f} {diff:>+9.1f}%")

        # Round amount rate
        p_v = subset_p["is_round_amount"].mean() * 100
        g_v = subset_g["is_round_amount"].mean() * 100
        print(f"  {'Round amount rate':30} {p_v:>11.1f}% {g_v:>11.1f}% {g_v - p_v:>+9.1f}%")

        # E-wallet rate
        p_v = subset_p["transaction_type"].str.startswith("EWALLET").mean() * 100
        g_v = subset_g["transaction_type"].str.startswith("EWALLET").mean() * 100
        print(f"  {'E-wallet rate':30} {p_v:>11.1f}% {g_v:>11.1f}% {g_v - p_v:>+9.1f}%")

        # Late night rate
        late_hours = {20, 21, 22, 23, 0, 1, 2}
        p_v = subset_p["tx_hour"].isin(late_hours).mean() * 100
        g_v = subset_g["tx_hour"].isin(late_hours).mean() * 100
        print(f"  {'Late night rate':30} {p_v:>11.1f}% {g_v:>11.1f}% {g_v - p_v:>+9.1f}%")

        # Weekend rate
        p_v = (subset_p["tx_day_of_week"] >= 5).mean() * 100
        g_v = (subset_g["tx_day_of_week"] >= 5).mean() * 100
        print(f"  {'Weekend rate':30} {p_v:>11.1f}% {g_v:>11.1f}% {g_v - p_v:>+9.1f}%")


def quality_verdict(df_p, df_g):
    """Overall quality assessment."""
    print("\n" + "=" * 70)
    print("  7. QUALITY VERDICT")
    print("=" * 70)

    issues = []
    warnings = []

    # Label ratio check
    p_judol = (df_p["label"] == 1).mean()
    g_judol = (df_g["label"] == 1).mean()
    if abs(g_judol - p_judol) > 0.03:
        issues.append(f"Label ratio drift: {p_judol:.1%} → {g_judol:.1%}")
    elif abs(g_judol - p_judol) > 0.01:
        warnings.append(f"Minor label ratio drift: {p_judol:.1%} → {g_judol:.1%}")

    # Amount mean check
    p_mean = df_p["amount"].mean()
    g_mean = df_g["amount"].mean()
    if abs(g_mean - p_mean) / p_mean > 0.10:
        issues.append(f"Amount mean drift: {p_mean:,.0f} → {g_mean:,.0f} ({(g_mean-p_mean)/p_mean:+.1%})")
    elif abs(g_mean - p_mean) / p_mean > 0.05:
        warnings.append(f"Minor amount drift: {(g_mean-p_mean)/p_mean:+.1%}")

    # Hour distribution check
    kl = kl_divergence(pct(df_p["tx_hour"]), pct(df_g["tx_hour"]))
    if kl > 0.1:
        issues.append(f"Hour distribution diverged (KL={kl:.4f})")
    elif kl > 0.02:
        warnings.append(f"Minor hour distribution shift (KL={kl:.4f})")

    # Province coverage
    p_prov = set(df_p["user_province"].unique())
    g_prov = set(df_g["user_province"].unique())
    missing = p_prov - g_prov
    if len(missing) > 3:
        issues.append(f"Missing {len(missing)} provinces in GAN output")
    elif len(missing) > 0:
        warnings.append(f"Missing {len(missing)} provinces: {missing}")

    # City coverage
    p_cities = df_p["user_city"].nunique()
    g_cities = df_g["user_city"].nunique()
    if g_cities < p_cities * 0.8:
        issues.append(f"City coverage dropped: {p_cities} → {g_cities}")

    # Transaction type coverage
    p_types = set(df_p["transaction_type"].unique())
    g_types = set(df_g["transaction_type"].unique())
    missing_types = p_types - g_types
    if missing_types:
        issues.append(f"Missing transaction types: {missing_types}")

    # Print verdict
    if not issues and not warnings:
        print("  ✅ PASS — GAN output closely matches parametric distribution")
    elif not issues:
        print("  ⚠️  PASS WITH WARNINGS")
        for w in warnings:
            print(f"    ⚠️  {w}")
    else:
        print("  ❌ ISSUES FOUND")
        for i in issues:
            print(f"    ❌ {i}")
        for w in warnings:
            print(f"    ⚠️  {w}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

class TeeOutput:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def main():
    parser = argparse.ArgumentParser(description="Compare Parametric vs GAN datasets")
    parser.add_argument("--parametric", "-p", type=str,
                        default="data/generated/parametric/pantau_dataset.csv",
                        help="Path to parametric dataset")
    parser.add_argument("--gan", "-g", type=str,
                        default="data/generated/gan/pantau_gan_ctgan.csv",
                        help="Path to GAN dataset")
    parser.add_argument("--output", "-o", type=str,
                        default="logs/comparison_report.txt",
                        help="Path to save report")
    args = parser.parse_args()

    if not os.path.exists(args.parametric):
        print(f"ERROR: Parametric dataset not found: {args.parametric}")
        sys.exit(1)
    if not os.path.exists(args.gan):
        print(f"ERROR: GAN dataset not found: {args.gan}")
        sys.exit(1)

    # Tee output to both terminal and file
    tee = TeeOutput(args.output)
    sys.stdout = tee

    print("=" * 70)
    print("  PANTAU — Dataset Quality Comparison")
    print("  Parametric vs GAN")
    print("=" * 70)

    print(f"\n  Loading parametric: {args.parametric}")
    df_p = pd.read_csv(args.parametric)
    df_p["is_round_amount"] = df_p["is_round_amount"].astype(bool)

    print(f"  Loading GAN:        {args.gan}")
    df_g = pd.read_csv(args.gan)
    df_g["is_round_amount"] = df_g["is_round_amount"].astype(bool)

    compare_basic(df_p, df_g)
    compare_amounts(df_p, df_g)
    compare_temporal(df_p, df_g)
    compare_geo(df_p, df_g)
    compare_transaction_type(df_p, df_g)
    compare_judol_patterns(df_p, df_g)
    quality_verdict(df_p, df_g)

    print(f"\n  Report saved to: {args.output}")
    print("  Done! ✓")

    tee.close()


if __name__ == "__main__":
    main()
