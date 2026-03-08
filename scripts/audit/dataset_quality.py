"""
Dataset Quality Audit — Statistical Tests for Synthetic Fraud Data
===================================================================
Measures whether a synthetic dataset is realistic or artificially separable.
Run AFTER generating data, BEFORE training, to catch quality issues early.

5 research-backed tests:
  1. Silhouette Score (Rousseeuw 1987)
  2. Baseline Model Performance (LR, Decision Stump, DT depth=3)
  3. Feature-Label Mutual Information + Information Value (Shannon 1948; Siddiqi 2006)
  4. Distribution Overlap — Bhattacharyya Coefficient (Bhattacharyya 1943)
  5. Borderline Ratio — N1 metric (Ho & Basu 2002, IEEE TPAMI)

Usage:
    python3 scripts/audit/dataset_quality.py --input data/generated/parametric/pantau_dataset.csv
    python3 scripts/audit/dataset_quality.py --input data/generated/gan/pantau_gan_ctgan.csv --output logs/audit_gan.txt
"""

import argparse
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

SEED = 42
FEATURES = ["amount", "log_amount", "is_round", "is_nighttime", "is_payday",
            "hour", "day_of_month", "day_of_week"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used across all tests."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_month"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_round"] = df["is_round_amount"].astype(int)
    df["is_nighttime"] = df["hour"].apply(lambda h: 1 if (h >= 20 or h < 2) else 0)
    df["is_payday"] = df["day_of_month"].apply(lambda d: 1 if (d <= 1 or d >= 25) else 0)
    df["log_amount"] = np.log1p(df["amount"])
    return df


def build_aggregates(df: pd.DataFrame):
    """Build user-level and merchant-level aggregate features."""
    user_agg = df.groupby("user_id").agg(
        tx_count=("amount", "count"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        round_ratio=("is_round", "mean"),
        night_ratio=("is_nighttime", "mean"),
        payday_ratio=("is_payday", "mean"),
        unique_merchants=("merchant_id", "nunique"),
    ).fillna(0)
    user_agg["label"] = df.groupby("user_id")["label"].max()

    merch_agg = df.groupby("merchant_id").agg(
        tx_count=("amount", "count"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        round_ratio=("is_round", "mean"),
        night_ratio=("is_nighttime", "mean"),
        payday_ratio=("is_payday", "mean"),
        unique_users=("user_id", "nunique"),
    ).fillna(0)
    merch_agg["label"] = df.groupby("merchant_id")["label"].max()

    return user_agg, merch_agg


# ============================================================
# TEST 1: Silhouette Score
# ============================================================
def test_silhouette(df, user_agg, merch_agg):
    print("\n" + "=" * 70)
    print("  TEST 1: Silhouette Score (Class Separability)")
    print("  Reference: Rousseeuw (1987), J. Computational & Applied Mathematics")
    print("=" * 70)

    np.random.seed(SEED)
    idx = np.random.choice(len(df), size=min(10000, len(df)), replace=False)
    tx_scaled = StandardScaler().fit_transform(df[FEATURES].iloc[idx].values)
    sil_tx = silhouette_score(tx_scaled, df["label"].iloc[idx].values,
                              sample_size=5000, random_state=SEED)

    user_scaled = StandardScaler().fit_transform(user_agg.drop(columns=["label"]).values)
    sil_user = silhouette_score(user_scaled, user_agg["label"].values,
                                sample_size=min(5000, len(user_agg)), random_state=SEED)

    merch_scaled = StandardScaler().fit_transform(merch_agg.drop(columns=["label"]).values)
    sil_merch = silhouette_score(merch_scaled, merch_agg["label"].values, random_state=SEED)

    print(f"\n  {'Level':<20} {'Score':>10} {'Verdict':>20}")
    print(f"  {'-'*50}")
    for name, score in [("Transaction", sil_tx), ("User", sil_user), ("Merchant", sil_merch)]:
        if score < 0.2:
            verdict = "✅ Realistic"
        elif score < 0.5:
            verdict = "⚠️ Moderate"
        else:
            verdict = "🔴 Too clean"
        print(f"  {name:<20} {score:>10.4f} {verdict:>20}")

    print(f"\n  Scale: 0.0-0.2 realistic | 0.2-0.5 moderate | 0.5-1.0 artificial")
    return {"tx": sil_tx, "user": sil_user, "merchant": sil_merch}


# ============================================================
# TEST 2: Baseline Models
# ============================================================
def test_baselines(df, user_agg, merch_agg):
    print("\n" + "=" * 70)
    print("  TEST 2: Baseline Model Performance (Can a dumb model solve it?)")
    print("  Method: Logistic Regression, Decision Stump (d=1), Decision Tree (d=3)")
    print("=" * 70)

    results = {}

    # Transaction-level
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES].values, df["label"].values,
        test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_tr_s, y_tr)
    lr_pred = lr.predict(X_te_s)
    lr_f1 = f1_score(y_te, lr_pred)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:, 1])

    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED, class_weight="balanced")
    stump.fit(X_tr, y_tr)
    st_f1 = f1_score(y_te, stump.predict(X_te))
    st_feat = FEATURES[stump.tree_.feature[0]]

    dt3 = DecisionTreeClassifier(max_depth=3, random_state=SEED, class_weight="balanced")
    dt3.fit(X_tr, y_tr)
    dt3_f1 = f1_score(y_te, dt3.predict(X_te))

    print(f"\n  Transaction-level:")
    print(f"  {'Model':<25} {'F1':>10} {'AUC-ROC':>10} {'Verdict':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Logistic Regression':<25} {lr_f1:>10.4f} {lr_auc:>10.4f} {'✅ OK' if lr_f1 < 0.80 else '🔴 Too easy':>15}")
    print(f"  {'Stump (d=1)':<25} {st_f1:>10.4f} {'N/A':>10} {'✅ OK' if st_f1 < 0.70 else '🔴 Too easy':>15}")
    print(f"  {'Tree (d=3)':<25} {dt3_f1:>10.4f} {'N/A':>10} {'✅ OK' if dt3_f1 < 0.80 else '🔴 Too easy':>15}")
    print(f"  Stump split feature: '{st_feat}'")

    # User-level
    for name, agg in [("User", user_agg), ("Merchant", merch_agg)]:
        feats = agg.drop(columns=["label"]).values
        labs = agg["label"].values
        Xa, Xb, ya, yb = train_test_split(feats, labs, test_size=0.2, random_state=SEED, stratify=labs)

        lr_a = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr_a.fit(StandardScaler().fit_transform(Xa), ya)
        la_f1 = f1_score(yb, lr_a.predict(StandardScaler().fit_transform(Xb)))

        st_a = DecisionTreeClassifier(max_depth=1, random_state=SEED, class_weight="balanced")
        st_a.fit(Xa, ya)
        sa_f1 = f1_score(yb, st_a.predict(Xb))
        sa_feat = agg.drop(columns=["label"]).columns[st_a.tree_.feature[0]]

        verdict_lr = "✅ OK" if la_f1 < 0.85 else "🔴 Too easy"
        verdict_st = "✅ OK" if sa_f1 < 0.70 else "🔴 Too easy"
        print(f"\n  {name}-level: LR F1={la_f1:.4f} {verdict_lr}, Stump F1={sa_f1:.4f} {verdict_st} (split: '{sa_feat}')")

    print(f"\n  Targets: TX LR F1 < 0.80 | Stump F1 < 0.70 | User/Merchant LR < 0.85")
    return {"tx_lr_f1": lr_f1, "tx_stump_f1": st_f1, "tx_lr_auc": lr_auc}


# ============================================================
# TEST 3: Mutual Information + Information Value
# ============================================================
def compute_iv(feature, label, bins=20):
    """Weight of Evidence / Information Value."""
    if feature.nunique() <= 2:
        groups = feature
    else:
        groups = pd.qcut(feature, q=bins, duplicates="drop")

    grouped = pd.DataFrame({"feature": groups, "label": label})
    agg = grouped.groupby("feature")["label"].agg(["sum", "count"])
    agg.columns = ["events", "total"]
    agg["non_events"] = agg["total"] - agg["events"]

    total_events = max(agg["events"].sum(), 1)
    total_non_events = max(agg["non_events"].sum(), 1)

    agg["event_rate"] = (agg["events"] / total_events).clip(lower=1e-6)
    agg["non_event_rate"] = (agg["non_events"] / total_non_events).clip(lower=1e-6)
    agg["woe"] = np.log(agg["non_event_rate"] / agg["event_rate"])
    agg["iv"] = (agg["non_event_rate"] - agg["event_rate"]) * agg["woe"]

    return agg["iv"].sum()


def test_feature_leakage(df):
    print("\n" + "=" * 70)
    print("  TEST 3: Feature-Label Leakage (MI + Information Value)")
    print("  References: Shannon (1948); Siddiqi (2006) Credit Risk Scorecards")
    print("=" * 70)

    np.random.seed(SEED)
    idx = np.random.choice(len(df), size=min(50000, len(df)), replace=False)
    X_sample = df[FEATURES].iloc[idx].values
    y_sample = df["label"].iloc[idx].values

    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=SEED, n_neighbors=5)

    print(f"\n  {'Feature':<20} {'MI':>8} {'IV':>8} {'Verdict':>15}")
    print(f"  {'-'*55}")

    for i, feat in enumerate(FEATURES):
        iv = compute_iv(df[feat].iloc[idx], df["label"].iloc[idx])
        mi = mi_scores[i]
        if iv > 0.5:
            verdict = "🔴 SUSPICIOUS"
        elif iv > 0.3:
            verdict = "⚠️ Strong"
        elif iv > 0.1:
            verdict = "Medium"
        else:
            verdict = "✅ Weak"
        print(f"  {feat:<20} {mi:>8.4f} {iv:>8.4f} {verdict:>15}")

    print(f"\n  IV scale: <0.02 useless | 0.02-0.10 weak | 0.10-0.30 medium | 0.30-0.50 strong | >0.50 suspicious")


# ============================================================
# TEST 4: Distribution Overlap (Bhattacharyya)
# ============================================================
def bhattacharyya_overlap(f0, f1, bins=50):
    """Bhattacharyya coefficient: 0=no overlap, 1=identical distributions."""
    range_min = min(f0.min(), f1.min())
    range_max = max(f0.max(), f1.max())

    hist0, _ = np.histogram(f0, bins=bins, range=(range_min, range_max), density=True)
    hist1, _ = np.histogram(f1, bins=bins, range=(range_min, range_max), density=True)

    hist0 = hist0 / (hist0.sum() + 1e-10)
    hist1 = hist1 / (hist1.sum() + 1e-10)

    return np.sum(np.sqrt(hist0 * hist1))


def test_distribution_overlap(df):
    print("\n" + "=" * 70)
    print("  TEST 4: Distribution Overlap (Bhattacharyya Coefficient)")
    print("  Reference: Bhattacharyya (1943), Bull. Calcutta Mathematical Society")
    print("=" * 70)

    normal = df[df["label"] == 0]
    judol = df[df["label"] == 1]

    print(f"\n  {'Feature':<20} {'Overlap':>10} {'Verdict':>25}")
    print(f"  {'-'*55}")

    for feat in FEATURES:
        bc = bhattacharyya_overlap(normal[feat].values, judol[feat].values)
        if bc > 0.85:
            verdict = "✅ High overlap (good)"
        elif bc > 0.5:
            verdict = "⚠️ Moderate overlap"
        elif bc > 0.2:
            verdict = "🔴 Low overlap (bad)"
        else:
            verdict = "🔴 No overlap (fake)"
        print(f"  {feat:<20} {bc:>10.4f} {verdict:>25}")

    print(f"\n  Scale: >0.85 realistic | 0.50-0.85 acceptable | <0.50 concerning | <0.20 fake")


# ============================================================
# TEST 5: Borderline Ratio (N1)
# ============================================================
def test_borderline_ratio(df):
    print("\n" + "=" * 70)
    print("  TEST 5: Borderline Ratio (N1 — Nearest Neighbor Boundary)")
    print("  Reference: Ho & Basu (2002), IEEE TPAMI; Lorena et al. (2019)")
    print("=" * 70)

    np.random.seed(SEED)
    idx = np.random.choice(len(df), size=min(20000, len(df)), replace=False)
    X = StandardScaler().fit_transform(df[FEATURES].iloc[idx].values)
    y = df["label"].iloc[idx].values

    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    nearest_labels = y[indices[:, 1]]
    borderline = (nearest_labels != y)
    overall = borderline.mean()
    normal_bl = borderline[y == 0].mean()
    judol_bl = borderline[y == 1].mean()

    if overall < 0.10:
        verdict = "🔴 Too clean"
    elif overall < 0.20:
        verdict = "⚠️ Moderate"
    elif overall < 0.35:
        verdict = "✅ Realistic"
    else:
        verdict = "⚠️ Very noisy"

    print(f"\n  Overall:  {overall:.4f} ({overall*100:.1f}%) — {verdict}")
    print(f"  Normal:   {normal_bl:.4f} ({normal_bl*100:.1f}%)")
    print(f"  Judol:    {judol_bl:.4f} ({judol_bl*100:.1f}%)")
    print(f"\n  Scale: <10% too clean | 10-20% moderate | 20-35% realistic | >35% noisy")

    return {"overall": overall, "normal": normal_bl, "judol": judol_bl}


# ============================================================
# SUMMARY
# ============================================================
def print_summary(sil, baselines, borderline):
    print("\n" + "=" * 70)
    print("  OVERALL DATASET QUALITY VERDICT")
    print("=" * 70)

    issues = 0
    checks = []

    if sil["merchant"] > 0.5:
        checks.append(f"  🔴 Merchant silhouette = {sil['merchant']:.2f} (too clean)")
        issues += 1
    if sil["user"] > 0.5:
        checks.append(f"  🔴 User silhouette = {sil['user']:.2f} (too clean)")
        issues += 1
    if baselines["tx_stump_f1"] > 0.70:
        checks.append(f"  🔴 Stump F1 = {baselines['tx_stump_f1']:.2f} (single feature solves it)")
        issues += 1
    if baselines["tx_lr_f1"] > 0.90:
        checks.append(f"  🔴 LR F1 = {baselines['tx_lr_f1']:.2f} (trivially separable)")
        issues += 1
    if borderline["overall"] < 0.10:
        checks.append(f"  🔴 Borderline ratio = {borderline['overall']*100:.1f}% (too few boundary cases)")
        issues += 1

    if sil["tx"] < 0.2:
        checks.append(f"  ✅ Transaction silhouette = {sil['tx']:.2f} (realistic)")
    if baselines["tx_lr_f1"] < 0.80:
        checks.append(f"  ✅ LR F1 = {baselines['tx_lr_f1']:.2f} (reasonable difficulty)")
    if borderline["overall"] >= 0.20:
        checks.append(f"  ✅ Borderline = {borderline['overall']*100:.1f}% (realistic)")

    for c in checks:
        print(c)

    if issues == 0:
        print(f"\n  VERDICT: ✅ Dataset appears realistic")
    elif issues <= 2:
        print(f"\n  VERDICT: ⚠️ Dataset has {issues} quality issue(s) — consider improving generator")
    else:
        print(f"\n  VERDICT: 🔴 Dataset has {issues} quality issues — generator needs redesign")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Dataset Quality Audit for Synthetic Fraud Data")
    parser.add_argument("--input", "-i", required=True, help="Path to dataset CSV")
    parser.add_argument("--output", "-o", help="Save output to file (optional)")
    args = parser.parse_args()

    # Redirect stdout if output file specified
    original_stdout = sys.stdout
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        # Tee to both console and file
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, text):
                for f in self.files:
                    f.write(text)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        sys.stdout = Tee(original_stdout, output_file)

    print("=" * 70)
    print("  PANTAU — Dataset Quality Audit")
    print("=" * 70)

    df = pd.read_csv(args.input)
    print(f"\n  Input: {args.input}")
    print(f"  Rows: {len(df):,}")
    print(f"  Fraud rate: {df['label'].mean()*100:.1f}%")
    print(f"  Users: {df['user_id'].nunique():,}")
    print(f"  Merchants: {df['merchant_id'].nunique():,}")

    df = engineer_features(df)
    user_agg, merch_agg = build_aggregates(df)

    sil = test_silhouette(df, user_agg, merch_agg)
    baselines = test_baselines(df, user_agg, merch_agg)
    test_feature_leakage(df)
    test_distribution_overlap(df)
    borderline = test_borderline_ratio(df)
    print_summary(sil, baselines, borderline)

    if output_file:
        output_file.close()
        sys.stdout = original_stdout
        print(f"\n  Results also saved to {args.output}")


if __name__ == "__main__":
    main()
