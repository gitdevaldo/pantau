"""
Dataset Quality Audit — Statistical Tests for Synthetic Fraud Data
===================================================================
Measures whether a synthetic dataset is realistic or artificially separable.
Run AFTER generating data, BEFORE training, to catch quality issues early.

9 research-backed tests covering all 6 ML layers:
  1. Silhouette Score (Rousseeuw 1987) — Layer 1 & 2
  2. Baseline Model Performance (LR, Decision Stump, DT depth=3) — Layer 1 & 2
  3. Feature-Label Mutual Information + Information Value (Shannon 1948; Siddiqi 2006)
  4. Distribution Overlap — Bhattacharyya Coefficient (Bhattacharyya 1943)
  5. Borderline Ratio — N1 metric (Ho & Basu 2002, IEEE TPAMI)
  6. Network/Graph Signal — fan-in, shared senders, geo diversity (Layer 3)
  7. Temporal Pattern Signal — burst detection & session clustering (Layer 4)
  8. Velocity/Delta Signal — cross-merchant rate anomalies (Layer 5)
  9. Money Flow Signal — fan-in concentration analysis (Layer 6)

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
# TEST 6: Network/Graph Signal (Layer 3)
# ============================================================
def test_network_signal(df):
    print("\n" + "=" * 70)
    print("  TEST 6: Network/Graph Signal (Layer 3 — Cluster Detection)")
    print("  Checks: graph structure, shared-user clusters, fan-in concentration")
    print("=" * 70)

    try:
        import networkx as nx
    except ImportError:
        print("\n  ⚠️ networkx not installed — skipping network test")
        return {}

    # Build bipartite user→merchant graph
    np.random.seed(SEED)
    sample = df.sample(n=min(100000, len(df)), random_state=SEED)
    G = nx.Graph()
    for _, row in sample.iterrows():
        G.add_edge(row["user_id"], row["merchant_id"])

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Connected components
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    cc_ratio = len(largest_cc) / n_nodes

    judol_merchants = set(df[df["label"] == 1]["merchant_id"].unique())
    normal_merchants = set(df[df["label"] == 0]["merchant_id"].unique()) - judol_merchants

    # Fan-in: unique users per merchant (the actual signal for Layer 3)
    deg_judol = [G.degree(m) for m in judol_merchants if m in G]
    deg_normal = [G.degree(m) for m in normal_merchants if m in G]
    mean_deg_judol = np.mean(deg_judol) if deg_judol else 0
    mean_deg_normal = np.mean(deg_normal) if deg_normal else 0

    # Shared senders: how many merchants share users with judol vs normal merchants
    def shared_sender_count(merchant_set):
        counts = []
        for m in list(merchant_set)[:200]:
            if m not in G:
                continue
            neighbors = set(G.neighbors(m))  # users of this merchant
            shared = 0
            for user in neighbors:
                shared += G.degree(user) - 1  # other merchants this user visits
            counts.append(shared / max(len(neighbors), 1))
        return np.mean(counts) if counts else 0

    shared_judol = shared_sender_count(judol_merchants)
    shared_normal = shared_sender_count(normal_merchants)

    # Geographic diversity: unique cities of senders per merchant
    geo_judol = sample[sample["merchant_id"].isin(judol_merchants)].groupby("merchant_id")["city"].nunique()
    geo_normal = sample[sample["merchant_id"].isin(normal_merchants)].groupby("merchant_id")["city"].nunique()
    mean_geo_judol = geo_judol.mean() if len(geo_judol) > 0 else 0
    mean_geo_normal = geo_normal.mean() if len(geo_normal) > 0 else 0

    print(f"\n  Graph: {n_nodes:,} nodes, {n_edges:,} edges")
    print(f"  Largest component: {len(largest_cc):,} nodes ({cc_ratio*100:.1f}%)")
    print(f"  Connected components: {len(components):,}")

    print(f"\n  {'Metric':<35} {'Judol':>10} {'Normal':>10} {'Verdict':>15}")
    print(f"  {'-'*70}")

    # Fan-in (unique users)
    deg_ratio = mean_deg_judol / (mean_deg_normal + 1e-10)
    if deg_ratio > 1.5:
        v_deg = "✅ Signal exists"
    elif deg_ratio > 1.1:
        v_deg = "⚠️ Weak signal"
    else:
        v_deg = "🔴 No signal"
    print(f"  {'Fan-in (unique users)':<35} {mean_deg_judol:>10.1f} {mean_deg_normal:>10.1f} {v_deg:>15}")

    # Shared senders
    shared_ratio = shared_judol / (shared_normal + 1e-10)
    if shared_ratio > 1.3:
        v_shared = "✅ Signal exists"
    elif shared_ratio > 1.05:
        v_shared = "⚠️ Weak signal"
    else:
        v_shared = "🔴 No signal"
    print(f"  {'Avg shared senders/user':<35} {shared_judol:>10.1f} {shared_normal:>10.1f} {v_shared:>15}")

    # Geographic diversity
    geo_ratio = mean_geo_judol / (mean_geo_normal + 1e-10)
    if geo_ratio > 1.5:
        v_geo = "✅ Signal exists"
    elif geo_ratio > 1.1:
        v_geo = "⚠️ Weak signal"
    else:
        v_geo = "🔴 No signal"
    print(f"  {'Sender city diversity':<35} {mean_geo_judol:>10.1f} {mean_geo_normal:>10.1f} {v_geo:>15}")

    # Connectivity
    if cc_ratio > 0.90:
        v_cc = "✅ Connected"
    elif cc_ratio > 0.50:
        v_cc = "⚠️ Fragmented"
    else:
        v_cc = "🔴 Disconnected"
    print(f"  {'Largest component coverage':<35} {cc_ratio*100:>10.1f}% {'':>10} {v_cc:>15}")

    return {"deg_ratio": deg_ratio, "shared_ratio": shared_ratio,
            "geo_ratio": geo_ratio, "cc_ratio": cc_ratio}


# ============================================================
# TEST 7: Temporal Pattern Signal (Layer 4)
# ============================================================
def test_temporal_signal(df):
    print("\n" + "=" * 70)
    print("  TEST 7: Temporal Pattern Signal (Layer 4 — Timing & Bursts)")
    print("  Checks: session clustering, rapid-fire bursts, hour entropy")
    print("=" * 70)

    judol_users = df[df["label"] == 1]["user_id"].unique()
    normal_only_users = set(df["user_id"].unique()) - set(judol_users)

    results = {}

    # Per-user burst detection: transactions within 30 min of each other
    def calc_burst_ratio(user_df):
        if len(user_df) < 3:
            return 0.0
        ts = user_df["timestamp"].sort_values()
        gaps = ts.diff().dt.total_seconds().dropna()
        bursts = (gaps <= 1800).sum()  # within 30 min
        return bursts / len(gaps) if len(gaps) > 0 else 0.0

    # Sample users for efficiency
    np.random.seed(SEED)
    sample_judol = np.random.choice(judol_users, size=min(500, len(judol_users)), replace=False)
    sample_normal = np.random.choice(list(normal_only_users),
                                      size=min(2000, len(normal_only_users)), replace=False)

    judol_txs = df[df["user_id"].isin(sample_judol)]
    normal_txs = df[df["user_id"].isin(sample_normal)]

    # Burst ratios (label=1 transactions only for judol users)
    judol_label1 = judol_txs[judol_txs["label"] == 1]
    burst_judol = judol_label1.groupby("user_id").apply(calc_burst_ratio)
    burst_normal = normal_txs.groupby("user_id").apply(calc_burst_ratio)

    mean_burst_judol = burst_judol.mean() if len(burst_judol) > 0 else 0
    mean_burst_normal = burst_normal.mean() if len(burst_normal) > 0 else 0

    # Hour entropy (how spread out are transactions across hours)
    def hour_entropy(user_df):
        counts = user_df["hour"].value_counts(normalize=True)
        return -(counts * np.log2(counts + 1e-10)).sum()

    ent_judol = judol_label1.groupby("user_id").apply(hour_entropy)
    ent_normal = normal_txs.groupby("user_id").apply(hour_entropy)
    mean_ent_judol = ent_judol.mean() if len(ent_judol) > 0 else 0
    mean_ent_normal = ent_normal.mean() if len(ent_normal) > 0 else 0

    # Night ratio comparison (20:00-02:00)
    night_judol = judol_label1["hour"].apply(lambda h: h >= 20 or h < 2).mean()
    night_normal = normal_txs["hour"].apply(lambda h: h >= 20 or h < 2).mean()

    # Togel timing correlation (13, 16, 19, 22)
    togel_hours = {13, 16, 19, 22}
    togel_judol = judol_label1["hour"].isin(togel_hours).mean()
    togel_normal = normal_txs["hour"].isin(togel_hours).mean()

    print(f"\n  {'Metric':<35} {'Judol':>10} {'Normal':>10} {'Verdict':>15}")
    print(f"  {'-'*70}")

    # Burst ratio
    if mean_burst_judol > mean_burst_normal * 1.5:
        v_burst = "✅ Signal exists"
    elif mean_burst_judol > mean_burst_normal * 1.1:
        v_burst = "⚠️ Weak signal"
    else:
        v_burst = "🔴 No signal"
    print(f"  {'Burst ratio (<30min gaps)':<35} {mean_burst_judol:>10.3f} {mean_burst_normal:>10.3f} {v_burst:>15}")

    # Hour entropy
    if abs(mean_ent_judol - mean_ent_normal) > 0.3:
        v_ent = "✅ Signal exists"
    elif abs(mean_ent_judol - mean_ent_normal) > 0.1:
        v_ent = "⚠️ Weak signal"
    else:
        v_ent = "🔴 No signal"
    print(f"  {'Hour entropy (bits)':<35} {mean_ent_judol:>10.2f} {mean_ent_normal:>10.2f} {v_ent:>15}")

    # Night ratio
    night_diff = night_judol - night_normal
    if night_diff > 0.10:
        v_night = "✅ Signal exists"
    elif night_diff > 0.03:
        v_night = "⚠️ Weak signal"
    else:
        v_night = "🔴 No signal"
    print(f"  {'Night ratio (20-02h)':<35} {night_judol:>10.3f} {night_normal:>10.3f} {v_night:>15}")

    # Togel timing
    togel_ratio = togel_judol / (togel_normal + 1e-10)
    if togel_ratio > 1.3:
        v_togel = "✅ Signal exists"
    elif togel_ratio > 1.05:
        v_togel = "⚠️ Weak signal"
    else:
        v_togel = "🔴 No signal"
    print(f"  {'Togel hours (13,16,19,22)':<35} {togel_judol:>10.3f} {togel_normal:>10.3f} {v_togel:>15}")

    return {"burst_ratio_j": mean_burst_judol, "burst_ratio_n": mean_burst_normal,
            "night_diff": night_diff}


# ============================================================
# TEST 8: Velocity/Delta Signal (Layer 5)
# ============================================================
def test_velocity_signal(df):
    print("\n" + "=" * 70)
    print("  TEST 8: Velocity/Delta Signal (Layer 5 — Rate Anomalies)")
    print("  Checks: tx velocity per merchant, inter-tx time gaps, amount variance")
    print("=" * 70)

    judol_merchants = set(df[df["label"] == 1]["merchant_id"].unique())
    normal_merchants = set(df["merchant_id"].unique()) - judol_merchants

    # Per-merchant velocity: tx per day
    date_range_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1

    merch_counts = df.groupby("merchant_id").agg(
        tx_count=("amount", "count"),
        unique_users=("user_id", "nunique"),
        std_amount=("amount", "std"),
        mean_gap=("timestamp", lambda x: x.sort_values().diff().dt.total_seconds().mean()
                  if len(x) > 1 else float("nan")),
    ).fillna(0)

    merch_counts["tx_per_day"] = merch_counts["tx_count"] / date_range_days
    merch_counts["is_judol"] = merch_counts.index.isin(judol_merchants).astype(int)

    judol_m = merch_counts[merch_counts["is_judol"] == 1]
    normal_m = merch_counts[merch_counts["is_judol"] == 0]

    # Velocity (tx/day)
    vel_judol = judol_m["tx_per_day"].mean()
    vel_normal = normal_m["tx_per_day"].mean()

    # Mean inter-tx gap (seconds)
    gap_judol = judol_m["mean_gap"].replace(0, np.nan).mean()
    gap_normal = normal_m["mean_gap"].replace(0, np.nan).mean()

    # Amount std (variance in transaction amounts)
    std_judol = judol_m["std_amount"].mean()
    std_normal = normal_m["std_amount"].mean()

    # Users per merchant
    upm_judol = judol_m["unique_users"].mean()
    upm_normal = normal_m["unique_users"].mean()

    print(f"\n  {'Metric':<35} {'Judol':>12} {'Normal':>12} {'Verdict':>15}")
    print(f"  {'-'*74}")

    # Velocity
    vel_ratio = vel_judol / (vel_normal + 1e-10)
    v_vel = "✅ Signal exists" if vel_ratio > 1.3 else ("⚠️ Weak signal" if vel_ratio > 1.05 else "🔴 No signal")
    print(f"  {'Tx/day per merchant':<35} {vel_judol:>12.2f} {vel_normal:>12.2f} {v_vel:>15}")

    # Gap
    if gap_judol > 0 and gap_normal > 0:
        gap_ratio = gap_normal / (gap_judol + 1e-10)
        v_gap = "✅ Signal exists" if gap_ratio > 1.3 else ("⚠️ Weak signal" if gap_ratio > 1.05 else "🔴 No signal")
        print(f"  {'Mean inter-tx gap (sec)':<35} {gap_judol:>12.0f} {gap_normal:>12.0f} {v_gap:>15}")

    # Amount std
    std_ratio = std_judol / (std_normal + 1e-10)
    v_std = "✅ Signal exists" if abs(std_ratio - 1) > 0.2 else ("⚠️ Weak signal" if abs(std_ratio - 1) > 0.05 else "🔴 No signal")
    print(f"  {'Amount std dev':<35} {std_judol:>12.0f} {std_normal:>12.0f} {v_std:>15}")

    # Users per merchant
    upm_ratio = upm_judol / (upm_normal + 1e-10)
    v_upm = "✅ Signal exists" if upm_ratio > 1.3 else ("⚠️ Weak signal" if upm_ratio > 1.05 else "🔴 No signal")
    print(f"  {'Unique users/merchant':<35} {upm_judol:>12.1f} {upm_normal:>12.1f} {v_upm:>15}")

    return {"vel_ratio": vel_ratio, "std_ratio": std_ratio, "upm_ratio": upm_ratio}


# ============================================================
# TEST 9: Money Flow Signal (Layer 6)
# ============================================================
def test_money_flow(df):
    print("\n" + "=" * 70)
    print("  TEST 9: Money Flow Signal (Layer 6 — Fan-in Analysis)")
    print("  Checks: inflow concentration, user fan-out, flow asymmetry")
    print("=" * 70)

    judol_merchants = set(df[df["label"] == 1]["merchant_id"].unique())
    normal_merchants = set(df["merchant_id"].unique()) - judol_merchants

    # Fan-in: total money flowing INTO each merchant
    merchant_inflow = df.groupby("merchant_id").agg(
        total_inflow=("amount", "sum"),
        tx_count=("amount", "count"),
        unique_senders=("user_id", "nunique"),
        mean_amount=("amount", "mean"),
    )
    merchant_inflow["is_judol"] = merchant_inflow.index.isin(judol_merchants).astype(int)

    judol_flow = merchant_inflow[merchant_inflow["is_judol"] == 1]
    normal_flow = merchant_inflow[merchant_inflow["is_judol"] == 0]

    # Concentration: top-user share of merchant's inflow
    def top_user_share(merchant_df):
        if len(merchant_df) == 0:
            return 0.0
        user_totals = merchant_df.groupby("user_id")["amount"].sum()
        if len(user_totals) == 0:
            return 0.0
        return user_totals.max() / (user_totals.sum() + 1e-10)

    np.random.seed(SEED)
    sample_judol_m = np.random.choice(list(judol_merchants),
                                       size=min(200, len(judol_merchants)), replace=False)
    sample_normal_m = np.random.choice(list(normal_merchants),
                                        size=min(1000, len(normal_merchants)), replace=False)

    conc_judol = [top_user_share(df[df["merchant_id"] == m]) for m in sample_judol_m]
    conc_normal = [top_user_share(df[df["merchant_id"] == m]) for m in sample_normal_m]

    mean_conc_judol = np.mean(conc_judol) if conc_judol else 0
    mean_conc_normal = np.mean(conc_normal) if conc_normal else 0

    # Fan-out: how many unique merchants does each user send to
    judol_users = set(df[df["label"] == 1]["user_id"].unique())
    normal_only_users = set(df["user_id"].unique()) - judol_users

    user_fanout = df.groupby("user_id")["merchant_id"].nunique()
    fanout_judol = user_fanout[user_fanout.index.isin(judol_users)].mean()
    fanout_normal = user_fanout[user_fanout.index.isin(normal_only_users)].mean()

    # Mean inflow per merchant
    inflow_judol = judol_flow["total_inflow"].mean()
    inflow_normal = normal_flow["total_inflow"].mean()

    print(f"\n  {'Metric':<35} {'Judol':>12} {'Normal':>12} {'Verdict':>15}")
    print(f"  {'-'*74}")

    # Top-user concentration
    conc_diff = mean_conc_judol - mean_conc_normal
    v_conc = "✅ Signal exists" if conc_diff > 0.05 else ("⚠️ Weak signal" if conc_diff > 0.01 else "🔴 No signal")
    print(f"  {'Top-user inflow share':<35} {mean_conc_judol:>12.3f} {mean_conc_normal:>12.3f} {v_conc:>15}")

    # Fan-out
    fanout_ratio = fanout_judol / (fanout_normal + 1e-10)
    v_fan = "✅ Signal exists" if fanout_ratio > 1.2 else ("⚠️ Weak signal" if fanout_ratio > 1.05 else "🔴 No signal")
    print(f"  {'User merchant fan-out':<35} {fanout_judol:>12.1f} {fanout_normal:>12.1f} {v_fan:>15}")

    # Total inflow
    inflow_ratio = inflow_judol / (inflow_normal + 1e-10)
    v_inflow = "✅ Signal exists" if inflow_ratio > 1.3 else ("⚠️ Weak signal" if inflow_ratio > 1.05 else "🔴 No signal")
    print(f"  {'Mean total inflow (Rp)':<35} {inflow_judol:>12,.0f} {inflow_normal:>12,.0f} {v_inflow:>15}")

    return {"conc_diff": conc_diff, "fanout_ratio": fanout_ratio, "inflow_ratio": inflow_ratio}


# ============================================================
# SUMMARY
# ============================================================
def print_summary(sil, baselines, borderline, network, temporal, velocity, money_flow):
    print("\n" + "=" * 70)
    print("  OVERALL DATASET QUALITY VERDICT")
    print("=" * 70)

    issues = 0
    checks = []

    # Separability checks (should NOT be too easy)
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

    # Signal checks (should HAVE signal for ML layers)
    signal_count = 0
    signal_total = 0

    if network:
        signal_total += 2
        if network.get("deg_ratio", 0) > 1.1:
            signal_count += 1
        if network.get("geo_ratio", 0) > 1.1:
            signal_count += 1

    if temporal:
        signal_total += 2
        if temporal.get("burst_ratio_j", 0) > temporal.get("burst_ratio_n", 0) * 1.5:
            signal_count += 1
        if temporal.get("night_diff", 0) > 0.03:
            signal_count += 1

    if velocity:
        signal_total += 2
        if velocity.get("vel_ratio", 0) > 1.3:
            signal_count += 1
        if velocity.get("upm_ratio", 0) > 1.3:
            signal_count += 1

    if money_flow:
        signal_total += 2
        if money_flow.get("conc_diff", 0) > 0.01:
            signal_count += 1
        if money_flow.get("fanout_ratio", 0) > 1.05:
            signal_count += 1

    if signal_total > 0 and signal_count < signal_total * 0.5:
        checks.append(f"  🔴 ML signal coverage: {signal_count}/{signal_total} signals detected (layers 3-6 may underperform)")
        issues += 1

    # Positive checks
    if sil["tx"] < 0.2:
        checks.append(f"  ✅ Transaction silhouette = {sil['tx']:.2f} (realistic)")
    if baselines["tx_lr_f1"] < 0.80:
        checks.append(f"  ✅ LR F1 = {baselines['tx_lr_f1']:.2f} (reasonable difficulty)")
    if borderline["overall"] >= 0.20:
        checks.append(f"  ✅ Borderline = {borderline['overall']*100:.1f}% (realistic)")
    if signal_total > 0 and signal_count >= signal_total * 0.5:
        checks.append(f"  ✅ ML signal coverage: {signal_count}/{signal_total} signals detected")

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
    network = test_network_signal(df)
    temporal = test_temporal_signal(df)
    velocity = test_velocity_signal(df)
    money_flow = test_money_flow(df)
    print_summary(sil, baselines, borderline, network, temporal, velocity, money_flow)

    if output_file:
        output_file.close()
        sys.stdout = original_stdout
        print(f"\n  Results also saved to {args.output}")


if __name__ == "__main__":
    main()
