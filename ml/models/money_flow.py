"""
Pantau ML — Layer 6: Money Flow Tracing (Directed Graph)
=========================================================
Analyzes money flow patterns per merchant using directed graph analysis.
Detects collection-point behavior and layering indicators.

Features per merchant (from PRD Section 7.7):
- Fan-in analysis (many users → one merchant)
- Amount concentration and uniformity
- Temporal inflow clustering
- Layering indicators (same-amount rapid sequences)
"""

import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-merchant money flow features."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    features = []

    for merchant_id, m_txs in df.groupby("merchant_id"):
        txs = m_txs.sort_values("timestamp")
        n = len(txs)

        # --- Fan-in analysis ---
        unique_senders = txs["user_id"].nunique()
        fan_in_ratio = unique_senders / max(n, 1)

        # --- Amount concentration ---
        amounts = txs["amount"].values
        avg_amount = amounts.mean()
        std_amount = amounts.std() if n > 1 else 0

        # Coefficient of variation (low = uniform amounts = suspicious)
        cv_amount = std_amount / max(avg_amount, 1)
        amount_uniformity = max(0, 1 - cv_amount)

        # Round amount dominance
        round_rate = txs["is_round_amount"].sum() / n

        # Top-amount concentration: what % of txs use the most common amount bucket
        amount_buckets = (txs["amount"] // 10000).value_counts()
        top_bucket_rate = amount_buckets.iloc[0] / n if len(amount_buckets) > 0 else 0

        # --- Temporal inflow clustering ---
        if n >= 2:
            gaps_sec = txs["timestamp"].diff().dt.total_seconds().dropna()
            avg_gap_minutes = gaps_sec.mean() / 60
            rapid_inflow_rate = (gaps_sec < 300).sum() / max(len(gaps_sec), 1)  # < 5 min
        else:
            avg_gap_minutes = 0
            rapid_inflow_rate = 0

        # --- Layering indicators ---
        # Same-amount sequences: consecutive txs with identical amounts
        if n >= 3:
            same_as_prev = (np.diff(amounts) == 0).sum()
            same_amount_rate = same_as_prev / max(n - 1, 1)
        else:
            same_amount_rate = 0

        # Near-identical amounts (within 5% of each other)
        if n >= 3:
            sorted_amounts = np.sort(amounts)
            diffs = np.diff(sorted_amounts)
            near_identical = (diffs < sorted_amounts[:-1] * 0.05).sum()
            near_identical_rate = near_identical / max(n - 1, 1)
        else:
            near_identical_rate = 0

        # --- Geographic dispersion of senders ---
        sender_provinces = txs["user_province"].nunique()
        sender_cities = txs["user_city"].nunique()
        geo_dispersion = sender_provinces / max(n, 1) * 10

        # --- Total volume ---
        total_volume = txs["amount"].sum()
        avg_daily_volume = total_volume / max(
            (txs["timestamp"].max() - txs["timestamp"].min()).days, 1
        )

        # --- Risk score ---
        risk = 0.0
        # High fan-in from many unique senders
        if unique_senders > 5:
            risk += min(20, unique_senders * 1.5)
        # Amount uniformity (judol often uses fixed amounts)
        risk += amount_uniformity * 15
        # Round amount dominance
        risk += round_rate * 10
        # Rapid inflow (< 5 min gaps)
        risk += rapid_inflow_rate * 20
        # Same-amount sequences
        risk += same_amount_rate * 15
        # Geographic dispersion (many provinces sending to one merchant)
        risk += min(15, geo_dispersion * 3)
        # Top bucket concentration
        risk += top_bucket_rate * 5

        risk_score = min(100, max(0, risk))

        features.append({
            "merchant_id": merchant_id,
            "flow_fan_in_senders": unique_senders,
            "flow_fan_in_ratio": round(fan_in_ratio, 4),
            "flow_avg_amount": round(avg_amount, 2),
            "flow_cv_amount": round(cv_amount, 4),
            "flow_amount_uniformity": round(amount_uniformity, 4),
            "flow_round_rate": round(round_rate, 4),
            "flow_top_bucket_rate": round(top_bucket_rate, 4),
            "flow_avg_gap_minutes": round(avg_gap_minutes, 2),
            "flow_rapid_inflow_rate": round(rapid_inflow_rate, 4),
            "flow_same_amount_rate": round(same_amount_rate, 4),
            "flow_near_identical_rate": round(near_identical_rate, 4),
            "flow_sender_provinces": sender_provinces,
            "flow_sender_cities": sender_cities,
            "flow_geo_dispersion": round(geo_dispersion, 4),
            "flow_total_volume": total_volume,
            "flow_avg_daily_volume": round(avg_daily_volume, 2),
            "risk_score": round(risk_score, 1),
        })

    return pd.DataFrame(features)


FEATURE_COLUMNS = [
    "flow_fan_in_senders", "flow_fan_in_ratio",
    "flow_avg_amount", "flow_cv_amount", "flow_amount_uniformity",
    "flow_round_rate", "flow_top_bucket_rate",
    "flow_avg_gap_minutes", "flow_rapid_inflow_rate",
    "flow_same_amount_rate", "flow_near_identical_rate",
    "flow_sender_provinces", "flow_sender_cities", "flow_geo_dispersion",
    "flow_total_volume", "flow_avg_daily_volume",
]


# ============================================================
# TRAIN (graph-based — evaluation against ground truth)
# ============================================================

def train(df: pd.DataFrame, threshold: float = 40.0) -> dict:
    """Compute flow features and evaluate against ground truth."""
    print("  [Money Flow] Computing flow features...")
    feature_df = engineer_flow_features(df)

    merchant_labels = df.groupby("merchant_id")["label"].mean()
    feature_df["label"] = feature_df["merchant_id"].map(merchant_labels).apply(
        lambda x: 1 if x > 0.5 else 0
    )

    feature_df["predicted_anomaly"] = (feature_df["risk_score"] >= threshold).astype(int)

    tp = ((feature_df["predicted_anomaly"] == 1) & (feature_df["label"] == 1)).sum()
    fp = ((feature_df["predicted_anomaly"] == 1) & (feature_df["label"] == 0)).sum()
    fn = ((feature_df["predicted_anomaly"] == 0) & (feature_df["label"] == 1)).sum()
    tn = ((feature_df["predicted_anomaly"] == 0) & (feature_df["label"] == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = {
        "total_merchants": len(feature_df),
        "flagged_merchants": int(feature_df["predicted_anomaly"].sum()),
        "threshold": threshold,
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    print(f"  [Money Flow] precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
    print(f"  [Money Flow] Flagged {feature_df['predicted_anomaly'].sum():,} / {len(feature_df):,} merchants")

    return {"model": None, "scaler": None, "feature_df": feature_df, "metrics": metrics}


# ============================================================
# PREDICTION
# ============================================================

def predict(df: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
    """Score new transactions with flow analysis (no model needed)."""
    feature_df = engineer_flow_features(df)
    feature_df["predicted_anomaly"] = (feature_df["risk_score"] >= threshold).astype(int)
    return feature_df


# ============================================================
# SAVE / LOAD
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def save(threshold: float = 40.0, path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "money_flow.pkl")
    with open(path, "wb") as f:
        pickle.dump({"threshold": threshold}, f)
    print(f"  [Money Flow] Saved to {path}")


def load(path: str = None):
    path = path or os.path.join(MODEL_DIR, "money_flow.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["threshold"]
