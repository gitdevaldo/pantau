"""
Pantau ML — Layer 5: Velocity Delta Detection (Cross-Merchant Z-Score)
======================================================================
Compares each merchant's transaction patterns against the global merchant
population to detect outliers. Uses cross-merchant z-scores instead of
per-merchant historical comparison (which requires long tx history).

Features per merchant (from PRD Section 7.6):
- Cross-merchant z-scores for count, amount, unique senders
- Velocity percentile rankings
- Spike indicators based on population statistics
"""

import os
import pickle

import numpy as np
import pandas as pd


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-merchant velocity features using cross-merchant z-scores."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Per-merchant aggregate stats
    merchant_agg = df.groupby("merchant_id").agg(
        total_tx=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        unique_users=("user_id", "nunique"),
        unique_days=("date", "nunique"),
        round_amount_rate=("is_round_amount", "mean"),
    ).reset_index()

    # Daily stats
    daily = df.groupby(["merchant_id", "date"]).agg(
        daily_count=("transaction_id", "count"),
        daily_amount=("amount", "sum"),
        daily_unique_users=("user_id", "nunique"),
    ).reset_index()

    daily_stats = daily.groupby("merchant_id").agg(
        avg_daily_count=("daily_count", "mean"),
        max_daily_count=("daily_count", "max"),
        std_daily_count=("daily_count", "std"),
        avg_daily_amount=("daily_amount", "mean"),
        max_daily_amount=("daily_amount", "max"),
    ).reset_index()

    daily_stats["std_daily_count"] = daily_stats["std_daily_count"].fillna(0)
    daily_stats["burstiness"] = (
        daily_stats["max_daily_count"] / daily_stats["avg_daily_count"].replace(0, 1)
    )

    features = merchant_agg.merge(daily_stats, on="merchant_id", how="left")
    features["std_amount"] = features["std_amount"].fillna(0)
    features["tx_per_day"] = features["total_tx"] / features["unique_days"].replace(0, 1)
    features["users_per_tx"] = features["unique_users"] / features["total_tx"].replace(0, 1)

    # --- Cross-merchant z-scores (compare each merchant vs ALL merchants) ---
    zscore_cols = ["total_tx", "total_amount", "unique_users", "avg_amount",
                   "tx_per_day", "round_amount_rate", "burstiness"]

    for col in zscore_cols:
        col_mean = features[col].mean()
        col_std = features[col].std()
        if col_std == 0:
            col_std = 1
        features[f"z_{col}"] = (features[col] - col_mean) / col_std

    # --- Composite risk score ---
    z_cols = [f"z_{c}" for c in zscore_cols]
    z_abs = features[z_cols].abs()

    # Higher weight on user count and burstiness (judol signals)
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.5, 2.0, 1.5])
    weighted_z = (z_abs.values * weights).mean(axis=1)

    features["risk_score"] = np.clip(
        weighted_z * 20 +
        (features["burstiness"] > 3).astype(int) * 10 +
        (features["round_amount_rate"] > 0.5).astype(int) * 10,
        0, 100,
    ).round(1)

    return features


FEATURE_COLUMNS = [
    "total_tx", "total_amount", "avg_amount", "std_amount",
    "unique_users", "unique_days", "round_amount_rate",
    "avg_daily_count", "max_daily_count", "burstiness", "tx_per_day",
    "z_total_tx", "z_total_amount", "z_unique_users", "z_avg_amount",
    "z_tx_per_day", "z_round_amount_rate", "z_burstiness",
]


# ============================================================
# TRAIN (statistical — no ML model, just evaluation)
# ============================================================

def train(df: pd.DataFrame, threshold: float = 40.0) -> dict:
    """Compute velocity features and evaluate against ground truth."""
    print("  [Velocity] Computing cross-merchant velocity features...")
    feature_df = engineer_velocity_features(df)

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

    print(f"  [Velocity] precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
    print(f"  [Velocity] Flagged {feature_df['predicted_anomaly'].sum():,} / {len(feature_df):,} merchants")

    return {"model": None, "scaler": None, "feature_df": feature_df, "metrics": metrics}


# ============================================================
# PREDICTION
# ============================================================

def predict(df: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
    """Score new transactions with velocity analysis (no model needed)."""
    feature_df = engineer_velocity_features(df)
    feature_df["predicted_anomaly"] = (feature_df["risk_score"] >= threshold).astype(int)
    return feature_df


# ============================================================
# SAVE / LOAD
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def save(threshold: float = 40.0, path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "velocity_delta.pkl")
    with open(path, "wb") as f:
        pickle.dump({"threshold": threshold}, f)
    print(f"  [Velocity] Saved to {path}")


def load(path: str = None):
    path = path or os.path.join(MODEL_DIR, "velocity_delta.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["threshold"]
