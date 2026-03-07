"""
Pantau ML — Layer 5: Velocity Delta Detection (Z-Score / SPC)
==============================================================
Monitors per-merchant transaction velocity using statistical process control.
Detects abnormal spikes that may indicate sudden gambling activity surges.

Features per merchant (from PRD Section 7.6):
- Z-score of daily velocity vs rolling average
- Velocity delta percentage
- Spike detection and magnitude
"""

import os
import pickle

import numpy as np
import pandas as pd


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-merchant velocity features using daily transaction counts."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Daily transaction count and amount per merchant
    daily = df.groupby(["merchant_id", "date"]).agg(
        daily_count=("transaction_id", "count"),
        daily_amount=("amount", "sum"),
        daily_unique_senders=("user_id", "nunique"),
    ).reset_index()

    # Per-merchant statistics
    merchant_stats = daily.groupby("merchant_id").agg(
        avg_daily_count=("daily_count", "mean"),
        std_daily_count=("daily_count", "std"),
        max_daily_count=("daily_count", "max"),
        min_daily_count=("daily_count", "min"),
        avg_daily_amount=("daily_amount", "mean"),
        std_daily_amount=("daily_amount", "std"),
        avg_daily_senders=("daily_unique_senders", "mean"),
        std_daily_senders=("daily_unique_senders", "std"),
        active_days=("date", "count"),
    ).reset_index()

    # Last active day stats per merchant
    last_day_idx = daily.groupby("merchant_id")["date"].idxmax()
    last_day = daily.loc[last_day_idx, ["merchant_id", "daily_count", "daily_amount",
                                         "daily_unique_senders"]].rename(columns={
        "daily_count": "last_day_count",
        "daily_amount": "last_day_amount",
        "daily_unique_senders": "last_day_senders",
    })

    features = merchant_stats.merge(last_day, on="merchant_id", how="left")

    # Fill NaN std (merchants with 1 active day)
    features["std_daily_count"] = features["std_daily_count"].fillna(0)
    features["std_daily_amount"] = features["std_daily_amount"].fillna(0)
    features["std_daily_senders"] = features["std_daily_senders"].fillna(0)

    # --- Z-scores for last day ---
    features["velocity_zscore_count"] = (
        (features["last_day_count"] - features["avg_daily_count"])
        / features["std_daily_count"].replace(0, 1)
    )
    features["velocity_zscore_amount"] = (
        (features["last_day_amount"] - features["avg_daily_amount"])
        / features["std_daily_amount"].replace(0, 1)
    )
    features["velocity_zscore_senders"] = (
        (features["last_day_senders"] - features["avg_daily_senders"])
        / features["std_daily_senders"].replace(0, 1)
    )

    # --- Velocity delta % (last day vs average) ---
    features["velocity_delta_pct"] = (
        (features["last_day_count"] - features["avg_daily_count"])
        / features["avg_daily_count"].replace(0, 1) * 100
    )

    # --- Spike detection ---
    features["spike_detected"] = (
        features["velocity_zscore_count"].abs() > 2
    ).astype(int)

    features["spike_magnitude"] = (
        features["last_day_count"] / features["avg_daily_count"].replace(0, 1)
    )

    # --- Burstiness: max / avg ratio ---
    features["burstiness"] = (
        features["max_daily_count"] / features["avg_daily_count"].replace(0, 1)
    )

    # --- Risk score: composite of z-scores and spike indicators ---
    z_abs = features[["velocity_zscore_count", "velocity_zscore_amount",
                       "velocity_zscore_senders"]].abs()

    features["risk_score"] = np.clip(
        z_abs.mean(axis=1) * 15 +
        features["spike_detected"] * 20 +
        features["burstiness"].clip(0, 5) * 5,
        0, 100,
    ).round(1)

    return features


FEATURE_COLUMNS = [
    "avg_daily_count", "std_daily_count", "max_daily_count",
    "avg_daily_amount", "std_daily_amount",
    "avg_daily_senders", "std_daily_senders",
    "active_days",
    "velocity_zscore_count", "velocity_zscore_amount", "velocity_zscore_senders",
    "velocity_delta_pct", "spike_detected", "spike_magnitude", "burstiness",
]


# ============================================================
# TRAIN (statistical — no ML model, just evaluation)
# ============================================================

def train(df: pd.DataFrame, threshold: float = 45.0) -> dict:
    """Compute velocity features and evaluate against ground truth."""
    print("  [Velocity] Computing velocity features...")
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

def predict(df: pd.DataFrame, threshold: float = 45.0) -> pd.DataFrame:
    """Score new transactions with velocity analysis (no model needed)."""
    feature_df = engineer_velocity_features(df)
    feature_df["predicted_anomaly"] = (feature_df["risk_score"] >= threshold).astype(int)
    return feature_df


# ============================================================
# SAVE / LOAD
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def save(threshold: float = 45.0, path: str = None):
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
