"""
Pantau ML — Layer 4: Temporal Pattern Detection (Rule-based)
=============================================================
Analyzes per-user transaction timing to detect gambling session patterns.

Rules detect (from PRD Section 7.5):
- Judol prime-time activity (20:00-02:00)
- Togel draw time alignment (13, 16, 19, 22)
- Gajian day concentration (1st, 25th-28th)
- Rapid-burst sessions (inter-tx < 60 min)
- Amount escalation within sessions (chasing losses)
- Time-of-day consistency (habitual behavior)
- Weekend concentration
"""

import os
import pickle

import numpy as np
import pandas as pd


# ============================================================
# TEMPORAL SCORING RULES
# ============================================================

TOGEL_HOURS = {13, 16, 19, 22}
LATE_NIGHT_HOURS = {20, 21, 22, 23, 0, 1, 2}
GAJIAN_DAYS = {1, 25, 26, 27, 28}
SESSION_GAP_MINUTES = 60
EXPECTED_GAJIAN_RATE = len(GAJIAN_DAYS) / 30  # ~16.7%


def score_user_temporal(user_txs: pd.DataFrame) -> dict:
    """
    Apply rule-based scoring to a single user's transactions.
    Returns dict with individual rule scores and final risk_score (0-100).
    """
    n = len(user_txs)
    if n == 0:
        return {"risk_score": 0}

    scores = {}

    # Rule 1: Late-night / prime-time rate (max 25 pts)
    late_rate = user_txs["tx_hour"].isin(LATE_NIGHT_HOURS).sum() / n
    scores["late_night_score"] = late_rate * 25

    # Rule 2: Togel draw-time alignment (max 15 pts)
    togel_rate = user_txs["tx_hour"].isin(TOGEL_HOURS).sum() / n
    expected_togel = len(TOGEL_HOURS) / 24
    scores["togel_score"] = max(0, (togel_rate - expected_togel) / max(1 - expected_togel, 0.01)) * 15

    # Rule 3: Gajian day concentration (max 15 pts)
    tx_days = pd.to_datetime(user_txs["timestamp"]).dt.day
    gajian_rate = tx_days.isin(GAJIAN_DAYS).sum() / n
    if gajian_rate > EXPECTED_GAJIAN_RATE * 1.5:
        scores["gajian_score"] = min(15, (gajian_rate - EXPECTED_GAJIAN_RATE) * 60)
    else:
        scores["gajian_score"] = 0

    # Rule 4: Rapid burst sessions — inter-tx < SESSION_GAP_MINUTES (max 20 pts)
    if n >= 3:
        timestamps = pd.to_datetime(user_txs["timestamp"]).sort_values()
        gaps_min = timestamps.diff().dt.total_seconds().dropna() / 60
        rapid_rate = (gaps_min < SESSION_GAP_MINUTES).sum() / max(len(gaps_min), 1)
        scores["burst_score"] = rapid_rate * 20
    else:
        scores["burst_score"] = 0

    # Rule 5: Amount escalation within sessions (max 10 pts)
    if n >= 3:
        amounts = user_txs.sort_values("timestamp")["amount"].values
        increases = sum(1 for i in range(1, len(amounts)) if amounts[i] > amounts[i - 1])
        increase_rate = increases / max(len(amounts) - 1, 1)
        scores["escalation_score"] = max(0, (increase_rate - 0.5) * 20)
    else:
        scores["escalation_score"] = 0

    # Rule 6: Weekend concentration (max 8 pts)
    weekend_rate = (user_txs["tx_day_of_week"] >= 5).sum() / n
    scores["weekend_score"] = max(0, (weekend_rate - 2 / 7)) * 24

    # Rule 7: Time-of-day consistency — low hour_std = habitual (max 7 pts)
    if n > 2:
        hour_std = user_txs["tx_hour"].std()
        if hour_std < 4:
            scores["consistency_score"] = (4 - hour_std) * 1.75
        else:
            scores["consistency_score"] = 0
    else:
        scores["consistency_score"] = 0

    risk_score = min(100, max(0, sum(scores.values())))
    scores["risk_score"] = round(risk_score, 1)
    return scores


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply temporal rules to each user and return per-user scored DataFrame."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    results = []
    for user_id, user_txs in df.groupby("user_id"):
        scores = score_user_temporal(user_txs)
        scores["user_id"] = user_id
        scores["tx_count"] = len(user_txs)
        results.append(scores)

    return pd.DataFrame(results)


# ============================================================
# TRAIN (rule-based — no model, just evaluation)
# ============================================================

def train(df: pd.DataFrame, threshold: float = 40.0) -> dict:
    """
    Apply temporal rules and evaluate against ground truth.
    threshold: risk_score above this is flagged as anomaly.
    """
    print("  [Temporal] Scoring users with temporal rules...")
    feature_df = engineer_temporal_features(df)

    user_labels = df.groupby("user_id")["label"].mean()
    feature_df["label"] = feature_df["user_id"].map(user_labels).apply(
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
        "total_users": len(feature_df),
        "flagged_users": int(feature_df["predicted_anomaly"].sum()),
        "threshold": threshold,
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    print(f"  [Temporal] precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
    print(f"  [Temporal] Flagged {feature_df['predicted_anomaly'].sum():,} / {len(feature_df):,} users")

    return {"model": None, "scaler": None, "feature_df": feature_df, "metrics": metrics}


# ============================================================
# PREDICTION
# ============================================================

def predict(df: pd.DataFrame, threshold: float = 40.0) -> pd.DataFrame:
    """Score new transactions with temporal rules (no model needed)."""
    feature_df = engineer_temporal_features(df)
    feature_df["predicted_anomaly"] = (feature_df["risk_score"] >= threshold).astype(int)
    return feature_df


# ============================================================
# SAVE / LOAD (saves threshold config)
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def save(threshold: float = 40.0, path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "temporal_pattern.pkl")
    with open(path, "wb") as f:
        pickle.dump({"threshold": threshold}, f)
    print(f"  [Temporal] Saved to {path}")


def load(path: str = None):
    path = path or os.path.join(MODEL_DIR, "temporal_pattern.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["threshold"]
