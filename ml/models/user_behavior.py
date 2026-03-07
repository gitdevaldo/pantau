"""
Pantau ML — Layer 1: User Behavior Anomaly Detection
=====================================================
Aggregates per-user behavioral features from raw transaction data,
then trains an Isolation Forest to detect anomalous user patterns.

Features engineered per user (from PRD Section 7.2):
- Transaction frequency (1hr, 24hr, 7d windows)
- Amount statistics (mean, std, deviation)
- Temporal patterns (peak hour, hour deviation, late-night rate)
- Merchant diversity (unique merchants, repeat rate)
- Round amount rate
- Geographic consistency
- Profile age
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw transactions into per-user behavioral features.
    Input: transaction-level DataFrame
    Output: user-level DataFrame with engineered features
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Date boundaries for relative window calculations
    max_date = df["timestamp"].max()
    date_7d = max_date - pd.Timedelta(days=7)
    date_30d = max_date - pd.Timedelta(days=30)

    user_features = []

    for user_id, user_txs in df.groupby("user_id"):
        txs = user_txs.sort_values("timestamp")
        n = len(txs)

        # --- Frequency features ---
        txs_7d = txs[txs["timestamp"] >= date_7d]
        txs_30d = txs[txs["timestamp"] >= date_30d]

        profile_span_days = max(
            (txs["timestamp"].max() - txs["timestamp"].min()).days, 1
        )
        avg_tx_per_day = n / profile_span_days
        tx_count_7d = len(txs_7d)
        tx_count_30d = len(txs_30d)

        # --- Amount features ---
        avg_amount_30d = txs_30d["amount"].mean() if len(txs_30d) > 0 else txs["amount"].mean()
        std_amount = txs["amount"].std() if n > 1 else 0
        amount_deviation = (
            (txs["amount"].iloc[-1] - avg_amount_30d) / max(std_amount, 1)
            if n > 1 else 0
        )

        # --- Temporal features ---
        hour_counts = txs["tx_hour"].value_counts()
        peak_hour = hour_counts.index[0] if len(hour_counts) > 0 else 12
        hour_std = txs["tx_hour"].std() if n > 1 else 0

        # Late night rate (22:00 - 04:00)
        late_night_mask = txs["tx_hour"].isin([22, 23, 0, 1, 2, 3, 4])
        late_night_rate = late_night_mask.sum() / n

        # Weekend rate
        weekend_rate = (txs["tx_day_of_week"] >= 5).sum() / n

        # --- Merchant diversity ---
        unique_merchants_7d = txs_7d["merchant_id"].nunique() if len(txs_7d) > 0 else 0
        unique_merchants_total = txs["merchant_id"].nunique()

        merchant_counts = txs["merchant_id"].value_counts()
        repeat_merchants = merchant_counts[merchant_counts > 1].index
        repeat_merchant_rate = txs["merchant_id"].isin(repeat_merchants).sum() / n

        # --- Round amount features ---
        round_amount_rate = txs["is_round_amount"].sum() / n

        # --- Geographic features ---
        unique_merchant_cities = txs["merchant_city"].nunique()
        unique_merchant_provinces = txs["merchant_province"].nunique()

        # Geo spread: higher = more spread out (suspicious)
        geo_spread = unique_merchant_provinces / max(n, 1) * 10

        # Same city rate: user_city == merchant_city
        same_city_rate = (txs["user_city"] == txs["merchant_city"]).sum() / n

        # --- E-wallet rate ---
        ewallet_rate = txs["transaction_type"].str.startswith("EWALLET").sum() / n

        # --- Profile age ---
        profile_age_days = profile_span_days

        # --- Assemble feature vector ---
        user_features.append({
            "user_id": user_id,
            "tx_count": n,
            "avg_tx_per_day": avg_tx_per_day,
            "tx_count_7d": tx_count_7d,
            "tx_count_30d": tx_count_30d,
            "avg_amount_30d": avg_amount_30d,
            "std_amount": std_amount,
            "amount_deviation": amount_deviation,
            "peak_hour": peak_hour,
            "hour_std": hour_std,
            "late_night_rate": late_night_rate,
            "weekend_rate": weekend_rate,
            "unique_merchants_7d": unique_merchants_7d,
            "unique_merchants_total": unique_merchants_total,
            "repeat_merchant_rate": repeat_merchant_rate,
            "round_amount_rate": round_amount_rate,
            "unique_merchant_cities": unique_merchant_cities,
            "unique_merchant_provinces": unique_merchant_provinces,
            "geo_spread": geo_spread,
            "same_city_rate": same_city_rate,
            "ewallet_rate": ewallet_rate,
            "profile_age_days": profile_age_days,
        })

    return pd.DataFrame(user_features)


# Feature columns used for Isolation Forest training
FEATURE_COLUMNS = [
    "tx_count", "avg_tx_per_day", "tx_count_7d", "tx_count_30d",
    "avg_amount_30d", "std_amount", "amount_deviation",
    "peak_hour", "hour_std", "late_night_rate", "weekend_rate",
    "unique_merchants_7d", "unique_merchants_total", "repeat_merchant_rate",
    "round_amount_rate", "unique_merchant_cities", "unique_merchant_provinces",
    "geo_spread", "same_city_rate", "ewallet_rate", "profile_age_days",
]


# ============================================================
# MODEL TRAINING
# ============================================================

def train(
    df: pd.DataFrame,
    contamination: float = 0.15,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Train Isolation Forest on user behavioral features.

    Args:
        df: Raw transaction DataFrame
        contamination: Expected fraction of anomalies (matches our 15% judol rate)
        n_estimators: Number of trees in the forest
        random_state: Seed for reproducibility

    Returns:
        dict with model, scaler, feature_df, and evaluation metrics
    """
    print("  [User Behavior] Engineering features...")
    feature_df = engineer_user_features(df)

    # Ground truth: user is judol if majority of their txs are label=1
    user_labels = df.groupby("user_id")["label"].mean()
    feature_df["label"] = feature_df["user_id"].map(user_labels).apply(
        lambda x: 1 if x > 0.5 else 0
    )

    X = feature_df[FEATURE_COLUMNS].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  [User Behavior] Training Isolation Forest ({len(X):,} users, {len(FEATURE_COLUMNS)} features)...")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Predict: -1 = anomaly, 1 = normal
    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    # Convert to 0-100 risk score (lower decision_function = higher risk)
    risk_scores = np.clip(
        (1 - (anomaly_scores - anomaly_scores.min()) /
         (anomaly_scores.max() - anomaly_scores.min())) * 100,
        0, 100
    ).astype(int)

    feature_df["predicted_anomaly"] = (predictions == -1).astype(int)
    feature_df["risk_score"] = risk_scores

    # Evaluation against ground truth
    true_positive = ((feature_df["predicted_anomaly"] == 1) & (feature_df["label"] == 1)).sum()
    false_positive = ((feature_df["predicted_anomaly"] == 1) & (feature_df["label"] == 0)).sum()
    false_negative = ((feature_df["predicted_anomaly"] == 0) & (feature_df["label"] == 1)).sum()
    true_negative = ((feature_df["predicted_anomaly"] == 0) & (feature_df["label"] == 0)).sum()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    metrics = {
        "total_users": len(feature_df),
        "flagged_users": int((predictions == -1).sum()),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    print(f"  [User Behavior] Results: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
    print(f"  [User Behavior] Flagged {(predictions == -1).sum():,} / {len(feature_df):,} users")

    return {
        "model": model,
        "scaler": scaler,
        "feature_df": feature_df,
        "metrics": metrics,
    }


# ============================================================
# PREDICTION (for API use)
# ============================================================

def predict(user_features: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Score users using a trained model.

    Args:
        user_features: DataFrame from engineer_user_features()
        model: Trained IsolationForest
        scaler: Fitted StandardScaler

    Returns:
        DataFrame with risk_score and predicted_anomaly columns added
    """
    X = user_features[FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    risk_scores = np.clip(
        (1 - (anomaly_scores - anomaly_scores.min()) /
         (anomaly_scores.max() - anomaly_scores.min())) * 100,
        0, 100
    ).astype(int)

    user_features = user_features.copy()
    user_features["predicted_anomaly"] = (predictions == -1).astype(int)
    user_features["risk_score"] = risk_scores

    return user_features


# ============================================================
# SAVE / LOAD
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def save(model, scaler, path: str = None):
    """Save trained model and scaler."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "user_behavior.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"  [User Behavior] Saved to {path}")


def load(path: str = None):
    """Load trained model and scaler."""
    path = path or os.path.join(MODEL_DIR, "user_behavior.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]
