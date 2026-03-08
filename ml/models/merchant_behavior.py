"""
Pantau ML — Layer 2: Merchant Behavior Anomaly Detection
=========================================================
Aggregates per-merchant behavioral features from raw transaction data,
then trains an Isolation Forest to detect anomalous merchant patterns.

Features engineered per merchant (from PRD Section 7.3):
- Transaction velocity (daily, 7d, 30d)
- Amount statistics (mean, std, coefficient of variation)
- Unique senders and repeat sender rate
- Temporal patterns (peak hour, night rate, hour spread)
- Geographic spread of senders
- Round amount rate, e-wallet rate
- Sender concentration
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

def engineer_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transactions into per-merchant behavioral features."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    max_date = df["timestamp"].max()
    date_7d = max_date - pd.Timedelta(days=7)
    date_30d = max_date - pd.Timedelta(days=30)

    merchant_features = []

    for merchant_id, m_txs in df.groupby("merchant_id"):
        txs = m_txs.sort_values("timestamp")
        n = len(txs)

        profile_span_days = max(
            (txs["timestamp"].max() - txs["timestamp"].min()).days, 1
        )

        # --- Velocity features ---
        txs_7d = txs[txs["timestamp"] >= date_7d]
        txs_30d = txs[txs["timestamp"] >= date_30d]

        avg_tx_per_day = n / profile_span_days
        tx_count_7d = len(txs_7d)
        tx_count_30d = len(txs_30d)

        # --- Amount features ---
        avg_amount = txs["amount"].mean()
        std_amount = txs["amount"].std() if n > 1 else 0
        cv_amount = std_amount / max(avg_amount, 1)

        # --- Sender features ---
        unique_senders = txs["user_id"].nunique()
        unique_senders_7d = txs_7d["user_id"].nunique() if len(txs_7d) > 0 else 0

        sender_counts = txs["user_id"].value_counts()
        repeat_senders = sender_counts[sender_counts > 1]
        repeat_sender_rate = len(repeat_senders) / max(unique_senders, 1)

        # --- Temporal features ---
        hour_counts = txs["tx_hour"].value_counts()
        peak_hour = hour_counts.index[0] if len(hour_counts) > 0 else 12
        night_mask = txs["tx_hour"].isin([22, 23, 0, 1, 2, 3, 4])
        night_tx_rate = night_mask.sum() / n
        hour_std = txs["tx_hour"].std() if n > 1 else 0

        # --- Geographic spread of senders ---
        unique_sender_cities = txs["user_city"].nunique()
        unique_sender_provinces = txs["user_province"].nunique()
        geo_spread = unique_sender_provinces / max(n, 1) * 10

        # --- Round amount & payment type ---
        round_amount_rate = txs["is_round_amount"].sum() / n
        ewallet_rate = txs["transaction_type"].str.startswith("EWALLET").sum() / n

        # --- Weekend rate ---
        weekend_rate = (txs["tx_day_of_week"] >= 5).sum() / n

        # --- Sender concentration (top sender % of total) ---
        top_sender_rate = sender_counts.iloc[0] / n if len(sender_counts) > 0 else 0

        merchant_features.append({
            "merchant_id": merchant_id,
            "tx_count": n,
            "avg_tx_per_day": avg_tx_per_day,
            "tx_count_7d": tx_count_7d,
            "tx_count_30d": tx_count_30d,
            "avg_amount": avg_amount,
            "std_amount": std_amount,
            "cv_amount": cv_amount,
            "unique_senders": unique_senders,
            "unique_senders_7d": unique_senders_7d,
            "repeat_sender_rate": repeat_sender_rate,
            "peak_hour": peak_hour,
            "night_tx_rate": night_tx_rate,
            "hour_std": hour_std,
            "unique_sender_cities": unique_sender_cities,
            "unique_sender_provinces": unique_sender_provinces,
            "geo_spread": geo_spread,
            "round_amount_rate": round_amount_rate,
            "ewallet_rate": ewallet_rate,
            "weekend_rate": weekend_rate,
            "top_sender_rate": top_sender_rate,
            "profile_age_days": profile_span_days,
        })

    return pd.DataFrame(merchant_features)


FEATURE_COLUMNS = [
    "tx_count", "avg_tx_per_day", "tx_count_7d", "tx_count_30d",
    "avg_amount", "std_amount", "cv_amount",
    "unique_senders", "unique_senders_7d", "repeat_sender_rate",
    "peak_hour", "night_tx_rate", "hour_std",
    "unique_sender_cities", "unique_sender_provinces", "geo_spread",
    "round_amount_rate", "ewallet_rate", "weekend_rate",
    "top_sender_rate", "profile_age_days",
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
    """Train Isolation Forest on merchant behavioral features."""
    print("  [Merchant Behavior] Engineering features...")
    feature_df = engineer_merchant_features(df)

    # Ground truth: merchant is judol if majority of their txs are label=1
    merchant_labels = df.groupby("merchant_id")["label"].mean()
    feature_df["label"] = feature_df["merchant_id"].map(merchant_labels).apply(
        lambda x: 1 if x > 0.5 else 0
    )

    X = feature_df[FEATURE_COLUMNS].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  [Merchant Behavior] Training IF ({len(X):,} merchants, {len(FEATURE_COLUMNS)} features)...")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    score_range = anomaly_scores.max() - anomaly_scores.min()
    risk_scores = np.clip(
        (1 - (anomaly_scores - anomaly_scores.min()) / max(score_range, 1e-9)) * 100,
        0, 100,
    ).astype(int)

    feature_df["predicted_anomaly"] = (predictions == -1).astype(int)
    feature_df["risk_score"] = risk_scores

    tp = ((feature_df["predicted_anomaly"] == 1) & (feature_df["label"] == 1)).sum()
    fp = ((feature_df["predicted_anomaly"] == 1) & (feature_df["label"] == 0)).sum()
    fn = ((feature_df["predicted_anomaly"] == 0) & (feature_df["label"] == 1)).sum()
    tn = ((feature_df["predicted_anomaly"] == 0) & (feature_df["label"] == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = {
        "total_merchants": len(feature_df),
        "flagged_merchants": int((predictions == -1).sum()),
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    print(f"  [Merchant Behavior] precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
    print(f"  [Merchant Behavior] Flagged {(predictions == -1).sum():,} / {len(feature_df):,} merchants")

    return {"model": model, "scaler": scaler, "feature_df": feature_df, "metrics": metrics}


# ============================================================
# PREDICTION
# ============================================================

def predict(merchant_features: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Score merchants using a trained model."""
    X = merchant_features[FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    score_range = anomaly_scores.max() - anomaly_scores.min()
    risk_scores = np.clip(
        (1 - (anomaly_scores - anomaly_scores.min()) / max(score_range, 1e-9)) * 100,
        0, 100,
    ).astype(int)

    merchant_features = merchant_features.copy()
    merchant_features["predicted_anomaly"] = (predictions == -1).astype(int)
    merchant_features["risk_score"] = risk_scores
    return merchant_features


# ============================================================
# SAVE / LOAD
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def save(model, scaler, path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = path or os.path.join(MODEL_DIR, "merchant_behavior.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"  [Merchant Behavior] Saved to {path}")


def load(path: str = None):
    path = path or os.path.join(MODEL_DIR, "merchant_behavior.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]
