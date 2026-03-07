"""
Pantau ML — Combined Risk Scoring Engine
==========================================
Combines scores from all 6 detection layers into a per-transaction
final risk score using the PRD Section 8.4 weighted formula.

Weights (from PRD):
  user_score     * 0.15
  merchant_score * 0.25
  network_score  * 0.25
  temporal_score * 0.10
  velocity_score * 0.15
  flow_score     * 0.10
  + cross_correlation_bonus (+15 if 4+ layers flag same entity)

Risk levels (PRD Section 9):
  0-50  : Normal ✅
  50-70 : Suspicious ⚠️
  70-90 : High Risk 🚩
  90-100: Critical 🧊
"""

import numpy as np
import pandas as pd


# ============================================================
# SCORE COMBINATION
# ============================================================

WEIGHTS = {
    "user": 0.15,
    "merchant": 0.25,
    "network": 0.25,
    "temporal": 0.10,
    "velocity": 0.15,
    "flow": 0.10,
}

CROSS_CORRELATION_THRESHOLD = 4
CROSS_CORRELATION_BONUS = 15


def combine_scores(df: pd.DataFrame, layer_results: dict) -> pd.DataFrame:
    """
    Combine all layer scores into per-transaction final risk score.

    Args:
        df: Raw transaction DataFrame
        layer_results: dict with keys 'user', 'merchant', 'network',
                       'temporal', 'velocity', 'flow' — each containing
                       a 'feature_df' with entity_id and 'risk_score'.

    Returns:
        DataFrame with per-transaction risk scores and risk levels.
    """
    scored = df[["transaction_id", "user_id", "merchant_id", "label"]].copy()

    # --- Build lookup dicts: entity_id → risk_score ---

    # User-level layers (keyed by user_id)
    user_scores = {}
    if "user" in layer_results:
        fdf = layer_results["user"]["feature_df"]
        user_scores = dict(zip(fdf["user_id"], fdf["risk_score"]))

    temporal_scores = {}
    if "temporal" in layer_results:
        fdf = layer_results["temporal"]["feature_df"]
        temporal_scores = dict(zip(fdf["user_id"], fdf["risk_score"]))

    # Merchant-level layers (keyed by merchant_id)
    merchant_scores = {}
    if "merchant" in layer_results:
        fdf = layer_results["merchant"]["feature_df"]
        merchant_scores = dict(zip(fdf["merchant_id"], fdf["risk_score"]))

    network_scores = {}
    if "network" in layer_results:
        fdf = layer_results["network"]["feature_df"]
        network_scores = dict(zip(fdf["merchant_id"], fdf["risk_score"]))

    velocity_scores = {}
    if "velocity" in layer_results:
        fdf = layer_results["velocity"]["feature_df"]
        velocity_scores = dict(zip(fdf["merchant_id"], fdf["risk_score"]))

    flow_scores = {}
    if "flow" in layer_results:
        fdf = layer_results["flow"]["feature_df"]
        flow_scores = dict(zip(fdf["merchant_id"], fdf["risk_score"]))

    # --- Map scores to transactions ---
    scored["user_score"] = scored["user_id"].map(user_scores).fillna(0)
    scored["temporal_score"] = scored["user_id"].map(temporal_scores).fillna(0)
    scored["merchant_score"] = scored["merchant_id"].map(merchant_scores).fillna(0)
    scored["network_score"] = scored["merchant_id"].map(network_scores).fillna(0)
    scored["velocity_score"] = scored["merchant_id"].map(velocity_scores).fillna(0)
    scored["flow_score"] = scored["merchant_id"].map(flow_scores).fillna(0)

    # --- Weighted combination ---
    scored["weighted_score"] = (
        scored["user_score"] * WEIGHTS["user"] +
        scored["merchant_score"] * WEIGHTS["merchant"] +
        scored["network_score"] * WEIGHTS["network"] +
        scored["temporal_score"] * WEIGHTS["temporal"] +
        scored["velocity_score"] * WEIGHTS["velocity"] +
        scored["flow_score"] * WEIGHTS["flow"]
    )

    # --- Cross-correlation bonus ---
    flag_threshold = 50
    layer_cols = ["user_score", "merchant_score", "network_score",
                  "temporal_score", "velocity_score", "flow_score"]
    scored["layers_flagged"] = (scored[layer_cols] >= flag_threshold).sum(axis=1)

    scored["cross_bonus"] = np.where(
        scored["layers_flagged"] >= CROSS_CORRELATION_THRESHOLD,
        CROSS_CORRELATION_BONUS, 0
    )

    scored["final_score"] = np.clip(
        scored["weighted_score"] + scored["cross_bonus"], 0, 100
    ).round(1)

    # --- Risk level ---
    scored["risk_level"] = pd.cut(
        scored["final_score"],
        bins=[-1, 50, 70, 90, 100],
        labels=["Normal", "Suspicious", "High Risk", "Critical"],
    )

    return scored


# ============================================================
# EVALUATION
# ============================================================

def evaluate(scored_df: pd.DataFrame, threshold: float = 50.0) -> dict:
    """Evaluate combined scoring against ground truth labels."""
    predicted = (scored_df["final_score"] >= threshold).astype(int)
    actual = scored_df["label"].astype(int)

    tp = ((predicted == 1) & (actual == 1)).sum()
    fp = ((predicted == 1) & (actual == 0)).sum()
    fn = ((predicted == 0) & (actual == 1)).sum()
    tn = ((predicted == 0) & (actual == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    risk_dist = scored_df["risk_level"].value_counts().to_dict()

    metrics = {
        "total_transactions": len(scored_df),
        "flagged_transactions": int(predicted.sum()),
        "threshold": threshold,
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "risk_distribution": risk_dist,
        "avg_score_normal": round(scored_df[actual == 0]["final_score"].mean(), 2),
        "avg_score_judol": round(scored_df[actual == 1]["final_score"].mean(), 2),
    }

    return metrics


def print_report(metrics: dict):
    """Pretty-print combined scoring report."""
    print("\n" + "=" * 60)
    print("  PANTAU — Combined Risk Scoring Report")
    print("=" * 60)
    print(f"  Total transactions: {metrics['total_transactions']:,}")
    print(f"  Flagged (score ≥ {metrics['threshold']}): {metrics['flagged_transactions']:,}")
    print(f"\n  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"\n  Avg score (normal): {metrics['avg_score_normal']:.1f}")
    print(f"  Avg score (judol):  {metrics['avg_score_judol']:.1f}")
    print(f"\n  Risk distribution:")
    for level, count in sorted(metrics.get("risk_distribution", {}).items()):
        print(f"    {level}: {count:,}")
    print("=" * 60)
