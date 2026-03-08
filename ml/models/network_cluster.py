"""
Pantau ML — Layer 3: Network Clustering & Ring Detection
========================================================
Builds a user→merchant transaction graph and analyzes network topology
to detect organized judol rings and suspicious merchant clusters.

Features per merchant:
- Degree centrality and fan-in (unique senders)
- Shared sender analysis between merchants
- Community/cluster detection
- Hub and ring scoring
"""

import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import label_propagation_communities
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_transaction_graph(df: pd.DataFrame) -> nx.Graph:
    """Build bipartite user-merchant graph weighted by transaction count."""
    edges = df.groupby(["user_id", "merchant_id"]).agg(
        tx_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
    ).reset_index()

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_node(row["user_id"], node_type="user")
        G.add_node(row["merchant_id"], node_type="merchant")
        G.add_edge(
            row["user_id"], row["merchant_id"],
            tx_count=row["tx_count"],
            total_amount=row["total_amount"],
        )
    return G


def build_merchant_projection(df: pd.DataFrame) -> nx.Graph:
    """Project to merchant-merchant graph (connected if they share users)."""
    um = df[["user_id", "merchant_id"]].drop_duplicates()

    # Filter users with too many merchants to prevent combinatorial explosion
    merchant_per_user = um.groupby("user_id")["merchant_id"].count()
    valid_users = merchant_per_user[merchant_per_user <= 50].index
    um = um[um["user_id"].isin(valid_users)]

    # Self-join to find merchant pairs sharing users
    pairs = um.merge(um, on="user_id", suffixes=("_1", "_2"))
    pairs = pairs[pairs["merchant_id_1"] < pairs["merchant_id_2"]]

    shared = (
        pairs.groupby(["merchant_id_1", "merchant_id_2"])
        .size()
        .reset_index(name="shared_users")
    )

    MG = nx.Graph()
    all_merchants = df["merchant_id"].unique()
    MG.add_nodes_from(all_merchants)

    for _, row in shared.iterrows():
        MG.add_edge(row["merchant_id_1"], row["merchant_id_2"],
                     shared_users=int(row["shared_users"]))
    return MG


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-merchant network features from transaction graph."""
    print("  [Network] Building transaction graph...")
    G = build_transaction_graph(df)

    print("  [Network] Building merchant projection...")
    MG = build_merchant_projection(df)

    merchant_nodes = [n for n, d in G.nodes(data=True)
                      if d.get("node_type") == "merchant"]

    # --- Centrality on bipartite graph ---
    print("  [Network] Computing centrality...")
    degree_cent = nx.degree_centrality(G)

    # --- Merchant projection metrics ---
    mg_degree = dict(MG.degree())
    mg_clustering = nx.clustering(MG) if MG.number_of_edges() > 0 else {}

    # --- Community detection ---
    print("  [Network] Detecting communities...")
    cluster_map = {}
    if MG.number_of_edges() > 0:
        try:
            communities = list(label_propagation_communities(MG))
            for cid, comm in enumerate(communities):
                for node in comm:
                    cluster_map[node] = cid
        except Exception:
            for i, m in enumerate(merchant_nodes):
                cluster_map[m] = i
    else:
        for i, m in enumerate(merchant_nodes):
            cluster_map[m] = i

    # --- Geographic sender diversity per merchant ---
    geo = df.groupby("merchant_id").agg(
        unique_sender_cities=("user_city", "nunique"),
        unique_sender_provinces=("user_province", "nunique"),
    ).to_dict("index")

    # --- Sender concentration per merchant ---
    def top_province_rate(grp):
        vc = grp.value_counts()
        return vc.iloc[0] / len(grp) if len(vc) > 0 else 0

    sender_conc = (
        df.groupby("merchant_id")["user_province"]
        .apply(top_province_rate)
        .to_dict()
    )

    # --- Build feature rows ---
    features = []
    for m_id in merchant_nodes:
        n_senders = G.degree(m_id) if m_id in G else 0

        shared_merchant_count = mg_degree.get(m_id, 0)
        total_shared_users = sum(
            MG[m_id][nb].get("shared_users", 0)
            for nb in MG.neighbors(m_id)
        ) if m_id in MG else 0

        g = geo.get(m_id, {})

        hub_score = min(100, (
            n_senders * 0.3 +
            shared_merchant_count * 5 +
            total_shared_users * 2
        ))

        ring_score = min(100, (
            mg_clustering.get(m_id, 0) * 30 +
            total_shared_users * 3 +
            shared_merchant_count * 2
        ))

        features.append({
            "merchant_id": m_id,
            "network_degree_centrality": degree_cent.get(m_id, 0),
            "network_unique_senders": n_senders,
            "network_unique_sender_cities": g.get("unique_sender_cities", 0),
            "network_unique_sender_provinces": g.get("unique_sender_provinces", 0),
            "network_sender_concentration": sender_conc.get(m_id, 0),
            "network_shared_merchant_count": shared_merchant_count,
            "network_total_shared_users": total_shared_users,
            "network_clustering_coeff": mg_clustering.get(m_id, 0),
            "network_cluster_id": cluster_map.get(m_id, -1),
            "network_hub_score": hub_score,
            "network_ring_score": ring_score,
        })

    return pd.DataFrame(features)


FEATURE_COLUMNS = [
    "network_degree_centrality",
    "network_unique_senders", "network_unique_sender_cities",
    "network_unique_sender_provinces", "network_sender_concentration",
    "network_shared_merchant_count", "network_total_shared_users",
    "network_clustering_coeff", "network_hub_score", "network_ring_score",
]


# ============================================================
# MODEL TRAINING
# ============================================================

def train(
    df: pd.DataFrame,
    contamination: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Compute network features then train Isolation Forest on them."""
    feature_df = engineer_network_features(df)

    merchant_labels = df.groupby("merchant_id")["label"].mean()
    feature_df["label"] = feature_df["merchant_id"].map(merchant_labels).apply(
        lambda x: 1 if x > 0.5 else 0
    )

    X = feature_df[FEATURE_COLUMNS].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  [Network] Training IF ({len(X):,} merchants, {len(FEATURE_COLUMNS)} features)...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
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

    print(f"  [Network] precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")
    return {"model": model, "scaler": scaler, "feature_df": feature_df, "metrics": metrics}


# ============================================================
# PREDICTION
# ============================================================

def predict(merchant_features: pd.DataFrame, model, scaler) -> pd.DataFrame:
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
    path = path or os.path.join(MODEL_DIR, "network_cluster.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"  [Network] Saved to {path}")


def load(path: str = None):
    path = path or os.path.join(MODEL_DIR, "network_cluster.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]
