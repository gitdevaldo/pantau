"""
Pantau ML — Training Orchestrator
===================================
Proper evaluation methodology:
  1. 80/20 stratified split → 80% train pool, 20% locked test set
  2. 5-Fold CV on train pool to grid-search hyperparameters
  3. Retrain final model on full train pool with best params
  4. Evaluate ONCE on locked test set → real metrics
  5. Save .pkl models

Usage:
    python3 -m ml.train [--input path/to/dataset.csv] [--sample N] [--tag name]
    python3 -m ml.train --no-tune   # skip K-Fold, use defaults
    python3 -m ml.train --folds 3   # fewer folds (faster)
    python3 -m ml.train --n-jobs 5  # parallel folds (default: -1 = all cores)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, StratifiedKFold

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models import (
    user_behavior,
    merchant_behavior,
    network_cluster,
    temporal_pattern,
    velocity_delta,
    money_flow,
)
from ml import scoring


SEED = 42
LAYER_KEYS = ["user", "merchant", "network", "temporal", "velocity", "flow"]


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

# IF contamination for layers 1-3
CONTAMINATION_GRID = [0.10, 0.15, 0.20]

# Per-layer threshold for rule-based layers 4-6
LAYER_THRESHOLD_GRID = [30.0, 40.0, 50.0]

# Combined scoring threshold
COMBINED_THRESHOLD_GRID = [30, 35, 40, 45]

# Number of random weight vectors to generate (Dirichlet distribution, all sum to 1.0)
N_WEIGHT_SAMPLES = 50


def generate_weight_samples(n: int = N_WEIGHT_SAMPLES) -> list:
    """Generate random weight vectors using Dirichlet distribution."""
    rng = np.random.RandomState(SEED)
    keys = LAYER_KEYS
    samples = []
    for _ in range(n):
        raw = rng.dirichlet(np.ones(len(keys)))
        raw = np.round(raw, 2)
        raw[-1] = round(1.0 - raw[:-1].sum(), 2)
        samples.append(dict(zip(keys, raw)))
    return samples


# Two-phase grid:
#   Phase 1 (expensive): contamination × layer_threshold = 9 combos → retrain per fold
#   Phase 2 (cheap):     50 weight samples × 4 combined_thresholds = 200 combos → score only
# Total retrains: 9 × 5 folds = 45
# Total scoring evals: 45 × 200 = 9,000 (fast, no retraining)



# ============================================================
# DATA LOADING & SPLITTING
# ============================================================

def load_dataset(path: str, sample: int = None) -> pd.DataFrame:
    """Load and prepare the transaction dataset."""
    print(f"\n[1/6] Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} transactions")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_round_amount"] = df["is_round_amount"].astype(bool)
    df["label"] = df["label"].astype(int)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=SEED).reset_index(drop=True)
        print(f"  Sampled {len(df):,} transactions")

    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.20):
    """Stratified 80/20 split. Test set is LOCKED — used only once at the end."""
    print(f"\n[2/6] Splitting dataset (train={1-test_size:.0%}, test={test_size:.0%} locked 🔒)...")
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df["label"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f"  Train pool: {len(df_train):,} rows ({(df_train['label']==1).mean()*100:.1f}% judol)")
    print(f"  Test (locked): {len(df_test):,} rows ({(df_test['label']==1).mean()*100:.1f}% judol)")
    return df_train, df_test


# ============================================================
# TRAINING (single run with given hyperparams)
# ============================================================

def train_layers(df: pd.DataFrame, contamination: float = 0.15,
                 layer_threshold: float = 40.0, verbose: bool = True) -> dict:
    """Train all 6 layers with given hyperparameters."""
    results = {}
    timings = {}

    layers = [
        ("user", "User Behavior", lambda d: user_behavior.train(d, contamination=contamination)),
        ("merchant", "Merchant Behavior", lambda d: merchant_behavior.train(d, contamination=contamination)),
        ("network", "Network Clustering", lambda d: network_cluster.train(d, contamination=contamination)),
        ("temporal", "Temporal Pattern", lambda d: temporal_pattern.train(d, threshold=layer_threshold)),
        ("velocity", "Velocity Delta", lambda d: velocity_delta.train(d, threshold=layer_threshold)),
        ("flow", "Money Flow", lambda d: money_flow.train(d, threshold=layer_threshold)),
    ]

    for key, name, train_fn in layers:
        if verbose:
            print(f"  Training {name}...", end=" ", flush=True)
        t0 = time.time()
        results[key] = train_fn(df)
        timings[key] = time.time() - t0
        if verbose:
            f1 = results[key]["metrics"]["f1_score"]
            print(f"F1={f1:.3f} ({timings[key]:.1f}s)")

    results["_timings"] = timings
    return results


def score_and_evaluate(df: pd.DataFrame, results: dict, weights: dict,
                       threshold: float) -> dict:
    """Score a dataset using trained layer results + given weights/threshold."""
    old_weights = scoring.WEIGHTS.copy()
    scoring.WEIGHTS.update(weights)

    scored_df = scoring.combine_scores(df, results)
    metrics = scoring.evaluate(scored_df, threshold=threshold)

    scoring.WEIGHTS.update(old_weights)
    return scored_df, metrics


# ============================================================
# K-FOLD CROSS-VALIDATION ON TRAIN POOL
# ============================================================

def _run_single_fold(df_train: pd.DataFrame, train_idx, val_idx,
                     contam: float, layer_thresh: float,
                     score_grid: list, fold_idx: int) -> dict:
    """Train one fold and sweep all scoring combos. Designed for joblib parallelism."""
    df_fold_train = df_train.iloc[train_idx].reset_index(drop=True)
    df_fold_val = df_train.iloc[val_idx].reset_index(drop=True)

    # Phase 1: Train layers (expensive)
    results = train_layers(
        df_fold_train,
        contamination=contam,
        layer_threshold=layer_thresh,
        verbose=False,
    )

    # Phase 2: Sweep scoring params (cheap — no retraining)
    fold_scores = {}
    for si, (weights, combo_thresh) in enumerate(score_grid):
        _, metrics = score_and_evaluate(
            df_fold_val, results,
            weights=weights,
            threshold=combo_thresh,
        )
        fold_scores[si] = metrics["f1_score"]

    print(f"    Fold {fold_idx+1} done", flush=True)
    return fold_scores


def kfold_tune(df_train: pd.DataFrame, n_folds: int = 5, n_jobs: int = -1) -> dict:
    """
    Two-phase grid search with K-Fold CV on the training pool.
    Phase 1: Retrain layers for each (contamination, layer_threshold) combo per fold.
    Phase 2: For each trained fold, sweep Dirichlet weight samples × combined thresholds (no retrain).

    Folds within each combo run in parallel using joblib (n_jobs=-1 = all cores).
    """
    train_grid = [(c, lt) for c in CONTAMINATION_GRID for lt in LAYER_THRESHOLD_GRID]
    weight_samples = generate_weight_samples(N_WEIGHT_SAMPLES)
    score_grid = [(w, ct) for w in weight_samples for ct in COMBINED_THRESHOLD_GRID]
    n_train_combos = len(train_grid)
    n_score_combos = len(score_grid)
    total_retrains = n_train_combos * n_folds
    total_evals = total_retrains * n_score_combos

    actual_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
    print(f"\n[3/6] K-Fold tuning ({n_folds} folds, {actual_jobs} parallel jobs):")
    print(f"  Phase 1: {n_train_combos} train combos × {n_folds} folds = {total_retrains} retrains")
    print(f"  Phase 2: {N_WEIGHT_SAMPLES} Dirichlet weights × {len(COMBINED_THRESHOLD_GRID)} thresholds "
          f"= {n_score_combos} score combos per retrain")
    print(f"  Total scoring evals: {total_evals:,}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(df_train, df_train["label"]))

    # Track best across ALL full param combos
    # Key: (train_combo_idx, score_combo_idx) → list of fold F1s
    all_results = {}

    for ti, (contam, layer_thresh) in enumerate(train_grid):
        print(f"\n  Train combo {ti+1}/{n_train_combos}: "
              f"contamination={contam}, layer_threshold={layer_thresh}")

        # Run all folds in parallel
        fold_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run_single_fold)(
                df_train, train_idx, val_idx,
                contam, layer_thresh, score_grid, fold_idx
            )
            for fold_idx, (train_idx, val_idx) in enumerate(fold_indices)
        )

        # Aggregate fold results
        for fold_scores in fold_results:
            for si, f1 in fold_scores.items():
                key = (ti, si)
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(f1)

    # Find best combo by average F1 across folds
    best_key = max(all_results, key=lambda k: np.mean(all_results[k]))
    best_ti, best_si = best_key
    best_fold_f1s = all_results[best_key]
    best_avg_f1 = np.mean(best_fold_f1s)
    best_std_f1 = np.std(best_fold_f1s)

    best_contam, best_layer_thresh = train_grid[best_ti]
    best_weights, best_combo_thresh = score_grid[best_si]

    best_params = {
        "contamination": best_contam,
        "layer_threshold": best_layer_thresh,
        "combined_threshold": best_combo_thresh,
        "weights": best_weights.copy(),
    }

    total_combos = n_train_combos * n_score_combos
    best_combo_num = best_ti * n_score_combos + best_si + 1

    print(f"\n  ✓ Best combo #{best_combo_num}/{total_combos}: "
          f"avg F1 = {best_avg_f1:.4f} ± {best_std_f1:.4f}")
    print(f"    contamination={best_contam}, layer_threshold={best_layer_thresh}, "
          f"combined_threshold={best_combo_thresh}")
    print(f"    weights: {best_weights}")
    print(f"    per-fold F1: {[f'{f:.4f}' for f in best_fold_f1s]}")

    return {
        "best_params": best_params,
        "best_avg_f1": best_avg_f1,
        "best_std_f1": best_std_f1,
        "best_fold_f1s": best_fold_f1s,
        "best_combo_idx": best_combo_num,
        "total_combos": total_combos,
        "n_folds": n_folds,
        "total_retrains": total_retrains,
        "total_evals": total_evals,
    }


# ============================================================
# SAVE / REPORT
# ============================================================

def save_all(results: dict, tag: str):
    """Save all trained models and scored data."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(project_root, "models", tag)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\nSaving models to models/{tag}/...")

    if "user" in results:
        user_behavior.save(results["user"]["model"], results["user"]["scaler"],
                           os.path.join(model_dir, "user_behavior.pkl"))
    if "merchant" in results:
        merchant_behavior.save(results["merchant"]["model"], results["merchant"]["scaler"],
                               os.path.join(model_dir, "merchant_behavior.pkl"))
    if "network" in results:
        network_cluster.save(results["network"]["model"], results["network"]["scaler"],
                             os.path.join(model_dir, "network_cluster.pkl"))

    best_params = results.get("_best_params", {})
    lt = best_params.get("layer_threshold", 40.0)
    temporal_pattern.save(threshold=lt, path=os.path.join(model_dir, "temporal_pattern.pkl"))
    velocity_delta.save(threshold=lt, path=os.path.join(model_dir, "velocity_delta.pkl"))
    money_flow.save(threshold=lt, path=os.path.join(model_dir, "money_flow.pkl"))

    if "_scored" in results:
        scored_dir = os.path.join(project_root, "data", "scored", tag)
        os.makedirs(scored_dir, exist_ok=True)
        out_path = os.path.join(scored_dir, "scored_transactions.csv")
        results["_scored"].to_csv(out_path, index=False)
        print(f"  Scored transactions saved to {out_path}")


def build_report(results: dict, tag: str, input_path: str, elapsed: float,
                 tune_result: dict = None) -> str:
    """Build a text report of training results."""
    lines = []
    lines.append("=" * 70)
    lines.append("  PANTAU — Training Report")
    lines.append(f"  Dataset: {tag} ({input_path})")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if tune_result:
        lines.append(f"  Method: {tune_result['n_folds']}-Fold CV on 80% train pool, "
                     f"final eval on 20% locked test set")
        lines.append(f"  Grid: {tune_result['total_combos']} combos × "
                     f"{tune_result['n_folds']} folds = "
                     f"{tune_result['total_combos'] * tune_result['n_folds']} evaluations")
    else:
        lines.append(f"  Method: 80/20 split, default params (no tuning)")
    lines.append("=" * 70)

    layer_names = {
        "user": "1. User Behavior",
        "merchant": "2. Merchant Behavior",
        "network": "3. Network Clustering",
        "temporal": "4. Temporal Pattern",
        "velocity": "5. Velocity Delta",
        "flow": "6. Money Flow",
    }

    timings = results.get("_timings", {})

    lines.append(f"\n  {'Layer':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>8}")
    lines.append("  " + "-" * 65)

    for key, name in layer_names.items():
        if key in results:
            m = results[key]["metrics"]
            t = timings.get(key, 0)
            lines.append(f"  {name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                         f"{m['f1_score']:>10.4f} {t:>7.1f}s")

    if "_combined_metrics" in results:
        cm = results["_combined_metrics"]
        total_time = sum(timings.values())
        lines.append("  " + "-" * 65)
        lines.append(f"  {'COMBINED':<25} {cm['precision']:>10.4f} {cm['recall']:>10.4f} "
                     f"{cm['f1_score']:>10.4f} {total_time:>7.1f}s")
        lines.append(f"\n  AUC-ROC:    {cm.get('auc_roc', 0):.4f}")
        lines.append(f"  PR-AUC:     {cm.get('pr_auc', 0):.4f}")
        lines.append(f"\n  Avg score (normal): {cm['avg_score_normal']:.1f}")
        lines.append(f"  Avg score (judol):  {cm['avg_score_judol']:.1f}")
        lines.append(f"\n  Risk distribution:")
        for level, count in sorted(cm.get("risk_distribution", {}).items()):
            lines.append(f"    {level}: {count:,}")

    if tune_result:
        bp = tune_result["best_params"]
        lines.append(f"\n  --- K-Fold Tuning Results ---")
        lines.append(f"  Retrains: {tune_result['total_retrains']} | "
                     f"Scoring evals: {tune_result['total_evals']}")
        lines.append(f"  Best combo: #{tune_result['best_combo_idx']} / {tune_result['total_combos']}")
        lines.append(f"  CV avg F1: {tune_result['best_avg_f1']:.4f} ± {tune_result['best_std_f1']:.4f}")
        lines.append(f"  Per-fold F1: {[f'{f:.4f}' for f in tune_result['best_fold_f1s']]}")
        lines.append(f"  contamination: {bp['contamination']}")
        lines.append(f"  layer_threshold: {bp['layer_threshold']}")
        lines.append(f"  combined_threshold: {bp['combined_threshold']}")
        lines.append(f"  weights:")
        for k, v in bp["weights"].items():
            lines.append(f"    {k}: {v:.2f}")

    lines.append(f"\n  Total time: {elapsed:.1f}s")
    lines.append("=" * 70)
    return "\n".join(lines)


def save_report(report: str, results: dict, tag: str, tune_result: dict = None):
    """Save training report (text) and metrics (JSON) to logs/."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    report_path = os.path.join(log_dir, f"training_report_{tag}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")

    metrics = {}
    for key in LAYER_KEYS:
        if key in results:
            metrics[key] = results[key]["metrics"]
    if "_combined_metrics" in results:
        metrics["combined"] = results["_combined_metrics"]
    metrics["timings"] = results.get("_timings", {})
    if tune_result:
        bp = tune_result["best_params"]
        metrics["tuning"] = {
            "method": f"{tune_result['n_folds']}-Fold CV",
            "total_combos": tune_result["total_combos"],
            "best_combo": tune_result["best_combo_idx"],
            "cv_avg_f1": tune_result["best_avg_f1"],
            "cv_std_f1": tune_result["best_std_f1"],
            "fold_f1s": tune_result["best_fold_f1s"],
            "contamination": bp["contamination"],
            "layer_threshold": bp["layer_threshold"],
            "combined_threshold": bp["combined_threshold"],
            "weights": {k: float(v) for k, v in bp["weights"].items()},
        }

    json_path = os.path.join(log_dir, f"training_metrics_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics saved to {json_path}")


def print_summary(results: dict, tune_result: dict = None):
    """Print final performance summary."""
    print("\n" + "=" * 70)
    print("  PANTAU — Final Evaluation (Locked Test Set 🔒)")
    print("=" * 70)
    print(f"  {'Layer':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>8}")
    print("  " + "-" * 65)

    layer_names = {
        "user": "1. User Behavior",
        "merchant": "2. Merchant Behavior",
        "network": "3. Network Clustering",
        "temporal": "4. Temporal Pattern",
        "velocity": "5. Velocity Delta",
        "flow": "6. Money Flow",
    }

    timings = results.get("_timings", {})
    for key, name in layer_names.items():
        if key in results:
            m = results[key]["metrics"]
            t = timings.get(key, 0)
            print(f"  {name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1_score']:>10.4f} {t:>7.1f}s")

    if "_combined_metrics" in results:
        cm = results["_combined_metrics"]
        total_time = sum(timings.values())
        print("  " + "-" * 65)
        print(f"  {'COMBINED':<25} {cm['precision']:>10.4f} {cm['recall']:>10.4f} "
              f"{cm['f1_score']:>10.4f} {total_time:>7.1f}s")
        print(f"  {'AUC-ROC':<25} {cm.get('auc_roc', 0):>10.4f}")
        print(f"  {'PR-AUC':<25} {cm.get('pr_auc', 0):>10.4f}")

    if tune_result:
        print(f"\n  Tuned params: contamination={tune_result['best_params']['contamination']}, "
              f"layer_thresh={tune_result['best_params']['layer_threshold']}, "
              f"combo_thresh={tune_result['best_params']['combined_threshold']}")
        print(f"  CV avg F1: {tune_result['best_avg_f1']:.4f} ± {tune_result['best_std_f1']:.4f}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pantau ML Training Pipeline")
    parser.add_argument("--input", "-i", type=str,
                        default="data/generated/gan/pantau_gan_ctgan.csv",
                        help="Path to training dataset CSV")
    parser.add_argument("--sample", "-s", type=int, default=None,
                        help="Sample N rows for faster testing")
    parser.add_argument("--tag", "-t", type=str, default=None,
                        help="Tag name for this run (default: auto-detect)")
    parser.add_argument("--folds", "-k", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving models")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip K-Fold tuning, use default params")
    parser.add_argument("--n-jobs", "-j", type=int, default=-1,
                        help="Parallel jobs for K-Fold CV (-1 = all cores, default: -1)")
    args = parser.parse_args()

    # Fallback to parametric dataset if GAN not available
    if not os.path.exists(args.input):
        fallback = "data/generated/parametric/pantau_dataset.csv"
        if os.path.exists(fallback):
            print(f"  GAN dataset not found, using parametric: {fallback}")
            args.input = fallback
        else:
            print(f"  ERROR: Dataset not found at {args.input}")
            sys.exit(1)

    # Auto-detect tag
    if not args.tag:
        if "gan" in args.input.lower():
            args.tag = "gan"
        elif "parametric" in args.input.lower():
            args.tag = "parametric"
        else:
            args.tag = os.path.splitext(os.path.basename(args.input))[0]

    start = time.time()

    # Step 1: Load
    df = load_dataset(args.input, sample=args.sample)

    # Step 2: 80/20 split — test set is LOCKED
    df_train, df_test = split_dataset(df, test_size=0.20)

    # Step 3: K-Fold tuning on train pool only
    tune_result = None
    if not args.no_tune:
        tune_result = kfold_tune(df_train, n_folds=args.folds, n_jobs=args.n_jobs)
        best_params = tune_result["best_params"]
    else:
        best_params = {
            "contamination": 0.15,
            "layer_threshold": 40.0,
            "combined_threshold": 40,
            "weights": scoring.WEIGHTS.copy(),
        }

    # Step 4: Retrain final model on FULL train pool with best params
    print(f"\n[4/6] Training final model on full train pool ({len(df_train):,} rows)...")
    results = train_layers(
        df_train,
        contamination=best_params["contamination"],
        layer_threshold=best_params["layer_threshold"],
        verbose=True,
    )
    results["_best_params"] = best_params

    # Step 5: Evaluate ONCE on locked test set
    print(f"\n[5/6] Final evaluation on locked test set ({len(df_test):,} rows) 🔒...")
    scored_df, combined_metrics = score_and_evaluate(
        df_test, results,
        weights=best_params["weights"],
        threshold=best_params["combined_threshold"],
    )
    scoring.print_report(combined_metrics)

    results["_scored"] = scored_df
    results["_combined_metrics"] = combined_metrics

    elapsed = time.time() - start

    # Step 6: Save
    print(f"\n[6/6] Saving artifacts...")
    if not args.no_save:
        save_all(results, tag=args.tag)

    print_summary(results, tune_result)

    report = build_report(results, tag=args.tag, input_path=args.input,
                          elapsed=elapsed, tune_result=tune_result)
    save_report(report, results, tag=args.tag, tune_result=tune_result)

    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print(f"  Tag: {args.tag}")
    print("  Done! ✓")


if __name__ == "__main__":
    main()
