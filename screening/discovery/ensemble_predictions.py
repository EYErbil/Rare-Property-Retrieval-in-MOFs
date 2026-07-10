#!/usr/bin/env python3
"""
Discovery ensemble: RRF + rank_avg on NN + ML + kNN predictions → top 25 for DFT.

Loads test_predictions.csv (or inference_predictions.csv) from each prediction dir,
applies Reciprocal Rank Fusion and rank averaging, outputs top 25.

No true-label evaluation (unlabeled set — no labels).

Usage:
  python discovery/ensemble_predictions.py \\
    --prediction_dirs ml_results/extra_trees ml_results/random_forest knn_results \\
    --nn_predictions discovery/Processed-data/inference_results/inference_predictions.csv \\
    --output_dir discovery/Processed-data/inference_results \\
    --top_k 25
"""

import os
import sys
import csv
import argparse
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def infer_score_direction(mode_str):
    """Same as ensemble_discovery: regression/knn → lower=better; multiclass → higher=better."""
    if not mode_str:
        return True
    mode_lower = str(mode_str).lower().strip()
    if any(tag in mode_lower for tag in ["regression", "knn", "sim_to_pos", "ensemble"]):
        return True
    return False


def load_predictions_from_csv(csv_path, normalize_lower_is_better=True):
    """
    Load cif_id -> score from test_predictions.csv or inference_predictions.csv.
    If normalize_lower_is_better=True (default), invert multiclass scores so that
    internally all scores are 'lower = better', matching ensemble_discovery and heatmap logic.
    """
    preds = {}
    mode_str = "regression"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("cif_id", "").strip()
            if not cid:
                continue
            try:
                preds[cid] = float(row["score"])
                if "mode" in row and row["mode"]:
                    mode_str = row["mode"]
            except (KeyError, ValueError):
                continue
    if not preds:
        return preds
    if normalize_lower_is_better and not infer_score_direction(mode_str):
        # multiclass: higher score = more positive → convert to lower = better
        max_s = max(preds.values())
        preds = {cid: max_s - s for cid, s in preds.items()}
    return preds


def find_predictions_csv(path):
    """Return path to test_predictions.csv or inference_predictions.csv under path."""
    path = os.path.normpath(path)
    for name in ["test_predictions.csv", "inference_predictions.csv"]:
        p = os.path.join(path, name)
        if os.path.isfile(p):
            return p
    return None


def discover_prediction_dirs(base_dir):
    """Find all dirs under base_dir that contain test_predictions.csv (one level deep)."""
    found = []
    base = os.path.normpath(base_dir)
    if find_predictions_csv(base):
        found.append((os.path.basename(base) or "predictions", base))
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            sub = os.path.join(base, name)
            if os.path.isdir(sub) and find_predictions_csv(sub):
                found.append((name, sub))
    return found


def score_to_rank(scores, lower_is_better=True):
    """Convert scores to ranks (1 = best). Handles ties by ordinal rank."""
    n = len(scores)
    if lower_is_better:
        order = np.argsort(scores)
    else:
        order = np.argsort(-np.array(scores))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks


def _fill_missing_scores(scores_arr, lower_is_better):
    """Replace NaN with value that gets worst rank."""
    arr = np.array(scores_arr, dtype=float)
    mask = np.isnan(arr)
    if not np.any(mask):
        return arr
    if lower_is_better:
        arr[mask] = np.nanmax(arr[~mask]) + 1e9 if np.any(~mask) else 1e9
    else:
        arr[mask] = np.nanmin(arr[~mask]) - 1e9 if np.any(~mask) else -1e9
    return arr


def reciprocal_rank_fusion(models, test_cids, k=60, lower_is_better=True):
    """RRF: score = sum(1/(k+rank_i)). Return {cid: combined_score} where lower = better."""
    cid_scores = defaultdict(float)
    for _name, model_scores in models.items():
        scores_arr = np.array([model_scores.get(cid, float("nan")) for cid in test_cids])
        filled = _fill_missing_scores(scores_arr, lower_is_better)
        ranks = score_to_rank(filled, lower_is_better=lower_is_better)
        for i, cid in enumerate(test_cids):
            cid_scores[cid] += 1.0 / (k + ranks[i])
    if not cid_scores:
        return {cid: 0.0 for cid in test_cids}
    max_score = max(cid_scores.values())
    return {cid: max_score - cid_scores[cid] for cid in test_cids}


def rank_averaging(models, test_cids, lower_is_better=True):
    """Average rank across models. Lower average rank = better."""
    cid_rank_sum = defaultdict(float)
    for _name, model_scores in models.items():
        scores_arr = np.array([model_scores.get(cid, float("nan")) for cid in test_cids])
        filled = _fill_missing_scores(scores_arr, lower_is_better)
        ranks = score_to_rank(filled, lower_is_better=lower_is_better)
        for i, cid in enumerate(test_cids):
            cid_rank_sum[cid] += ranks[i]
    n_models = len(models)
    return {cid: cid_rank_sum[cid] / n_models for cid in test_cids}


def infer_model_type(csv_path):
    """Detect model type (NN / ML) from the 'mode' column of a predictions CSV."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = (row.get("mode") or "").strip().lower()
            if mode:
                return "NN" if mode == "regression" else "ML"
    return "ML"  # default


def type_balanced_rrf(models, test_cids, model_types, k=60):
    """
    Two-stage RRF for type-balanced ensemble.

    Stage 1: RRF within each model type (NN, ML) independently.
    Stage 2: RRF across the 2 type-level scores (equal weight per type).

    This ensures each model *family* contributes 50 %, regardless of how
    many individual models exist in that family (e.g. 3 NN vs 2 ML).

    Parameters
    ----------
    models : dict[str, dict[str, float]]
        {model_name: {cid: score}} — all scores lower-is-better.
    test_cids : list[str]
    model_types : dict[str, str]
        {model_name: 'NN' | 'ML'}
    k : int
        RRF smoothing constant.
    """
    groups = defaultdict(dict)
    for name, scores in models.items():
        t = model_types.get(name, "ML")
        groups[t][name] = scores

    if len(groups) < 2:
        # Only one type present — fall back to plain RRF
        return reciprocal_rank_fusion(models, test_cids, k=k, lower_is_better=True)

    # Stage 1: within-type RRF
    type_scores = {}
    for t, grp in groups.items():
        type_scores[t] = reciprocal_rank_fusion(grp, test_cids, k=k,
                                                lower_is_better=True)

    # Stage 2: across-type RRF (each type = 1 vote)
    return reciprocal_rank_fusion(type_scores, test_cids, k=k,
                                  lower_is_better=True)


def main():
    parser = argparse.ArgumentParser(
        description="Discovery ensemble: RRF + rank_avg → top 25 for DFT"
    )
    parser.add_argument(
        "--prediction_dirs",
        nargs="+",
        default=[],
        help="Directories each containing test_predictions.csv",
    )
    parser.add_argument(
        "--nn_predictions",
        type=str,
        default=None,
        help="Path to NN inference_predictions.csv (or dir containing it)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for top25 files",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[25, 50, 100],
        help="Top-K values to output (default: 25 50 100)",
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF smoothing parameter k (default 60)",
    )
    args = parser.parse_args()

    base = PROJECT_ROOT
    output_dir = os.path.join(base, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    models = {}
    all_cids = set()

    def add_predictions(label, path):
        if not path or not os.path.exists(path):
            return
        if os.path.isdir(path):
            path = find_predictions_csv(path)
        if path and os.path.isfile(path):
            preds = load_predictions_from_csv(path)
            if preds:
                models[label] = preds
                all_cids.update(preds.keys())

    for d in args.prediction_dirs:
        p = os.path.join(base, d) if not os.path.isabs(d) else d
        if find_predictions_csv(p):
            add_predictions(os.path.basename(p.rstrip("/\\")) or "pred", p)
        else:
            for name, sub in discover_prediction_dirs(p):
                add_predictions(name, sub)

    if args.nn_predictions:
        nn_path = os.path.join(base, args.nn_predictions) if not os.path.isabs(args.nn_predictions) else args.nn_predictions
        add_predictions("nn", nn_path)

    if not models:
        raise ValueError("No prediction files found. Provide --prediction_dirs and/or --nn_predictions.")

    test_cids = sorted(all_cids)
    top_k_list = sorted(set(args.top_k))
    print("=" * 70)
    print(f"  Discovery ensemble: RRF + rank_avg → top {top_k_list} for DFT")
    print("=" * 70)
    print(f"  Models: {list(models.keys())}")
    print(f"  Test CIFs: {len(test_cids)}")
    print(f"  Top-K: {top_k_list}")

    rrf_scores = reciprocal_rank_fusion(models, test_cids, k=args.rrf_k, lower_is_better=True)
    rank_avg_scores = rank_averaging(models, test_cids, lower_is_better=True)

    rrf_sorted = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c])
    rank_avg_sorted = sorted(rank_avg_scores.keys(), key=lambda c: rank_avg_scores[c])

    for method_name, sorted_cids in [("rrf", rrf_sorted), ("rank_avg", rank_avg_sorted)]:
        for k in top_k_list:
            top_cids = sorted_cids[:k]
            out_path = os.path.join(output_dir, f"top{k}_for_DFT_{method_name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for cid in top_cids:
                    f.write(cid + "\n")
            print(f"  Saved {os.path.basename(out_path)}: {out_path}")

    max_k = max(top_k_list)
    print("=" * 70)
    print(f"  Top {max_k} (RRF):")
    for i, cid in enumerate(rrf_sorted[:max_k], 1):
        print(f"    {i:2d}. {cid}")
    print("=" * 70)
    print("  Done.")


if __name__ == "__main__":
    main()
