#!/usr/bin/env python3
"""
Discovery pipeline: run saved ML + optional NN on a new test set (no ground truth),
output top-25/50/100 per individual model and per ensemble, plus agreement report.

Assumes unlabeled embeddings .npz is available (you do mapping). Creates:
  individual/<model>/top25.txt, top50.txt, top100.txt
  ensemble/rrf/top25.txt, ...
  ensemble/rank_avg/top25.txt, ...
  discovery_agreement_report.txt

Usage:
  python discovery/discovery_pipeline.py \\
    --discovery_dir /path/to/discovery \\
    --embeddings_path Processed-data/unlabeled_embeddings.npz \\
    [--ml_models_dir ...] [--nn_predictions ...] [--output_dir ...]
"""

import os
import sys
import csv
import argparse
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Include src/ for predict_with_embedding_classifier; include SCRIPT_DIR for ensemble_predictions
for d in (SCRIPT_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if d not in sys.path:
        sys.path.insert(0, d)

from predict_with_embedding_classifier import load_embeddings, load_model_artifacts, predict_scores
import ensemble_predictions as ep6

TOP_K_LIST = [25, 50, 100]


def discover_ml_methods(ml_dir):
    """List method names that have model.joblib or artifacts.joblib."""
    methods = []
    if not os.path.isdir(ml_dir):
        return methods
    for name in sorted(os.listdir(ml_dir)):
        method_dir = os.path.join(ml_dir, name)
        if not os.path.isdir(method_dir):
            continue
        if os.path.isfile(os.path.join(method_dir, "model.joblib")) or os.path.isfile(
            os.path.join(method_dir, "artifacts.joblib")
        ):
            methods.append(name)
    return methods


def scores_to_top_k(cid_scores, higher_is_better=True, k=100):
    """Return list of top-k CIF ids. For ML higher=better, for NN lower=better."""
    if higher_is_better:
        sorted_cids = sorted(cid_scores.keys(), key=lambda c: cid_scores[c], reverse=True)
    else:
        sorted_cids = sorted(cid_scores.keys(), key=lambda c: cid_scores[c])
    return sorted_cids[:k]


def run_individual_ml(embeddings_path, ml_models_dir, output_dir):
    """Run each saved ML model on embeddings; write individual/<method>/top25.txt, etc. Returns {method: {cid: score}} (higher=better)."""
    cif_ids, embeddings, _ = load_embeddings(embeddings_path)
    methods = discover_ml_methods(ml_models_dir)
    all_scores = {}
    for method in methods:
        method_dir = os.path.join(ml_models_dir, method)
        try:
            model, scaler, pca, artifacts = load_model_artifacts(method_dir)
            if model is None and "mu_pos" not in artifacts and "knn_model" not in artifacts:
                continue
            scores_arr = predict_scores(embeddings, model, scaler=scaler, pca=pca, artifacts=artifacts)
            scores_dict = {cid: float(scores_arr[i]) for i, cid in enumerate(cif_ids)}
            all_scores[method] = scores_dict
            ind_dir = os.path.join(output_dir, "individual", method)
            os.makedirs(ind_dir, exist_ok=True)
            for k in TOP_K_LIST:
                top = scores_to_top_k(scores_dict, higher_is_better=True, k=k)
                path = os.path.join(ind_dir, f"top{k}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    for cid in top:
                        f.write(cid + "\n")
            # Write inference_predictions.csv for custom ensemble step
            csv_path = os.path.join(ind_dir, "inference_predictions.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("cif_id,score,predicted_binary,true_label,mode\n")
                for cid in cif_ids:
                    sc = scores_dict.get(cid, 0.0)
                    pb = 1 if sc >= 0.5 else 0
                    f.write(f"{cid},{sc:.6f},{pb},0.0,multiclass\n")
        except Exception as e:
            print(f"  Warning: {method} failed: {e}")
    return all_scores


def run_nn(nn_predictions_path, output_dir, all_cids):
    """Load NN CSV (lower=better), write individual/nn/top25.txt, etc. Returns {cid: score} or None."""
    if not nn_predictions_path or not os.path.isfile(nn_predictions_path):
        return None
    preds = ep6.load_predictions_from_csv(nn_predictions_path, normalize_lower_is_better=True)
    if not preds:
        return None
    ind_dir = os.path.join(output_dir, "individual", "nn")
    os.makedirs(ind_dir, exist_ok=True)
    for k in TOP_K_LIST:
        top = scores_to_top_k(preds, higher_is_better=False, k=k)
        path = os.path.join(ind_dir, f"top{k}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for cid in top:
                f.write(cid + "\n")
    return preds


def run_ensemble(models_lower_better, all_cids, output_dir, rrf_k=60):
    """models_lower_better: {name: {cid: score}} with lower=better. Write ensemble/rrf and ensemble/rank_avg top25/50/100."""
    if not models_lower_better or not all_cids:
        return
    test_cids = sorted(all_cids)
    rrf_scores = ep6.reciprocal_rank_fusion(models_lower_better, test_cids, k=rrf_k, lower_is_better=True)
    rank_avg_scores = ep6.rank_averaging(models_lower_better, test_cids, lower_is_better=True)
    for method_name, scores in [("rrf", rrf_scores), ("rank_avg", rank_avg_scores)]:
        sorted_cids = sorted(scores.keys(), key=lambda c: scores[c])
        ens_dir = os.path.join(output_dir, "ensemble", method_name)
        os.makedirs(ens_dir, exist_ok=True)
        for k in TOP_K_LIST:
            top = sorted_cids[:k]
            path = os.path.join(ens_dir, f"top{k}.txt")
            with open(path, "w", encoding="utf-8") as f:
                for cid in top:
                    f.write(cid + "\n")


def collect_trial_top_k_lists(output_dir):
    """Collect for each K the set of (trial_name, set of CIFs in top-K)."""
    trials_by_k = {k: [] for k in TOP_K_LIST}
    ind_dir = os.path.join(output_dir, "individual")
    ens_dir = os.path.join(output_dir, "ensemble")
    for k in TOP_K_LIST:
        # individual
        if os.path.isdir(ind_dir):
            for name in sorted(os.listdir(ind_dir)):
                path = os.path.join(ind_dir, name, f"top{k}.txt")
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        cids = set(line.strip() for line in f if line.strip())
                    trials_by_k[k].append((f"individual/{name}", cids))
        # ensemble
        if os.path.isdir(ens_dir):
            for name in sorted(os.listdir(ens_dir)):
                path = os.path.join(ens_dir, name, f"top{k}.txt")
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        cids = set(line.strip() for line in f if line.strip())
                    trials_by_k[k].append((f"ensemble/{name}", cids))
    return trials_by_k


def write_agreement_report(output_dir, trials_by_k):
    """Write discovery_agreement_report.txt: for each K, in_all, in >=80%, in >=50%, and CIF counts."""
    path = os.path.join(output_dir, "discovery_agreement_report.txt")
    n_trials = {k: len(trials_by_k[k]) for k in TOP_K_LIST}
    with open(path, "w", encoding="utf-8") as f:
        f.write("Discovery agreement report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Structures that appear in all trials (or in >=80%, >=50%) for each top-K.\n\n")
        for k in TOP_K_LIST:
            trials = trials_by_k[k]
            if not trials:
                f.write(f"Top-{k}: no trials found.\n\n")
                continue
            all_sets = [s for _, s in trials]
            in_all = set.intersection(*all_sets) if all_sets else set()
            n = len(trials)
            in_80 = set()
            in_50 = set()
            count_per_cid = defaultdict(int)
            for _, s in trials:
                for cid in s:
                    count_per_cid[cid] += 1
            for cid, cnt in count_per_cid.items():
                if cnt >= max(1, int(0.8 * n + 0.5)):
                    in_80.add(cid)
                if cnt >= max(1, int(0.5 * n + 0.5)):
                    in_50.add(cid)
            f.write(f"Top-{k} (n_trials={n}):\n")
            f.write(f"  In all trials:     {len(in_all)} structures\n")
            f.write(f"  In >= 80% trials:  {len(in_80)} structures\n")
            f.write(f"  In >= 50% trials:  {len(in_50)} structures\n")
            if in_all:
                f.write("  CIFs in all: " + ", ".join(sorted(in_all)[:20]))
                if len(in_all) > 20:
                    f.write(f" ... (+{len(in_all) - 20} more)")
                f.write("\n")
            f.write("\n")
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Discovery pipeline: ML + NN → top-25/50/100 and agreement report")
    parser.add_argument("--discovery_dir", type=str, default=None,
                        help="Discovery root (default: dir of this script)")
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to unlabeled embeddings .npz")
    parser.add_argument("--ml_models_dir", type=str, default=None,
                        help="Path to embedding_classifiers/strategy_d_farthest_point (default: discovery_dir/../embedding_classifiers/strategy_d_farthest_point or discovery_dir/embedding_classifiers/strategy_d_farthest_point)")
    parser.add_argument("--nn_predictions", type=str, default=None,
                        help="Path to inference_predictions.csv (optional)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: discovery_dir/Processed-data/discovery_output)")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF k parameter")
    args = parser.parse_args()

    discovery_dir = os.path.abspath(args.discovery_dir or SCRIPT_DIR)
    emb_path = args.embeddings_path
    if not os.path.isabs(emb_path):
        emb_path = os.path.join(discovery_dir, emb_path)
    if not os.path.isfile(emb_path):
        print(f"ERROR: Embeddings not found: {emb_path}")
        return 1

    ml_dir = args.ml_models_dir
    if ml_dir is None:
        for candidate in [
            os.path.join(discovery_dir, "embedding_classifiers", "strategy_d_farthest_point"),
            os.path.join(PROJECT_ROOT, "embedding_classifiers", "strategy_d_farthest_point"),
        ]:
            if os.path.isdir(candidate):
                ml_dir = candidate
                break
        if ml_dir is None:
            ml_dir = os.path.join(discovery_dir, "embedding_classifiers", "strategy_d_farthest_point")
    ml_dir = os.path.abspath(ml_dir)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(discovery_dir, "Processed-data", "discovery_output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    nn_path = args.nn_predictions
    if nn_path and not os.path.isabs(nn_path):
        nn_path = os.path.join(discovery_dir, nn_path)

    cif_ids, _, _ = load_embeddings(emb_path)
    all_cids = set(cif_ids)
    print(f"Discovery: {len(all_cids)} structures")
    print(f"  Embeddings: {emb_path}")
    print(f"  ML models:  {ml_dir}")
    print(f"  Output:     {output_dir}")

    # Individual ML
    ml_scores = run_individual_ml(emb_path, ml_dir, output_dir)
    print(f"  ML methods run: {len(ml_scores)}")

    # NN
    nn_scores = run_nn(nn_path, output_dir, all_cids)
    if nn_scores is not None:
        print("  NN predictions loaded and top-K written")

    # Build models dict with lower=better for ensemble
    models_lower = {}
    for method, scores in ml_scores.items():
        max_s = max(scores.values())
        models_lower[method] = {cid: max_s - s for cid, s in scores.items()}
    if nn_scores is not None:
        models_lower["nn"] = nn_scores

    run_ensemble(models_lower, all_cids, output_dir, rrf_k=args.rrf_k)
    trials_by_k = collect_trial_top_k_lists(output_dir)
    write_agreement_report(output_dir, trials_by_k)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
