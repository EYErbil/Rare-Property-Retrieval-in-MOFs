#!/usr/bin/env python3
"""
Re-infer each saved ML method on a provided embeddings .npz.

Uses the same loading and prediction path as verify_ml_heatmap.py and
predict_with_embedding_classifier.py: load_embeddings, load_model_artifacts,
predict_scores. Writes test_predictions.csv (and minimal final_results.json)
into each method directory under clf_dir so downstream scripts see the new
inferences.

Usage:
  python reinfer_ml.py \\
    --embeddings_path /path/to/project/embeddings/pmt_embeddings_qmof_unlabeled.npz \\
    --clf_dir /path/to/project/embedding_classifiers/strategy_d_farthest_point
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Same predictor logic as verify_ml_heatmap / predict_with_embedding_classifier
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, PROJECT_ROOT):
    if d not in sys.path:
        sys.path.insert(0, d)
from predict_with_embedding_classifier import load_embeddings, load_model_artifacts, predict_scores


def discover_ml_methods(clf_dir):
    """Method names that have model.joblib or artifacts.joblib (no need for existing test_predictions)."""
    methods = []
    if not os.path.isdir(clf_dir):
        return methods
    for name in sorted(os.listdir(clf_dir)):
        method_dir = os.path.join(clf_dir, name)
        if not os.path.isdir(method_dir):
            continue
        if os.path.isfile(os.path.join(method_dir, "model.joblib")) or os.path.isfile(
            os.path.join(method_dir, "artifacts.joblib")
        ):
            methods.append(name)
    return methods


def main():
    parser = argparse.ArgumentParser(
        description="Re-infer each ML method on the provided embeddings .npz."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Path to embeddings .npz (must contain cif_ids + embeddings).",
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        default=None,
        help="Backward-compatible fallback: directory containing "
        "pmt_embeddings_qmof_unlabeled.npz (or the legacy Phase6_embeddings.npz).",
    )
    parser.add_argument(
        "--clf_dir",
        type=str,
        default="/path/to/embedding_classifiers/strategy_d_farthest_point",
        help="Directory with one subdir per method (model.joblib or artifacts.joblib)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional output root for re-inference files. "
            "If set, writes to <output_dir>/<method>/test_predictions.csv "
            "and final_results.json. If omitted, writes back into each method dir in --clf_dir."
        ),
    )
    args = parser.parse_args()

    npz_path = args.embeddings_path
    if npz_path is None:
        if args.npz_dir is None:
            parser.error("Provide --embeddings_path (or legacy --npz_dir).")
        npz_path = os.path.join(args.npz_dir, "pmt_embeddings_qmof_unlabeled.npz")
        if not os.path.isfile(npz_path):  # legacy archive name
            legacy = os.path.join(args.npz_dir, "Phase6_embeddings.npz")
            if os.path.isfile(legacy):
                npz_path = legacy

    npz_path = os.path.abspath(npz_path)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Embeddings npz not found: {npz_path}")

    print(f"Loading embeddings: {npz_path}")
    cif_ids, embeddings, bandgaps = load_embeddings(npz_path)
    print(f"  {len(cif_ids)} samples, dim {embeddings.shape[1]}")

    methods = discover_ml_methods(args.clf_dir)
    if not methods:
        print(f"No ML methods found under {args.clf_dir} (need model.joblib or artifacts.joblib per subdir)")
        return

    print(f"Re-inferring {len(methods)} methods: {methods}")

    for method_name in methods:
        method_dir = os.path.join(args.clf_dir, method_name)
        if args.output_dir:
            out_method_dir = Path(args.output_dir) / method_name
            out_method_dir.mkdir(parents=True, exist_ok=True)
            write_dir = str(out_method_dir)
        else:
            write_dir = method_dir
        try:
            model, scaler, pca, artifacts = load_model_artifacts(method_dir)
            if model is None and "mu_pos" not in artifacts and "knn_model" not in artifacts:
                print(f"  SKIP {method_name}: no model/artifacts")
                continue
            scores = predict_scores(embeddings, model, scaler=scaler, pca=pca, artifacts=artifacts)
        except Exception as e:
            print(f"  FAIL {method_name}: {e}")
            continue

        # Same format as embedding_classifier.save_predictions (test_predictions.csv)
        csv_path = os.path.join(write_dir, "test_predictions.csv")
        true_label = 0.0  # the screening pool has no ground truth
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("cif_id,score,predicted_binary,true_label,mode\n")
            for i, cid in enumerate(cif_ids):
                score = float(scores[i])
                pred_bin = 1 if score > 0.5 else 0
                f.write(f"{cid},{score:.6f},{pred_bin},{true_label},multiclass\n")
        print(f"  OK {method_name}: {csv_path}")

        # Minimal final_results.json so load_val_metric() etc. do not break
        results_path = os.path.join(write_dir, "final_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "method": method_name,
                "checkpoints": {"best_auc_recall_score": 0.5},
            }, f, indent=2)

    if args.output_dir:
        print(f"Done. Re-inference files written under {args.output_dir}")
    else:
        print("Done. Each method dir now has test_predictions.csv and final_results.json for the screening pool.")


if __name__ == "__main__":
    main()
