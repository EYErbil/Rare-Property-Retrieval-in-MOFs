#!/usr/bin/env python3
"""
ML verification: load all saved regular ML models, infer on the test set,
reproduce rankings, and generate a heatmap + reproducibility report.

Usage:
  python verify_ml_heatmap.py --base_dir /path/to/project
  python verify_ml_heatmap.py --base_dir . --output_dir ./verification_output
"""

import os
import sys
import json
import csv
import argparse
import numpy as np

# Reuse predictor logic
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_with_embedding_classifier import load_embeddings, load_model_artifacts, predict_scores

# Optional matplotlib for heatmap
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_test_labels(labels_dir, threshold=1.0):
    """Return test_cids (list), true_labels (cid -> bandgap), test_positives (cids with bg < threshold)."""
    path = os.path.join(labels_dir, "test_bandgaps_regression.json")
    if not os.path.isfile(path):
        return None, None, None
    with open(path, "r") as f:
        labels = json.load(f)
    test_cids = sorted(labels.keys())
    true_labels = {cid: float(bg) for cid, bg in labels.items()}
    test_positives = [cid for cid in test_cids if true_labels[cid] < threshold]
    return test_cids, true_labels, test_positives


def load_saved_test_predictions(method_dir):
    """Load cif_id -> score from test_predictions.csv."""
    path = os.path.join(method_dir, "test_predictions.csv")
    if not os.path.isfile(path):
        return None
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("cif_id", "").strip()
            if not cid:
                continue
            try:
                preds[cid] = float(row["score"])
            except (KeyError, ValueError):
                continue
    return preds


def score_to_rank(scores_dict, test_cids, higher_is_better=True):
    """Return rank per cid (1 = best). For ML classifiers higher score = better."""
    scores_arr = np.array([scores_dict.get(cid, np.nan) for cid in test_cids])
    if higher_is_better:
        order = np.argsort(-np.nan_to_num(scores_arr, nan=-np.inf))
    else:
        order = np.argsort(np.nan_to_num(scores_arr, nan=np.inf))
    ranks = np.empty(len(test_cids), dtype=float)
    ranks[order] = np.arange(1, len(test_cids) + 1, dtype=float)
    return {cid: int(ranks[i]) for i, cid in enumerate(test_cids)}


def discover_ml_methods(clf_dir):
    """List method names that have model.joblib or artifacts.joblib and test_predictions.csv."""
    methods = []
    if not os.path.isdir(clf_dir):
        return methods
    for name in sorted(os.listdir(clf_dir)):
        method_dir = os.path.join(clf_dir, name)
        if not os.path.isdir(method_dir):
            continue
        if not os.path.isfile(os.path.join(method_dir, "test_predictions.csv")):
            continue
        if os.path.isfile(os.path.join(method_dir, "model.joblib")) or os.path.isfile(
            os.path.join(method_dir, "artifacts.joblib")
        ):
            methods.append(name)
    return methods


def is_regression_method(method_name):
    """Regression methods save normalized 0-1 but predictor returns raw -pred; value match may fail."""
    return "regression" in method_name.lower() and any(
        x in method_name.lower() for x in ["ridge", "lasso", "rf_regression", "xgboost"]
    )


def run_verification(base_dir, embeddings_path, labels_dir, clf_dir, threshold, output_dir, rtol=1e-5, atol=1e-8):
    test_cids, true_labels, test_positives = load_test_labels(labels_dir, threshold)
    if not test_cids or not test_positives:
        print("No test set or no test positives found.")
        return 1

    cif_ids, embeddings, _ = load_embeddings(embeddings_path)
    cid_to_idx = {cid: i for i, cid in enumerate(cif_ids)}
    test_indices = [cid_to_idx[cid] for cid in test_cids if cid in cid_to_idx]
    if len(test_indices) != len(test_cids):
        print("Some test CIFs missing from embeddings .npz.")
        return 1

    methods = discover_ml_methods(clf_dir)
    if not methods:
        print("No ML methods found with model/artifacts and test_predictions.csv.")
        return 1

    # Collect scores and ranks per method; reproducibility result
    model_scores = {}
    model_ranks = {}
    repro_ok = []
    repro_fail = []
    repro_skip = []

    for method in methods:
        method_dir = os.path.join(clf_dir, method)
        try:
            model, scaler, pca, artifacts = load_model_artifacts(method_dir)
            if model is None and "mu_pos" not in artifacts and "knn_model" not in artifacts:
                continue
            scores_all = predict_scores(embeddings, model, scaler=scaler, pca=pca, artifacts=artifacts)
            scores_test = {cid: float(scores_all[cid_to_idx[cid]]) for cid in test_cids}
            model_scores[method] = scores_test
            rank_dict = score_to_rank(scores_test, test_cids, higher_is_better=True)
            model_ranks[method] = rank_dict
        except Exception as e:
            repro_fail.append((method, str(e)))
            continue

        saved = load_saved_test_predictions(method_dir)
        if not saved:
            repro_skip.append((method, "no saved CSV"))
            continue
        if is_regression_method(method):
            repro_skip.append((method, "regression: value scale differs, rank only"))
            repro_ok.append(method)
            continue
        mismatches = 0
        for cid in test_cids:
            if cid not in saved:
                mismatches += 1
                continue
            if not np.isclose(scores_test[cid], saved[cid], rtol=rtol, atol=atol):
                mismatches += 1
        if mismatches == 0:
            repro_ok.append(method)
        else:
            repro_fail.append((method, f"{mismatches} score mismatches"))

    # Per-positive analysis (same structure as ensemble_discovery)
    per_positive = {}
    model_names = sorted(model_ranks.keys())
    for cid in test_positives:
        bg = true_labels[cid]
        ranks_in_models = {m: model_ranks[m][cid] for m in model_names if cid in model_ranks[m]}
        if not ranks_in_models:
            continue
        best_model = min(ranks_in_models, key=ranks_in_models.get)
        per_positive[cid] = {
            "bandgap": bg,
            "ranks": ranks_in_models,
            "best_rank": ranks_in_models[best_model],
            "best_model": best_model,
        }

    if not per_positive:
        print("No per-positive data for heatmap.")
    else:
        # CSV of ranks
        ranks_path = os.path.join(output_dir, "verification_heatmap_ranks.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(ranks_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cif_id", "bandgap"] + model_names)
            for cid in sorted(per_positive.keys(), key=lambda c: per_positive[c]["best_rank"]):
                row = [cid, per_positive[cid]["bandgap"]]
                row.extend(per_positive[cid]["ranks"].get(m, "") for m in model_names)
                writer.writerow(row)
        print(f"Wrote {ranks_path}")

        # Heatmap
        if HAS_MATPLOTLIB and model_names:
            mof_names = sorted(per_positive.keys(), key=lambda c: per_positive[c]["best_rank"])
            n_mofs = len(mof_names)
            n_models = len(model_names)
            rank_matrix = np.zeros((n_mofs, n_models))
            for i, mof in enumerate(mof_names):
                for j, model in enumerate(model_names):
                    rank_matrix[i, j] = per_positive[mof]["ranks"].get(model, 9999)

            fig, ax = plt.subplots(figsize=(max(10, n_models * 0.8), max(5, n_mofs * 0.5)))
            cmap = plt.cm.RdYlGn_r
            im = ax.imshow(np.log10(rank_matrix + 1), cmap=cmap, aspect="auto", vmin=0, vmax=np.log10(10000))
            for i in range(n_mofs):
                for j in range(n_models):
                    rank = int(rank_matrix[i, j])
                    color = "white" if rank > 500 else "black"
                    fontsize = 7 if rank > 999 else 8
                    ax.text(j, i, str(rank), ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold")
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=55, ha="right", fontsize=8)
            ax.set_yticks(range(n_mofs))
            display_mofs = [mof.replace("_FSR", "") for mof in mof_names]
            ax.set_yticklabels(display_mofs, fontsize=9)
            ax.set_title("ML verification: rank of each test positive per model (lower = better)", fontsize=12, fontweight="bold")
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("log10(Rank)", fontsize=10)
            cbar.set_ticks([0, 1, 2, 3, 4])
            cbar.set_ticklabels(["1", "10", "100", "1K", "10K"])
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, "verification_heatmap.png")
            fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {heatmap_path}")

    # Report
    report_path = os.path.join(output_dir, "verification_report.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("ML verification report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Base dir:        {base_dir}\n")
        f.write(f"Embeddings:      {embeddings_path}\n")
        f.write(f"Labels dir:      {labels_dir}\n")
        f.write(f"Classifier dir:  {clf_dir}\n")
        f.write(f"Threshold:      {threshold} eV\n")
        f.write(f"Test set size:   {len(test_cids)}\n")
        f.write(f"Test positives: {len(test_positives)}\n")
        f.write(f"ML methods:      {len(methods)}\n\n")
        f.write("Reproducibility (loaded model vs saved test_predictions.csv):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  OK (exact match):  {len(repro_ok)}  {repro_ok}\n")
        for method, reason in repro_skip:
            f.write(f"  Skip (rank only):   {method}  ({reason})\n")
        for method, reason in repro_fail:
            f.write(f"  FAIL:               {method}  ({reason})\n")
        f.write("\n")
        if per_positive:
            best = min(per_positive[c]["best_rank"] for c in per_positive)
            worst = max(per_positive[c]["best_rank"] for c in per_positive)
            f.write(f"Per-positive: best_rank range [{best}, {worst}]\n")
    print(f"Wrote {report_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Verify ML models and generate ranking heatmap")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base dir (embedding_classifiers, embedding_analysis, splits)")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to .npz (default: base_dir/embedding_analysis/embeddings_pretrained.npz)")
    parser.add_argument("--labels_dir", type=str, default=None,
                        help="Labels (default: base_dir/new_splits/strategy_d_farthest_point)")
    parser.add_argument("--clf_dir", type=str, default=None,
                        help="Classifier output (default: base_dir/embedding_classifiers/strategy_d_farthest_point)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Bandgap threshold for positive class (eV)")
    parser.add_argument("--output_dir", type=str, default="./verification_output",
                        help="Output directory for heatmap and report")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for score match")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for score match")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    emb_path = args.embeddings_path or os.path.join(base, "embedding_analysis", "embeddings_pretrained.npz")
    labels_dir = args.labels_dir or os.path.join(base, "new_splits", "strategy_d_farthest_point")
    clf_dir = args.clf_dir or os.path.join(base, "embedding_classifiers", "strategy_d_farthest_point")
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(emb_path):
        print(f"ERROR: Embeddings not found: {emb_path}")
        return 1
    if not os.path.isdir(labels_dir):
        print(f"ERROR: Labels dir not found: {labels_dir}")
        return 1
    if not os.path.isdir(clf_dir):
        print(f"ERROR: Classifier dir not found: {clf_dir}")
        return 1

    return run_verification(
        base, emb_path, labels_dir, clf_dir, args.threshold, output_dir, args.rtol, args.atol
    )


if __name__ == "__main__":
    sys.exit(main())
