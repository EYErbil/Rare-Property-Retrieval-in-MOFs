#!/usr/bin/env python3
"""
Run a saved embedding classifier (Random Forest, Extra Trees, etc.) on new embeddings.
Use this for discovery: train with embedding_classifier.py, then score a new .npz
(e.g. from analyze_embeddings on a candidate set) with this script.

Requires: embeddings .npz with keys 'cif_ids' and 'embeddings' (same 768-dim format
as embeddings_pretrained.npz). Optional: 'bandgaps' for evaluation.

Usage:
  python predict_with_embedding_classifier.py \\
      --model_dir ./embedding_classifiers/random_forest \\
      --embeddings_path ./discovery_embeddings.npz \\
      --output ./discovery_scores.csv

  # Score with Extra Trees and write top-100 CIFs
  python predict_with_embedding_classifier.py \\
      --model_dir ./embedding_classifiers/extra_trees \\
      --embeddings_path ./discovery_embeddings.npz \\
      --output ./discovery_scores.csv \\
      --top_k 100
"""

import os
import argparse
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None


def load_embeddings(npz_path):
    """Load cif_ids and embeddings from .npz (optional bandgaps)."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = list(data['cif_ids'])
    embeddings = np.asarray(data['embeddings'])
    bandgaps = data['bandgaps'] if 'bandgaps' in data else None
    return cif_ids, embeddings, bandgaps


def load_model_artifacts(model_dir):
    """Load model, scaler, pca, and optional artifacts from a method directory."""
    model_path = os.path.join(model_dir, 'model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    pca_path = os.path.join(model_dir, 'pca.joblib')
    artifacts_path = os.path.join(model_dir, 'artifacts.joblib')

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
    artifacts = joblib.load(artifacts_path) if os.path.exists(artifacts_path) else {}

    return model, scaler, pca, artifacts


def predict_mahalanobis(embeddings, scaler, artifacts):
    """Score using Mahalanobis distance to positive class (saved in artifacts)."""
    mu_pos = artifacts['mu_pos']
    cov_inv = artifacts['cov_inv']
    X = scaler.transform(embeddings)
    diff = X - mu_pos
    mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    return -mahal  # lower distance -> higher score


def predict_two_stage(embeddings, scaler, artifacts):
    """Score using two-stage pipeline: kNN pre-filter then ExtraTrees re-rank."""
    knn = artifacts['knn_model']
    et = artifacts['et_model']
    prefilter_k = int(artifacts['prefilter_k'])
    X = scaler.transform(embeddings)
    knn_probs = knn.predict_proba(X)[:, 1]
    top_indices = np.argsort(-knn_probs)[:prefilter_k]
    X_cand = X[top_indices]
    et_probs = et.predict_proba(X_cand)[:, 1]
    knn_filt = knn_probs[top_indices]
    knn_n = (knn_filt - knn_filt.min()) / (knn_filt.max() - knn_filt.min() + 1e-12)
    et_n = (et_probs - et_probs.min()) / (et_probs.max() - et_probs.min() + 1e-12)
    combined = 0.4 * knn_n + 0.6 * et_n
    full_scores = np.zeros(len(embeddings), dtype=np.float64)
    full_scores[top_indices] = combined + 1.0
    return full_scores


def predict_scores(embeddings, model, scaler=None, pca=None, artifacts=None):
    """Get discovery scores (higher = more positive-like)."""
    artifacts = artifacts or {}
    # Mahalanobis: no sklearn model, use scaler + mu_pos, cov_inv
    if 'mu_pos' in artifacts and 'cov_inv' in artifacts and scaler is not None:
        return predict_mahalanobis(embeddings, scaler, artifacts)
    # Two-stage: kNN + ExtraTrees saved in artifacts
    if 'knn_model' in artifacts and 'et_model' in artifacts and scaler is not None:
        return predict_two_stage(embeddings, scaler, artifacts)

    X = np.asarray(embeddings, dtype=np.float64)
    # Feature-selected: apply top_idx mask first
    if 'top_idx' in artifacts:
        X = X[:, np.asarray(artifacts['top_idx'])]
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)

    if model is None:
        raise ValueError("No model and no Mahalanobis/two_stage artifacts in model_dir")

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
        return probs
    # Regression models: predicted bandgap; lower bg = more positive -> score = -pred
    preds = model.predict(X)
    return -np.asarray(preds, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description='Run a saved embedding classifier on new embeddings (discovery).')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to method directory (e.g. embedding_classifiers/random_forest) '
                             'containing model.joblib, scaler.joblib, etc.')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to .npz with cif_ids and embeddings (768-dim)')
    parser.add_argument('--output', type=str, default='discovery_scores.csv',
                        help='Output CSV: cif_id, score [, bandgap if in .npz]')
    parser.add_argument('--top_k', type=int, default=None,
                        help='If set, also print top-k CIF IDs by score to stdout')
    args = parser.parse_args()

    if joblib is None:
        raise RuntimeError("pip install joblib to use this script")

    print(f"Loading embeddings from {args.embeddings_path}")
    cif_ids, embeddings, bandgaps = load_embeddings(args.embeddings_path)
    print(f"  {len(cif_ids)} samples, embedding dim {embeddings.shape[1]}")

    print(f"Loading model from {args.model_dir}")
    model, scaler, pca, artifacts = load_model_artifacts(args.model_dir)
    if model is None and 'mu_pos' not in artifacts and 'knn_model' not in artifacts:
        raise FileNotFoundError(
            f"No model.joblib and no Mahalanobis/two_stage artifacts in {args.model_dir}. "
            "Train with embedding_classifier.py first (models are now saved automatically).")

    print("Computing scores...")
    scores = predict_scores(embeddings, model, scaler=scaler, pca=pca, artifacts=artifacts)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        header = "cif_id,score"
        if bandgaps is not None:
            header += ",bandgap"
        f.write(header + "\n")
        for i, cid in enumerate(cif_ids):
            row = f"{cid},{scores[i]:.6f}"
            if bandgaps is not None:
                row += f",{bandgaps[i]:.4f}"
            f.write(row + "\n")
    print(f"Wrote {args.output}")

    if args.top_k is not None and args.top_k > 0:
        order = np.argsort(-scores)[: args.top_k]
        print(f"\nTop-{args.top_k} by score:")
        for idx in order:
            line = f"  {cif_ids[idx]}  {scores[idx]:.6f}"
            if bandgaps is not None:
                line += f"  (bg={bandgaps[idx]:.3f})"
            print(line)


if __name__ == '__main__':
    main()
