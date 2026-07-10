#!/usr/bin/env python3
"""
k-NN Baseline & Hybrid Ranking for MOF Discovery
===================================================

Three zero-training approaches that leverage pretrained embeddings:

1. k-NN REGRESSION BASELINE
   - For each test MOF, find K nearest training MOFs by cosine similarity
   - Predict bandgap as distance-weighted average of neighbors' bandgaps
   - Rank test MOFs by predicted bandgap (lowest = most likely positive)
   - No GPU training needed — runs in seconds

2. SIMILARITY-TO-POSITIVE RANKING
   - For each test MOF, compute max cosine similarity to ANY train positive
   - Rank by: highest similarity to a known positive → most likely positive
   - Pure structural matching, no bandgap prediction

3. HYBRID RANKING
   - Combine neural network predictions with k-NN structural similarity
   - hybrid_score = alpha * normalized_nn_score + (1-alpha) * knn_score
   - Different alpha values tested automatically
   - Finds the sweet spot between learned patterns and structural matching

4. NOVELTY-AWARE RANKING
   - Flag test MOFs that are far from ALL training data
   - For these "novel" MOFs, the NN prediction is unreliable
   - Boost novel MOFs in ranking (they're worth investigating precisely
     because the model is uncertain about them)

Usage:
  # k-NN baseline only (needs embeddings from analyze_embeddings.py):
  python knn_baseline.py --embeddings_path ./embedding_analysis/embeddings_pretrained.npz

  # Hybrid with NN predictions:
  python knn_baseline.py --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --predictions_dir ./experiments/exp233_best \\
      --alpha 0.3

  # Search over alpha values:
  python knn_baseline.py --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --predictions_dir ./experiments/exp233_best \\
      --sweep_alpha

  # Multi-model hybrid (combine multiple NN models with k-NN):
  python knn_baseline.py --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --predictions_dir ./experiments/exp233_best ./experiments/exp201_best \\
      --sweep_alpha
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
from collections import defaultdict


# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings(npz_path):
    """Load embeddings from analyze_embeddings.py output."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = list(data['cif_ids'])
    embeddings = data['embeddings']
    bandgaps = data['bandgaps']
    splits = list(data['splits'])
    return cif_ids, embeddings, bandgaps, splits


def override_splits_from_labels(cif_ids, bandgaps, labels_dir):
    """Override split assignments by reading label JSON files from labels_dir.
    
    The labels_dir should contain:
      train_bandgaps_regression.json  (cif_id -> bandgap)
      val_bandgaps_regression.json
      test_bandgaps_regression.json
    
    CIF IDs present in a label file are assigned to that split.
    CIF IDs not in any label file are excluded (split='unused').
    Bandgap values are also updated from the label files.
    """
    new_splits = ['unused'] * len(cif_ids)
    new_bandgaps = bandgaps.copy()
    cid_to_idx = {cid: i for i, cid in enumerate(cif_ids)}
    
    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(labels_dir, f'{split_name}_bandgaps_regression.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: {json_path} not found, skipping {split_name} split")
            continue
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        matched = 0
        for cid, bg in label_data.items():
            if cid in cid_to_idx:
                idx = cid_to_idx[cid]
                new_splits[idx] = split_name
                new_bandgaps[idx] = float(bg)
                matched += 1
        n_pos = sum(1 for bg in label_data.values() if float(bg) < 1.0)
        print(f"  {split_name}: {len(label_data)} in labels, {matched} matched to embeddings, {n_pos} positives")
    
    assigned = sum(1 for s in new_splits if s != 'unused')
    print(f"  Total assigned: {assigned}/{len(cif_ids)}")
    return new_splits, new_bandgaps


def load_nn_predictions(csv_path):
    """Load neural network predictions from test_predictions.csv."""
    preds = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['cif_id']
            preds[cid] = {
                'score': float(row['score']),
                'true_label': float(row['true_label']),
            }
    return preds


def cosine_similarity_matrix(A, B):
    """Cosine similarity between rows of A and rows of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


# =============================================================================
# RANKING METRICS
# =============================================================================

def compute_ranking_metrics(cif_ids, scores, true_bandgaps, threshold=1.0,
                            Ks=[10, 25, 50, 100, 200]):
    """
    Compute standard discovery ranking metrics.
    Lower score = predicted as more likely positive (lower bandgap).
    """
    # Sort by score (ascending — lowest predicted bandgap first)
    order = np.argsort(scores)
    sorted_cids = [cif_ids[i] for i in order]
    sorted_true = true_bandgaps[order]
    is_positive = sorted_true < threshold

    n_total = len(scores)
    n_positive = int(is_positive.sum())
    prevalence = n_positive / n_total if n_total > 0 else 0

    metrics = {
        'n_total': n_total,
        'n_positive': n_positive,
        'prevalence': prevalence,
    }

    # Recall@K, Precision@K
    for K in Ks:
        if K > n_total:
            continue
        top_k_pos = int(is_positive[:K].sum())
        metrics[f'recall@{K}'] = top_k_pos / n_positive if n_positive > 0 else 0
        metrics[f'precision@{K}'] = top_k_pos / K
        metrics[f'enrichment@{K}'] = (top_k_pos / K) / prevalence if prevalence > 0 else 0
        metrics[f'hits@{K}'] = top_k_pos

    # First hit rank
    hit_ranks = np.where(is_positive)[0] + 1  # 1-indexed
    if len(hit_ranks) > 0:
        metrics['first_hit_rank'] = int(hit_ranks[0])
        metrics['median_hit_rank'] = int(np.median(hit_ranks))
        metrics['mean_hit_rank'] = float(np.mean(hit_ranks))
        metrics['last_hit_rank'] = int(hit_ranks[-1])
        metrics['mrr'] = float(np.mean(1.0 / hit_ranks))
    else:
        metrics['first_hit_rank'] = n_total
        metrics['mrr'] = 0

    # Spearman rank correlation
    from scipy.stats import spearmanr, kendalltau
    rho, p_rho = spearmanr(scores, true_bandgaps)
    tau, p_tau = kendalltau(scores, true_bandgaps)
    metrics['spearman_rho'] = float(rho) if not np.isnan(rho) else 0
    metrics['kendall_tau'] = float(tau) if not np.isnan(tau) else 0

    # Which positives were found in top-K?
    for K in Ks:
        found = []
        for i in range(min(K, n_total)):
            if is_positive[i]:
                found.append((sorted_cids[i], float(sorted_true[i]), i + 1))
        metrics[f'found_in_top_{K}'] = found

    return metrics, sorted_cids, sorted_true


# =============================================================================
# METHOD 1: k-NN REGRESSION
# =============================================================================

def knn_regression(test_embs, train_embs, train_bandgaps, K=10, distance_weighting=True):
    """
    Predict bandgap for test MOFs using K nearest train neighbors.

    Args:
        test_embs: [N_test, D] test embeddings
        train_embs: [N_train, D] train embeddings
        train_bandgaps: [N_train] true bandgaps
        K: number of neighbors
        distance_weighting: weight by similarity (True) or uniform (False)

    Returns:
        predictions: [N_test] predicted bandgaps
        nn_info: list of dicts with neighbor details
    """
    sim = cosine_similarity_matrix(test_embs, train_embs)  # [N_test, N_train]

    predictions = np.zeros(len(test_embs))
    nn_info = []

    for i in range(len(test_embs)):
        # Top-K most similar train MOFs
        top_k_idx = np.argsort(-sim[i])[:K]
        top_k_sims = sim[i, top_k_idx]
        top_k_bgs = train_bandgaps[top_k_idx]

        if distance_weighting:
            # Weight by similarity (higher sim → more influence)
            weights = np.maximum(top_k_sims, 0)  # clip negative similarities
            if weights.sum() > 0:
                predictions[i] = np.average(top_k_bgs, weights=weights)
            else:
                predictions[i] = np.mean(top_k_bgs)
        else:
            predictions[i] = np.mean(top_k_bgs)

        nn_info.append({
            'neighbor_ids': top_k_idx.tolist(),
            'neighbor_sims': top_k_sims.tolist(),
            'neighbor_bgs': top_k_bgs.tolist(),
        })

    return predictions, nn_info


# =============================================================================
# METHOD 2: SIMILARITY TO POSITIVE
# =============================================================================

def similarity_to_positive_ranking(test_embs, train_pos_embs):
    """
    Score each test MOF by max cosine similarity to any train positive.
    Lower score (negative of max similarity) → rank higher.

    Returns:
        scores: [N_test] — negative of max-similarity (lower = more similar to positive)
    """
    sim = cosine_similarity_matrix(test_embs, train_pos_embs)  # [N_test, N_pos]
    max_sim = sim.max(axis=1)  # [N_test]
    # Negate so that "lower score = more likely positive" convention holds
    return -max_sim


# =============================================================================
# METHOD 3: HYBRID RANKING
# =============================================================================

def hybrid_ranking(nn_scores, knn_scores, alpha=0.5):
    """
    Combine NN predictions with k-NN scores.

    hybrid = alpha * normalized_nn + (1-alpha) * normalized_knn

    Both scores follow "lower = more likely positive" convention.
    """
    # Normalize both to [0, 1] range
    nn_norm = (nn_scores - nn_scores.min()) / (nn_scores.max() - nn_scores.min() + 1e-12)
    knn_norm = (knn_scores - knn_scores.min()) / (knn_scores.max() - knn_scores.min() + 1e-12)

    return alpha * nn_norm + (1 - alpha) * knn_norm


# =============================================================================
# METHOD 4: NOVELTY-AWARE RANKING
# =============================================================================

def novelty_aware_ranking(test_embs, train_embs, nn_scores, novelty_boost=0.5):
    """
    Boost test MOFs that are far from ALL training data.

    Rationale: if a MOF is structurally novel (low max similarity to any
    training sample), the NN prediction is unreliable. We should investigate
    these MOFs because they represent unexplored chemistry.

    Score = nn_score - novelty_boost * (1 - max_sim_to_train)

    Lower score → higher rank. Novel MOFs get score reduction (boosted up).
    """
    sim = cosine_similarity_matrix(test_embs, train_embs)
    max_sim = sim.max(axis=1)  # max similarity to ANY train sample

    # Novelty: how far from training distribution
    novelty = 1 - max_sim  # higher = more novel

    nn_norm = (nn_scores - nn_scores.min()) / (nn_scores.max() - nn_scores.min() + 1e-12)

    return nn_norm - novelty_boost * novelty


# =============================================================================
# REPORT
# =============================================================================

def print_metrics_report(method_name, metrics, width=80):
    """Print a formatted ranking metrics report."""
    print(f"\n{'='*width}")
    print(f"  {method_name}")
    print(f"{'='*width}")
    print(f"  Total: {metrics['n_total']}  |  Positives: {metrics['n_positive']}  |  "
          f"Prevalence: {metrics['prevalence']:.4f}")
    print(f"  First Hit Rank: {metrics.get('first_hit_rank', 'N/A')}")
    print(f"  MRR: {metrics.get('mrr', 0):.4f}")
    print(f"  Spearman rho: {metrics.get('spearman_rho', 0):.4f}")
    print(f"  Kendall tau: {metrics.get('kendall_tau', 0):.4f}")

    print(f"\n  {'K':>5s}  {'Hits':>5s}  {'Recall':>8s}  {'Precision':>10s}  {'Enrichment':>11s}")
    print(f"  {'-'*45}")
    for K in [10, 25, 50, 100, 200]:
        key = f'recall@{K}'
        if key in metrics:
            print(f"  {K:>5d}  {metrics[f'hits@{K}']:>5d}  "
                  f"{metrics[f'recall@{K}']:>8.3f}  "
                  f"{metrics[f'precision@{K}']:>10.4f}  "
                  f"{metrics[f'enrichment@{K}']:>11.1f}x")

    # Show which positives were found
    for K in [25, 50, 100, 200]:
        found_key = f'found_in_top_{K}'
        if found_key in metrics and metrics[found_key]:
            found_str = ', '.join(f"{cid}({bg:.3f}eV,rank={r})"
                                  for cid, bg, r in metrics[found_key])
            print(f"\n  Found in top-{K}: {found_str}")

    print(f"{'='*width}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='k-NN Baseline & Hybrid Ranking for MOF Discovery')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to embeddings .npz from analyze_embeddings.py')
    parser.add_argument('--predictions_dir', type=str, nargs='*', default=None,
                        help='Path(s) to experiment dir(s) containing test_predictions.csv '
                             '(for hybrid mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as embeddings)')
    parser.add_argument('--K', type=int, default=10,
                        help='Number of neighbors for k-NN regression')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive class (eV)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Hybrid weight: alpha * NN + (1-alpha) * kNN')
    parser.add_argument('--sweep_alpha', action='store_true',
                        help='Sweep alpha values to find optimal hybrid weight')
    parser.add_argument('--novelty_boost', type=float, default=0.3,
                        help='How much to boost novel MOFs in ranking')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Override split assignments from label JSON files in this '
                             'directory. Use for embedding-informed splits (D/E/F). '
                             'Expects {train,val,test}_bandgaps_regression.json files.')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.embeddings_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    cif_ids, embeddings, bandgaps, splits = load_embeddings(args.embeddings_path)

    # Override splits if labels_dir provided
    if args.labels_dir is not None:
        print(f"Overriding splits from: {args.labels_dir}")
        splits, bandgaps = override_splits_from_labels(cif_ids, bandgaps, args.labels_dir)

    # Separate splits
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])
    pos_mask = bandgaps < args.threshold

    train_idx = np.where(train_mask)[0]
    train_pos_idx = np.where(train_mask & pos_mask)[0]
    test_idx = np.where(test_mask)[0]

    train_embs = embeddings[train_idx]
    train_pos_embs = embeddings[train_pos_idx]
    train_bgs = bandgaps[train_idx]
    test_embs = embeddings[test_idx]
    test_bgs = bandgaps[test_idx]
    test_cids = [cif_ids[i] for i in test_idx]

    n_test_pos = int((test_bgs < args.threshold).sum())
    print(f"  Train: {len(train_idx)} ({len(train_pos_idx)} pos)")
    print(f"  Test:  {len(test_idx)} ({n_test_pos} pos)")

    all_results = {}

    # =========================================================================
    # METHOD 1: k-NN Regression
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# METHOD 1: k-NN Regression (K={args.K})")
    print(f"{'#'*70}")

    for K in [5, 10, 20, 50]:
        knn_preds, _ = knn_regression(test_embs, train_embs, train_bgs,
                                       K=K, distance_weighting=True)
        metrics, _, _ = compute_ranking_metrics(test_cids, knn_preds, test_bgs,
                                                args.threshold)
        print_metrics_report(f"k-NN Regression (K={K}, distance-weighted)", metrics)
        all_results[f'knn_K{K}'] = metrics

    # Also uniform weighting
    knn_preds_uniform, _ = knn_regression(test_embs, train_embs, train_bgs,
                                           K=10, distance_weighting=False)
    metrics_uniform, _, _ = compute_ranking_metrics(test_cids, knn_preds_uniform,
                                                     test_bgs, args.threshold)
    print_metrics_report("k-NN Regression (K=10, uniform)", metrics_uniform)
    all_results['knn_K10_uniform'] = metrics_uniform

    # =========================================================================
    # METHOD 2: Similarity to Positive
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# METHOD 2: Max Similarity to Train Positive")
    print(f"{'#'*70}")

    sim_scores = similarity_to_positive_ranking(test_embs, train_pos_embs)
    metrics_sim, _, _ = compute_ranking_metrics(test_cids, sim_scores, test_bgs,
                                                 args.threshold)
    print_metrics_report("Similarity to Nearest Train Positive", metrics_sim)
    all_results['sim_to_positive'] = metrics_sim

    # =========================================================================
    # METHOD 3: Hybrid (if NN predictions available)
    # =========================================================================
    if args.predictions_dir:
        print(f"\n{'#'*70}")
        print(f"# METHOD 3: Hybrid NN + k-NN")
        print(f"{'#'*70}")

        for pred_dir in args.predictions_dir:
            csv_path = os.path.join(pred_dir, 'test_predictions.csv')
            if not os.path.exists(csv_path):
                print(f"  WARNING: {csv_path} not found, skipping")
                continue

            exp_name = os.path.basename(pred_dir)
            print(f"\n  Loading NN predictions from {exp_name}...")
            nn_preds = load_nn_predictions(csv_path)

            # Match test ordering
            nn_scores = np.array([nn_preds.get(cid, {}).get('score', 5.0)
                                  for cid in test_cids])

            # NN-only baseline
            metrics_nn, _, _ = compute_ranking_metrics(test_cids, nn_scores, test_bgs,
                                                        args.threshold)
            print_metrics_report(f"NN Only ({exp_name})", metrics_nn)
            all_results[f'nn_{exp_name}'] = metrics_nn

            # k-NN score for hybrid
            knn_scores, _ = knn_regression(test_embs, train_embs, train_bgs,
                                            K=args.K, distance_weighting=True)

            if args.sweep_alpha:
                # Sweep alpha values
                print(f"\n  Alpha sweep for {exp_name}:")
                print(f"  {'Alpha':>6s}  {'FHR':>5s}  {'R@25':>6s}  {'R@50':>6s}  "
                      f"{'R@100':>6s}  {'R@200':>6s}  {'Spearman':>8s}")
                print(f"  {'-'*55}")

                best_alpha = 0
                best_recall_100 = 0

                for alpha_pct in range(0, 105, 5):
                    alpha = alpha_pct / 100.0
                    hybrid = hybrid_ranking(nn_scores, knn_scores, alpha)
                    m, _, _ = compute_ranking_metrics(test_cids, hybrid, test_bgs,
                                                       args.threshold)
                    r25 = m.get('recall@25', 0)
                    r50 = m.get('recall@50', 0)
                    r100 = m.get('recall@100', 0)
                    r200 = m.get('recall@200', 0)
                    fhr = m.get('first_hit_rank', 9999)
                    rho = m.get('spearman_rho', 0)

                    marker = ""
                    if r100 > best_recall_100 or (r100 == best_recall_100 and fhr < m.get('first_hit_rank', 9999)):
                        best_recall_100 = r100
                        best_alpha = alpha
                        marker = " <-- best"

                    print(f"  {alpha:>6.2f}  {fhr:>5d}  {r25:>6.3f}  {r50:>6.3f}  "
                          f"{r100:>6.3f}  {r200:>6.3f}  {rho:>8.4f}{marker}")

                # Full report for best alpha
                hybrid_best = hybrid_ranking(nn_scores, knn_scores, best_alpha)
                metrics_hybrid, _, _ = compute_ranking_metrics(
                    test_cids, hybrid_best, test_bgs, args.threshold)
                print_metrics_report(
                    f"Hybrid NN+kNN ({exp_name}, alpha={best_alpha:.2f})",
                    metrics_hybrid)
                all_results[f'hybrid_{exp_name}_alpha{best_alpha:.2f}'] = metrics_hybrid
            else:
                # Single alpha
                hybrid = hybrid_ranking(nn_scores, knn_scores, args.alpha)
                metrics_hybrid, _, _ = compute_ranking_metrics(
                    test_cids, hybrid, test_bgs, args.threshold)
                print_metrics_report(
                    f"Hybrid NN+kNN ({exp_name}, alpha={args.alpha:.2f})",
                    metrics_hybrid)
                all_results[f'hybrid_{exp_name}'] = metrics_hybrid

            # =========================================================
            # METHOD 4: Novelty-aware
            # =========================================================
            print(f"\n  --- Novelty-aware ranking ({exp_name}) ---")
            for nb in [0.1, 0.3, 0.5, 1.0]:
                novelty_scores = novelty_aware_ranking(
                    test_embs, train_embs, nn_scores, novelty_boost=nb)
                m_nov, _, _ = compute_ranking_metrics(
                    test_cids, novelty_scores, test_bgs, args.threshold)
                r100 = m_nov.get('recall@100', 0)
                fhr = m_nov.get('first_hit_rank', 9999)
                print(f"  novelty_boost={nb:.1f}: FHR={fhr}, R@100={r100:.3f}, "
                      f"R@200={m_nov.get('recall@200', 0):.3f}")
                all_results[f'novelty_{exp_name}_nb{nb}'] = m_nov

    # =========================================================================
    # MULTI-MODEL HYBRID (if multiple prediction dirs)
    # =========================================================================
    if args.predictions_dir and len(args.predictions_dir) > 1:
        print(f"\n{'#'*70}")
        print(f"# MULTI-MODEL HYBRID: Average NN scores + k-NN")
        print(f"{'#'*70}")

        # Average NN scores across models
        all_nn_scores = []
        for pred_dir in args.predictions_dir:
            csv_path = os.path.join(pred_dir, 'test_predictions.csv')
            if not os.path.exists(csv_path):
                continue
            nn_preds = load_nn_predictions(csv_path)
            scores = np.array([nn_preds.get(cid, {}).get('score', 5.0)
                               for cid in test_cids])
            # Normalize to [0,1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            all_nn_scores.append(scores)

        if len(all_nn_scores) > 1:
            avg_nn = np.mean(all_nn_scores, axis=0)
            knn_scores, _ = knn_regression(test_embs, train_embs, train_bgs,
                                            K=args.K, distance_weighting=True)

            for alpha_pct in [0, 10, 20, 30, 50, 70, 100]:
                alpha = alpha_pct / 100.0
                hybrid = hybrid_ranking(avg_nn, knn_scores, alpha)
                m, _, _ = compute_ranking_metrics(test_cids, hybrid, test_bgs,
                                                   args.threshold)
                print(f"  Multi-model alpha={alpha:.2f}: FHR={m['first_hit_rank']}, "
                      f"R@50={m.get('recall@50', 0):.3f}, "
                      f"R@100={m.get('recall@100', 0):.3f}, "
                      f"R@200={m.get('recall@200', 0):.3f}")

    # =========================================================================
    # SAVE k-NN PREDICTIONS (so they can be used in ensemble_discovery.py)
    # =========================================================================
    print(f"\n{'#'*70}")
    print(f"# Saving k-NN predictions as test_predictions.csv")
    print(f"{'#'*70}")

    knn_preds_final, _ = knn_regression(test_embs, train_embs, train_bgs,
                                         K=args.K, distance_weighting=True)

    knn_csv_path = os.path.join(args.output_dir, 'test_predictions.csv')
    with open(knn_csv_path, 'w', newline='') as f:
        f.write("cif_id,score,predicted_binary,true_label,mode\n")
        for i, cid in enumerate(test_cids):
            score = knn_preds_final[i]
            pred_bin = 1 if score < args.threshold else 0
            true_label = test_bgs[i]
            f.write(f"{cid},{score:.6f},{pred_bin},{true_label},{{'knn_K{args.K}'}}\n")

    print(f"  Saved: {knn_csv_path}")
    print(f"  (Can be used as a model in ensemble_discovery.py ensemble)")

    # Save sim-to-positive predictions too
    sim_csv_path = os.path.join(args.output_dir, 'test_predictions_sim_to_pos.csv')
    with open(sim_csv_path, 'w', newline='') as f:
        f.write("cif_id,score,predicted_binary,true_label,mode\n")
        for i, cid in enumerate(test_cids):
            # Convert back: sim_scores = -max_sim, so score = -sim_scores = max_sim
            # But we want "lower score = more positive"
            # Actually, knn predicted bandgap works fine. Let's save the raw sim score.
            max_sim = -sim_scores[i]
            # For CSV, save as "predicted bandgap" = 1 - max_sim (lower = more similar to pos)
            pseudo_bg = 1.0 - max_sim
            pred_bin = 1 if max_sim > 0.75 else 0
            true_label = test_bgs[i]
            f.write(f"{cid},{pseudo_bg:.6f},{pred_bin},{true_label},sim_to_pos\n")

    print(f"  Saved: {sim_csv_path}")

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Method':<45s}  {'FHR':>5s}  {'R@25':>5s}  {'R@50':>5s}  "
          f"{'R@100':>5s}  {'R@200':>5s}  {'Spear':>6s}")
    print(f"  {'-'*80}")

    for name, m in sorted(all_results.items()):
        fhr = m.get('first_hit_rank', 9999)
        r25 = m.get('recall@25', 0)
        r50 = m.get('recall@50', 0)
        r100 = m.get('recall@100', 0)
        r200 = m.get('recall@200', 0)
        rho = m.get('spearman_rho', 0)
        print(f"  {name:<45s}  {fhr:>5d}  {r25:>5.3f}  {r50:>5.3f}  "
              f"{r100:>5.3f}  {r200:>5.3f}  {rho:>6.3f}")

    print(f"  {'-'*80}")
    print(f"{'='*80}")

    # Save full results
    results_path = os.path.join(args.output_dir, 'knn_hybrid_results.json')
    serializable = {}
    for name, m in all_results.items():
        sm = {}
        for k, v in m.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                sm[k] = float(v)
            elif isinstance(v, list):
                sm[k] = [(str(a), float(b), int(c)) for a, b, c in v]
        serializable[name] = sm

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()
