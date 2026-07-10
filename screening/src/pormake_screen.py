#!/usr/bin/env python3
"""
PORMAKE MOF Screening with Two-Signal Approach
=================================================

Combines Neural Network bandgap prediction with kNN structural similarity
to select candidates for DFT validation.

Signal 1 (NN): Predicted bandgap from fine-tuned MOFTransformer.
  - Captures learned bandgap-structure relationships (Spearman ~0.7)
  - Can potentially identify conductors from novel structural families

Signal 2 (kNN): Cosine similarity to nearest known conductor embedding.
  - Exploits structural proximity in pretrained embedding space
  - Finds MOFs that "look like" known conductors

Selection: Union of top-K from each signal. A MOF is a candidate if
EITHER the NN predicts low bandgap OR it's structurally similar to a
known conductor.

Usage:
  python pormake_screen.py \\
      --nn_predictions experiments/exp364_fulltune/test_predictions.csv \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --labels_dir ./data/splits/strategy_d_farthest_point \\
      --output_dir ./screening_results \\
      --n_candidates 25

  # Multiple NN models (averaged):
  python pormake_screen.py \\
      --nn_predictions experiments/exp364_*/test_predictions.csv \\
                       experiments/exp362_*/test_predictions.csv \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --labels_dir ./data/splits/strategy_d_farthest_point \\
      --output_dir ./screening_results \\
      --n_candidates 25
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from collections import defaultdict


def load_nn_predictions(csv_paths):
    """Load and optionally average predictions from one or more NN models."""
    all_preds = {}

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        preds = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row['cif_id']
                preds[cid] = {
                    'score': float(row['score']),
                    'true_label': float(row.get('true_label', -1)),
                }
        model_name = os.path.basename(os.path.dirname(csv_path))
        all_preds[model_name] = preds
        print(f"  Loaded NN: {model_name} ({len(preds)} MOFs)")

    if not all_preds:
        return None

    common_cids = set.intersection(*[set(p.keys()) for p in all_preds.values()])
    print(f"  Common MOFs across {len(all_preds)} NN models: {len(common_cids)}")

    averaged = {}
    for cid in common_cids:
        scores = [all_preds[m][cid]['score'] for m in all_preds]
        true_label = next(iter(all_preds.values()))[cid]['true_label']
        averaged[cid] = {
            'nn_score': np.mean(scores),
            'nn_std': np.std(scores) if len(scores) > 1 else 0.0,
            'true_label': true_label,
            'n_models': len(scores),
        }
    return averaged


def load_embeddings(npz_path):
    """Load embeddings from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = list(data['cif_ids'])
    embeddings = data['embeddings']
    bandgaps = data['bandgaps']
    splits = list(data['splits'])
    return cif_ids, embeddings, bandgaps, splits


def load_labels(labels_dir):
    """Load split assignments from label JSON files."""
    labels = {}
    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(labels_dir, f'{split_name}_bandgaps_regression.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        for cid, bg in data.items():
            labels[cid] = {'bandgap': float(bg), 'split': split_name}
    return labels


def compute_knn_scores(embeddings_dict, labels, threshold=1.0):
    """Compute kNN structural similarity to nearest known conductor.

    For each MOF, computes cosine similarity to the nearest training positive.
    Returns dict: cif_id -> similarity score (higher = more similar to a conductor).
    """
    train_pos_cids = []
    train_pos_embs = []
    all_train_cids = []
    all_train_embs = []

    for cid, info in labels.items():
        if info['split'] in ('train', 'val') and cid in embeddings_dict:
            all_train_cids.append(cid)
            all_train_embs.append(embeddings_dict[cid])
            if info['bandgap'] < threshold:
                train_pos_cids.append(cid)
                train_pos_embs.append(embeddings_dict[cid])

    if not train_pos_embs:
        print("  ERROR: No training positives found")
        return None, None

    train_pos_embs = np.array(train_pos_embs)
    all_train_embs = np.array(all_train_embs)

    train_pos_norms = train_pos_embs / (np.linalg.norm(train_pos_embs, axis=1, keepdims=True) + 1e-12)
    all_train_norms = all_train_embs / (np.linalg.norm(all_train_embs, axis=1, keepdims=True) + 1e-12)

    print(f"  Training positives: {len(train_pos_cids)}")
    print(f"  Total training MOFs: {len(all_train_cids)}")

    return train_pos_norms, all_train_norms


def score_candidates(candidate_cids, candidate_embs, train_pos_norms, all_train_norms):
    """Score candidates by similarity to known conductors and applicability domain."""
    cand_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-12)

    sim_to_pos = cand_norms @ train_pos_norms.T
    max_sim_to_pos = sim_to_pos.max(axis=1)
    nearest_pos_idx = sim_to_pos.argmax(axis=1)
    mean_sim_to_top5_pos = np.sort(sim_to_pos, axis=1)[:, -min(5, sim_to_pos.shape[1]):].mean(axis=1)

    sim_to_all = cand_norms @ all_train_norms.T
    max_sim_to_any = sim_to_all.max(axis=1)

    return {
        'max_sim_to_pos': max_sim_to_pos,
        'nearest_pos_idx': nearest_pos_idx,
        'mean_sim_top5_pos': mean_sim_to_top5_pos,
        'max_sim_to_any_train': max_sim_to_any,
    }


def two_signal_selection(nn_preds, sim_scores, candidate_cids, train_pos_cids,
                         n_candidates=25, nn_weight=0.5, threshold=1.0,
                         min_diversity=0.3):
    """Select candidates using union of top-K from NN and kNN signals.

    Args:
        nn_preds: dict cif_id -> {nn_score, nn_std, true_label, ...}
        sim_scores: dict from score_candidates
        candidate_cids: list of candidate cif_ids
        train_pos_cids: list of training positive cif_ids
        n_candidates: target number of candidates
        nn_weight: ignored (both signals contribute via union)
        threshold: bandgap threshold
        min_diversity: minimum cosine distance between selected candidates

    Returns:
        selected: list of dicts with candidate details
    """
    n = len(candidate_cids)
    k_per_signal = n_candidates

    nn_scores = np.array([nn_preds[cid]['nn_score'] for cid in candidate_cids])
    nn_order = np.argsort(nn_scores)
    nn_topk = set(nn_order[:k_per_signal])

    sim_vals = sim_scores['max_sim_to_pos']
    sim_order = np.argsort(-sim_vals)
    sim_topk = set(sim_order[:k_per_signal])

    union_indices = sorted(nn_topk | sim_topk)

    print(f"\n  Signal 1 (NN bandgap): top-{k_per_signal} candidates")
    print(f"  Signal 2 (kNN similarity): top-{k_per_signal} candidates")
    print(f"  Union: {len(union_indices)} unique candidates")
    print(f"  Overlap: {len(nn_topk & sim_topk)} in both signals")

    candidates = []
    for idx in union_indices:
        cid = candidate_cids[idx]
        info = nn_preds[cid]
        nn_rank = int(np.where(nn_order == idx)[0][0]) + 1
        sim_rank = int(np.where(sim_order == idx)[0][0]) + 1

        in_nn = idx in nn_topk
        in_sim = idx in sim_topk
        if in_nn and in_sim:
            source = "BOTH"
            confidence = "HIGH"
        elif in_nn:
            source = "NN_only"
            confidence = "MEDIUM"
        else:
            source = "SIM_only"
            confidence = "MEDIUM"

        ood_flag = sim_scores['max_sim_to_any_train'][idx] < 0.5
        if ood_flag:
            confidence = "LOW (OOD)"

        nearest_pos = train_pos_cids[sim_scores['nearest_pos_idx'][idx]]

        candidates.append({
            'cif_id': cid,
            'nn_predicted_bg': float(info['nn_score']),
            'nn_uncertainty': float(info.get('nn_std', 0)),
            'nn_rank': nn_rank,
            'sim_to_nearest_pos': float(sim_scores['max_sim_to_pos'][idx]),
            'nearest_pos_cid': nearest_pos,
            'sim_rank': sim_rank,
            'max_sim_to_any_train': float(sim_scores['max_sim_to_any_train'][idx]),
            'mean_sim_top5_pos': float(sim_scores['mean_sim_top5_pos'][idx]),
            'source': source,
            'confidence': confidence,
            'true_bandgap': float(info.get('true_label', -1)),
        })

    candidates.sort(key=lambda c: (
        -{'HIGH': 2, 'MEDIUM': 1, 'LOW (OOD)': 0}[c['confidence']],
        c['nn_predicted_bg'],
    ))

    if len(candidates) > n_candidates and min_diversity > 0:
        print(f"  Enforcing diversity (min cosine distance {min_diversity})...")
        candidates = enforce_diversity(candidates, nn_preds, sim_scores,
                                       candidate_cids, n_candidates, min_diversity)

    return candidates[:n_candidates]


def enforce_diversity(candidates, nn_preds, sim_scores, candidate_cids, n_target, min_dist):
    """Remove candidates that are too similar to already-selected ones."""
    cid_to_idx = {cid: i for i, cid in enumerate(candidate_cids)}
    selected = [candidates[0]]

    for cand in candidates[1:]:
        if len(selected) >= n_target:
            break
        idx = cid_to_idx.get(cand['cif_id'])
        if idx is None:
            selected.append(cand)
            continue

        too_close = False
        for sel in selected:
            sel_idx = cid_to_idx.get(sel['cif_id'])
            if sel_idx is None:
                continue
            sim = sim_scores['max_sim_to_any_train'][idx]
            sel_sim = sim_scores['max_sim_to_any_train'][sel_idx]
            if abs(sim - sel_sim) < 0.01:
                too_close = True
                break
        if not too_close:
            selected.append(cand)

    remaining = [c for c in candidates if c not in selected]
    while len(selected) < n_target and remaining:
        selected.append(remaining.pop(0))

    return selected


def print_screening_report(candidates, threshold=1.0):
    """Print formatted screening results."""
    print(f"\n{'#'*90}")
    print(f"  PORMAKE SCREENING RESULTS -- TOP {len(candidates)} CANDIDATES")
    print(f"{'#'*90}")

    print(f"\n  {'Rank':<5s}  {'CIF ID':<20s}  {'NN bg':>6s}  {'NN rank':>7s}  "
          f"{'Sim':>5s}  {'Sim rank':>8s}  {'Source':<10s}  {'Conf':<12s}  {'True bg':>7s}")
    print(f"  {'-'*88}")

    n_true_pos = 0
    for i, c in enumerate(candidates):
        true_bg = c.get('true_bandgap', -1)
        true_str = f"{true_bg:.3f}" if true_bg >= 0 else "?"
        is_pos = true_bg >= 0 and true_bg < threshold
        if is_pos:
            n_true_pos += 1
            marker = " <-- HIT"
        else:
            marker = ""

        print(f"  {i+1:<5d}  {c['cif_id']:<20s}  {c['nn_predicted_bg']:>6.3f}  "
              f"{c['nn_rank']:>7d}  {c['sim_to_nearest_pos']:>5.3f}  "
              f"{c['sim_rank']:>8d}  {c['source']:<10s}  {c['confidence']:<12s}  "
              f"{true_str:>7s}{marker}")

    print(f"  {'-'*88}")

    if any(c.get('true_bandgap', -1) >= 0 for c in candidates):
        print(f"\n  True positives found: {n_true_pos}/{len(candidates)} candidates")
        total_pos = sum(1 for c in candidates if c.get('true_bandgap', -1) >= 0
                        and c['true_bandgap'] < threshold)
        print(f"  (This is a retrospective check -- in deployment, true labels are unknown)")

    n_both = sum(1 for c in candidates if c['source'] == 'BOTH')
    n_nn = sum(1 for c in candidates if c['source'] == 'NN_only')
    n_sim = sum(1 for c in candidates if c['source'] == 'SIM_only')
    print(f"\n  Selection breakdown:")
    print(f"    Both signals:     {n_both}")
    print(f"    NN only:          {n_nn}")
    print(f"    Similarity only:  {n_sim}")

    n_ood = sum(1 for c in candidates if 'OOD' in c.get('confidence', ''))
    if n_ood:
        print(f"    Out-of-domain:    {n_ood} (predictions less reliable)")


def save_results(candidates, output_dir, nn_predictions=None):
    """Save screening results to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'screening_candidates.csv')
    with open(csv_path, 'w', newline='') as f:
        if candidates:
            writer = csv.DictWriter(f, fieldnames=candidates[0].keys())
            writer.writeheader()
            writer.writerows(candidates)
    print(f"\n  Saved CSV: {csv_path}")

    json_path = os.path.join(output_dir, 'screening_results.json')
    results = {
        'n_candidates': len(candidates),
        'candidates': candidates,
        'sources': {
            'both': sum(1 for c in candidates if c['source'] == 'BOTH'),
            'nn_only': sum(1 for c in candidates if c['source'] == 'NN_only'),
            'sim_only': sum(1 for c in candidates if c['source'] == 'SIM_only'),
        },
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='PORMAKE MOF Screening with Two-Signal Approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Screen using NN + embeddings:
  python pormake_screen.py \\
      --nn_predictions experiments/exp364_fulltune/test_predictions.csv \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --labels_dir ./data/splits/strategy_d_farthest_point \\
      --n_candidates 25

  # Multiple NN models (averaged for uncertainty):
  python pormake_screen.py \\
      --nn_predictions experiments/exp364_*/test_predictions.csv \\
                       experiments/exp362_*/test_predictions.csv \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --labels_dir ./data/splits/strategy_d_farthest_point \\
      --n_candidates 25 \\
      --output_dir ./screening_results
""")
    parser.add_argument('--nn_predictions', type=str, nargs='+', required=True,
                        help='Path(s) to NN test_predictions.csv files')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to embeddings .npz (pretrained)')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directory with train/val/test label JSON files')
    parser.add_argument('--output_dir', type=str, default='./screening_results',
                        help='Output directory')
    parser.add_argument('--n_candidates', type=int, default=25,
                        help='Number of candidates to select (default: 25)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive class (eV)')
    parser.add_argument('--min_diversity', type=float, default=0.0,
                        help='Minimum cosine distance between candidates (0=off)')
    args = parser.parse_args()

    print("=" * 90)
    print("  PORMAKE MOF SCREENING -- TWO-SIGNAL APPROACH")
    print("=" * 90)

    # 1. Load NN predictions
    print("\n--- Loading NN predictions ---")
    nn_preds = load_nn_predictions(args.nn_predictions)
    if nn_preds is None:
        print("ERROR: No NN predictions loaded")
        sys.exit(1)

    # 2. Load embeddings
    print("\n--- Loading embeddings ---")
    cif_ids, embeddings, bandgaps, splits = load_embeddings(args.embeddings_path)
    emb_dict = {cid: embeddings[i] for i, cid in enumerate(cif_ids)}
    print(f"  Loaded {len(emb_dict)} embeddings (dim={embeddings.shape[1]})")

    # 3. Load labels for train/val identification
    print("\n--- Loading split labels ---")
    labels = load_labels(args.labels_dir)
    n_train_pos = sum(1 for v in labels.values()
                      if v['split'] in ('train', 'val') and v['bandgap'] < args.threshold)
    print(f"  Total labeled: {len(labels)}")
    print(f"  Training positives (conductors): {n_train_pos}")

    # 4. Compute kNN structural similarity scores
    print("\n--- Computing structural similarity ---")
    train_pos_norms, all_train_norms = compute_knn_scores(
        emb_dict, labels, args.threshold)

    train_pos_cids = [cid for cid, v in labels.items()
                      if v['split'] in ('train', 'val') and v['bandgap'] < args.threshold
                      and cid in emb_dict]

    candidate_cids = [cid for cid in nn_preds if cid in emb_dict]
    candidate_embs = np.array([emb_dict[cid] for cid in candidate_cids])
    print(f"  Screening {len(candidate_cids)} candidate MOFs")

    sim_scores = score_candidates(
        candidate_cids, candidate_embs, train_pos_norms, all_train_norms)

    # 5. Two-signal selection
    print("\n--- Two-signal candidate selection ---")
    candidates = two_signal_selection(
        nn_preds, sim_scores, candidate_cids, train_pos_cids,
        n_candidates=args.n_candidates,
        threshold=args.threshold,
        min_diversity=args.min_diversity,
    )

    # 6. Report
    print_screening_report(candidates, args.threshold)

    # 7. Save
    save_results(candidates, args.output_dir)

    print(f"\n{'='*90}")
    print(f"  Screening complete. {len(candidates)} candidates selected.")
    print(f"  Next step: Run DFT on these candidates to validate bandgap predictions.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
