#!/usr/bin/env python3
"""
UMAP Analysis for Original Split (new_subset)
===============================================

Standalone UMAP visualization of the ORIGINAL train/val/test split
(the one used before embedding-informed splits D/E/F were created).

This helps diagnose WHY models trained on the original split struggled:
- Are test positives isolated from train positives in embedding space?
- Do test positives cluster near train negatives?
- Which test positives are structurally novel vs. well-covered?

Outputs:
  - umap_original_split_bandgap.png   (colored by bandgap, stars for positives)
  - umap_original_split_split.png     (colored by train/val/test)
  - umap_original_split_diagnosis.png (highlights isolated test positives)
  - umap_original_split_report.txt    (nearest-neighbor diagnosis table)

Usage:
  # If embeddings_pretrained.npz already exists (recommended):
  python umap_analysis_original_split.py \
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \
      --original_splits_dir ./new_subset \
      --output_dir ./umap_original_split

  # If you also want to compare with Split D:
  python umap_analysis_original_split.py \
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \
      --original_splits_dir ./new_subset \
      --splitd_dir ./new_splits/strategy_d_farthest_point \
      --output_dir ./umap_original_split
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_embeddings(npz_path):
    """Load embeddings from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = list(data['cif_ids'])
    embeddings = data['embeddings']
    bandgaps = data['bandgaps']
    splits_from_npz = list(data['splits'])
    return cif_ids, embeddings, bandgaps, splits_from_npz


def load_split_labels(splits_dir, downstream='bandgaps_regression'):
    """Load train/val/test labels from JSON files in a split directory."""
    labels = {}
    split_assignments = {}
    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(splits_dir, f'{split_name}_{downstream}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            for cid, bg in data.items():
                labels[cid] = float(bg)
                split_assignments[cid] = split_name
    return labels, split_assignments


def cosine_similarity(a, b):
    """Cosine similarity between vectors a and b."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def cosine_sim_matrix(A, B):
    """Pairwise cosine similarity between rows of A and rows of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def compute_umap(embeddings, n_neighbors=30, min_dist=0.3, metric='cosine', seed=42):
    """Compute 2D UMAP projection."""
    try:
        from umap import UMAP
    except ImportError:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    print("  Computing UMAP projection...")
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    coords = reducer.fit_transform(embeddings)
    print(f"  UMAP done: {coords.shape}")
    return coords


def plot_bandgap_umap(coords, bandgaps, splits, threshold, output_path):
    """Panel 1: All points colored by bandgap. Stars for test positives."""
    pos_mask = bandgaps < threshold
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Left: colored by bandgap
    ax = axes[0]
    bg_clipped = np.clip(bandgaps, 0, 5)
    neg_mask = ~pos_mask
    scatter = ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
                         c=bg_clipped[neg_mask], cmap='RdYlBu',
                         s=3, alpha=0.3, rasterized=True, vmin=0, vmax=5)

    trp = train_mask & pos_mask
    ax.scatter(coords[trp, 0], coords[trp, 1], c='blue', s=80,
               marker='^', edgecolors='black', linewidths=0.8, zorder=4,
               label=f'Train positive ({trp.sum()})')

    vp = val_mask & pos_mask
    if vp.sum() > 0:
        ax.scatter(coords[vp, 0], coords[vp, 1], c='orange', s=80,
                   marker='D', edgecolors='black', linewidths=0.8, zorder=4,
                   label=f'Val positive ({vp.sum()})')

    tp = test_mask & pos_mask
    ax.scatter(coords[tp, 0], coords[tp, 1], c='red', s=150,
               marker='*', edgecolors='black', linewidths=1.0, zorder=5,
               label=f'Test positive ({tp.sum()})')

    ax.set_title('UMAP -- Colored by Bandgap (eV)\nOriginal Split (new_subset)', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    plt.colorbar(scatter, ax=ax, label='Bandgap (eV)', shrink=0.8)

    # Right: colored by split
    ax = axes[1]
    colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:gray'}
    for split_name in ['test', 'train', 'val']:
        mask = np.array([s == split_name for s in splits])
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[split_name],
                   s=3, alpha=0.2, label=f'{split_name} ({mask.sum()})',
                   rasterized=True)
    ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1], c='red', s=60,
               marker='*', edgecolors='black', linewidths=0.5, zorder=5,
               label=f'All positives ({pos_mask.sum()})')
    ax.set_title('UMAP -- Colored by Split\nOriginal Split (new_subset)', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_diagnosis_umap(coords, bandgaps, splits, cif_ids, nn_results,
                        threshold, output_path):
    """
    Highlight each test positive with color-coded isolation status.
    Green = well-covered (nearest train pos sim > 0.85)
    Yellow = borderline (0.7 - 0.85)
    Red = isolated (< 0.7)
    """
    pos_mask = bandgaps < threshold
    train_mask = np.array([s == 'train' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])

    fig, ax = plt.subplots(figsize=(14, 10))

    neg_mask = ~pos_mask
    ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
               c='lightgray', s=2, alpha=0.2, rasterized=True, label='Negatives')

    trp = train_mask & pos_mask
    ax.scatter(coords[trp, 0], coords[trp, 1], c='blue', s=60,
               marker='^', edgecolors='black', linewidths=0.5, zorder=4,
               label=f'Train positives ({trp.sum()})')

    cid_to_idx = {cid: i for i, cid in enumerate(cif_ids)}

    for result in nn_results:
        cid = result['cif_id']
        idx = cid_to_idx.get(cid)
        if idx is None:
            continue

        sim = result['nearest_pos_sim']
        gap = result['pos_neg_gap']

        if sim >= 0.85:
            color, status = 'limegreen', 'COVERED'
        elif sim >= 0.70:
            color, status = 'gold', 'BORDERLINE'
        else:
            color, status = 'red', 'ISOLATED'

        ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=250,
                   marker='*', edgecolors='black', linewidths=1.5, zorder=6)

        label = f"{cid[:12]}\nbg={result['bandgap']:.2f}\nsim={sim:.2f}"
        ax.annotate(label, (coords[idx, 0], coords[idx, 1]),
                    fontsize=7, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    zorder=7)

    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
               markersize=10, label=f'Train positive ({trp.sum()})'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='limegreen',
               markersize=15, label='Test pos: COVERED (sim >= 0.85)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
               markersize=15, label='Test pos: BORDERLINE (0.70-0.85)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=15, label='Test pos: ISOLATED (sim < 0.70)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    ax.set_title('UMAP Diagnosis -- Original Split\n'
                 'Why did models fail? Test positives colored by structural coverage',
                 fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison_umap(coords, bandgaps, splits_original, splits_d,
                         cif_ids, threshold, output_path):
    """Side-by-side: original split vs Split D assignments on same UMAP."""
    pos_mask = bandgaps < threshold
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, splits, title in [
        (axes[0], splits_original, 'Original Split (new_subset)'),
        (axes[1], splits_d, 'Split D (farthest-point coverage)')
    ]:
        train_mask = np.array([s == 'train' for s in splits])
        test_mask = np.array([s == 'test' for s in splits])

        neg_mask = ~pos_mask
        ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
                   c='lightgray', s=2, alpha=0.2, rasterized=True)

        trp = train_mask & pos_mask
        ax.scatter(coords[trp, 0], coords[trp, 1], c='blue', s=80,
                   marker='^', edgecolors='black', linewidths=0.5, zorder=4,
                   label=f'Train pos ({trp.sum()})')

        tp = test_mask & pos_mask
        ax.scatter(coords[tp, 0], coords[tp, 1], c='red', s=150,
                   marker='*', edgecolors='black', linewidths=1.0, zorder=5,
                   label=f'Test pos ({tp.sum()})')

        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)

    plt.suptitle('Same UMAP projection, different split assignments\n'
                 'Split D guarantees each test positive has a nearby train positive',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def nn_diagnosis(cif_ids, embeddings, bandgaps, splits, threshold=1.0):
    """
    For each test positive, find nearest train positive and nearest train negative.
    Returns list of dicts with diagnosis per test positive.
    """
    train_mask = np.array([s == 'train' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])
    pos_mask = bandgaps < threshold

    train_pos_idx = np.where(train_mask & pos_mask)[0]
    train_neg_idx = np.where(train_mask & ~pos_mask)[0]
    train_all_idx = np.where(train_mask)[0]
    test_pos_idx = np.where(test_mask & pos_mask)[0]

    print(f"\n  Split stats:")
    print(f"    Train total:    {train_mask.sum()}")
    print(f"    Train positive: {len(train_pos_idx)}")
    print(f"    Train negative: {len(train_neg_idx)}")
    print(f"    Test positive:  {len(test_pos_idx)}")
    print(f"    Test total:     {test_mask.sum()}")

    results = []

    for tp_i in test_pos_idx:
        tp_emb = embeddings[tp_i:tp_i+1]
        tp_cid = cif_ids[tp_i]
        tp_bg = bandgaps[tp_i]

        sim_to_train_pos = cosine_sim_matrix(tp_emb, embeddings[train_pos_idx])[0]
        sim_to_train_neg = cosine_sim_matrix(tp_emb, embeddings[train_neg_idx])[0]
        sim_to_train_all = cosine_sim_matrix(tp_emb, embeddings[train_all_idx])[0]

        # Nearest train positive
        best_pos_j = np.argmax(sim_to_train_pos)
        nearest_pos_sim = float(sim_to_train_pos[best_pos_j])
        nearest_pos_cid = cif_ids[train_pos_idx[best_pos_j]]
        nearest_pos_bg = float(bandgaps[train_pos_idx[best_pos_j]])

        # Nearest train negative
        best_neg_j = np.argmax(sim_to_train_neg)
        nearest_neg_sim = float(sim_to_train_neg[best_neg_j])
        nearest_neg_cid = cif_ids[train_neg_idx[best_neg_j]]
        nearest_neg_bg = float(bandgaps[train_neg_idx[best_neg_j]])

        # Rank of nearest positive among all train
        rank_of_pos = int((sim_to_train_all > nearest_pos_sim).sum()) + 1

        # Top-5 nearest in all train
        top5_all = np.argsort(-sim_to_train_all)[:5]
        nn_all = []
        for idx in top5_all:
            gi = train_all_idx[idx]
            nn_all.append({
                'cif_id': cif_ids[gi],
                'bandgap': float(bandgaps[gi]),
                'is_positive': bool(bandgaps[gi] < threshold),
                'cosine_sim': float(sim_to_train_all[idx]),
            })

        results.append({
            'cif_id': tp_cid,
            'bandgap': float(tp_bg),
            'nearest_pos_cid': nearest_pos_cid,
            'nearest_pos_bg': nearest_pos_bg,
            'nearest_pos_sim': nearest_pos_sim,
            'nearest_neg_cid': nearest_neg_cid,
            'nearest_neg_bg': nearest_neg_bg,
            'nearest_neg_sim': nearest_neg_sim,
            'pos_neg_gap': nearest_pos_sim - nearest_neg_sim,
            'rank_of_nearest_pos': rank_of_pos,
            'top5_neighbors': nn_all,
        })

    return results


def generate_report(nn_results, output_path, split_name="Original"):
    """Generate text report."""
    lines = []
    W = 100

    lines.append("=" * W)
    lines.append(f"UMAP NEAREST-NEIGHBOR DIAGNOSIS -- {split_name} Split")
    lines.append("=" * W)
    lines.append("")
    lines.append("For each test positive, we check:")
    lines.append("  - Nearest train POSITIVE: is there a structural anchor?")
    lines.append("  - Nearest train NEGATIVE: is the model confused?")
    lines.append("  - Gap (pos_sim - neg_sim): positive = good, negative = trouble")
    lines.append("")

    lines.append(f"{'Test CIF':<22s}  {'BG':>5s}  {'NN+ CIF':<22s}  {'NN+ BG':>5s}  "
                 f"{'CosSim':>7s}  {'NN- Sim':>7s}  {'Gap':>7s}  {'PosRank':>7s}  Diagnosis")
    lines.append("-" * W)

    for r in sorted(nn_results, key=lambda x: x['pos_neg_gap']):
        gap = r['pos_neg_gap']
        if r['nearest_pos_sim'] >= 0.85:
            diag = "COVERED"
        elif r['nearest_pos_sim'] >= 0.70:
            diag = "BORDERLINE"
        else:
            diag = "ISOLATED"

        if gap < -0.05:
            diag += " + NEG_CLOSER"

        lines.append(
            f"{r['cif_id']:<22s}  {r['bandgap']:>5.3f}  "
            f"{r['nearest_pos_cid']:<22s}  {r['nearest_pos_bg']:>5.3f}  "
            f"{r['nearest_pos_sim']:>7.4f}  {r['nearest_neg_sim']:>7.4f}  "
            f"{gap:>+7.4f}  {r['rank_of_nearest_pos']:>7d}  {diag}")

    lines.append("")
    lines.append("=" * W)
    lines.append("INTERPRETATION")
    lines.append("=" * W)
    lines.append("  COVERED:    sim >= 0.85 to a train positive. Model should learn this.")
    lines.append("  BORDERLINE: sim 0.70-0.85. Model might learn it with enough epochs.")
    lines.append("  ISOLATED:   sim < 0.70. Structurally novel -- model cannot learn this")
    lines.append("              from the current training set. This is WHY re-splitting helps.")
    lines.append("  NEG_CLOSER: Nearest train negative is closer than nearest train positive.")
    lines.append("              The model sees this test positive as 'looking like a negative'.")
    lines.append("")
    lines.append("  PosRank: Rank of the nearest train positive among ALL train neighbors.")
    lines.append("           PosRank=1 means the closest training sample is a positive (ideal).")
    lines.append("           PosRank>>1 means many negatives are closer than any positive.")
    lines.append("")

    n_covered = sum(1 for r in nn_results if r['nearest_pos_sim'] >= 0.85)
    n_border = sum(1 for r in nn_results if 0.70 <= r['nearest_pos_sim'] < 0.85)
    n_isolated = sum(1 for r in nn_results if r['nearest_pos_sim'] < 0.70)
    n_neg_closer = sum(1 for r in nn_results if r['pos_neg_gap'] < -0.05)

    lines.append("=" * W)
    lines.append("SUMMARY")
    lines.append("=" * W)
    lines.append(f"  Total test positives: {len(nn_results)}")
    lines.append(f"  COVERED (sim >= 0.85):    {n_covered}")
    lines.append(f"  BORDERLINE (0.70-0.85):   {n_border}")
    lines.append(f"  ISOLATED (sim < 0.70):    {n_isolated}")
    lines.append(f"  Nearest neg closer:       {n_neg_closer}")
    lines.append("")

    if n_isolated > 0:
        lines.append(f"  >> {n_isolated}/{len(nn_results)} test positives are STRUCTURALLY ISOLATED.")
        lines.append(f"     No model trained on this split can learn to find them.")
        lines.append(f"     This is the root cause of poor recall on the original split.")
        lines.append(f"     Solution: embedding-informed splitting (Split D) ensures coverage.")
    else:
        lines.append(f"  >> All test positives have structural coverage in training.")
        lines.append(f"     Poor recall (if any) is due to model capacity, not data splitting.")

    lines.append("")

    # Per-positive details
    lines.append("=" * W)
    lines.append("DETAILED PER-POSITIVE NEIGHBORS")
    lines.append("=" * W)

    for r in sorted(nn_results, key=lambda x: x['bandgap']):
        lines.append(f"\n  --- {r['cif_id']} (bandgap = {r['bandgap']:.3f} eV) ---")
        lines.append(f"  Nearest train positive: {r['nearest_pos_cid']} "
                     f"(bg={r['nearest_pos_bg']:.3f}, sim={r['nearest_pos_sim']:.4f})")
        lines.append(f"  Nearest train negative: {r['nearest_neg_cid']} "
                     f"(bg={r['nearest_neg_bg']:.3f}, sim={r['nearest_neg_sim']:.4f})")
        lines.append(f"  Top-5 nearest in all training:")
        for nn in r['top5_neighbors']:
            pos_str = "POS" if nn['is_positive'] else "neg"
            lines.append(f"    {nn['cif_id']:<22s}  bg={nn['bandgap']:.3f}  "
                         f"sim={nn['cosine_sim']:.4f}  [{pos_str}]")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='UMAP Analysis for Original Split (new_subset)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python umap_analysis_original_split.py \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --original_splits_dir ./new_subset \\
      --output_dir ./umap_original_split

  python umap_analysis_original_split.py \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --original_splits_dir ./new_subset \\
      --splitd_dir ./new_splits/strategy_d_farthest_point \\
      --output_dir ./umap_original_split
""")
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to embeddings_pretrained.npz')
    parser.add_argument('--original_splits_dir', type=str, required=True,
                        help='Path to new_subset/ with original {train,val,test}_bandgaps_regression.json')
    parser.add_argument('--splitd_dir', type=str, default=None,
                        help='(Optional) Path to strategy_d_farthest_point/ for comparison')
    parser.add_argument('--output_dir', type=str, default='./umap_original_split',
                        help='Output directory')
    parser.add_argument('--downstream', type=str, default='bandgaps_regression',
                        help='Downstream task name')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive class (eV)')
    parser.add_argument('--n_neighbors', type=int, default=30,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.3,
                        help='UMAP min_dist parameter')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # 1. Load embeddings
    # =========================================================================
    print("=" * 80)
    print("  UMAP ANALYSIS -- ORIGINAL SPLIT (new_subset)")
    print("=" * 80)

    print(f"\n  Loading embeddings from: {args.embeddings_path}")
    cif_ids, embeddings, bandgaps, splits_npz = load_embeddings(args.embeddings_path)
    print(f"  Loaded: {len(cif_ids)} MOFs, embedding dim={embeddings.shape[1]}")

    # =========================================================================
    # 2. Load original split labels
    # =========================================================================
    print(f"\n  Loading original split labels from: {args.original_splits_dir}")
    labels_orig, split_assign_orig = load_split_labels(
        args.original_splits_dir, args.downstream)

    # Map splits onto the embedding order
    splits_original = []
    for cid in cif_ids:
        if cid in split_assign_orig:
            splits_original.append(split_assign_orig[cid])
        else:
            splits_original.append(splits_npz[cif_ids.index(cid)]
                                   if cid in cif_ids else 'unknown')

    # Use bandgaps from labels if available
    bandgaps_override = np.array([
        labels_orig.get(cid, bandgaps[i]) for i, cid in enumerate(cif_ids)
    ])

    n_train = sum(1 for s in splits_original if s == 'train')
    n_val = sum(1 for s in splits_original if s == 'val')
    n_test = sum(1 for s in splits_original if s == 'test')
    n_pos = sum(1 for bg in bandgaps_override if bg < args.threshold)
    n_train_pos = sum(1 for i, s in enumerate(splits_original)
                      if s == 'train' and bandgaps_override[i] < args.threshold)
    n_test_pos = sum(1 for i, s in enumerate(splits_original)
                     if s == 'test' and bandgaps_override[i] < args.threshold)

    print(f"  Train: {n_train} ({n_train_pos} pos)")
    print(f"  Val:   {n_val}")
    print(f"  Test:  {n_test} ({n_test_pos} pos)")
    print(f"  Total positives: {n_pos}")

    # =========================================================================
    # 3. UMAP projection (shared across all plots)
    # =========================================================================
    print(f"\n--- UMAP Projection ---")
    coords = compute_umap(embeddings, args.n_neighbors, args.min_dist)

    np.savez_compressed(
        os.path.join(args.output_dir, 'umap_coords.npz'),
        coords=coords, cif_ids=np.array(cif_ids),
    )

    # =========================================================================
    # 4. Nearest-neighbor diagnosis
    # =========================================================================
    print(f"\n--- Nearest-Neighbor Diagnosis ---")
    nn_results = nn_diagnosis(cif_ids, embeddings, bandgaps_override,
                              splits_original, args.threshold)

    # =========================================================================
    # 5. Plots
    # =========================================================================
    print(f"\n--- Generating Plots ---")

    plot_bandgap_umap(coords, bandgaps_override, splits_original, args.threshold,
                      os.path.join(args.output_dir, 'umap_original_split_bandgap.png'))

    plot_diagnosis_umap(coords, bandgaps_override, splits_original, cif_ids,
                        nn_results, args.threshold,
                        os.path.join(args.output_dir, 'umap_original_split_diagnosis.png'))

    # =========================================================================
    # 6. Comparison with Split D (if provided)
    # =========================================================================
    if args.splitd_dir:
        print(f"\n--- Comparison with Split D ---")
        _, split_assign_d = load_split_labels(args.splitd_dir, args.downstream)

        splits_d = []
        for cid in cif_ids:
            if cid in split_assign_d:
                splits_d.append(split_assign_d[cid])
            else:
                splits_d.append('unknown')

        n_train_d = sum(1 for s in splits_d if s == 'train')
        n_test_d = sum(1 for s in splits_d if s == 'test')
        n_train_pos_d = sum(1 for i, s in enumerate(splits_d)
                            if s == 'train' and bandgaps_override[i] < args.threshold)
        n_test_pos_d = sum(1 for i, s in enumerate(splits_d)
                           if s == 'test' and bandgaps_override[i] < args.threshold)
        print(f"  Split D: Train {n_train_d} ({n_train_pos_d} pos), "
              f"Test {n_test_d} ({n_test_pos_d} pos)")

        plot_comparison_umap(coords, bandgaps_override, splits_original, splits_d,
                             cif_ids, args.threshold,
                             os.path.join(args.output_dir, 'umap_comparison_original_vs_d.png'))

        nn_results_d = nn_diagnosis(cif_ids, embeddings, bandgaps_override,
                                    splits_d, args.threshold)
        generate_report(nn_results_d,
                        os.path.join(args.output_dir, 'umap_splitd_report.txt'),
                        split_name="Split D")

    # =========================================================================
    # 7. Text report
    # =========================================================================
    print(f"\n--- Generating Report ---")
    generate_report(nn_results,
                    os.path.join(args.output_dir, 'umap_original_split_report.txt'),
                    split_name="Original")

    # Save JSON summary
    summary = {
        'split': 'original (new_subset)',
        'n_total': len(cif_ids),
        'n_train': n_train,
        'n_test': n_test,
        'n_positive': n_pos,
        'n_train_positive': n_train_pos,
        'n_test_positive': n_test_pos,
        'test_positives': [],
    }
    for r in sorted(nn_results, key=lambda x: x['pos_neg_gap']):
        summary['test_positives'].append({
            'cif_id': r['cif_id'],
            'bandgap': r['bandgap'],
            'nearest_train_pos_sim': r['nearest_pos_sim'],
            'nearest_train_neg_sim': r['nearest_neg_sim'],
            'gap': r['pos_neg_gap'],
            'rank_of_nearest_pos': r['rank_of_nearest_pos'],
        })

    with open(os.path.join(args.output_dir, 'umap_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"  DONE. All outputs in: {args.output_dir}")
    print(f"{'=' * 80}")
    print(f"  Files:")
    print(f"    umap_original_split_bandgap.png    -- UMAP colored by bandgap + splits")
    print(f"    umap_original_split_diagnosis.png  -- Test positives colored by coverage")
    if args.splitd_dir:
        print(f"    umap_comparison_original_vs_d.png  -- Side-by-side original vs Split D")
        print(f"    umap_splitd_report.txt             -- NN diagnosis for Split D")
    print(f"    umap_original_split_report.txt     -- Full NN diagnosis table")
    print(f"    umap_analysis_summary.json         -- Machine-readable summary")
    print(f"    umap_coords.npz                    -- UMAP coordinates (reusable)")


if __name__ == "__main__":
    main()
