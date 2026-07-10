#!/usr/bin/env python3
"""
MOF Embedding Analysis & Nearest-Neighbor Diagnosis
=====================================================

Extracts MOFTransformer embeddings (768-dim CLS features) for ALL MOFs
in the dataset (train + val + test), then runs nearest-neighbor analysis
to diagnose WHY certain test positives are missed by every model.

Key questions answered:
1. Are the missed test positives structurally similar to any training positive?
2. Do the missed test positives cluster near training negatives?
3. Is the problem "unseen structural motif" or "feature representation gap"?

Outputs:
  - embeddings_pretrained.npz  (raw MOFTransformer embeddings, no fine-tuning)
    This file can be reused for different split strategies: run once with any
    splits_dir that covers your full MOF set; downstream scripts (e.g.
    embedding_classifier.py) accept --labels_dir to override train/val/test.
  - embeddings_finetuned.npz   (from best fine-tuned checkpoint, if provided)
  - embedding_analysis_report.txt  (nearest-neighbor tables)
  - embedding_umap.png           (2D UMAP plot colored by bandgap)
  - embedding_tsne.png           (2D t-SNE plot colored by bandgap)
  - pairwise_similarity.png      (similarity heatmap among positives)

Usage:
  # Pretrained only (no checkpoint needed):
  python analyze_embeddings.py --data_dir /path/to/dataset --splits_dir /path/to/splits

  # Also compare with fine-tuned model:
  python analyze_embeddings.py --data_dir /path/to/dataset --splits_dir /path/to/splits \\
      --checkpoint /path/to/best_model.ckpt

  # Example:
  python analyze_embeddings.py \\
      --data_dir ./data/raw \\
      --splits_dir ./data/raw \\
      --checkpoint ./experiments/exp364_fulltune/best_es.ckpt \\
      --output_dir ./data/embeddings
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# MOFTransformer imports
from moftransformer.modules.module import Module
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def create_model(config, checkpoint_path=None):
    """Create MOFTransformer model, optionally loading a fine-tuned checkpoint."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading fine-tuned checkpoint: {checkpoint_path}")
        # Load the checkpoint — the model class may vary (MOFRegressor, etc.)
        # We load just the Module base class for feature extraction
        try:
            model = Module.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        except Exception:
            # If Module doesn't work, try loading state dict into a fresh Module
            model = Module(config)
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # Filter out head-specific keys
            base_keys = {k: v for k, v in state_dict.items()
                         if not any(h in k for h in ['regression_head', 'classification_head',
                                                      'ordinal_head', 'discovery_head'])}
            model.load_state_dict(base_keys, strict=False)
        print(f"  Loaded checkpoint successfully")
    else:
        print(f"  Using pretrained MOFTransformer (pmtransformer)")
        model = Module(config)

    model.eval()
    return model


def extract_embeddings(model, data_dir, splits_dir, downstream, config,
                       batch_size=8, num_workers=2, device='cuda'):
    """
    Extract 768-dim CLS embeddings for all samples across train/val/test.

    Returns:
        cif_ids: list of str
        embeddings: np.ndarray [N, 768]
        bandgaps: np.ndarray [N]
        split_labels: list of str ('train', 'val', 'test')
    """
    model = model.to(device)
    model.eval()

    all_cif_ids = []
    all_embeddings = []
    all_bandgaps = []
    all_splits = []

    # Load bandgap labels for all splits
    bandgap_lookup = {}
    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(splits_dir, f'{split_name}_{downstream}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                labels = json.load(f)
            bandgap_lookup.update(labels)
            print(f"  Loaded {len(labels)} labels from {split_name}")

    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(data_dir, f'{split_name}_{downstream}.json')
        if not os.path.exists(json_path):
            # Try splits_dir
            json_path = os.path.join(splits_dir, f'{split_name}_{downstream}.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: No {split_name} split found, skipping")
            continue

        ds = Dataset(
            data_dir,
            split=split_name,
            downstream=downstream,
            nbr_fea_len=config.get("nbr_fea_len", 64),
            draw_false_grid=False,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: Dataset.collate(x, config.get("img_size", 30)),
            pin_memory=True,
        )

        print(f"  Extracting {split_name}: {len(ds)} samples...")
        n_done = 0

        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                output = model.infer(batch)
                cls_feats = output["cls_feats"]  # [B, 768]

                # Also extract mean-pooled features for comparison
                cif_ids_batch = output["cif_id"]

                cls_np = cls_feats.cpu().numpy()

                for i, cid in enumerate(cif_ids_batch):
                    all_cif_ids.append(cid)
                    all_embeddings.append(cls_np[i])
                    all_bandgaps.append(float(bandgap_lookup.get(cid, -1.0)))
                    all_splits.append(split_name)

                n_done += len(cif_ids_batch)
                if n_done % 500 == 0:
                    print(f"    {n_done}/{len(ds)} done")

        print(f"    {split_name}: extracted {n_done} embeddings")

    embeddings = np.stack(all_embeddings, axis=0)
    bandgaps = np.array(all_bandgaps)

    return all_cif_ids, embeddings, bandgaps, all_splits


# =============================================================================
# NEAREST-NEIGHBOR ANALYSIS
# =============================================================================

def cosine_similarity_matrix(A, B):
    """Compute cosine similarity between rows of A and rows of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def nn_analysis(cif_ids, embeddings, bandgaps, splits, threshold=1.0):
    """
    For each test positive, find nearest neighbors in training set.

    Returns a list of dicts with analysis per test positive.
    """
    # Separate into train positives, train negatives, test positives
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])
    pos_mask = bandgaps < threshold

    train_pos_mask = train_mask & pos_mask
    train_neg_mask = train_mask & ~pos_mask
    val_pos_mask = val_mask & pos_mask
    test_pos_mask = test_mask & pos_mask

    train_pos_idx = np.where(train_pos_mask)[0]
    train_neg_idx = np.where(train_neg_mask)[0]
    train_all_idx = np.where(train_mask)[0]
    val_pos_idx = np.where(val_pos_mask)[0]
    test_pos_idx = np.where(test_pos_mask)[0]

    print(f"\n  Train positives: {len(train_pos_idx)}")
    print(f"  Train negatives: {len(train_neg_idx)}")
    print(f"  Val positives:   {len(val_pos_idx)}")
    print(f"  Test positives:  {len(test_pos_idx)}")

    results = []

    # For each test positive
    for tp_i in test_pos_idx:
        tp_emb = embeddings[tp_i:tp_i+1]  # [1, 768]
        tp_cid = cif_ids[tp_i]
        tp_bg = bandgaps[tp_i]

        # Cosine similarity to all train samples
        sim_to_train = cosine_similarity_matrix(tp_emb, embeddings[train_all_idx])[0]
        # Cosine similarity to train positives only
        sim_to_train_pos = cosine_similarity_matrix(tp_emb, embeddings[train_pos_idx])[0]
        # Cosine similarity to train negatives
        sim_to_train_neg = cosine_similarity_matrix(tp_emb, embeddings[train_neg_idx])[0]

        # Top-5 nearest in ALL train
        top5_all = np.argsort(-sim_to_train)[:5]
        nn_all = []
        for rank, idx in enumerate(top5_all):
            global_idx = train_all_idx[idx]
            nn_all.append({
                'rank': rank + 1,
                'cif_id': cif_ids[global_idx],
                'bandgap': float(bandgaps[global_idx]),
                'is_positive': bool(bandgaps[global_idx] < threshold),
                'cosine_sim': float(sim_to_train[idx]),
            })

        # Top-5 nearest among train POSITIVES
        top5_pos = np.argsort(-sim_to_train_pos)[:5]
        nn_pos = []
        for rank, idx in enumerate(top5_pos):
            global_idx = train_pos_idx[idx]
            nn_pos.append({
                'rank': rank + 1,
                'cif_id': cif_ids[global_idx],
                'bandgap': float(bandgaps[global_idx]),
                'cosine_sim': float(sim_to_train_pos[idx]),
            })

        # Top-5 nearest among train NEGATIVES
        top5_neg = np.argsort(-sim_to_train_neg)[:5]
        nn_neg = []
        for rank, idx in enumerate(top5_neg):
            global_idx = train_neg_idx[idx]
            nn_neg.append({
                'rank': rank + 1,
                'cif_id': cif_ids[global_idx],
                'bandgap': float(bandgaps[global_idx]),
                'cosine_sim': float(sim_to_train_neg[idx]),
            })

        # Key diagnostic: is nearest positive closer or further than nearest negative?
        nearest_pos_sim = nn_pos[0]['cosine_sim'] if nn_pos else 0
        nearest_neg_sim = nn_neg[0]['cosine_sim'] if nn_neg else 0
        pos_neg_gap = nearest_pos_sim - nearest_neg_sim

        # Rank of nearest positive among ALL train neighbors
        pos_sims_in_all = sim_to_train_pos.max()
        rank_of_nearest_pos = int((sim_to_train > pos_sims_in_all).sum()) + 1

        results.append({
            'cif_id': tp_cid,
            'bandgap': tp_bg,
            'nearest_any': nn_all,
            'nearest_positive': nn_pos,
            'nearest_negative': nn_neg,
            'nearest_pos_sim': nearest_pos_sim,
            'nearest_neg_sim': nearest_neg_sim,
            'pos_neg_gap': pos_neg_gap,
            'rank_of_nearest_pos_in_all_train': rank_of_nearest_pos,
        })

    return results


# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================

def distribution_analysis(cif_ids, embeddings, bandgaps, splits, threshold=1.0):
    """Analyze embedding distribution differences between groups."""
    train_mask = np.array([s == 'train' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])
    pos_mask = bandgaps < threshold

    train_pos = embeddings[train_mask & pos_mask]
    train_neg = embeddings[train_mask & ~pos_mask]
    test_pos = embeddings[test_mask & pos_mask]
    test_neg = embeddings[test_mask & ~pos_mask]

    results = {}

    # Embedding norms
    for name, embs in [('train_pos', train_pos), ('train_neg', train_neg),
                        ('test_pos', test_pos), ('test_neg', test_neg)]:
        norms = np.linalg.norm(embs, axis=1)
        results[f'{name}_norm_mean'] = float(np.mean(norms))
        results[f'{name}_norm_std'] = float(np.std(norms))
        results[f'{name}_count'] = len(embs)

    # Mean embedding distance between groups
    if len(train_pos) > 0 and len(test_pos) > 0:
        train_pos_centroid = train_pos.mean(axis=0)
        test_pos_centroid = test_pos.mean(axis=0)
        centroid_cos = float(cosine_similarity_matrix(
            train_pos_centroid.reshape(1, -1),
            test_pos_centroid.reshape(1, -1)
        )[0, 0])
        results['train_pos_vs_test_pos_centroid_cosine'] = centroid_cos

    if len(train_neg) > 0 and len(test_pos) > 0:
        train_neg_centroid = train_neg.mean(axis=0)
        test_pos_centroid = test_pos.mean(axis=0)
        centroid_cos_neg = float(cosine_similarity_matrix(
            train_neg_centroid.reshape(1, -1),
            test_pos_centroid.reshape(1, -1)
        )[0, 0])
        results['train_neg_vs_test_pos_centroid_cosine'] = centroid_cos_neg

    # Average pairwise similarity within train positives
    if len(train_pos) > 1:
        sim_mat = cosine_similarity_matrix(train_pos, train_pos)
        np.fill_diagonal(sim_mat, 0)
        results['train_pos_intra_sim_mean'] = float(sim_mat.sum() / (len(train_pos) * (len(train_pos) - 1)))

    # Average similarity of test positives to nearest train positive
    if len(train_pos) > 0 and len(test_pos) > 0:
        cross_sim = cosine_similarity_matrix(test_pos, train_pos)
        results['test_pos_to_nearest_train_pos_mean'] = float(cross_sim.max(axis=1).mean())
        results['test_pos_to_nearest_train_pos_min'] = float(cross_sim.max(axis=1).min())

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_umap(embeddings, bandgaps, splits, threshold, output_path):
    """2D UMAP colored by bandgap with markers for positives."""
    try:
        from umap import UMAP
    except ImportError:
        print("  UMAP not installed (pip install umap-learn), skipping UMAP plot")
        return

    print("  Computing UMAP projection...")
    reducer = UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
    coords = reducer.fit_transform(embeddings)

    pos_mask = bandgaps < threshold
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Panel 1: All points colored by bandgap
    ax = axes[0]
    # Clip bandgaps for colormap
    bg_clipped = np.clip(bandgaps, 0, 5)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=bg_clipped, cmap='RdYlBu',
                         s=3, alpha=0.4, rasterized=True)
    # Highlight test positives
    tp_mask = test_mask & pos_mask
    ax.scatter(coords[tp_mask, 0], coords[tp_mask, 1], c='red', s=100,
               marker='*', edgecolors='black', linewidths=1, zorder=5, label='Test positive')
    # Highlight train positives
    trp_mask = train_mask & pos_mask
    ax.scatter(coords[trp_mask, 0], coords[trp_mask, 1], c='blue', s=60,
               marker='^', edgecolors='black', linewidths=0.5, zorder=4, label='Train positive')
    ax.set_title('UMAP — colored by bandgap (eV)', fontsize=14)
    ax.legend(fontsize=10)
    plt.colorbar(scatter, ax=ax, label='Bandgap (eV)')

    # Panel 2: Colored by split
    ax = axes[1]
    colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:gray'}
    for split_name in ['test', 'train', 'val']:  # test first (background)
        mask = np.array([s == split_name for s in splits])
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[split_name],
                   s=3, alpha=0.3, label=f'{split_name} ({mask.sum()})', rasterized=True)
    # Highlight all positives
    ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1], c='red', s=40,
               marker='*', edgecolors='black', linewidths=0.5, zorder=5,
               label=f'All positives ({pos_mask.sum()})')
    ax.set_title('UMAP — colored by split', fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved UMAP plot: {output_path}")


def plot_similarity_heatmap(cif_ids, embeddings, bandgaps, splits, threshold, output_path):
    """Pairwise cosine similarity heatmap among ALL positives (train+val+test)."""
    pos_mask = bandgaps < threshold
    pos_idx = np.where(pos_mask)[0]

    if len(pos_idx) < 2:
        print("  Not enough positives for similarity heatmap")
        return

    pos_embs = embeddings[pos_idx]
    sim_mat = cosine_similarity_matrix(pos_embs, pos_embs)

    pos_cids = [cif_ids[i] for i in pos_idx]
    pos_splits = [splits[i] for i in pos_idx]
    pos_bgs = bandgaps[pos_idx]

    # Sort by split (train first, then val, then test) then by bandgap
    split_order = {'train': 0, 'val': 1, 'test': 2}
    sort_idx = sorted(range(len(pos_idx)), key=lambda i: (split_order.get(pos_splits[i], 3), pos_bgs[i]))

    sim_sorted = sim_mat[np.ix_(sort_idx, sort_idx)]
    labels = [f"{pos_splits[i][:2]}|{pos_cids[i][:8]}|{pos_bgs[i]:.2f}" for i in sort_idx]

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(sim_sorted, cmap='RdYlBu_r', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f'Pairwise Cosine Similarity — All {len(pos_idx)} Positives\n'
                 f'(tr=train, va=val, te=test | CIF_ID | bandgap eV)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Draw split boundaries
    n_train = sum(1 for s in pos_splits for _ in [] if False)  # count manually
    counts = defaultdict(int)
    for i in sort_idx:
        counts[pos_splits[i]] += 1
    cum = 0
    for split in ['train', 'val', 'test']:
        if counts[split] > 0:
            ax.axhline(cum + counts[split] - 0.5, color='black', linewidth=1, linestyle='--')
            ax.axvline(cum + counts[split] - 0.5, color='black', linewidth=1, linestyle='--')
            cum += counts[split]

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved similarity heatmap: {output_path}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(nn_results, dist_results, output_path, model_name="pretrained"):
    """Generate a text report summarizing the nearest-neighbor analysis."""
    lines = []
    W = 100

    lines.append("=" * W)
    lines.append(f"MOF EMBEDDING NEAREST-NEIGHBOR ANALYSIS ({model_name})")
    lines.append("=" * W)

    # Distribution summary
    lines.append(f"\n--- EMBEDDING DISTRIBUTION ---")
    for k, v in sorted(dist_results.items()):
        lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    key_metric = dist_results.get('test_pos_to_nearest_train_pos_min', 0)
    if key_metric < 0.7:
        lines.append(f"\n  *** WARNING: At least one test positive has very LOW similarity")
        lines.append(f"      to its nearest train positive (min cosine = {key_metric:.3f}).")
        lines.append(f"      This confirms STRUCTURAL NOVELTY — the model has never seen")
        lines.append(f"      a similar positive during training. ***")

    # Per-test-positive analysis
    lines.append(f"\n{'='*W}")
    lines.append("PER-TEST-POSITIVE NEAREST-NEIGHBOR ANALYSIS")
    lines.append(f"{'='*W}")
    lines.append(f"  For each of the {len(nn_results)} test positives, we show:")
    lines.append(f"  - 5 nearest neighbors among ALL training samples")
    lines.append(f"  - 5 nearest neighbors among TRAIN POSITIVES only")
    lines.append(f"  - 5 nearest neighbors among TRAIN NEGATIVES only")
    lines.append(f"  - Whether nearest positive is closer than nearest negative")

    for r in sorted(nn_results, key=lambda x: x['bandgap']):
        lines.append(f"\n  {'─'*90}")
        lines.append(f"  TEST POSITIVE: {r['cif_id']}  (bandgap = {r['bandgap']:.3f} eV)")
        lines.append(f"  {'─'*90}")

        # Diagnosis
        gap = r['pos_neg_gap']
        if gap > 0.05:
            diagnosis = "GOOD — nearest positive is closer than nearest negative"
        elif gap > -0.05:
            diagnosis = "AMBIGUOUS — nearest positive and negative are equidistant"
        else:
            diagnosis = "BAD — nearest NEGATIVE is closer than nearest positive (model likely confused)"

        lines.append(f"  Nearest train pos sim: {r['nearest_pos_sim']:.4f}")
        lines.append(f"  Nearest train neg sim: {r['nearest_neg_sim']:.4f}")
        lines.append(f"  Gap (pos - neg):       {gap:+.4f}  → {diagnosis}")
        lines.append(f"  Rank of nearest pos in all train: {r['rank_of_nearest_pos_in_all_train']}")

        lines.append(f"\n  Top-5 nearest in ALL training:")
        lines.append(f"    {'Rank':>4s}  {'CIF_ID':<20s}  {'Bandgap':>8s}  {'Positive?':>9s}  {'Cosine':>7s}")
        for nn in r['nearest_any']:
            pos_str = "YES" if nn['is_positive'] else "no"
            lines.append(f"    {nn['rank']:>4d}  {nn['cif_id']:<20s}  {nn['bandgap']:>8.3f}  "
                         f"{pos_str:>9s}  {nn['cosine_sim']:>7.4f}")

        lines.append(f"\n  Top-5 nearest TRAIN POSITIVES:")
        lines.append(f"    {'Rank':>4s}  {'CIF_ID':<20s}  {'Bandgap':>8s}  {'Cosine':>7s}")
        for nn in r['nearest_positive']:
            lines.append(f"    {nn['rank']:>4d}  {nn['cif_id']:<20s}  {nn['bandgap']:>8.3f}  "
                         f"{nn['cosine_sim']:>7.4f}")

        lines.append(f"\n  Top-5 nearest TRAIN NEGATIVES:")
        lines.append(f"    {'Rank':>4s}  {'CIF_ID':<20s}  {'Bandgap':>8s}  {'Cosine':>7s}")
        for nn in r['nearest_negative']:
            lines.append(f"    {nn['rank']:>4d}  {nn['cif_id']:<20s}  {nn['bandgap']:>8.3f}  "
                         f"{nn['cosine_sim']:>7.4f}")

    # Summary table
    lines.append(f"\n{'='*W}")
    lines.append("SUMMARY TABLE: Test Positive ↔ Nearest Train Positive")
    lines.append(f"{'='*W}")
    lines.append(f"  {'Test CIF_ID':<20s}  {'BG':>6s}  {'NN+ CIF_ID':<20s}  {'NN+ BG':>6s}  "
                 f"{'CosSim':>7s}  {'NN- Sim':>7s}  {'Gap':>7s}  {'Diagnosis'}")
    lines.append(f"  {'-'*95}")

    for r in sorted(nn_results, key=lambda x: x['pos_neg_gap']):
        gap = r['pos_neg_gap']
        if gap > 0.05:
            diag = "OK"
        elif gap > -0.05:
            diag = "BORDERLINE"
        else:
            diag = "ISOLATED"

        nn_pos = r['nearest_positive'][0] if r['nearest_positive'] else {'cif_id': '?', 'bandgap': 0, 'cosine_sim': 0}
        lines.append(f"  {r['cif_id']:<20s}  {r['bandgap']:>6.3f}  {nn_pos['cif_id']:<20s}  "
                     f"{nn_pos['bandgap']:>6.3f}  {r['nearest_pos_sim']:>7.4f}  "
                     f"{r['nearest_neg_sim']:>7.4f}  {gap:>+7.4f}  {diag}")

    lines.append(f"\n{'='*W}")
    lines.append("INTERPRETATION GUIDE")
    lines.append(f"{'='*W}")
    lines.append("  CosSim > 0.85: Very similar structure (model should recognize)")
    lines.append("  CosSim 0.7-0.85: Moderately similar (model might recognize)")
    lines.append("  CosSim < 0.7: Structurally different (model likely cannot generalize)")
    lines.append("")
    lines.append("  Gap > 0: Nearest positive closer than nearest negative → model has positive reference")
    lines.append("  Gap < 0: Nearest negative closer → test positive LOOKS LIKE a negative to the model")
    lines.append("  Gap << -0.1: Severely isolated in negative territory → fundamentally hard case")
    lines.append(f"\n{'='*W}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved report: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MOF Embedding Analysis & Nearest-Neighbor Diagnosis')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory (contains .graphdata, .griddata16 files)')
    parser.add_argument('--splits_dir', type=str, default=None,
                        help='Path to directory with split JSON files '
                             '(default: same as data_dir)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned model checkpoint (.ckpt). '
                             'If not provided, uses pretrained MOFTransformer only.')
    parser.add_argument('--output_dir', type=str, default='./embedding_analysis',
                        help='Output directory for results')
    parser.add_argument('--downstream', type=str, default='bandgaps_regression',
                        help='Downstream task name (for label files)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive class (eV)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for embedding extraction')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    args = parser.parse_args()

    if args.splits_dir is None:
        args.splits_dir = args.data_dir

    os.makedirs(args.output_dir, exist_ok=True)

    # Build config
    config = default_config_fn()
    config = json.loads(json.dumps(config))
    config["data_dir"] = args.data_dir
    config["downstream"] = args.downstream
    config["load_path"] = "pmtransformer"
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config = get_valid_config(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ===========================
    # Step 1: Pretrained embeddings
    # ===========================
    print(f"\n{'='*70}")
    print("STEP 1: Extract pretrained MOFTransformer embeddings")
    print(f"{'='*70}")

    model_pt = create_model(config, checkpoint_path=None)
    cif_ids, embs_pt, bandgaps, split_labels = extract_embeddings(
        model_pt, args.data_dir, args.splits_dir, args.downstream, config,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device
    )
    del model_pt
    torch.cuda.empty_cache() if device == 'cuda' else None

    # Save
    np.savez_compressed(
        os.path.join(args.output_dir, 'embeddings_pretrained.npz'),
        cif_ids=np.array(cif_ids),
        embeddings=embs_pt,
        bandgaps=bandgaps,
        splits=np.array(split_labels),
    )
    print(f"  Saved pretrained embeddings: {embs_pt.shape}")

    # Nearest-neighbor analysis (pretrained)
    print(f"\n{'='*70}")
    print("STEP 2: Nearest-neighbor analysis (pretrained)")
    print(f"{'='*70}")

    nn_pt = nn_analysis(cif_ids, embs_pt, bandgaps, split_labels, args.threshold)
    dist_pt = distribution_analysis(cif_ids, embs_pt, bandgaps, split_labels, args.threshold)
    generate_report(nn_pt, dist_pt,
                    os.path.join(args.output_dir, 'embedding_analysis_report_pretrained.txt'),
                    model_name="Pretrained MOFTransformer")

    # Plots (pretrained)
    print(f"\n{'='*70}")
    print("STEP 3: Visualization (pretrained)")
    print(f"{'='*70}")

    plot_umap(embs_pt, bandgaps, split_labels, args.threshold,
              os.path.join(args.output_dir, 'embedding_umap_pretrained.png'))
    plot_similarity_heatmap(cif_ids, embs_pt, bandgaps, split_labels, args.threshold,
                            os.path.join(args.output_dir, 'similarity_heatmap_pretrained.png'))

    # ===========================
    # Step 2: Fine-tuned embeddings (optional)
    # ===========================
    if args.checkpoint:
        print(f"\n{'='*70}")
        print("STEP 4: Extract fine-tuned model embeddings")
        print(f"{'='*70}")

        model_ft = create_model(config, checkpoint_path=args.checkpoint)
        _, embs_ft, _, _ = extract_embeddings(
            model_ft, args.data_dir, args.splits_dir, args.downstream, config,
            batch_size=args.batch_size, num_workers=args.num_workers, device=device
        )
        del model_ft
        torch.cuda.empty_cache() if device == 'cuda' else None

        np.savez_compressed(
            os.path.join(args.output_dir, 'embeddings_finetuned.npz'),
            cif_ids=np.array(cif_ids),
            embeddings=embs_ft,
            bandgaps=bandgaps,
            splits=np.array(split_labels),
        )
        print(f"  Saved fine-tuned embeddings: {embs_ft.shape}")

        nn_ft = nn_analysis(cif_ids, embs_ft, bandgaps, split_labels, args.threshold)
        dist_ft = distribution_analysis(cif_ids, embs_ft, bandgaps, split_labels, args.threshold)
        generate_report(nn_ft, dist_ft,
                        os.path.join(args.output_dir, 'embedding_analysis_report_finetuned.txt'),
                        model_name="Fine-tuned Model")

        plot_umap(embs_ft, bandgaps, split_labels, args.threshold,
                  os.path.join(args.output_dir, 'embedding_umap_finetuned.png'))
        plot_similarity_heatmap(cif_ids, embs_ft, bandgaps, split_labels, args.threshold,
                                os.path.join(args.output_dir, 'similarity_heatmap_finetuned.png'))

        # Compare pretrained vs fine-tuned
        print(f"\n{'='*70}")
        print("STEP 5: Pretrained vs Fine-tuned comparison")
        print(f"{'='*70}")

        # How much do embeddings change after fine-tuning?
        cos_change = np.array([
            float(cosine_similarity_matrix(embs_pt[i:i+1], embs_ft[i:i+1])[0, 0])
            for i in range(len(cif_ids))
        ])

        pos_mask = bandgaps < args.threshold
        train_mask = np.array([s == 'train' for s in split_labels])
        test_mask = np.array([s == 'test' for s in split_labels])

        print(f"  Embedding change (cosine sim between pretrained & fine-tuned):")
        print(f"    All samples:     {cos_change.mean():.4f} ± {cos_change.std():.4f}")
        print(f"    Train positives: {cos_change[train_mask & pos_mask].mean():.4f}")
        print(f"    Train negatives: {cos_change[train_mask & ~pos_mask].mean():.4f}")
        print(f"    Test positives:  {cos_change[test_mask & pos_mask].mean():.4f}")
        print(f"    Test negatives:  {cos_change[test_mask & ~pos_mask].mean():.4f}")

    print(f"\n{'='*70}")
    print(f"DONE. All results saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
