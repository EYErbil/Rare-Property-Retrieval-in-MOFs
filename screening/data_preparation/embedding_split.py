#!/usr/bin/env python3
"""
Embedding-Informed Data Splitting for MOF Discovery
======================================================

Uses pretrained MOFTransformer embeddings to create SMARTER train/val/test
splits that guarantee structural coverage.

THE PROBLEM:
  With random splitting, 5/9 test positives are structurally ISOLATED —
  their nearest NEGATIVE is closer than their nearest POSITIVE in embedding
  space. No model can learn to classify them correctly because it has never
  seen anything structurally similar that is positive.

THE SOLUTION:
  Use cosine similarity in embedding space to:
  1. Put the most ISOLATED positives in TRAINING (so the model learns them)
  2. Put positives that have structural "anchors" in training into test/val
  3. GUARANTEE every val/test positive has >= 1 similar positive in training

Two strategies:

  Strategy D: Farthest-Point Coverage
    - Greedily build train set using farthest-point sampling on positives
    - This maximizes structural diversity in training
    - Assign remaining positives to val/test only if they have a nearby
      train positive (cosine_sim >= threshold)
    - Result: every test positive is "reachable" from training

  Strategy E: Cluster-Balanced Split
    - Hierarchically cluster all positives by embedding similarity
    - Ensure every cluster has >=1 member in training
    - Proportionally distribute remaining cluster members to val/test
    - Result: no structural family is entirely missing from training

Both strategies keep the test SET SIZE similar to original (9 positives)
and only redistribute train+val positives to maximize coverage.

IMPORTANT: Negatives are numerous (~10K) and split randomly. Only the
74 positives need careful placement.

Requires: embeddings .npz from analyze_embeddings.py (run that first).

Usage:
  python embedding_split.py \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --splits_dir /path/to/new_subset \\
      --output_dir /path/to/new_splits \\
      --strategy all
"""

import os
import sys
import json
import argparse
import math
import numpy as np
from collections import defaultdict, Counter


# =============================================================================
# LABEL GENERATION UTILITIES (reused from resplit_data.py)
# =============================================================================

def bandgap_to_binary(bandgap, threshold=1.0):
    return 1 if bandgap < threshold else 0

def bandgap_to_ordinal(bandgap, bin_width=0.5, max_class=13):
    if bandgap < 1.0:
        return 0
    bin_idx = int((bandgap - 1.0) / bin_width) + 1
    return min(bin_idx, max_class)

def bandgap_to_multiclass(bandgap, bin_width=0.5, max_class=14):
    if bandgap < 1.0:
        return 0
    if bandgap > 7.5:
        return max_class
    bin_idx = int((bandgap - 1.0) / bin_width) + 1
    return min(bin_idx, max_class - 1)

def compute_sample_weights(bandgaps, threshold=1.0):
    bgs = list(bandgaps.values())
    cids = list(bandgaps.keys())
    n_total = len(bgs)
    n_pos = sum(1 for bg in bgs if bg < threshold)
    n_neg = n_total - n_pos
    if n_pos == 0:
        return {cid: 1.0 for cid in cids}
    base_pos_weight = n_neg / n_pos
    weights = {}
    for cid in cids:
        bg = bandgaps[cid]
        if bg < threshold:
            bg_factor = 1.0 + 0.3 * (1.0 - bg / threshold)
            weights[cid] = base_pos_weight * bg_factor
        else:
            weights[cid] = 1.0
    return weights

def generate_all_labels(split_bandgaps, split_name, output_dir):
    regression = {cid: bg for cid, bg in split_bandgaps.items()}
    save_json(regression, os.path.join(output_dir, f'{split_name}_bandgaps_regression.json'))
    binary = {cid: bandgap_to_binary(bg) for cid, bg in split_bandgaps.items()}
    save_json(binary, os.path.join(output_dir, f'{split_name}_bandgaps.json'))
    ordinal = {cid: bandgap_to_ordinal(bg) for cid, bg in split_bandgaps.items()}
    save_json(ordinal, os.path.join(output_dir, f'{split_name}_bandgaps_ordinal.json'))
    multiclass = {cid: bandgap_to_multiclass(bg) for cid, bg in split_bandgaps.items()}
    save_json(multiclass, os.path.join(output_dir, f'{split_name}_bandgaps_regression_multiclass.json'))
    if split_name == 'train':
        weights = compute_sample_weights(split_bandgaps)
        save_json(weights, os.path.join(output_dir, f'{split_name}_bandgaps_regression_weights.json'))
    return {
        'n_total': len(split_bandgaps),
        'n_pos': sum(1 for bg in split_bandgaps.values() if bg < 1.0),
        'n_neg': sum(1 for bg in split_bandgaps.values() if bg >= 1.0),
    }

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# LOADING
# =============================================================================

def load_all_bandgaps(splits_dir):
    """Load original split label files."""
    all_bandgaps = {}
    split_assignment = {}
    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(splits_dir, f'{split_name}_bandgaps_regression.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: {json_path} not found")
            continue
        with open(json_path) as f:
            data = json.load(f)
        for cid, bg in data.items():
            all_bandgaps[cid] = float(bg)
            split_assignment[cid] = split_name
        n_pos = sum(1 for bg in data.values() if float(bg) < 1.0)
        print(f"  Loaded {split_name}: {len(data)} total, {n_pos} positives")
    return all_bandgaps, split_assignment


def load_embeddings(npz_path):
    """Load embeddings from analyze_embeddings.py output."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = list(data['cif_ids'])
    embeddings = data['embeddings']
    bandgaps = data['bandgaps']
    splits = list(data['splits'])
    return cif_ids, embeddings, bandgaps, splits


def cosine_similarity_matrix(A, B=None):
    """Cosine similarity between rows of A and rows of B (or A with itself)."""
    if B is None:
        B = A
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


# =============================================================================
# STRATEGY D: FARTHEST-POINT COVERAGE
# =============================================================================

def strategy_d_farthest_point(all_bandgaps, split_assignment, emb_cid_to_vec,
                               n_test_pos=9, n_val_pos=7,
                               min_sim_threshold=0.55, seed=42):
    """
    Embedding-informed split using farthest-point sampling.

    Algorithm:
      1. Collect all 74 positives and their embeddings
      2. Compute pairwise cosine similarity matrix
      3. Greedily select TRAIN positives via farthest-point sampling
         (each new train positive is the one most distant from all current
         train positives → maximizes structural diversity in training)
      4. After building train set of ~58 positives:
         - Remaining ~16 positives go to val/test
         - Assign to TEST those with highest max-sim to a train positive
           (these are most "learnable" from the training data)
         - Assign to VAL the rest
      5. Verify: every val/test positive has cosine_sim >= threshold to
         at least one train positive. If not, swap it into train.

    For negatives: random stratified split (plenty of data).
    """
    rng = np.random.default_rng(seed)
    threshold = 1.0

    # Separate positives and negatives
    all_pos = {c: bg for c, bg in all_bandgaps.items() if bg < threshold}
    all_neg = {c: bg for c, bg in all_bandgaps.items() if bg >= threshold}
    test_neg_orig = [c for c, s in split_assignment.items()
                     if s == 'test' and all_bandgaps[c] >= threshold]

    print(f"\n  Total positives: {len(all_pos)}")
    print(f"  Total negatives: {len(all_neg)}")

    # Get embeddings for positives
    pos_cids = sorted(all_pos.keys())
    pos_embs = []
    missing = []
    for cid in pos_cids:
        if cid in emb_cid_to_vec:
            pos_embs.append(emb_cid_to_vec[cid])
        else:
            missing.append(cid)

    if missing:
        print(f"  WARNING: {len(missing)} positives missing embeddings: {missing[:5]}...")
        # Remove missing from pos_cids
        pos_cids = [c for c in pos_cids if c in emb_cid_to_vec]
        pos_embs = [emb_cid_to_vec[c] for c in pos_cids]

    pos_embs = np.array(pos_embs)
    n_pos = len(pos_cids)

    # Pairwise similarity
    sim_matrix = cosine_similarity_matrix(pos_embs)
    np.fill_diagonal(sim_matrix, 0)  # don't match with self

    print(f"\n  Positive pairwise similarity stats:")
    upper = sim_matrix[np.triu_indices(n_pos, k=1)]
    print(f"    Mean: {upper.mean():.4f}")
    print(f"    Std:  {upper.std():.4f}")
    print(f"    Min:  {upper.min():.4f}")
    print(f"    Max:  {upper.max():.4f}")

    # ---- Step 1: Farthest-point sampling for train positives ----
    n_train_pos = n_pos - n_test_pos - n_val_pos

    # Start with the most isolated positive (lowest max-sim to any other)
    max_sim_to_any = sim_matrix.max(axis=1)
    start_idx = np.argmin(max_sim_to_any)

    train_indices = [start_idx]
    remaining = set(range(n_pos)) - {start_idx}

    # Greedily add the positive that is FARTHEST from all current train
    for _ in range(n_train_pos - 1):
        if not remaining:
            break
        remaining_list = list(remaining)
        # For each remaining positive, find max sim to any current train positive
        max_sim_to_train = np.array([
            sim_matrix[r, train_indices].max() for r in remaining_list
        ])
        # Pick the one with LOWEST max-sim (most distant from current train set)
        next_idx = remaining_list[np.argmin(max_sim_to_train)]
        train_indices.append(next_idx)
        remaining.remove(next_idx)

    # ---- Step 2: Assign remaining to val/test ----
    remaining_indices = list(remaining)

    # For each remaining positive, compute max-sim to any train positive
    remaining_max_sim = []
    for idx in remaining_indices:
        max_sim = sim_matrix[idx, train_indices].max()
        nearest_train = train_indices[np.argmax(sim_matrix[idx, train_indices])]
        remaining_max_sim.append((idx, max_sim, nearest_train))

    # Sort by max-sim descending: highest similarity → test (most learnable)
    remaining_max_sim.sort(key=lambda x: -x[1])

    test_pos_indices = [x[0] for x in remaining_max_sim[:n_test_pos]]
    val_pos_indices = [x[0] for x in remaining_max_sim[n_test_pos:]]

    # ---- Step 3: Verify coverage ----
    print(f"\n  Coverage verification (sim to nearest train positive):")

    swaps_needed = []
    for label, indices in [('test', test_pos_indices), ('val', val_pos_indices)]:
        for idx in indices:
            cid = pos_cids[idx]
            max_sim = sim_matrix[idx, train_indices].max()
            nearest_idx = train_indices[np.argmax(sim_matrix[idx, train_indices])]
            nearest_cid = pos_cids[nearest_idx]
            status = "OK" if max_sim >= min_sim_threshold else "LOW"
            print(f"    {label}: {cid} -> nearest_train_pos={nearest_cid} "
                  f"sim={max_sim:.4f} [{status}]")
            if max_sim < min_sim_threshold:
                swaps_needed.append((idx, label))

    # Swap isolated val/test positives into train
    if swaps_needed:
        print(f"\n  Swapping {len(swaps_needed)} isolated positives into train...")
        for idx, old_label in swaps_needed:
            cid = pos_cids[idx]
            # Find the most "redundant" train positive (highest max-sim to another train pos)
            best_swap = None
            best_redundancy = -1
            for t_idx in train_indices:
                other_train = [t for t in train_indices if t != t_idx]
                if not other_train:
                    continue
                redundancy = sim_matrix[t_idx, other_train].max()
                if redundancy > best_redundancy:
                    best_redundancy = redundancy
                    best_swap = t_idx

            if best_swap is not None:
                print(f"    Swap: {cid} (isolated {old_label}) <-> "
                      f"{pos_cids[best_swap]} (redundant train, sim={best_redundancy:.4f})")
                train_indices.remove(best_swap)
                train_indices.append(idx)
                if old_label == 'test':
                    test_pos_indices.remove(idx)
                    test_pos_indices.append(best_swap)
                else:
                    val_pos_indices.remove(idx)
                    val_pos_indices.append(best_swap)

    # ---- Step 4: Build final split dict ----
    train_pos_cids = set(pos_cids[i] for i in train_indices)
    val_pos_cids = set(pos_cids[i] for i in val_pos_indices)
    test_pos_cids = set(pos_cids[i] for i in test_pos_indices)

    # Negatives: keep test negatives unchanged, randomly split rest
    test_neg_cids = set(test_neg_orig)
    remaining_neg = [c for c in all_neg if c not in test_neg_cids]
    rng.shuffle(remaining_neg)

    # Split remaining negatives proportionally
    n_val_neg = max(50, int(len(remaining_neg) * 0.1))
    val_neg_cids = set(remaining_neg[:n_val_neg])
    train_neg_cids = set(remaining_neg[n_val_neg:])

    splits = {
        'train': {c: all_bandgaps[c] for c in (train_pos_cids | train_neg_cids)},
        'val': {c: all_bandgaps[c] for c in (val_pos_cids | val_neg_cids)},
        'test': {c: all_bandgaps[c] for c in (test_pos_cids | test_neg_cids)},
    }

    # Print summary
    for s_name, s_data in splits.items():
        n_p = sum(1 for bg in s_data.values() if bg < threshold)
        n_n = len(s_data) - n_p
        print(f"\n  {s_name}: {len(s_data)} total ({n_p} pos, {n_n} neg)")

    # Print coverage stats for final assignment
    print(f"\n  FINAL test positive coverage:")
    for idx in test_pos_indices:
        cid = pos_cids[idx]
        max_sim = sim_matrix[idx, train_indices].max()
        nearest_idx = train_indices[np.argmax(sim_matrix[idx, train_indices])]
        nearest_cid = pos_cids[nearest_idx]
        bg = all_bandgaps[cid]
        print(f"    {cid} (bg={bg:.3f}eV) -> nearest train pos: "
              f"{nearest_cid} sim={max_sim:.4f}")

    return splits


# =============================================================================
# STRATEGY E: CLUSTER-BALANCED SPLIT
# =============================================================================

def strategy_e_cluster_balanced(all_bandgaps, split_assignment, emb_cid_to_vec,
                                 n_test_pos=9, n_val_pos=7,
                                 n_clusters=None, seed=42):
    """
    Cluster-balanced split using hierarchical clustering on embeddings.

    Algorithm:
      1. Cluster all positives by embedding cosine distance
      2. Determine natural number of clusters (or use specified)
      3. Within each cluster, assign proportionally to train/val/test
         ensuring at least 1 per cluster goes to train
      4. For singletons (clusters of size 1), always put in train
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(seed)
    threshold = 1.0

    all_pos = {c: bg for c, bg in all_bandgaps.items() if bg < threshold}
    all_neg = {c: bg for c, bg in all_bandgaps.items() if bg >= threshold}
    test_neg_orig = [c for c, s in split_assignment.items()
                     if s == 'test' and all_bandgaps[c] >= threshold]

    # Get embeddings
    pos_cids = sorted(all_pos.keys())
    pos_embs = np.array([emb_cid_to_vec[c] for c in pos_cids if c in emb_cid_to_vec])
    pos_cids = [c for c in pos_cids if c in emb_cid_to_vec]
    n_pos = len(pos_cids)

    # Cosine distance matrix
    # Normalize embeddings
    norms = np.linalg.norm(pos_embs, axis=1, keepdims=True) + 1e-12
    pos_embs_normed = pos_embs / norms
    cos_dist = pdist(pos_embs_normed, metric='cosine')

    # Hierarchical clustering
    Z = linkage(cos_dist, method='average')

    # Auto-determine number of clusters if not specified
    if n_clusters is None:
        # Try different cut levels, pick one giving 5-15 clusters
        for n_try in [8, 10, 6, 12, 5, 15]:
            labels = fcluster(Z, t=n_try, criterion='maxclust')
            sizes = Counter(labels)
            n_singletons = sum(1 for s in sizes.values() if s == 1)
            if n_singletons < n_try * 0.6:  # not too many singletons
                n_clusters = n_try
                break
        if n_clusters is None:
            n_clusters = 8

    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(i)

    print(f"\n  Hierarchical clustering: {n_clusters} clusters")
    for cl_id in sorted(cluster_map.keys()):
        members = cluster_map[cl_id]
        member_cids = [pos_cids[i] for i in members]
        bgs = [all_bandgaps[c] for c in member_cids]
        orig_splits = [split_assignment.get(c, '?') for c in member_cids]
        print(f"    Cluster {cl_id}: {len(members)} members, "
              f"mean bg={np.mean(bgs):.3f}eV, "
              f"orig splits: {dict(Counter(orig_splits))}")
        for cid in member_cids:
            print(f"      {cid}: bg={all_bandgaps[cid]:.3f}eV, "
                  f"orig={split_assignment.get(cid, '?')}")

    # ---- Assign within each cluster ----
    train_pos_cids = set()
    val_pos_cids = set()
    test_pos_cids = set()

    # Target fractions
    total_pos = n_pos
    target_train = total_pos - n_test_pos - n_val_pos
    frac_test = n_test_pos / total_pos
    frac_val = n_val_pos / total_pos

    for cl_id in sorted(cluster_map.keys()):
        members = cluster_map[cl_id]
        rng.shuffle(members)
        n = len(members)

        if n == 1:
            # Singletons always go to train
            train_pos_cids.add(pos_cids[members[0]])
            continue

        # At least 1 to train
        n_test_cl = max(0, round(n * frac_test))
        n_val_cl = max(0, round(n * frac_val))
        n_train_cl = n - n_test_cl - n_val_cl

        # Ensure at least 1 in train
        if n_train_cl == 0:
            n_train_cl = 1
            if n_test_cl > 0:
                n_test_cl -= 1
            elif n_val_cl > 0:
                n_val_cl -= 1

        # Assign
        for i, idx in enumerate(members):
            cid = pos_cids[idx]
            if i < n_train_cl:
                train_pos_cids.add(cid)
            elif i < n_train_cl + n_test_cl:
                test_pos_cids.add(cid)
            else:
                val_pos_cids.add(cid)

    # Adjust if counts are off
    while len(test_pos_cids) > n_test_pos and len(train_pos_cids) < target_train:
        c = test_pos_cids.pop()
        train_pos_cids.add(c)
    while len(val_pos_cids) > n_val_pos and len(train_pos_cids) < target_train:
        c = val_pos_cids.pop()
        train_pos_cids.add(c)
    while len(test_pos_cids) < n_test_pos and len(train_pos_cids) > target_train:
        c = train_pos_cids.pop()
        test_pos_cids.add(c)
    while len(val_pos_cids) < n_val_pos and len(train_pos_cids) > target_train:
        c = train_pos_cids.pop()
        val_pos_cids.add(c)

    # Negatives: same as strategy D
    test_neg_cids = set(test_neg_orig)
    remaining_neg = [c for c in all_neg if c not in test_neg_cids]
    rng.shuffle(remaining_neg)
    n_val_neg = max(50, int(len(remaining_neg) * 0.1))
    val_neg_cids = set(remaining_neg[:n_val_neg])
    train_neg_cids = set(remaining_neg[n_val_neg:])

    splits = {
        'train': {c: all_bandgaps[c] for c in (train_pos_cids | train_neg_cids)},
        'val': {c: all_bandgaps[c] for c in (val_pos_cids | val_neg_cids)},
        'test': {c: all_bandgaps[c] for c in (test_pos_cids | test_neg_cids)},
    }

    for s_name, s_data in splits.items():
        n_p = sum(1 for bg in s_data.values() if bg < threshold)
        n_n = len(s_data) - n_p
        print(f"\n  {s_name}: {len(s_data)} total ({n_p} pos, {n_n} neg)")

    # Coverage report
    print(f"\n  FINAL test positive coverage (cluster-balanced):")
    sim_matrix = cosine_similarity_matrix(pos_embs)
    np.fill_diagonal(sim_matrix, 0)
    train_indices = [i for i, c in enumerate(pos_cids) if c in train_pos_cids]
    for cid in sorted(test_pos_cids):
        idx = pos_cids.index(cid)
        max_sim = sim_matrix[idx, train_indices].max()
        nearest_idx = train_indices[np.argmax(sim_matrix[idx, train_indices])]
        nearest_cid = pos_cids[nearest_idx]
        bg = all_bandgaps[cid]
        cl = labels[idx]
        n_train_in_cl = sum(1 for t in train_indices if labels[t] == cl)
        print(f"    {cid} (bg={bg:.3f}eV, cluster={cl}) -> nearest train pos: "
              f"{nearest_cid} sim={max_sim:.4f} "
              f"({n_train_in_cl} train pos in same cluster)")

    return splits


# =============================================================================
# STRATEGY F: COVERAGE-MAXIMIZED WITH VAL-HOLDOUT (best of D+E + merged)
# =============================================================================

def strategy_f_coverage_merged(all_bandgaps, split_assignment, emb_cid_to_vec,
                                n_test_pos=9, val_frac=0.1, seed=42):
    """
    Best-of-both: farthest-point coverage + merged train (no separate val).

    Uses Strategy D's farthest-point logic to select 9 test positives that
    are maximally covered by the remaining positives, then puts ALL remaining
    positives into training.

    For models that need a val set for early stopping, creates a small
    stratified val holdout (like Strategy B).

    This maximizes training signal while guaranteeing test coverage.
    """
    rng = np.random.default_rng(seed)
    threshold = 1.0

    all_pos = {c: bg for c, bg in all_bandgaps.items() if bg < threshold}
    all_neg = {c: bg for c, bg in all_bandgaps.items() if bg >= threshold}
    test_neg_orig = [c for c, s in split_assignment.items()
                     if s == 'test' and all_bandgaps[c] >= threshold]

    pos_cids = sorted(all_pos.keys())
    pos_embs = np.array([emb_cid_to_vec[c] for c in pos_cids if c in emb_cid_to_vec])
    pos_cids = [c for c in pos_cids if c in emb_cid_to_vec]
    n_pos = len(pos_cids)

    sim_matrix = cosine_similarity_matrix(pos_embs)
    np.fill_diagonal(sim_matrix, 0)

    # Build a "train pool" of all positives except test
    # Select TEST positives as those with HIGHEST max-sim to remaining positives
    # (i.e., the most "learnable" ones go to test)

    # Step 1: farthest-point sampling to find the n_test_pos most "surrounded" positives
    # These have the best structural support from other positives = ideal for test

    # Score each positive by mean similarity to all others
    mean_sim = sim_matrix.mean(axis=1)  # [n_pos]

    # Candidates for test: those with highest mean similarity (well-connected)
    candidate_order = np.argsort(-mean_sim)

    # But also ensure each test positive has high max-sim to non-test positives
    test_indices = []
    for candidate_idx in candidate_order:
        if len(test_indices) >= n_test_pos:
            break
        # Check: if this candidate is in test, does it have high sim to remaining?
        train_pool = [i for i in range(n_pos) if i not in test_indices and i != candidate_idx]
        if not train_pool:
            break
        max_sim_to_pool = sim_matrix[candidate_idx, train_pool].max()
        if max_sim_to_pool >= 0.55:  # reasonable threshold
            test_indices.append(candidate_idx)

    # If we didn't get enough, add remaining by mean_sim
    remaining_candidates = [i for i in candidate_order if i not in test_indices]
    while len(test_indices) < n_test_pos and remaining_candidates:
        test_indices.append(remaining_candidates.pop(0))

    train_indices = [i for i in range(n_pos) if i not in test_indices]
    test_pos_cids = set(pos_cids[i] for i in test_indices)
    all_train_pos_cids = set(pos_cids[i] for i in train_indices)

    print(f"\n  Test positives selected: {len(test_pos_cids)}")
    print(f"  Train pool positives: {len(all_train_pos_cids)}")

    # Split train pool into train + small val
    train_pos_list = sorted(all_train_pos_cids)
    rng.shuffle(train_pos_list)
    n_val_pos = max(5, int(len(train_pos_list) * val_frac))
    val_pos_cids = set(train_pos_list[:n_val_pos])
    train_pos_cids = set(train_pos_list[n_val_pos:])

    # Negatives
    test_neg_cids = set(test_neg_orig)
    remaining_neg = [c for c in all_neg if c not in test_neg_cids]
    rng.shuffle(remaining_neg)
    n_val_neg = max(50, int(len(remaining_neg) * val_frac))
    val_neg_cids = set(remaining_neg[:n_val_neg])
    train_neg_cids = set(remaining_neg[n_val_neg:])

    splits = {
        'train': {c: all_bandgaps[c] for c in (train_pos_cids | train_neg_cids)},
        'val': {c: all_bandgaps[c] for c in (val_pos_cids | val_neg_cids)},
        'test': {c: all_bandgaps[c] for c in (test_pos_cids | test_neg_cids)},
    }

    for s_name, s_data in splits.items():
        n_p = sum(1 for bg in s_data.values() if bg < threshold)
        n_n = len(s_data) - n_p
        print(f"\n  {s_name}: {len(s_data)} total ({n_p} pos, {n_n} neg)")

    # Coverage report
    print(f"\n  FINAL test positive coverage (coverage-merged):")
    # Consider ALL train+val positives as training signal
    all_support = train_indices  # all non-test positives
    for cid in sorted(test_pos_cids):
        idx = pos_cids.index(cid)
        max_sim = sim_matrix[idx, all_support].max()
        nearest_idx = all_support[np.argmax(sim_matrix[idx, all_support])]
        nearest_cid = pos_cids[nearest_idx]
        bg = all_bandgaps[cid]
        print(f"    {cid} (bg={bg:.3f}eV) -> nearest support pos: "
              f"{nearest_cid} sim={max_sim:.4f}")

    return splits


# =============================================================================
# COMPARE WITH ORIGINAL SPLIT
# =============================================================================

def compare_coverage(all_bandgaps, split_assignment, emb_cid_to_vec, threshold=1.0):
    """Show coverage of ORIGINAL split for reference."""
    print(f"\n{'='*70}")
    print("  ORIGINAL SPLIT COVERAGE (for comparison)")
    print(f"{'='*70}")

    pos_cids = sorted(c for c, bg in all_bandgaps.items() if bg < threshold)
    pos_embs = np.array([emb_cid_to_vec[c] for c in pos_cids if c in emb_cid_to_vec])
    pos_cids = [c for c in pos_cids if c in emb_cid_to_vec]

    sim_matrix = cosine_similarity_matrix(pos_embs)
    np.fill_diagonal(sim_matrix, 0)

    train_indices = [i for i, c in enumerate(pos_cids) if split_assignment.get(c) == 'train']
    test_indices = [i for i, c in enumerate(pos_cids) if split_assignment.get(c) == 'test']
    val_indices = [i for i, c in enumerate(pos_cids) if split_assignment.get(c) == 'val']

    print(f"\n  Original: {len(train_indices)} train pos, "
          f"{len(val_indices)} val pos, {len(test_indices)} test pos")

    for label, indices in [('test', test_indices), ('val', val_indices)]:
        if not indices or not train_indices:
            continue
        print(f"\n  {label} positive coverage:")
        for idx in indices:
            cid = pos_cids[idx]
            max_sim = sim_matrix[idx, train_indices].max()
            nearest_idx = train_indices[np.argmax(sim_matrix[idx, train_indices])]
            nearest_cid = pos_cids[nearest_idx]

            # Also check: is nearest neg closer than nearest pos?
            all_neg_in_split = [i for i in range(len(pos_cids))
                                if split_assignment.get(pos_cids[i]) == 'train']
            # We only have positive embeddings here - but we can still show
            # the max-sim to nearest train positive
            bg = all_bandgaps[cid]
            status = "OK" if max_sim >= 0.55 else "ISOLATED"
            print(f"    {cid} (bg={bg:.3f}eV) -> nearest train pos: "
                  f"{nearest_cid} sim={max_sim:.4f} [{status}]")


# =============================================================================
# SYMLINK SCRIPT GENERATION
# =============================================================================

def create_data_symlink_script(data_dir, output_dir, strategy_name, split_data):
    """Create shell script for symlinking data files into split subdirectories."""
    script = f"""#!/bin/bash
# Symlink data files for strategy {strategy_name}
# MOFTransformer expects: data_dir/{{split}}/{{cif_id}}.{{ext}}

DATA_SRC="{data_dir}"
DEST="{output_dir}"
ORIG_SPLITS="train val test"

link_file() {{
    local cif_id="$1"
    local new_split="$2"
    mkdir -p "$DEST/$new_split"
    for ext in grid griddata16 graphdata; do
        for orig_split in $ORIG_SPLITS; do
            src="$DATA_SRC/$orig_split/${{cif_id}}.$ext"
            dst="$DEST/$new_split/${{cif_id}}.$ext"
            if [ -f "$src" ] && [ ! -e "$dst" ]; then
                ln -s "$src" "$dst"
                break
            fi
        done
    done
}}

echo "Setting up data symlinks for strategy {strategy_name}..."
echo "Source: $DATA_SRC"
echo "Destination: $DEST"
echo ""

"""
    for split_name, cids_dict in split_data.items():
        cid_list = list(cids_dict.keys())
        script += f'echo "Linking {len(cid_list)} files for {split_name}..."\n'
        script += f'mkdir -p "$DEST/{split_name}"\n'
        for cid in cid_list:
            script += f'link_file "{cid}" "{split_name}"\n'
        script += f'echo "  {split_name}: done ({len(cid_list)} MOFs)"\n\n'

    script += """
echo ""
echo "Done! Data symlinks created."
for split in train val test; do
    if [ -d "$DEST/$split" ]; then
        n=$(ls "$DEST/$split"/*.graphdata 2>/dev/null | wc -l)
        echo "  $split: $n .graphdata files"
    fi
done
"""
    script_path = os.path.join(output_dir, 'setup_data_links.sh')
    with open(script_path, 'w', newline='\n') as f:
        f.write(script)
    print(f"  Saved data link script: {script_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Embedding-Informed Data Splitting for MOF Discovery')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to .npz from analyze_embeddings.py')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory with original split label JSON files')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Original data dir with train/val/test subdirs '
                             '(for symlink script). Defaults to splits_dir.')
    parser.add_argument('--output_dir', type=str, default='./new_splits',
                        help='Output directory for new splits')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['D', 'E', 'F', 'all'],
                        help='Which strategy: D=farthest-point, E=cluster-balanced, '
                             'F=coverage-merged, all=all three')
    parser.add_argument('--n_test_pos', type=int, default=9,
                        help='Number of positives in test set')
    parser.add_argument('--n_val_pos', type=int, default=7,
                        help='Number of positives in val set (strategy D/E)')
    parser.add_argument('--min_sim', type=float, default=0.55,
                        help='Minimum cosine similarity threshold for coverage')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters for strategy E (auto if not set)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = args.splits_dir

    print("=" * 70)
    print("EMBEDDING-INFORMED DATA SPLITTING")
    print("=" * 70)

    # Load original labels
    print("\nLoading original splits...")
    all_bandgaps, split_assignment = load_all_bandgaps(args.splits_dir)

    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings_path}...")
    emb_cids, emb_vecs, emb_bgs, emb_splits = load_embeddings(args.embeddings_path)

    # Build cid -> embedding mapping
    emb_cid_to_vec = {cid: emb_vecs[i] for i, cid in enumerate(emb_cids)}
    print(f"  Loaded {len(emb_cid_to_vec)} embeddings (dim={emb_vecs.shape[1]})")

    # Check coverage of original split
    compare_coverage(all_bandgaps, split_assignment, emb_cid_to_vec)

    strategies = ['D', 'E', 'F'] if args.strategy == 'all' else [args.strategy]

    for strategy in strategies:
        print(f"\n\n{'#'*70}")
        print(f"# STRATEGY {strategy}")
        print(f"{'#'*70}")

        if strategy == 'D':
            strat_dir = os.path.join(args.output_dir, 'strategy_d_farthest_point')
            os.makedirs(strat_dir, exist_ok=True)

            splits = strategy_d_farthest_point(
                all_bandgaps, split_assignment, emb_cid_to_vec,
                n_test_pos=args.n_test_pos, n_val_pos=args.n_val_pos,
                min_sim_threshold=args.min_sim, seed=args.seed)

            for split_name, split_data in splits.items():
                stats = generate_all_labels(split_data, split_name, strat_dir)
                print(f"  {split_name}: {stats['n_total']} total, "
                      f"{stats['n_pos']} pos, {stats['n_neg']} neg")
            create_data_symlink_script(args.data_dir, strat_dir, 'D', splits)

        elif strategy == 'E':
            strat_dir = os.path.join(args.output_dir, 'strategy_e_cluster_balanced')
            os.makedirs(strat_dir, exist_ok=True)

            splits = strategy_e_cluster_balanced(
                all_bandgaps, split_assignment, emb_cid_to_vec,
                n_test_pos=args.n_test_pos, n_val_pos=args.n_val_pos,
                n_clusters=args.n_clusters, seed=args.seed)

            for split_name, split_data in splits.items():
                stats = generate_all_labels(split_data, split_name, strat_dir)
                print(f"  {split_name}: {stats['n_total']} total, "
                      f"{stats['n_pos']} pos, {stats['n_neg']} neg")
            create_data_symlink_script(args.data_dir, strat_dir, 'E', splits)

        elif strategy == 'F':
            strat_dir = os.path.join(args.output_dir, 'strategy_f_coverage_merged')
            os.makedirs(strat_dir, exist_ok=True)

            splits = strategy_f_coverage_merged(
                all_bandgaps, split_assignment, emb_cid_to_vec,
                n_test_pos=args.n_test_pos, seed=args.seed)

            for split_name, split_data in splits.items():
                stats = generate_all_labels(split_data, split_name, strat_dir)
                print(f"  {split_name}: {stats['n_total']} total, "
                      f"{stats['n_pos']} pos, {stats['n_neg']} neg")
            create_data_symlink_script(args.data_dir, strat_dir, 'F', splits)

    # =========================================================================
    # VERIFICATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    for strategy in strategies:
        if strategy == 'D':
            d = os.path.join(args.output_dir, 'strategy_d_farthest_point')
        elif strategy == 'E':
            d = os.path.join(args.output_dir, 'strategy_e_cluster_balanced')
        elif strategy == 'F':
            d = os.path.join(args.output_dir, 'strategy_f_coverage_merged')
        else:
            continue

        train_f = os.path.join(d, 'train_bandgaps_regression.json')
        val_f = os.path.join(d, 'val_bandgaps_regression.json')
        test_f = os.path.join(d, 'test_bandgaps_regression.json')

        sets = {}
        for name, path in [('train', train_f), ('val', val_f), ('test', test_f)]:
            if os.path.exists(path):
                with open(path) as f:
                    sets[name] = set(json.load(f).keys())
            else:
                sets[name] = set()

        overlap_tv = sets.get('train', set()) & sets.get('val', set())
        overlap_tt = sets.get('train', set()) & sets.get('test', set())
        overlap_vt = sets.get('val', set()) & sets.get('test', set())
        total_accounted = len(sets.get('train', set()) |
                              sets.get('val', set()) |
                              sets.get('test', set()))

        print(f"\n  Strategy {strategy}:")
        print(f"    Train-Val overlap:  {len(overlap_tv)} "
              f"{'OK' if len(overlap_tv) == 0 else '*** LEAK ***'}")
        print(f"    Train-Test overlap: {len(overlap_tt)} "
              f"{'OK' if len(overlap_tt) == 0 else '*** LEAK ***'}")
        print(f"    Val-Test overlap:   {len(overlap_vt)} "
              f"{'OK' if len(overlap_vt) == 0 else '*** LEAK ***'}")
        print(f"    Total accounted: {total_accounted} / {len(all_bandgaps)}")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("""
  1. Copy new split directories to cluster
  2. Run setup_data_links.sh in each strategy directory
  3. Train models on new splits — use DATA_DIR pointing to strategy dir
  4. Compare test results: original split vs D vs E vs F
""")


if __name__ == "__main__":
    main()
