#!/usr/bin/env python3
"""
MOF Data Re-Splitting Script
==============================

Creates alternative data splits from the full QMOF dataset to address the
fundamental problem: only 58 training positives and 7 validation positives
make the current split unreliable.

Three strategies:
  A) bigger_train — Steal positives from val to boost training:
       train: 62 positives, val: 3 positives, test: 9 positives (untouched)

  B) merged_trainval — Merge train+val, use test only for final evaluation:
       train: 65 positives (all from old train+val), test: 9 positives (untouched)
       No dedicated val set — use train loss / Spearman ρ on train for early stopping,
       or use internal k-fold CV.

  C) kfold — 5-fold cross-validation on train+val combined:
       Each fold: ~52 train positives, ~13 val positives
       Test: 9 positives (untouched, never used for model selection)

For each strategy, generates ALL label file variants:
  - {split}_bandgaps_regression.json     (float bandgap values)
  - {split}_bandgaps.json                (binary 0/1, threshold=1.0 eV)
  - {split}_bandgaps_ordinal.json        (ordinal bins, 0.5 eV width)
  - {split}_bandgaps_regression_multiclass.json  (multiclass ordinal)
  - train_bandgaps_regression_weights.json (sample weights)

Usage:
  python resplit_data.py --splits_dir ./splits --output_dir ./new_splits
  python resplit_data.py --splits_dir ./splits --output_dir ./new_splits --strategy all
  python resplit_data.py --splits_dir ./splits --output_dir ./new_splits --strategy B
"""

import os
import sys
import json
import argparse
import math
import numpy as np
from collections import defaultdict, Counter


# =============================================================================
# LABEL GENERATION UTILITIES
# =============================================================================

def bandgap_to_binary(bandgap, threshold=1.0):
    """Convert bandgap to binary label: 1 if < threshold, 0 otherwise."""
    return 1 if bandgap < threshold else 0


def bandgap_to_ordinal(bandgap, bin_width=0.5, max_class=13):
    """
    Convert bandgap to ordinal class:
      Class 0: bandgap < 1.0 eV  (the discovery-positive class)
      Class 1: 1.0 <= bg < 1.5
      Class 2: 1.5 <= bg < 2.0
      ...
      Class 13: bg >= 7.0
    """
    if bandgap < 1.0:
        return 0
    # For bandgap >= 1.0, compute bin
    bin_idx = int((bandgap - 1.0) / bin_width) + 1
    return min(bin_idx, max_class)


def bandgap_to_multiclass(bandgap, bin_width=0.5, max_class=14):
    """
    Same as ordinal but with an extra outlier class.
    Class 14: extreme outliers (bg > 7.5 or whatever threshold)
    """
    if bandgap < 1.0:
        return 0
    if bandgap > 7.5:
        return max_class
    bin_idx = int((bandgap - 1.0) / bin_width) + 1
    return min(bin_idx, max_class - 1)


def compute_sample_weights(bandgaps, threshold=1.0):
    """
    Compute sample weights for regression with inverse-frequency weighting.
    Positives (bg < threshold) get higher weight proportional to rarity.
    Negatives get weight 1.0.

    The weighting scheme gives each positive a weight based on how extreme
    its bandgap is (lower = rarer = higher weight), similar to the original
    weights which range from ~18 to ~23 for positives.
    """
    bgs = list(bandgaps.values())
    cids = list(bandgaps.keys())

    n_total = len(bgs)
    n_pos = sum(1 for bg in bgs if bg < threshold)
    n_neg = n_total - n_pos

    if n_pos == 0:
        return {cid: 1.0 for cid in cids}

    # Base weight for positives: n_neg / n_pos (class balance ratio)
    base_pos_weight = n_neg / n_pos

    weights = {}
    for cid in cids:
        bg = bandgaps[cid]
        if bg < threshold:
            # Scale weight: lower bandgap → slightly higher weight
            # Normalized so that mean positive weight ≈ base_pos_weight
            # Use log-scale to avoid extreme weights
            bg_factor = 1.0 + 0.3 * (1.0 - bg / threshold)
            weights[cid] = base_pos_weight * bg_factor
        else:
            weights[cid] = 1.0

    return weights


def generate_all_labels(split_bandgaps, split_name, output_dir):
    """
    Given {cif_id: bandgap} for a split, generate all label file variants
    and save them to output_dir.
    """
    # 1. Regression labels (raw bandgap)
    regression = {cid: bg for cid, bg in split_bandgaps.items()}
    save_json(regression, os.path.join(output_dir, f'{split_name}_bandgaps_regression.json'))

    # 2. Binary labels
    binary = {cid: bandgap_to_binary(bg) for cid, bg in split_bandgaps.items()}
    save_json(binary, os.path.join(output_dir, f'{split_name}_bandgaps.json'))

    # 3. Ordinal labels
    ordinal = {cid: bandgap_to_ordinal(bg) for cid, bg in split_bandgaps.items()}
    save_json(ordinal, os.path.join(output_dir, f'{split_name}_bandgaps_ordinal.json'))

    # 4. Multiclass labels
    multiclass = {cid: bandgap_to_multiclass(bg) for cid, bg in split_bandgaps.items()}
    save_json(multiclass, os.path.join(output_dir, f'{split_name}_bandgaps_regression_multiclass.json'))

    # 5. Sample weights (only for train)
    if split_name == 'train':
        weights = compute_sample_weights(split_bandgaps)
        save_json(weights, os.path.join(output_dir, f'{split_name}_bandgaps_regression_weights.json'))

    return {
        'n_total': len(split_bandgaps),
        'n_pos': sum(1 for bg in split_bandgaps.values() if bg < 1.0),
        'n_neg': sum(1 for bg in split_bandgaps.values() if bg >= 1.0),
    }


def save_json(data, path):
    """Save dict as JSON with 2-space indent."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# LOADING EXISTING SPLITS
# =============================================================================

def load_all_bandgaps(splits_dir):
    """
    Load all three split files and merge into a single dict.
    Returns: {cif_id: bandgap}, split_assignment: {cif_id: 'train'/'val'/'test'}
    """
    all_bandgaps = {}
    split_assignment = {}

    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(splits_dir, f'{split_name}_bandgaps_regression.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: {json_path} not found, skipping {split_name}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        for cid, bg in data.items():
            all_bandgaps[cid] = float(bg)
            split_assignment[cid] = split_name

        n_pos = sum(1 for bg in data.values() if bg < 1.0)
        print(f"  Loaded {split_name}: {len(data)} total, {n_pos} positives")

    return all_bandgaps, split_assignment


# =============================================================================
# STRATEGY A: BIGGER TRAIN
# =============================================================================

def strategy_a_bigger_train(all_bandgaps, split_assignment, seed=42):
    """
    Move some val positives to train to increase training positives.
    Keep test untouched. Move 4 of 7 val positives to train.

    Result: train ~62 pos, val ~3 pos, test 9 pos (unchanged)
    """
    rng = np.random.default_rng(seed)

    # Separate by original split
    train_cids = [c for c, s in split_assignment.items() if s == 'train']
    val_cids = [c for c, s in split_assignment.items() if s == 'val']
    test_cids = [c for c, s in split_assignment.items() if s == 'test']

    # Identify val positives
    val_pos = [c for c in val_cids if all_bandgaps[c] < 1.0]
    val_neg = [c for c in val_cids if all_bandgaps[c] >= 1.0]

    # Move 4 of 7 val positives to train
    n_move = min(4, len(val_pos))
    rng.shuffle(val_pos)
    moved_to_train = val_pos[:n_move]
    remaining_val_pos = val_pos[n_move:]

    new_train = train_cids + moved_to_train
    new_val = remaining_val_pos + val_neg

    return {
        'train': {c: all_bandgaps[c] for c in new_train},
        'val': {c: all_bandgaps[c] for c in new_val},
        'test': {c: all_bandgaps[c] for c in test_cids},
    }


# =============================================================================
# STRATEGY B: MERGED TRAIN+VAL
# =============================================================================

def strategy_b_merged(all_bandgaps, split_assignment):
    """
    Merge train + val into one big training set.
    Test stays untouched. No dedicated val set.
    Use k-fold CV or training metrics for model selection.

    Result: train 65 pos (1260 total), test 9 pos (9550 total)
    """
    train_cids = [c for c, s in split_assignment.items() if s == 'train']
    val_cids = [c for c, s in split_assignment.items() if s == 'val']
    test_cids = [c for c, s in split_assignment.items() if s == 'test']

    new_train = train_cids + val_cids

    # For experiments that need a val set for early stopping,
    # create a small pseudo-val set by holding out a stratified portion
    # Actually: we'll provide a "val" file that is identical to train
    # so Lightning doesn't crash, but early stopping should use train metrics.
    # Better approach: create a small stratified val holdout (10% of new train)
    # This way we still get val metrics for early stopping.

    return {
        'train': {c: all_bandgaps[c] for c in new_train},
        'test': {c: all_bandgaps[c] for c in test_cids},
    }


def strategy_b_with_val(all_bandgaps, split_assignment, val_frac=0.1, seed=42):
    """
    Merge train+val, then hold out a small stratified fraction as new val.
    This gives us early stopping capability while maximizing training data.

    Result: train ~58 pos, val ~7 pos (stratified), test 9 pos (unchanged)
    """
    rng = np.random.default_rng(seed)

    train_cids = [c for c, s in split_assignment.items() if s == 'train']
    val_cids = [c for c, s in split_assignment.items() if s == 'val']
    test_cids = [c for c, s in split_assignment.items() if s == 'test']

    # Pool all train+val
    pool = train_cids + val_cids
    pool_pos = [c for c in pool if all_bandgaps[c] < 1.0]
    pool_neg = [c for c in pool if all_bandgaps[c] >= 1.0]

    # Stratified split: ~10% of each class to val
    rng.shuffle(pool_pos)
    rng.shuffle(pool_neg)

    n_val_pos = max(5, int(len(pool_pos) * val_frac))  # at least 5 positives in val
    n_val_neg = int(len(pool_neg) * val_frac)

    new_val_cids = list(pool_pos[:n_val_pos]) + list(pool_neg[:n_val_neg])
    new_train_cids = list(pool_pos[n_val_pos:]) + list(pool_neg[n_val_neg:])

    return {
        'train': {c: all_bandgaps[c] for c in new_train_cids},
        'val': {c: all_bandgaps[c] for c in new_val_cids},
        'test': {c: all_bandgaps[c] for c in test_cids},
    }


# =============================================================================
# STRATEGY C: K-FOLD CROSS-VALIDATION
# =============================================================================

def strategy_c_kfold(all_bandgaps, split_assignment, n_folds=5, seed=42):
    """
    K-fold CV on train+val pool. Test is never touched.

    For each fold, generates complete split files.
    Returns a list of fold dicts.
    """
    rng = np.random.default_rng(seed)

    train_cids = [c for c, s in split_assignment.items() if s == 'train']
    val_cids = [c for c, s in split_assignment.items() if s == 'val']
    test_cids = [c for c, s in split_assignment.items() if s == 'test']

    # Pool all train+val
    pool = train_cids + val_cids
    pool_pos = [c for c in pool if all_bandgaps[c] < 1.0]
    pool_neg = [c for c in pool if all_bandgaps[c] >= 1.0]

    # Shuffle for randomization
    rng.shuffle(pool_pos)
    rng.shuffle(pool_neg)

    # Create stratified folds
    folds = []
    for fold_i in range(n_folds):
        # Get val indices for this fold
        pos_start = fold_i * len(pool_pos) // n_folds
        pos_end = (fold_i + 1) * len(pool_pos) // n_folds
        neg_start = fold_i * len(pool_neg) // n_folds
        neg_end = (fold_i + 1) * len(pool_neg) // n_folds

        val_pos = pool_pos[pos_start:pos_end]
        val_neg = pool_neg[neg_start:neg_end]
        train_pos = pool_pos[:pos_start] + pool_pos[pos_end:]
        train_neg = pool_neg[:neg_start] + pool_neg[neg_end:]

        fold = {
            'train': {c: all_bandgaps[c] for c in train_pos + train_neg},
            'val': {c: all_bandgaps[c] for c in val_pos + val_neg},
            'test': {c: all_bandgaps[c] for c in test_cids},
        }
        folds.append(fold)

    return folds


# =============================================================================
# CREATE SYMLINKS FOR DATA FILES
# =============================================================================

def create_data_symlink_script(data_dir, output_dir, strategy_name, split_data=None):
    """
    Create a shell script that symlinks the actual data files (.graphdata,
    .griddata16, .grid) into split-specific subdirectories.

    MOFTransformer Dataset expects: data_dir/{split}/{cif_id}.{ext}
    The original data has:          data_dir/{original_split}/{cif_id}.{ext}

    Since re-splitting moves MOFs between train/val/test, we need to create
    new {split}/ subdirectories with symlinks pointing to whichever original
    split subdirectory actually contains each MOF's data files.

    Args:
        data_dir: Original data directory (with train/, val/, test/ subdirs)
        output_dir: New split directory to set up
        strategy_name: Label for the script
        split_data: dict of {split_name: {cif_id: bandgap}} — if provided,
                    generates a targeted script that only links needed files.
    """
    script = f"""#!/bin/bash
# Symlink data files for strategy {strategy_name}
# Run this ONCE on the cluster to set up the data directory
#
# MOFTransformer expects: data_dir/{{split}}/{{cif_id}}.{{ext}}
# This script creates train/, val/, test/ subdirs with symlinks
# pointing to the ORIGINAL split directories where each MOF's
# data files actually live.
#
# Usage: bash setup_data_links.sh

DATA_SRC="{data_dir}"
DEST="{output_dir}"

# Original split subdirectories to search for data files
ORIG_SPLITS="train val test"

link_file() {{
    # Link a single CIF's data files from original split dirs into new split dir
    local cif_id="$1"
    local new_split="$2"

    mkdir -p "$DEST/$new_split"

    for ext in grid griddata16 graphdata; do
        # Search all original split dirs for this file
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

    # If we have the split assignments, generate targeted link commands
    if split_data:
        for split_name, cids_dict in split_data.items():
            cid_list = list(cids_dict.keys())
            script += f'echo "Linking {len(cid_list)} files for {split_name}..."\n'
            script += f'mkdir -p "$DEST/{split_name}"\n'
            
            # Write CIF IDs to a temp approach — but for shell script,
            # just loop directly (may be slow for 9550 test but works)
            for cid in cid_list:
                script += f'link_file "{cid}" "{split_name}"\n'
            script += f'echo "  {split_name}: done ({len(cid_list)} MOFs)"\n\n'
    else:
        # Fallback: link everything from each original split into matching new split
        script += """
# Fallback: link all files from original splits
for split in train val test; do
    mkdir -p "$DEST/$split"
    for ext in grid griddata16 graphdata; do
        echo "Linking $split/*.$ext ..."
        for f in "$DATA_SRC/$split"/*.$ext; do
            [ -f "$f" ] || continue
            fname=$(basename "$f")
            dst="$DEST/$split/$fname"
            [ -e "$dst" ] || ln -s "$f" "$dst"
        done
    done
done
"""

    script += """
echo ""
echo "=========================================="
echo "Done! Data symlinks created."
echo "=========================================="
# Verify
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
    parser = argparse.ArgumentParser(description='MOF Data Re-Splitting')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='Directory containing original split JSON files')
    parser.add_argument('--output_dir', type=str, default='./new_splits',
                        help='Output directory for new splits')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Original data directory (for symlink script). '
                             'If not set, uses splits_dir.')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['A', 'B', 'C', 'all'],
                        help='Which strategy to generate (A/B/C/all)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for strategy C')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of train+val to hold out as val in strategy B')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive class (eV)')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = args.splits_dir

    print("=" * 70)
    print("MOF DATA RE-SPLITTING")
    print("=" * 70)

    # Load all existing data
    print("\nLoading existing splits...")
    all_bandgaps, split_assignment = load_all_bandgaps(args.splits_dir)
    total_pos = sum(1 for bg in all_bandgaps.values() if bg < args.threshold)
    total_neg = sum(1 for bg in all_bandgaps.values() if bg >= args.threshold)
    print(f"\n  Total MOFs: {len(all_bandgaps)}")
    print(f"  Total positives (< {args.threshold} eV): {total_pos}")
    print(f"  Total negatives (>= {args.threshold} eV): {total_neg}")

    strategies_to_run = ['A', 'B', 'C'] if args.strategy == 'all' else [args.strategy]

    for strategy in strategies_to_run:
        print(f"\n{'='*70}")
        print(f"STRATEGY {strategy}")
        print(f"{'='*70}")

        if strategy == 'A':
            strat_dir = os.path.join(args.output_dir, 'strategy_a_bigger_train')
            os.makedirs(strat_dir, exist_ok=True)

            splits = strategy_a_bigger_train(all_bandgaps, split_assignment, args.seed)

            for split_name, split_data in splits.items():
                stats = generate_all_labels(split_data, split_name, strat_dir)
                print(f"  {split_name}: {stats['n_total']} total, "
                      f"{stats['n_pos']} pos, {stats['n_neg']} neg")

            create_data_symlink_script(args.data_dir, strat_dir, 'A', splits)

        elif strategy == 'B':
            strat_dir = os.path.join(args.output_dir, 'strategy_b_merged')
            os.makedirs(strat_dir, exist_ok=True)

            splits = strategy_b_with_val(all_bandgaps, split_assignment,
                                          args.val_frac, args.seed)

            for split_name, split_data in splits.items():
                stats = generate_all_labels(split_data, split_name, strat_dir)
                print(f"  {split_name}: {stats['n_total']} total, "
                      f"{stats['n_pos']} pos, {stats['n_neg']} neg")

            create_data_symlink_script(args.data_dir, strat_dir, 'B', splits)

            # Also generate a "no-val" variant where we train on everything
            strat_dir_noval = os.path.join(args.output_dir, 'strategy_b_no_val')
            os.makedirs(strat_dir_noval, exist_ok=True)

            splits_noval = strategy_b_merged(all_bandgaps, split_assignment)

            # For the train-only variant, we still need a "val" file for PyTorch Lightning
            # Use a copy of train as val (monitor train metrics for stopping)
            splits_noval['val'] = splits_noval['train'].copy()

            for split_name, split_data in splits_noval.items():
                stats = generate_all_labels(split_data, split_name, strat_dir_noval)
                print(f"  [no-val] {split_name}: {stats['n_total']} total, "
                      f"{stats['n_pos']} pos, {stats['n_neg']} neg")

            create_data_symlink_script(args.data_dir, strat_dir_noval, 'B_no_val', splits_noval)

        elif strategy == 'C':
            strat_dir_base = os.path.join(args.output_dir, 'strategy_c_kfold')
            os.makedirs(strat_dir_base, exist_ok=True)

            folds = strategy_c_kfold(all_bandgaps, split_assignment,
                                      args.n_folds, args.seed)

            for fold_i, fold_data in enumerate(folds, 1):
                fold_dir = os.path.join(strat_dir_base, f'fold_{fold_i}')
                os.makedirs(fold_dir, exist_ok=True)

                print(f"\n  Fold {fold_i}:")
                for split_name, split_data in fold_data.items():
                    stats = generate_all_labels(split_data, split_name, fold_dir)
                    print(f"    {split_name}: {stats['n_total']} total, "
                          f"{stats['n_pos']} pos, {stats['n_neg']} neg")

                create_data_symlink_script(args.data_dir, fold_dir, f'C_fold_{fold_i}', fold_data)

    # =========================================================================
    # VERIFICATION: ensure no data leakage
    # =========================================================================
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    # For each strategy, verify:
    # 1. Train and val don't overlap
    # 2. Test is identical to original
    # 3. All CIF IDs accounted for

    original_test = set(c for c, s in split_assignment.items() if s == 'test')

    for strategy in strategies_to_run:
        if strategy == 'A':
            d = os.path.join(args.output_dir, 'strategy_a_bigger_train')
        elif strategy == 'B':
            d = os.path.join(args.output_dir, 'strategy_b_merged')
        elif strategy == 'C':
            d = os.path.join(args.output_dir, 'strategy_c_kfold', 'fold_1')
        else:
            continue

        # Load generated splits
        train_f = os.path.join(d, 'train_bandgaps_regression.json')
        val_f = os.path.join(d, 'val_bandgaps_regression.json')
        test_f = os.path.join(d, 'test_bandgaps_regression.json')

        if os.path.exists(train_f):
            with open(train_f) as f: train_ids = set(json.load(f).keys())
        else:
            train_ids = set()

        if os.path.exists(val_f):
            with open(val_f) as f: val_ids = set(json.load(f).keys())
        else:
            val_ids = set()

        if os.path.exists(test_f):
            with open(test_f) as f: test_ids = set(json.load(f).keys())
        else:
            test_ids = set()

        overlap_tv = train_ids & val_ids
        overlap_tt = train_ids & test_ids
        overlap_vt = val_ids & test_ids

        test_match = test_ids == original_test

        print(f"\n  Strategy {strategy}:")
        print(f"    Train-Val overlap:  {len(overlap_tv)} "
              f"{'OK' if len(overlap_tv) == 0 else '*** LEAK ***'}")
        print(f"    Train-Test overlap: {len(overlap_tt)} "
              f"{'OK' if len(overlap_tt) == 0 else '*** LEAK ***'}")
        print(f"    Val-Test overlap:   {len(overlap_vt)} "
              f"{'OK' if len(overlap_vt) == 0 else '*** LEAK ***'}")
        print(f"    Test matches original: {'OK' if test_match else '*** MISMATCH ***'}")
        print(f"    Total accounted: {len(train_ids | val_ids | test_ids)} / {len(all_bandgaps)}")

    # Print summary of what to do next
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("""
  1. Copy the generated split directories to the cluster:
       scp -r new_splits/ cluster:/path/to/splits/

  2. Run the setup_data_links.sh script in each split directory
     to create symlinks to the actual .graphdata/.griddata16 files:
       cd /path/to/new_splits/strategy_b_merged/
       bash setup_data_links.sh

  3. Update experiment run.py files to point DATA_DIR to the new split:
       DATA_DIR = "/path/to/new_splits/strategy_b_merged"

  4. For Strategy C (k-fold), use the run_kfold() function:
       from train_regressor import run_kfold
       run_kfold(kfold_dir="/path/to/new_splits/strategy_c_kfold", ...)
""")


if __name__ == "__main__":
    main()
