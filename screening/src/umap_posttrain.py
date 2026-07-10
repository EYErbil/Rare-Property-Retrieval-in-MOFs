#!/usr/bin/env python3
"""
UMAP on post-training representations (learned embeddings after training).
==========================================================================

After training a regressor, run this to project the model's internal
representations (pooled/CLS features before the regression head) with UMAP
and produce the same diagnostic figures as the pretrained-embedding UMAP.

Outputs (into --output_dir):
  - umap_posttrain_bandgap.png
  - umap_posttrain_diagnosis.png
  - umap_posttrain_report.txt
  - umap_posttrain_analysis_summary.json
  - umap_coords.npz
  - embeddings_posttrain.npz (optional, with --save_embeddings)

Usage:
  python umap_posttrain.py --checkpoint experiments/exp364_fulltune/best_es-epoch=XX.ckpt \\
      --data_dir data/splits/strategy_d_farthest_point --output_dir ./umap_posttrain_exp364

  python umap_posttrain.py --experiment exp364_fulltune \\
      --data_dir data/splits/strategy_d_farthest_point --output_dir ./umap_posttrain_exp364
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_regressor import MOFRegressor
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config

# Reuse UMAP and plotting from Split D script
from umap_analysis_split_d import (
    load_split_labels,
    cosine_sim_matrix,
    compute_umap,
    plot_bandgap_umap,
    plot_diagnosis_umap,
    nn_diagnosis,
    generate_report,
)

THRESHOLD = 1.0


def find_best_checkpoint(exp_dir):
    """Spearman-best checkpoint (same as reinfer_nn)."""
    results_path = os.path.join(exp_dir, "final_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        best = results.get("checkpoints", {}).get("best", "")
        if best:
            if not os.path.isabs(best):
                best = os.path.join(exp_dir, best)
            if os.path.exists(best):
                return best
    for pattern in ["best_es-*.ckpt", "best_*.ckpt"]:
        matches = glob.glob(os.path.join(exp_dir, pattern))
        if matches:
            return sorted(matches)[-1]
    last = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    return None


def build_config(data_dir):
    """Config for loading model and data (same as reinfer_nn)."""
    config = default_config_fn()
    config = json.loads(json.dumps(config))
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config["data_dir"] = data_dir
    config["downstream"] = "bandgaps_regression"
    config["threshold"] = THRESHOLD
    config["pooling_type"] = "mean"
    config["dropout"] = 0.0
    config["per_gpu_batchsize"] = 16
    config["batch_size"] = 32
    config["load_path"] = "pmtransformer"
    config = get_valid_config(config)
    return config


def extract_embeddings(model, loaders, device, split_names):
    """Run forward_features on all batches; return cif_ids, embeddings, bandgaps, splits."""
    model.eval()
    all_cif_ids = []
    all_embeddings = []
    all_bandgaps = []
    all_splits = []

    with torch.no_grad():
        for loader, split_name in zip(loaders, split_names):
            if loader is None:
                continue
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                pooled, output = model.forward_features(batch)
                targets = batch["target"]
                cif_ids = output.get("cif_id", output.get("name", None))

                all_embeddings.append(pooled.cpu().numpy())
                all_bandgaps.extend([float(t) for t in targets])
                all_splits.extend([split_name] * len(targets))
                if cif_ids:
                    if isinstance(cif_ids, (list, tuple)):
                        all_cif_ids.extend(cif_ids)
                    else:
                        all_cif_ids.append(cif_ids)

    cif_ids = all_cif_ids
    embeddings = np.concatenate(all_embeddings, axis=0)
    bandgaps = np.array(all_bandgaps)
    splits = all_splits
    return cif_ids, embeddings, bandgaps, splits


def main():
    parser = argparse.ArgumentParser(
        description='UMAP on post-training (learned) representations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .ckpt file')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name (e.g. exp364_fulltune); '
                             'checkpoint = experiments/<name>/best_es-*.ckpt')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'new_splits', 'strategy_d_farthest_point'),
                        help='Split directory (train/val/test)')
    parser.add_argument('--output_dir', type=str, default='./umap_posttrain',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive (eV)')
    parser.add_argument('--n_neighbors', type=int, default=30,
                        help='UMAP n_neighbors')
    parser.add_argument('--min_dist', type=float, default=0.3,
                        help='UMAP min_dist')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save embeddings_posttrain.npz')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(base_dir, args.data_dir)

    ckpt_path = args.checkpoint
    if ckpt_path is None and args.experiment:
        exp_dir = os.path.join(base_dir, 'experiments', args.experiment)
        ckpt_path = find_best_checkpoint(exp_dir)
        if ckpt_path is None:
            print(f"ERROR: No checkpoint found for experiment {args.experiment}")
            sys.exit(1)
    if ckpt_path is None:
        print("ERROR: Provide --checkpoint or --experiment")
        sys.exit(1)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("  UMAP ON POST-TRAINING REPRESENTATIONS")
    print("=" * 80)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output:     {args.output_dir}")

    config = build_config(args.data_dir)
    collate = lambda x: Dataset.collate(x, config["img_size"])
    batch_size = config.get("per_gpu_batchsize", 16)

    train_ds = Dataset(
        args.data_dir, split="train",
        downstream=config["downstream"],
        nbr_fea_len=config["nbr_fea_len"],
        draw_false_grid=False,
    )
    val_ds = Dataset(
        args.data_dir, split="val",
        downstream=config["downstream"],
        nbr_fea_len=config["nbr_fea_len"],
        draw_false_grid=False,
    )
    try:
        test_ds = Dataset(
            args.data_dir, split="test",
            downstream=config["downstream"],
            nbr_fea_len=config["nbr_fea_len"],
            draw_false_grid=False,
        )
    except Exception:
        test_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate) if test_ds else None

    loaders = [train_loader, val_loader, test_loader]
    split_names = ['train', 'val', 'test']

    print(f"\n  Loading model from checkpoint...")
    model = MOFRegressor.load_from_checkpoint(ckpt_path, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"  Extracting embeddings (train + val + test)...")
    cif_ids, embeddings, bandgaps, splits = extract_embeddings(
        model, loaders, device, split_names)
    print(f"  Total: {len(cif_ids)} MOFs, embedding dim={embeddings.shape[1]}")

    del model
    torch.cuda.empty_cache()

    if args.save_embeddings:
        npz_path = os.path.join(args.output_dir, 'embeddings_posttrain.npz')
        np.savez_compressed(
            npz_path,
            cif_ids=np.array(cif_ids),
            embeddings=embeddings,
            bandgaps=bandgaps,
            splits=np.array(splits),
        )
        print(f"  Saved: {npz_path}")

    print(f"\n--- UMAP projection ---")
    coords = compute_umap(embeddings, args.n_neighbors, args.min_dist)
    np.savez_compressed(
        os.path.join(args.output_dir, 'umap_coords.npz'),
        coords=coords, cif_ids=np.array(cif_ids),
    )

    print(f"\n--- Nearest-neighbor diagnosis ---")
    nn_results = nn_diagnosis(cif_ids, embeddings, bandgaps, splits, args.threshold)

    print(f"\n--- Plots ---")
    plot_bandgap_umap(
        coords, bandgaps, splits, args.threshold,
        os.path.join(args.output_dir, 'umap_posttrain_bandgap.png'),
        split_label="Post-training (learned representations)")
    plot_diagnosis_umap(
        coords, bandgaps, splits, cif_ids, nn_results, args.threshold,
        os.path.join(args.output_dir, 'umap_posttrain_diagnosis.png'),
        split_label="Post-training")

    print(f"\n--- Report ---")
    generate_report(
        nn_results,
        os.path.join(args.output_dir, 'umap_posttrain_report.txt'),
        split_name="Post-training")

    n_train = sum(1 for s in splits if s == 'train')
    n_test = sum(1 for s in splits if s == 'test')
    n_pos = sum(1 for bg in bandgaps if bg < args.threshold)
    n_train_pos = sum(1 for i, s in enumerate(splits)
                      if s == 'train' and bandgaps[i] < args.threshold)
    n_test_pos = sum(1 for i, s in enumerate(splits)
                     if s == 'test' and bandgaps[i] < args.threshold)

    summary = {
        'split': 'Post-training UMAP (learned representations)',
        'checkpoint': ckpt_path,
        'n_total': len(cif_ids),
        'n_train': n_train,
        'n_test': n_test,
        'n_positive': int(n_pos),
        'n_train_positive': n_train_pos,
        'n_test_positive': n_test_pos,
        'test_positives': [],
    }
    for r in sorted(nn_results, key=lambda x: x['pos_neg_gap']):
        summary['test_positives'].append({
            'cif_id': r['cif_id'],
            'bandgap': float(r['bandgap']),
            'nearest_train_pos_sim': r['nearest_pos_sim'],
            'nearest_train_neg_sim': r['nearest_neg_sim'],
            'gap': r['pos_neg_gap'],
            'rank_of_nearest_pos': r['rank_of_nearest_pos'],
        })
    with open(os.path.join(args.output_dir, 'umap_posttrain_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"  DONE. Outputs in: {args.output_dir}")
    print(f"{'=' * 80}")
    print(f"    umap_posttrain_bandgap.png")
    print(f"    umap_posttrain_diagnosis.png")
    print(f"    umap_posttrain_report.txt")
    print(f"    umap_posttrain_analysis_summary.json")
    print(f"    umap_coords.npz")


if __name__ == "__main__":
    main()
