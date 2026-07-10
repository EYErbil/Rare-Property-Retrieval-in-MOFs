#!/usr/bin/env python3
"""
Extract unlabeled embeddings using RAW, non-finetuned MOFTransformer/pmtransformer.

Same logic as analyze_embeddings.py (create_model + extract_embeddings) — self-contained,
no import from analyze_embeddings (which may not exist on cluster).

All outputs go into embedding_analysis (JUST LIKE analyze_embeddings.py):
  - unlabeled_embeddings.npz   (cif_ids, embeddings, bandgaps, splits=test)
  - embedding_umap_unlabeled.png       (UMAP plot — compare with embedding_umap_pretrained.png)
  - similarity_heatmap_unlabeled.png   (similarity heatmap — compare with similarity_heatmap_pretrained.png)

Usage:
  python data_preparation/extract_unlabeled_embeddings.py \\
    --data_dir data_preparation/Processed-data \\
    --output_dir embedding_analysis
"""

import os
import json
import shutil
import argparse
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moftransformer.modules.module import Module
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config


def create_model(config, checkpoint_path=None):
    """Same as analyze_embeddings.py — pretrained MOFTransformer when checkpoint_path is None."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        raise NotImplementedError("Fine-tuned checkpoint not used for unlabeled set (pretrained only)")
    model = Module(config)
    model.eval()
    return model


def extract_embeddings(model, data_dir, splits_dir, downstream, config,
                       batch_size=8, num_workers=2, device='cuda'):
    """
    Same as analyze_embeddings.py — extract 768-dim CLS embeddings for train/val/test.
    Returns: cif_ids, embeddings, bandgaps, split_labels
    """
    model = model.to(device)
    model.eval()

    all_cif_ids = []
    all_embeddings = []
    all_bandgaps = []
    all_splits = []

    # Load bandgap labels from splits_dir (same as analyze_embeddings)
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
        n_skipped = 0
        loader_iter = iter(loader)

        with torch.no_grad():
            for idx in range(len(ds)):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                except RuntimeError as e:
                    if "shape" in str(e) or "invalid" in str(e):
                        try:
                            sample = ds[idx]
                            cif_id = sample.get("cif_id", sample.get("name", sample.get("cif_id", f"index_{idx}")))
                            if isinstance(cif_id, (list, tuple)):
                                cif_id = cif_id[0] if cif_id else f"index_{idx}"
                        except Exception:
                            cif_id = f"index_{idx}"
                        print(f"  SKIPPED (grid shape mismatch): {cif_id}")
                        n_skipped += 1
                        continue
                    raise

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                output = model.infer(batch)
                cls_feats = output["cls_feats"]
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

        if n_skipped > 0:
            print(f"    {split_name}: extracted {n_done}, skipped {n_skipped} (grid shape mismatch)")
        else:
            print(f"    {split_name}: extracted {n_done} embeddings")

    if not all_embeddings:
        raise RuntimeError(
            "No embeddings extracted. All samples failed (grid shape mismatch). "
            "Check SKIPPED messages above and fix/remove problematic MOF files."
        )
    embeddings = np.stack(all_embeddings, axis=0)
    bandgaps = np.array(all_bandgaps)
    return all_cif_ids, embeddings, bandgaps, all_splits


def cosine_similarity_matrix(A, B):
    """Compute cosine similarity between rows of A and rows of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def plot_umap(embeddings, bandgaps, splits, threshold, output_path):
    """2D UMAP colored by bandgap — same as analyze_embeddings.py."""
    try:
        from umap import UMAP
    except ImportError:
        print("  UMAP not installed (pip install umap-learn), skipping UMAP plot")
        return

    print("  Computing UMAP projection...")
    n_neighbors = min(30, max(2, len(embeddings) - 1))
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.3, metric='cosine', random_state=42)
    coords = reducer.fit_transform(embeddings)

    pos_mask = bandgaps < threshold
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    bg_clipped = np.clip(bandgaps, 0, 5)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=bg_clipped, cmap='RdYlBu',
                         s=3, alpha=0.4, rasterized=True)
    tp_mask = test_mask & pos_mask
    ax.scatter(coords[tp_mask, 0], coords[tp_mask, 1], c='red', s=100,
               marker='*', edgecolors='black', linewidths=1, zorder=5, label='Test positive')
    trp_mask = train_mask & pos_mask
    ax.scatter(coords[trp_mask, 0], coords[trp_mask, 1], c='blue', s=60,
               marker='^', edgecolors='black', linewidths=0.5, zorder=4, label='Train positive')
    ax.set_title('UMAP — colored by bandgap (eV)', fontsize=14)
    ax.legend(fontsize=10)
    plt.colorbar(scatter, ax=ax, label='Bandgap (eV)')

    ax = axes[1]
    colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:gray'}
    for split_name in ['test', 'train', 'val']:
        mask = np.array([s == split_name for s in splits])
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=colors.get(split_name, 'tab:gray'),
                       s=3, alpha=0.3, label=f'{split_name} ({mask.sum()})', rasterized=True)
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
    """Pairwise cosine similarity heatmap among positives — same as analyze_embeddings.py."""
    pos_mask = bandgaps < threshold
    pos_idx = np.where(pos_mask)[0]

    if len(pos_idx) < 2:
        print("  Not enough positives for similarity heatmap (need >= 2)")
        return

    pos_embs = embeddings[pos_idx]
    sim_mat = cosine_similarity_matrix(pos_embs, pos_embs)

    pos_cids = [cif_ids[i] for i in pos_idx]
    pos_splits = [splits[i] for i in pos_idx]
    pos_bgs = bandgaps[pos_idx]

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
    ax.set_title(f'Pairwise Cosine Similarity — All {len(pos_idx)} Positives (unlabeled)\n'
                 f'(tr=train, va=val, te=test | CIF_ID | bandgap eV)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

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


def ensure_test_json(data_dir):
    """If test_bandgaps_regression.json is missing, copy from inference_bandgaps_regression.json."""
    test_json = os.path.join(data_dir, "test_bandgaps_regression.json")
    if os.path.exists(test_json):
        return
    inf_json = os.path.join(data_dir, "inference_bandgaps_regression.json")
    if not os.path.exists(inf_json):
        raise FileNotFoundError(
            f"Neither {test_json} nor {inf_json} found. Run build_inference_targets.py and collect_inference_structures.sh first."
        )
    shutil.copy2(inf_json, test_json)
    print(f"  Created {test_json} from inference_bandgaps_regression.json")


def main():
    parser = argparse.ArgumentParser(
        description="Extract unlabeled embeddings (same logic as analyze_embeddings)"
    )
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "Processed-data"),
                        help="Unlabeled Processed-data directory")
    parser.add_argument("--splits_dir", type=str, default=None,
                        help="Splits dir (default: same as data_dir)")
    parser.add_argument("--output_dir", type=str, default="embedding_analysis2",
                        help="Output directory for npz, UMAP, heatmap. Same folder as analyze_embeddings.py.")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Bandgap threshold for positive class (eV)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Use 1 to log skipped MOF names correctly")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 required for skip logging (error occurs in collate)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    splits_dir = os.path.abspath(args.splits_dir) if args.splits_dir else data_dir
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ensure_test_json(data_dir)

    # Config exactly as analyze_embeddings.py
    config = default_config_fn()
    config = json.loads(json.dumps(config))
    config["data_dir"] = data_dir
    config["downstream"] = "bandgaps_regression"
    config["load_path"] = "pmtransformer"
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config = get_valid_config(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Splits dir: {splits_dir}")
    print(f"  Output dir: {output_dir}")

    model = create_model(config, checkpoint_path=None)
    cif_ids, embeddings, bandgaps, split_labels = extract_embeddings(
        model, data_dir, splits_dir, config["downstream"], config,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device,
    )
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    npz_path = os.path.join(output_dir, "unlabeled_embeddings.npz")
    np.savez_compressed(
        npz_path,
        cif_ids=np.array(cif_ids),
        embeddings=embeddings,
        bandgaps=bandgaps,
        splits=np.array(split_labels),
    )
    print(f"  Saved: {npz_path} ({len(cif_ids)} unlabeled structures, dim={embeddings.shape[1]})")

    # Same PNG outputs as analyze_embeddings.py — so you can compare datasets visually
    print(f"\n  Generating UMAP and similarity heatmap (like analyze_embeddings)...")
    plot_umap(embeddings, bandgaps, split_labels, args.threshold,
              os.path.join(output_dir, "embedding_umap_unlabeled.png"))
    plot_similarity_heatmap(cif_ids, embeddings, bandgaps, split_labels, args.threshold,
                            os.path.join(output_dir, "similarity_heatmap_unlabeled.png"))

    print(f"\n  DONE. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
