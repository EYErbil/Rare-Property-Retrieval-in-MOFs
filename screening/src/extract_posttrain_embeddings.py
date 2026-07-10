#!/usr/bin/env python3
"""
Extract post-training embeddings and save to .npz (same format as embeddings_pretrained.npz).
==========================================================================================
Then run umap_analysis_split_d.py with --embeddings_path pointing to this npz to get
all UMAP plots and reports (same as the working pretrained UMAP pipeline).

Usage:
  python extract_posttrain_embeddings.py --experiment exp364_fulltune \\
      --data_dir ./data/splits/strategy_d_farthest_point \\
      --output_npz ./umap_posttrain_exp364/embeddings_posttrain.npz

  # Then run the existing UMAP script (same as run_umap_original.sh uses for Split D):
  python umap_analysis_split_d.py \\
      --embeddings_path ./umap_posttrain_exp364/embeddings_posttrain.npz \\
      --splitd_dir ./data/splits/strategy_d_farthest_point \\
      --output_dir ./umap_posttrain_exp364
"""

import os
import sys
import json
import glob
import argparse
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_regressor import MOFRegressor
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config

THRESHOLD = 1.0
DOWNSTREAM = "bandgaps_regression"


def load_bandgap_lookup(data_dir):
    """Load canonical bandgaps (eV) from Split D JSONs, same as analyze_embeddings / UMAP script."""
    lookup = {}
    for split_name in ["train", "val", "test"]:
        json_path = os.path.join(data_dir, f"{split_name}_{DOWNSTREAM}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                labels = json.load(f)
            for cid, bg in labels.items():
                lookup[cid] = float(bg)
    return lookup


def _bandgap_for_cid(cid, bandgap_lookup, batch_target):
    """Resolve bandgap in eV: prefer canonical JSON; try cid and basename variants."""
    if bandgap_lookup is None:
        return float(batch_target)
    candidates = [cid]
    if isinstance(cid, str):
        base = os.path.basename(cid)
        if base != cid:
            candidates.append(base)
        if cid.endswith(".graphdata"):
            candidates.append(cid.replace(".graphdata", ""))
        if base.endswith(".graphdata"):
            candidates.append(base.replace(".graphdata", ""))
    for key in candidates:
        if key in bandgap_lookup:
            return bandgap_lookup[key]
    return float(batch_target)


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


def extract_embeddings(model, loaders, device, split_names, bandgap_lookup=None):
    """Run forward_features on all batches; return cif_ids, embeddings, bandgaps, splits.
    Bandgaps are taken from bandgap_lookup (canonical eV from Split D JSON) when available,
    so the UMAP colorbar shows true bandgap in eV, not whatever the Dataset puts in batch['target']."""
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
                # Use canonical bandgap from JSON when possible (same as pretrained pipeline)
                if cif_ids is not None:
                    if isinstance(cif_ids, (list, tuple)):
                        for i, cid in enumerate(cif_ids):
                            t = targets[i] if i < len(targets) else targets[0]
                            all_bandgaps.append(_bandgap_for_cid(cid, bandgap_lookup, t))
                    else:
                        all_bandgaps.append(_bandgap_for_cid(cif_ids, bandgap_lookup, targets[0]))
                else:
                    all_bandgaps.extend([float(t) for t in targets])
                all_splits.extend([split_name] * len(targets))
                if cif_ids is not None:
                    if isinstance(cif_ids, (list, tuple)):
                        all_cif_ids.extend(list(cif_ids))
                    else:
                        all_cif_ids.append(str(cif_ids))
                else:
                    # Fallback: no cif_id in output, use placeholder so lengths match
                    n = len(targets)
                    all_cif_ids.extend([f"sample_{split_name}_{len(all_cif_ids)+i}" for i in range(n)])

    embeddings = np.concatenate(all_embeddings, axis=0)
    bandgaps = np.array(all_bandgaps)
    return all_cif_ids, embeddings, bandgaps, all_splits


def main():
    parser = argparse.ArgumentParser(description="Extract post-training embeddings to .npz")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt")
    parser.add_argument("--experiment", type=str, default=None,
                        help="e.g. exp364_fulltune; finds best_es in experiments/<name>")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Split directory (train/val/test with .graphdata etc.)")
    parser.add_argument("--output_npz", type=str, required=True,
                        help="Output path for embeddings_posttrain.npz")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(args.data_dir) if os.path.isabs(args.data_dir) else os.path.join(base_dir, args.data_dir)
    output_npz = os.path.abspath(args.output_npz) if os.path.isabs(args.output_npz) else os.path.join(os.getcwd(), args.output_npz)
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)

    ckpt_path = args.checkpoint
    if ckpt_path is None and args.experiment:
        exp_dir = os.path.join(base_dir, "experiments", args.experiment)
        ckpt_path = find_best_checkpoint(exp_dir)
        if ckpt_path is None:
            print(f"ERROR: No checkpoint for experiment {args.experiment}", file=sys.stderr)
            sys.exit(1)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("ERROR: Provide --checkpoint or --experiment with a valid checkpoint", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  EXTRACT POST-TRAINING EMBEDDINGS")
    print("=" * 70)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output:     {output_npz}")

    try:
        config = build_config(data_dir)
        collate = lambda x: Dataset.collate(x, config["img_size"])
        batch_size = config.get("per_gpu_batchsize", 16)

        train_ds = Dataset(data_dir, split="train", downstream=config["downstream"],
                           nbr_fea_len=config["nbr_fea_len"], draw_false_grid=False)
        val_ds = Dataset(data_dir, split="val", downstream=config["downstream"],
                         nbr_fea_len=config["nbr_fea_len"], draw_false_grid=False)
        try:
            test_ds = Dataset(data_dir, split="test", downstream=config["downstream"],
                             nbr_fea_len=config["nbr_fea_len"], draw_false_grid=False)
        except Exception:
            test_ds = None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate) if test_ds else None

        loaders = [train_loader, val_loader, test_loader]
        split_names = ["train", "val", "test"]

        print("  Loading canonical bandgaps from Split D JSONs (for correct UMAP colorbar in eV)...")
        bandgap_lookup = load_bandgap_lookup(data_dir)
        print(f"  Loaded {len(bandgap_lookup)} bandgaps from {data_dir}")

        print("  Loading model...")
        model = MOFRegressor.load_from_checkpoint(ckpt_path, config=config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print("  Extracting embeddings (train + val + test)...")
        cif_ids, embeddings, bandgaps, splits = extract_embeddings(
            model, loaders, device, split_names, bandgap_lookup=bandgap_lookup
        )
        print(f"  Total: {len(cif_ids)} MOFs, dim={embeddings.shape[1]}")

        del model
        torch.cuda.empty_cache()

        # Same format as embeddings_pretrained.npz (load_embeddings in umap_analysis_split_d.py)
        np.savez_compressed(
            output_npz,
            cif_ids=np.array(cif_ids),
            embeddings=embeddings,
            bandgaps=bandgaps,
            splits=np.array(splits),
        )
        print(f"  Saved: {output_npz}")
        print("=" * 70)
        print("  Next: run umap_analysis_split_d.py with --embeddings_path this file")
        print("=" * 70)
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
