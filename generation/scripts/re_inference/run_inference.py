#!/usr/bin/env python3
"""
Screening-pool inference: rank all test structures and output top-K for DFT.

Uses trained regressor checkpoints to infer on a MOFTransformer-ready
inference set. Ranks by predicted bandgap (lower = more likely conductive)
and writes the top 25 (or --top_k) structure IDs for DFT.

Usage (cluster):
  python scripts/re_inference/run_inference.py \\
    --data_dir /path/to/PMtransformer_Files \\
    --experiments exp364_embsplit_d_fulltune \\
    --top_k 25 \\
    --output_dir /path/to/inference_results

  python scripts/re_inference/run_inference.py \\
    --data_dir /path/to/PMtransformer_Files \\
    --checkpoints experiments/exp364_embsplit_d_fulltune/best_es-epoch=09.ckpt \\
    --top_k 25
"""

import os
import sys
import json
import glob
import shutil
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Project root = parent of this script's directory (or --base_dir when given)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_regressor import MOFRegressor
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config

THRESHOLD = 1.0


def find_best_checkpoint(exp_dir):
    """Find Spearman-best checkpoint (same logic as reinfer_nn)."""
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


def build_config(data_dir, downstream):
    """Build inference config for given data_dir (same architecture as reinfer_nn)."""
    config = default_config_fn()
    config = json.loads(json.dumps(config))
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config["data_dir"] = data_dir
    config["downstream"] = downstream
    config["threshold"] = THRESHOLD
    config["pooling_type"] = "mean"
    config["dropout"] = 0.0
    config["per_gpu_batchsize"] = 8
    config["batch_size"] = 32
    config["load_path"] = "pmtransformer"
    config = get_valid_config(config)
    return config


def resolve_downstream(data_dir, downstream_arg):
    """Resolve downstream tag from CLI or existing JSON files."""
    if downstream_arg and downstream_arg != "auto":
        return downstream_arg

    if os.path.exists(os.path.join(data_dir, "test_bandgaps.json")) or os.path.exists(
        os.path.join(data_dir, "inference_bandgaps.json")
    ):
        return "bandgaps"
    if os.path.exists(os.path.join(data_dir, "test_bandgaps_regression.json")) or os.path.exists(
        os.path.join(data_dir, "inference_bandgaps_regression.json")
    ):
        return "bandgaps_regression"
    return "bandgaps_regression"


def ensure_test_json(data_dir, downstream):
    """Ensure test_<downstream>.json exists, with practical fallbacks."""
    test_json = os.path.join(data_dir, f"test_{downstream}.json")
    if os.path.exists(test_json):
        return test_json

    inf_json = os.path.join(data_dir, f"inference_{downstream}.json")
    if os.path.exists(inf_json):
        shutil.copy2(inf_json, test_json)
        print(f"  Created {test_json} from inference JSON")
        return test_json

    # Cross-compatibility fallback for common naming mismatch.
    alt = "bandgaps" if downstream == "bandgaps_regression" else "bandgaps_regression"
    alt_test = os.path.join(data_dir, f"test_{alt}.json")
    if os.path.exists(alt_test):
        shutil.copy2(alt_test, test_json)
        print(f"  Created {test_json} from {os.path.basename(alt_test)}")
        return test_json
    alt_inf = os.path.join(data_dir, f"inference_{alt}.json")
    if os.path.exists(alt_inf):
        shutil.copy2(alt_inf, test_json)
        print(f"  Created {test_json} from {os.path.basename(alt_inf)}")
        return test_json

    raise FileNotFoundError(
        f"Missing {os.path.basename(test_json)}. Checked inference/test JSONs for "
        f"'{downstream}' and fallback '{alt}'."
    )


def run_inference(model, loader, device):
    """Run model on loader; return (cif_ids, scores) arrays."""
    model.eval()
    all_preds = []
    all_cif_ids = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            pooled, output = model.forward_features(batch)
            preds = model.regression_head(pooled).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            cif_ids = output.get("cif_id", output.get("name", None))
            all_preds.append(preds.cpu().numpy())
            if cif_ids:
                if isinstance(cif_ids, (list, tuple)):
                    all_cif_ids.extend(cif_ids)
                else:
                    all_cif_ids.append(cif_ids)
    preds_arr = np.concatenate(all_preds)
    return all_cif_ids, preds_arr


def main():
    parser = argparse.ArgumentParser(
        description="Screening-pool inference: rank structures and output top-K for DFT"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base project dir (for experiments, imports). Default: parent of this script's dir.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "Processed-data"),
        help="Data directory (Processed-data with inference_bandgaps_regression.json and test/)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Paths to .ckpt files (relative to cwd or absolute)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment names (e.g. exp364_embsplit_d_fulltune); uses experiments/ under project root",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=25,
        help="Number of top structures to output for DFT (default 25)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <data_dir>/inference_results)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CUDA/cuDNN for reproducibility",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="auto",
        help=(
            "Downstream tag for dataset JSON names (e.g. bandgaps or bandgaps_regression). "
            "Default 'auto' infers from files in --data_dir."
        ),
    )
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir) if args.base_dir else PROJECT_ROOT
    if base not in sys.path:
        sys.path.insert(0, base)

    data_dir = os.path.abspath(args.data_dir)
    if args.output_dir is None:
        output_dir = os.path.join(data_dir, "inference_results")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve checkpoints
    checkpoints = []
    if args.checkpoints:
        for c in args.checkpoints:
            p = os.path.abspath(c) if not os.path.isabs(c) else c
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            checkpoints.append(p)
    if args.experiments:
        experiments_dir = os.path.join(base, "experiments")
        for name in args.experiments:
            exp_dir = os.path.join(experiments_dir, name)
            if not os.path.isdir(exp_dir):
                raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
            ckpt = find_best_checkpoint(exp_dir)
            if ckpt is None:
                raise FileNotFoundError(f"No checkpoint found in {exp_dir}")
            checkpoints.append(ckpt)
    if not checkpoints:
        raise ValueError("Provide either --checkpoints or --experiments")

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        print("  Deterministic mode ON")

    print("=" * 70)
    print("  Screening-pool inference: rank test structures, output top-K for DFT")
    print("=" * 70)
    print(f"  Data dir:    {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Checkpoints: {len(checkpoints)}")
    print(f"  Top-K:       {args.top_k}")

    downstream = resolve_downstream(data_dir, args.downstream)
    print(f"  Downstream:  {downstream}")
    ensure_test_json(data_dir, downstream)
    config = build_config(data_dir, downstream)

    test_ds = Dataset(
        data_dir,
        split="test",
        downstream=downstream,
        nbr_fea_len=config["nbr_fea_len"],
        draw_false_grid=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["per_gpu_batchsize"],
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: Dataset.collate(x, config["img_size"]),
        pin_memory=True,
    )
    print(f"  Test samples: {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_scores_by_cid = {}  # cif_id -> list of scores (one per checkpoint)
    cif_id_order = None

    for i, ckpt_path in enumerate(checkpoints):
        print(f"\n  Loading checkpoint {i+1}/{len(checkpoints)}: {os.path.basename(ckpt_path)}")
        model = MOFRegressor.load_from_checkpoint(ckpt_path, config=config)
        model = model.to(device)
        cif_ids, scores = run_inference(model, test_loader, device)
        del model
        torch.cuda.empty_cache()

        if cif_id_order is None:
            cif_id_order = list(cif_ids)
        for j in range(len(scores)):
            cid = cif_ids[j] if j < len(cif_ids) else f"sample_{j}"
            all_scores_by_cid.setdefault(cid, []).append(float(scores[j]))

    # Aggregate: mean score across checkpoints (lower = better)
    agg_scores = []
    for cid in cif_id_order:
        sc_list = all_scores_by_cid.get(cid, [])
        agg_scores.append(np.mean(sc_list))

    # Full predictions CSV
    full_path = os.path.join(output_dir, "inference_predictions.csv")
    lines = ["cif_id,score,predicted_binary,true_label,mode"]
    for i, cid in enumerate(cif_id_order):
        sc = float(agg_scores[i])
        pb = 1 if sc < THRESHOLD else 0
        lines.append(f"{cid},{sc:.6f},{pb},0.0,regression")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved full predictions: {full_path}")

    # Rank by score ascending (lower predicted bandgap = higher priority for DFT)
    order = np.argsort(agg_scores)
    top_k = min(args.top_k, len(cif_id_order))
    top_indices = order[:top_k]
    top_ids = [cif_id_order[i] for i in top_indices]
    top_scores = [agg_scores[i] for i in top_indices]

    # top25_for_DFT.txt (or top{K}_for_DFT.txt)
    txt_name = f"top{top_k}_for_DFT.txt"
    txt_path = os.path.join(output_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        for cid in top_ids:
            f.write(cid + "\n")
    print(f"  Saved {txt_name}: {txt_path}")

    # top25_for_DFT.csv
    csv_name = f"top{top_k}_for_DFT.csv"
    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,cif_id,score\n")
        for r, (cid, sc) in enumerate(zip(top_ids, top_scores), start=1):
            f.write(f"{r},{cid},{sc:.6f}\n")
    print(f"  Saved {csv_name}: {csv_path}")

    # Full ranked CSV (all structures + rank + score)
    ranked_path = os.path.join(output_dir, "inference_ranked.csv")
    with open(ranked_path, "w", encoding="utf-8") as f:
        f.write("rank,cif_id,score\n")
        for r, idx in enumerate(order, start=1):
            f.write(f"{r},{cif_id_order[idx]},{float(agg_scores[idx]):.6f}\n")
    print(f"  Saved full ranked list: {ranked_path}")

    print(f"\n{'=' * 70}")
    print(f"  Top {top_k} for DFT (lowest predicted bandgap):")
    print(f"{'=' * 70}")
    for r, (cid, sc) in enumerate(zip(top_ids, top_scores), start=1):
        print(f"  {r:2d}. {cid}  {sc:.6f}")
    print(f"{'=' * 70}")
    print("  Done.")


if __name__ == "__main__":
    main()
