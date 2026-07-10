#!/usr/bin/env python3
"""
Run NN inference using the best_es checkpoint in the current folder.

Run from the experiment folder (same place as run.py):
  cd experiments/exp364_fulltune
  python ../../discovery/run_inference_from_cwd.py

Uses best_es checkpoint (Spearman-best). Writes inference_predictions.csv, inference_ranked.csv, top{N}_for_DFT.txt directly into the current folder.
"""
import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is parent of discovery/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Override with --data_dir; default: data/unlabeled/ relative to repo root
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "unlabeled")

TOP_K = 25

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_regressor import MOFRegressor
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config

THRESHOLD = 1.0


def find_best_es(exp_dir):
    """Find Spearman-best checkpoint (best_es)."""
    results_path = os.path.join(exp_dir, "final_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            best = json.load(f).get("checkpoints", {}).get("best", "")
        if best:
            p = best if os.path.isabs(best) else os.path.join(exp_dir, best)
            if os.path.exists(p):
                return p
    for pattern in ["best_es-*.ckpt", "best_*.ckpt"]:
        matches = glob.glob(os.path.join(exp_dir, pattern))
        if matches:
            return sorted(matches)[-1]
    last = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    return None


def run_inference(model, loader, device):
    model.eval()
    all_preds, all_cif_ids = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            pooled, output = model.forward_features(batch)
            preds = model.regression_head(pooled).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            all_preds.append(preds.cpu().numpy())
            cids = output.get("cif_id", output.get("name", None))
            if cids:
                if isinstance(cids, (list, tuple)):
                    all_cif_ids.extend(cids)
                else:
                    all_cif_ids.append(cids)
    return all_cif_ids, np.concatenate(all_preds)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run NN inference using best_es checkpoint in current folder.")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to folder with structures and test_bandgaps_regression.json")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of top candidates to output")
    args = parser.parse_args()

    exp_dir = os.getcwd()
    ckpt = find_best_es(exp_dir)
    if not ckpt:
        raise SystemExit(f"No best_es checkpoint in {exp_dir}")

    data_dir = os.path.abspath(args.data_dir)
    output_dir = exp_dir

    # Ensure test JSON
    test_json = os.path.join(data_dir, "test_bandgaps_regression.json")
    if not os.path.exists(test_json):
        inf_json = os.path.join(data_dir, "inference_bandgaps_regression.json")
        if not os.path.exists(inf_json):
            raise SystemExit(f"Neither {test_json} nor {inf_json} found")
        import shutil
        shutil.copy2(inf_json, test_json)

    # Config
    config = default_config_fn()
    config = json.loads(json.dumps(config))
    config["loss_names"] = {"ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0, "regression": 1, "classification": 0}
    config["data_dir"] = data_dir
    config["downstream"] = "bandgaps_regression"
    config["threshold"] = THRESHOLD
    config["pooling_type"] = "mean"
    config["dropout"] = 0.0
    config["per_gpu_batchsize"] = 8
    config["batch_size"] = 32
    config["load_path"] = "pmtransformer"
    config = get_valid_config(config)

    ds = Dataset(data_dir, split="test", downstream="bandgaps_regression", nbr_fea_len=config["nbr_fea_len"], draw_false_grid=False)
    loader = DataLoader(ds, batch_size=config["per_gpu_batchsize"], shuffle=False, num_workers=4,
                        collate_fn=lambda x: Dataset.collate(x, config["img_size"]), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOFRegressor.load_from_checkpoint(ckpt, config=config).to(device)
    cif_ids, scores = run_inference(model, loader, device)

    # CSV
    path = os.path.join(exp_dir, "inference_predictions.csv")
    lines = ["cif_id,score,predicted_binary,true_label,mode"]
    for i, cid in enumerate(cif_ids):
        sc = float(scores[i])
        pb = 1 if sc < THRESHOLD else 0
        lines.append(f"{cid},{sc:.6f},{pb},0.0,regression")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")

    # Ranked
    order = np.argsort(scores)
    ranked = os.path.join(output_dir, "inference_ranked.csv")
    with open(ranked, "w", encoding="utf-8") as f:
        f.write("rank,cif_id,score\n")
        for r, idx in enumerate(order, start=1):
            f.write(f"{r},{cif_ids[idx]},{float(scores[idx]):.6f}\n")
    print(f"Saved {ranked}")

    # Top-K
    top_ids = [cif_ids[i] for i in order[:TOP_K]]
    txt = os.path.join(exp_dir, f"top{TOP_K}_for_DFT.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(top_ids) + "\n")
    print(f"Saved {txt}")


if __name__ == "__main__":
    main()
