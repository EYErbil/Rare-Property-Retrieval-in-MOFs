#!/usr/bin/env python3
"""
Re-run NN test inference after split modification.

Loads each trained NN checkpoint and re-runs test_step on the current
split under ``--data_dir``. Model weights are not modified.

If a reference ``hideffs/test_predictions.csv`` exists per experiment, the
script can verify that re-inference matches for overlapping MOFs (optional).

Why can re-inference differ from the reference?
  - GPU non-determinism: CUDA/cuDNN often use non-deterministic reductions,
    so the same model + same input can yield tiny differences (e.g. 1e-5).
  - Run with --deterministic to force reproducible ops (slower, usually exact match).
  - Small differences (< ~1e-5) are numerical noise; large ones would indicate
    wrong data or wrong checkpoint.

Usage (on the cluster):
  python reinfer_nn.py
  python reinfer_nn.py --deterministic    # reproducible (slower)
  python reinfer_nn.py --tolerance 1e-5   # allow small GPU noise
  python reinfer_nn.py --experiments exp364_fulltune
"""

import os
import sys
import json
import glob
import shutil
import argparse
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Non-interactive backend for plots on cluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_regressor import (
    MOFRegressor,
    _plot_test_scatter,
    _plot_discovery_curve_regression,
)
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
DATA_DIR = os.path.join(_REPO_ROOT, "data", "splits", "strategy_d_farthest_point")
EXPERIMENTS_DIR = os.path.join(_REPO_ROOT, "experiments")
THRESHOLD = 1.0


def find_split_d_experiments():
    """Auto-discover all Split D experiment directories that have checkpoints."""
    found = {}
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        if not name.startswith("exp"):
            continue
        exp_dir = os.path.join(EXPERIMENTS_DIR, name)
        if not os.path.isdir(exp_dir):
            continue
        ckpt = find_best_checkpoint(exp_dir)
        if ckpt:
            found[name] = exp_dir
    return found


def find_best_checkpoint(exp_dir):
    """Find the checkpoint used for final test inference (Spearman-best).

    Training uses es_monitor='val/spearman_rho', so the early-stopping checkpoint
    (best_es) is the one that maximizes validation Spearman. Final test and
    test_predictions.csv are produced with that checkpoint. We use the same here.
    """
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

    # Prefer best_es (early-stopping = Spearman when es_monitor=val/spearman_rho)
    for pattern in ["best_es-*.ckpt", "best_*.ckpt"]:
        matches = glob.glob(os.path.join(exp_dir, pattern))
        if matches:
            return sorted(matches)[-1]

    # last.ckpt only if no Spearman checkpoint exists (e.g. run interrupted)
    last = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(last):
        return last

    return None


def build_config():
    """Build inference config.

    All Split D experiments share the same architecture:
      pooling_type = "mean"
      dropout      = 0.0

    These are the ONLY config values that affect the forward pass.
    Everything else (loss_type, freeze_layers, use_sample_weights,
    learning_rate, etc.) is training-only and irrelevant for inference.
    """
    config = default_config_fn()
    config = json.loads(json.dumps(config))

    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config["data_dir"] = DATA_DIR
    config["downstream"] = "bandgaps_regression"
    config["threshold"] = THRESHOLD
    config["pooling_type"] = "mean"
    config["dropout"] = 0.0
    config["per_gpu_batchsize"] = 8
    config["batch_size"] = 32
    config["load_path"] = "pmtransformer"

    config = get_valid_config(config)
    return config


# Reference (pre-swap) predictions are stored here for verification
HIDEFFS_DIR = "hideffs"
REFERENCE_PREDICTIONS_FILENAME = "test_predictions.csv"


def get_reference_predictions_path(exp_dir):
    """Path to the hidden reference test_predictions.csv (pre-swap)."""
    return os.path.join(exp_dir, HIDEFFS_DIR, REFERENCE_PREDICTIONS_FILENAME)


def load_reference_predictions(exp_dir):
    """Load reference (pre-swap) test_predictions.csv from hideffs/.

    Expected layout: {exp_dir}/hideffs/test_predictions.csv
    Returns dict {cif_id: score} or None if file missing.
    """
    ref_path = get_reference_predictions_path(exp_dir)
    if not os.path.exists(ref_path):
        return None
    old = {}
    with open(ref_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("cif_id") or not line:
                continue
            parts = line.split(",")
            old[parts[0]] = float(parts[1])
    return old


def reinfer(exp_name, exp_dir, config, test_loader, verify_tol=1e-6):
    """Load checkpoint and re-run test inference for one experiment."""
    ckpt_path = find_best_checkpoint(exp_dir)
    if ckpt_path is None:
        print(f"  SKIP: no checkpoint found")
        return False

    print(f"  Checkpoint: {ckpt_path}")

    # Load reference (pre-swap) predictions from hideffs/ for verification
    ref_path = get_reference_predictions_path(exp_dir)
    old_preds = load_reference_predictions(exp_dir)

    # Load model from checkpoint
    print(f"  Loading model...")
    model = MOFRegressor.load_from_checkpoint(ckpt_path, config=config)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds = []
    all_targets = []
    all_cif_ids = []

    print(f"  Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            pooled, output = model.forward_features(batch)
            preds = model.regression_head(pooled).squeeze()
            targets = torch.tensor(
                batch["target"], device=device, dtype=torch.float
            ).squeeze()

            cif_ids = output.get("cif_id", output.get("name", None))

            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            if cif_ids:
                if isinstance(cif_ids, (list, tuple)):
                    all_cif_ids.extend(cif_ids)
                else:
                    all_cif_ids.append(cif_ids)

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    # Sanity checks
    has_badxej = "BADXEJ01_FSR" in all_cif_ids
    has_iledio = "ILEDIO_FSR" in all_cif_ids
    print(f"  Total predictions: {len(preds_arr)}")
    print(f"  BADXEJ01_FSR present: {has_badxej}")
    print(f"  ILEDIO_FSR absent:    {not has_iledio}")

    if has_iledio:
        print("  ERROR: ILEDIO_FSR still in test! Run fix_split_symlinks.sh first.")
        return False
    if not has_badxej:
        print("  ERROR: BADXEJ01_FSR not in test! Check symlinks.")
        return False

    pred_path = os.path.join(exp_dir, "test_predictions.csv")

    # --- Always write CSV and plots first (so something is saved even if verification fails) ---
    # Backup current file if it exists (and is not already in hideffs)
    if os.path.exists(pred_path):
        backup = pred_path + ".bak_pre_swap"
        if not os.path.exists(backup):
            shutil.copy2(pred_path, backup)
            print(f"  Backed up: {backup}")

    # Write new predictions (same format as train_regressor.py L1404-1413)
    lines = ["cif_id,score,predicted_binary,true_label,mode"]
    for i in range(len(preds_arr)):
        cid = all_cif_ids[i] if i < len(all_cif_ids) else f"sample_{i}"
        sc = float(preds_arr[i])
        pb = 1 if sc < THRESHOLD else 0
        tl = float(targets_arr[i])
        lines.append(f"{cid},{sc:.6f},{pb},{tl},regression")

    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Saved {len(preds_arr)} predictions to: {pred_path}")

    # Save plots (same as trainer after test)
    class _FakeTestMetrics:
        def __init__(self, preds, targets):
            self.preds = list(preds)
            self.targets = list(targets)

    fake_metrics = _FakeTestMetrics(preds_arr, targets_arr)
    try:
        _plot_test_scatter(fake_metrics, THRESHOLD, exp_dir)
        print(f"  Saved: {os.path.join(exp_dir, 'test_scatter.png')}")
    except Exception as e:
        print(f"  WARNING: test_scatter plot failed: {e}")
    try:
        _plot_discovery_curve_regression(fake_metrics, THRESHOLD, exp_dir)
        print(f"  Saved: {os.path.join(exp_dir, 'discovery_curve.png')}")
    except Exception as e:
        print(f"  WARNING: discovery_curve plot failed: {e}")

    if has_badxej:
        idx = all_cif_ids.index("BADXEJ01_FSR")
        print(f"  >> BADXEJ01_FSR: predicted={preds_arr[idx]:.6f}, "
              f"true={targets_arr[idx]:.6f}")

    # Free GPU memory before verification
    del model
    torch.cuda.empty_cache()

    # --- Verification against reference (hideffs) ---
    new_preds_dict = {all_cif_ids[i]: float(preds_arr[i])
                      for i in range(len(all_cif_ids))}

    if old_preds is None:
        print(f"  Verification: FAIL (reference not found: {ref_path})")
        print(f"                Put pre-swap test_predictions.csv in {HIDEFFS_DIR}/")
        return False

    shared = set(old_preds.keys()) & set(new_preds_dict.keys())
    diffs = [abs(old_preds[cid] - new_preds_dict[cid]) for cid in shared]
    max_diff = max(diffs) if diffs else 0.0
    n_diffs = sum(1 for d in diffs if d > verify_tol)

    # Histogram of differences (helps tell noise from bug)
    buckets = [(0, 1e-7), (1e-7, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, float("inf"))]
    hist = []
    for lo, hi in buckets:
        count = sum(1 for d in diffs if lo <= d < hi)
        hist.append((lo, hi, count))

    print(f"  Verification (reference: {ref_path}, tolerance={verify_tol:.0e}):")
    print(f"    shared MOFs: {len(shared)}  (expected 9549)")
    print(f"    max |new - ref|: {max_diff:.2e}")
    print(f"    mismatches (>{verify_tol:.0e}): {n_diffs}")
    print(f"    Difference histogram:")
    for lo, hi, count in hist:
        if count > 0:
            label = f"[{lo:.0e}, {hi:.0e})" if hi < 1e10 else f">= {lo:.0e}"
            print(f"      {label}: {count} MOFs")

    if n_diffs > 0:
        print(f"    FAIL: inferences differ from reference.")
        if max_diff < 1e-4:
            print(f"    (Differences are small — likely GPU non-determinism. Try: python reinfer_nn.py --deterministic)")
        else:
            print(f"    (Large differences — check same checkpoint, same data_dir, and no code changes.)")
        shown = 0
        for cid in sorted(shared):
            if abs(old_preds[cid] - new_preds_dict[cid]) > verify_tol and shown < 5:
                print(f"      {cid}: ref={old_preds[cid]:.6f} new={new_preds_dict[cid]:.6f}")
                shown += 1
        return False

    print(f"    PASS: all {len(shared)} shared predictions match reference (within {verify_tol:.0e}).")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Re-run NN test inference after split swap")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiment names (default: auto-discover all with checkpoints)")
    parser.add_argument("--experiments_dir", type=str, default=None,
                        help="Directory containing experiment folders (default: <repo>/experiments)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Split directory with test_bandgaps_regression.json (default: <repo>/data/splits/strategy_d_farthest_point)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic CUDA/cuDNN (reproducible, slower)")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Verification tolerance |new-ref| (default 1e-6; use 1e-5 for GPU noise)")
    args = parser.parse_args()

    global DATA_DIR, EXPERIMENTS_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir
    if args.experiments_dir:
        EXPERIMENTS_DIR = args.experiments_dir

    pl.seed_everything(42)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        print("  Deterministic mode ON (reproducible inference)")

    print("=" * 70)
    print("  RE-INFERENCE: Updating test_predictions.csv for NN models")
    print("=" * 70)

    # Verify split files first
    test_json = os.path.join(DATA_DIR, "test_bandgaps_regression.json")
    if not os.path.exists(test_json):
        print(f"  ERROR: {test_json} not found")
        sys.exit(1)

    with open(test_json) as f:
        test_data = json.load(f)
    test_pos = [k for k, v in test_data.items() if float(v) < THRESHOLD]

    print(f"  Data dir:  {DATA_DIR}")
    print(f"  Test set:  {len(test_data)} MOFs ({len(test_pos)} positives)")

    # Discover experiments
    if args.experiments:
        experiments = {name: os.path.join(EXPERIMENTS_DIR, name)
                       for name in args.experiments}
    else:
        experiments = find_split_d_experiments()

    if not experiments:
        print("  No Split D experiments with checkpoints found.")
        sys.exit(1)

    print(f"\n  Experiments found ({len(experiments)}):")
    for name in experiments:
        ckpt = find_best_checkpoint(experiments[name])
        print(f"    {name}: {ckpt}")

    # Build shared config (all experiments use same architecture)
    config = build_config()

    # Create test DataLoader (shared across all experiments)
    print(f"\n  Creating test dataset...")
    test_ds = Dataset(
        DATA_DIR,
        split="test",
        downstream="bandgaps_regression",
        nbr_fea_len=config["nbr_fea_len"],
        draw_false_grid=False,
    )
    print(f"  Test samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=config["per_gpu_batchsize"],
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: Dataset.collate(x, config["img_size"]),
        pin_memory=True,
    )

    # Re-infer each experiment
    success = 0
    for exp_name, exp_dir in sorted(experiments.items()):
        print(f"\n--- {exp_name} ---")
        if reinfer(exp_name, exp_dir, config, test_loader, verify_tol=args.tolerance):
            success += 1

    print(f"\n{'=' * 70}")
    print(f"  Done: {success}/{len(experiments)} experiments re-inferred")
    print(f"{'=' * 70}")

    if success < len(experiments):
        sys.exit(1)


if __name__ == "__main__":
    main()
