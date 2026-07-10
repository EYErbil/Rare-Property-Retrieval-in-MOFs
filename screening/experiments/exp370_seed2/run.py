#!/usr/bin/env python
"""
Experiment 370: Embedding Split D - Full finetune, seed=123
-------------------------------------------------------------
Strategy D: Farthest-point coverage split.
Same config as exp364 (best performer) but with different random seed
for ensemble diversity.
"""

import sys, os, argparse

# Add src/ to Python path (repo_root/src/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from train_regressor import run

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "splits", "strategy_d_farthest_point")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to split directory containing train/val/test JSONs and structure files")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        downstream="bandgaps_regression",
        threshold=1.0,
        loss_type="huber",
        pooling_type="mean",
        freeze_layers=0,
        use_sample_weights=False,
        es_monitor="val/spearman_rho",
        es_mode="max",
        batch_size=32,
        per_gpu_batchsize=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_mult=10.0,
        max_epochs=100,
        patience=15,
        log_dir=".",
        seed=123,
        num_workers=4,
    )
