#!/usr/bin/env python3
"""
MOF Bandgap Regression Training with Ranking Evaluation
=========================================================

This script trains a regressor to predict MOF bandgap values (continuous).
The model is evaluated using RANKING metrics - we rank MOFs by predicted bandgap
and measure how many true low-bandgap MOFs are in the top-K.

WHY REGRESSION FOR DISCOVERY:
-----------------------------
Classification with extreme imbalance (0.09% positives) is very hard.
Regression naturally produces a ranking - just predict bandgap and sort.
No threshold needed at training time!

EVALUATION METRICS (same as classifier - discovery focused):
------------------------------------------------------------
- Recall@K: Of all true low-gap MOFs, how many are in top-K (lowest predicted)?
- Precision@K: Of top-K predictions, how many are truly low-gap?
- Enrichment@K: How much better than random?
- MAE/RMSE: Standard regression metrics (secondary)

LOSS OPTIONS:
-------------
- MSE: Standard mean squared error
- Huber: Robust to outliers (good for bandgaps)
- Weighted MSE: Higher weight for low-bandgap samples

USAGE:
------
    from train_regressor import run
    run(data_dir="./dataset", downstream="bandgaps", ...)

Author: Generated for MOF bandgap discovery (regression approach)
"""

import os
import sys
import json
import logging
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, kendalltau

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress torchmetrics warnings
warnings.filterwarnings("ignore", message=".*compute without update.*")
warnings.filterwarnings("ignore", message=".*No positive samples.*")

# MOFTransformer imports
from moftransformer.modules.module import Module
from moftransformer.modules import heads, objectives
from moftransformer.datamodules.dataset import Dataset
from moftransformer.config import config as default_config_fn
from moftransformer.utils.validation import get_valid_config, get_num_devices


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: str, name: str = "regressor") -> logging.Logger:
    """Setup dual logging to console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    return logger


# =============================================================================
# GRADIENT TRACKING CALLBACK
# =============================================================================

class GradientTracker(Callback):
    """Track gradient norms, losses, and produce training dashboards."""
    
    def __init__(self, log_dir: str, log_every_n_steps: int = 50):
        super().__init__()
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self.gradient_history = []
        self.loss_history = []
        self.val_metrics_history = []
        self.epoch_train_losses = []
        self.max_grad_norm = 0.0
        self.grad_explosion_count = 0
    
    def on_after_backward(self, trainer, pl_module):
        """Log gradient norms after backward pass."""
        if trainer.global_step % self.log_every_n_steps == 0:
            total_norm = 0.0
            layer_norms = {}
            
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm(2).item()
                    total_norm += norm ** 2
                    
                    # Track key layers
                    if 'regression_head' in name:
                        layer_norms[name] = norm
                    elif 'blocks.11' in name or 'blocks.10' in name:
                        layer_norms[name] = norm
            
            total_norm = total_norm ** 0.5
            self.max_grad_norm = max(self.max_grad_norm, total_norm)
            
            self.gradient_history.append({
                'step': trainer.global_step,
                'epoch': trainer.current_epoch,
                'total_norm': total_norm,
                'layer_norms': layer_norms
            })
            
            # Warn if gradient explosion
            if total_norm > 100:
                self.grad_explosion_count += 1
                print(f"[WARNING] Large gradient norm: {total_norm:.2f} at step {trainer.global_step}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track training loss."""
        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].item() if torch.is_tensor(outputs['loss']) else outputs['loss']
        elif torch.is_tensor(outputs):
            loss = outputs.item()
        
        if loss is not None:
            self.loss_history.append({
                'step': trainer.global_step,
                'epoch': trainer.current_epoch,
                'loss': loss
            })
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Track epoch-level training loss."""
        epoch = trainer.current_epoch
        epoch_losses = [h['loss'] for h in self.loss_history if h['epoch'] == epoch]
        if epoch_losses:
            self.epoch_train_losses.append({
                'epoch': epoch,
                'avg_loss': float(np.mean(epoch_losses)),
                'min_loss': float(np.min(epoch_losses)),
                'max_loss': float(np.max(epoch_losses)),
                'std_loss': float(np.std(epoch_losses)),
            })

    def on_validation_epoch_end(self, trainer, pl_module):
        """Track validation metrics."""
        metrics = {k: v.item() if torch.is_tensor(v) else v 
                   for k, v in trainer.callback_metrics.items()}
        metrics['epoch'] = trainer.current_epoch
        self.val_metrics_history.append(metrics)
    
    def save_history(self):
        """Save all history to JSON and generate plots."""
        os.makedirs(self.log_dir, exist_ok=True)
        history = {
            'gradients': self.gradient_history,
            'train_loss': self.loss_history,
            'epoch_train_losses': self.epoch_train_losses,
            'val_metrics': self.val_metrics_history,
            'analysis': {
                'max_gradient_norm': self.max_grad_norm,
                'gradient_explosions': self.grad_explosion_count,
            },
        }
        
        path = os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self._plot_dashboard()
        return path

    def _plot_dashboard(self):
        """Generate 4-panel training dashboard."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Regression Training Dashboard', fontsize=14, fontweight='bold')

            # 1. Training loss curve (per-batch + smoothed)
            ax = axes[0, 0]
            if self.loss_history:
                steps = [h['step'] for h in self.loss_history]
                losses = [h['loss'] for h in self.loss_history]
                ax.plot(steps, losses, alpha=0.3, color='blue', label='Per-batch')
                w = min(50, len(losses) // 10 + 1)
                if len(losses) > w:
                    sm = np.convolve(losses, np.ones(w)/w, mode='valid')
                    ax.plot(steps[w-1:], sm, color='red', lw=2, label='Smoothed')
            ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
            ax.legend(); ax.grid(True, alpha=0.3)

            # 2. Gradient norms
            ax = axes[0, 1]
            if self.gradient_history:
                steps = [h['step'] for h in self.gradient_history]
                norms = [h['total_norm'] for h in self.gradient_history]
                ax.plot(steps, norms, color='green', alpha=0.7)
                ax.axhline(1.0, color='orange', ls='--', label='Clip')
                ax.axhline(10.0, color='red', ls='--', label='Warning')
            ax.set_xlabel('Step'); ax.set_ylabel('Norm')
            ax.set_title(f'Gradient Norms (max={self.max_grad_norm:.2f})')
            ax.legend(); ax.grid(True, alpha=0.3)

            # 3. Val metrics (ranking + correlation)
            ax = axes[1, 0]
            if self.val_metrics_history:
                epochs = [m['epoch'] for m in self.val_metrics_history]
                for key, color, label in [
                    ('val/spearman_rho', 'purple', 'Spearman ρ'),
                    ('val/recall@100', 'green', 'Recall@100'),
                    ('val/recall@50', 'blue', 'Recall@50'),
                    ('val/recall@200', 'orange', 'Recall@200'),
                ]:
                    vals = [m.get(key) for m in self.val_metrics_history]
                    ve = [e for e, v in zip(epochs, vals) if v is not None]
                    vv = [v for v in vals if v is not None]
                    if vv:
                        ax.plot(ve, vv, marker='o', ms=3, label=label, color=color)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
            ax.set_title('Validation Metrics'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            # 4. Train vs Val loss per epoch
            ax = axes[1, 1]
            if self.epoch_train_losses:
                epochs = [e['epoch'] for e in self.epoch_train_losses]
                ax.plot(epochs, [e['avg_loss'] for e in self.epoch_train_losses],
                        'b-o', ms=3, label='Train')
                ax.fill_between(epochs,
                                [e['min_loss'] for e in self.epoch_train_losses],
                                [e['max_loss'] for e in self.epoch_train_losses],
                                alpha=0.2, color='blue')
            if self.val_metrics_history:
                ve, vl = [], []
                for m in self.val_metrics_history:
                    v = m.get('val/loss', m.get('val_loss'))
                    if v is not None:
                        ve.append(m['epoch']); vl.append(v)
                if vl:
                    ax.plot(ve, vl, 'r-o', ms=3, label='Val')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title('Train vs Val Loss'); ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'training_dashboard.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            # Also generate learning curves
            self._plot_learning_curves()

        except Exception as e:
            print(f"[GradientTracker] Plot error: {e}")

    def _plot_learning_curves(self):
        """Generate detailed learning curves for ranking & regression metrics."""
        try:
            if not self.val_metrics_history:
                return
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Panel 1: Spearman + Pearson over epochs
            ax = axes[0]
            epochs = [m['epoch'] for m in self.val_metrics_history]
            for key, color, label in [
                ('val/spearman_rho', 'purple', 'Spearman ρ'),
                ('val/pearson_r', 'blue', 'Pearson r'),
                ('val/r2', 'green', 'R²'),
            ]:
                vals = [m.get(key) for m in self.val_metrics_history]
                ve = [e for e, v in zip(epochs, vals) if v is not None]
                vv = [v for v in vals if v is not None]
                if vv:
                    ax.plot(ve, vv, color=color, marker='o', ms=4, label=label)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Correlation')
            ax.set_title('Ranking / Correlation Quality')
            ax.legend(); ax.grid(True, alpha=0.3)

            # Panel 2: Recall@K over epochs
            ax = axes[1]
            for k, color in [(50, 'blue'), (100, 'green'), (200, 'red')]:
                vals = [m.get(f'val/recall@{k}') for m in self.val_metrics_history]
                ve = [e for e, v in zip(epochs, vals) if v is not None]
                vv = [v for v in vals if v is not None]
                if vv:
                    ax.plot(ve, vv, color=color, marker='o', ms=3, label=f'R@{k}')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Recall')
            ax.set_title('Discovery: Recall@K')
            ax.legend(); ax.grid(True, alpha=0.3)

            # Panel 3: MAE / RMSE over epochs
            ax = axes[2]
            for key, color, label in [
                ('val/mae', 'blue', 'MAE'),
                ('val/rmse', 'red', 'RMSE'),
            ]:
                vals = [m.get(key) for m in self.val_metrics_history]
                ve = [e for e, v in zip(epochs, vals) if v is not None]
                vv = [v for v in vals if v is not None]
                if vv:
                    ax.plot(ve, vv, color=color, marker='o', ms=3, label=label)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Error (eV)')
            ax.set_title('Regression Error')
            ax.legend(); ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'learning_curves.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[GradientTracker] Learning curve plot error: {e}")


# =============================================================================
# RANKING METRICS: Discovery-focused evaluation for regression
# =============================================================================

class RankingMetrics:
    """
    Compute discovery metrics for regression via ranking.
    
    We rank MOFs by PREDICTED bandgap (ascending - lowest first).
    Then measure how many TRUE low-gap MOFs are in top-K.
    
    This is the same as classification metrics but using regression predictions!
    """
    
    def __init__(self, k_values: List[int] = [50, 100, 200], threshold: float = 1.0):
        self.k_values = k_values
        self.threshold = threshold  # For defining "positive" (true bandgap < threshold)
        self.reset()
    
    def reset(self):
        self.preds = []  # Predicted bandgaps
        self.targets = []  # True bandgaps
        self.cif_ids = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, cif_ids: List[str] = None):
        """Accumulate predictions."""
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        self.preds.extend(preds.flatten().tolist())
        self.targets.extend(targets.flatten().tolist())
        if cif_ids:
            self.cif_ids.extend(cif_ids)
    
    def compute(self) -> Dict[str, float]:
        """Compute ranking-based discovery metrics."""
        if len(self.preds) == 0:
            return {}
        
        preds = np.array(self.preds)
        targets = np.array(self.targets)
        
        # Binary labels based on threshold
        labels = (targets < self.threshold).astype(int)
        
        results = {}
        
        # Sort by PREDICTED bandgap (ascending - lowest predicted first)
        sorted_idx = np.argsort(preds)  # Ascending!
        sorted_labels = labels[sorted_idx]
        sorted_targets = targets[sorted_idx]
        
        n_total = len(labels)
        n_positive = labels.sum()
        baseline_prevalence = n_positive / n_total if n_total > 0 else 0
        
        results['n_total'] = n_total
        results['n_positive'] = int(n_positive)
        results['baseline_prevalence'] = baseline_prevalence
        
        # Recall@K and Precision@K (based on ranking by predicted bandgap)
        for k in self.k_values:
            k_actual = min(k, n_total)
            
            top_k_labels = sorted_labels[:k_actual]
            true_positives_at_k = top_k_labels.sum()
            
            # Recall@K: TP@K / total_positives
            recall_at_k = true_positives_at_k / n_positive if n_positive > 0 else 0
            results[f'recall@{k}'] = recall_at_k
            
            # Precision@K: TP@K / K
            precision_at_k = true_positives_at_k / k_actual
            results[f'precision@{k}'] = precision_at_k
            
            # Enrichment@K
            enrichment_at_k = precision_at_k / baseline_prevalence if baseline_prevalence > 0 else 0
            results[f'enrichment@{k}'] = enrichment_at_k
        
        # Standard regression metrics
        results['mae'] = mean_absolute_error(targets, preds)
        results['rmse'] = np.sqrt(mean_squared_error(targets, preds))
        results['r2'] = r2_score(targets, preds)
        
        # Correlation metrics (CRITICAL for ranking quality)
        if len(preds) > 1:
            results['pearson_r'] = np.corrcoef(targets, preds)[0, 1]
            
            # Spearman correlation - measures ranking quality
            spearman_rho, spearman_p = spearmanr(targets, preds)
            results['spearman_rho'] = spearman_rho
            results['spearman_pvalue'] = spearman_p
            
            # Kendall tau - alternative ranking metric
            kendall_tau, kendall_p = kendalltau(targets, preds)
            results['kendall_tau'] = kendall_tau
        else:
            results['pearson_r'] = 0.0
            results['spearman_rho'] = 0.0
            results['kendall_tau'] = 0.0

        # === NEF (Normalized Enrichment Factor) at fraction of library ===
        for frac_pct in [1, 2, 5, 10]:
            k_frac = max(1, int(n_total * frac_pct / 100))
            top_k_labels = sorted_labels[:k_frac]
            tp = int(top_k_labels.sum())
            if n_positive > 0:
                hit_rate = tp / n_positive
                expected = frac_pct / 100
                results[f'nef@{frac_pct}pct'] = float(hit_rate / expected) if expected > 0 else 0.0
                results[f'hits@{frac_pct}pct'] = tp
            else:
                results[f'nef@{frac_pct}pct'] = 0.0
                results[f'hits@{frac_pct}pct'] = 0

        # === Rank statistics for positives ===
        if n_positive > 0:
            positive_ranks = []
            for rank_pos, label in enumerate(sorted_labels):
                if label == 1:  # positive
                    positive_ranks.append(rank_pos + 1)  # 1-indexed

            results['first_hit_rank'] = int(positive_ranks[0]) if positive_ranks else n_total
            results['median_hit_rank'] = int(np.median(positive_ranks)) if positive_ranks else n_total
            results['mean_hit_rank'] = float(np.mean(positive_ranks)) if positive_ranks else float(n_total)
            results['last_hit_rank'] = int(positive_ranks[-1]) if positive_ranks else n_total
            results['screen_all_frac'] = float(positive_ranks[-1] / n_total) if positive_ranks else 1.0
            results['mrr'] = float(np.mean([1.0 / r for r in positive_ranks])) if positive_ranks else 0.0
        else:
            results['first_hit_rank'] = n_total
            results['median_hit_rank'] = n_total
            results['mean_hit_rank'] = float(n_total)
            results['last_hit_rank'] = n_total
            results['screen_all_frac'] = 1.0
            results['mrr'] = 0.0

        # === AUC of recall curve ===
        if n_positive > 0:
            cumulative_hits = np.cumsum(sorted_labels)
            recall_curve = cumulative_hits / n_positive
            results['auc_recall'] = float(np.mean(recall_curve))  # random=0.5, perfect~1.0
        else:
            results['auc_recall'] = 0.0

        # === Precision/Recall for class 0 (thresholded) ===
        # "class 0 correct / class 0 total guessed"
        guessed_pos = (preds < self.threshold)
        truly_pos = (targets < self.threshold)
        n_guessed = int(guessed_pos.sum())
        n_correct = int((guessed_pos & truly_pos).sum())
        results['precision_class0'] = float(n_correct / n_guessed) if n_guessed > 0 else 0.0
        results['recall_class0'] = float(n_correct / n_positive) if n_positive > 0 else 0.0
        results['n_guessed_class0'] = n_guessed

        return results
    
    def get_top_k_predictions(self, k: int = 100) -> List[Tuple[str, float, float]]:
        """Return top-K predictions: (cif_id, predicted_bandgap, true_bandgap)."""
        if len(self.preds) == 0 or len(self.cif_ids) == 0:
            return []
        
        preds = np.array(self.preds)
        targets = np.array(self.targets)
        
        sorted_idx = np.argsort(preds)[:k]
        
        return [
            (self.cif_ids[i], self.preds[i], self.targets[i])
            for i in sorted_idx
        ]
    
    def get_failure_analysis(self) -> Dict:
        """
        Analyze failure modes for regression.
        
        Failures for ranking:
        - Missed positives: true bandgap < threshold but predicted high (ranked low)
        - False alarms: true bandgap >= threshold but predicted low (ranked high)
        - Prediction error analysis: how far off are we?
        """
        if len(self.preds) == 0:
            return {}
        
        preds = np.array(self.preds)
        targets = np.array(self.targets)
        errors = preds - targets  # positive = over-predicted, negative = under-predicted
        
        # Sort by predicted (what we'd rank by)
        sorted_idx = np.argsort(preds)
        
        # True positives and negatives
        is_positive = targets < self.threshold
        is_negative = ~is_positive
        
        failures = {
            'missed_positives': [],     # True positive ranked too low
            'false_alarms': [],         # True negative ranked too high
            'worst_underpredictions': [],  # Predicted much lower than actual
            'worst_overpredictions': [],   # Predicted much higher than actual
        }
        
        # Get rank for each sample
        ranks = np.zeros(len(preds), dtype=int)
        ranks[sorted_idx] = np.arange(len(preds))
        
        # Analyze failures
        for i in range(len(preds)):
            cif_id = self.cif_ids[i] if i < len(self.cif_ids) else f"sample_{i}"
            info = {
                'cif_id': cif_id,
                'true_bandgap': float(targets[i]),
                'pred_bandgap': float(preds[i]),
                'error': float(errors[i]),
                'rank': int(ranks[i])
            }
            
            # Missed positives: true positive but ranked in bottom half
            if is_positive[i] and ranks[i] > len(preds) // 2:
                failures['missed_positives'].append(info)
            
            # False alarms: true negative but ranked in top 100
            if is_negative[i] and ranks[i] < 100:
                failures['false_alarms'].append(info)
        
        # Worst prediction errors
        error_sorted = np.argsort(errors)
        
        # Worst underpredictions (predicted too low - error most negative)
        for i in error_sorted[:10]:
            cif_id = self.cif_ids[i] if i < len(self.cif_ids) else f"sample_{i}"
            failures['worst_underpredictions'].append({
                'cif_id': cif_id,
                'true_bandgap': float(targets[i]),
                'pred_bandgap': float(preds[i]),
                'error': float(errors[i])
            })
        
        # Worst overpredictions (predicted too high - error most positive)
        for i in error_sorted[-10:][::-1]:
            cif_id = self.cif_ids[i] if i < len(self.cif_ids) else f"sample_{i}"
            failures['worst_overpredictions'].append({
                'cif_id': cif_id,
                'true_bandgap': float(targets[i]),
                'pred_bandgap': float(preds[i]),
                'error': float(errors[i])
            })
        
        # Summary stats
        failures['summary'] = {
            'n_missed_positives': len(failures['missed_positives']),
            'n_false_alarms': len(failures['false_alarms']),
            'mean_abs_error': float(np.abs(errors).mean()),
            'mean_error_positives': float(errors[is_positive].mean()) if is_positive.any() else 0,
            'mean_error_negatives': float(errors[is_negative].mean()) if is_negative.any() else 0,
        }
        
        return failures

    # -----------------------------------------------------------------
    # _compute_ranking_core  (ranking metrics on an arbitrary subset)
    # -----------------------------------------------------------------
    def _compute_ranking_core(self, preds_sub: np.ndarray,
                              targets_sub: np.ndarray) -> Dict[str, float]:
        """Compute ranking-based discovery metrics on a subset of predictions."""
        labels = (targets_sub < self.threshold).astype(int)
        sorted_idx = np.argsort(preds_sub)  # ascending predicted bandgap
        sorted_labels = labels[sorted_idx]
        n_total = len(labels)
        n_positive = int(labels.sum())
        prev = n_positive / n_total if n_total > 0 else 0

        results: Dict[str, float] = {
            'n_total': n_total,
            'n_positive': n_positive,
            'baseline_prevalence': float(prev),
        }

        for k in [25, 50, 100, 200, 500]:
            k_actual = min(k, n_total)
            tp = int(sorted_labels[:k_actual].sum())
            recall_k = tp / n_positive if n_positive > 0 else 0
            precision_k = tp / k_actual
            enrichment_k = precision_k / prev if prev > 0 else 0
            results[f'recall@{k}'] = float(recall_k)
            results[f'precision@{k}'] = float(precision_k)
            results[f'enrichment@{k}'] = float(enrichment_k)

        for frac_pct in [1, 2, 5, 10]:
            k_frac = max(1, int(n_total * frac_pct / 100))
            tp = int(sorted_labels[:k_frac].sum())
            if n_positive > 0:
                results[f'nef@{frac_pct}pct'] = float((tp / n_positive) / (frac_pct / 100))
                results[f'hits@{frac_pct}pct'] = tp
            else:
                results[f'nef@{frac_pct}pct'] = 0.0
                results[f'hits@{frac_pct}pct'] = 0

        if n_positive > 0:
            positive_ranks = [r + 1 for r, v in enumerate(sorted_labels) if v == 1]
            results['first_hit_rank'] = int(positive_ranks[0]) if positive_ranks else n_total
            results['median_hit_rank'] = int(np.median(positive_ranks)) if positive_ranks else n_total
            results['mean_hit_rank'] = float(np.mean(positive_ranks)) if positive_ranks else float(n_total)
            results['last_hit_rank'] = int(positive_ranks[-1]) if positive_ranks else n_total
            results['mrr'] = float(np.mean([1.0 / r for r in positive_ranks])) if positive_ranks else 0.0
            cumulative = np.cumsum(sorted_labels)
            results['auc_recall'] = float(np.mean(cumulative / n_positive))
        else:
            results['first_hit_rank'] = n_total
            results['median_hit_rank'] = n_total
            results['mean_hit_rank'] = float(n_total)
            results['last_hit_rank'] = n_total
            results['mrr'] = 0.0
            results['auc_recall'] = 0.0

        if len(preds_sub) > 1:
            sp, _ = spearmanr(targets_sub, preds_sub)
            results['spearman_rho'] = float(sp) if not np.isnan(sp) else 0.0
        else:
            results['spearman_rho'] = 0.0

        return results

    # -----------------------------------------------------------------
    # evaluate_subsampled
    # -----------------------------------------------------------------
    def evaluate_subsampled(self, n_subsample: int = 1500,
                            n_resamples: int = 30, seed: int = 42) -> Dict:
        """Subsampled evaluation: keep all positives, sample negatives to n_subsample."""
        if len(self.preds) == 0:
            return {}

        preds = np.array(self.preds)
        targets = np.array(self.targets)
        labels = (targets < self.threshold).astype(int)

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_pos = len(pos_idx)
        if n_pos == 0 or n_subsample <= n_pos:
            return {}

        n_neg_sample = min(n_subsample - n_pos, len(neg_idx))
        rng = np.random.RandomState(seed)

        all_results = []
        for _ in range(n_resamples):
            neg_sample = rng.choice(neg_idx, n_neg_sample, replace=False)
            idx = np.concatenate([pos_idx, neg_sample])
            rng.shuffle(idx)
            metrics = self._compute_ranking_core(preds[idx], targets[idx])
            all_results.append(metrics)

        summary: Dict = {
            'n_subsample': n_subsample,
            'n_resamples': n_resamples,
            'n_positives_per_resample': n_pos,
            'subsample_prevalence': float(n_pos / n_subsample),
        }
        if all_results:
            all_keys = set()
            for r in all_results:
                all_keys.update(r.keys())
            for k in sorted(all_keys):
                vals = [float(r[k]) for r in all_results
                        if isinstance(r.get(k), (int, float))
                        and not np.isnan(float(r.get(k, 0)))]
                if vals:
                    summary[f'{k}_mean'] = float(np.mean(vals))
                    summary[f'{k}_std'] = float(np.std(vals))
                    summary[f'{k}_min'] = float(np.min(vals))
                    summary[f'{k}_max'] = float(np.max(vals))
        return summary

    # -----------------------------------------------------------------
    # evaluate_mini_splits
    # -----------------------------------------------------------------
    def evaluate_mini_splits(self, n_splits: int = 5, seed: int = 42) -> Dict:
        """Disjoint mini-split evaluation (needle-in-haystack).

        Partitions negatives into n_splits disjoint groups, adds ALL positives
        to each. Reports per-split metrics + mean/std aggregates.
        """
        if len(self.preds) == 0:
            return {}

        preds = np.array(self.preds)
        targets = np.array(self.targets)
        labels = (targets < self.threshold).astype(int)

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            return {}

        rng = np.random.RandomState(seed)
        shuffled_neg = neg_idx.copy()
        rng.shuffle(shuffled_neg)
        neg_groups = np.array_split(shuffled_neg, n_splits)

        per_split_results = []
        per_split_details = []
        for split_i, neg_group in enumerate(neg_groups):
            idx = np.concatenate([pos_idx, neg_group])
            metrics = self._compute_ranking_core(preds[idx], targets[idx])
            per_split_results.append(metrics)
            per_split_details.append({
                'split_idx': split_i,
                'n_samples': len(idx),
                'n_positives': n_pos,
                'n_negatives': len(neg_group),
                'prevalence': float(n_pos / len(idx)),
            })

        summary: Dict = {
            'n_splits': n_splits,
            'n_positives': n_pos,
            'n_total_test': len(labels),
            'n_negatives_total': n_neg,
        }
        for i, (detail, metrics) in enumerate(zip(per_split_details, per_split_results)):
            summary[f'split_{i}_n_samples'] = detail['n_samples']
            summary[f'split_{i}_prevalence'] = detail['prevalence']
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    summary[f'split_{i}_{k}'] = float(v)

        if per_split_results:
            all_keys = set()
            for r in per_split_results:
                all_keys.update(r.keys())
            for k in sorted(all_keys):
                vals = [float(r[k]) for r in per_split_results
                        if isinstance(r.get(k), (int, float))
                        and not np.isnan(float(r.get(k, 0)))]
                if vals:
                    summary[f'{k}_mean'] = float(np.mean(vals))
                    summary[f'{k}_std'] = float(np.std(vals))
                    summary[f'{k}_min'] = float(np.min(vals))
                    summary[f'{k}_max'] = float(np.max(vals))
        return summary


# =============================================================================
# MODEL: Regressor with Layer Freezing
# =============================================================================

class MOFRegressor(Module):
    """
    MOF Bandgap Regressor with Layer Freezing support.
    
    Predicts continuous bandgap values.
    Evaluated using ranking metrics (top-K recall).
    Supports CLS token or mean pooling over atom tokens.
    """
    
    def __init__(self, config):
        # Override loss_names to use regression
        config["loss_names"] = {
            "ggm": 0,
            "mpp": 0,
            "mtp": 0,
            "vfp": 0,
            "moc": 0,
            "bbc": 0,
            "regression": 1,  # Our task
            "classification": 0,
        }
        
        load_path = config.get("load_path", "")
        if load_path:
            print(f"[MOFRegressor] Loading pretrained weights from: {load_path}")
        
        super().__init__(config)
        
        print(f"[MOFRegressor] Model initialized with {len(self.transformer.blocks)} transformer blocks")
        
        self.freeze_config = config.get("freeze_layers", 0)
        self.threshold = config.get("threshold", 1.0)
        self.loss_type = config.get("loss_type", "huber")
        self.use_sample_weights = config.get("use_sample_weights", True)
        self.pooling_type = config.get("pooling_type", "cls")
        self.dropout_rate = config.get("dropout", 0.0)
        
        hid_dim = config.get("hid_dim", 768)
        
        # Replace default regression_head with a deeper one if dropout > 0
        if self.dropout_rate > 0:
            self.regression_head = nn.Sequential(
                nn.Linear(hid_dim, hid_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hid_dim // 2, hid_dim // 4),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hid_dim // 4, 1),
            )
            print(f"[MOFRegressor] Custom deep regression head (dropout={self.dropout_rate})")
        
        print(f"[MOFRegressor] Pooling: {self.pooling_type}")
        
        # Load sample weights
        self.sample_weights = {}
        if self.use_sample_weights:
            data_dir = config.get("data_dir", "")
            downstream = config.get("downstream", "bandgaps")
            if data_dir:
                weight_path = os.path.join(data_dir, f"train_{downstream}_weights.json")
                if os.path.exists(weight_path):
                    with open(weight_path, 'r') as f:
                        self.sample_weights = json.load(f)
                    print(f"[MOFRegressor] Loaded sample weights for {len(self.sample_weights)} MOFs")
        
        # Ranking metrics
        self.val_metrics = RankingMetrics(k_values=[25, 50, 100, 200], threshold=self.threshold)
        self.test_metrics = RankingMetrics(k_values=[25, 50, 100, 200, 500], threshold=self.threshold)
        
        # Storage
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        self.test_cif_ids = []
    
    def forward_features(self, batch):
        """Extract features with CLS or mean pooling."""
        output = self.infer(batch)
        
        if self.pooling_type == "mean":
            graph_feats = output["graph_feats"]    # [B, max_graph_len+1, hid_dim]
            graph_masks = output["graph_masks"]    # [B, max_graph_len+1]
            masked = graph_feats * graph_masks.unsqueeze(-1)
            pooled = masked.sum(dim=1) / graph_masks.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = output["cls_feats"]           # [B, hid_dim]
        
        return pooled, output
    
    def freeze_layers(self, freeze_mode: Union[int, str], logger: Optional[logging.Logger] = None):
        """Freeze backbone layers for transfer learning."""
        log = logger.info if logger else print
        
        n_blocks = len(self.transformer.blocks)
        
        # First, make everything trainable
        for param in self.parameters():
            param.requires_grad = True
        
        if freeze_mode == 0:
            log(f"[Freeze] Mode 0: All {n_blocks} blocks TRAINABLE (full fine-tuning)")
            return
        
        if freeze_mode == "all" or freeze_mode == n_blocks:
            # Freeze everything except regression head
            for name, param in self.named_parameters():
                if "regression_head" not in name:
                    param.requires_grad = False
            
            log(f"[Freeze] Mode 'all': Only regression head TRAINABLE (linear probing)")
            return
        
        if isinstance(freeze_mode, int) and 0 < freeze_mode < n_blocks:
            n_trainable = freeze_mode
            n_frozen = n_blocks - n_trainable
            
            # Freeze embeddings
            for name, param in self.named_parameters():
                if "transformer.blocks" not in name and "regression_head" not in name:
                    param.requires_grad = False
            
            # Freeze first n_frozen blocks
            for i in range(n_frozen):
                for param in self.transformer.blocks[i].parameters():
                    param.requires_grad = False
            
            log(f"[Freeze] Blocks 0-{n_frozen-1} FROZEN, blocks {n_frozen}-{n_blocks-1} TRAINABLE")
    
    def compute_regression_loss(self, preds, targets, cif_ids=None):
        """Compute weighted regression loss."""
        # Get sample weights
        if self.use_sample_weights and cif_ids and self.sample_weights:
            weights = torch.tensor([
                self.sample_weights.get(cif_id, 1.0) for cif_id in cif_ids
            ], device=preds.device, dtype=preds.dtype)
            
            # Debug: print first batch info
            if not hasattr(self, '_debug_logged'):
                self._debug_logged = True
                n_found = sum(1 for c in cif_ids if c in self.sample_weights)
                n_total = len(cif_ids)
                print(f"\n[DEBUG] Sample weights lookup: {n_found}/{n_total} MOFs found in weights dict")
                print(f"[DEBUG] First 3 cif_ids: {cif_ids[:3]}")
                print(f"[DEBUG] First 3 weights: {weights[:3].tolist()}")
                print(f"[DEBUG] Weight range in batch: [{weights.min():.3f}, {weights.max():.3f}]")
                print(f"[DEBUG] Total MOFs in weights dict: {len(self.sample_weights)}\n")
        else:
            weights = torch.ones_like(preds)
        
        if self.loss_type == "mse":
            loss = F.mse_loss(preds, targets, reduction='none')
        elif self.loss_type == "huber":
            loss = F.huber_loss(preds, targets, reduction='none', delta=1.0)
        else:
            loss = F.mse_loss(preds, targets, reduction='none')
        
        # Apply weights
        weighted_loss = (loss.squeeze() * weights).mean()
        return weighted_loss
    
    def training_step(self, batch, batch_idx):
        """Training step with weighted loss."""
        pooled, output = self.forward_features(batch)
        
        preds = self.regression_head(pooled).squeeze(-1)
        targets = torch.tensor(batch["target"], device=preds.device, dtype=torch.float).squeeze(-1)
        cif_ids = output.get("cif_id", None)
        
        loss = self.compute_regression_loss(preds, targets, cif_ids)
        
        self.log("train/loss", loss, prog_bar=True, batch_size=targets.numel())
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - collect predictions for ranking."""
        pooled, output = self.forward_features(batch)
        
        preds = self.regression_head(pooled).squeeze(-1)
        targets = torch.tensor(batch["target"], device=preds.device, dtype=torch.float).squeeze(-1)
        
        loss = F.huber_loss(preds, targets)
        self.log("val/loss", loss, prog_bar=True, batch_size=targets.numel())
        
        self.val_preds.append(preds.detach())
        self.val_targets.append(targets.detach())
        
        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        """Compute ranking metrics at epoch end."""
        if not self.val_preds:
            return
        
        all_preds = torch.cat(self.val_preds)
        all_targets = torch.cat(self.val_targets)
        
        self.val_metrics.reset()
        self.val_metrics.update(all_preds, all_targets)
        metrics = self.val_metrics.compute()
        
        # Log metrics
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith(('recall', 'precision', 'enrichment', 'mae', 'rmse',
                             'spearman', 'pearson', 'kendall', 'r2',
                             'auc_recall', 'first_hit', 'median_hit', 'mean_hit',
                             'last_hit', 'mrr', 'nef', 'hits', 'screen',
                             'n_positive', 'n_total', 'prevalence', 'n_guessed')):
                self.log(f"val/{k}", float(v), prog_bar=False)
        
        # Print report
        self._print_ranking_report("VALIDATION", metrics)
        
        # Clear storage
        self.val_preds = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        pooled, output = self.forward_features(batch)
        
        preds = self.regression_head(pooled).squeeze(-1)
        targets = torch.tensor(batch["target"], device=preds.device, dtype=torch.float).squeeze(-1)
        cif_ids = output.get("cif_id", output.get("name", None))
        
        self.test_preds.append(preds.detach())
        self.test_targets.append(targets.detach())
        if cif_ids:
            if isinstance(cif_ids, (list, tuple)):
                self.test_cif_ids.extend(cif_ids)
            else:
                self.test_cif_ids.append(cif_ids)
    
    def on_test_epoch_end(self):
        """Compute final test ranking metrics."""
        if not self.test_preds:
            return
        
        all_preds = torch.cat(self.test_preds)
        all_targets = torch.cat(self.test_targets)
        
        self.test_metrics.reset()
        self.test_metrics.update(all_preds, all_targets, self.test_cif_ids if self.test_cif_ids else None)
        metrics = self.test_metrics.compute()
        
        self._print_ranking_report("TEST", metrics, verbose=True)
        
        # Log for checkpointing
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log(f"test/{k}", float(v))
    
    def _print_ranking_report(self, split: str, metrics: Dict, verbose: bool = False):
        """Print ranking metrics report with discovery screening details."""
        n_total = metrics.get('n_total', 0)
        n_positive = metrics.get('n_positive', 0)
        prevalence = metrics.get('baseline_prevalence', 0)
        
        print(f"\n{'=' * 70}")
        print(f"{split} RANKING REPORT (Epoch {self.current_epoch})")
        print(f"{'=' * 70}")
        print(f"Total samples: {n_total}")
        print(f"True <{self.threshold}eV MOFs: {n_positive} ({prevalence:.4%})")

        print(f"\n--- Regression Quality ---")
        print(f"MAE: {metrics.get('mae', 0):.4f} eV | RMSE: {metrics.get('rmse', 0):.4f} eV | R²: {metrics.get('r2', 0):.4f}")
        print(f"Pearson r: {metrics.get('pearson_r', 0):.4f} | "
              f"Spearman ρ: {metrics.get('spearman_rho', 0):.4f} | "
              f"Kendall τ: {metrics.get('kendall_tau', 0):.4f}")

        print(f"\n--- Discovery Screening ---")
        print(f"{'K':>6s} {'Recall':>8s} {'Precision':>10s} {'EF':>8s}  {'Hits':>5s}")
        for k in [50, 100, 200, 500]:
            if f'recall@{k}' in metrics:
                recall = metrics[f'recall@{k}']
                prec = metrics.get(f'precision@{k}', 0)
                ef = metrics.get(f'enrichment@{k}', 0)
                tp = int(recall * n_positive)
                print(f"  {k:4d}  {recall:8.4f}  {prec:10.4f}  {ef:7.2f}x  {tp:4d}/{n_positive}")

        if verbose:
            print(f"\n{'Frac%':>6s} {'NEF':>8s} {'Hits':>6s}  (Normalized Enrichment Factor)")
            for frac_pct in [1, 2, 5, 10]:
                nef = metrics.get(f'nef@{frac_pct}pct', 0)
                hits = metrics.get(f'hits@{frac_pct}pct', 0)
                k_frac = max(1, int(n_total * frac_pct / 100))
                print(f"  {frac_pct:3d}%  {nef:8.2f}  {hits:6d}  (top {k_frac} of {n_total})")

            print(f"\n--- Rank Statistics ---")
            print(f"First hit rank:  {metrics.get('first_hit_rank', '?')}")
            print(f"Median hit rank: {metrics.get('median_hit_rank', '?')}")
            print(f"Mean hit rank:   {metrics.get('mean_hit_rank', 0):.1f}")
            print(f"Last hit rank:   {metrics.get('last_hit_rank', '?')}")
            print(f"Screen to find all: {metrics.get('screen_all_frac', 1):.2%} of library")
            print(f"MRR: {metrics.get('mrr', 0):.6f}")
            print(f"AUC recall: {metrics.get('auc_recall', 0):.4f} (random=0.5)")
        print(f"{'=' * 70}\n")


# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run(
    data_dir: str,
    downstream: str = "bandgaps_regression",
    log_dir: str = "./logs",
    threshold: float = 1.0,
    freeze_layers: Union[int, str] = 2,
    use_sample_weights: bool = True,
    loss_type: str = "huber",
    pooling_type: str = "cls",
    dropout: float = 0.0,
    batch_size: int = 32,
    per_gpu_batchsize: int = 16,  # Increased from 8
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,   # Increased regularization
    lr_mult: float = 10.0,
    max_epochs: int = 100,
    patience: int = 15,
    gradient_clip_val: float = 1.0,  # Gradient clipping
    seed: int = 42,
    num_workers: int = 4,
    es_monitor: str = "val/recall@100",
    es_mode: str = "max",
    **kwargs
):
    """
    Main training function for MOF bandgap regression with ranking evaluation.
    
    Enhanced with:
    - CLS token or mean pooling over atom tokens
    - Optional deep regression head with dropout
    - Spearman/Kendall correlation metrics
    - R² score
    - Training dashboards and learning curve plots
    - Gradient tracking and clipping
    - Failure mode analysis
    """
    pl.seed_everything(seed)
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(log_dir)
    
    logger.info("=" * 70)
    logger.info("MOF BANDGAP REGRESSION (RANKING EVALUATION)")
    logger.info("=" * 70)
    logger.info(f"Goal: Rank MOFs by predicted bandgap, evaluate top-K recall")
    logger.info(f"Threshold for 'positive': {threshold} eV")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Downstream: {downstream}")
    logger.info(f"Loss type: {loss_type}")
    logger.info(f"Pooling: {pooling_type} | Dropout: {dropout}")
    logger.info(f"Batch size: {per_gpu_batchsize}")
    logger.info(f"Gradient clipping: {gradient_clip_val}")
    logger.info("")
    
    # Create sample weights if needed
    if use_sample_weights:
        _create_regression_weights(data_dir, downstream, threshold, logger)
    
    # Build config
    config = default_config_fn()
    config = json.loads(json.dumps(config))
    
    config.update(kwargs)
    
    config["loss_names"] = {
        "ggm": 0, "mpp": 0, "mtp": 0, "vfp": 0, "moc": 0, "bbc": 0,
        "regression": 1, "classification": 0,
    }
    config["log_dir"] = log_dir
    config["data_dir"] = data_dir
    config["downstream"] = downstream
    config["threshold"] = threshold
    config["freeze_layers"] = freeze_layers
    config["use_sample_weights"] = use_sample_weights
    config["loss_type"] = loss_type
    config["pooling_type"] = pooling_type
    config["dropout"] = dropout
    config["batch_size"] = batch_size
    config["per_gpu_batchsize"] = per_gpu_batchsize
    config["learning_rate"] = learning_rate
    config["weight_decay"] = weight_decay
    config["lr_mult"] = lr_mult
    config["max_epochs"] = max_epochs
    config["gradient_clip_val"] = gradient_clip_val
    
    # Ensure load_path is set for pretrained weights
    if "load_path" not in config or not config["load_path"]:
        config["load_path"] = "pmtransformer"
    
    # Validate config
    config = get_valid_config(config)
    
    # Create model
    logger.info("Creating MOFRegressor model...")
    model = MOFRegressor(config)
    model.freeze_layers(freeze_layers, logger)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
    
    # Setup data - use Dataset directly like classifier does
    logger.info("Setting up data loaders...")
    from torch.utils.data import DataLoader
    
    train_ds = Dataset(
        data_dir,
        split="train",
        downstream=downstream,
        nbr_fea_len=config["nbr_fea_len"],
        draw_false_grid=False,
    )
    
    val_ds = Dataset(
        data_dir,
        split="val",
        downstream=downstream,
        nbr_fea_len=config["nbr_fea_len"],
        draw_false_grid=False,
    )
    
    physical_batch_size = config["per_gpu_batchsize"]
    logger.info(f"  Train samples: {len(train_ds)}")
    logger.info(f"  Val samples: {len(val_ds)}")
    logger.info(f"  Physical batch size: {physical_batch_size}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: Dataset.collate(x, config["img_size"]),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=physical_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: Dataset.collate(x, config["img_size"]),
        pin_memory=True,
    )
    
    # Test loader
    test_loader = None
    try:
        test_ds = Dataset(
            data_dir,
            split="test",
            downstream=downstream,
            nbr_fea_len=config["nbr_fea_len"],
            draw_false_grid=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=physical_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: Dataset.collate(x, config["img_size"]),
            pin_memory=True,
        )
        logger.info(f"  Test samples: {len(test_ds)}")
    except Exception as e:
        logger.info(f"  Test set not found: {e}")
    
    # Callbacks
    logger.info(f"Early stopping: monitor={es_monitor}, mode={es_mode}, patience={patience}")
    
    # Multiple checkpoints (matching discovery/multiclass pattern)
    checkpoint_metrics = [
        (es_monitor, es_mode, "best_es"),
        ("val/precision_class0", "max", "best_prec_c0"),
        ("val/auc_recall", "max", "best_auc_recall"),
        ("val/loss", "min", "best_loss"),
    ]
    checkpoint_callbacks = []
    for metric_name, metric_mode, ckpt_name in checkpoint_metrics:
        cb = ModelCheckpoint(
            dirpath=log_dir,
            filename=f"{ckpt_name}-{{epoch:02d}}",
            monitor=metric_name,
            mode=metric_mode,
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
            verbose=True,
        )
        checkpoint_callbacks.append(cb)
        logger.info(f"  Checkpoint: {ckpt_name} -> monitor={metric_name} ({metric_mode})")

    # Also save last
    last_cb = ModelCheckpoint(
        dirpath=log_dir, filename="last", save_last=True, every_n_epochs=1,
    )
    checkpoint_callbacks.append(last_cb)

    early_stop = EarlyStopping(
        monitor=es_monitor,
        patience=patience,
        mode=es_mode,
        verbose=True,
        min_delta=0.001,
    )
    
    # Gradient tracking callback
    gradient_tracker = GradientTracker(log_dir, log_every_n_steps=50)
    
    # Trainer with gradient clipping
    num_devices = get_num_devices(config)
    gradient_clip = config.get("gradient_clip_val", 1.0)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=num_devices,
        callbacks=checkpoint_callbacks + [early_stop, gradient_tracker],
        default_root_dir=log_dir,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 70)
    
    # Load early-stopping best (Spearman when es_monitor='val/spearman_rho'); used for final test and test_predictions.csv
    best_path = checkpoint_callbacks[0].best_model_path
    if best_path:
        logger.info(f"Loading best model (Spearman checkpoint): {best_path}")
        model = MOFRegressor.load_from_checkpoint(best_path, config=config)
    
    if test_loader:
        trainer.test(model, test_loader)
        
        # Get detailed test results and save
        metrics = model.test_metrics.compute()
        logger.info("\nTEST RESULTS (Ranking Metrics):")
        logger.info("-" * 40)
        logger.info(f"  Total samples: {metrics.get('n_total', 0)}")
        logger.info(f"  Positive samples (<{threshold}eV): {metrics.get('n_positive', 0)}")
        logger.info(f"  Baseline prevalence: {metrics.get('baseline_prevalence', 0):.1%}")
        logger.info("")
        for k in [25, 50, 100, 200, 500]:
            if f'recall@{k}' in metrics:
                logger.info(f"  Recall@{k}: {metrics.get(f'recall@{k}', 0):.1%}")
                logger.info(f"  Precision@{k}: {metrics.get(f'precision@{k}', 0):.1%}")
                logger.info(f"  Enrichment@{k}: {metrics.get(f'enrichment@{k}', 0):.2f}x")
                logger.info("")
        logger.info(f"  MAE: {metrics.get('mae', 0):.4f} eV")
        logger.info(f"  RMSE: {metrics.get('rmse', 0):.4f} eV")
        logger.info(f"  R²: {metrics.get('r2', 0):.4f}")
        logger.info(f"  Pearson r: {metrics.get('pearson_r', 0):.4f}")
        logger.info(f"  Spearman ρ: {metrics.get('spearman_rho', 0):.4f}")
        logger.info(f"  Kendall τ: {metrics.get('kendall_tau', 0):.4f}")
        logger.info("-" * 40)
        
        # Failure analysis
        failures = model.test_metrics.get_failure_analysis()
        if failures:
            logger.info("\nFAILURE ANALYSIS:")
            logger.info(f"  Missed positives (true <{threshold}eV, ranked low): {failures['summary']['n_missed_positives']}")
            logger.info(f"  False alarms (true >={threshold}eV, in top 100): {failures['summary']['n_false_alarms']}")
            logger.info(f"  Mean error on positives: {failures['summary']['mean_error_positives']:.4f} eV")
            logger.info(f"  Mean error on negatives: {failures['summary']['mean_error_negatives']:.4f} eV")
            
            if failures['worst_overpredictions']:
                logger.info("\n  Top 5 worst over-predictions (predicted too high):")
                for f in failures['worst_overpredictions'][:5]:
                    logger.info(f"    {f['cif_id']}: true={f['true_bandgap']:.3f}, pred={f['pred_bandgap']:.3f}, err={f['error']:.3f}")
            
            if failures['worst_underpredictions']:
                logger.info("\n  Top 5 worst under-predictions (predicted too low):")
                for f in failures['worst_underpredictions'][:5]:
                    logger.info(f"    {f['cif_id']}: true={f['true_bandgap']:.3f}, pred={f['pred_bandgap']:.3f}, err={f['error']:.3f}")
        
        # Generate test scatter plot (predicted vs true bandgap)
        _plot_test_scatter(model.test_metrics, threshold, log_dir)
        logger.info(f"Test scatter plot saved to: {os.path.join(log_dir, 'test_scatter.png')}")

        # Generate discovery recall curve
        _plot_discovery_curve_regression(model.test_metrics, threshold, log_dir)

        # Generate bootstrap CI / metric summary bar chart
        _plot_bootstrap_ci_regression(metrics, log_dir)

        # === Save per-sample test predictions for ensemble/voting ===
        try:
            _preds = np.array(model.test_metrics.preds)
            _targets = np.array(model.test_metrics.targets)
            _cids_flat = []
            for _batch in model.test_metrics.cif_ids:
                if isinstance(_batch, (list, tuple)):
                    _cids_flat.extend(_batch)
                else:
                    _cids_flat.append(_batch)
            _pred_lines = ["cif_id,score,predicted_binary,true_label,mode"]
            for _i in range(len(_preds)):
                _cid = _cids_flat[_i] if _i < len(_cids_flat) else f"sample_{_i}"
                _sc = float(_preds[_i])
                _pb = 1 if _sc < threshold else 0
                _tl = float(_targets[_i])
                _pred_lines.append(f"{_cid},{_sc:.6f},{_pb},{_tl},regression")
            _pred_path = os.path.join(log_dir, 'test_predictions.csv')
            with open(_pred_path, 'w', encoding='utf-8') as _f:
                _f.write("\n".join(_pred_lines) + "\n")
            logger.info(f"  Saved {len(_preds)} test predictions to test_predictions.csv")
        except Exception as _e:
            logger.info(f"  WARNING: Could not save test predictions: {_e}")

        # === Subsampled + Mini-split evaluation ===
        subsampled_results = {}
        mini_split_results = {}

        logger.info("\nRunning subsampled test evaluation (1500 samples, 30 resamples)...")
        subsampled_results = model.test_metrics.evaluate_subsampled(
            n_subsample=1500, n_resamples=30, seed=seed
        )
        if subsampled_results:
            n_sub = subsampled_results.get('n_subsample', 1500)
            n_pos_sub = subsampled_results.get('n_positives_per_resample', 0)
            prev_sub = subsampled_results.get('subsample_prevalence', 0)
            logger.info(f"  Subsampled: {n_pos_sub} positives in {n_sub} "
                        f"(prevalence={prev_sub:.2%}, 30 resamples)")
            logger.info(f"\n  {'Metric':<25s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
            logger.info(f"  {'-'*65}")
            for key_name in ['recall@25', 'recall@50', 'recall@100', 'recall@200',
                             'enrichment@25', 'enrichment@50', 'enrichment@100',
                             'nef@1pct', 'nef@5pct', 'spearman_rho', 'auc_recall',
                             'first_hit_rank', 'mrr']:
                mean_key = f'{key_name}_mean'
                if mean_key in subsampled_results:
                    m = subsampled_results[mean_key]
                    s = subsampled_results.get(f'{key_name}_std', 0)
                    mn = subsampled_results.get(f'{key_name}_min', m)
                    mx = subsampled_results.get(f'{key_name}_max', m)
                    logger.info(f"  {key_name:<25s} {m:10.4f} {s:10.4f} {mn:10.4f} {mx:10.4f}")
            logger.info("")

        logger.info("\nRunning mini-split evaluation (5 disjoint splits, all positives in each)...")
        mini_split_results = model.test_metrics.evaluate_mini_splits(
            n_splits=5, seed=seed
        )
        if mini_split_results:
            n_sp = mini_split_results.get('n_splits', 5)
            n_pos_ms = mini_split_results.get('n_positives', 0)
            logger.info(f"  {n_sp} disjoint splits, {n_pos_ms} positives in each")
            for si in range(n_sp):
                n_samp = mini_split_results.get(f'split_{si}_n_samples', 0)
                prev_si = mini_split_results.get(f'split_{si}_prevalence', 0)
                auc_si = mini_split_results.get(f'split_{si}_auc_recall', 0)
                r100_si = mini_split_results.get(f'split_{si}_recall@100', 0)
                fhr_si = mini_split_results.get(f'split_{si}_first_hit_rank', 0)
                sp_si = mini_split_results.get(f'split_{si}_spearman_rho', 0)
                logger.info(f"  Split {si}: n={n_samp}, prev={prev_si:.2%}, "
                            f"AUC={auc_si:.4f}, R@100={r100_si:.4f}, "
                            f"FHR={fhr_si}, Spearman={sp_si:.4f}")
            logger.info(f"\n  {'Metric':<25s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
            logger.info(f"  {'-'*65}")
            for key_name in ['recall@25', 'recall@50', 'recall@100', 'recall@200',
                             'enrichment@25', 'enrichment@50', 'enrichment@100',
                             'nef@1pct', 'nef@5pct', 'spearman_rho', 'auc_recall',
                             'first_hit_rank', 'mrr']:
                mean_key = f'{key_name}_mean'
                if mean_key in mini_split_results:
                    m = mini_split_results[mean_key]
                    s = mini_split_results.get(f'{key_name}_std', 0)
                    mn = mini_split_results.get(f'{key_name}_min', m)
                    mx = mini_split_results.get(f'{key_name}_max', m)
                    logger.info(f"  {key_name:<25s} {m:10.4f} {s:10.4f} {mn:10.4f} {mx:10.4f}")
            logger.info("")

        # Save final results to JSON
        results_path = os.path.join(log_dir, "final_results.json")
        final_results = {
            'config': {
                'mode': 'regression',
                'threshold': threshold,
                'freeze_layers': str(freeze_layers),
                'loss_type': loss_type,
                'pooling_type': pooling_type,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'lr_mult': lr_mult,
                'gradient_clip_val': gradient_clip_val,
                'use_sample_weights': use_sample_weights,
                'es_monitor': es_monitor,
                'patience': patience,
                'seed': seed,
            },
            'test_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                            for k, v in metrics.items()},
            'discovery_summary': {
                'n_positives': int(metrics.get('n_positive', 0)),
                'n_total': int(metrics.get('n_total', 0)),
                'prevalence': float(metrics.get('baseline_prevalence', 0)),
                'recall@25': float(metrics.get('recall@25', 0)),
                'recall@50': float(metrics.get('recall@50', 0)),
                'recall@100': float(metrics.get('recall@100', 0)),
                'recall@200': float(metrics.get('recall@200', 0)),
                'recall@500': float(metrics.get('recall@500', 0)),
                'enrichment@25': float(metrics.get('enrichment@25', 0)),
                'enrichment@50': float(metrics.get('enrichment@50', 0)),
                'enrichment@100': float(metrics.get('enrichment@100', 0)),
                'nef@1pct': float(metrics.get('nef@1pct', 0)),
                'nef@5pct': float(metrics.get('nef@5pct', 0)),
                'first_hit_rank': int(metrics.get('first_hit_rank', 0)),
                'median_hit_rank': int(metrics.get('median_hit_rank', 0)),
                'auc_recall': float(metrics.get('auc_recall', 0)),
                'spearman_rho': float(metrics.get('spearman_rho', 0)),
            },
            'subsampled_test': subsampled_results if subsampled_results else {},
            'mini_split_test': mini_split_results if mini_split_results else {},
            'failure_summary': failures.get('summary', {}) if failures else {},
            'training_analysis': {
                'max_gradient_norm': gradient_tracker.max_grad_norm,
                'gradient_explosions': gradient_tracker.grad_explosion_count,
                'epochs_trained': len(gradient_tracker.epoch_train_losses),
            },
            'checkpoints': {
                'best': best_path,
                'last': os.path.join(log_dir, 'last.ckpt'),
            },
        }
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")
    else:
        logger.info("No test set available")
    
    # Always save training history + plots (dashboard, learning curves)
    # even if no test set is available
    history_path = gradient_tracker.save_history()
    logger.info(f"\nTraining history saved to: {history_path}")
    
    logger.info("")
    logger.info(f"Checkpoints saved to: {log_dir}")
    logger.info(f"Best model: {best_path}")
    logger.info(f"Last model: {os.path.join(log_dir, 'last.ckpt')}")
    logger.info("=" * 70)
    logger.info("Training complete!")
    
    return model, trainer


def _plot_test_scatter(test_metrics, threshold, log_dir):
    """Generate predicted vs true bandgap scatter plot with ranking highlights."""
    try:
        if len(test_metrics.preds) == 0:
            return
        preds = np.array(test_metrics.preds)
        targets = np.array(test_metrics.targets)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel 1: Full scatter
        ax = axes[0]
        is_pos = targets < threshold
        ax.scatter(targets[~is_pos], preds[~is_pos], s=8, alpha=0.3, c='blue', label=f'Negative (≥{threshold}eV)')
        ax.scatter(targets[is_pos], preds[is_pos], s=30, alpha=0.8, c='red', marker='*', label=f'Positive (<{threshold}eV)')
        lims = [min(targets.min(), preds.min()) - 0.2, max(targets.max(), preds.max()) + 0.2]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect')
        ax.axhline(threshold, color='orange', ls=':', alpha=0.5)
        ax.axvline(threshold, color='orange', ls=':', alpha=0.5)
        ax.set_xlabel('True Bandgap (eV)'); ax.set_ylabel('Predicted Bandgap (eV)')
        ax.set_title('Predicted vs True Bandgap')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        
        # Panel 2: Ranking curve — cumulative recall as we walk down the ranked list
        ax = axes[1]
        sorted_idx = np.argsort(preds)
        sorted_labels = (targets[sorted_idx] < threshold).astype(int)
        cum_recall = np.cumsum(sorted_labels) / max(sorted_labels.sum(), 1)
        k_values = np.arange(1, len(cum_recall) + 1)
        ax.plot(k_values, cum_recall, color='green', lw=2)
        for k_mark in [50, 100, 200, 500]:
            if k_mark < len(cum_recall):
                ax.axvline(k_mark, color='gray', ls='--', alpha=0.3)
                ax.annotate(f'R@{k_mark}={cum_recall[k_mark-1]:.1%}',
                           (k_mark, cum_recall[k_mark-1]), fontsize=7,
                           textcoords='offset points', xytext=(5, 5))
        ax.set_xlabel('Top-K'); ax.set_ylabel('Cumulative Recall')
        ax.set_title('Discovery Curve: Recall vs Ranking Position')
        ax.set_xlim(0, min(1000, len(cum_recall))); ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'test_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"[Plot] Test scatter error: {e}")


def _plot_discovery_curve_regression(test_metrics, threshold, log_dir):
    """Generate discovery recall curve: fraction screened vs fraction found."""
    try:
        if len(test_metrics.preds) == 0:
            return
        preds = np.array(test_metrics.preds)
        targets = np.array(test_metrics.targets)
        labels = (targets < threshold).astype(int)
        n_positive = labels.sum()
        n_total = len(labels)

        if n_positive == 0:
            return

        sorted_idx = np.argsort(preds)
        sorted_labels = labels[sorted_idx]
        cumulative_hits = np.cumsum(sorted_labels)
        recall_curve = cumulative_hits / n_positive
        frac_screened = np.arange(1, n_total + 1) / n_total

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Full recall curve
        ax = axes[0]
        ax.plot(frac_screened, recall_curve, 'b-', lw=2, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ideal_frac = n_positive / n_total
        ax.plot([0, ideal_frac, 1], [0, 1, 1], 'g--', alpha=0.5, label='Perfect')
        ax.set_xlabel('Fraction of Library Screened')
        ax.set_ylabel('Fraction of Positives Found')
        ax.set_title(f'Discovery Recall Curve (n_pos={n_positive}, n_total={n_total})')
        for frac_pct, color in [(1, 'red'), (5, 'orange'), (10, 'purple')]:
            idx = min(int(n_total * frac_pct / 100), n_total - 1)
            ax.axvline(frac_pct / 100, color=color, ls=':', alpha=0.5)
            ax.plot(frac_pct / 100, recall_curve[idx], 'o', color=color, ms=8,
                    label=f'{frac_pct}%: recall={recall_curve[idx]:.3f}')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

        # Zoomed to top 10%
        ax = axes[1]
        top_idx = max(1, int(n_total * 0.10))
        ax.plot(frac_screened[:top_idx] * 100, recall_curve[:top_idx], 'b-', lw=2, label='Model')
        ax.plot([0, 10], [0, 10 * n_positive / n_total], 'k--', alpha=0.5, label='Random')
        ax.set_xlabel('% of Library Screened')
        ax.set_ylabel('Fraction of Positives Found')
        ax.set_title(f'Zoomed: Top 10% ({top_idx} samples)')
        positive_ranks = np.where(sorted_labels == 1)[0] + 1
        for i, rank in enumerate(positive_ranks):
            if rank <= top_idx:
                ax.plot(rank / n_total * 100, (i + 1) / n_positive, 'r*', ms=12)
                ax.annotate(f'hit#{i+1} r={rank}', (rank/n_total*100, (i+1)/n_positive),
                           textcoords="offset points", xytext=(10, -5), fontsize=7, color='red')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'discovery_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Plot] Discovery curve saved: {os.path.join(log_dir, 'discovery_curve.png')}")
    except Exception as e:
        print(f"[Plot] Discovery curve error: {e}")


def _plot_bootstrap_ci_regression(test_metrics, log_dir):
    """Plot key test metrics as a bar chart for regression (analogous to bootstrap CI)."""
    try:
        metrics = test_metrics if isinstance(test_metrics, dict) else {}
        keys = ['recall@25', 'recall@50', 'recall@100', 'recall@200',
                'enrichment@25', 'enrichment@50', 'enrichment@100',
                'nef@1pct', 'nef@5pct', 'spearman_rho', 'auc_recall']

        vals, labels_list = [], []
        for k in keys:
            if k in metrics:
                vals.append(float(metrics[k]))
                labels_list.append(k.replace('_', ' '))

        if not vals:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(vals))
        ax.bar(x, vals, color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=30, ha='right')
        ax.set_ylabel('Value')
        ax.set_title('Test Metrics Summary (Regression)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_path = os.path.join(log_dir, 'bootstrap_ci.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Plot] Metric summary (bootstrap_ci) saved: {save_path}")
    except Exception as e:
        print(f"[Plot] Bootstrap CI error: {e}")


def _create_regression_weights(data_dir: str, downstream: str, threshold: float, logger):
    """
    Create sample weights for regression with INVERSE FREQUENCY + GRADIENT boosting.
    
    For severely imbalanced data (8.5% positive in train, 0.09% in test):
    
    Strategy:
    1. Inverse frequency: positives get weight = N_total / N_positive (~11.8x)
    2. Gradient boost: within positives, lower bandgap = even higher weight
    3. Negatives: weight = 1.0 (baseline)
    
    This ensures:
    - Model focuses on learning positives (they're rare and important)
    - Very low bandgap MOFs get highest priority
    - Negatives still contribute but don't dominate the loss
    """
    json_path = os.path.join(data_dir, f"train_{downstream}.json")
    weight_path = os.path.join(data_dir, f"train_{downstream}_weights.json")
    
    # ALWAYS recreate weights (remove old logic check)
    # We want to ensure the new weighting strategy is applied
    
    if not os.path.exists(json_path):
        logger.info(f"No training data found at {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Count positives and negatives
    n_total = len(data)
    n_positive = sum(1 for v in data.values() if float(v) < threshold)
    n_negative = n_total - n_positive
    
    # Inverse frequency base weight for positives
    if n_positive > 0:
        inv_freq_weight = n_total / n_positive  # e.g., 685/58 ≈ 11.8
    else:
        inv_freq_weight = 1.0
    
    logger.info(f"Sample weight calculation:")
    logger.info(f"  Total samples: {n_total}")
    logger.info(f"  Positives (< {threshold} eV): {n_positive} ({100*n_positive/n_total:.1f}%)")
    logger.info(f"  Negatives (>= {threshold} eV): {n_negative} ({100*n_negative/n_total:.1f}%)")
    logger.info(f"  Inverse frequency weight for positives: {inv_freq_weight:.2f}x")
    
    weights = {}
    min_bandgap = min(float(v) for v in data.values() if float(v) < threshold) if n_positive > 0 else 0
    
    for mof_name, bandgap in data.items():
        bandgap = float(bandgap)
        if bandgap < threshold:
            # POSITIVE: inverse frequency + gradient boost
            # Base: inv_freq_weight (e.g., 11.8)
            # Gradient: 1.0 (at threshold) to 2.0 (at 0 eV) 
            gradient_boost = 1.0 + (threshold - bandgap) / threshold
            weight = inv_freq_weight * gradient_boost
            # Result: weights range from ~11.8 (near threshold) to ~23.6 (near 0)
        else:
            # NEGATIVE: baseline weight
            weight = 1.0
        
        weights[mof_name] = weight
    
    # Log weight distribution
    pos_weights = [w for m, w in weights.items() if float(data[m]) < threshold]
    neg_weights = [w for m, w in weights.items() if float(data[m]) >= threshold]
    
    if pos_weights:
        logger.info(f"  Positive weight range: [{min(pos_weights):.2f}, {max(pos_weights):.2f}]")
    if neg_weights:
        logger.info(f"  Negative weight range: [{min(neg_weights):.2f}, {max(neg_weights):.2f}]")
    
    with open(weight_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    logger.info(f"Created sample weights: {weight_path}")


# =============================================================================
# K-FOLD CROSS-VALIDATION
# =============================================================================

def run_kfold(
    kfold_dir: str,
    downstream: str = "bandgaps_regression",
    log_dir: str = "logs_kfold/",
    threshold: float = 1.0,
    n_folds: int = 5,
    **kwargs
):
    """
    Run K-Fold cross-validation for MOF bandgap regression with ranking evaluation.
    
    Args:
        kfold_dir: Path to kfold directory containing fold_1/, fold_2/, etc.
                   Each fold directory must have train/val/test split JSON files.
        downstream: Downstream task suffix (default: bandgaps_regression)
        log_dir: Base directory for logs
        threshold: Bandgap threshold for positive class (eV)
        n_folds: Number of folds
        **kwargs: Additional config overrides passed to run()
    
    Returns:
        Dictionary with fold-wise and aggregate metrics, or None if all folds fail.
    """
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION (REGRESSION)")
    print("=" * 70)
    print(f"K-fold directory: {kfold_dir}")
    print(f"Number of folds: {n_folds}")
    print(f"Threshold: {threshold} eV")
    print(f"Extra config: {kwargs}")
    print("=" * 70 + "\n")
    
    all_fold_metrics = []
    
    for fold in range(1, n_folds + 1):
        fold_dir = os.path.join(kfold_dir, f"fold_{fold}")
        fold_log_dir = os.path.join(log_dir, f"fold_{fold}")
        
        if not os.path.exists(fold_dir):
            print(f"WARNING: Fold directory not found: {fold_dir}")
            continue
        
        print("\n" + "#" * 70)
        print(f"# FOLD {fold}/{n_folds}")
        print("#" * 70 + "\n")
        
        try:
            model, trainer = run(
                data_dir=fold_dir,
                downstream=downstream,
                log_dir=fold_log_dir,
                threshold=threshold,
                **kwargs
            )
            
            # Collect validation metrics from last epoch
            fold_metrics = {'fold': fold}
            if hasattr(model, 'val_metrics'):
                try:
                    metrics = model.val_metrics.compute()
                    fold_metrics.update({k: float(v) for k, v in metrics.items()
                                        if isinstance(v, (int, float, np.floating))})
                except Exception:
                    pass
            
            # Also collect from trainer logged metrics
            if trainer.callback_metrics:
                for k, v in trainer.callback_metrics.items():
                    if k.startswith('val/'):
                        clean_k = k.replace('val/', '')
                        if clean_k not in fold_metrics:
                            try:
                                fold_metrics[clean_k] = float(v)
                            except (TypeError, ValueError):
                                pass
            
            all_fold_metrics.append(fold_metrics)
            
        except Exception as e:
            print(f"ERROR in fold {fold}: {e}")
            import traceback
            traceback.print_exc()
    
    # Aggregate results
    if all_fold_metrics:
        print("\n" + "=" * 70)
        print("K-FOLD CROSS-VALIDATION SUMMARY (REGRESSION)")
        print("=" * 70)
        
        # Key regression + ranking metrics to aggregate
        key_metrics = [
            'recall@25', 'recall@50', 'recall@100', 'recall@200',
            'precision@25', 'precision@50', 'precision@100',
            'enrichment@25', 'enrichment@50', 'enrichment@100',
            'spearman_rho', 'pearson_r', 'kendall_tau',
            'r2', 'mae', 'rmse',
            'auc_recall', 'first_hit_rank', 'mrr',
        ]
        
        print(f"\nResults across {len(all_fold_metrics)} folds:")
        print("-" * 60)
        
        summary = {}
        for metric in key_metrics:
            values = [m.get(metric, None) for m in all_fold_metrics]
            values = [v for v in values if v is not None]
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                summary[metric] = {'mean': mean_val, 'std': std_val, 'values': values}
                print(f"  {metric:<25s}: {mean_val:.4f} ± {std_val:.4f}  "
                      f"(per-fold: {', '.join(f'{v:.3f}' for v in values)})")
        
        print("-" * 60)
        print("=" * 70 + "\n")
        
        # Save summary
        os.makedirs(log_dir, exist_ok=True)
        summary_path = os.path.join(log_dir, "kfold_summary.json")
        
        # Make values JSON-serializable
        serializable_folds = []
        for fm in all_fold_metrics:
            sfm = {}
            for k, v in fm.items():
                try:
                    sfm[k] = float(v)
                except (TypeError, ValueError):
                    sfm[k] = str(v)
            serializable_folds.append(sfm)
        
        with open(summary_path, 'w') as f:
            json.dump({
                'n_folds': len(all_fold_metrics),
                'fold_metrics': serializable_folds,
                'summary': {k: {'mean': v['mean'], 'std': v['std']}
                           for k, v in summary.items()},
            }, f, indent=2)
        print(f"Summary saved to: {summary_path}")
        
        return {'fold_metrics': all_fold_metrics, 'summary': summary}
    
    print("WARNING: No folds completed successfully!")
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--downstream", default="bandgaps_regression")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--freeze_layers", default="2")
    parser.add_argument("--loss_type", default="huber")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    args = parser.parse_args()
    
    freeze = args.freeze_layers
    if freeze != "all":
        freeze = int(freeze)
    
    run(
        data_dir=args.data_dir,
        downstream=args.downstream,
        log_dir=args.log_dir,
        threshold=args.threshold,
        freeze_layers=freeze,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
    )
