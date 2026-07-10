#!/usr/bin/env python3
"""
Comprehensive Result Comparison for MOF Discovery
====================================================

THE master reporting script that consolidates ALL results from:
  - Old NN experiments (exp01–exp60, original split)
  - Resplit experiments (exp350–361, strategies A/B/C)
  - Embedding-informed split experiments (exp362–369, strategies D/E/F)
  - Embedding classifiers (sklearn on fixed embeddings)
  - k-NN baselines (knn_baseline.py outputs)

Key Design Decisions:
  1. Results are GROUPED BY SPLIT — metrics across different test sets
     are NOT directly comparable. This script explicitly marks which
     comparisons are fair (same test set) and which are cross-split.
  2. Per-positive analysis: for each of the 74 known positives (bg<1 eV),
     which methods detect it when it appears in the test set.
  3. Feature importance from tree-based sklearn classifiers (RF, GBT).
  4. Meta-comparison: "Did educated splitting improve things?"

Usage:
  # Auto-scan everything:
  python compare_results.py \\
      --experiments_base ./experiments \\
      --old_results ./Results/all_experiment_results \\
      --embedding_classifiers ./embedding_classifiers \\
      --knn_results ./knn_baselines \\
      --output_dir ./comparison_report

  # Only specific experiment groups:
  python compare_results.py \\
      --experiments_base ./experiments \\
      --output_dir ./comparison_report \\
      --split_filter original,D,E,F

  # Minimal: just scan experiments directory:
  python compare_results.py --experiments_base ./experiments
"""

import os
import sys
import json
import csv
import re
import argparse
import glob
from collections import defaultdict
from datetime import datetime

import numpy as np


# =============================================================================
# CONSTANTS & SPLIT METADATA
# =============================================================================

# Which experiments belong to which split and category
EXPERIMENT_METADATA = {
    # Original split — classification (exp01-exp30)
    'exp01': {'split': 'original', 'type': 'classification', 'desc': 'Baseline, weights ON'},
    'exp02': {'split': 'original', 'type': 'classification', 'desc': 'Baseline, weights OFF'},
    'exp03': {'split': 'original', 'type': 'classification', 'desc': 'Linear probe, weights ON'},
    'exp04': {'split': 'original', 'type': 'classification', 'desc': 'Linear probe, weights OFF'},
    'exp05': {'split': 'original', 'type': 'classification', 'desc': 'Freeze 1, weights ON'},
    'exp06': {'split': 'original', 'type': 'classification', 'desc': 'Freeze 1, weights OFF'},
    'exp07': {'split': 'original', 'type': 'classification', 'desc': 'Freeze 3, weights ON'},
    'exp08': {'split': 'original', 'type': 'classification', 'desc': 'Freeze 3, weights OFF'},
    'exp09': {'split': 'original', 'type': 'classification', 'desc': 'Full finetune, weights ON'},
    'exp10': {'split': 'original', 'type': 'classification', 'desc': 'Full finetune, weights OFF'},
    'exp11': {'split': 'original', 'type': 'classification', 'desc': 'Low LR, weights ON'},
    'exp12': {'split': 'original', 'type': 'classification', 'desc': 'Low LR, weights OFF'},
    'exp13': {'split': 'original', 'type': 'classification', 'desc': 'High reg, weights ON'},
    'exp14': {'split': 'original', 'type': 'classification', 'desc': 'High reg, weights OFF'},
    'exp15': {'split': 'original', 'type': 'classification', 'desc': 'Agg head, weights ON'},
    'exp16': {'split': 'original', 'type': 'classification', 'desc': 'Agg head, weights OFF'},
    'exp17': {'split': 'original', 'type': 'classification', 'desc': 'Low reg, weights ON'},
    'exp18': {'split': 'original', 'type': 'classification', 'desc': 'Low reg, weights OFF'},
    'exp19': {'split': 'original', 'type': 'classification', 'desc': 'High LR, weights ON'},
    'exp20': {'split': 'original', 'type': 'classification', 'desc': 'High LR, weights OFF'},
    'exp21': {'split': 'original', 'type': 'classification', 'desc': 'LinProbe high LR, w ON'},
    'exp22': {'split': 'original', 'type': 'classification', 'desc': 'LinProbe high LR, w OFF'},
    'exp23': {'split': 'original', 'type': 'classification', 'desc': 'LinProbe low LR, w ON'},
    'exp24': {'split': 'original', 'type': 'classification', 'desc': 'LinProbe low LR, w OFF'},
    'exp25': {'split': 'original', 'type': 'classification', 'desc': 'LinProbe high reg, w ON'},
    'exp26': {'split': 'original', 'type': 'classification', 'desc': 'LinProbe high reg, w OFF'},
    'exp27': {'split': 'original', 'type': 'focal', 'desc': 'Focal standard'},
    'exp28': {'split': 'original', 'type': 'focal', 'desc': 'Focal aggressive'},
    'exp29': {'split': 'original', 'type': 'classification', 'desc': 'Pos weight 50x'},
    'exp30': {'split': 'original', 'type': 'classification', 'desc': 'Pos weight 100x'},
    # Original split — regression (exp31-exp45)
    'exp31': {'split': 'original', 'type': 'regression', 'desc': 'Regression baseline'},
    'exp32': {'split': 'original', 'type': 'regression', 'desc': 'Regression linprobe'},
    'exp33': {'split': 'original', 'type': 'regression', 'desc': 'Regression fulltune'},
    'exp34': {'split': 'original', 'type': 'focal', 'desc': 'Focal combo'},
    'exp35': {'split': 'original', 'type': 'regression', 'desc': 'Regression MSE'},
    'exp36': {'split': 'original', 'type': 'regression', 'desc': 'Regression freeze1'},
    'exp37': {'split': 'original', 'type': 'regression', 'desc': 'Regression freeze3'},
    'exp38': {'split': 'original', 'type': 'regression', 'desc': 'Regression freeze6'},
    'exp39': {'split': 'original', 'type': 'regression', 'desc': 'Regression no weights'},
    'exp40': {'split': 'original', 'type': 'regression', 'desc': 'Regression high LR'},
    'exp41': {'split': 'original', 'type': 'regression', 'desc': 'Regression low LR'},
    'exp42': {'split': 'original', 'type': 'regression', 'desc': 'Regression high reg'},
    'exp43': {'split': 'original', 'type': 'regression', 'desc': 'Regression low reg'},
    'exp44': {'split': 'original', 'type': 'regression', 'desc': 'Regression linprobe high LR'},
    'exp45': {'split': 'original', 'type': 'regression', 'desc': 'Regression MSE no weights'},
    # Original split — multiclass (exp46-exp60)
    'exp46': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass baseline'},
    'exp47': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass mean pool'},
    'exp48': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass linprobe'},
    'exp49': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass freeze1'},
    'exp50': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass freeze3'},
    'exp51': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass fulltune'},
    'exp52': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass mean linprobe'},
    'exp53': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass no weights'},
    'exp54': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass high LR'},
    'exp55': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass low LR'},
    'exp56': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass high reg'},
    'exp57': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass freeze6'},
    'exp58': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass strong clip'},
    'exp59': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass mean fulltune'},
    'exp60': {'split': 'original', 'type': 'multiclass', 'desc': 'Multiclass aggressive head'},
    # Resplit experiments (B/C)
    'exp350': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, freeze2, weights ON'},
    'exp351': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, fulltune, weights ON'},
    'exp352': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, linprobe, weights ON'},
    'exp353': {'split': 'C', 'type': 'regression', 'desc': 'Strategy C, kfold, weights ON'},
    'exp354': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, freeze1, weights ON'},
    'exp355': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, auc monitor, weights ON'},
    'exp356': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, freeze2, weights OFF'},
    'exp357': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, fulltune, weights OFF'},
    'exp358': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, linprobe, weights OFF'},
    'exp359': {'split': 'C', 'type': 'regression', 'desc': 'Strategy C, kfold, weights OFF'},
    'exp360': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, freeze1, weights OFF'},
    'exp361': {'split': 'B', 'type': 'regression', 'desc': 'Strategy B, auc monitor, weights OFF'},
    # Embedding-informed splits (D/E/F)
    'exp362': {'split': 'D', 'type': 'regression', 'desc': 'Split D(farthest), freeze2'},
    'exp363': {'split': 'D', 'type': 'regression', 'desc': 'Split D(farthest), linprobe'},
    'exp364': {'split': 'D', 'type': 'regression', 'desc': 'Split D(farthest), fulltune'},
    'exp365': {'split': 'E', 'type': 'regression', 'desc': 'Split E(cluster), freeze2'},
    'exp366': {'split': 'E', 'type': 'regression', 'desc': 'Split E(cluster), linprobe'},
    'exp367': {'split': 'F', 'type': 'regression', 'desc': 'Split F(coverage), freeze2'},
    'exp368': {'split': 'F', 'type': 'regression', 'desc': 'Split F(coverage), linprobe'},
    'exp369': {'split': 'F', 'type': 'regression', 'desc': 'Split F(coverage), fulltune'},
}


# =============================================================================
# RESULT LOADING
# =============================================================================

def load_test_predictions(csv_path):
    """Load test_predictions.csv -> list of dicts."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'cif_id': row['cif_id'],
                'score': float(row['score']),
                'predicted_binary': int(row.get('predicted_binary', 0)),
                'true_label': float(row['true_label']),
                'mode': row.get('mode', 'unknown'),
            })
    return results


def load_final_results(json_path):
    """Load final_results.json."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_discovery_metrics(predictions, threshold=1.0):
    """
    Compute standard discovery ranking metrics from predictions.
    
    For regression mode: lower score = predicted lower bandgap = more likely positive.
    For multiclass/classifier mode: higher score = more likely positive.
    """
    if not predictions:
        return {}
    
    cif_ids = [p['cif_id'] for p in predictions]
    scores = np.array([p['score'] for p in predictions])
    true_labels = np.array([p['true_label'] for p in predictions])
    mode = predictions[0].get('mode', 'regression')
    
    # Determine sort order
    if mode in ('regression',):
        # Lower score = more positive (predicted bandgap)
        order = np.argsort(scores)
    else:
        # Higher score = more positive (probability/confidence)
        order = np.argsort(-scores)
    
    sorted_cids = [cif_ids[i] for i in order]
    sorted_true = true_labels[order]
    is_positive = sorted_true < threshold
    
    n_total = len(scores)
    n_positive = int(is_positive.sum())
    
    if n_positive == 0:
        return {'n_total': n_total, 'n_positive': 0, 'error': 'no positives in test set'}
    
    prevalence = n_positive / n_total
    
    metrics = {
        'n_total': n_total,
        'n_positive': n_positive,
        'prevalence': prevalence,
        'mode': mode,
    }
    
    # Recall@K, Precision@K, Enrichment@K
    for K in [25, 50, 100, 200, 500]:
        if K > n_total:
            continue
        hits = int(is_positive[:K].sum())
        recall = hits / n_positive if n_positive > 0 else 0
        precision = hits / K
        enrichment = precision / prevalence if prevalence > 0 else 0
        metrics[f'recall@{K}'] = recall
        metrics[f'precision@{K}'] = precision
        metrics[f'enrichment@{K}'] = enrichment
        metrics[f'hits@{K}'] = hits
    
    # First hit rank (1-indexed), Mean/Median hit rank
    pos_ranks = np.where(is_positive)[0] + 1  # 1-indexed
    if len(pos_ranks) > 0:
        metrics['first_hit_rank'] = int(pos_ranks[0])
        metrics['median_hit_rank'] = int(np.median(pos_ranks))
        metrics['mean_hit_rank'] = float(np.mean(pos_ranks))
        metrics['last_hit_rank'] = int(pos_ranks[-1])
        metrics['mrr'] = float(1.0 / pos_ranks[0])
    
    # NEF (Normalized Enrichment Factor)
    for pct in [1, 2, 5, 10]:
        k = max(1, int(n_total * pct / 100))
        hits = int(is_positive[:k].sum())
        nef = (hits / n_positive) / (k / n_total) if (n_positive > 0 and n_total > 0) else 0
        metrics[f'nef@{pct}pct'] = nef
    
    # AUC of recall curve
    recall_curve = np.cumsum(is_positive) / n_positive
    metrics['auc_recall'] = float(np.mean(recall_curve))
    
    # Spearman rank correlation (predicted score vs true bandgap)
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(scores, true_labels)
        if mode in ('regression',):
            metrics['spearman_rho'] = float(rho) if not np.isnan(rho) else 0.0
        else:
            metrics['spearman_rho'] = float(-rho) if not np.isnan(rho) else 0.0
    except Exception:
        metrics['spearman_rho'] = 0.0
    
    # Per-positive detail: which positives found in top-K
    metrics['found_positives'] = {}
    for K in [25, 50, 100, 200, 500]:
        if K > n_total:
            continue
        found = []
        for i in range(min(K, n_total)):
            if is_positive[i]:
                found.append({
                    'cif_id': sorted_cids[i],
                    'rank': i + 1,
                    'true_bandgap': float(sorted_true[i]),
                })
        metrics['found_positives'][f'top_{K}'] = found
    
    return metrics


def load_old_csv_results(csv_path, is_regression=False):
    """Load results from the pre-existing classification_results.csv or 
    regression_results.csv files."""
    results = {}
    if not os.path.exists(csv_path):
        return results
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row.get('exp_name', row.get('filename', 'unknown'))
            # Extract experiment number
            match = re.search(r'exp(\d+)', exp_name)
            exp_key = f'exp{match.group(1)}' if match else exp_name
            
            entry = {
                'source': 'old_csv',
                'exp_name': exp_name,
                'is_regression': is_regression or row.get('is_regression', '').lower() == 'true',
                'weights': row.get('weights', 'N/A'),
                'completed': row.get('completed', '').lower() == 'true',
            }
            
            # Extract key metrics
            for k in ['test_recall@50', 'test_recall@100', 'test_recall@200', 'test_recall@500',
                       'test_precision@50', 'test_precision@100', 'test_precision@200',
                       'test_enrichment@50', 'test_enrichment@100', 'test_enrichment@200',
                       'test_sniped@50', 'test_sniped@100', 'test_sniped@200', 'test_sniped@500',
                       'test_pr_auc', 'test_roc_auc',
                       'test_mae', 'test_rmse', 'test_pearson_r',
                       'val_recall@50', 'val_recall@100', 'val_recall@200',
                       'val_sniped@50', 'val_sniped@100', 'val_sniped@200',
                       'val_pr_auc', 'val_mae', 'val_rmse', 'val_pearson_r']:
                val = row.get(k, '')
                if val and val != '':
                    try:
                        entry[k] = float(val)
                    except ValueError:
                        pass
            
            if exp_key not in results:
                results[exp_key] = []
            results[exp_key].append(entry)
    
    return results


def scan_experiments_dir(exp_base):
    """Scan experiments directory for test_predictions.csv and final_results.json."""
    results = {}
    
    if not os.path.isdir(exp_base):
        print(f"  WARNING: {exp_base} not found")
        return results
    
    for entry in sorted(os.listdir(exp_base)):
        exp_dir = os.path.join(exp_base, entry)
        if not os.path.isdir(exp_dir):
            continue
        
        # Skip non-experiment directories
        if not entry.startswith('exp'):
            continue
        
        pred_csv = os.path.join(exp_dir, 'test_predictions.csv')
        results_json = os.path.join(exp_dir, 'final_results.json')
        
        exp_match = re.match(r'exp(\d+)', entry)
        exp_key = f'exp{exp_match.group(1)}' if exp_match else entry
        
        exp_result = {
            'dir_name': entry,
            'dir_path': exp_dir,
            'exp_key': exp_key,
            'has_predictions': os.path.exists(pred_csv),
            'has_results': os.path.exists(results_json),
        }
        
        if os.path.exists(pred_csv):
            try:
                exp_result['predictions'] = load_test_predictions(pred_csv)
                exp_result['metrics'] = compute_discovery_metrics(exp_result['predictions'])
            except Exception as e:
                exp_result['load_error'] = str(e)
        
        if os.path.exists(results_json):
            try:
                exp_result['final_results'] = load_final_results(results_json)
            except Exception as e:
                exp_result['json_error'] = str(e)
        
        # Get metadata
        meta = EXPERIMENT_METADATA.get(exp_key, {})
        exp_result['split'] = meta.get('split', infer_split(entry))
        exp_result['type'] = meta.get('type', 'unknown')
        exp_result['desc'] = meta.get('desc', entry)
        
        if exp_key not in results:
            results[exp_key] = []
        results[exp_key].append(exp_result)
    
    return results


def infer_split(dirname):
    """Infer split from directory name for experiments not in metadata."""
    dn = dirname.lower()
    if 'splitd' in dn or 'farthest' in dn or 'fulltune' in dn:
        return 'D'
    elif 'splite' in dn or 'embsplit_e' in dn or 'cluster' in dn:
        return 'E'
    elif 'splitf' in dn or 'embsplit_f' in dn or 'coverage' in dn:
        return 'F'
    elif 'strategy_a' in dn:
        return 'A'
    elif 'strategy_b' in dn or 'resplit' in dn:
        return 'B'
    elif 'strategy_c' in dn or 'kfold' in dn:
        return 'C'
    else:
        return 'original'


def scan_embedding_classifiers(clf_dir):
    """Scan embedding_classifiers output directory."""
    results = {}
    
    if not os.path.isdir(clf_dir):
        return results
    
    # Look for all_results.json (master file)
    master_json = os.path.join(clf_dir, 'all_results.json')
    if os.path.exists(master_json):
        with open(master_json, 'r') as f:
            results['_master'] = json.load(f)
    
    # Find individual method directories with test_predictions.csv
    for entry in sorted(os.listdir(clf_dir)):
        sub_dir = os.path.join(clf_dir, entry)
        pred_csv = os.path.join(sub_dir, 'test_predictions.csv')
        if os.path.isdir(sub_dir) and os.path.exists(pred_csv):
            try:
                preds = load_test_predictions(pred_csv)
                metrics = compute_discovery_metrics(preds)
                results[entry] = {
                    'predictions': preds,
                    'metrics': metrics,
                    'method': entry,
                }
            except Exception as e:
                results[entry] = {'error': str(e)}
    
    return results


def scan_knn_results(knn_dir):
    """Scan knn_baseline.py output directory."""
    results = {}
    
    if not os.path.isdir(knn_dir):
        return results
    
    # Look for knn_hybrid_results.json
    json_path = os.path.join(knn_dir, 'knn_hybrid_results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results['_hybrid_master'] = json.load(f)
    
    # Load main test_predictions.csv (k-NN regression)
    pred_csv = os.path.join(knn_dir, 'test_predictions.csv')
    if os.path.exists(pred_csv):
        try:
            preds = load_test_predictions(pred_csv)
            results['knn_regression'] = {
                'predictions': preds,
                'metrics': compute_discovery_metrics(preds),
            }
        except Exception as e:
            results['knn_regression'] = {'error': str(e)}
    
    # Load sim_to_pos predictions
    sim_csv = os.path.join(knn_dir, 'test_predictions_sim_to_pos.csv')
    if os.path.exists(sim_csv):
        try:
            preds = load_test_predictions(sim_csv)
            results['sim_to_positive'] = {
                'predictions': preds,
                'metrics': compute_discovery_metrics(preds),
            }
        except Exception as e:
            results['sim_to_positive'] = {'error': str(e)}
    
    return results


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def format_metric(value, fmt='.3f', na='  N/A'):
    """Format a metric value, handling None/missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return na
    return f'{value:{fmt}}'


def format_pct(value, na='  N/A'):
    """Format as percentage."""
    if value is None:
        return na
    return f'{value*100:5.1f}%'


def print_section_header(title, char='=', width=100):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_comparison_table(entries, title="Comparison"):
    """Print a formatted comparison table.
    
    entries: list of dicts with keys: name, metrics (dict with recall@K, etc.)
    """
    if not entries:
        print("  (no results to display)")
        return
    
    print_section_header(title)
    
    header = (f"  {'Method':<45s}  {'Split':>5s}  {'FHR':>5s}  {'R@25':>6s}  "
              f"{'R@50':>6s}  {'R@100':>6s}  {'R@200':>6s}  {'MRR':>6s}  "
              f"{'Spear':>6s}  {'AUC':>6s}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    
    for e in entries:
        m = e.get('metrics', {})
        if not m or 'error' in m:
            continue
        name = e.get('name', 'unknown')[:45]
        split = e.get('split', '?')[:5]
        fhr = m.get('first_hit_rank', None)
        r25 = m.get('recall@25', None)
        r50 = m.get('recall@50', None)
        r100 = m.get('recall@100', None)
        r200 = m.get('recall@200', None)
        mrr = m.get('mrr', None)
        rho = m.get('spearman_rho', None)
        auc = m.get('auc_recall', None)
        
        fhr_s = f'{fhr:>5d}' if fhr is not None else '    -'
        r25_s = format_metric(r25)
        r50_s = format_metric(r50)
        r100_s = format_metric(r100)
        r200_s = format_metric(r200)
        mrr_s = format_metric(mrr, '.4f')
        rho_s = format_metric(rho)
        auc_s = format_metric(auc)
        
        print(f"  {name:<45s}  {split:>5s}  {fhr_s}  {r25_s:>6s}  "
              f"{r50_s:>6s}  {r100_s:>6s}  {r200_s:>6s}  {mrr_s:>6s}  "
              f"{rho_s:>6s}  {auc_s:>6s}")
    
    print(f"  {'-' * (len(header) - 2)}")


def print_old_results_table(old_results, title="Old Experiment Results (Original Split)"):
    """Print old CSV results in a comparable format."""
    if not old_results:
        print(f"\n  {title}: no results found")
        return
    
    print_section_header(title)
    
    # Old results use percentages (0-100) for recall
    header = (f"  {'Experiment':<40s}  {'Weights':>7s}  {'S@50':>4s}  {'R@50':>6s}  "
              f"{'S@100':>5s}  {'R@100':>6s}  {'S@200':>5s}  {'R@200':>6s}  "
              f"{'PR-AUC':>7s}  {'ROC-AUC':>7s}  {'MAE':>6s}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    
    for exp_key in sorted(old_results.keys()):
        for entry in old_results[exp_key]:
            name = entry.get('exp_name', exp_key)[:40]
            w = entry.get('weights', '-')[:7]
            s50 = entry.get('test_sniped@50', None)
            r50 = entry.get('test_recall@50', None)
            s100 = entry.get('test_sniped@100', None)
            r100 = entry.get('test_recall@100', None)
            s200 = entry.get('test_sniped@200', None)
            r200 = entry.get('test_recall@200', None)
            prauc = entry.get('test_pr_auc', None)
            rocauc = entry.get('test_roc_auc', None)
            mae = entry.get('test_mae', None)
            
            s50_s = f'{int(s50):>4d}' if s50 is not None else '   -'
            r50_s = f'{r50:>5.1f}%' if r50 is not None else '    -'
            s100_s = f'{int(s100):>5d}' if s100 is not None else '    -'
            r100_s = f'{r100:>5.1f}%' if r100 is not None else '    -'
            s200_s = f'{int(s200):>5d}' if s200 is not None else '    -'
            r200_s = f'{r200:>5.1f}%' if r200 is not None else '    -'
            prauc_s = format_metric(prauc, '.4f', '      -')
            rocauc_s = format_metric(rocauc, '.4f', '      -')
            mae_s = format_metric(mae, '.3f', '     -')
            
            print(f"  {name:<40s}  {w:>7s}  {s50_s}  {r50_s:>6s}  "
                  f"{s100_s}  {r100_s:>6s}  {s200_s}  {r200_s:>6s}  "
                  f"{prauc_s:>7s}  {rocauc_s:>7s}  {mae_s:>6s}")
    
    print(f"  {'-' * (len(header) - 2)}")


def print_per_positive_analysis(all_experiments, threshold=1.0):
    """For each positive in any test set, show which methods found it."""
    print_section_header("PER-POSITIVE ANALYSIS: Which Methods Find Each Target?", '#')
    
    # Collect all positives across all experiments
    positive_hits = defaultdict(lambda: defaultdict(dict))
    # positive_hits[cif_id][method_name] = {'rank': ..., 'score': ...}
    
    for exp_name, exp_data_list in all_experiments.items():
        for exp_data in exp_data_list:
            preds = exp_data.get('predictions', [])
            if not preds:
                continue
            
            mode = preds[0].get('mode', 'regression') if preds else 'regression'
            
            # Sort predictions the same way as metrics computation
            if mode in ('regression',):
                sorted_preds = sorted(preds, key=lambda x: x['score'])
            else:
                sorted_preds = sorted(preds, key=lambda x: -x['score'])
            
            for rank, p in enumerate(sorted_preds, 1):
                if p['true_label'] < threshold:
                    cid = p['cif_id']
                    method_label = exp_data.get('desc', exp_name)
                    split = exp_data.get('split', '?')
                    key = f"{method_label} [{split}]"
                    positive_hits[cid][key] = {
                        'rank': rank,
                        'score': p['score'],
                        'true_bg': p['true_label'],
                    }
    
    if not positive_hits:
        print("  No positives found in any experiment predictions.")
        return
    
    # Print per-positive report
    for cid in sorted(positive_hits.keys()):
        methods = positive_hits[cid]
        best_rank = min(m['rank'] for m in methods.values())
        true_bg = list(methods.values())[0]['true_bg']
        n_methods = len(methods)
        
        # Color-code based on best rank
        status = "EASY" if best_rank <= 50 else ("MODERATE" if best_rank <= 200 else "HARD")
        
        print(f"\n  {cid} (bg={true_bg:.3f} eV) — Best rank: {best_rank}, "
              f"Found by {n_methods} methods — [{status}]")
        
        # Show top 10 methods sorted by rank
        sorted_methods = sorted(methods.items(), key=lambda x: x[1]['rank'])
        for method_name, info in sorted_methods[:15]:
            r = info['rank']
            marker = " *** " if r <= 100 else ""
            print(f"    Rank {r:>5d}: {method_name}{marker}")
        
        if len(sorted_methods) > 15:
            print(f"    ... and {len(sorted_methods) - 15} more methods")


def print_split_comparison(all_experiments):
    """Meta-comparison: how do different splits perform?"""
    print_section_header(
        "SPLIT META-COMPARISON: Did Educated Splitting Help?", '#')
    
    print("\n  CAUTION: Metrics across different splits are NOT directly comparable!")
    print("  Different splits have different test sets with different positives.")
    print("  A higher recall@100 on split D does NOT necessarily mean D is better —")
    print("  it might just have 'easier' test positives.\n")
    
    split_summary = defaultdict(list)
    
    for exp_name, exp_data_list in all_experiments.items():
        for exp_data in exp_data_list:
            metrics = exp_data.get('metrics', {})
            if not metrics or 'error' in metrics:
                continue
            split = exp_data.get('split', 'unknown')
            split_summary[split].append({
                'name': exp_data.get('desc', exp_name),
                'metrics': metrics,
            })
    
    for split in sorted(split_summary.keys()):
        entries = split_summary[split]
        n = len(entries)
        
        # Average metrics across experiments in this split
        r100_vals = [e['metrics'].get('recall@100', 0) for e in entries if 'recall@100' in e['metrics']]
        fhr_vals = [e['metrics'].get('first_hit_rank', 9999) for e in entries if 'first_hit_rank' in e['metrics']]
        rho_vals = [e['metrics'].get('spearman_rho', 0) for e in entries if 'spearman_rho' in e['metrics']]
        auc_vals = [e['metrics'].get('auc_recall', 0) for e in entries if 'auc_recall' in e['metrics']]
        
        # Test set info
        n_pos = entries[0]['metrics'].get('n_positive', 0) if entries else 0
        n_total = entries[0]['metrics'].get('n_total', 0) if entries else 0
        
        print(f"  Split '{split}': {n} experiments, test={n_total} ({n_pos} positives)")
        
        if r100_vals:
            print(f"    Recall@100:  mean={np.mean(r100_vals):.3f}, "
                  f"best={max(r100_vals):.3f}, worst={min(r100_vals):.3f}")
        if fhr_vals:
            fhr_clean = [v for v in fhr_vals if v < 9999]
            if fhr_clean:
                print(f"    FHR:         mean={np.mean(fhr_clean):.0f}, "
                      f"best={min(fhr_clean)}, worst={max(fhr_clean)}")
        if rho_vals:
            print(f"    Spearman:    mean={np.mean(rho_vals):.3f}, "
                  f"best={max(rho_vals):.3f}, worst={min(rho_vals):.3f}")
        if auc_vals:
            print(f"    AUC-Recall:  mean={np.mean(auc_vals):.4f}, "
                  f"best={max(auc_vals):.4f}, worst={min(auc_vals):.4f}")
        
        # Best method in this split
        best = max(entries, key=lambda e: e['metrics'].get('recall@100', 0))
        print(f"    Best method: {best['name']} "
              f"(R@100={best['metrics'].get('recall@100', 0):.3f}, "
              f"FHR={best['metrics'].get('first_hit_rank', 'N/A')})")
        print()


def print_feature_importance(clf_dir):
    """Print feature importance from tree-based sklearn classifiers if available."""
    if not os.path.isdir(clf_dir):
        return
    
    print_section_header("FEATURE IMPORTANCE: Tree-Based Classifiers", '#')
    
    # Look for feature importance files
    for method in ['random_forest', 'gradient_boosting']:
        # Check if there's a feature importance file
        fi_path = os.path.join(clf_dir, method, 'feature_importance.json')
        fi_npy = os.path.join(clf_dir, method, 'feature_importance.npy')
        
        if os.path.exists(fi_path):
            with open(fi_path) as f:
                fi = json.load(f)
            if isinstance(fi, dict):
                sorted_fi = sorted(fi.items(), key=lambda x: -abs(float(x[1])))[:20]
                print(f"\n  {method.replace('_', ' ').title()} — Top 20 Important Features:")
                for fname, fval in sorted_fi:
                    print(f"    {fname:<30s}  {float(fval):.6f}")
        elif os.path.exists(fi_npy):
            fi = np.load(fi_npy)
            top_idx = np.argsort(-np.abs(fi))[:20]
            print(f"\n  {method.replace('_', ' ').title()} — Top 20 Important Embedding Dims:")
            for idx in top_idx:
                print(f"    dim_{idx:<5d}  {fi[idx]:.6f}")
    
    # Also look for all_results.json which may have analysis
    master_json = os.path.join(clf_dir, 'all_results.json')
    if os.path.exists(master_json):
        with open(master_json) as f:
            master = json.load(f)
        
        print(f"\n  Embedding Classifier Summary:")
        print(f"  {'Method':<30s}  {'FHR':>5s}  {'R@50':>6s}  {'R@100':>6s}  "
              f"{'R@200':>6s}  {'Spearman':>8s}")
        print(f"  {'-'*75}")
        
        for name in sorted(master.keys()):
            m = master[name]
            fhr = m.get('first_hit_rank', None)
            r50 = m.get('recall@50', None)
            r100 = m.get('recall@100', None)
            r200 = m.get('recall@200', None)
            rho = m.get('spearman_rho', None)
            
            fhr_s = f'{int(fhr):>5d}' if fhr is not None else '    -'
            r50_s = format_metric(r50) if r50 is not None else '  N/A'
            r100_s = format_metric(r100) if r100 is not None else '  N/A'
            r200_s = format_metric(r200) if r200 is not None else '  N/A'
            rho_s = format_metric(rho) if rho is not None else '  N/A'
            
            print(f"  {name:<30s}  {fhr_s}  {r50_s:>6s}  {r100_s:>6s}  "
                  f"{r200_s:>6s}  {rho_s:>8s}")


def print_executive_summary(all_experiments, old_results, clf_results, knn_results):
    """Print a high-level executive summary at the end."""
    print_section_header("EXECUTIVE SUMMARY", '#', 100)
    
    # Count results by category
    n_old_classification = 0
    n_old_regression = 0
    for exp_key, entries in old_results.items():
        for e in entries:
            if e.get('is_regression'):
                n_old_regression += 1
            else:
                n_old_classification += 1
    
    n_new_with_results = 0
    n_new_pending = 0
    for exp_key, exp_list in all_experiments.items():
        for e in exp_list:
            if e.get('has_predictions'):
                n_new_with_results += 1
            elif e.get('has_results'):
                n_new_with_results += 1
            else:
                n_new_pending += 1
    
    n_clf_methods = len([k for k in clf_results.keys() if not k.startswith('_')])
    n_knn_methods = len([k for k in knn_results.keys() if not k.startswith('_')])
    
    print(f"\n  COVERAGE:")
    print(f"    Old NN experiments (original split):   {n_old_classification} classification + "
          f"{n_old_regression} regression")
    print(f"    New NN experiments (exp350+):           {n_new_with_results} with results, "
          f"{n_new_pending} pending")
    print(f"    Embedding classifiers (sklearn):        {n_clf_methods} methods")
    print(f"    k-NN baselines:                         {n_knn_methods} methods")
    
    # Find global best across all methods
    print(f"\n  BEST RESULTS BY CATEGORY:")
    
    # Best from old experiments
    best_old_r100 = 0
    best_old_name = None
    for exp_key, entries in old_results.items():
        for e in entries:
            r100 = e.get('test_recall@100', 0)
            if r100 and r100 > best_old_r100:
                best_old_r100 = r100
                best_old_name = e.get('exp_name', exp_key)
    
    if best_old_name:
        print(f"    Old NN (original split): {best_old_name} — "
              f"Test Recall@100 = {best_old_r100:.1f}%")
    else:
        print(f"    Old NN (original split): No results with recall@100 > 0")
    
    # Best from new experiments
    best_new_r100 = 0
    best_new_name = None
    best_new_split = None
    for exp_key, exp_list in all_experiments.items():
        for e in exp_list:
            m = e.get('metrics', {})
            r100 = m.get('recall@100', 0)
            if r100 > best_new_r100:
                best_new_r100 = r100
                best_new_name = e.get('desc', exp_key)
                best_new_split = e.get('split', '?')
    
    if best_new_name:
        print(f"    New NN (split {best_new_split}): {best_new_name} — "
              f"Recall@100 = {best_new_r100:.3f}")
    else:
        print(f"    New NN experiments: No results yet (pending cluster execution)")
    
    # Best from embedding classifiers
    best_clf_r100 = 0
    best_clf_name = None
    for name, data in clf_results.items():
        if name.startswith('_'):
            continue
        m = data.get('metrics', {})
        r100 = m.get('recall@100', 0)
        if r100 > best_clf_r100:
            best_clf_r100 = r100
            best_clf_name = name
    
    if best_clf_name:
        print(f"    Embedding classifiers: {best_clf_name} — "
              f"Recall@100 = {best_clf_r100:.3f}")
    
    # Best from kNN
    best_knn_r100 = 0
    best_knn_name = None
    for name, data in knn_results.items():
        if name.startswith('_'):
            continue
        m = data.get('metrics', {})
        r100 = m.get('recall@100', 0)
        if r100 > best_knn_r100:
            best_knn_r100 = r100
            best_knn_name = name
    
    if best_knn_name:
        print(f"    k-NN baselines: {best_knn_name} — "
              f"Recall@100 = {best_knn_r100:.3f}")
    
    # Key takeaways
    print(f"\n  KEY COMPARISONS:")
    print(f"    Original split test: 9 positives out of ~2,163 (0.42% prevalence)")
    print(f"    Educated splits have DIFFERENT test sets — direct metric comparison "
          f"is misleading.")
    print(f"    The per-positive analysis above shows which targets are structurally")
    print(f"    findable vs genuinely challenging.\n")
    
    if best_old_r100 == 0 and best_new_r100 == 0:
        print(f"  STATUS: No experiments have achieved recall@100 > 0 yet.")
        print(f"  This means NO positive MOF has been ranked in the top 100 by any method.")
        print(f"  The embedding classifiers and k-NN baselines may provide insight into")
        print(f"  whether the problem is the model or the data structure.\n")


# =============================================================================
# SAVE REPORT
# =============================================================================

def save_report_json(output_dir, all_experiments, old_results, clf_results, knn_results):
    """Save machine-readable JSON report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'generated': datetime.now().isoformat(),
        'summary': {},
        'by_split': defaultdict(list),
        'old_results': {},
        'embedding_classifiers': {},
        'knn_baselines': {},
    }
    
    # Summarize new experiments
    for exp_key, exp_list in all_experiments.items():
        for e in exp_list:
            metrics = e.get('metrics', {})
            # Clean non-serializable items
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    clean_metrics[k] = v
                elif isinstance(v, (np.integer,)):
                    clean_metrics[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_metrics[k] = float(v)
                elif isinstance(v, dict):
                    # found_positives etc.
                    try:
                        json.dumps(v)  # test serializable
                        clean_metrics[k] = v
                    except (TypeError, ValueError):
                        pass
            
            entry = {
                'exp_key': exp_key,
                'dir_name': e.get('dir_name', ''),
                'split': e.get('split', 'unknown'),
                'type': e.get('type', 'unknown'),
                'desc': e.get('desc', ''),
                'has_predictions': e.get('has_predictions', False),
                'metrics': clean_metrics,
            }
            report['by_split'][e.get('split', 'unknown')].append(entry)
    
    # Old results summary
    for exp_key, entries in old_results.items():
        report['old_results'][exp_key] = [
            {k: v for k, v in e.items() if isinstance(v, (int, float, str, bool))}
            for e in entries
        ]
    
    # Embedding classifiers
    for name, data in clf_results.items():
        if name.startswith('_'):
            if isinstance(data, dict):
                try:
                    json.dumps(data)
                    report['embedding_classifiers'][name] = data
                except (TypeError, ValueError):
                    pass
            continue
        m = data.get('metrics', {})
        clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                 for k, v in m.items() 
                 if isinstance(v, (int, float, str, bool, np.integer, np.floating))}
        report['embedding_classifiers'][name] = clean
    
    # k-NN results
    for name, data in knn_results.items():
        if name.startswith('_'):
            if isinstance(data, dict):
                try:
                    json.dumps(data)
                    report['knn_baselines'][name] = data
                except (TypeError, ValueError):
                    pass
            continue
        m = data.get('metrics', {})
        clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                 for k, v in m.items() 
                 if isinstance(v, (int, float, str, bool, np.integer, np.floating))}
        report['knn_baselines'][name] = clean
    
    report_path = os.path.join(output_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  JSON report saved to: {report_path}")
    
    return report


def save_summary_csv(output_dir, all_experiments, old_results, clf_results, knn_results):
    """Save a flat CSV with one row per method for easy Excel/plotting."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'comparison_summary.csv')
    
    rows = []
    
    # Old results
    for exp_key, entries in old_results.items():
        for e in entries:
            meta = EXPERIMENT_METADATA.get(exp_key, {})
            rows.append({
                'source': 'old_nn',
                'method': e.get('exp_name', exp_key),
                'split': 'original',
                'type': meta.get('type', 'classification'),
                'description': meta.get('desc', ''),
                'n_positive': 9,  # original split
                'n_total': 2163,
                'recall@50': e.get('test_recall@50', ''),
                'recall@100': e.get('test_recall@100', ''),
                'recall@200': e.get('test_recall@200', ''),
                'recall@500': e.get('test_recall@500', ''),
                'hits@50': e.get('test_sniped@50', ''),
                'hits@100': e.get('test_sniped@100', ''),
                'hits@200': e.get('test_sniped@200', ''),
                'first_hit_rank': '',
                'mrr': '',
                'spearman_rho': '',
                'auc_recall': '',
                'pr_auc': e.get('test_pr_auc', ''),
                'roc_auc': e.get('test_roc_auc', ''),
                'mae': e.get('test_mae', ''),
                'rmse': e.get('test_rmse', ''),
            })
    
    # New experiments
    for exp_key, exp_list in all_experiments.items():
        for e in exp_list:
            m = e.get('metrics', {})
            rows.append({
                'source': 'new_nn',
                'method': e.get('dir_name', exp_key),
                'split': e.get('split', 'unknown'),
                'type': e.get('type', 'unknown'),
                'description': e.get('desc', ''),
                'n_positive': m.get('n_positive', ''),
                'n_total': m.get('n_total', ''),
                'recall@50': m.get('recall@50', ''),
                'recall@100': m.get('recall@100', ''),
                'recall@200': m.get('recall@200', ''),
                'recall@500': m.get('recall@500', ''),
                'hits@50': m.get('hits@50', ''),
                'hits@100': m.get('hits@100', ''),
                'hits@200': m.get('hits@200', ''),
                'first_hit_rank': m.get('first_hit_rank', ''),
                'mrr': m.get('mrr', ''),
                'spearman_rho': m.get('spearman_rho', ''),
                'auc_recall': m.get('auc_recall', ''),
                'pr_auc': '',
                'roc_auc': '',
                'mae': '',
                'rmse': '',
            })
    
    # Embedding classifiers
    for name, data in clf_results.items():
        if name.startswith('_'):
            continue
        m = data.get('metrics', {})
        rows.append({
            'source': 'embedding_clf',
            'method': name,
            'split': 'original',  # default; may be overridden if labels_dir used
            'type': 'sklearn',
            'description': name,
            'n_positive': m.get('n_positive', ''),
            'n_total': m.get('n_total', ''),
            'recall@50': m.get('recall@50', ''),
            'recall@100': m.get('recall@100', ''),
            'recall@200': m.get('recall@200', ''),
            'recall@500': m.get('recall@500', ''),
            'hits@50': m.get('hits@50', ''),
            'hits@100': m.get('hits@100', ''),
            'hits@200': m.get('hits@200', ''),
            'first_hit_rank': m.get('first_hit_rank', ''),
            'mrr': m.get('mrr', ''),
            'spearman_rho': m.get('spearman_rho', ''),
            'auc_recall': m.get('auc_recall', ''),
            'pr_auc': '',
            'roc_auc': '',
            'mae': '',
            'rmse': '',
        })
    
    # kNN results
    for name, data in knn_results.items():
        if name.startswith('_'):
            continue
        m = data.get('metrics', {})
        rows.append({
            'source': 'knn_baseline',
            'method': name,
            'split': 'original',
            'type': 'knn',
            'description': name,
            'n_positive': m.get('n_positive', ''),
            'n_total': m.get('n_total', ''),
            'recall@50': m.get('recall@50', ''),
            'recall@100': m.get('recall@100', ''),
            'recall@200': m.get('recall@200', ''),
            'recall@500': m.get('recall@500', ''),
            'hits@50': m.get('hits@50', ''),
            'hits@100': m.get('hits@100', ''),
            'hits@200': m.get('hits@200', ''),
            'first_hit_rank': m.get('first_hit_rank', ''),
            'mrr': m.get('mrr', ''),
            'spearman_rho': m.get('spearman_rho', ''),
            'auc_recall': m.get('auc_recall', ''),
            'pr_auc': '',
            'roc_auc': '',
            'mae': '',
            'rmse': '',
        })
    
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"  CSV summary saved to: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Result Comparison for MOF Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan everything:
  python compare_results.py --experiments_base ./experiments \\
      --old_results ./Results/all_experiment_results \\
      --output_dir ./comparison_report
      
  # After cluster run, add embedding classifiers + kNN:
  python compare_results.py --experiments_base ./experiments \\
      --old_results ./Results/all_experiment_results \\
      --embedding_classifiers ./embedding_classifiers \\
      --knn_results ./knn_baselines \\
      --output_dir ./comparison_report
""")
    parser.add_argument('--experiments_base', type=str, default='./experiments',
                        help='Base directory containing experiment folders')
    parser.add_argument('--old_results', type=str, default=None,
                        help='Directory with old classification_results.csv and '
                             'regression_results.csv')
    parser.add_argument('--embedding_classifiers', type=str, nargs='*', default=None,
                        help='Path(s) to embedding classifier output directories. '
                             'Can specify multiple for different splits.')
    parser.add_argument('--knn_results', type=str, nargs='*', default=None,
                        help='Path(s) to k-NN baseline output directories')
    parser.add_argument('--output_dir', type=str, default='./comparison_report',
                        help='Output directory for reports')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive class (eV)')
    parser.add_argument('--split_filter', type=str, default=None,
                        help='Comma-separated list of splits to include '
                             '(e.g., "original,D,E,F")')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-method output')
    args = parser.parse_args()
    
    split_filter = None
    if args.split_filter:
        split_filter = set(args.split_filter.split(','))
    
    print("=" * 100)
    print("  MOF DISCOVERY — COMPREHENSIVE RESULT COMPARISON")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    # =========================================================================
    # 1. LOAD OLD RESULTS
    # =========================================================================
    old_clf_results = {}
    old_reg_results = {}
    
    if args.old_results and os.path.isdir(args.old_results):
        print(f"\nLoading old results from: {args.old_results}")
        
        clf_csv = os.path.join(args.old_results, 'classification_results.csv')
        reg_csv = os.path.join(args.old_results, 'regression_results.csv')
        all_csv = os.path.join(args.old_results, 'all_experiment_results.csv')
        
        if os.path.exists(clf_csv):
            old_clf_results = load_old_csv_results(clf_csv, is_regression=False)
            print(f"  Loaded {sum(len(v) for v in old_clf_results.values())} "
                  f"classification results from {len(old_clf_results)} experiments")
        
        if os.path.exists(reg_csv):
            old_reg_results = load_old_csv_results(reg_csv, is_regression=True)
            print(f"  Loaded {sum(len(v) for v in old_reg_results.values())} "
                  f"regression results from {len(old_reg_results)} experiments")
        elif os.path.exists(all_csv):
            # Fallback: load all from combined CSV
            all_old = load_old_csv_results(all_csv)
            print(f"  Loaded {sum(len(v) for v in all_old.values())} "
                  f"results from all_experiment_results.csv")
            old_clf_results = all_old
    else:
        # Try auto-detect
        auto_path = os.path.join(os.path.dirname(args.experiments_base), 
                                 'Results', 'all_experiment_results')
        if os.path.isdir(auto_path):
            print(f"\nAuto-detected old results: {auto_path}")
            clf_csv = os.path.join(auto_path, 'classification_results.csv')
            reg_csv = os.path.join(auto_path, 'regression_results.csv')
            if os.path.exists(clf_csv):
                old_clf_results = load_old_csv_results(clf_csv, is_regression=False)
                print(f"  Loaded {sum(len(v) for v in old_clf_results.values())} "
                      f"classification results")
            if os.path.exists(reg_csv):
                old_reg_results = load_old_csv_results(reg_csv, is_regression=True)
                print(f"  Loaded {sum(len(v) for v in old_reg_results.values())} "
                      f"regression results")
    
    # Merge old results
    old_results = {**old_clf_results}
    for k, v in old_reg_results.items():
        if k in old_results:
            old_results[k].extend(v)
        else:
            old_results[k] = v
    
    # =========================================================================
    # 2. SCAN NEW EXPERIMENTS
    # =========================================================================
    print(f"\nScanning experiments directory: {args.experiments_base}")
    all_experiments = scan_experiments_dir(args.experiments_base)
    
    n_with_results = sum(1 for exp_list in all_experiments.values() 
                         for e in exp_list if e.get('has_predictions'))
    n_pending = sum(1 for exp_list in all_experiments.values() 
                    for e in exp_list if not e.get('has_predictions'))
    print(f"  Found {len(all_experiments)} experiment groups: "
          f"{n_with_results} with results, {n_pending} pending")
    
    # =========================================================================
    # 3. LOAD EMBEDDING CLASSIFIERS
    # =========================================================================
    clf_results = {}
    if args.embedding_classifiers:
        for clf_dir in args.embedding_classifiers:
            print(f"\nLoading embedding classifiers from: {clf_dir}")
            results = scan_embedding_classifiers(clf_dir)
            n_methods = len([k for k in results if not k.startswith('_')])
            print(f"  Found {n_methods} classifier methods")
            # Prefix with directory name if multiple
            if len(args.embedding_classifiers) > 1:
                prefix = os.path.basename(clf_dir)
                results = {f'{prefix}/{k}': v for k, v in results.items()}
            clf_results.update(results)
    else:
        # Auto-detect
        for candidate in ['./embedding_classifiers', 
                           os.path.join(os.path.dirname(args.experiments_base), 
                                        'embedding_classifiers')]:
            if os.path.isdir(candidate):
                print(f"\nAuto-detected embedding classifiers: {candidate}")
                clf_results = scan_embedding_classifiers(candidate)
                n_methods = len([k for k in clf_results if not k.startswith('_')])
                print(f"  Found {n_methods} classifier methods")
                break
    
    # =========================================================================
    # 4. LOAD k-NN RESULTS
    # =========================================================================
    knn_results = {}
    if args.knn_results:
        for knn_dir in args.knn_results:
            print(f"\nLoading k-NN results from: {knn_dir}")
            results = scan_knn_results(knn_dir)
            n_methods = len([k for k in results if not k.startswith('_')])
            print(f"  Found {n_methods} k-NN methods")
            if len(args.knn_results) > 1:
                prefix = os.path.basename(knn_dir)
                results = {f'{prefix}/{k}': v for k, v in results.items()}
            knn_results.update(results)
    else:
        # Auto-detect
        for candidate in ['./knn_baselines', './embedding_analysis',
                           os.path.join(os.path.dirname(args.experiments_base), 
                                        'knn_baselines')]:
            if os.path.isdir(candidate):
                results = scan_knn_results(candidate)
                n_methods = len([k for k in results if not k.startswith('_')])
                if n_methods > 0:
                    print(f"\nAuto-detected k-NN results: {candidate}")
                    print(f"  Found {n_methods} k-NN methods")
                    knn_results = results
                    break
    
    # =========================================================================
    # 5. APPLY SPLIT FILTER
    # =========================================================================
    if split_filter:
        all_experiments = {
            k: [e for e in v if e.get('split', 'unknown') in split_filter]
            for k, v in all_experiments.items()
        }
        all_experiments = {k: v for k, v in all_experiments.items() if v}
    
    # =========================================================================
    # 6. PRINT REPORTS 
    # =========================================================================
    
    # Section A: Old results (original split)
    if old_results:
        print_old_results_table(old_clf_results, "OLD CLASSIFICATION EXPERIMENTS (Original Split)")
        if old_reg_results:
            print_old_results_table(old_reg_results, "OLD REGRESSION EXPERIMENTS (Original Split)")
    
    # Section B: New experiments grouped by split
    splits_seen = set()
    for exp_list in all_experiments.values():
        for e in exp_list:
            splits_seen.add(e.get('split', 'unknown'))
    
    for split in sorted(splits_seen):
        entries = []
        for exp_key, exp_list in sorted(all_experiments.items()):
            for e in exp_list:
                if e.get('split') == split and e.get('metrics'):
                    entries.append({
                        'name': f"{e['dir_name']}",
                        'split': split,
                        'metrics': e['metrics'],
                    })
        
        if entries:
            # Get test set info from first entry
            m0 = entries[0]['metrics']
            n_pos = m0.get('n_positive', '?')
            n_total = m0.get('n_total', '?')
            print_comparison_table(entries, 
                f"NEW NN EXPERIMENTS — Split '{split}' "
                f"(test: {n_total} samples, {n_pos} positives)")
    
    # Section C: Embedding classifiers
    if clf_results:
        clf_entries = []
        for name, data in sorted(clf_results.items()):
            if name.startswith('_'):
                continue
            m = data.get('metrics', {})
            if m:
                clf_entries.append({
                    'name': name,
                    'split': 'orig',
                    'metrics': m,
                })
        if clf_entries:
            print_comparison_table(clf_entries, "EMBEDDING CLASSIFIERS (sklearn on fixed embeddings)")
    
    # Section D: k-NN baselines
    if knn_results:
        knn_entries = []
        for name, data in sorted(knn_results.items()):
            if name.startswith('_'):
                continue
            m = data.get('metrics', {})
            if m:
                knn_entries.append({
                    'name': name,
                    'split': 'orig',
                    'metrics': m,
                })
        if knn_entries:
            print_comparison_table(knn_entries, "k-NN BASELINES (zero-training)")
    
    # Section E: Feature importance
    if args.embedding_classifiers:
        for clf_dir in args.embedding_classifiers:
            print_feature_importance(clf_dir)
    
    # Section F: Per-positive analysis
    if args.verbose:
        print_per_positive_analysis(
            {k: v for k, v in all_experiments.items() 
             if any(e.get('predictions') for e in v)},
            args.threshold)
    
    # Section G: Split meta-comparison
    exp_with_metrics = {
        k: v for k, v in all_experiments.items()
        if any(e.get('metrics') and 'error' not in e.get('metrics', {}) for e in v)
    }
    if len(splits_seen) > 1 and exp_with_metrics:
        print_split_comparison(exp_with_metrics)
    
    # Section H: Executive summary
    print_executive_summary(all_experiments, old_results, clf_results, knn_results)
    
    # =========================================================================
    # 7. SAVE REPORTS
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    save_report_json(args.output_dir, all_experiments, old_results, clf_results, knn_results)
    save_summary_csv(args.output_dir, all_experiments, old_results, clf_results, knn_results)
    
    print(f"\n{'=' * 100}")
    print(f"  Report complete. Output directory: {args.output_dir}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
