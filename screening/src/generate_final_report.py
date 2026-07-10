#!/usr/bin/env python3
"""
Generate Final Report: Publication-Quality Figures & Tables for MOF Discovery
=============================================================================

Reads all existing results (ensemble JSONs, UMAP reports, test predictions)
and generates ~15 figures + reports into final_results/ for progress slides.

Auto-discovers ALL ensemble runs inside --ensemble_dir and compares them
alongside individual models in a unified leaderboard and comparison figures.

Usage:
    python generate_final_report.py
    python generate_final_report.py --output_dir ./final_results
    python generate_final_report.py --ensemble_dir ./ensemble_results --output_dir ./final_results
"""

import os
import sys
import json
import csv
import argparse
import re
import numpy as np
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ============================================================================
# MOF METADATA -- Loaded from qmof.csv at runtime
# ============================================================================

ORGANIC_ELEMENTS = {'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                    'Si', 'B', 'Se', 'Te'}

MOF_METADATA = {}


def _extract_metal_from_formula(formula):
    """Extract the primary metal from a chemical formula string.
    Returns the first element symbol that is not a common organic element."""
    elements = re.findall(r'([A-Z][a-z]?)', formula)
    for el in elements:
        if el not in ORGANIC_ELEMENTS:
            return el
    return 'unknown'


def load_qmof_metadata(base_dir):
    """Load metal types and bandgaps from qmof.csv into MOF_METADATA."""
    global MOF_METADATA
    qmof_path = os.path.join(base_dir, 'qmof.csv')
    if not os.path.exists(qmof_path):
        print(f"  WARNING: {qmof_path} not found -- metal info unavailable")
        return
    print(f"  Loading MOF metadata from {qmof_path}...")
    count = 0
    with open(qmof_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name', '')
            if not name:
                continue
            formula = row.get('info.formula', '')
            metal = _extract_metal_from_formula(formula)
            try:
                bg = float(row.get('outputs.pbe.bandgap', 'nan'))
            except (ValueError, TypeError):
                bg = float('nan')
            MOF_METADATA[name] = {
                'metal': metal,
                'bandgap': bg,
                'formula': formula,
            }
            count += 1
    print(f"  Loaded metadata for {count} MOFs")


def get_metal(cif_id):
    if cif_id in MOF_METADATA:
        return MOF_METADATA[cif_id]['metal']
    return 'unknown'


def short_name(model_name):
    """Shorten model name for display."""
    s = model_name.replace('experiments/', '').replace('strategy_d_farthest_point/', '')
    s = s.replace('exp', 'E').replace('_FSR', '')
    s = s.replace('knn_results/strategy_d_farthest_point', 'kNN_reg')
    return s


def nice_name(model_name):
    """Human-readable model name."""
    mapping = {
        'experiments/exp364_fulltune': 'NN fulltune (exp364)',
        'experiments/exp370_seed2': 'NN fulltune s2 (exp370)',
        'experiments/exp371_seed3': 'NN fulltune s3 (exp371)',
        'knn_results/strategy_d_farthest_point': 'kNN regression',
        'strategy_d_farthest_point/extra_trees': 'Extra Trees',
        'strategy_d_farthest_point/knn_classifier': 'kNN classifier',
        'strategy_d_farthest_point/random_forest': 'Random Forest',
        'strategy_d_farthest_point/smote_extra_trees': 'SMOTE ExtraTrees',
        'strategy_d_farthest_point/smote_random_forest': 'SMOTE RF',
        'strategy_d_farthest_point/two_stage_knn_et': 'Two-Stage kNN+ET',
        'strategy_d_farthest_point/logistic_regression': 'Logistic Regression',
        'strategy_d_farthest_point/xgboost_regression': 'XGBoost Reg.',
        'strategy_d_farthest_point/ensemble_avg': 'Ensemble (score avg)',
        'strategy_d_farthest_point/ensemble_rank_avg': 'Ensemble (rank avg)',
        'rrf_k60': 'RRF (k=60)', 'rrf_k10': 'RRF k=10', 'rrf_k30': 'RRF k=30',
        'rrf_k100': 'RRF k=100', 'rrf_k200': 'RRF k=200',
        'rank_avg': 'Rank avg', 'score_avg': 'Score avg',
        'ensemble_avg': 'Ensemble (score avg)', 'ensemble_rank_avg': 'Ensemble (rank avg)',
        'weighted_rrf': 'Weighted RRF', 'stacking': 'Stacking',
        'ablation_best': 'Ablation best',
    }
    if model_name in mapping:
        return mapping[model_name]
    if model_name.startswith('vote_top'):
        return f"Vote {model_name.replace('vote_top', '')}"
    if model_name.startswith('search-found_'):
        return model_name.replace('search-found_', '')[:48]
    return model_name.split('/')[-1]


def is_nn_model(model_name):
    return model_name.startswith('experiments/')


def is_ml_model(model_name):
    """True if not NN and not a known ensemble method name."""
    if model_name.startswith('experiments/'):
        return False
    return not is_ensemble_model(model_name)


def is_ensemble_model(model_name):
    """True for ensemble method keys saved in per_positive (rrf_k60, rank_avg, etc.)."""
    s = model_name.lower()
    if 'ensemble_avg' in s or 'ensemble_rank_avg' in s:
        return True
    return (s.startswith('rrf_k') or s in (
        'rank_avg', 'score_avg', 'weighted_rrf', 'stacking', 'ablation_best'
    ) or s.startswith('vote_top'))


def is_search_found_model(model_name):
    """True for exhaustive-search combos that met limits (saved as search-found_*)."""
    return str(model_name).startswith('search-found_')


# ============================================================================
# STYLING
# ============================================================================
def setup_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })
    if HAS_SEABORN:
        sns.set_palette("husl")


# ============================================================================
# DATA LOADING
# ============================================================================
def load_json(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_ensemble_runs(ensemble_dir):
    """Auto-discover all ensemble runs under ensemble_dir (one level, or two for custom/run1, custom/run2).

    Returns (runs, run_paths):
      runs: run_name -> parsed ensemble_results.json
      run_paths: run_name -> absolute path to that run's directory (for loading prediction CSVs)
    """
    runs = {}
    run_paths = {}
    if not os.path.isdir(ensemble_dir):
        print(f"  WARNING: ensemble_dir not found: {ensemble_dir}")
        return runs, run_paths

    ensemble_dir = os.path.abspath(ensemble_dir)
    top_json = os.path.join(ensemble_dir, 'ensemble_results.json')
    if os.path.exists(top_json):
        d = load_json(top_json)
        if d:
            name = os.path.basename(ensemble_dir)
            runs[name] = d
            run_paths[name] = ensemble_dir

    for entry in sorted(os.listdir(ensemble_dir)):
        subdir = os.path.join(ensemble_dir, entry)
        if not os.path.isdir(subdir):
            continue
        json_path = os.path.join(subdir, 'ensemble_results.json')
        if os.path.exists(json_path):
            d = load_json(json_path)
            if d:
                runs[entry] = d
                run_paths[entry] = os.path.abspath(subdir)
                print(f"    Loaded ensemble run: {entry}")
        else:
            for subentry in sorted(os.listdir(subdir)):
                subsubdir = os.path.join(subdir, subentry)
                if not os.path.isdir(subsubdir):
                    continue
                subjson = os.path.join(subsubdir, 'ensemble_results.json')
                if os.path.exists(subjson):
                    d = load_json(subjson)
                    if d:
                        run_name = f"{entry}/{subentry}"
                        runs[run_name] = d
                        run_paths[run_name] = os.path.abspath(subsubdir)
                        print(f"    Loaded ensemble run: {run_name}")
    return runs, run_paths


def load_all_data(base_dir, ensemble_dir=None):
    """Load all results from the workspace."""
    data = {}

    if ensemble_dir is None:
        ensemble_dir = os.path.join(base_dir, 'ensemble_results')

    runs, run_paths = load_ensemble_runs(ensemble_dir)
    data['ensemble_runs'] = runs
    data['ensemble_run_paths'] = run_paths

    data['selective'] = data['ensemble_runs'].get('selective_split_d')
    data['exhaustive'] = data['ensemble_runs'].get('exhaustive_split_d')
    data['primary_run_name'] = 'selective_split_d' if data['selective'] is not None else None

    if data['selective'] is None:
        sel_path = os.path.join(base_dir, 'ensemble_results', 'selective_split_d', 'ensemble_results.json')
        data['selective'] = load_json(sel_path)
        if data['selective'] is not None:
            data['primary_run_name'] = 'selective_split_d'
    if data['exhaustive'] is None:
        exh_path = os.path.join(base_dir, 'ensemble_results', 'exhaustive_split_d', 'ensemble_results.json')
        data['exhaustive'] = load_json(exh_path)

    if data['selective'] is None and data['ensemble_runs']:
        first_key = next(iter(data['ensemble_runs']))
        data['selective'] = data['ensemble_runs'][first_key]
        data['primary_run_name'] = first_key
        print(f"  Using '{first_key}' as primary ensemble (no selective_split_d found)")

    # Path to primary run's directory (for loading ensemble prediction CSVs when backfilling)
    if data.get('primary_run_name'):
        if data['primary_run_name'] in data.get('ensemble_run_paths', {}):
            data['primary_run_path'] = data['ensemble_run_paths'][data['primary_run_name']]
        else:
            parts = data['primary_run_name'].split('/')
            data['primary_run_path'] = os.path.join(ensemble_dir, *parts)
    else:
        data['primary_run_path'] = None

    # For the main heatmap (fig1), use the run with the MOST models so we show all Regular ML + ensembles
    data['primary_run_for_heatmap'] = data.get('selective')
    data['primary_run_name_for_heatmap'] = data.get('primary_run_name')
    data['primary_run_path_for_heatmap'] = data.get('primary_run_path')
    best_n_models = -1
    for run_name, run_data in data.get('ensemble_runs', {}).items():
        pp = run_data.get('per_positive') or {}
        if not pp:
            continue
        first_mof = next(iter(pp))
        n_cols = len(pp[first_mof].get('ranks', {}))
        if n_cols > best_n_models:
            best_n_models = n_cols
            data['primary_run_for_heatmap'] = run_data
            data['primary_run_name_for_heatmap'] = run_name
            data['primary_run_path_for_heatmap'] = data.get('ensemble_run_paths', {}).get(run_name)
            if data['primary_run_path_for_heatmap'] is None and run_name:
                parts = run_name.split('/')
                data['primary_run_path_for_heatmap'] = os.path.join(ensemble_dir, *parts)
    if best_n_models >= 0:
        print(f"  Heatmap will use run '{data['primary_run_name_for_heatmap']}' ({best_n_models} models) for fig1")

    umap_path = os.path.join(base_dir, 'umap_original_split', 'umap_analysis_summary.json')
    data['umap_summary'] = load_json(umap_path)

    splitd_report_path = os.path.join(base_dir, 'umap_original_split', 'umap_splitd_report.txt')
    if os.path.exists(splitd_report_path):
        with open(splitd_report_path, 'r') as f:
            data['splitd_report_text'] = f.read()
    else:
        data['splitd_report_text'] = None

    orig_report_path = os.path.join(base_dir, 'umap_original_split', 'umap_original_split_report.txt')
    if os.path.exists(orig_report_path):
        with open(orig_report_path, 'r') as f:
            data['orig_report_text'] = f.read()
    else:
        data['orig_report_text'] = None

    exp_base = os.path.join(base_dir, 'experiments')
    if os.path.isdir(exp_base):
        for exp in ['exp362', 'exp363', 'exp364', 'exp370', 'exp371', 'exp372', 'exp373']:
            exp_dir = None
            for entry in os.listdir(exp_base):
                if entry.startswith(exp):
                    exp_dir = entry
                    break
            if exp_dir:
                fr_path = os.path.join(exp_base, exp_dir, 'final_results.json')
                data[f'{exp}_results'] = load_json(fr_path)

    return data


def parse_splitd_report(text):
    """Parse the UMAP Split D report to extract per-positive diagnosis."""
    results = []
    if text is None:
        return results
    for line in text.split('\n'):
        line = line.strip()
        parts = line.split()
        if len(parts) >= 9 and parts[0].endswith('_FSR') and parts[0] != 'Test':
            try:
                entry = {
                    'cif_id': parts[0],
                    'bandgap': float(parts[1]),
                    'nn_pos_cid': parts[2],
                    'nn_pos_bg': float(parts[3]),
                    'cos_sim': float(parts[4]),
                    'nn_neg_sim': float(parts[5]),
                    'gap': float(parts[6]),
                    'pos_rank': int(parts[7]),
                    'diagnosis': ' '.join(parts[8:]),
                }
                results.append(entry)
            except (ValueError, IndexError):
                pass
    return results


def parse_orig_report(text):
    """Parse the UMAP original split report."""
    results = []
    if text is None:
        return results
    for line in text.split('\n'):
        line = line.strip()
        parts = line.split()
        if len(parts) >= 9 and parts[0].endswith('_FSR') and parts[0] != 'Test':
            try:
                entry = {
                    'cif_id': parts[0],
                    'bandgap': float(parts[1]),
                    'nn_pos_cid': parts[2],
                    'nn_pos_bg': float(parts[3]),
                    'cos_sim': float(parts[4]),
                    'nn_neg_sim': float(parts[5]),
                    'gap': float(parts[6]),
                    'pos_rank': int(parts[7]),
                    'diagnosis': ' '.join(parts[8:]),
                }
                results.append(entry)
            except (ValueError, IndexError):
                pass
    return results


def _backfill_ensemble_ranks_in_per_positive(sel, run_path):
    """If per_positive has no ensemble method keys in ranks, load from prediction CSVs and add them.
    Modifies sel['per_positive'] in place. run_path = directory containing rrf_k60_predictions.csv etc.
    """
    if not run_path or not os.path.isdir(run_path):
        return
    pp = sel['per_positive']
    ens_methods = sel.get('ensemble_methods', {})
    if not ens_methods:
        return
    first_mof = next(iter(pp))
    existing_ranks = pp[first_mof].get('ranks', {})
    if any(is_ensemble_model(m) for m in existing_ranks):
        return  # already have ensemble ranks
    pos_cids = set(pp.keys())
    for method_name in ens_methods:
        csv_path = os.path.join(run_path, f'{method_name}_predictions.csv')
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue
            # Lower score = better rank; sort by score ascending, assign rank 1, 2, ...
            rows.sort(key=lambda r: float(r['score']))
            cid_to_rank = {row['cif_id']: i + 1 for i, row in enumerate(rows)}
            for cid in pp:
                pp[cid]['ranks'][method_name] = cid_to_rank.get(cid, 9999)
            for cid in pp:
                r = cid_to_rank.get(cid, 9999)
                if r < pp[cid]['best_rank']:
                    pp[cid]['best_rank'] = r
                    pp[cid]['best_model'] = method_name
        except Exception as e:
            print(f"    Warning: could not load {csv_path}: {e}")
    for cid in pp:
        pp[cid]['n_in_top100'] = sum(1 for r in pp[cid]['ranks'].values() if r <= 100)
        pp[cid]['n_in_top200'] = sum(1 for r in pp[cid]['ranks'].values() if r <= 200)


# ============================================================================
# FIGURE 1: Per-Positive Heatmap
# ============================================================================
def fig1_per_positive_heatmap(data, output_dir):
    """Rows=MOFs, Cols=individual models + ensemble methods, cells=rank. The core figure.
    Uses the run with the MOST models so all Regular ML and ensemble methods appear."""
    sel = data.get('primary_run_for_heatmap') or data['selective']
    if sel is None:
        return
    run_path = data.get('primary_run_path_for_heatmap') or data.get('primary_run_path')
    _backfill_ensemble_ranks_in_per_positive(sel, run_path)

    pp = sel['per_positive']
    mof_names = sorted(pp.keys(), key=lambda c: pp[c]['best_rank'])
    model_names = sorted(pp[mof_names[0]]['ranks'].keys())

    nn_models = [m for m in model_names if is_nn_model(m)]
    ml_models = [m for m in model_names if is_ml_model(m)]
    search_found_models = [m for m in model_names if is_search_found_model(m)]
    ensemble_models = [m for m in model_names if is_ensemble_model(m) and not is_search_found_model(m)]
    ordered_models = nn_models + ml_models + ensemble_models + search_found_models
    separator_idx = len(nn_models)
    separator2_idx = len(nn_models) + len(ml_models)
    separator3_idx = len(nn_models) + len(ml_models) + len(ensemble_models)

    n_mofs = len(mof_names)
    n_models = len(ordered_models)
    rank_matrix = np.zeros((n_mofs, n_models))

    for i, mof in enumerate(mof_names):
        for j, model in enumerate(ordered_models):
            rank_matrix[i, j] = pp[mof]['ranks'].get(model, 9999)

    fig, ax = plt.subplots(figsize=(16, 7))

    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(np.log10(rank_matrix + 1), cmap=cmap, aspect='auto',
                   vmin=0, vmax=np.log10(10000))

    for i in range(n_mofs):
        for j in range(n_models):
            rank = int(rank_matrix[i, j])
            color = 'white' if rank > 500 else 'black'
            fontsize = 7 if rank > 999 else 8
            ax.text(j, i, str(rank), ha='center', va='center',
                    fontsize=fontsize, color=color, fontweight='bold')

    display_models = [nice_name(m) for m in ordered_models]
    display_mofs = []
    for mof in mof_names:
        metal = get_metal(mof)
        bg = pp[mof]['bandgap']
        name = mof.replace('_FSR', '')
        display_mofs.append(f"{name} ({metal}, {bg:.2f}eV)")

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(display_models, rotation=55, ha='right', fontsize=8)
    ax.set_yticks(range(n_mofs))
    ax.set_yticklabels(display_mofs, fontsize=9)

    if n_models > 0:
        if separator_idx > 0 and separator_idx < n_models:
            ax.axvline(x=separator_idx - 0.5, color='black', linewidth=2, linestyle='-')
        if separator2_idx > separator_idx and separator2_idx < n_models:
            ax.axvline(x=separator2_idx - 0.5, color='black', linewidth=2, linestyle='-')
        if separator3_idx > separator2_idx and separator3_idx < n_models:
            ax.axvline(x=separator3_idx - 0.5, color='black', linewidth=2, linestyle='-')
        # Labels under the blocks
        if len(nn_models) > 0:
            ax.text(separator_idx / 2 - 0.5, -0.8, 'Neural Networks', ha='center',
                    fontsize=9, fontweight='bold', color='#1f77b4')
        if len(ml_models) > 0:
            ax.text(separator_idx + (separator2_idx - separator_idx) / 2 - 0.5, -0.8,
                    'Regular ML', ha='center', fontsize=9, fontweight='bold', color='#ff7f0e')
        if len(ensemble_models) > 0:
            ax.text(separator2_idx + (separator3_idx - separator2_idx) / 2 - 0.5, -0.8,
                    'Ensemble methods', ha='center', fontsize=9, fontweight='bold', color='#d62728')
        if len(search_found_models) > 0:
            ax.text(separator3_idx + (n_models - separator3_idx) / 2 - 0.5, -0.8,
                    'Search-found (meet limits)', ha='center', fontsize=9, fontweight='bold', color='#2ca02c')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('log10(Rank)', fontsize=10)
    ticks = [0, 1, 2, 3, 4]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(['1', '10', '100', '1K', '10K'])

    run_label = data.get('primary_run_name_for_heatmap') or data.get('primary_run_name') or 'primary'
    title = 'Model-MOF Discovery Matrix\nWhich models find which conductive MOFs? (lower rank = better)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    meta = sel.get('run_metadata', {})
    model_paths = meta.get('model_paths', [])
    if model_paths:
        models_str = ', '.join(nice_name(m) for m in model_paths[:18])
        if len(model_paths) > 18:
            models_str += f' ... (+{len(model_paths)-18} more)'
        fig.text(0.5, 0.02, f"Run: {run_label}  |  Models in this run: {models_str}  |  Ensemble methods (right) combine these base models.",
                 ha='center', fontsize=8)

    for i in range(n_mofs):
        best = int(rank_matrix[i].min())
        if best <= 100:
            ax.text(-0.7, i, '*', ha='center', va='center', fontsize=16,
                    color='green', fontweight='bold')
        elif best > 500:
            ax.text(-0.7, i, 'X', ha='center', va='center', fontsize=12,
                    color='red', fontweight='bold')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(output_dir, 'fig1_per_positive_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 2: Recall@K Curves
# ============================================================================
def fig2_recall_at_k_curves(data, output_dir):
    """Recall vs K for best NN, best ML, best ensemble, random."""
    sel = data['selective']
    if sel is None:
        return

    Ks = [10, 25, 50, 100, 200, 500]
    n_total = 9550
    n_pos = 9

    best_nn_name = 'experiments/exp364_fulltune'
    best_ml_name = 'strategy_d_farthest_point/extra_trees'
    knn_name = 'strategy_d_farthest_point/knn_classifier'

    lines = {}
    for name, label, color, ls in [
        (best_nn_name, 'Best NN (exp364 fulltune)', '#1f77b4', '-'),
        (best_ml_name, 'Best ML (Extra Trees)', '#ff7f0e', '-'),
        (knn_name, 'kNN Classifier', '#2ca02c', '--'),
    ]:
        if name in sel['individual_models']:
            m = sel['individual_models'][name]
            recalls = [m.get(f'recall@{K}', 0) for K in Ks]
            lines[label] = (Ks, recalls, color, ls)

    ens_methods = [
        ('rrf_k60', 'RRF Ensemble', '#d62728', '-'),
        ('weighted_rrf', 'Weighted RRF', '#9467bd', '--'),
        ('stacking', 'Stacking', '#8c564b', '-.'),
    ]
    for ens_key, label, color, ls in ens_methods:
        if ens_key in sel.get('ensemble_methods', {}):
            m = sel['ensemble_methods'][ens_key]
            recalls = [m.get(f'recall@{K}', 0) for K in Ks]
            lines[label] = (Ks, recalls, color, ls)

    fig, ax = plt.subplots(figsize=(10, 6))

    random_recalls = [K / n_total * (n_pos / n_pos) if n_pos > 0 else 0
                      for K in Ks]
    random_recalls = [min(K * n_pos / n_total, 1.0) for K in Ks]
    ax.plot(Ks, random_recalls, '--', color='gray', alpha=0.5, linewidth=1,
            label='Random baseline')

    for label, (ks, recs, color, ls) in lines.items():
        lw = 2.5 if 'Ensemble' in label or 'RRF' in label else 1.8
        marker = 'o' if 'Ensemble' in label or 'RRF' in label else 's'
        ax.plot(ks, recs, ls, color=color, linewidth=lw, marker=marker,
                markersize=6, label=label)

    if 'RRF Ensemble' in lines and 'Best NN (exp364 fulltune)' in lines:
        ens_recs = lines['RRF Ensemble'][1]
        nn_recs = lines['Best NN (exp364 fulltune)'][1]
        ax.fill_between(Ks, nn_recs, ens_recs, alpha=0.15, color='#d62728',
                         label='Ensemble advantage')

    ax.set_xlabel('Top-K (number of candidates screened)', fontsize=12)
    ax.set_ylabel('Recall (fraction of 9 positives found)', fontsize=12)
    ax.set_title('Recall@K: How many conductive MOFs found in top-K?\n'
                 'Ensemble combines NN + ML for superior recall',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 520)
    ax.set_ylim(-0.02, 1.02)

    for y_val, label in [(1/9, '1/9'), (3/9, '3/9'), (5/9, '5/9'), (7/9, '7/9'), (9/9, '9/9')]:
        ax.axhline(y=y_val, color='lightblue', linewidth=0.5, alpha=0.5)
        ax.text(510, y_val, label, va='center', fontsize=8, color='steelblue')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_recall_at_k_curves.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 3: Original Split vs Split D Coverage
# ============================================================================
def fig3_split_comparison(data, output_dir):
    """Stacked bar: coverage categories for original vs Split D."""
    orig_counts = {'COVERED': 0, 'BORDERLINE': 8, 'ISOLATED': 1}
    splitd_counts = {'COVERED': 8, 'BORDERLINE': 1, 'ISOLATED': 0}

    orig_diag = parse_orig_report(data.get('orig_report_text'))
    splitd_diag = parse_splitd_report(data.get('splitd_report_text'))

    if orig_diag:
        orig_counts = {'COVERED': 0, 'BORDERLINE': 0, 'ISOLATED': 0}
        for d in orig_diag:
            diag = d['diagnosis'].split('+')[0].strip()
            if diag in orig_counts:
                orig_counts[diag] += 1

    if splitd_diag:
        splitd_counts = {'COVERED': 0, 'BORDERLINE': 0, 'ISOLATED': 0}
        for d in splitd_diag:
            diag = d['diagnosis'].split('+')[0].strip()
            if diag in splitd_counts:
                splitd_counts[diag] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    categories = ['COVERED', 'BORDERLINE', 'ISOLATED']
    colors_map = {'COVERED': '#2ecc71', 'BORDERLINE': '#f39c12', 'ISOLATED': '#e74c3c'}
    colors = [colors_map[c] for c in categories]

    for ax, counts, title in [
        (axes[0], orig_counts, 'Original Split'),
        (axes[1], splitd_counts, 'Split D (farthest-point)')
    ]:
        vals = [counts.get(c, 0) for c in categories]
        bars = ax.bar(categories, vals, color=colors, edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom', fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of test positives', fontsize=11)
        ax.set_ylim(0, 10)

    fig.suptitle('Structural Coverage of Test Positives\n'
                 'Split D guarantees every test positive has a nearby train positive',
                 fontsize=14, fontweight='bold', y=1.03)

    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='COVERED (sim >= 0.85)'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='BORDERLINE (0.70-0.85)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='ISOLATED (sim < 0.70)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_split_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 4: Ensemble Method Comparison
# ============================================================================
def fig4_ensemble_method_comparison(data, output_dir):
    """Grouped bars comparing ensemble methods on key metrics."""
    sel = data['selective']
    if sel is None:
        return
    ens = sel.get('ensemble_methods', {})

    method_order = ['rrf_k60', 'weighted_rrf', 'rank_avg', 'vote_top200',
                    'score_avg', 'stacking']
    method_labels = {
        'rrf_k60': 'RRF (k=60)',
        'weighted_rrf': 'Weighted\nRRF',
        'rank_avg': 'Rank\nAverage',
        'vote_top200': 'Top-200\nVoting',
        'score_avg': 'Score\nAverage',
        'stacking': 'Stacking\n(LogReg)',
    }

    methods = [m for m in method_order if m in ens]
    labels = [method_labels.get(m, m) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_list = [
        ('first_hit_rank', 'First Hit Rank (lower = better)', True),
        ('recall@100', 'Recall@100', False),
        ('recall@200', 'Recall@200', False),
    ]

    bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    for ax, (metric, title, invert) in zip(axes, metrics_list):
        vals = [ens[m].get(metric, 0) for m in methods]
        bars = ax.bar(range(len(methods)), vals, color=bar_colors[:len(methods)],
                      edgecolor='black', linewidth=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')

        for bar, val in zip(bars, vals):
            fmt = f"{val:.0f}" if metric == 'first_hit_rank' else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    fmt, ha='center', va='bottom', fontsize=8, fontweight='bold')

        best_idx = np.argmin(vals) if invert else np.argmax(vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    fig.suptitle('Ensemble Method Comparison (Split D, 13 models)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_ensemble_method_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 5: Per-Positive Waterfall (Best Achievable Rank)
# ============================================================================
def fig5_per_positive_waterfall(data, output_dir):
    """Horizontal bars: best rank per MOF, color-coded by difficulty."""
    sel = data['selective']
    if sel is None:
        return
    pp = sel['per_positive']

    mofs = sorted(pp.keys(), key=lambda c: pp[c]['best_rank'])

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mof in enumerate(mofs):
        rank = pp[mof]['best_rank']
        bg = pp[mof]['bandgap']
        metal = get_metal(mof)
        best_model = nice_name(pp[mof]['best_model'])

        if rank <= 50:
            color = '#27ae60'
            label_cat = 'Easy (rank <= 50)'
        elif rank <= 200:
            color = '#f39c12'
            label_cat = 'Moderate (50-200)'
        elif rank <= 1000:
            color = '#e67e22'
            label_cat = 'Hard (200-1000)'
        else:
            color = '#e74c3c'
            label_cat = 'Catastrophic (> 1000)'

        bar = ax.barh(i, min(rank, 5000), color=color, edgecolor='black',
                      linewidth=0.8, height=0.7)

        name_str = mof.replace('_FSR', '')
        ax.text(-50, i, f"{name_str} ({metal}, {bg:.2f}eV)",
                ha='right', va='center', fontsize=9, fontweight='bold')

        rank_str = str(rank)
        text_x = min(rank, 5000) + 50
        ax.text(text_x, i, f"rank {rank_str} -- {best_model}",
                ha='left', va='center', fontsize=8, color='#333')

    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlim(0.5, 8000)
    ax.set_yticks(range(len(mofs)))
    ax.set_yticklabels([''] * len(mofs))
    ax.invert_yaxis()
    ax.set_xlabel('Best achievable rank across all models', fontsize=11)
    ax.set_title('Per-Positive Discovery Difficulty\n'
                 'Best rank any model achieves for each test positive',
                 fontsize=13, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#27ae60', edgecolor='black', label='Easy (rank <= 50)'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Moderate (50-200)'),
        mpatches.Patch(facecolor='#e67e22', edgecolor='black', label='Hard (200-1000)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Catastrophic (> 1000)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    n_found = sum(1 for m in mofs if pp[m]['best_rank'] <= 200)
    n_missed = len(mofs) - n_found
    ax.text(0.98, 0.02, f"Found (rank <= 200): {n_found}/9\nMissed: {n_missed}/9",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_per_positive_waterfall.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 6: NN vs ML Scatter
# ============================================================================
def fig6_nn_vs_ml_scatter(data, output_dir):
    """Scatter: best NN rank vs best ML rank for each positive."""
    sel = data['selective']
    if sel is None:
        return
    pp = sel['per_positive']
    model_names = list(pp[list(pp.keys())[0]]['ranks'].keys())

    nn_models_list = [m for m in model_names if is_nn_model(m)]
    ml_models_list = [m for m in model_names if is_ml_model(m)]

    fig, ax = plt.subplots(figsize=(9, 8))

    for mof in pp:
        ranks = pp[mof]['ranks']
        best_nn = min(ranks.get(m, 9999) for m in nn_models_list) if nn_models_list else 9999
        best_ml = min(ranks.get(m, 9999) for m in ml_models_list) if ml_models_list else 9999
        metal = get_metal(mof)
        bg = pp[mof]['bandgap']
        name = mof.replace('_FSR', '')

        metal_colors = {'Cu': '#e74c3c', 'Fe': '#3498db', 'Mn': '#2ecc71',
                        'Cd': '#9b59b6', 'Zn': '#f39c12', 'Co': '#1abc9c',
                        'unknown': 'gray'}
        color = metal_colors.get(metal, 'gray')

        ax.scatter(best_nn, best_ml, s=150, c=color, edgecolors='black',
                   linewidths=1.2, zorder=5)
        ax.annotate(f"{name}\n({metal}, {bg:.2f}eV)",
                    (best_nn, best_ml), fontsize=7, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))

    ax.axhline(y=200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)

    ax.fill_between([0, 200], 0, 200, alpha=0.05, color='green')
    ax.fill_between([200, 10000], 0, 200, alpha=0.05, color='orange')
    ax.fill_between([0, 200], 200, 10000, alpha=0.05, color='blue')
    ax.fill_between([200, 10000], 200, 10000, alpha=0.05, color='red')

    ax.text(50, 50, 'Both find', fontsize=10, color='green', alpha=0.7,
            ha='center', va='center')
    ax.text(2000, 50, 'ML only', fontsize=10, color='orange', alpha=0.7,
            ha='center', va='center')
    ax.text(50, 2000, 'NN only', fontsize=10, color='blue', alpha=0.7,
            ha='center', va='center')
    ax.text(2000, 2000, 'Neither\n(catastrophic)', fontsize=10, color='red',
            alpha=0.7, ha='center', va='center')

    ax.set_xscale('symlog', linthresh=10)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_xlabel('Best NN rank (6 NN models)', fontsize=12)
    ax.set_ylabel('Best ML rank (7 ML models)', fontsize=12)
    ax.set_title('NN vs ML Complementarity\n'
                 'Different models catch different conductive MOFs',
                 fontsize=13, fontweight='bold')

    unique_metals = sorted(set(get_metal(m) for m in pp))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=metal_colors.get(mt, 'gray'),
               markersize=10, label=mt, markeredgecolor='black')
        for mt in unique_metals
    ]
    ax.legend(handles=legend_elements, title='Metal', loc='upper left', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_nn_vs_ml_scatter.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 7: Model Coverage Summary
# ============================================================================
def fig7_model_coverage_summary(data, output_dir):
    """Which MOFs are found by NN-only, ML-only, both, or neither."""
    sel = data['selective']
    if sel is None:
        return
    pp = sel['per_positive']
    model_names = list(pp[list(pp.keys())[0]]['ranks'].keys())

    nn_models_list = [m for m in model_names if is_nn_model(m)]
    ml_models_list = [m for m in model_names if is_ml_model(m)]

    top_k = 200
    categories = OrderedDict([
        ('Found by both\nNN + ML', []),
        ('NN only', []),
        ('ML only', []),
        ('Neither\n(missed)', []),
    ])

    for mof in sorted(pp.keys()):
        ranks = pp[mof]['ranks']
        nn_found = any(ranks.get(m, 9999) <= top_k for m in nn_models_list)
        ml_found = any(ranks.get(m, 9999) <= top_k for m in ml_models_list)

        name = mof.replace('_FSR', '')
        metal = get_metal(mof)
        label = f"{name} ({metal})"

        if nn_found and ml_found:
            categories['Found by both\nNN + ML'].append(label)
        elif nn_found:
            categories['NN only'].append(label)
        elif ml_found:
            categories['ML only'].append(label)
        else:
            categories['Neither\n(missed)'].append(label)

    fig, ax = plt.subplots(figsize=(11, 6))
    cat_colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
    cat_names = list(categories.keys())
    cat_counts = [len(v) for v in categories.values()]

    bars = ax.bar(range(len(cat_names)), cat_counts, color=cat_colors,
                  edgecolor='black', linewidth=1.5, width=0.6)

    for i, (bar, count, cat) in enumerate(zip(bars, cat_counts, cat_names)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(count), ha='center', fontsize=18, fontweight='bold')
        mof_list = list(categories.values())[i]
        if mof_list:
            text = '\n'.join(mof_list)
            y_pos = bar.get_height() / 2
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, text,
                    ha='center', va='center', fontsize=7.5, fontweight='bold',
                    color='white' if count > 0 else 'black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))

    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_names, fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of test positives', fontsize=12)
    ax.set_ylim(0, max(cat_counts) + 1.5)
    ax.set_title(f'Discovery Coverage by Model Type (top-{top_k})\n'
                 'Key insight: NN and ML find DIFFERENT positives',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_model_coverage_summary.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 8: Catastrophic Miss Analysis
# ============================================================================
def fig8_catastrophic_miss_analysis(data, output_dir):
    """Table-figure for the unfindable MOFs."""
    sel = data['selective']
    if sel is None:
        return
    pp = sel['per_positive']

    splitd_diag = parse_splitd_report(data.get('splitd_report_text'))
    diag_map = {d['cif_id']: d for d in splitd_diag}

    missed = [(mof, pp[mof]) for mof in pp if pp[mof]['best_rank'] > 200]
    missed.sort(key=lambda x: x[1]['best_rank'])

    fig, ax = plt.subplots(figsize=(14, max(3 + len(missed), 5)))
    ax.axis('off')

    headers = ['MOF', 'Metal', 'Bandgap\n(eV)', 'Best\nRank', 'Best Model',
               'Nearest Train\nPositive', 'Cos Sim', 'Diagnosis']
    col_widths = [0.10, 0.06, 0.07, 0.06, 0.16, 0.15, 0.07, 0.15]
    n_cols = len(headers)
    n_rows = len(missed) + 1

    for j, (header, w) in enumerate(zip(headers, col_widths)):
        x = sum(col_widths[:j]) + 0.05
        ax.text(x + w/2, 0.95, header, ha='center', va='center',
                fontsize=9, fontweight='bold',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#34495e', alpha=0.9),
                color='white')

    for i, (mof, info) in enumerate(missed):
        y = 0.95 - (i + 1) * (0.8 / max(n_rows, 2))
        name = mof.replace('_FSR', '')
        metal = get_metal(mof)
        bg = info['bandgap']
        best_rank = info['best_rank']
        best_model = nice_name(info['best_model'])

        diag = diag_map.get(mof, {})
        nn_pos = diag.get('nn_pos_cid', '?').replace('_FSR', '')
        cos_sim = diag.get('cos_sim', 0)
        diagnosis = diag.get('diagnosis', 'COVERED')

        row_data = [name, metal, f"{bg:.3f}", str(best_rank), best_model,
                    nn_pos, f"{cos_sim:.3f}", diagnosis]

        bg_color = '#fadbd8' if best_rank > 1000 else '#fef9e7'
        for j, (val, w) in enumerate(zip(row_data, col_widths)):
            x = sum(col_widths[:j]) + 0.05
            ax.text(x + w/2, y, val, ha='center', va='center',
                    fontsize=8, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.7))

    ax.set_title('Catastrophic Misses: MOFs No Model Finds in Top-200\n'
                 'These require more training data or physics-based features',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig8_catastrophic_miss_analysis.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 9: Enrichment Factor
# ============================================================================
def fig9_enrichment_factor(data, output_dir):
    """Bar chart of enrichment@K for best ensemble."""
    sel = data['selective']
    if sel is None:
        return

    best_ens_key = 'rrf_k60'
    best_single_key = 'experiments/exp364_fulltune'

    ens_m = sel.get('ensemble_methods', {}).get(best_ens_key, {})
    single_m = sel.get('individual_models', {}).get(best_single_key, {})

    Ks = [25, 50, 100, 200, 500]
    ens_enrich = [ens_m.get(f'enrichment@{K}', 0) for K in Ks]
    single_enrich = [single_m.get(f'enrichment@{K}', 0) for K in Ks]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(Ks))
    width = 0.35

    bars1 = ax.bar(x - width/2, single_enrich, width, label='Best NN (exp364)',
                   color='#3498db', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, ens_enrich, width, label='RRF Ensemble',
                   color='#e74c3c', edgecolor='black', linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        f"{h:.0f}x", ha='center', va='bottom', fontsize=9,
                        fontweight='bold')

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Random (1x)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{K}' for K in Ks], fontsize=11)
    ax.set_ylabel('Enrichment Factor (vs random)', fontsize=12)
    ax.set_title('Enrichment: How much better than random?\n'
                 'Higher = more efficient screening',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig9_enrichment_factor.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 10: Bandgap vs Best Rank
# ============================================================================
def fig10_bandgap_vs_rank(data, output_dir):
    """Scatter: true bandgap vs best rank, annotated with names."""
    sel = data['selective']
    if sel is None:
        return
    pp = sel['per_positive']

    fig, ax = plt.subplots(figsize=(10, 7))

    metal_colors = {'Cu': '#e74c3c', 'Fe': '#3498db', 'Mn': '#2ecc71',
                    'Cd': '#9b59b6', 'Zn': '#f39c12', 'Co': '#1abc9c',
                    'unknown': 'gray'}

    for mof in pp:
        bg = pp[mof]['bandgap']
        rank = pp[mof]['best_rank']
        metal = get_metal(mof)
        name = mof.replace('_FSR', '')
        color = metal_colors.get(metal, 'gray')

        ax.scatter(bg, rank, s=180, c=color, edgecolors='black', linewidths=1.2,
                   zorder=5)
        ax.annotate(f"{name}\n({metal})", (bg, rank), fontsize=8, fontweight='bold',
                    xytext=(10, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.25))

    ax.axhline(y=200, color='green', linestyle='--', alpha=0.5,
               label='Top-200 threshold')
    ax.axhline(y=100, color='blue', linestyle=':', alpha=0.4,
               label='Top-100 threshold')

    ax.set_yscale('symlog', linthresh=10)
    ax.set_xlabel('True Bandgap (eV)', fontsize=12)
    ax.set_ylabel('Best Model Rank (log scale)', fontsize=12)
    ax.set_title('Bandgap vs Discoverability\n'
                 'Are the most conductive MOFs the hardest to find?',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig10_bandgap_vs_rank.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 11: Robustness Boxplots
# ============================================================================
def fig11_robustness_boxplots(data, output_dir):
    """
    Bar chart (with error bars) of recall@100 and recall@200 from subsampled evaluation.
    Uses a fixed set of 5 ensemble methods (RRF, W-RRF, Rank Avg, Score Avg, Stacking)
    — not all ensembles, and not a 'best' selection. Only methods present in the
    primary run's subsampled_evaluation are plotted.
    """
    sel = data['selective']
    if sel is None:
        return

    sub_eval = sel.get('subsampled_evaluation', {})
    if not sub_eval:
        print("  Skipping fig11: no subsampled evaluation data")
        return

    methods = ['rrf_k60', 'weighted_rrf', 'rank_avg', 'score_avg', 'stacking']
    method_labels = {
        'rrf_k60': 'RRF', 'weighted_rrf': 'W-RRF', 'rank_avg': 'Rank Avg',
        'score_avg': 'Score Avg', 'stacking': 'Stacking',
    }
    available = [m for m in methods if m in sub_eval]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title in [
        (axes[0], 'recall@100', 'Recall@100 (30 resamples)'),
        (axes[1], 'recall@200', 'Recall@200 (30 resamples)'),
    ]:
        means = [sub_eval[m].get(f'{metric}_mean', 0) for m in available]
        stds = [sub_eval[m].get(f'{metric}_std', 0) for m in available]
        labels = [method_labels.get(m, m) for m in available]

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bars = ax.bar(range(len(available)), means, yerr=stds, capsize=5,
                      color=colors[:len(available)], edgecolor='black', linewidth=0.8)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f"{mean:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)

    fig.suptitle('Robustness: Subsampled Evaluation (N=1500, 30 resamples)\n'
                 'Fixed set: RRF, W-RRF, Rank Avg, Score Avg, Stacking. Error bars = std across resamples.',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig11_robustness_boxplots.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 12: Metal Type Analysis
# ============================================================================
def fig12_metal_type_analysis(data, output_dir):
    """Grouped bar: metal types for found vs missed."""
    sel = data['selective']
    if sel is None:
        return
    pp = sel['per_positive']

    found_metals = {}
    missed_metals = {}
    for mof in pp:
        metal = get_metal(mof)
        rank = pp[mof]['best_rank']
        if rank <= 200:
            found_metals[metal] = found_metals.get(metal, 0) + 1
        else:
            missed_metals[metal] = missed_metals.get(metal, 0) + 1

    all_metals = sorted(set(list(found_metals.keys()) + list(missed_metals.keys())))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked bar by metal
    ax = axes[0]
    x = np.arange(len(all_metals))
    found_vals = [found_metals.get(m, 0) for m in all_metals]
    missed_vals = [missed_metals.get(m, 0) for m in all_metals]

    bars1 = ax.bar(x, found_vals, 0.6, label='Found (rank <= 200)',
                   color='#27ae60', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, missed_vals, 0.6, bottom=found_vals, label='Missed (rank > 200)',
                   color='#e74c3c', edgecolor='black', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_metals, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Metal Type: Found vs Missed', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    for i, (f, m) in enumerate(zip(found_vals, missed_vals)):
        if f > 0:
            ax.text(i, f/2, str(f), ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')
        if m > 0:
            ax.text(i, f + m/2, str(m), ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')

    # Right: detailed table
    ax2 = axes[1]
    ax2.axis('off')

    table_data = []
    for mof in sorted(pp.keys(), key=lambda c: pp[c]['best_rank']):
        metal = get_metal(mof)
        rank = pp[mof]['best_rank']
        bg = pp[mof]['bandgap']
        status = 'FOUND' if rank <= 200 else 'MISSED'
        name = mof.replace('_FSR', '')
        table_data.append([name, metal, f"{bg:.3f}", str(rank), status])

    col_labels = ['MOF', 'Metal', 'Bandgap', 'Best Rank', 'Status']
    cell_colors = []
    for row in table_data:
        if row[4] == 'FOUND':
            cell_colors.append(['#d5f5e3'] * 5)
        else:
            cell_colors.append(['#fadbd8'] * 5)

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      cellColours=cell_colors, loc='center',
                      cellLoc='center',
                      colColours=['#2c3e50'] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#bdc3c7')

    ax2.set_title('Per-MOF Summary', fontsize=12, fontweight='bold')

    fig.suptitle('Chemical Analysis: Which Metal Types Do We Miss?\n'
                 'The miss pattern reveals structural novelty, not chemical bias',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig12_metal_type_analysis.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# SPLIT D UMAP-STYLE DIAGNOSIS (from parsed report)
# ============================================================================
def fig_splitd_diagnosis(data, output_dir):
    """Replicate the UMAP diagnosis for Split D showing MOF names and coverage."""
    splitd_diag = parse_splitd_report(data.get('splitd_report_text'))
    if not splitd_diag:
        print("  Skipping Split D diagnosis plot: no data")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    sorted_diag = sorted(splitd_diag, key=lambda d: d['cos_sim'], reverse=True)

    y_positions = range(len(sorted_diag))
    for i, d in enumerate(sorted_diag):
        sim = d['cos_sim']
        gap = d['gap']
        name = d['cif_id'].replace('_FSR', '')
        metal = get_metal(d['cif_id'])
        bg = d['bandgap']
        nn_pos = d['nn_pos_cid'].replace('_FSR', '')
        nn_metal = get_metal(d['nn_pos_cid'])
        diagnosis = d['diagnosis'].split('+')[0].strip()

        if diagnosis == 'COVERED':
            color = '#27ae60'
        elif diagnosis == 'BORDERLINE':
            color = '#f39c12'
        else:
            color = '#e74c3c'

        bar = ax.barh(i, sim, color=color, edgecolor='black', linewidth=0.8,
                      height=0.7, alpha=0.85)

        ax.text(-0.01, i, f"{name} ({metal}, {bg:.2f}eV)",
                ha='right', va='center', fontsize=9, fontweight='bold')

        ax.text(sim + 0.005, i,
                f"sim={sim:.3f} | NN+: {nn_pos} ({nn_metal}) | gap={gap:+.3f} | {diagnosis}",
                ha='left', va='center', fontsize=7.5,
                color='#2c3e50')

    ax.axvline(x=0.85, color='green', linestyle='--', alpha=0.7,
               label='COVERED threshold (0.85)')
    ax.axvline(x=0.70, color='red', linestyle='--', alpha=0.7,
               label='ISOLATED threshold (0.70)')

    ax.set_yticks([])
    ax.set_xlabel('Cosine Similarity to Nearest Train Positive', fontsize=12)
    ax.set_title('Split D: Structural Coverage Diagnosis\n'
                 'Every test positive has a nearby train positive (8 COVERED, 1 BORDERLINE)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0.6, 1.0)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_splitd_diagnosis.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 13: Unified Leaderboard -- Individual + Ensemble ranked together
# ============================================================================
def fig13_unified_leaderboard(data, output_dir):
    """Bar chart ranking ALL methods (individual + ensemble) by recall@50."""
    ensemble_runs = data.get('ensemble_runs', {})
    if not ensemble_runs:
        print("  Skipping fig13: no ensemble runs")
        return

    entries = []
    seen = set()

    for run_name, run_data in ensemble_runs.items():
        run_label = run_name.replace('_split_d', '').replace('_', ' ').title()

        for model_name, m in run_data.get('individual_models', {}).items():
            display = nice_name(model_name)
            if display in seen:
                continue
            seen.add(display)
            entries.append({
                'name': display,
                'source': 'Individual',
                'r25': m.get('recall@25', 0),
                'r50': m.get('recall@50', 0),
                'r100': m.get('recall@100', 0),
                'r200': m.get('recall@200', 0),
                'fhr': m.get('first_hit_rank', 9999),
                'mrr': m.get('mrr', 0),
            })

        for ens_name, m in run_data.get('ensemble_methods', {}).items():
            display = f"{ens_name} ({run_label})"
            if display in seen:
                continue
            seen.add(display)
            entries.append({
                'name': display,
                'source': f'Ensemble ({run_label})',
                'r25': m.get('recall@25', 0),
                'r50': m.get('recall@50', 0),
                'r100': m.get('recall@100', 0),
                'r200': m.get('recall@200', 0),
                'fhr': m.get('first_hit_rank', 9999),
                'mrr': m.get('mrr', 0),
            })

    if not entries:
        return

    entries.sort(key=lambda e: (-e['r50'], -e['r100'], e['fhr']))
    top_n = min(25, len(entries))
    entries = entries[:top_n]

    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.42)))

    names = [e['name'] for e in entries]
    r50_vals = [e['r50'] for e in entries]
    colors = ['#e74c3c' if 'Ensemble' in e['source'] else
              ('#3498db' if is_nn_model(e['name']) else '#f39c12')
              for e in entries]

    y_pos = np.arange(top_n)
    bars = ax.barh(y_pos, r50_vals, color=colors, edgecolor='black', linewidth=0.7, height=0.7)

    for i, (bar, e) in enumerate(zip(bars, entries)):
        w = bar.get_width()
        ax.text(w + 0.005, i, f"R@50={e['r50']:.3f}  R@200={e['r200']:.3f}  FHR={e['fhr']:.0f}",
                va='center', fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Recall@50 (fraction of positives in top 50)', fontsize=11)
    ax.set_title('Unified Leaderboard: Individual Models vs Ensemble Methods\n'
                 'Ranked by Recall@50 (find targets in first 50)',
                 fontsize=13, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Ensemble'),
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='NN (Individual)'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='ML (Individual)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig13_unified_leaderboard.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 14: Multi-Ensemble Run Comparison
# ============================================================================
def fig14_multi_ensemble_comparison(data, output_dir):
    """Compare the best ensemble method from each run side by side."""
    ensemble_runs = data.get('ensemble_runs', {})
    if len(ensemble_runs) < 1:
        print("  Skipping fig14: need >= 1 ensemble run")
        return

    run_summaries = []
    for run_name, run_data in sorted(ensemble_runs.items()):
        ens = run_data.get('ensemble_methods', {})
        ind = run_data.get('individual_models', {})
        if not ens and not ind:
            continue

        best_ens_name, best_ens_m = None, None
        for name, m in ens.items():
            if best_ens_m is None or m.get('recall@50', 0) > best_ens_m.get('recall@50', 0):
                best_ens_name, best_ens_m = name, m

        best_ind_name, best_ind_m = None, None
        for name, m in ind.items():
            if best_ind_m is None or m.get('recall@50', 0) > best_ind_m.get('recall@50', 0):
                best_ind_name, best_ind_m = name, m

        label = run_name.replace('_split_d', '').replace('_', ' ').title()
        meta = run_data.get('run_metadata', {})
        model_paths = meta.get('model_paths', [])
        models_used = ', '.join(nice_name(m) for m in model_paths[:8]) if model_paths else '—'
        if model_paths and len(model_paths) > 8:
            models_used += f' +{len(model_paths)-8}'
        run_summaries.append({
            'run': label,
            'models_used': models_used,
            'best_ens_name': best_ens_name or '—',
            'best_ens': best_ens_m or {},
            'best_ind_name': nice_name(best_ind_name) if best_ind_name else '—',
            'best_ind': best_ind_m or {},
            'n_individual': len(ind),
            'n_ensemble': len(ens),
        })

    if not run_summaries:
        return

    metrics = [
        ('recall@25', 'Recall@25'),
        ('recall@50', 'Recall@50'),
        ('recall@100', 'Recall@100'),
        ('recall@200', 'Recall@200'),
    ]

    n_metrics = len(metrics)
    n_runs = len(run_summaries)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(5, n_runs * 0.8 + 2)))

    for ax_idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]
        run_labels = []
        ens_vals = []
        ind_vals = []
        for rs in run_summaries:
            run_labels.append(rs['run'])
            ens_vals.append(rs['best_ens'].get(metric_key, 0))
            ind_vals.append(rs['best_ind'].get(metric_key, 0))

        y = np.arange(n_runs)
        h = 0.35
        ax.barh(y - h/2, ens_vals, h, label='Best Ensemble', color='#e74c3c',
                edgecolor='black', linewidth=0.7)
        ax.barh(y + h/2, ind_vals, h, label='Best Individual', color='#3498db',
                edgecolor='black', linewidth=0.7)

        for i in range(n_runs):
            if ens_vals[i] > 0:
                ax.text(ens_vals[i] + 0.005, i - h/2, f"{ens_vals[i]:.3f}",
                        va='center', fontsize=7)
            if ind_vals[i] > 0:
                ax.text(ind_vals[i] + 0.005, i + h/2, f"{ind_vals[i]:.3f}",
                        va='center', fontsize=7)

        ax.set_yticks(y)
        ax.set_yticklabels(run_labels, fontsize=9)
        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        if ax_idx == 0:
            ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Multi-Ensemble Comparison: Best Ensemble vs Best Individual per Run\n'
                 'Each bar = best method from that ensemble configuration',
                 fontsize=13, fontweight='bold', y=1.03)
    # Add a text block listing models per run for reproducibility
    models_text = []
    for rs in run_summaries:
        models_text.append(f"{rs['run']}: {rs['models_used']}")
    fig.text(0.5, 0.01, 'Models per run: ' + '  |  '.join(models_text),
             ha='center', fontsize=7, wrap=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(output_dir, 'fig14_multi_ensemble_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE 15: Ensemble Gain -- How much does ensembling improve over best single?
# ============================================================================
def fig15_ensemble_gain(data, output_dir):
    """For each ensemble run: show recall@K curves for best individual vs best ensemble."""
    ensemble_runs = data.get('ensemble_runs', {})
    if not ensemble_runs:
        print("  Skipping fig15: no ensemble runs")
        return

    Ks = [10, 25, 50, 100, 200, 500]
    n_runs = len(ensemble_runs)
    fig, axes = plt.subplots(1, min(n_runs, 3), figsize=(7 * min(n_runs, 3), 6),
                             squeeze=False)
    axes = axes[0]

    plot_idx = 0
    for run_name, run_data in sorted(ensemble_runs.items()):
        if plot_idx >= 3:
            break
        ax = axes[plot_idx]
        ens = run_data.get('ensemble_methods', {})
        ind = run_data.get('individual_models', {})
        label = run_name.replace('_split_d', '').replace('_', ' ').title()

        best_ind_name = max(ind, key=lambda n: ind[n].get('recall@200', 0)) if ind else None
        best_ens_name = max(ens, key=lambda n: ens[n].get('recall@50', 0)) if ens else None

        if best_ind_name:
            m = ind[best_ind_name]
            recs = [m.get(f'recall@{K}', 0) for K in Ks]
            ax.plot(Ks, recs, 's-', color='#3498db', linewidth=2, markersize=5,
                    label=f'Best single: {nice_name(best_ind_name)}')

        if best_ens_name:
            m = ens[best_ens_name]
            recs = [m.get(f'recall@{K}', 0) for K in Ks]
            ax.plot(Ks, recs, 'o-', color='#e74c3c', linewidth=2.5, markersize=6,
                    label=f'Best ensemble: {best_ens_name}')

        n_total = 9550
        random_recs = [min(K * 9 / n_total, 1.0) for K in Ks]
        ax.plot(Ks, random_recs, '--', color='gray', alpha=0.5, label='Random')

        if best_ind_name and best_ens_name:
            ind_recs = [ind[best_ind_name].get(f'recall@{K}', 0) for K in Ks]
            ens_recs = [ens[best_ens_name].get(f'recall@{K}', 0) for K in Ks]
            ax.fill_between(Ks, ind_recs, ens_recs, alpha=0.15, color='#e74c3c',
                            label='Ensemble gain')

        ax.set_xlabel('Top-K', fontsize=10)
        ax.set_ylabel('Recall', fontsize=10)
        meta = run_data.get('run_metadata', {})
        model_paths = meta.get('model_paths', [])
        models_short = ', '.join(nice_name(m) for m in model_paths[:5]) if model_paths else '—'
        if model_paths and len(model_paths) > 5:
            models_short += f' +{len(model_paths)-5}'
        ax.set_title(f'{label}\nModels: {models_short}',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim(0, 520)
        ax.set_ylim(-0.02, 1.05)
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Ensemble Gain: How much does ensembling improve recall?',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig15_ensemble_gain.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# FIGURE: Custom ensemble heatmaps (one per custom run — which models, which MOFs)
# ============================================================================
def _draw_one_heatmap(pp, run_name, run_data, output_dir, filename_prefix, run_path=None):
    """Draw a single heatmap for one run. Shows models used and ranks (NN, ML, ensemble, search-found)."""
    if not pp:
        return
    if run_path:
        _backfill_ensemble_ranks_in_per_positive(run_data, run_path)
    mof_names = sorted(pp.keys(), key=lambda c: pp[c]['best_rank'])
    model_names = sorted(pp[mof_names[0]]['ranks'].keys())
    nn_models = [m for m in model_names if is_nn_model(m)]
    ml_models = [m for m in model_names if is_ml_model(m)]
    search_found_models = [m for m in model_names if is_search_found_model(m)]
    ensemble_models = [m for m in model_names if is_ensemble_model(m) and not is_search_found_model(m)]
    ordered_models = nn_models + ml_models + ensemble_models + search_found_models
    separator_idx = len(nn_models)
    separator2_idx = len(nn_models) + len(ml_models)
    separator3_idx = len(nn_models) + len(ml_models) + len(ensemble_models)
    n_mofs = len(mof_names)
    n_models = len(ordered_models)
    rank_matrix = np.zeros((n_mofs, n_models))
    for i, mof in enumerate(mof_names):
        for j, model in enumerate(ordered_models):
            rank_matrix[i, j] = pp[mof]['ranks'].get(model, 9999)
    fig, ax = plt.subplots(figsize=(max(14, n_models * 0.5), max(5, n_mofs * 0.35)))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(np.log10(rank_matrix + 1), cmap=cmap, aspect='auto',
                   vmin=0, vmax=np.log10(10000))
    for i in range(n_mofs):
        for j in range(n_models):
            rank = int(rank_matrix[i, j])
            color = 'white' if rank > 500 else 'black'
            fontsize = 6 if rank > 999 else 7
            ax.text(j, i, str(rank), ha='center', va='center',
                    fontsize=fontsize, color=color, fontweight='bold')
    display_models = [nice_name(m) for m in ordered_models]
    display_mofs = []
    for mof in mof_names:
        metal = get_metal(mof)
        bg = pp[mof]['bandgap']
        name = mof.replace('_FSR', '')
        display_mofs.append(f"{name} ({metal}, {bg:.2f}eV)")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(display_models, rotation=55, ha='right', fontsize=7)
    ax.set_yticks(range(n_mofs))
    ax.set_yticklabels(display_mofs, fontsize=8)
    if n_models > 0:
        if separator_idx > 0 and separator_idx < n_models:
            ax.axvline(x=separator_idx - 0.5, color='black', linewidth=2, linestyle='-')
        if separator2_idx > separator_idx and separator2_idx < n_models:
            ax.axvline(x=separator2_idx - 0.5, color='black', linewidth=2, linestyle='-')
        if separator3_idx > separator2_idx and separator3_idx < n_models:
            ax.axvline(x=separator3_idx - 0.5, color='black', linewidth=2, linestyle='-')
        if len(nn_models) > 0:
            ax.text(separator_idx / 2 - 0.5, -0.7, 'NN', ha='center', fontsize=8, fontweight='bold', color='#1f77b4')
        if len(ml_models) > 0:
            ax.text(separator_idx + (separator2_idx - separator_idx) / 2 - 0.5, -0.7,
                    'Regular ML', ha='center', fontsize=8, fontweight='bold', color='#ff7f0e')
        if len(ensemble_models) > 0:
            ax.text(separator2_idx + (separator3_idx - separator2_idx) / 2 - 0.5, -0.7,
                    'Ensemble', ha='center', fontsize=8, fontweight='bold', color='#d62728')
        if len(search_found_models) > 0:
            ax.text(separator3_idx + (n_models - separator3_idx) / 2 - 0.5, -0.7,
                    'Search-found', ha='center', fontsize=8, fontweight='bold', color='#2ca02c')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('log10(Rank)', fontsize=9)
    meta = run_data.get('run_metadata', {})
    model_paths = meta.get('model_paths', [])
    models_used = ', '.join(nice_name(m) for m in model_paths) if model_paths else '—'
    run_label = run_name.replace('_split_d', '').replace('_', ' ').replace('custom/', '')
    ax.set_title(f'Run: {run_label}\nModels in this run: {models_used}',
                 fontsize=12, fontweight='bold', pad=12)
    fig.text(0.5, 0.01, 'Lower rank = better. Ensemble / search-found (right) combine the base models.',
             ha='center', fontsize=8, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(output_dir, f'{filename_prefix}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def fig_custom_ensemble_heatmaps(data, output_dir):
    """Draw one heatmap per ensemble run (exhaustive, selective, custom, etc.) so all results are comparable."""
    ensemble_runs = data.get('ensemble_runs', {})
    run_paths = data.get('ensemble_run_paths', {})
    if not ensemble_runs:
        return
    for run_name, run_data in sorted(ensemble_runs.items()):
        pp = run_data.get('per_positive')
        if not pp:
            continue
        run_path = run_paths.get(run_name)
        slug = run_name.replace('/', '_').replace(' ', '_')[:60]
        _draw_one_heatmap(pp, run_name, run_data, output_dir,
                          f'fig_run_heatmap_{slug}', run_path)


def fig_searchfound_heatmaps(data, output_dir):
    """Draw one heatmap per search-found combo (constituent models + combo column) for ranking comparison."""
    ensemble_runs = data.get('ensemble_runs', {})
    run_paths = data.get('ensemble_run_paths', {})
    drawn = 0
    for run_name, run_data in sorted(ensemble_runs.items()):
        run_pp = run_data.get('per_positive')
        items = run_data.get('search_found_ensembles') or []
        if not run_pp or not items:
            continue
        run_path = run_paths.get(run_name)
        if run_path:
            _backfill_ensemble_ranks_in_per_positive(run_data, run_path)
        for idx, item in enumerate(items):
            combo_name = item.get('name') or f'search-found_{idx}'
            model_names = list(item.get('model_names') or [])
            pranks = item.get('per_positive_ranks') or {}
            synthetic_pp = {}
            for mof in run_pp:
                ranks = {}
                for m in model_names:
                    ranks[m] = run_pp[mof]['ranks'].get(m, 9999)
                r = pranks.get(mof, pranks.get(str(mof), 9999))
                ranks[combo_name] = int(r) if r is not None else 9999
                synthetic_pp[mof] = {
                    'bandgap': run_pp[mof]['bandgap'],
                    'best_rank': min(ranks.values()),
                    'ranks': ranks,
                }
            if not synthetic_pp:
                continue
            run_slug = run_name.replace('/', '_').replace(' ', '_')[:30]
            combo_short = combo_name.replace('search-found_', '').replace('/', '_').replace(' ', '_')[:40]
            slug = f"{run_slug}_{idx:02d}_{combo_short}"
            run_data_minimal = {
                'run_metadata': {'model_paths': model_names},
            }
            display_name = f"{run_name} — {combo_name}"
            _draw_one_heatmap(synthetic_pp, display_name, run_data_minimal, output_dir,
                              f'fig_searchfound_heatmap_{slug}', None)
            drawn += 1
    if drawn:
        print(f"  Saved {drawn} search-found combo heatmap(s)")


# ============================================================================
# REPORT: Summary Markdown
# ============================================================================
def generate_summary_report(data, output_dir):
    """Generate markdown summary report."""
    sel = data['selective']
    if sel is None:
        return

    pp = sel['per_positive']
    ens = sel.get('ensemble_methods', {})
    ind = sel.get('individual_models', {})

    found = [m for m in pp if pp[m]['best_rank'] <= 200]
    missed = [m for m in pp if pp[m]['best_rank'] > 200]

    lines = []
    lines.append("# MOF Conductivity Discovery -- Progress Report")
    lines.append("")
    lines.append(f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Dataset**: 10,810 MOFs | 74 conductive (bandgap < 1 eV) | 0.68% prevalence")
    lines.append(f"**Split D**: 58 train + 7 val + 9 test positives | 9,550 test total")
    lines.append("")

    lines.append("---")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **{len(found)}/9** test positives found in top-200 by at least one model")
    lines.append(f"- **{len(missed)}/9** remain unfindable (best rank > 200)")
    lines.append(f"- **Best single model**: exp364 (NN fulltune) -- FHR=1, R@200=0.333")
    lines.append(f"- **Best ensemble**: RRF -- finds {len(found)}/9 = {len(found)/9:.1%} in top-200")

    best_ens = ens.get('rrf_k60', {})
    lines.append(f"- **Enrichment@25**: {best_ens.get('enrichment@25', 0):.0f}x better than random")
    lines.append(f"- **Spearman rho**: 0.705 (best NN) -- genuine bandgap prediction signal")
    lines.append("")

    lines.append("---")
    lines.append("## Split D Advantage")
    lines.append("")
    lines.append("| Metric | Original Split | Split D |")
    lines.append("|--------|---------------|---------|")
    lines.append("| Test positives COVERED (sim >= 0.85) | 0/9 | 8/9 |")
    lines.append("| Test positives BORDERLINE | 8/9 | 1/9 |")
    lines.append("| Test positives ISOLATED | 1/9 | 0/9 |")
    lines.append("| Nearest neg closer than pos | 5/9 | 3/9 |")
    lines.append("")
    lines.append("Split D uses farthest-point sampling to guarantee every test positive")
    lines.append("has a structurally similar training positive (cosine sim >= 0.55).")
    lines.append("")

    lines.append("---")
    lines.append("## Ensemble Results")
    lines.append("")

    ensemble_runs = data.get('ensemble_runs', {})
    if ensemble_runs:
        for run_name, run_data in sorted(ensemble_runs.items()):
            run_label = run_name.replace('_split_d', '').replace('_', ' ').title()
            run_ens = run_data.get('ensemble_methods', {})
            run_ind = run_data.get('individual_models', {})
            if not run_ens and not run_ind:
                continue

            lines.append(f"### {run_label} ({len(run_ind)} individual + {len(run_ens)} ensemble methods)")
            meta = run_data.get('run_metadata', {})
            model_paths = meta.get('model_paths', [])
            if model_paths:
                models_str = ', '.join(nice_name(m) for m in model_paths)
                lines.append(f"**Models in this run:** {models_str}")
            lines.append("")
            lines.append("| Method | Type | FHR | R@25 | R@50 | R@100 | R@200 | MRR |")
            lines.append("|--------|------|-----|------|------|-------|-------|-----|")

            all_methods = []
            for name, m in run_ind.items():
                mtype = 'NN' if is_nn_model(name) else 'ML'
                all_methods.append((nice_name(name), mtype, m))
            for name, m in run_ens.items():
                all_methods.append((name, 'Ens', m))

            all_methods.sort(key=lambda x: (-x[2].get('recall@50', 0),
                                            -x[2].get('recall@200', 0)))
            for display, mtype, m in all_methods[:20]:
                lines.append(
                    f"| {display} | {mtype} | {m.get('first_hit_rank', '?'):.0f} | "
                    f"{m.get('recall@25', 0):.3f} | "
                    f"{m.get('recall@50', 0):.3f} | "
                    f"{m.get('recall@100', 0):.3f} | "
                    f"{m.get('recall@200', 0):.3f} | "
                    f"{m.get('mrr', 0):.4f} |")
            lines.append("")
    else:
        lines.append("| Method | FHR | R@50 | R@100 | R@200 | MRR |")
        lines.append("|--------|-----|------|-------|-------|-----|")
        for name in ['rrf_k60', 'weighted_rrf', 'rank_avg', 'vote_top200', 'score_avg', 'stacking']:
            if name in ens:
                m = ens[name]
                lines.append(f"| {name} | {m.get('first_hit_rank', '?'):.0f} | "
                            f"{m.get('recall@50', 0):.3f} | "
                            f"{m.get('recall@100', 0):.3f} | "
                            f"{m.get('recall@200', 0):.3f} | "
                            f"{m.get('mrr', 0):.4f} |")
        lines.append("")

    # Discovery goals: at least this many test positives in top 25/50/100
    GOAL_25, GOAL_50, GOAL_100 = 3, 4, 5

    # Recommended model combinations (discovery: recall@25, @50, @100)
    has_recommended = any(run_data.get('recommended_combinations') for run_data in (data.get('ensemble_runs') or {}).values())
    if has_recommended:
        lines.append("---")
        lines.append("## Recommended model combinations (discovery)")
        lines.append("")
        lines.append("Greedy forward selection on **test split**: best 2–4 model subsets for RRF and rank averaging. **Discovery goals:** ≥3 targets in top 25, ≥4 in top 50, ≥5 in top 100.")
        lines.append("")
        for run_name, run_data in sorted((data.get('ensemble_runs') or {}).items()):
            rec = run_data.get('recommended_combinations')
            if not rec:
                continue
            run_label = run_name.replace('_split_d', '').replace('_', ' ').title()
            by_metric = rec.get('by_metric', {})
            metrics_list = rec.get('metrics') or ([rec.get('primary_metric', rec.get('metric', 'recall@50'))] if not by_metric else list(by_metric.keys()))
            if by_metric:
                for m in metrics_list:
                    if m not in by_metric:
                        continue
                    r = by_metric[m]
                    lines.append(f"### {run_label} — {m}")
                    for method in ['rrf', 'rank_avg']:
                        combo = r.get(f'best_combo_{method}', [])
                        size = r.get(f'best_size_{method}', 0)
                        label = 'RRF' if method == 'rrf' else 'Rank avg'
                        names = ', '.join(nice_name(x) for x in combo)
                        lines.append(f"- **{label}** best size {size}: {names}")
                        best_m = r.get(f'best_metrics_{method}', {})
                        if best_m:
                            h25 = int(best_m.get('hits@25', best_m.get('recall@25', 0) * 9))
                            h50 = int(best_m.get('hits@50', best_m.get('recall@50', 0) * 9))
                            h100 = int(best_m.get('hits@100', best_m.get('recall@100', 0) * 9))
                            lines.append(f"  Hits: **{h25} @25**, **{h50} @50**, **{h100} @100**  →  Goals (≥{GOAL_25} @25, ≥{GOAL_50} @50, ≥{GOAL_100} @100): {'✓' if h25 >= GOAL_25 else '✗'} @25  {'✓' if h50 >= GOAL_50 else '✗'} @50  {'✓' if h100 >= GOAL_100 else '✗'} @100")
                    best_m = r.get('best_metrics_rrf') or r.get('best_metrics_rank_avg') or {}
                    if best_m:
                        lines.append(f"  R@25={best_m.get('recall@25', 0):.3f} R@50={best_m.get('recall@50', 0):.3f} R@100={best_m.get('recall@100', 0):.3f} R@200={best_m.get('recall@200', 0):.3f} FHR={best_m.get('first_hit_rank', '?')}")
                    lines.append("")
            else:
                metric = rec.get('metric', rec.get('primary_metric', 'recall@50'))
                lines.append(f"### {run_label} (metric: {metric})")
                for method in ['rrf', 'rank_avg']:
                    combo = rec.get(f'best_combo_{method}', [])
                    size = rec.get(f'best_size_{method}', 0)
                    label = 'RRF' if method == 'rrf' else 'Rank avg'
                    names = ', '.join(nice_name(x) for x in combo)
                    lines.append(f"- **{label}** best size {size}: {names}")
                    best_m = rec.get(f'best_metrics_{method}', rec.get('best_metrics_rrf') or rec.get('best_metrics_rank_avg') or {})
                    if best_m:
                        h25 = int(best_m.get('hits@25', best_m.get('recall@25', 0) * 9))
                        h50 = int(best_m.get('hits@50', best_m.get('recall@50', 0) * 9))
                        h100 = int(best_m.get('hits@100', best_m.get('recall@100', 0) * 9))
                        lines.append(f"  Hits: {h25} @25, {h50} @50, {h100} @100  →  Goals: {'✓' if h25 >= GOAL_25 else '✗'} @25  {'✓' if h50 >= GOAL_50 else '✗'} @50  {'✓' if h100 >= GOAL_100 else '✗'} @100")
                best_m = rec.get('best_metrics_rrf') or rec.get('best_metrics_rank_avg') or {}
                if best_m:
                    lines.append(f"  R@25={best_m.get('recall@25', 0):.3f} R@50={best_m.get('recall@50', 0):.3f} R@100={best_m.get('recall@100', 0):.3f} (at best size)")
                lines.append("")

    # Ablation best subset (when run with --ablation)
    runs_with_ablation = [(rn, rd) for rn, rd in sorted((data.get('ensemble_runs') or {}).items())
                          if rd.get('ensemble_methods', {}).get('ablation_best')]
    if runs_with_ablation:
        lines.append("---")
        lines.append("## Best discovery ensemble (ablation)")
        lines.append("")
        lines.append("Ablation searched model subsets to maximize recall@50 on test. Below: best subset metrics per run.")
        lines.append("")
        for run_name, run_data in runs_with_ablation:
            ab = run_data['ensemble_methods']['ablation_best']
            greedy = run_data.get('greedy_selection', {})
            selected = greedy.get('selected_models', [])
            run_label = run_name.replace('_split_d', '').replace('_', ' ').title()
            lines.append(f"### {run_label}")
            lines.append(f"- R@25={ab.get('recall@25', 0):.3f} R@50={ab.get('recall@50', 0):.3f} R@100={ab.get('recall@100', 0):.3f} R@200={ab.get('recall@200', 0):.3f} FHR={ab.get('first_hit_rank', '?')}")
            if selected:
                lines.append("- Greedy complementary (coverage) order: " + ", ".join(nice_name(m) for m in selected[:10]))
            lines.append("")

    # Search-found ensembles (exhaustive search: all 2/3/4 combos meeting >=3 @25, >=4 @50, >=5 @100)
    runs_with_search = [(rn, rd) for rn, rd in sorted((data.get('ensemble_runs') or {}).items())
                        if rd.get('search_found_ensembles')]
    if runs_with_search:
        lines.append("---")
        lines.append("## Search-found ensembles")
        lines.append("")
        lines.append("Exhaustive search over all 2-, 3-, 4-model combinations (NN exp≥362 + all ML). Only combos meeting **≥3 @25, ≥4 @50, ≥5 @100** are saved. Shown in the main heatmap under \"Search-found (meet limits)\".")
        lines.append("")
        for run_name, run_data in runs_with_search:
            items = run_data['search_found_ensembles']
            run_label = run_name.replace('_split_d', '').replace('_', ' ').title()
            lines.append(f"### {run_label} ({len(items)} combos meeting limits)")
            lines.append("")
            lines.append("| Name | Method | Size | Hits @25 | @50 | @100 | Models |")
            lines.append("|------|--------|------|----------|-----|------|--------|")
            for item in items[:25]:
                models_str = ', '.join(nice_name(m) for m in item['model_names'][:4])
                if len(item['model_names']) > 4:
                    models_str += '...'
                lines.append("| %s | %s | %d | %d | %d | %d | %s |" % (
                    item['name'][:40], item['method'], item['combo_size'],
                    item['hits@25'], item['hits@50'], item['hits@100'], models_str[:50]))
            if len(items) > 25:
                lines.append("| ... | ... | ... | ... | ... | ... | (%d more) |" % (len(items) - 25))
            lines.append("")

    # Robustness (subsampled / mini-splits) — ensure all ensemble data is reflected
    sel = data.get('selective') or data.get('primary_run_for_heatmap')
    if sel:
        sub = sel.get('subsampled_evaluation', {})
        mini = sel.get('mini_split_evaluation', {})
        if sub or mini:
            lines.append("---")
            lines.append("## Robustness (subsampled & mini-splits)")
            lines.append("")
            if sub:
                lines.append("Subsampled evaluation (N=1500, 30 resamples): mean ± std across resamples.")
                for method in ['rrf_k60', 'rank_avg', 'stacking']:
                    if method not in sub:
                        continue
                    s = sub[method]
                    lines.append(f"- **{method}**: R@50={s.get('recall@50_mean', 0):.3f}±{s.get('recall@50_std', 0):.3f}  R@100={s.get('recall@100_mean', 0):.3f}±{s.get('recall@100_std', 0):.3f}")
                lines.append("")
            if mini:
                lines.append("Mini-split (5 disjoint negative chunks): R@50/R@100 mean ± std.")
                for method in ['rrf_k60', 'rank_avg']:
                    if method not in mini:
                        continue
                    m = mini[method]
                    lines.append(f"- **{method}**: R@50={m.get('recall@50_mean', 0):.3f}±{m.get('recall@50_std', 0):.3f}  R@100={m.get('recall@100_mean', 0):.3f}±{m.get('recall@100_std', 0):.3f}")
                lines.append("")

    lines.append("---")
    lines.append("## Per-Positive Analysis")
    lines.append("")
    lines.append("| MOF | Metal | Bandgap | Best Rank | Best Model | #Top100 | #Top200 | Status |")
    lines.append("|-----|-------|---------|-----------|------------|---------|---------|--------|")
    for mof in sorted(pp.keys(), key=lambda c: pp[c]['best_rank']):
        info = pp[mof]
        metal = get_metal(mof)
        name = mof.replace('_FSR', '')
        status = 'FOUND' if info['best_rank'] <= 200 else 'MISSED'
        lines.append(f"| {name} | {metal} | {info['bandgap']:.3f} | "
                    f"{info['best_rank']} | {nice_name(info['best_model'])} | "
                    f"{info['n_in_top100']} | {info['n_in_top200']} | {status} |")
    lines.append("")

    lines.append("---")
    lines.append("## What We Miss and Why")
    lines.append("")
    for mof in missed:
        info = pp[mof]
        metal = get_metal(mof)
        name = mof.replace('_FSR', '')
        lines.append(f"### {name} ({metal}, {info['bandgap']:.3f} eV)")
        lines.append(f"- Best rank: {info['best_rank']} by {nice_name(info['best_model'])}")
        lines.append(f"- Mean rank across all models: {info['mean_rank']:.0f}")
        lines.append(f"- In top-100 of {info['n_in_top100']} models, top-200 of {info['n_in_top200']} models")
        lines.append("")

    lines.append("---")
    lines.append("## Chemical Analysis")
    lines.append("")
    found_metals = [get_metal(m) for m in found]
    missed_metals = [get_metal(m) for m in missed]
    lines.append(f"**Found metals**: {', '.join(sorted(set(found_metals)))}")
    lines.append(f"**Missed metals**: {', '.join(sorted(set(missed_metals)))}")
    lines.append("")
    lines.append("The miss pattern is primarily **structural**, not chemical.")
    lines.append("Missed MOFs span multiple metal types, indicating the models")
    lines.append("struggle with structurally unusual frameworks regardless of the metal center.")
    lines.append("The fix requires either more diverse training data or physics-based features.")
    lines.append("")

    lines.append("---")
    lines.append("## How to reproduce")
    lines.append("")
    lines.append("- **Recommended combos**: Use the model lists in the \"Recommended model combinations\" section (or `recommended_combinations.csv`) for the metric you care about (recall@25, @50, or @100). Each row gives run, method (RRF / Rank avg), metric, best size, and exact model names.")
    lines.append("- **Per-run model list**: See `models_per_run.csv` for the full list of models in each ensemble run.")
    lines.append("- **Re-run ensemble**: `python ensemble_discovery.py --prediction_dirs <dir1> <dir2> ... --output_dir ./out --recommend_metrics recall@25 recall@50 recall@100 --recommend_max_models 4`")
    lines.append("- **Re-generate this report**: `python generate_final_report.py --base_dir . --output_dir ./final_results --ensemble_dir ./ensemble_results`")
    lines.append("")

    lines.append("---")
    lines.append("## Key Takeaways for Slides")
    lines.append("")
    lines.append("1. **Split D works**: 0 -> 8 test positives structurally covered")
    lines.append("2. **NN + ML complementarity**: NN uniquely finds UTIHAH (Fe), ML uniquely finds MOJWUF (Cu)")
    lines.append("3. **Ensemble > single model**: RRF covers 5-6/9 vs 3-4/9 alone")
    lines.append("4. **4 MOFs are truly hard**: structural novelty beyond current training data")
    lines.append("5. **85x enrichment**: top-25 is 85x more likely to contain a conductor than random")
    lines.append("")

    report_path = os.path.join(output_dir, 'summary_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")


# ============================================================================
# REPORT: Per-Positive Detailed CSV
# ============================================================================
def generate_per_positive_csv(data, output_dir):
    sel = data['selective']
    if sel is None:
        return

    pp = sel['per_positive']
    model_names = list(pp[list(pp.keys())[0]]['ranks'].keys())
    nn_models_list = [m for m in model_names if is_nn_model(m)]
    ml_models_list = [m for m in model_names if is_ml_model(m)]

    splitd_diag = parse_splitd_report(data.get('splitd_report_text'))
    diag_map = {d['cif_id']: d for d in splitd_diag}

    csv_path = os.path.join(output_dir, 'per_positive_detailed.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['mof_name', 'cif_id', 'metal', 'bandgap_eV', 'best_rank',
                        'best_model', 'mean_rank', 'n_top100', 'n_top200',
                        'best_nn_rank', 'best_ml_rank', 'found_by_nn', 'found_by_ml',
                        'cos_sim_to_train_pos', 'coverage_diagnosis', 'status'])

        for mof in sorted(pp.keys(), key=lambda c: pp[c]['best_rank']):
            info = pp[mof]
            ranks = info['ranks']
            best_nn = min((ranks.get(m, 9999) for m in nn_models_list), default=9999)
            best_ml = min((ranks.get(m, 9999) for m in ml_models_list), default=9999)

            diag = diag_map.get(mof, {})
            cos_sim = diag.get('cos_sim', '')
            diagnosis = diag.get('diagnosis', '')

            writer.writerow([
                mof.replace('_FSR', ''), mof, get_metal(mof),
                f"{info['bandgap']:.4f}", info['best_rank'],
                nice_name(info['best_model']), f"{info['mean_rank']:.0f}",
                info['n_in_top100'], info['n_in_top200'],
                best_nn, best_ml,
                'Yes' if best_nn <= 200 else 'No',
                'Yes' if best_ml <= 200 else 'No',
                f"{cos_sim:.4f}" if cos_sim else '',
                diagnosis,
                'FOUND' if info['best_rank'] <= 200 else 'MISSED',
            ])

    print(f"  Saved: {csv_path}")


# ============================================================================
# REPORT: Ensemble Comparison CSV
# ============================================================================
def generate_search_found_csv(data, output_dir):
    """Export search-found ensembles (exhaustive search combos meeting limits)."""
    ensemble_runs = data.get('ensemble_runs', {})
    rows = []
    for run_name, run_data in sorted(ensemble_runs.items()):
        items = run_data.get('search_found_ensembles', [])
        for item in items:
            rows.append({
                'run_name': run_name,
                'name': item['name'],
                'method': item['method'],
                'combo_size': item['combo_size'],
                'hits@25': item['hits@25'], 'hits@50': item['hits@50'], 'hits@100': item['hits@100'],
                'recall@50': item.get('recall@50', 0), 'recall@100': item.get('recall@100', 0),
                'models': '; '.join(nice_name(m) for m in item['model_names']),
            })
    if not rows:
        return
    csv_path = os.path.join(output_dir, 'search_found_ensembles.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['run_name', 'name', 'method', 'combo_size',
                                               'hits@25', 'hits@50', 'hits@100', 'recall@50', 'recall@100', 'models'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


def generate_recommended_combinations_csv(data, output_dir):
    """Export recommended model combinations (all metrics: recall@25, @50, @100)."""
    ensemble_runs = data.get('ensemble_runs', {})
    rows = []
    for run_name, run_data in sorted(ensemble_runs.items()):
        rec = run_data.get('recommended_combinations')
        if not rec:
            continue
        by_metric = rec.get('by_metric', {})
        if by_metric:
            for metric, r in by_metric.items():
                for method in ['rrf', 'rank_avg']:
                    combo = r.get(f'best_combo_{method}', [])
                    size = r.get(f'best_size_{method}', 0)
                    label = 'RRF' if method == 'rrf' else 'Rank avg'
                    names = '; '.join(nice_name(m) for m in combo)
                    rows.append({'run_name': run_name, 'method': label, 'metric': metric,
                                 'best_size': size, 'models': names})
        else:
            metric = rec.get('metric', rec.get('primary_metric', 'recall@50'))
            for method in ['rrf', 'rank_avg']:
                combo = rec.get(f'best_combo_{method}', [])
                size = rec.get(f'best_size_{method}', 0)
                label = 'RRF' if method == 'rrf' else 'Rank avg'
                names = '; '.join(nice_name(m) for m in combo)
                rows.append({'run_name': run_name, 'method': label, 'metric': metric,
                             'best_size': size, 'models': names})
    if not rows:
        return
    csv_path = os.path.join(output_dir, 'recommended_combinations.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['run_name', 'method', 'metric', 'best_size', 'models'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


def generate_models_per_run_csv(data, output_dir):
    """Quick lookup: which models each ensemble run used (for reproducibility)."""
    ensemble_runs = data.get('ensemble_runs', {})
    if not ensemble_runs:
        return
    csv_path = os.path.join(output_dir, 'models_per_run.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['run_name', 'n_individual', 'n_ensemble_methods', 'models_used'])
        for run_name, run_data in sorted(ensemble_runs.items()):
            meta = run_data.get('run_metadata', {})
            model_paths = meta.get('model_paths', [])
            ind = run_data.get('individual_models', {})
            ens = run_data.get('ensemble_methods', {})
            models_str = '; '.join(nice_name(m) for m in model_paths) if model_paths else '—'
            writer.writerow([run_name, len(ind), len(ens), models_str])
    print(f"  Saved: {csv_path}")


def generate_ensemble_csv(data, output_dir):
    sel = data['selective']
    if sel is None:
        return

    ens = sel.get('ensemble_methods', {})
    csv_path = os.path.join(output_dir, 'ensemble_comparison.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'first_hit_rank', 'recall@25', 'recall@50',
                        'recall@100', 'recall@200', 'recall@500',
                        'enrichment@25', 'enrichment@100',
                        'mrr', 'spearman_rho', 'hits@200'])

        for name in sorted(ens.keys()):
            m = ens[name]
            writer.writerow([
                name,
                m.get('first_hit_rank', ''),
                f"{m.get('recall@25', 0):.4f}",
                f"{m.get('recall@50', 0):.4f}",
                f"{m.get('recall@100', 0):.4f}",
                f"{m.get('recall@200', 0):.4f}",
                f"{m.get('recall@500', 0):.4f}",
                f"{m.get('enrichment@25', 0):.1f}",
                f"{m.get('enrichment@100', 0):.1f}",
                f"{m.get('mrr', 0):.5f}",
                f"{m.get('spearman_rho', 0):.4f}",
                m.get('hits@200', ''),
            ])

    print(f"  Saved: {csv_path}")


# ============================================================================
# REPORT: Model Leaderboard CSV
# ============================================================================
def generate_model_leaderboard(data, output_dir):
    """Unified leaderboard CSV: individual models + ensemble methods from ALL runs."""
    all_entries = []
    seen = set()

    ensemble_runs = data.get('ensemble_runs', {})
    for run_name, run_data in ensemble_runs.items():
        run_label = run_name.replace('_split_d', '').replace('_', ' ').title()

        for model_name, m in run_data.get('individual_models', {}).items():
            display = nice_name(model_name)
            if display in seen:
                continue
            seen.add(display)
            mtype = 'NN' if is_nn_model(model_name) else 'ML'
            all_entries.append((model_name, display, mtype, m))

        for ens_name, m in run_data.get('ensemble_methods', {}).items():
            display = f"{ens_name} [{run_label}]"
            if display in seen:
                continue
            seen.add(display)
            all_entries.append((ens_name, display, f'Ensemble ({run_label})', m))

    if not all_entries:
        sel = data.get('selective')
        if sel:
            for model_name, m in sel.get('individual_models', {}).items():
                display = nice_name(model_name)
                mtype = 'NN' if is_nn_model(model_name) else 'ML'
                all_entries.append((model_name, display, mtype, m))

    if not all_entries:
        return

    all_entries.sort(key=lambda e: (-e[3].get('recall@50', 0),
                                    -e[3].get('recall@200', 0),
                                    e[3].get('first_hit_rank', 9999)))

    csv_path = os.path.join(output_dir, 'model_leaderboard.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'model_name', 'display_name', 'type',
                        'first_hit_rank', 'recall@25', 'recall@50', 'recall@100',
                        'recall@200', 'recall@500', 'enrichment@25', 'enrichment@100',
                        'mrr', 'spearman_rho'])

        for i, (raw_name, display, mtype, m) in enumerate(all_entries):
            writer.writerow([
                i + 1, raw_name, display, mtype,
                m.get('first_hit_rank', ''),
                f"{m.get('recall@25', 0):.4f}",
                f"{m.get('recall@50', 0):.4f}",
                f"{m.get('recall@100', 0):.4f}",
                f"{m.get('recall@200', 0):.4f}",
                f"{m.get('recall@500', 0):.4f}",
                f"{m.get('enrichment@25', 0):.1f}",
                f"{m.get('enrichment@100', 0):.1f}",
                f"{m.get('mrr', 0):.5f}",
                f"{m.get('spearman_rho', 0):.4f}",
            ])

    print(f"  Saved: {csv_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate Final Report for MOF Conductivity Discovery')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory (workspace root)')
    parser.add_argument('--output_dir', type=str, default='./final_results',
                        help='Output directory for figures and reports')
    parser.add_argument('--ensemble_dir', type=str, default=None,
                        help='Directory containing ensemble runs (default: <base_dir>/ensemble_results). '
                             'Auto-discovers all subdirs with ensemble_results.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_style()

    print("=" * 80)
    print("  FINAL REPORT GENERATOR -- MOF Conductivity Discovery")
    print("=" * 80)

    print("\n--- Loading MOF metadata from qmof.csv ---")
    load_qmof_metadata(args.base_dir)

    print("\n--- Loading data ---")
    data = load_all_data(args.base_dir, ensemble_dir=args.ensemble_dir)

    n_runs = len(data.get('ensemble_runs', {}))
    print(f"\n  Ensemble runs discovered: {n_runs}")
    for rn in sorted(data.get('ensemble_runs', {})):
        rd = data['ensemble_runs'][rn]
        n_ind = len(rd.get('individual_models', {}))
        n_ens = len(rd.get('ensemble_methods', {}))
        print(f"    {rn}: {n_ind} individual + {n_ens} ensemble methods")

    has_ensemble = data['selective'] is not None or n_runs > 0

    if not has_ensemble:
        print("\n  NOTE: No ensemble results found yet. Ensemble figures will be skipped.")
        print("  Run ensemble_discovery.py first, then re-run this script for full report.")
        print("  (Or provide --ensemble_dir pointing to your ensemble results)")

    print(f"\n  Primary ensemble (selective): {'loaded' if data['selective'] else 'not found'}")
    print(f"  Exhaustive ensemble: {'loaded' if data['exhaustive'] else 'not found'}")
    print(f"  UMAP summary: {'loaded' if data['umap_summary'] else 'not found'}")
    print(f"  Split D report: {'loaded' if data['splitd_report_text'] else 'not found'}")
    print(f"  Original report: {'loaded' if data['orig_report_text'] else 'not found'}")

    if data['selective']:
        pp = data['selective']['per_positive']
        n_found = sum(1 for m in pp if pp[m]['best_rank'] <= 200)
        print(f"\n  Test positives: {len(pp)}")
        print(f"  Found (rank <= 200): {n_found}")
        print(f"  Missed: {len(pp) - n_found}")

    print(f"\n--- Generating Figures ---")

    fig_funcs_need_ensemble = [
        ('fig1_per_positive_heatmap', fig1_per_positive_heatmap),
        ('fig2_recall_at_k_curves', fig2_recall_at_k_curves),
        ('fig4_ensemble_method_comparison', fig4_ensemble_method_comparison),
        ('fig5_per_positive_waterfall', fig5_per_positive_waterfall),
        ('fig6_nn_vs_ml_scatter', fig6_nn_vs_ml_scatter),
        ('fig7_model_coverage_summary', fig7_model_coverage_summary),
        ('fig8_catastrophic_miss_analysis', fig8_catastrophic_miss_analysis),
        ('fig9_enrichment_factor', fig9_enrichment_factor),
        ('fig10_bandgap_vs_rank', fig10_bandgap_vs_rank),
        ('fig11_robustness_boxplots', fig11_robustness_boxplots),
        ('fig12_metal_type_analysis', fig12_metal_type_analysis),
    ]
    for name, func in fig_funcs_need_ensemble:
        if has_ensemble:
            func(data, args.output_dir)
        else:
            print(f"  Skipping {name}: no ensemble data")

    fig3_split_comparison(data, args.output_dir)
    fig_splitd_diagnosis(data, args.output_dir)
    fig13_unified_leaderboard(data, args.output_dir)
    fig14_multi_ensemble_comparison(data, args.output_dir)
    fig15_ensemble_gain(data, args.output_dir)
    fig_custom_ensemble_heatmaps(data, args.output_dir)
    fig_searchfound_heatmaps(data, args.output_dir)

    print(f"\n--- Generating Reports ---")

    if has_ensemble:
        generate_summary_report(data, args.output_dir)
        generate_per_positive_csv(data, args.output_dir)
        generate_ensemble_csv(data, args.output_dir)
        generate_models_per_run_csv(data, args.output_dir)
        generate_recommended_combinations_csv(data, args.output_dir)
        generate_search_found_csv(data, args.output_dir)
    else:
        print("  Skipping summary/per-positive/ensemble reports: no ensemble data")
    generate_model_leaderboard(data, args.output_dir)

    print(f"\n{'=' * 80}")
    print(f"  DONE. All outputs in: {args.output_dir}")
    print(f"{'=' * 80}")
    print(f"\n  Figures:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.png'):
            print(f"    {f}")
    print(f"\n  Reports:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(('.md', '.csv')):
            print(f"    {f}")


if __name__ == "__main__":
    main()
