#!/usr/bin/env python3
"""
Publication-quality plots for regular ML (embedding) classifiers
===============================================================

Reads existing embedding_classifier outputs (test_predictions.csv per method)
and optionally refits tree models to generate:

  FROM SAVED PREDICTIONS (always):
  - Discovery metrics dashboard: Recall@25,50,100,200 + Enrichment@25,50,100,200 (grouped bars)
  - Recall summary bars: Recall@25, 50, 100, 200 per method
  - Recall@K curves (line plot)
  - Enrichment@K curves (fold over random in top-K)
  - ROC curves (all methods, AUC in legend)
  - Precision–Recall curves (imbalanced discovery)
  - Predicted score vs true bandgap (scatter)
  - Calibration (reliability diagram)

  WITH --embeddings_path (optional refit):
  - Feature importance (Random Forest, Extra Trees) — which embedding dims matter
  - Learning curves (e.g. Extra Trees: train size vs val recall@100)

Usage:
  # From existing embedding_classifier run only:
  python ml_training_plots.py --clf_dir ./embedding_classifiers/strategy_d_farthest_point \\
      --output_dir ./figures/ml_training_plots --threshold 1.0

  # Include feature importance and learning curves (refit on embeddings):
  python ml_training_plots.py --clf_dir ./embedding_classifiers/strategy_d_farthest_point \\
      --output_dir ./figures/ml_training_plots \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --labels_dir ./new_splits/strategy_d_farthest_point
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try seaborn for beautiful defaults (optional)
try:
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False

# Publication-friendly style: larger fonts and DPI for readable figures
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
# Minimum figure dimensions for readability
FIG_WIDTH_PER_METHOD = 1.4   # inches per method for bar charts
FIG_HEIGHT_PANEL = 5.5      # height per panel for dashboards

# Distinct color palette for methods (repeat if many)
METHOD_COLORS = [
    '#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12',
    '#1abc9c', '#34495e', '#e91e63', '#00bcd4', '#ff9800',
    '#795548', '#607d8b', '#4caf50', '#2196f3', '#3f51b5',
]
# Sequential for K values (Recall@25 -> darker)
K_COLORS = ['#81d4fa', '#4fc3f7', '#29b6f6', '#039be5', '#0288d1']  # light to dark blue
ENRICHMENT_COLORS = ['#a5d6a7', '#66bb6a', '#43a047', '#2e7d32']   # greens


# -----------------------------------------------------------------------------
# Method name -> display name
# -----------------------------------------------------------------------------
METHOD_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'svm_rbf': 'SVM (RBF)',
    'random_forest': 'Random Forest',
    'extra_trees': 'Extra Trees',
    'gradient_boosting': 'Gradient Boosting',
    'lda': 'LDA',
    'mahalanobis': 'Mahalanobis',
    'isolation_forest': 'Isolation Forest',
    'adaboost': 'AdaBoost',
    'mlp_classifier': 'MLP',
    'gaussian_nb': 'Gaussian NB',
    'ridge_regression': 'Ridge Reg.',
    'lasso_regression': 'Lasso Reg.',
    'rf_regression': 'RF Reg.',
    'xgboost_regression': 'XGBoost Reg.',
    'pca_logreg': 'PCA + LogReg',
    'ensemble_avg': 'Ensemble (avg)',
    'ensemble_rank_avg': 'Ensemble (rank avg)',
    'smote_extra_trees': 'SMOTE Extra Trees',
    'smote_random_forest': 'SMOTE RF',
}


def load_predictions_from_clf_dir(clf_dir, threshold=1.0):
    """
    Scan clf_dir for method subdirs containing test_predictions.csv.
    Returns: dict method_name -> {'cif_ids': list, 'scores': array, 'true_bandgaps': array, 'y_binary': array}
    """
    result = {}
    if not os.path.isdir(clf_dir):
        return result
    for name in sorted(os.listdir(clf_dir)):
        subdir = os.path.join(clf_dir, name)
        if not os.path.isdir(subdir):
            continue
        csv_path = os.path.join(subdir, 'test_predictions.csv')
        if not os.path.isfile(csv_path):
            continue
        cids, scores, true_bg = [], [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cids.append(row['cif_id'])
                    scores.append(float(row['score']))
                    true_bg.append(float(row['true_label']))
                except (KeyError, ValueError):
                    continue
        if not cids:
            continue
        scores = np.array(scores)
        true_bg = np.array(true_bg)
        y_binary = (true_bg < threshold).astype(int)
        result[name] = {
            'cif_ids': cids,
            'scores': scores,
            'true_bandgaps': true_bg,
            'y_binary': y_binary,
        }
    return result


def nice_name(method_key):
    return METHOD_LABELS.get(method_key, method_key.replace('_', ' ').title())


# -----------------------------------------------------------------------------
# 1. ROC curve
# -----------------------------------------------------------------------------
def plot_roc(data_by_method, output_path, threshold=1.0):
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, d) in enumerate(data_by_method.items()):
        y_true = d['y_binary']
        scores = d['scores']
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, label=f"{nice_name(name)} (AUC={roc_auc:.3f})", color=_method_color(i))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC curves — ML classifiers on embeddings', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 2. Precision–Recall curve
# -----------------------------------------------------------------------------
def plot_pr_curve(data_by_method, output_path, threshold=1.0):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, d) in enumerate(data_by_method.items()):
        y_true = d['y_binary']
        scores = d['scores']
        if y_true.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(rec, prec, lw=2.5, label=f"{nice_name(name)} (AP={ap:.3f})", color=_method_color(i))
    prevalence = next((d['y_binary'].mean() for d in data_by_method.values()), 0.01)
    ax.axhline(y=prevalence, color='gray', linestyle='--', lw=1.5, label=f'No skill ({prevalence:.3f})')
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision–Recall curves (imbalanced discovery)', fontsize=15, fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 3. Recall@k curve
# -----------------------------------------------------------------------------
def recall_at_k(scores, y_binary, k_values=(25, 50, 100, 200)):
    order = np.argsort(-scores)
    n_pos = int(y_binary.sum())
    if n_pos == 0:
        return {k: 0.0 for k in k_values}
    sorted_pos = y_binary[order]
    result = {}
    for k in k_values:
        if k > len(scores):
            result[k] = 1.0 if n_pos > 0 else 0.0
        else:
            hits = int(sorted_pos[:k].sum())
            result[k] = hits / n_pos
    return result


def enrichment_at_k(scores, y_binary, k_values=(25, 50, 100, 200)):
    """Enrichment@K = (hits/K) / prevalence; >1 means better than random."""
    order = np.argsort(-scores)
    n_total = len(scores)
    n_pos = int(y_binary.sum())
    prevalence = n_pos / n_total if n_total > 0 else 1e-6
    sorted_pos = y_binary[order]
    result = {}
    for k in k_values:
        if k > n_total or prevalence <= 0:
            result[k] = 0.0
        else:
            hits = int(sorted_pos[:k].sum())
            result[k] = (hits / k) / prevalence
    return result


def _method_color(i):
    return METHOD_COLORS[i % len(METHOD_COLORS)]


def plot_recall_at_k(data_by_method, output_path, k_values=(25, 50, 100, 200)):
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (name, d) in enumerate(data_by_method.items()):
        r = recall_at_k(d['scores'], d['y_binary'], k_values)
        ax.plot(list(r.keys()), list(r.values()), 'o-', lw=3, label=nice_name(name),
                color=_method_color(i), markersize=10)
    ax.set_xlabel('Top-K', fontsize=14)
    ax.set_ylabel('Recall (fraction of positives in top-K)', fontsize=14)
    ax.set_title('Recall@K — discovery ranking', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='gray', fontsize=11)
    ax.set_xticks(list(k_values))
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(k_values) - 2, max(k_values) + 2)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 4. Predicted score vs true bandgap
# -----------------------------------------------------------------------------
def plot_score_vs_bandgap(data_by_method, output_path, threshold=1.0, max_methods=8):
    methods = list(data_by_method.keys())[:max_methods]
    ncol = 2
    nrow = (len(methods) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5.5 * nrow))
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes.reshape(1, -1)
    for idx, name in enumerate(methods):
        ax = axes.flat[idx]
        d = data_by_method[name]
        x = d['true_bandgaps']
        y = d['scores']
        pos = d['y_binary'].astype(bool)
        ax.scatter(x[~pos], y[~pos], s=18, alpha=0.35, c='#95a5a6', label='Neg', edgecolors='none')
        ax.scatter(x[pos], y[pos], s=60, alpha=0.9, c='#e74c3c', label='Pos', zorder=3, edgecolors='white', linewidths=0.8)
        ax.axvline(threshold, color='#2c3e50', linestyle='--', alpha=0.6, lw=1.5)
        ax.set_xlabel('True bandgap (eV)', fontsize=13)
        ax.set_ylabel('Predicted score', fontsize=13)
        ax.set_title(nice_name(name), fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=11)
    for idx in range(len(methods), len(axes.flat)):
        axes.flat[idx].set_visible(False)
    plt.suptitle('Predicted score vs true bandgap (red = positives)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 4b. Discovery metrics dashboard (Recall@25,50,100,200 + Enrichment@25,50,100)
# -----------------------------------------------------------------------------
def plot_discovery_metrics_dashboard(data_by_method, output_path,
                                     recall_ks=(25, 50, 100, 200),
                                     enrich_ks=(25, 50, 100, 200)):
    """Single figure: Recall@K and Enrichment@K grouped bars — centerpiece for papers."""
    methods = list(data_by_method.keys())
    n_methods = len(methods)
    # Figure width: enough that each method has room; height for two clear panels
    fig_w = max(14, n_methods * FIG_WIDTH_PER_METHOD)
    fig_h = FIG_HEIGHT_PANEL * 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

    x = np.arange(n_methods)
    width = 0.18
    off = width * 1.6

    # --- Row 1: Recall@25, 50, 100, 200 ---
    r25 = [recall_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], recall_ks)[recall_ks[0]] for m in methods]
    r50 = [recall_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], recall_ks)[50] for m in methods]
    r100 = [recall_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], recall_ks)[100] for m in methods]
    r200 = [recall_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], recall_ks)[200] for m in methods]
    ax1.bar(x - off, r25, width, label='Recall@25', color=K_COLORS[0], edgecolor='white', linewidth=0.6)
    ax1.bar(x - width/2, r50, width, label='Recall@50', color=K_COLORS[1], edgecolor='white', linewidth=0.6)
    ax1.bar(x + width/2, r100, width, label='Recall@100', color=K_COLORS[2], edgecolor='white', linewidth=0.6)
    ax1.bar(x + off, r200, width, label='Recall@200', color=K_COLORS[3], edgecolor='white', linewidth=0.6)
    ax1.set_ylabel('Recall', fontsize=14)
    ax1.set_title('Discovery metrics: Recall@K (fraction of positives in top-K)', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', ncol=4, framealpha=0.95, fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([nice_name(m) for m in methods], rotation=90, ha='center', fontsize=12)
    ax1.set_ylim(0, 1.08)
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.tick_params(axis='both', labelsize=12)

    # --- Row 2: Enrichment@25, 50, 100, 200 ---
    e25 = [enrichment_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], enrich_ks)[25] for m in methods]
    e50 = [enrichment_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], enrich_ks)[50] for m in methods]
    e100 = [enrichment_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], enrich_ks)[100] for m in methods]
    e200 = [enrichment_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], enrich_ks)[200] for m in methods]
    ax2.bar(x - off, e25, width, label='Enrich@25', color=ENRICHMENT_COLORS[0], edgecolor='white', linewidth=0.6)
    ax2.bar(x - width/2, e50, width, label='Enrich@50', color=ENRICHMENT_COLORS[1], edgecolor='white', linewidth=0.6)
    ax2.bar(x + width/2, e100, width, label='Enrich@100', color=ENRICHMENT_COLORS[2], edgecolor='white', linewidth=0.6)
    ax2.bar(x + off, e200, width, label='Enrich@200', color=ENRICHMENT_COLORS[3], edgecolor='white', linewidth=0.6)
    ax2.set_ylabel('Enrichment factor', fontsize=14)
    ax2.set_xlabel('Method', fontsize=14)
    ax2.set_title('Enrichment@K (fold over random in top-K)', fontsize=15, fontweight='bold')
    ax2.legend(loc='upper right', ncol=4, framealpha=0.95, fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([nice_name(m) for m in methods], rotation=90, ha='center', fontsize=12)
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylim(0, max(1, max(e25 + e50 + e100 + e200) * 1.15))
    ax2.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 4c. Enrichment@K curves (line plot)
# -----------------------------------------------------------------------------
def plot_enrichment_at_k(data_by_method, output_path, k_values=(25, 50, 100, 200)):
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (name, d) in enumerate(data_by_method.items()):
        e = enrichment_at_k(d['scores'], d['y_binary'], k_values)
        ax.plot(list(e.keys()), list(e.values()), 's-', lw=3, label=nice_name(name),
                color=_method_color(i), markersize=10)
    ax.axhline(1.0, color='gray', linestyle='--', lw=1.5, label='Random (1×)')
    ax.set_xlabel('Top-K', fontsize=14)
    ax.set_ylabel('Enrichment factor', fontsize=14)
    ax.set_title('Enrichment@K — fold over random in top-K', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=11)
    ax.set_xticks(list(k_values))
    ax.set_xlim(min(k_values) - 2, max(k_values) + 2)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 5. Summary bar chart: Recall@25, 50, 100, 200 per method
# -----------------------------------------------------------------------------
def plot_recall_summary_bars(data_by_method, output_path, k_values=(25, 50, 100, 200)):
    """Bar chart of recall@k per method — quick comparison for papers."""
    methods = list(data_by_method.keys())
    n_k = len(k_values)
    width = 0.85 / n_k
    x = np.arange(len(methods))
    fig_w = max(14, len(methods) * FIG_WIDTH_PER_METHOD)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    for ki, k in enumerate(k_values):
        r = [recall_at_k(data_by_method[m]['scores'], data_by_method[m]['y_binary'], (k,))[k] for m in methods]
        offset = (ki - (n_k - 1) / 2) * width
        ax.bar(x + offset, r, width, label=f'Recall@{k}', color=K_COLORS[ki % len(K_COLORS)],
               edgecolor='white', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([nice_name(m) for m in methods], rotation=90, ha='center', fontsize=12)
    ax.set_ylabel('Recall', fontsize=14)
    ax.set_title('Discovery recall by method (Recall@25, 50, 100, 200)', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, framealpha=0.95, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 6. Calibration plot (reliability diagram)
# -----------------------------------------------------------------------------
def plot_calibration(data_by_method, output_path, n_bins=10):
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, d) in enumerate(data_by_method.items()):
        prob = np.clip(d['scores'], 1e-6, 1 - 1e-6)
        y_true = d['y_binary']
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_means = []
        bin_freq = []
        for j in range(n_bins):
            mask = bin_indices == j
            if mask.sum() > 0:
                bin_means.append(y_true[mask].mean())
                bin_freq.append(prob[mask].mean())
            else:
                bin_means.append(np.nan)
                bin_freq.append(np.nan)
        bin_means = np.array(bin_means)
        bin_freq = np.array(bin_freq)
        valid = ~np.isnan(bin_means)
        if valid.sum() < 2:
            continue
        ax.plot(bin_freq[valid], bin_means[valid], 'o-', lw=2, markersize=6, label=nice_name(name), color=_method_color(i))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted score (bin)', fontsize=14)
    ax.set_ylabel('Fraction of positives (bin)', fontsize=14)
    ax.set_title('Calibration (reliability diagram)', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# 7. Feature importance (requires refit)
# -----------------------------------------------------------------------------
# What "dim 453" means: the model's embedding is a 768-dimensional vector (one number
# per dimension). "dim 453" is the 454th component (index 453). These dimensions are
# learned by the transformer; they have no built-in chemical label. The tree models
# use these 768 numbers as features; importance shows which indices the splits rely on.
EMBEDDING_DIM_CAPTION = (
    "Embedding index 0–767: each dimension is one component of the model’s 768-d "
    "representation. Indices are learned (no fixed chemical meaning); importance "
    "shows which components the tree splits use most."
)


def plot_feature_importance(embeddings_path, labels_dir, output_dir, threshold=1.0, top_k=25):
    """Refit RF and Extra Trees, plot top embedding-dimension importances."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from embedding_classifier import load_embeddings, prepare_data
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    cif_ids, embeddings, bandgaps, splits = load_embeddings(embeddings_path)
    data = prepare_data(cif_ids, embeddings, bandgaps, splits, threshold, labels_dir=labels_dir)
    X_train = data['X_trainval']
    y_train = data['y_trainval']

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))

    for ax, (model_name, Model) in zip(axes, [
        ('Random Forest', RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced_subsample', min_samples_leaf=3, random_state=42)),
        ('Extra Trees', ExtraTreesClassifier(n_estimators=200, max_depth=None, class_weight='balanced_subsample', min_samples_leaf=3, random_state=42)),
    ]):
        model = Model
        model.fit(X_train, y_train)
        imp = model.feature_importances_
        order = np.argsort(-imp)[:top_k]
        imp_ordered = imp[order]
        colors = plt.cm.Blues(0.35 + 0.6 * np.linspace(1, 0, top_k))
        ax.barh(range(top_k), imp_ordered, color=colors, edgecolor='white', linewidth=0.8, height=0.75)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"Embedding index {order[i]}" for i in range(top_k)], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Feature importance', fontsize=14)
        ax.set_title(model_name, fontsize=15, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=11)
    fig.suptitle('Which embedding dimensions drive the trees?', fontsize=16, fontweight='bold', y=1.02)
    fig.text(0.5, 0.01,
             "Embedding index 0–767: one component of the 768-d representation. "
             "Indices are learned (no fixed chemical meaning).",
             ha='center', fontsize=10, style='italic', color='#555')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(output_dir, 'ml_feature_importance.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    # Save caption for papers
    caption_path = os.path.join(output_dir, 'ml_feature_importance_caption.txt')
    with open(caption_path, 'w') as f:
        f.write("Figure: Feature importance for Random Forest and Extra Trees on MOF embeddings.\n\n")
        f.write(EMBEDDING_DIM_CAPTION + "\n")
    print(f"  Saved: {path}")
    print(f"  Caption: {caption_path}")


# -----------------------------------------------------------------------------
# 8. Learning curve (train size vs val recall@100)
# -----------------------------------------------------------------------------
def plot_learning_curve(embeddings_path, labels_dir, output_path, threshold=1.0, n_sizes=8):
    """Train Extra Trees with varying train set size, plot val recall@100."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from embedding_classifier import load_embeddings, prepare_data, compute_ranking_metrics
    from sklearn.ensemble import ExtraTreesClassifier

    cif_ids, embeddings, bandgaps, splits = load_embeddings(embeddings_path)
    data = prepare_data(cif_ids, embeddings, bandgaps, splits, threshold, labels_dir=labels_dir)
    X_train, X_val = data['X_train'], data['X_val']
    y_train = data['y_train']
    y_val_bg = data['y_val_bg']
    val_cids = data.get('val_cids', [])
    if len(X_val) == 0 or len(val_cids) == 0:
        print("  Skipping learning curve: no validation set.")
        return
    n_train = len(X_train)
    train_sizes = np.linspace(max(20, n_train // 10), n_train, n_sizes).astype(int)
    train_sizes = np.unique(train_sizes)

    recalls = []
    for size in train_sizes:
        idx = np.random.RandomState(42).permutation(n_train)[:size]
        X_t = X_train[idx]
        y_t = y_train[idx]
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=None, class_weight='balanced_subsample',
            min_samples_leaf=3, random_state=42,
        )
        model.fit(X_t, y_t)
        scores = model.predict_proba(X_val)[:, 1]
        m = compute_ranking_metrics(val_cids, scores, y_val_bg, threshold)
        recalls.append(m.get('recall@100', 0))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, recalls, 'o-', lw=3, color='#3498db', markersize=12, markeredgecolor='white', markeredgewidth=1.2)
    ax.fill_between(train_sizes, recalls, alpha=0.2, color='#3498db')
    ax.set_xlabel('Training set size', fontsize=14)
    ax.set_ylabel('Val Recall@100', fontsize=14)
    ax.set_title('Learning curve (Extra Trees)', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Publication plots for ML embedding classifiers')
    ap.add_argument('--clf_dir', type=str, required=True,
                    help='Embedding classifier output dir (e.g. embedding_classifiers/strategy_d_farthest_point)')
    ap.add_argument('--output_dir', type=str, default='./figures/ml_training_plots',
                    help='Where to save figures')
    ap.add_argument('--threshold', type=float, default=1.0, help='Bandgap threshold (eV) for positive')
    ap.add_argument('--embeddings_path', type=str, default=None,
                    help='If set, refit trees for feature importance and learning curve')
    ap.add_argument('--labels_dir', type=str, default=None,
                    help='Labels dir for split override (required if --embeddings_path)')
    ap.add_argument('--skip_refit', action='store_true', help='Do not refit even if embeddings_path given')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading predictions from:", args.clf_dir)
    data_by_method = load_predictions_from_clf_dir(args.clf_dir, args.threshold)
    if not data_by_method:
        print("ERROR: No test_predictions.csv found under clf_dir.")
        sys.exit(1)
    print(f"  Found {len(data_by_method)} methods: {list(data_by_method.keys())}")

    print("\n--- Discovery metrics (Recall@25, 50, 100, 200 + Enrichment) ---")
    plot_discovery_metrics_dashboard(
        data_by_method,
        os.path.join(args.output_dir, 'ml_discovery_metrics_dashboard.png'),
    )
    plot_recall_summary_bars(
        data_by_method,
        os.path.join(args.output_dir, 'ml_recall_summary_bars.png'),
        k_values=(25, 50, 100, 200),
    )
    plot_recall_at_k(data_by_method, os.path.join(args.output_dir, 'ml_recall_at_k.png'))
    plot_enrichment_at_k(data_by_method, os.path.join(args.output_dir, 'ml_enrichment_at_k.png'))

    print("\n--- ROC, PR, score vs bandgap, calibration ---")
    plot_roc(data_by_method, os.path.join(args.output_dir, 'ml_roc_curves.png'), args.threshold)
    plot_pr_curve(data_by_method, os.path.join(args.output_dir, 'ml_precision_recall_curves.png'), args.threshold)
    plot_score_vs_bandgap(data_by_method, os.path.join(args.output_dir, 'ml_score_vs_bandgap.png'), args.threshold)
    plot_calibration(data_by_method, os.path.join(args.output_dir, 'ml_calibration.png'))

    if args.embeddings_path and not args.skip_refit:
        if not args.labels_dir:
            print("  WARNING: --labels_dir required for refit plots; skipping feature importance and learning curve.")
        else:
            print("\n--- Refit-based figures (feature importance, learning curve) ---")
            plot_feature_importance(
                args.embeddings_path, args.labels_dir, args.output_dir,
                threshold=args.threshold, top_k=25,
            )
            plot_learning_curve(
                args.embeddings_path, args.labels_dir,
                os.path.join(args.output_dir, 'ml_learning_curve.png'),
                threshold=args.threshold,
            )

    print("\nDone. All figures in:", args.output_dir)


if __name__ == '__main__':
    main()
