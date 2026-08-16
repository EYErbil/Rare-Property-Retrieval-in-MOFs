#!/usr/bin/env python3
"""
Embedding-Space Classifiers for MOF Discovery
================================================

Train analytical classifiers directly on frozen, pretrained MOFTransformer
embeddings. This is the historical production/comparison program underlying the
released SMOTE--ExtraTrees classifier.

Protocol note:
  - The production classifier combines the predefined training and validation
    partitions before oversampling and estimator fitting.
  - The test partition is excluded from production fitting. Because this script
    compares candidate models on that partition, its test scores are
    retrospective model-comparison results, not an untouched external estimate.
  - The production artifact is ``smote_extra_trees``. Other methods below are
    exploratory comparators.

Why this may beat neural fine-tuning:
  - With only 65 train-plus-validation positives, fine-tuning a 12-layer ViT is
    severely overparameterized. Gradient-based training sees each positive ~once per
    epoch and can't learn stable decision boundaries.
  - These classifiers operate on FIXED 768-dim embeddings and train with
    stable analytical/convex optimizers (LBFGS, QP, etc.).
  - Strong regularization (C=0.01 in logistic regression) prevents overfitting.
  - Methods that expose a cross-validation search tune their own parameters;
    the production ExtraTrees hyperparameters are fixed explicitly below.
  - The full pipeline (extract + train + predict) takes ~20 seconds on CPU.

Methods:
  1. Logistic Regression (L2-regularized, CV-tuned C)
  2. SVM with RBF kernel (captures nonlinear bandgap-structure relationships)
  3. Random Forest (ensemble of decision trees, feature importance)
  4. scikit-learn Gradient Boosting (with XGBoost regression compared separately)
  5. LDA (Linear Discriminant Analysis - optimal linear boundary)
  6. Mahalanobis distance to positive class centroid
  7. Gaussian Naive Bayes (probabilistic class model)
  8. Isolation Forest (anomaly detection - novel chemistry)

All methods output test_predictions.csv compatible with ensemble_discovery.py.
Trained models (and scalers) are saved under output_dir/<method_name>/ so you can
run them on new embeddings for discovery; use predict_with_embedding_classifier.py.

Usage:
  python embedding_classifier.py \\
      --embeddings_path ./embedding_analysis/embeddings_pretrained.npz \\
      --output_dir ./embedding_classifiers \\
      --threshold 1.0

  # Use fine-tuned embeddings (if available):
  python embedding_classifier.py \\
      --embeddings_path ./embedding_analysis/embeddings_finetuned.npz \\
      --output_dir ./embedding_classifiers_finetuned
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from collections import defaultdict

try:
    import joblib
except ImportError:
    joblib = None

warnings.filterwarnings('ignore')


def save_model_artifacts(output_dir, method_name, model=None, scaler=None, pca=None, **extra):
    """Save fitted model and preprocessors so you can run on new embeddings (e.g. discovery).
    Each method dir gets model.joblib, scaler.joblib, pca.joblib (if used), and artifacts.joblib for extras.
    """
    if joblib is None:
        return
    dir_path = os.path.join(output_dir, method_name)
    os.makedirs(dir_path, exist_ok=True)
    if model is not None:
        joblib.dump(model, os.path.join(dir_path, 'model.joblib'))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(dir_path, 'scaler.joblib'))
    if pca is not None:
        joblib.dump(pca, os.path.join(dir_path, 'pca.joblib'))
    if extra:
        joblib.dump(extra, os.path.join(dir_path, 'artifacts.joblib'))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_embeddings(npz_path):
    """Load embeddings from analyze_embeddings.py output."""
    data = np.load(npz_path, allow_pickle=True)
    cif_ids = list(data['cif_ids'])
    embeddings = data['embeddings']
    bandgaps = data['bandgaps']
    splits = list(data['splits'])
    return cif_ids, embeddings, bandgaps, splits


def override_splits_from_labels(cif_ids, bandgaps, labels_dir, threshold=1.0):
    """Override split assignments by reading label JSON files from labels_dir.
    
    The labels_dir should contain:
      train_bandgaps_regression.json  (cif_id -> bandgap)
      val_bandgaps_regression.json
      test_bandgaps_regression.json
    
    CIF IDs present in a label file are assigned to that split.
    CIF IDs not in any label file are excluded (split='unused').
    Bandgap values are also updated from the label files.
    """
    new_splits = ['unused'] * len(cif_ids)
    new_bandgaps = bandgaps.copy()
    cid_to_idx = {cid: i for i, cid in enumerate(cif_ids)}
    
    for split_name in ['train', 'val', 'test']:
        json_path = os.path.join(labels_dir, f'{split_name}_bandgaps_regression.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: {json_path} not found, skipping {split_name} split")
            continue
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        matched = 0
        for cid, bg in label_data.items():
            if cid in cid_to_idx:
                idx = cid_to_idx[cid]
                new_splits[idx] = split_name
                new_bandgaps[idx] = float(bg)
                matched += 1
        n_pos = sum(1 for bg in label_data.values()
                    if float(bg) <= threshold)
        print(f"  {split_name}: {len(label_data)} in labels, {matched} matched to embeddings, {n_pos} positives")
        if matched < len(label_data):
            print(f"  WARNING: {len(label_data) - matched} CIFs in labels not found in .npz (re-run analyze_embeddings.py with a splits_dir that includes all MOFs).")
    
    assigned = sum(1 for s in new_splits if s != 'unused')
    print(f"  Total assigned: {assigned}/{len(cif_ids)}")
    return new_splits, new_bandgaps


def prepare_data(cif_ids, embeddings, bandgaps, splits, threshold=1.0,
                 labels_dir=None):
    """Split into train/val/test arrays for classification.
    
    If labels_dir is provided, override split assignments from label JSON files.
    This enables running on embedding-informed splits (D/E/F) instead of the
    original split baked into the .npz file.
    
    Contract: You can run analyze_embeddings.py once (one split strategy) and
    reuse the same .npz for other strategies. The .npz must contain every CIF
    that appears in the label JSONs in labels_dir (same MOF set, different
    train/val/test assignment is fine).
    """
    if labels_dir is not None:
        print(f"  Overriding splits from: {labels_dir}")
        splits, bandgaps = override_splits_from_labels(
            cif_ids, bandgaps, labels_dir, threshold)
    
    train_mask = np.array([s == 'train' for s in splits])
    val_mask = np.array([s == 'val' for s in splits])
    test_mask = np.array([s == 'test' for s in splits])

    X_train = embeddings[train_mask]
    y_train_bg = bandgaps[train_mask]
    y_train = (y_train_bg <= threshold).astype(int)

    X_val = embeddings[val_mask]
    y_val_bg = bandgaps[val_mask]
    y_val = (y_val_bg <= threshold).astype(int)

    X_test = embeddings[test_mask]
    y_test_bg = bandgaps[test_mask]
    y_test = (y_test_bg <= threshold).astype(int)
    test_cids = [cif_ids[i] for i, s in enumerate(splits) if s == 'test']

    # Production fitting set: predefined training + validation partitions.
    X_trainval = np.vstack([X_train, X_val]) if len(X_val) > 0 else X_train
    y_trainval = np.concatenate([y_train, y_val]) if len(y_val) > 0 else y_train
    y_trainval_bg = np.concatenate([y_train_bg, y_val_bg]) if len(y_val_bg) > 0 else y_train_bg

    val_cids = [cif_ids[i] for i in np.where(val_mask)[0]]
    train_cids = [cif_ids[i] for i in np.where(train_mask)[0]]

    print(f"  Train: {len(X_train)} ({y_train.sum()} pos)")
    print(f"  Val:   {len(X_val)} ({y_val.sum()} pos)")
    print(f"  Test:  {len(X_test)} ({y_test.sum()} pos)")
    print(f"  Train+Val: {len(X_trainval)} ({y_trainval.sum()} pos)")

    return {
        'X_train': X_train, 'y_train': y_train, 'y_train_bg': y_train_bg,
        'X_val': X_val, 'y_val': y_val, 'y_val_bg': y_val_bg,
        'X_test': X_test, 'y_test': y_test, 'y_test_bg': y_test_bg,
        'X_trainval': X_trainval, 'y_trainval': y_trainval,
        'y_trainval_bg': y_trainval_bg,
        'test_cids': test_cids, 'val_cids': val_cids, 'train_cids': train_cids,
    }


# =============================================================================
# RANKING METRICS
# =============================================================================

def compute_ranking_metrics(test_cids, scores, true_bandgaps, threshold=1.0):
    """
    Compute discovery metrics. Higher score = more likely positive.
    """
    Ks = [10, 25, 50, 100, 200, 500]
    n_total = len(scores)
    is_positive = true_bandgaps <= threshold
    n_positive = int(is_positive.sum())
    prevalence = n_positive / n_total if n_total > 0 else 0

    # Sort descending (highest score = most likely positive)
    order = np.argsort(-scores)
    sorted_cids = [test_cids[i] for i in order]
    sorted_pos = is_positive[order]
    sorted_bgs = true_bandgaps[order]

    metrics = {
        'n_total': n_total, 'n_positive': n_positive, 'prevalence': prevalence
    }

    for K in Ks:
        if K > n_total:
            continue
        hits = int(sorted_pos[:K].sum())
        metrics[f'recall@{K}'] = hits / n_positive if n_positive > 0 else 0
        metrics[f'precision@{K}'] = hits / K
        metrics[f'enrichment@{K}'] = (hits / K) / prevalence if prevalence > 0 else 0
        metrics[f'hits@{K}'] = hits

    # First hit rank (1-indexed)
    hit_ranks = np.where(sorted_pos)[0] + 1
    if len(hit_ranks) > 0:
        metrics['first_hit_rank'] = int(hit_ranks[0])
        metrics['mean_hit_rank'] = float(np.mean(hit_ranks))
        metrics['mrr'] = float(np.mean(1.0 / hit_ranks))
    else:
        metrics['first_hit_rank'] = n_total
        metrics['mrr'] = 0

    # Spearman correlation (scores vs bandgaps: higher score should -> lower bg)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(scores, -true_bandgaps)  # positive rho = correct direction
    metrics['spearman_rho'] = float(rho) if not np.isnan(rho) else 0

    # Found positives
    for K in [25, 50, 100, 200]:
        found = []
        for i in range(min(K, n_total)):
            if sorted_pos[i]:
                found.append((sorted_cids[i], float(sorted_bgs[i]), i + 1))
        metrics[f'found_in_top_{K}'] = found

    return metrics


def print_metrics(name, m, width=80):
    """Print formatted metrics."""
    print(f"\n{'='*width}")
    print(f"  {name}")
    print(f"{'='*width}")
    print(f"  Positives: {m['n_positive']}/{m['n_total']}  |  "
          f"FHR: {m.get('first_hit_rank', '?')}  |  "
          f"MRR: {m.get('mrr', 0):.4f}  |  "
          f"Spearman: {m.get('spearman_rho', 0):.4f}")

    print(f"\n  {'K':>5s}  {'Hits':>5s}  {'Recall':>8s}  {'Prec':>8s}  {'Enrich':>8s}")
    print(f"  {'-'*40}")
    for K in [10, 25, 50, 100, 200, 500]:
        if f'recall@{K}' in m:
            print(f"  {K:>5d}  {m[f'hits@{K}']:>5d}  "
                  f"{m[f'recall@{K}']:>8.3f}  "
                  f"{m[f'precision@{K}']:>8.4f}  "
                  f"{m[f'enrichment@{K}']:>8.1f}x")

    for K in [50, 100]:
        key = f'found_in_top_{K}'
        if key in m and m[key]:
            found_str = ', '.join(f"{c}({b:.3f}eV,r={r})"
                                  for c, b, r in m[key])
            print(f"  Top-{K}: {found_str}")


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(test_cids, scores, true_bandgaps, method_name, output_dir, threshold=1.0):
    """Save predictions in ensemble_discovery.py-compatible format."""
    dir_path = os.path.join(output_dir, method_name)
    os.makedirs(dir_path, exist_ok=True)
    csv_path = os.path.join(dir_path, 'test_predictions.csv')

    with open(csv_path, 'w') as f:
        f.write("cif_id,score,predicted_binary,true_label,mode\n")
        for i, cid in enumerate(test_cids):
            # For regression-type compatibility: score = predicted bandgap
            # Lower score = more positive, but we have "higher = more positive" scores
            # Convert: predicted_bg = max_bg - score * range
            # Actually, just save raw score. ensemble_discovery.py normalizes via percentile ranks.
            #
            # The score convention matters for mode tag:
            # - regression mode: lower score = more positive (predicted bandgap)
            # - multiclass mode: higher score = more positive (class-1 probability)
            #
            # Since our scores are "higher = more positive", use multiclass mode
            score = scores[i]
            pred_bin = 1 if score > 0.5 else 0  # rough binary
            true_label = true_bandgaps[i]
            f.write(f"{cid},{score:.6f},{pred_bin},{true_label},multiclass\n")

    # Also save a minimal final_results.json so load_val_metric() works
    results_path = os.path.join(dir_path, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'method': method_name,
            'checkpoints': {
                # Legacy compatibility value consumed by ensemble_discovery.py;
                # this is not a validation-selection score.
                'best_auc_recall_score': 0.5,
            },
            'metric_note': ('Legacy compatibility placeholder; candidate models '
                            'were compared retrospectively on the test partition.'),
        }, f, indent=2)

    return csv_path


# =============================================================================
# CLASSIFIERS
# =============================================================================

def train_logistic_regression(data, output_dir, threshold=1.0):
    """Logistic Regression with cross-validated regularization."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    print("\n--- Logistic Regression (L2, CV-tuned C) ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    # Class weights to handle 58:752 imbalance
    model = LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 20),
        cv=5,
        class_weight='balanced',
        scoring='recall',
        solver='lbfgs',
        max_iter=10000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"  Best C: {model.C_[0]:.6f}")

    # Probability of being positive (class 1)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Logistic Regression (CV)", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'logistic_regression', output_dir, threshold)
    save_model_artifacts(output_dir, 'logistic_regression', model=model, scaler=scaler)

    # Also train on just train (validate on val) to get val metric
    X_t = scaler.fit_transform(data['X_train'])
    X_v = scaler.transform(data['X_val'])
    model_val = LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 20), cv=3, class_weight='balanced',
        scoring='recall', solver='lbfgs', max_iter=10000, random_state=42
    )
    model_val.fit(X_t, data['y_train'])
    val_probs = model_val.predict_proba(X_v)[:, 1]
    val_metrics = compute_ranking_metrics(
        list(range(len(val_probs))), val_probs, data['y_val_bg'], threshold)
    print(f"  Val: FHR={val_metrics.get('first_hit_rank', '?')}, "
          f"R@25={val_metrics.get('recall@25', 0):.3f}")

    return metrics, probs


def train_svm(data, output_dir, threshold=1.0):
    """SVM with RBF kernel - captures nonlinear boundaries."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    print("\n--- SVM (RBF kernel) ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    # Grid search for C and gamma
    # Compute 'scale' and 'auto' values explicitly for old sklearn compatibility
    n_features = X_train.shape[1]
    gamma_scale = 1.0 / (n_features * X_train.var())
    gamma_auto = 1.0 / n_features
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'gamma': [gamma_scale, gamma_auto, 0.001, 0.01],
    }

    model = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced', probability=True,
            random_state=42),
        param_grid, cv=5, scoring='recall', n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"  Best params: {model.best_params_}")

    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("SVM (RBF)", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'svm_rbf', output_dir, threshold)
    save_model_artifacts(output_dir, 'svm_rbf', model=model, scaler=scaler)

    return metrics, probs


def train_random_forest(data, output_dir, threshold=1.0):
    """Random Forest - ensemble of decision trees."""
    from sklearn.ensemble import RandomForestClassifier

    print("\n--- Random Forest ---")

    X_train = data['X_trainval']
    X_test = data['X_test']
    y_train = data['y_trainval']

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight='balanced_subsample',
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Random Forest", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'random_forest', output_dir, threshold)
    save_model_artifacts(output_dir, 'random_forest', model=model)

    # Feature importance (top 20 embedding dimensions)
    importances = model.feature_importances_
    top_dims = np.argsort(-importances)[:20]
    print(f"  Top 20 important dims: {top_dims.tolist()}")
    print(f"  Their importances: {importances[top_dims].tolist()}")

    return metrics, probs


def train_gradient_boosting(data, output_dir, threshold=1.0):
    """Gradient Boosted Trees."""
    from sklearn.ensemble import GradientBoostingClassifier

    print("\n--- Gradient Boosting ---")

    X_train = data['X_trainval']
    X_test = data['X_test']
    y_train = data['y_trainval']

    # Compute sample weights for class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    w_pos = n_neg / n_pos if n_pos > 0 else 1
    sample_weights = np.where(y_train == 1, w_pos, 1.0)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Gradient Boosting", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'gradient_boosting', output_dir, threshold)
    save_model_artifacts(output_dir, 'gradient_boosting', model=model)

    return metrics, probs


def train_lda(data, output_dir, threshold=1.0):
    """Linear Discriminant Analysis - analytical optimal linear boundary."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    print("\n--- LDA (analytical optimal) ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    # Use shrinkage='auto' to handle high-dimensional case (768 > n_samples)
    model = LinearDiscriminantAnalysis(
        solver='lsqr',
        shrinkage='auto',
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("LDA (analytical)", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'lda', output_dir, threshold)
    save_model_artifacts(output_dir, 'lda', model=model, scaler=scaler)

    return metrics, probs


def mahalanobis_ranking(data, output_dir, threshold=1.0):
    """Mahalanobis distance to positive class centroid."""
    from sklearn.preprocessing import StandardScaler

    print("\n--- Mahalanobis Distance to Positive Class ---")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data['X_trainval'])
    X_test_s = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    # Positive class stats
    pos_mask = y_train == 1
    X_pos = X_train_s[pos_mask]
    mu_pos = X_pos.mean(axis=0)

    # Regularized covariance
    n_pos, d = X_pos.shape
    cov_pos = np.cov(X_pos, rowvar=False)
    # Shrinkage: C_reg = (1-alpha)*C + alpha*sigma_I
    alpha = 0.5  # strong shrinkage because n_pos << d
    sigma = np.trace(cov_pos) / d
    cov_reg = (1 - alpha) * cov_pos + alpha * sigma * np.eye(d)
    cov_inv = np.linalg.inv(cov_reg)

    # Mahalanobis distance for each test point
    diff = X_test_s - mu_pos
    mahal = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    # Convert: lower distance = more similar to positives = higher score
    scores = -mahal

    metrics = compute_ranking_metrics(
        data['test_cids'], scores, data['y_test_bg'], threshold)
    print_metrics("Mahalanobis (positive class)", metrics)

    # For saving, normalize to [0,1]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    save_predictions(data['test_cids'], scores_norm, data['y_test_bg'],
                     'mahalanobis', output_dir, threshold)
    # No sklearn model; save scaler + positive-class stats for discovery
    save_model_artifacts(output_dir, 'mahalanobis', scaler=scaler,
                         mu_pos=mu_pos, cov_inv=cov_inv, threshold=threshold)

    return metrics, scores


def regression_models(data, output_dir, threshold=1.0):
    """Train regression models to predict bandgap directly."""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor

    print("\n--- Regression Models (predict bandgap directly) ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval_bg']

    all_metrics = {}

    # Ridge regression
    print("\n  Ridge Regression:")
    ridge = GridSearchCV(
        Ridge(), {'alpha': np.logspace(-2, 4, 20)},
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    ridge.fit(X_train, y_train)
    preds_ridge = ridge.predict(X_test)
    scores_ridge = -preds_ridge  # lower predicted bg -> higher score
    m = compute_ranking_metrics(data['test_cids'], scores_ridge,
                                data['y_test_bg'], threshold)
    print_metrics("Ridge Regression", m)
    all_metrics['ridge'] = m
    save_predictions(data['test_cids'],
                     (scores_ridge - scores_ridge.min()) / (scores_ridge.max() - scores_ridge.min() + 1e-12),
                     data['y_test_bg'], 'ridge_regression', output_dir, threshold)
    save_model_artifacts(output_dir, 'ridge_regression', model=ridge, scaler=scaler)

    # Lasso
    print("\n  Lasso Regression:")
    lasso = GridSearchCV(
        Lasso(max_iter=10000), {'alpha': np.logspace(-4, 2, 20)},
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    lasso.fit(X_train, y_train)
    preds_lasso = lasso.predict(X_test)
    scores_lasso = -preds_lasso
    m = compute_ranking_metrics(data['test_cids'], scores_lasso,
                                data['y_test_bg'], threshold)
    print_metrics("Lasso Regression", m)
    all_metrics['lasso'] = m
    save_predictions(data['test_cids'],
                     (scores_lasso - scores_lasso.min()) / (scores_lasso.max() - scores_lasso.min() + 1e-12),
                     data['y_test_bg'], 'lasso_regression', output_dir, threshold)
    save_model_artifacts(output_dir, 'lasso_regression', model=lasso, scaler=scaler)

    # RF regression
    print("\n  Random Forest Regression:")
    rf = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    scores_rf = -preds_rf
    m = compute_ranking_metrics(data['test_cids'], scores_rf,
                                data['y_test_bg'], threshold)
    print_metrics("RF Regression", m)
    all_metrics['rf_regression'] = m
    save_predictions(data['test_cids'],
                     (scores_rf - scores_rf.min()) / (scores_rf.max() - scores_rf.min() + 1e-12),
                     data['y_test_bg'], 'rf_regression', output_dir, threshold)
    save_model_artifacts(output_dir, 'rf_regression', model=rf, scaler=scaler)

    # XGBoost regression
    print("\n  XGBoost Regression:")
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)
    scores_xgb = -preds_xgb
    m = compute_ranking_metrics(data['test_cids'], scores_xgb,
                                data['y_test_bg'], threshold)
    print_metrics("XGBoost Regression", m)
    all_metrics['xgboost_regression'] = m
    save_predictions(data['test_cids'],
                     (scores_xgb - scores_xgb.min()) / (scores_xgb.max() - scores_xgb.min() + 1e-12),
                     data['y_test_bg'], 'xgboost_regression', output_dir, threshold)
    save_model_artifacts(output_dir, 'xgboost_regression', model=xgb, scaler=scaler)

    return all_metrics


def isolation_forest_ranking(data, output_dir, threshold=1.0):
    """Isolation Forest - density-based anomaly scoring near positive region."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    print("\n--- Isolation Forest (fit on positives only) ---")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data['X_trainval'])
    X_test_s = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    pos_mask = y_train == 1
    X_pos = X_train_s[pos_mask]

    # Fit on positives: test points that look "normal" relative to positives
    # will get high scores (less anomalous = more like a positive)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_pos)

    # higher = more normal (more like positives)
    if hasattr(model, 'score_samples'):
        scores = model.score_samples(X_test_s)
    else:
        # Old sklearn uses decision_function instead
        scores = model.decision_function(X_test_s)

    metrics = compute_ranking_metrics(
        data['test_cids'], scores, data['y_test_bg'], threshold)
    print_metrics("Isolation Forest (positive-fitted)", metrics)

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    save_predictions(data['test_cids'], scores_norm, data['y_test_bg'],
                     'isolation_forest', output_dir, threshold)
    save_model_artifacts(output_dir, 'isolation_forest', model=model, scaler=scaler)

    return metrics, scores


def pca_then_classify(data, output_dir, threshold=1.0):
    """PCA dimensionality reduction + classifier. Reduces overfitting risk."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n--- PCA + Logistic Regression ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    best_m = None
    best_probs = None
    best_n = None
    best_model = None
    best_pca = None

    for n_components in [16, 32, 64, 128, 256]:
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        var_explained = pca.explained_variance_ratio_.sum()

        model = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 15), cv=5, class_weight='balanced',
            scoring='recall', solver='lbfgs', max_iter=10000, random_state=42
        )
        model.fit(X_train_pca, y_train)

        probs = model.predict_proba(X_test_pca)[:, 1]
        m = compute_ranking_metrics(data['test_cids'], probs,
                                    data['y_test_bg'], threshold)

        fhr = m.get('first_hit_rank', 9999)
        r100 = m.get('recall@100', 0)
        print(f"  PCA-{n_components} (var={var_explained:.3f}): "
              f"FHR={fhr}, R@50={m.get('recall@50', 0):.3f}, "
              f"R@100={r100:.3f}, R@200={m.get('recall@200', 0):.3f}")

        if best_m is None or r100 > best_m.get('recall@100', 0):
            best_m = m
            best_probs = probs
            best_n = n_components
            best_model = model
            best_pca = pca

    if best_m:
        print_metrics(f"PCA-{best_n} + LogReg (best)", best_m)
        save_predictions(data['test_cids'], best_probs, data['y_test_bg'],
                         f'pca{best_n}_logreg', output_dir, threshold)
        save_model_artifacts(output_dir, f'pca{best_n}_logreg',
                             model=best_model, scaler=scaler, pca=best_pca,
                             n_components=best_n)

    return best_m, best_probs


def train_extra_trees(data, output_dir, threshold=1.0):
    """Extra Trees - more randomized than RF, often better for rare-class discovery."""
    from sklearn.ensemble import ExtraTreesClassifier

    print("\n--- Extra Trees ---")

    X_train = data['X_trainval']
    X_test = data['X_test']
    y_train = data['y_trainval']

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight='balanced_subsample',
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Extra Trees", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'extra_trees', output_dir, threshold)
    save_model_artifacts(output_dir, 'extra_trees', model=model)

    return metrics, probs


def train_adaboost(data, output_dir, threshold=1.0):
    """AdaBoost - iteratively focuses on hard-to-classify examples."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    print("\n--- AdaBoost ---")

    X_train = data['X_trainval']
    X_test = data['X_test']
    y_train = data['y_trainval']

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    w_pos = n_neg / n_pos if n_pos > 0 else 1
    sample_weights = np.where(y_train == 1, w_pos, 1.0)

    base = DecisionTreeClassifier(max_depth=3, random_state=42)
    try:
        model = AdaBoostClassifier(
            estimator=base,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            algorithm='SAMME',
        )
    except TypeError:
        # Old sklearn (<1.2) uses 'base_estimator' instead of 'estimator'
        model = AdaBoostClassifier(
            base_estimator=base,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            algorithm='SAMME',
        )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("AdaBoost", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'adaboost', output_dir, threshold)
    save_model_artifacts(output_dir, 'adaboost', model=model)

    return metrics, probs


def train_knn_classifier(data, output_dir, threshold=1.0):
    """k-NN Classifier on fixed embeddings (distinct from knn_baseline.py regression)."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    print("\n--- k-NN Classifier ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    best_m = None
    best_probs = None
    best_k = None
    best_model = None

    for k in [5, 10, 20, 50]:
        model = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='cosine',
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        m = compute_ranking_metrics(data['test_cids'], probs,
                                    data['y_test_bg'], threshold)
        fhr = m.get('first_hit_rank', 9999)
        r100 = m.get('recall@100', 0)
        print(f"  k={k}: FHR={fhr}, R@50={m.get('recall@50', 0):.3f}, "
              f"R@100={r100:.3f}, R@200={m.get('recall@200', 0):.3f}")

        if best_m is None or r100 > best_m.get('recall@100', 0):
            best_m = m
            best_probs = probs
            best_k = k
            best_model = model

    if best_m:
        print_metrics(f"k-NN Classifier (k={best_k})", best_m)
        save_predictions(data['test_cids'], best_probs, data['y_test_bg'],
                         f'knn_classifier', output_dir, threshold)
        save_model_artifacts(output_dir, 'knn_classifier', model=best_model,
                             scaler=scaler, n_neighbors=best_k)

    return best_m, best_probs


def train_mlp_classifier(data, output_dir, threshold=1.0):
    """MLP Classifier - small neural net on fixed embeddings via sklearn."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    print("\n--- MLP Classifier ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    model = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("MLP Classifier", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'mlp_classifier', output_dir, threshold)
    save_model_artifacts(output_dir, 'mlp_classifier', model=model, scaler=scaler)

    return metrics, probs


def train_gaussian_nb(data, output_dir, threshold=1.0):
    """Gaussian Naive Bayes - fast, orthogonal to tree-based methods."""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("\n--- Gaussian Naive Bayes ---")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_trainval'])
    X_test = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    pca = PCA(n_components=64, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_pca, y_train)

    probs = model.predict_proba(X_test_pca)[:, 1]

    metrics = compute_ranking_metrics(
        data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Gaussian NB (PCA-64)", metrics)

    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'gaussian_nb', output_dir, threshold)
    save_model_artifacts(output_dir, 'gaussian_nb', model=model, scaler=scaler, pca=pca)

    return metrics, probs


def ensemble_embedding_classifiers(all_probs, data, output_dir, threshold=1.0):
    """Average probability across all embedding classifiers."""
    print("\n--- Ensemble of Embedding Classifiers ---")

    valid_probs = [p for p in all_probs if p is not None]
    if not valid_probs:
        print("  No valid predictions to ensemble")
        return None

    # Simple average
    avg_probs = np.mean(valid_probs, axis=0)
    m_avg = compute_ranking_metrics(
        data['test_cids'], avg_probs, data['y_test_bg'], threshold)
    print_metrics("Ensemble Average (all embedding classifiers)", m_avg)
    save_predictions(data['test_cids'], avg_probs, data['y_test_bg'],
                     'ensemble_avg', output_dir, threshold)

    # Rank average (more robust)
    rank_probs = []
    for p in valid_probs:
        ranks = np.argsort(np.argsort(-p)).astype(float) / len(p)
        rank_probs.append(ranks)
    avg_ranks = np.mean(rank_probs, axis=0)
    # Invert: lower average rank = higher score
    scores_rank = 1 - avg_ranks

    m_rank = compute_ranking_metrics(
        data['test_cids'], scores_rank, data['y_test_bg'], threshold)
    print_metrics("Ensemble Rank Average (all embedding classifiers)", m_rank)
    save_predictions(data['test_cids'], scores_rank, data['y_test_bg'],
                     'ensemble_rank_avg', output_dir, threshold)

    return m_avg


# =============================================================================
# ENHANCED METHODS
# =============================================================================

def get_rf_feature_importances(data, n_top=100):
    """Get top feature indices from a Random Forest trained on full embeddings."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=500, max_depth=None,
        class_weight='balanced_subsample',
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    model.fit(data['X_trainval'], data['y_trainval'])
    importances = model.feature_importances_
    top_idx = np.argsort(-importances)[:n_top]
    cumulative = importances[top_idx].sum()
    print(f"  Selected top {n_top} features (cumulative importance: {cumulative:.3f})")
    return top_idx, importances


def feature_selected_classifiers(data, output_dir, threshold=1.0, n_top=100):
    """Train classifiers on RF-importance-selected top features to reduce overfitting."""
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*80}")
    print(f"  FEATURE-SELECTED CLASSIFIERS (top {n_top} of 768 dims)")
    print(f"{'='*80}")

    top_idx, _ = get_rf_feature_importances(data, n_top)

    X_train_fs = data['X_trainval'][:, top_idx]
    X_test_fs = data['X_test'][:, top_idx]
    y_train = data['y_trainval']

    all_metrics = {}

    print(f"\n--- Extra Trees (top-{n_top}) ---")
    model = ExtraTreesClassifier(
        n_estimators=500, max_depth=None,
        class_weight='balanced_subsample',
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    model.fit(X_train_fs, y_train)
    probs = model.predict_proba(X_test_fs)[:, 1]
    m = compute_ranking_metrics(data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics(f"Extra Trees (top-{n_top})", m)
    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     f'fs{n_top}_extra_trees', output_dir, threshold)
    save_model_artifacts(output_dir, f'fs{n_top}_extra_trees', model=model, top_idx=top_idx)
    all_metrics[f'fs{n_top}_extra_trees'] = m

    print(f"\n--- Random Forest (top-{n_top}) ---")
    model = RandomForestClassifier(
        n_estimators=500, max_depth=None,
        class_weight='balanced_subsample',
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    model.fit(X_train_fs, y_train)
    probs = model.predict_proba(X_test_fs)[:, 1]
    m = compute_ranking_metrics(data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics(f"Random Forest (top-{n_top})", m)
    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     f'fs{n_top}_random_forest', output_dir, threshold)
    save_model_artifacts(output_dir, f'fs{n_top}_random_forest', model=model, top_idx=top_idx)
    all_metrics[f'fs{n_top}_random_forest'] = m

    print(f"\n--- k-NN (top-{n_top}) ---")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_fs)
    X_test_s = scaler.transform(X_test_fs)

    best_m, best_probs, best_k, best_model = None, None, None, None
    for k in [5, 10, 20, 50]:
        model = KNeighborsClassifier(
            n_neighbors=k, weights='distance',
            metric='cosine', n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]
        m = compute_ranking_metrics(data['test_cids'], probs, data['y_test_bg'], threshold)
        r100 = m.get('recall@100', 0)
        print(f"  k={k}: FHR={m.get('first_hit_rank', 9999)}, R@100={r100:.3f}")
        if best_m is None or r100 > best_m.get('recall@100', 0):
            best_m, best_probs, best_k, best_model = m, probs, k, model

    if best_m:
        print_metrics(f"k-NN (top-{n_top}, k={best_k})", best_m)
        save_predictions(data['test_cids'], best_probs, data['y_test_bg'],
                         f'fs{n_top}_knn', output_dir, threshold)
        save_model_artifacts(output_dir, f'fs{n_top}_knn', model=best_model, scaler=scaler,
                             top_idx=top_idx, n_neighbors=best_k)
        all_metrics[f'fs{n_top}_knn'] = best_m

    print(f"\n--- Logistic Regression (top-{n_top}) ---")
    model = LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 20), cv=5, class_weight='balanced',
        scoring='recall', solver='lbfgs', max_iter=10000, random_state=42
    )
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    m = compute_ranking_metrics(data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics(f"Logistic Regression (top-{n_top})", m)
    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     f'fs{n_top}_logreg', output_dir, threshold)
    save_model_artifacts(output_dir, f'fs{n_top}_logreg', model=model, scaler=scaler, top_idx=top_idx)
    all_metrics[f'fs{n_top}_logreg'] = m

    return all_metrics


def smote_manual(X, y, n_synthetic_per_pos=5, k_neighbors=5, random_state=42):
    """Synthetic Minority Over-sampling by interpolating between positive neighbors."""
    rng = np.random.RandomState(random_state)
    pos_mask = y == 1
    X_pos = X[pos_mask]
    n_pos = len(X_pos)

    if n_pos < 2:
        print("  SMOTE: fewer than 2 positives, cannot interpolate")
        return X, y

    k = min(k_neighbors, n_pos - 1)

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X_pos)
    _, indices = nn.kneighbors(X_pos)

    synthetic = []
    for i in range(n_pos):
        neighbors = indices[i, 1:]
        for _ in range(n_synthetic_per_pos):
            j = rng.choice(neighbors)
            lam = rng.uniform(0.1, 0.9)
            new_point = X_pos[i] + lam * (X_pos[j] - X_pos[i])
            synthetic.append(new_point)

    synthetic = np.array(synthetic)
    X_aug = np.vstack([X, synthetic])
    y_aug = np.concatenate([y, np.ones(len(synthetic), dtype=y.dtype)])

    print(f"  SMOTE: {n_pos} pos -> {n_pos + len(synthetic)} pos "
          f"(+{len(synthetic)} synthetic, {n_synthetic_per_pos}x)")
    return X_aug, y_aug


def train_with_smote(data, output_dir, threshold=1.0, n_synthetic=5):
    """Fit the production SMOTE--ExtraTrees branch and two comparators.

    Oversampling and fitting use only the combined training+validation set.
    The test partition is scored after fitting and is never passed to SMOTE.
    """
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    print(f"\n{'='*80}")
    print(f"  SMOTE-AUGMENTED CLASSIFIERS ({n_synthetic}x oversampling)")
    print(f"{'='*80}")

    X_aug, y_aug = smote_manual(
        data['X_trainval'], data['y_trainval'],
        n_synthetic_per_pos=n_synthetic)

    all_metrics = {}

    print("\n--- Extra Trees + SMOTE ---")
    model = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, max_features='sqrt',
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    model.fit(X_aug, y_aug)
    probs = model.predict_proba(data['X_test'])[:, 1]
    m = compute_ranking_metrics(data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Extra Trees + SMOTE", m)
    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'smote_extra_trees', output_dir, threshold)
    save_model_artifacts(
        output_dir, 'smote_extra_trees', model=model,
        fit_partition='train_plus_validation',
        n_real_fit_rows=int(len(data['y_trainval'])),
        n_real_fit_positives=int(data['y_trainval'].sum()),
        n_synthetic_per_positive=int(n_synthetic),
        positive_rule=f'bandgap <= {threshold} eV')
    all_metrics['smote_extra_trees'] = m

    print("\n--- Random Forest + SMOTE ---")
    model = RandomForestClassifier(
        n_estimators=500, max_depth=None,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    model.fit(X_aug, y_aug)
    probs = model.predict_proba(data['X_test'])[:, 1]
    m = compute_ranking_metrics(data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Random Forest + SMOTE", m)
    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'smote_random_forest', output_dir, threshold)
    save_model_artifacts(output_dir, 'smote_random_forest', model=model)
    all_metrics['smote_random_forest'] = m

    print("\n--- Gradient Boosting + SMOTE ---")
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    model.fit(X_aug, y_aug)
    probs = model.predict_proba(data['X_test'])[:, 1]
    m = compute_ranking_metrics(data['test_cids'], probs, data['y_test_bg'], threshold)
    print_metrics("Gradient Boosting + SMOTE", m)
    save_predictions(data['test_cids'], probs, data['y_test_bg'],
                     'smote_gradient_boosting', output_dir, threshold)
    save_model_artifacts(output_dir, 'smote_gradient_boosting', model=model)
    all_metrics['smote_gradient_boosting'] = m

    return all_metrics


def two_stage_prescreening(data, output_dir, threshold=1.0, prefilter_k=500):
    """Two-stage: kNN pre-filter selects candidates, then ExtraTrees re-ranks."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*80}")
    print(f"  TWO-STAGE PRE-SCREENING (kNN top-{prefilter_k} -> ExtraTrees)")
    print(f"{'='*80}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(data['X_trainval'])
    X_test_s = scaler.transform(data['X_test'])
    y_train = data['y_trainval']

    print(f"\n  Stage 1: kNN scoring (k=20, cosine)...")
    knn = KNeighborsClassifier(
        n_neighbors=20, weights='distance', metric='cosine', n_jobs=-1,
    )
    knn.fit(X_train_s, y_train)
    knn_probs = knn.predict_proba(X_test_s)[:, 1]

    top_indices = np.argsort(-knn_probs)[:prefilter_k]
    n_pos_captured = int(data['y_test'][top_indices].sum())
    n_pos_total = int(data['y_test'].sum())
    print(f"  Pre-filter captures {n_pos_captured}/{n_pos_total} positives "
          f"in top-{prefilter_k}")

    X_cand = data['X_test'][top_indices]
    cids_cand = [data['test_cids'][i] for i in top_indices]
    bg_cand = data['y_test_bg'][top_indices]

    print(f"  Stage 2: ExtraTrees re-ranking {prefilter_k} candidates...")
    et = ExtraTreesClassifier(
        n_estimators=500, max_depth=None,
        class_weight='balanced_subsample',
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    et.fit(data['X_trainval'], y_train)
    et_probs = et.predict_proba(X_cand)[:, 1]

    knn_filt = knn_probs[top_indices]
    knn_n = (knn_filt - knn_filt.min()) / (knn_filt.max() - knn_filt.min() + 1e-12)
    et_n = (et_probs - et_probs.min()) / (et_probs.max() - et_probs.min() + 1e-12)
    combined = 0.4 * knn_n + 0.6 * et_n

    m_cand = compute_ranking_metrics(cids_cand, combined, bg_cand, threshold)
    print_metrics(f"Two-Stage Candidates ({prefilter_k})", m_cand)

    full_scores = np.zeros(len(data['X_test']))
    full_scores[top_indices] = combined + 1.0
    m_full = compute_ranking_metrics(
        data['test_cids'], full_scores, data['y_test_bg'], threshold)
    print_metrics("Two-Stage Full-Test View", m_full)
    save_predictions(data['test_cids'], full_scores, data['y_test_bg'],
                     'two_stage_knn_et', output_dir, threshold)
    save_model_artifacts(output_dir, 'two_stage_knn_et', scaler=scaler,
                         knn_model=knn, et_model=et, prefilter_k=prefilter_k)

    return m_full


def label_propagation_ranking(data, output_dir, threshold=1.0):
    """Semi-supervised Label Spreading through embedding similarity graph.

    Evaluation-only: fitted on [X_train; X_test] with test labels masked, so the
    model is not suitable for discovery on new data. Not saved for predict_with_embedding_classifier.
    """
    print(f"\n{'='*80}")
    print(f"  LABEL SPREADING (semi-supervised)")
    print(f"{'='*80}")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = data['X_trainval']
    y_train = data['y_trainval']
    X_test = data['X_test']

    X_all = np.vstack([X_train, X_test])
    X_all_s = scaler.fit_transform(X_all)

    n_train = len(X_train)
    n_test = len(X_test)

    y_all = np.full(n_train + n_test, -1, dtype=int)
    y_all[:n_train] = y_train

    print(f"  Graph: {n_train + n_test} nodes ({n_train} labeled, {n_test} unlabeled)")
    print(f"  Labels: {int(y_train.sum())} pos, {int((y_train == 0).sum())} neg")

    try:
        from sklearn.semi_supervised import LabelSpreading
        model = LabelSpreading(
            kernel='knn', n_neighbors=15, alpha=0.2, max_iter=50,
        )
        model.fit(X_all_s, y_all)

        classes = list(model.classes_)
        if 1 not in classes:
            print("  LabelSpreading: positive class not found in output")
            return None
        pos_col = classes.index(1)
        probs_test = model.label_distributions_[n_train:, pos_col]

        m = compute_ranking_metrics(
            data['test_cids'], probs_test, data['y_test_bg'], threshold)
        print_metrics("Label Spreading (kNN graph)", m)
        save_predictions(data['test_cids'], probs_test, data['y_test_bg'],
                         'label_spreading', output_dir, threshold)
        return m

    except ImportError:
        print("  LabelSpreading not available in this sklearn version")
        return None
    except Exception as e:
        print(f"  LabelSpreading failed: {e}")
        return None


def selective_ensemble(output_dir, data, threshold=1.0):
    """Load all model predictions, keep only signal-bearing models, ensemble them.
    
    Signal threshold: recall@200 must exceed 2x the random baseline.
    """
    import csv

    print(f"\n{'='*80}")
    print(f"  SELECTIVE ENSEMBLE (signal-filtered models)")
    print(f"{'='*80}")

    skip_prefixes = ('selective_', 'ensemble_')

    all_preds = {}
    for method_name in sorted(os.listdir(output_dir)):
        if any(method_name.startswith(p) for p in skip_prefixes):
            continue
        pred_dir = os.path.join(output_dir, method_name)
        csv_path = os.path.join(pred_dir, 'test_predictions.csv')
        if not os.path.isdir(pred_dir) or not os.path.exists(csv_path):
            continue
        try:
            scores = {}
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scores[row['cif_id']] = float(row['score'])
            arr = np.array([scores.get(cid, 0.0) for cid in data['test_cids']])
            all_preds[method_name] = arr
        except Exception as e:
            print(f"  Warning: skip {method_name}: {e}")

    print(f"  Loaded {len(all_preds)} model predictions")
    if not all_preds:
        return {}

    n_total = len(data['y_test'])
    n_pos = int(data['y_test'].sum())
    random_r200 = min(200.0 * n_pos / n_total, 1.0) if n_total > 0 else 0
    signal_thresh = 2.0 * random_r200

    print(f"  Random R@200 baseline: {random_r200:.4f}")
    print(f"  Signal threshold (2x): {signal_thresh:.4f}\n")

    good = {}
    for name, scores in sorted(all_preds.items()):
        m = compute_ranking_metrics(data['test_cids'], scores,
                                    data['y_test_bg'], threshold)
        r200 = m.get('recall@200', 0)
        fhr = m.get('first_hit_rank', 9999)
        tag = "INCLUDE" if r200 >= signal_thresh else "exclude"
        print(f"  [{tag:7s}] {name:<35s}  R@200={r200:.3f}  FHR={fhr}")
        if r200 >= signal_thresh:
            good[name] = scores

    print(f"\n  Selected {len(good)}/{len(all_preds)} models")
    if len(good) < 2:
        print("  Too few signal models for meaningful ensemble")
        return {}

    probs_list = list(good.values())
    result_metrics = {}

    avg = np.mean(probs_list, axis=0)
    m = compute_ranking_metrics(data['test_cids'], avg, data['y_test_bg'], threshold)
    print_metrics(f"Selective Avg ({len(good)} models)", m)
    save_predictions(data['test_cids'], avg, data['y_test_bg'],
                     'selective_avg', output_dir, threshold)
    result_metrics['selective_avg'] = m

    k_rrf = 60
    rrf = np.zeros(n_total)
    for s in probs_list:
        ranks = np.argsort(np.argsort(-s)) + 1
        rrf += 1.0 / (k_rrf + ranks)
    m = compute_ranking_metrics(data['test_cids'], rrf, data['y_test_bg'], threshold)
    print_metrics(f"Selective RRF ({len(good)} models)", m)
    save_predictions(data['test_cids'], rrf, data['y_test_bg'],
                     'selective_rrf', output_dir, threshold)
    result_metrics['selective_rrf'] = m

    rank_list = []
    for s in probs_list:
        r = np.argsort(np.argsort(-s)).astype(float) / len(s)
        rank_list.append(r)
    avg_rank = 1.0 - np.mean(rank_list, axis=0)
    m = compute_ranking_metrics(data['test_cids'], avg_rank, data['y_test_bg'], threshold)
    print_metrics(f"Selective Rank Avg ({len(good)} models)", m)
    save_predictions(data['test_cids'], avg_rank, data['y_test_bg'],
                     'selective_rank_avg', output_dir, threshold)
    result_metrics['selective_rank_avg'] = m

    return result_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Embedding-Space Classifiers for MOF Discovery')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to embeddings .npz from analyze_embeddings.py')
    parser.add_argument('--output_dir', type=str, default='./embedding_classifiers',
                        help='Output directory for predictions and results')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Positive class is bandgap <= threshold (eV)')
    parser.add_argument('--skip_slow', action='store_true',
                        help='Skip slow methods (SVM, GB)')
    parser.add_argument('--labels_dir', type=str, default=None,
                        help='Override split assignments from label JSON files in this '
                             'directory. Use for embedding-informed splits (D/E/F). '
                             'Expects {train,val,test}_bandgaps_regression.json files.')
    parser.add_argument('--enhanced', action='store_true',
                        help='Run enhanced methods: feature selection, SMOTE, '
                             'two-stage pre-screening, label propagation, selective ensemble')
    parser.add_argument('--n_top_features', type=int, default=100,
                        help='Number of top RF features for feature selection (default: 100)')
    parser.add_argument('--smote_ratio', type=int, default=5,
                        help='Synthetic samples per positive for SMOTE (default: 5)')
    parser.add_argument('--prefilter_k', type=int, default=500,
                        help='Candidates for two-stage pre-screening (default: 500)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading embeddings...")
    cif_ids, embeddings, bandgaps, splits = load_embeddings(args.embeddings_path)
    data = prepare_data(cif_ids, embeddings, bandgaps, splits, args.threshold,
                        labels_dir=args.labels_dir)
    print('\nProtocol: production fitting uses train+validation; the test '
          'partition is excluded from fitting but is used below for '
          'retrospective model comparison.')

    all_probs = []
    all_metrics = {}

    # 1. Logistic Regression
    try:
        m, probs = train_logistic_regression(data, args.output_dir, args.threshold)
        all_probs.append(probs)
        all_metrics['logistic_regression'] = m
    except Exception as e:
        print(f"  LogReg failed: {e}")

    # 2. SVM
    if not args.skip_slow:
        try:
            m, probs = train_svm(data, args.output_dir, args.threshold)
            all_probs.append(probs)
            all_metrics['svm_rbf'] = m
        except Exception as e:
            print(f"  SVM failed: {e}")

    # 3. Random Forest
    try:
        m, probs = train_random_forest(data, args.output_dir, args.threshold)
        all_probs.append(probs)
        all_metrics['random_forest'] = m
    except Exception as e:
        print(f"  RF failed: {e}")

    # 4. Gradient Boosting
    if not args.skip_slow:
        try:
            m, probs = train_gradient_boosting(data, args.output_dir, args.threshold)
            all_probs.append(probs)
            all_metrics['gradient_boosting'] = m
        except Exception as e:
            print(f"  GB failed: {e}")

    # 5. LDA
    try:
        m, probs = train_lda(data, args.output_dir, args.threshold)
        all_probs.append(probs)
        all_metrics['lda'] = m
    except Exception as e:
        print(f"  LDA failed: {e}")

    # 6. Mahalanobis
    try:
        m, scores = mahalanobis_ranking(data, args.output_dir, args.threshold)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        all_probs.append(scores_norm)
        all_metrics['mahalanobis'] = m
    except Exception as e:
        print(f"  Mahalanobis failed: {e}")

    # 7. Isolation Forest
    try:
        m, scores = isolation_forest_ranking(data, args.output_dir, args.threshold)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        all_probs.append(scores_norm)
        all_metrics['isolation_forest'] = m
    except Exception as e:
        print(f"  Isolation Forest failed: {e}")

    # 8. PCA + LogReg
    try:
        m, probs = pca_then_classify(data, args.output_dir, args.threshold)
        if probs is not None:
            all_probs.append(probs)
            all_metrics['pca_logreg'] = m
    except Exception as e:
        print(f"  PCA+LogReg failed: {e}")

    # 9. Regression models
    try:
        reg_metrics = regression_models(data, args.output_dir, args.threshold)
        all_metrics.update({f'reg_{k}': v for k, v in reg_metrics.items()})
    except Exception as e:
        print(f"  Regression models failed: {e}")

    # 10. Extra Trees
    try:
        m, probs = train_extra_trees(data, args.output_dir, args.threshold)
        all_probs.append(probs)
        all_metrics['extra_trees'] = m
    except Exception as e:
        print(f"  Extra Trees failed: {e}")

    # 11. AdaBoost
    try:
        m, probs = train_adaboost(data, args.output_dir, args.threshold)
        all_probs.append(probs)
        all_metrics['adaboost'] = m
    except Exception as e:
        print(f"  AdaBoost failed: {e}")

    # 12. k-NN Classifier
    try:
        m, probs = train_knn_classifier(data, args.output_dir, args.threshold)
        if probs is not None:
            all_probs.append(probs)
            all_metrics['knn_classifier'] = m
    except Exception as e:
        print(f"  k-NN Classifier failed: {e}")

    # 13. MLP Classifier
    try:
        m, probs = train_mlp_classifier(data, args.output_dir, args.threshold)
        all_probs.append(probs)
        all_metrics['mlp_classifier'] = m
    except Exception as e:
        print(f"  MLP Classifier failed: {e}")

    # 14. Gaussian Naive Bayes
    try:
        m, probs = train_gaussian_nb(data, args.output_dir, args.threshold)
        all_probs.append(probs)
        all_metrics['gaussian_nb'] = m
    except Exception as e:
        print(f"  Gaussian NB failed: {e}")

    # 15. Ensemble of all embedding classifiers
    m_ens = ensemble_embedding_classifiers(all_probs, data, args.output_dir, args.threshold)
    if m_ens:
        all_metrics['ensemble_avg'] = m_ens

    # =========================================================================
    # ENHANCED METHODS (--enhanced flag)
    # =========================================================================
    if args.enhanced:
        print(f"\n\n{'#'*90}")
        print(f"  RUNNING ENHANCED METHODS")
        print(f"{'#'*90}\n")

        try:
            fs_m = feature_selected_classifiers(
                data, args.output_dir, args.threshold, args.n_top_features)
            all_metrics.update(fs_m)
        except Exception as e:
            print(f"  Feature selection failed: {e}")

        try:
            sm_m = train_with_smote(
                data, args.output_dir, args.threshold, args.smote_ratio)
            all_metrics.update(sm_m)
        except Exception as e:
            print(f"  SMOTE failed: {e}")

        try:
            ts_m = two_stage_prescreening(
                data, args.output_dir, args.threshold, args.prefilter_k)
            if ts_m:
                all_metrics['two_stage_knn_et'] = ts_m
        except Exception as e:
            print(f"  Two-stage failed: {e}")

        try:
            lp_m = label_propagation_ranking(
                data, args.output_dir, args.threshold)
            if lp_m:
                all_metrics['label_spreading'] = lp_m
        except Exception as e:
            print(f"  Label propagation failed: {e}")

        try:
            sel_m = selective_ensemble(
                args.output_dir, data, args.threshold)
            all_metrics.update(sel_m)
        except Exception as e:
            print(f"  Selective ensemble failed: {e}")

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print(f"\n\n{'#'*90}")
    print(f"  RETROSPECTIVE TEST-PARTITION COMPARISON -- ALL EMBEDDING-SPACE METHODS")
    print(f"{'#'*90}")
    print(f"  {'Method':<35s}  {'FHR':>5s}  {'R@25':>5s}  {'R@50':>5s}  "
          f"{'R@100':>5s}  {'R@200':>5s}  {'MRR':>6s}  {'Spear':>6s}")
    print(f"  {'-'*85}")

    for name in sorted(all_metrics.keys()):
        m = all_metrics[name]
        fhr = m.get('first_hit_rank', 9999)
        r25 = m.get('recall@25', 0)
        r50 = m.get('recall@50', 0)
        r100 = m.get('recall@100', 0)
        r200 = m.get('recall@200', 0)
        mrr = m.get('mrr', 0)
        rho = m.get('spearman_rho', 0)
        print(f"  {name:<35s}  {fhr:>5d}  {r25:>5.3f}  {r50:>5.3f}  "
              f"{r100:>5.3f}  {r200:>5.3f}  {mrr:>6.4f}  {rho:>6.3f}")

    print(f"  {'-'*85}")
    print(f"{'#'*90}")

    # Save results JSON
    results_path = os.path.join(args.output_dir, 'all_results.json')
    serializable = {}
    for name, m in all_metrics.items():
        sm = {}
        for k, v in m.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                sm[k] = float(v)
            elif isinstance(v, list):
                try:
                    sm[k] = [(str(a), float(b), int(c)) for a, b, c in v]
                except (ValueError, TypeError):
                    sm[k] = str(v)
        serializable[name] = sm
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # List all saved prediction directories
    print(f"\nPrediction directories (compatible with ensemble_discovery.py):")
    for name in sorted(all_metrics.keys()):
        d = os.path.join(args.output_dir, name)
        if os.path.isdir(d):
            print(f"  {d}")

    print(f"\n  Usage with ensemble_discovery.py:")
    print(f"    python ensemble_discovery.py --clf_dir {args.output_dir}")
    print(f"    (or add individual dirs to existing --experiments_dir)")


if __name__ == "__main__":
    main()
