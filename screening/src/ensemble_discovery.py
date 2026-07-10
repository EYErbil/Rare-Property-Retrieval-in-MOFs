#!/usr/bin/env python3
"""
Ensemble Discovery for MOF Conductivity Prediction
=====================================================

Combines predictions from multiple heterogeneous models (neural nets,
sklearn classifiers, k-NN baselines) on the SAME test set to produce
a single, stronger ranking for discovering conductive MOFs.

The key insight: different models capture different positive MOFs.
A neural net may rank positives 1,3,5 highly while k-NN catches 2,4,6.
Ensembling recovers positives that any single model misses.

Ensemble methods implemented (parity with past ensemble script):
  1. Reciprocal Rank Fusion (RRF) - gold standard for heterogeneous rankings
  2. Rank averaging - simple, robust
  3. Top-K majority voting - threshold-free voting
  4. Weighted RRF - weights by individual model quality (recall@200)
  5. Score averaging - normalize-then-average
  6. Stacking - meta-learner on model ranks
  + Greedy complementary model selection (set-cover in top-K)
  + Ablation over model subsets (best combo by recall)
  + Per-positive analysis, complementarity matrix, robustness (subsampled/mini-splits)
  + Saves ensemble_results.json with individual + all ensemble metrics

Usage:
  # Custom run: each call creates a NEW subfolder under output_dir (no overwrite):
  python ensemble_discovery.py \\
      --base_dir /path/to/project \\
      --models exp364 exp362 extra_trees random_forest \\
      --output_dir ./ensemble_results/custom
  # -> writes to ./ensemble_results/custom/exp362_exp364_extra_trees_random_forest/
  #    (or .../exp362_exp364_extra_trees_random_forest_2 if that exists)
  #    Saves run_metadata.json + ensemble_results.json (includes model list).
  # Use --no_run_subfolder to write directly to output_dir (e.g. for exhaustive/selective scripts).

  # By full paths:
  python ensemble_discovery.py \\
      --prediction_dirs experiments/exp364_fulltune \\
                        experiments/exp370_seed2 \\
                        embedding_classifiers/strategy_d_farthest_point/random_forest \\
                        knn_results/strategy_d_farthest_point \\
      --output_dir ./ensemble_results/split_d

  # Auto-discover all models for a split:
  python ensemble_discovery.py \\
      --auto_discover \\
      --nn_dirs experiments/exp364_fulltune \\
                experiments/exp370_seed2 \\
                experiments/exp371_seed3 \\
      --clf_dir embedding_classifiers/strategy_d_farthest_point \\
      --knn_dir knn_results/strategy_d_farthest_point \\
      --output_dir ./ensemble_results/split_d
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
from collections import defaultdict


# =============================================================================
# PREDICTION LOADING
# =============================================================================

def load_predictions(csv_path):
    """
    Load predictions from test_predictions.csv.

    Returns dict: cif_id -> {'score': float, 'true_label': float, 'mode': str}
    """
    preds = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['cif_id']
            preds[cid] = {
                'score': float(row['score']),
                'true_label': float(row['true_label']),
                'mode': row.get('mode', 'regression'),
            }
    return preds


def score_to_rank(scores, lower_is_better=True):
    """
    Convert scores to ranks (1 = best / most likely positive).
    Handles ties by averaging ranks.
    """
    n = len(scores)
    if lower_is_better:
        order = np.argsort(scores)
    else:
        order = np.argsort(-scores)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks


def _scores_to_rank_dict(scores_dict, test_cids, lower_is_better=True):
    """Convert cid -> score dict to cid -> rank dict (for ensemble methods)."""
    scores_arr = np.array([scores_dict.get(cid, float('nan')) for cid in test_cids])
    ranks = score_to_rank(scores_arr, lower_is_better=lower_is_better)
    return {cid: int(ranks[i]) for i, cid in enumerate(test_cids)}


def filter_models_exp_min(models, min_exp=362):
    """Keep only NN experiments with exp number >= min_exp; keep all non-NN (ML, kNN)."""
    import re
    filtered = {}
    for name in models:
        if not name.startswith('experiments/') or 'exp' not in name:
            filtered[name] = models[name]
            continue
        m = re.match(r'experiments/exp(\d+)', name)
        if m and int(m.group(1)) >= min_exp:
            filtered[name] = models[name]
    return filtered


def exhaustive_search_meeting_limits(models, test_cids, true_labels, threshold=1.0,
                                     rrf_k=60, goal_25=3, goal_50=4, goal_100=5,
                                     max_combo_size=4, min_combo_size=2):
    """
    Try ALL combinations of size 2, 3, ..., max_combo_size. Keep only those
    that meet limits: hits@25 >= goal_25, hits@50 >= goal_50, hits@100 >= goal_100.
    Returns list of dicts: { name, model_names, method, per_positive_ranks, metrics }.
    """
    from itertools import combinations
    model_names = sorted(models.keys())
    n_pos = sum(1 for cid in test_cids if true_labels.get(cid, 99) < threshold)
    found = []
    seen_combo = set()  # (tuple(sorted names), method) to avoid duplicate

    for size in range(min_combo_size, min(max_combo_size + 1, len(model_names) + 1)):
        for combo in combinations(model_names, size):
            combo = list(combo)
            sub = {n: models[n] for n in combo}
            for method_name, get_scores in [
                ('rrf', lambda: reciprocal_rank_fusion(sub, test_cids, k=rrf_k)),
                ('rank_avg', lambda: rank_averaging(sub, test_cids)),
            ]:
                key = (tuple(sorted(combo)), method_name)
                if key in seen_combo:
                    continue
                scores = get_scores()
                metrics = compute_ranking_metrics(test_cids, scores, true_labels, threshold)
                h25 = metrics.get('hits@25', 0)
                h50 = metrics.get('hits@50', 0)
                h100 = metrics.get('hits@100', 0)
                if h25 >= goal_25 and h50 >= goal_50 and h100 >= goal_100:
                    seen_combo.add(key)
                    ranks = _scores_to_rank_dict(scores, test_cids, lower_is_better=True)
                    short = '+'.join((n.split('/')[-1][:18] for n in combo))
                    name = "search-found_%s_%s" % (method_name, short[:55])
                    found.append({
                        'name': name,
                        'model_names': combo,
                        'method': method_name,
                        'per_positive_ranks': ranks,
                        'hits@25': h25, 'hits@50': h50, 'hits@100': h100,
                        'recall@25': metrics.get('recall@25', 0), 'recall@50': metrics.get('recall@50', 0),
                        'recall@100': metrics.get('recall@100', 0), 'recall@200': metrics.get('recall@200', 0),
                        'first_hit_rank': metrics.get('first_hit_rank', 9999),
                        'mrr': metrics.get('mrr', 0),
                        'combo_size': size,
                    })
    return found


def infer_score_direction(mode_str):
    """
    Determine if lower score means more positive.
    regression / knn: lower score (predicted bandgap) = more positive
    multiclass / classification: higher score (probability) = more positive
    """
    mode_lower = mode_str.lower().strip().strip("'{}")
    if any(tag in mode_lower for tag in ['regression', 'knn', 'sim_to_pos']):
        return True
    return False


def collect_predictions(prediction_dirs):
    """
    Load predictions from all specified directories.

    Returns:
        models: dict of model_name -> {cif_id -> score} (lower = more positive)
        true_labels: dict of cif_id -> true_bandgap
        test_cids: list of cif_ids (consistent ordering)
    """
    models = {}
    true_labels = {}
    all_cid_sets = []

    for pred_dir in prediction_dirs:
        csv_path = os.path.join(pred_dir, 'test_predictions.csv')
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        model_name = os.path.basename(pred_dir)
        parent = os.path.basename(os.path.dirname(pred_dir))
        if parent and parent != '.':
            model_name = f"{parent}/{model_name}"

        preds = load_predictions(csv_path)
        if not preds:
            print(f"  WARNING: {csv_path} is empty, skipping")
            continue

        first_mode = next(iter(preds.values()))['mode']
        lower_is_better = infer_score_direction(first_mode)

        for cid, p in preds.items():
            true_labels[cid] = p['true_label']

        raw_scores = {cid: p['score'] for cid, p in preds.items()}

        if not lower_is_better:
            max_score = max(raw_scores.values())
            raw_scores = {cid: max_score - s for cid, s in raw_scores.items()}

        models[model_name] = raw_scores
        all_cid_sets.append(set(raw_scores.keys()))

        n_test = len(preds)
        n_pos = sum(1 for p in preds.values() if p['true_label'] < 1.0)
        direction = "lower=positive" if lower_is_better else "higher=positive (inverted)"
        print(f"  Loaded {model_name}: {n_test} samples, {n_pos} pos, mode={first_mode}, {direction}")

    if not models:
        print("ERROR: No models loaded.")
        sys.exit(1)

    common_cids = set.intersection(*all_cid_sets) if all_cid_sets else set()
    if len(common_cids) < len(all_cid_sets[0]) * 0.9:
        print(f"  WARNING: Models have different test sets. Using intersection: {len(common_cids)} samples")
    test_cids = sorted(common_cids)

    for name in list(models.keys()):
        models[name] = {cid: models[name][cid] for cid in test_cids if cid in models[name]}

    print(f"\n  Total models: {len(models)}")
    print(f"  Common test samples: {len(test_cids)}")
    n_pos = sum(1 for cid in test_cids if true_labels.get(cid, 99) < 1.0)
    print(f"  Test positives: {n_pos}")

    return models, true_labels, test_cids


def auto_discover_models(nn_dirs=None, clf_dir=None, knn_dir=None):
    """Auto-discover all prediction directories."""
    dirs = []

    if nn_dirs:
        for d in nn_dirs:
            if os.path.exists(os.path.join(d, 'test_predictions.csv')):
                dirs.append(d)

    if clf_dir and os.path.isdir(clf_dir):
        for entry in sorted(os.listdir(clf_dir)):
            sub = os.path.join(clf_dir, entry)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, 'test_predictions.csv')):
                dirs.append(sub)

    if knn_dir and os.path.isdir(knn_dir):
        if os.path.exists(os.path.join(knn_dir, 'test_predictions.csv')):
            dirs.append(knn_dir)
        for entry in sorted(os.listdir(knn_dir)):
            sub = os.path.join(knn_dir, entry)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, 'test_predictions.csv')):
                dirs.append(sub)

    return dirs


def _run_slug_from_dirs(pred_dirs):
    """Build a short, filesystem-safe slug from prediction dirs for run subfolder names."""
    def short_name(path):
        b = os.path.basename(path.rstrip(os.path.sep))
        if b.startswith('exp') and '_' in b:
            return b.split('_')[0]
        return b.replace(' ', '_')
    parts = sorted(set(short_name(d) for d in pred_dirs))
    slug = "_".join(parts)
    if not slug:
        slug = "ensemble"
    return slug[:80]


def resolve_models_to_dirs(base_dir, model_names):
    """
    Resolve short model names to full paths containing test_predictions.csv.

    Supports:
      - NN experiments: exp364, exp362, exp364_fulltune, etc.
        -> base_dir/experiments/<matching dir>
      - ML classifiers: extra_trees, random_forest, knn_classifier, etc.
        -> base_dir/embedding_classifiers/strategy_d_farthest_point/<name>
      - kNN baseline: knn, knn_baseline
        -> base_dir/knn_results/strategy_d_farthest_point
      - Any path that exists and contains test_predictions.csv is used as-is.

    Returns list of resolved absolute paths.
    """
    import glob
    resolved = []
    base_dir = os.path.abspath(base_dir or '.')
    experiments_dir = os.path.join(base_dir, 'experiments')
    clf_base = os.path.join(base_dir, 'embedding_classifiers', 'strategy_d_farthest_point')
    knn_base = os.path.join(base_dir, 'knn_results', 'strategy_d_farthest_point')

    for name in model_names:
        name = name.strip()
        if not name:
            continue
        # Already a path?
        if os.path.sep in name or (os.path.exists(name) and os.path.isdir(name)):
            cand = os.path.abspath(name)
            if os.path.exists(os.path.join(cand, 'test_predictions.csv')):
                resolved.append(cand)
                continue
            # relative to base_dir
            cand = os.path.join(base_dir, name)
            if os.path.exists(os.path.join(cand, 'test_predictions.csv')):
                resolved.append(os.path.abspath(cand))
                continue
        # kNN baseline
        if name.lower() in ('knn', 'knn_baseline', 'knn_regression'):
            if os.path.exists(os.path.join(knn_base, 'test_predictions.csv')):
                resolved.append(knn_base)
                continue
        # NN experiment: exp364 or exp364_fulltune
        if name.lower().startswith('exp') and any(c.isdigit() for c in name):
            if os.path.isdir(experiments_dir):
                exact = os.path.join(experiments_dir, name)
                if os.path.isdir(exact) and os.path.exists(os.path.join(exact, 'test_predictions.csv')):
                    resolved.append(os.path.abspath(exact))
                else:
                    # prefix match: exp364 -> exp364_fulltune
                    for d in sorted(glob.glob(os.path.join(experiments_dir, name + '*'))):
                        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'test_predictions.csv')):
                            resolved.append(os.path.abspath(d))
                            break
                continue
        # ML classifier by name (e.g. extra_trees, random_forest)
        cand = os.path.join(clf_base, name)
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, 'test_predictions.csv')):
            resolved.append(os.path.abspath(cand))
            continue
        # Subdir under knn (e.g. hybrid_post_training)
        cand = os.path.join(knn_base, name)
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, 'test_predictions.csv')):
            resolved.append(os.path.abspath(cand))
            continue

        print(f"  WARNING: Could not resolve model '{name}' to a dir with test_predictions.csv")
    return resolved


# =============================================================================
# RANKING METRICS
# =============================================================================

def compute_ranking_metrics(test_cids, scores, true_labels, threshold=1.0,
                            Ks=[10, 25, 50, 100, 200, 500]):
    """
    Compute discovery ranking metrics.
    Lower score = more likely positive.
    """
    true_bgs = np.array([true_labels[cid] for cid in test_cids])
    scores_arr = np.array([scores[cid] for cid in test_cids])

    order = np.argsort(scores_arr)
    sorted_cids = [test_cids[i] for i in order]
    sorted_true = true_bgs[order]
    is_positive = sorted_true < threshold

    n_total = len(scores_arr)
    n_positive = int(is_positive.sum())
    prevalence = n_positive / n_total if n_total > 0 else 0

    metrics = {
        'n_total': n_total,
        'n_positive': n_positive,
        'prevalence': prevalence,
    }

    for K in Ks:
        if K > n_total:
            continue
        hits = int(is_positive[:K].sum())
        metrics[f'recall@{K}'] = hits / n_positive if n_positive > 0 else 0
        metrics[f'precision@{K}'] = hits / K
        metrics[f'enrichment@{K}'] = (hits / K) / prevalence if prevalence > 0 else 0
        metrics[f'hits@{K}'] = hits

    hit_ranks = np.where(is_positive)[0] + 1
    if len(hit_ranks) > 0:
        metrics['first_hit_rank'] = int(hit_ranks[0])
        metrics['median_hit_rank'] = int(np.median(hit_ranks))
        metrics['mean_hit_rank'] = float(np.mean(hit_ranks))
        metrics['last_hit_rank'] = int(hit_ranks[-1])
        metrics['mrr'] = float(np.mean(1.0 / hit_ranks))
    else:
        metrics['first_hit_rank'] = n_total
        metrics['mrr'] = 0

    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(scores_arr, true_bgs)
        metrics['spearman_rho'] = float(rho) if not np.isnan(rho) else 0
    except ImportError:
        metrics['spearman_rho'] = 0

    for K in [25, 50, 100, 200]:
        found = []
        for i in range(min(K, n_total)):
            if is_positive[i]:
                found.append((sorted_cids[i], float(sorted_true[i]), i + 1))
        metrics[f'found_in_top_{K}'] = found

    return metrics


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def reciprocal_rank_fusion(models, test_cids, k=60):
    """
    Reciprocal Rank Fusion: score = sum(1 / (k + rank_i)) for each model i.
    Lower RRF score = higher rank (we negate so lower = more positive).
    """
    cid_scores = defaultdict(float)
    for model_name, model_scores in models.items():
        scores_arr = np.array([model_scores[cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        for i, cid in enumerate(test_cids):
            cid_scores[cid] += 1.0 / (k + ranks[i])

    max_score = max(cid_scores.values())
    return {cid: max_score - cid_scores[cid] for cid in test_cids}


def rank_averaging(models, test_cids):
    """Average rank across all models. Lower average rank = more positive."""
    cid_rank_sum = defaultdict(float)
    for model_name, model_scores in models.items():
        scores_arr = np.array([model_scores[cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        for i, cid in enumerate(test_cids):
            cid_rank_sum[cid] += ranks[i]

    n_models = len(models)
    return {cid: cid_rank_sum[cid] / n_models for cid in test_cids}


def top_k_voting(models, test_cids, K=100):
    """
    Count how many models place each MOF in their top-K.
    More votes = more likely positive (negate for lower=better convention).
    """
    cid_votes = defaultdict(int)
    for model_name, model_scores in models.items():
        scores_arr = np.array([model_scores[cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        for i, cid in enumerate(test_cids):
            if ranks[i] <= K:
                cid_votes[cid] += 1

    max_votes = max(cid_votes.values()) if cid_votes else 1
    result = {}
    for cid in test_cids:
        votes = cid_votes.get(cid, 0)
        scores_arr = np.array([models[m][cid] for m in models])
        avg_score = np.mean(scores_arr)
        result[cid] = -(votes * 1e6) + avg_score
    return result


def weighted_rrf(models, test_cids, model_weights, k=60):
    """
    Weighted RRF: score = sum(w_i / (k + rank_i)).
    model_weights: dict of model_name -> weight (higher = more trusted).
    Use uniform weights for valid test evaluation; weights derived from test
    performance would cause data leakage.
    """
    cid_scores = defaultdict(float)
    for model_name, model_scores in models.items():
        w = model_weights.get(model_name, 1.0)
        scores_arr = np.array([model_scores[cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        for i, cid in enumerate(test_cids):
            cid_scores[cid] += w / (k + ranks[i])

    max_score = max(cid_scores.values())
    return {cid: max_score - cid_scores[cid] for cid in test_cids}


def score_averaging(models, test_cids):
    """Normalize all scores to [0,1] then average. Lower = more positive."""
    normalized = {}
    for model_name, model_scores in models.items():
        scores_arr = np.array([model_scores[cid] for cid in test_cids])
        s_min, s_max = scores_arr.min(), scores_arr.max()
        s_range = s_max - s_min + 1e-12
        normalized[model_name] = (scores_arr - s_min) / s_range

    avg_scores = np.mean(list(normalized.values()), axis=0)
    return {cid: avg_scores[i] for i, cid in enumerate(test_cids)}


def stacking_ensemble(models, test_cids, true_labels, threshold=1.0):
    """
    Train a logistic regression meta-learner on model scores.
    Uses leave-one-out on the test set so the meta-learner never sees the
    left-out sample's label when predicting it (avoids data leakage).
    Falls back to rank averaging if stacking fails.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  sklearn not available, falling back to rank averaging")
        return rank_averaging(models, test_cids)

    model_names = sorted(models.keys())
    n_models = len(model_names)
    n_samples = len(test_cids)

    X = np.zeros((n_samples, n_models))
    for j, name in enumerate(model_names):
        scores_arr = np.array([models[name][cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        X[:, j] = ranks / n_samples

    y = np.array([1 if true_labels[cid] < threshold else 0 for cid in test_cids])

    if y.sum() < 2:
        print("  Too few positives for stacking, falling back to rank averaging")
        return rank_averaging(models, test_cids)

    meta_probs = np.zeros(n_samples)
    meta = LogisticRegression(
        C=0.1, class_weight='balanced', max_iter=10000, solver='lbfgs'
    )
    print("  Stacking: leave-one-out (no test leakage), %d fits..." % n_samples)
    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        y_train = y[mask]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        xi = scaler.transform(X[i : i + 1])
        meta.fit(X_train_scaled, y_train)
        meta_probs[i] = meta.predict_proba(xi)[0, 1]

    max_prob = meta_probs.max()
    return {cid: max_prob - meta_probs[i] for i, cid in enumerate(test_cids)}


def greedy_ensemble_forward_selection(models, test_cids, true_labels, threshold=1.0,
                                       metric='recall@50', max_models=15, rrf_k=60):
    """
    Greedy forward selection: at each step add the model that most improves
    ensemble performance (RRF or rank_avg) on the chosen metric.
    Use this to get recommended model combinations without trying all 2^N subsets.

    Note: Best combo is selected using test-set performance, so the reported
    "best" metric is optimistic (selection bias). Use the recommended combo
    on a held-out set for an unbiased estimate.

    Returns dict with keys:
      - order_rrf, order_rank_avg: list of model names in order added
      - curve_rrf, curve_rank_avg: list of (subset_size, metric_value)
      - best_combo_rrf, best_combo_rank_avg: best subset (up to best size)
      - best_size_rrf, best_size_rank_avg: size that achieved best metric
    """
    model_names = sorted(models.keys())
    if not model_names:
        return {}

    def score_metric(m):
        if metric == 'composite':
            return m.get('recall@200', 0) * 100 + m.get('recall@100', 0) * 10 + m.get('mrr', 0)
        return m.get(metric, 0)

    def eval_rrf(subset):
        if not subset:
            return {}
        sub = {n: models[n] for n in subset}
        scores = reciprocal_rank_fusion(sub, test_cids, k=rrf_k)
        return compute_ranking_metrics(test_cids, scores, true_labels, threshold)

    def eval_rank_avg(subset):
        if not subset:
            return {}
        sub = {n: models[n] for n in subset}
        scores = rank_averaging(sub, test_cids)
        return compute_ranking_metrics(test_cids, scores, true_labels, threshold)

    result = {}
    for method_name, eval_fn in [('rrf', eval_rrf), ('rank_avg', eval_rank_avg)]:
        selected = []
        remaining = list(model_names)
        curve = []

        for _ in range(min(max_models, len(model_names))):
            best_candidate = None
            best_score = -1
            best_metrics = None
            for c in remaining:
                combo = selected + [c]
                m = eval_fn(combo)
                s = score_metric(m)
                if s > best_score:
                    best_score = s
                    best_candidate = c
                    best_metrics = m
            if best_candidate is None:
                break
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            curve.append((len(selected), best_score))

        if not curve:
            continue
        # Best size = size that maximizes the metric; require at least 2 models for "ensemble"
        candidates = [(i + 1, curve[i][1]) for i in range(len(curve)) if (i + 1) >= 2]
        if not candidates:
            candidates = [(i + 1, curve[i][1]) for i in range(len(curve))]
        best_size = max(candidates, key=lambda x: x[1])[0]
        result[f'order_{method_name}'] = selected
        result[f'curve_{method_name}'] = curve
        result[f'best_size_{method_name}'] = best_size
        result[f'best_combo_{method_name}'] = selected[:best_size]
        # Store full metrics at best size for reporting
        best_sub = selected[:best_size]
        result[f'best_metrics_{method_name}'] = eval_fn(best_sub)

    return result


def ablation_rrf(models, test_cids, true_labels, threshold=1.0, k=60, metric='recall@50'):
    """
    Try RRF with every subset of models to find the best combination.
    Only practical with <= 15 models.

    Note: Best subset is selected on the test set, so the reported metric
    is optimistic (selection bias). For unbiased evaluation, use the
    chosen subset on a held-out set.

    metric: 'recall@25', 'recall@50', 'recall@100', 'recall@200', or 'composite'.
      - Single metric: maximize that recall (e.g. recall@50 = find targets in first 50).
      - 'composite': score = recall@200*100 + recall@100*10 + mrr (original behavior).
    """
    from itertools import combinations

    model_names = sorted(models.keys())
    n_models = len(model_names)
    if n_models > 15:
        print(f"  Too many models ({n_models}) for exhaustive ablation, using top-10 by FHR")
        individual_fhr = {}
        for name in model_names:
            m = compute_ranking_metrics(test_cids, models[name], true_labels, threshold)
            individual_fhr[name] = m.get('first_hit_rank', 99999)
        model_names = sorted(individual_fhr, key=individual_fhr.get)[:10]
        n_models = len(model_names)

    best_score = -1
    best_combo = None
    best_metrics = None

    for size in range(2, min(n_models + 1, 8)):
        for combo in combinations(model_names, size):
            sub_models = {name: models[name] for name in combo}
            ens_scores = reciprocal_rank_fusion(sub_models, test_cids, k=k)
            m = compute_ranking_metrics(test_cids, ens_scores, true_labels, threshold)
            if metric == 'composite':
                score = m.get('recall@200', 0) * 100 + m.get('recall@100', 0) * 10 + m.get('mrr', 0)
            else:
                score = m.get(metric, 0)
            if score > best_score:
                best_score = score
                best_combo = combo
                best_metrics = m

    return best_combo, best_metrics


# =============================================================================
# PER-POSITIVE ANALYSIS
# =============================================================================

def per_positive_analysis(models, test_cids, true_labels, threshold=1.0):
    """
    For each test positive, show its rank in every model.
    Reveals which models are complementary.
    """
    pos_cids = [cid for cid in test_cids if true_labels.get(cid, 99) < threshold]
    if not pos_cids:
        print("  No test positives found.")
        return {}

    model_names = sorted(models.keys())
    model_ranks = {}
    for name in model_names:
        scores_arr = np.array([models[name][cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        rank_dict = {cid: int(ranks[i]) for i, cid in enumerate(test_cids)}
        model_ranks[name] = rank_dict

    analysis = {}
    for cid in pos_cids:
        bg = true_labels[cid]
        ranks_in_models = {name: model_ranks[name][cid] for name in model_names}
        best_model = min(ranks_in_models, key=ranks_in_models.get)
        worst_model = max(ranks_in_models, key=ranks_in_models.get)
        analysis[cid] = {
            'bandgap': bg,
            'ranks': ranks_in_models,
            'best_rank': ranks_in_models[best_model],
            'best_model': best_model,
            'worst_rank': ranks_in_models[worst_model],
            'worst_model': worst_model,
            'mean_rank': np.mean(list(ranks_in_models.values())),
            'n_in_top100': sum(1 for r in ranks_in_models.values() if r <= 100),
            'n_in_top200': sum(1 for r in ranks_in_models.values() if r <= 200),
        }

    return analysis


def complementarity_analysis(models, test_cids, true_labels, threshold=1.0, top_k=200):
    """
    Compute pairwise complementarity: how many positives does combining
    two models find (in top-K) that neither finds alone?
    """
    pos_cids = set(cid for cid in test_cids if true_labels.get(cid, 99) < threshold)
    model_names = sorted(models.keys())

    model_topk_pos = {}
    for name in model_names:
        scores_arr = np.array([models[name][cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        rank_dict = {cid: int(ranks[i]) for i, cid in enumerate(test_cids)}
        found = {cid for cid in pos_cids if rank_dict[cid] <= top_k}
        model_topk_pos[name] = found

    n = len(model_names)
    comp_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            union = model_topk_pos[model_names[i]] | model_topk_pos[model_names[j]]
            comp_matrix[i, j] = len(union)

    return model_names, comp_matrix, model_topk_pos


# =============================================================================
# REPORTING
# =============================================================================

def print_metrics_report(method_name, metrics, width=90):
    """Print formatted ranking metrics."""
    print(f"\n{'='*width}")
    print(f"  {method_name}")
    print(f"{'='*width}")
    print(f"  Total: {metrics['n_total']}  |  Positives: {metrics['n_positive']}  |  "
          f"FHR: {metrics.get('first_hit_rank', '?')}  |  "
          f"MRR: {metrics.get('mrr', 0):.4f}  |  "
          f"Spearman: {metrics.get('spearman_rho', 0):.4f}")

    print(f"\n  {'K':>5s}  {'Hits':>5s}  {'Recall':>8s}  {'Prec':>8s}  {'Enrich':>8s}")
    print(f"  {'-'*42}")
    for K in [10, 25, 50, 100, 200, 500]:
        if f'recall@{K}' in metrics:
            print(f"  {K:>5d}  {metrics[f'hits@{K}']:>5d}  "
                  f"{metrics[f'recall@{K}']:>8.3f}  "
                  f"{metrics[f'precision@{K}']:>8.4f}  "
                  f"{metrics[f'enrichment@{K}']:>8.1f}x")

    for K in [50, 100, 200]:
        key = f'found_in_top_{K}'
        if key in metrics and metrics[key]:
            found_str = ', '.join(f"{c}({b:.3f}eV,rank={r})"
                                  for c, b, r in metrics[key])
            print(f"\n  Found in top-{K}: {found_str}")


def print_comparison_table(all_results, title="COMPARISON"):
    """Print side-by-side comparison table."""
    print(f"\n\n{'#'*100}")
    print(f"  {title}")
    print(f"{'#'*100}")
    print(f"  {'Method':<50s}  {'FHR':>5s}  {'R@25':>5s}  {'R@50':>5s}  "
          f"{'R@100':>6s}  {'R@200':>6s}  {'R@500':>6s}  {'MRR':>6s}")
    print(f"  {'-'*95}")

    for name, m in sorted(all_results.items()):
        fhr = m.get('first_hit_rank', 9999)
        r25 = m.get('recall@25', 0)
        r50 = m.get('recall@50', 0)
        r100 = m.get('recall@100', 0)
        r200 = m.get('recall@200', 0)
        r500 = m.get('recall@500', 0)
        mrr = m.get('mrr', 0)
        print(f"  {name:<50s}  {fhr:>5d}  {r25:>5.3f}  {r50:>5.3f}  "
              f"{r100:>6.3f}  {r200:>6.3f}  {r500:>6.3f}  {mrr:>6.4f}")

    print(f"  {'-'*95}")


def print_per_positive_report(analysis, model_names):
    """Print per-positive analysis showing which models find each positive."""
    print(f"\n\n{'#'*100}")
    print(f"  PER-POSITIVE ANALYSIS")
    print(f"{'#'*100}")

    short_names = []
    for name in model_names:
        short = name.split('/')[-1][:12]
        short_names.append(short)

    header = f"  {'Positive CIF':<25s} {'BG':>5s}  {'Best':>6s}  {'Mean':>7s}  {'#top100':>7s}  {'#top200':>7s}"
    print(header)
    print(f"  {'-'*70}")

    for cid in sorted(analysis.keys(), key=lambda c: analysis[c]['mean_rank']):
        a = analysis[cid]
        print(f"  {cid:<25s} {a['bandgap']:>5.3f}  {a['best_rank']:>6d}  "
              f"{a['mean_rank']:>7.0f}  {a['n_in_top100']:>7d}  {a['n_in_top200']:>7d}")

    print(f"\n  Detailed ranks per model:")
    max_name_len = max(len(n) for n in model_names)
    for cid in sorted(analysis.keys(), key=lambda c: analysis[c]['mean_rank']):
        a = analysis[cid]
        print(f"\n    {cid} (bg={a['bandgap']:.3f}eV):")
        for name in model_names:
            rank = a['ranks'][name]
            marker = " ***" if rank <= 100 else (" *" if rank <= 200 else "")
            print(f"      {name:<{max_name_len}s}  rank={rank:>5d}{marker}")


def print_complementarity_report(model_names, comp_matrix, model_topk_pos, n_pos):
    """Print complementarity matrix."""
    print(f"\n\n{'#'*100}")
    print(f"  COMPLEMENTARITY ANALYSIS (positives found in top-200 by union of two models)")
    print(f"{'#'*100}")

    short = [n.split('/')[-1][:10] for n in model_names]

    print(f"\n  Individual coverage (top-200):")
    for i, name in enumerate(model_names):
        n_found = len(model_topk_pos[name])
        found_cids = ', '.join(sorted(model_topk_pos[name]))
        print(f"    {name:<45s}: {n_found}/{n_pos} -- {found_cids}")

    all_found = set()
    for found in model_topk_pos.values():
        all_found |= found
    print(f"\n  Union of ALL models: {len(all_found)}/{n_pos}")
    never_found = set(cid for cid in model_topk_pos[model_names[0]]) if model_names else set()
    print(f"  Never in top-200 by any model: {n_pos - len(all_found)}")

    if len(model_names) <= 12:
        print(f"\n  Pairwise union matrix (top-200 hits):")
        print(f"  {'':>12s}  " + "  ".join(f"{s:>10s}" for s in short))
        for i in range(len(model_names)):
            row = f"  {short[i]:>12s}  "
            for j in range(len(model_names)):
                row += f"  {comp_matrix[i,j]:>10d}"
            print(row)


def evaluate_subsampled(test_cids, scores, true_labels, threshold=1.0,
                        n_subsample=1500, n_resamples=30, seed=42):
    """
    Evaluate on random subsamples to estimate metric variance.
    Keeps ALL positives, subsamples negatives to create realistic-prevalence pools.
    """
    rng = np.random.RandomState(seed)
    true_bgs = np.array([true_labels[cid] for cid in test_cids])
    scores_arr = np.array([scores[cid] for cid in test_cids])
    is_pos = true_bgs < threshold

    pos_idx = np.where(is_pos)[0]
    neg_idx = np.where(~is_pos)[0]
    n_pos = len(pos_idx)

    if n_pos == 0 or n_subsample > len(test_cids):
        return {}

    n_neg_sample = min(n_subsample - n_pos, len(neg_idx))
    if n_neg_sample <= 0:
        return {}

    all_results = []
    for _ in range(n_resamples):
        neg_sample = rng.choice(neg_idx, size=n_neg_sample, replace=False)
        idx = np.concatenate([pos_idx, neg_sample])
        sub_cids = [test_cids[i] for i in idx]
        sub_scores = {test_cids[i]: scores_arr[i] for i in idx}
        sub_labels = {test_cids[i]: true_bgs[i] for i in idx}
        m = compute_ranking_metrics(sub_cids, sub_scores, sub_labels, threshold)
        all_results.append(m)

    result = {
        'n_subsample': n_subsample, 'n_resamples': n_resamples,
        'n_positives': n_pos, 'prevalence': n_pos / n_subsample,
    }
    keys = ['recall@25', 'recall@50', 'recall@100', 'recall@200',
            'enrichment@25', 'enrichment@50', 'enrichment@100',
            'first_hit_rank', 'mrr']
    for k in keys:
        vals = [m.get(k, 0) for m in all_results if k in m]
        if vals:
            result[f'{k}_mean'] = float(np.mean(vals))
            result[f'{k}_std'] = float(np.std(vals))
    return result


def evaluate_mini_splits(test_cids, scores, true_labels, threshold=1.0,
                         n_splits=5, seed=42):
    """
    Evaluate on disjoint mini-splits: all positives + disjoint negative chunks.
    Gives robust performance estimate across different negative backgrounds.
    """
    rng = np.random.RandomState(seed)
    true_bgs = np.array([true_labels[cid] for cid in test_cids])
    scores_arr = np.array([scores[cid] for cid in test_cids])
    is_pos = true_bgs < threshold

    pos_idx = np.where(is_pos)[0]
    neg_idx = np.where(~is_pos)[0]
    n_pos = len(pos_idx)

    if n_pos == 0:
        return {}

    neg_shuffled = rng.permutation(neg_idx)
    chunk_size = len(neg_shuffled) // n_splits
    if chunk_size == 0:
        return {}

    all_results = []
    result = {'n_splits': n_splits, 'n_positives': n_pos}

    for si in range(n_splits):
        start = si * chunk_size
        end = start + chunk_size if si < n_splits - 1 else len(neg_shuffled)
        neg_chunk = neg_shuffled[start:end]
        idx = np.concatenate([pos_idx, neg_chunk])
        sub_cids = [test_cids[i] for i in idx]
        sub_scores = {test_cids[i]: scores_arr[i] for i in idx}
        sub_labels = {test_cids[i]: true_bgs[i] for i in idx}
        m = compute_ranking_metrics(sub_cids, sub_scores, sub_labels, threshold)
        all_results.append(m)
        result[f'split_{si}_size'] = len(idx)
        result[f'split_{si}_recall@50'] = m.get('recall@50', 0)
        result[f'split_{si}_recall@100'] = m.get('recall@100', 0)
        result[f'split_{si}_first_hit_rank'] = m.get('first_hit_rank', 0)

    keys = ['recall@25', 'recall@50', 'recall@100', 'recall@200',
            'first_hit_rank', 'mrr']
    for k in keys:
        vals = [m.get(k, 0) for m in all_results if k in m]
        if vals:
            result[f'{k}_mean'] = float(np.mean(vals))
            result[f'{k}_std'] = float(np.std(vals))
    return result


def select_complementary_models(models, test_cids, true_labels, threshold=1.0,
                                top_k=200, max_models=10):
    """
    Greedy set-cover model selection: pick models that together maximize
    coverage of positive MOFs in top-K.

    1. For each model, find which positives appear in its top-K.
    2. Pick the model that covers the most positives.
    3. Pick the next model adding the most NEW positives.
    4. Repeat until all positives covered or max_models reached.

    Returns: (selected_names, trace, target_cids, model_topk_targets)
    """
    model_names = sorted(models.keys())
    pos_cids = set(cid for cid in test_cids if true_labels.get(cid, 99) < threshold)

    model_topk_targets = {}
    model_all_ranks = {}

    for name in model_names:
        scores_arr = np.array([models[name][cid] for cid in test_cids])
        ranks = score_to_rank(scores_arr, lower_is_better=True)
        rank_dict = {cid: int(ranks[i]) for i, cid in enumerate(test_cids)}
        model_all_ranks[name] = rank_dict
        model_topk_targets[name] = {cid for cid in pos_cids if rank_dict[cid] <= top_k}

    remaining = set(pos_cids)
    available = set(model_names)
    selected = []
    trace = []

    for step in range(max_models):
        if not remaining or not available:
            break

        best_name = None
        best_new = set()
        for name in available:
            new = model_topk_targets[name] & remaining
            if len(new) > len(best_new) or (len(new) == len(best_new) and best_name and
                    len(model_topk_targets[name]) > len(model_topk_targets.get(best_name, set()))):
                best_name = name
                best_new = new

        if best_name is None or len(best_new) == 0:
            break

        selected.append(best_name)
        remaining -= best_new
        available.discard(best_name)

        covered = pos_cids - remaining
        new_cids_with_ranks = {c: model_all_ranks[best_name].get(c, '?') for c in best_new}
        trace.append({
            'step': step + 1,
            'model': best_name,
            'new_targets': sorted(best_new),
            'new_target_ranks': new_cids_with_ranks,
            'cumulative_coverage': len(covered),
            'total_targets': len(pos_cids),
        })

    return selected, trace, pos_cids, model_topk_targets, model_all_ranks


def save_ensemble_predictions(test_cids, scores, true_labels, method_name,
                              output_dir, threshold=1.0):
    """Save ensemble predictions in standard format."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{method_name}_predictions.csv')

    with open(csv_path, 'w', newline='') as f:
        f.write("cif_id,score,predicted_binary,true_label,mode\n")
        for cid in test_cids:
            score = scores[cid]
            pred_bin = 1 if score < threshold else 0
            true_label = true_labels[cid]
            f.write(f"{cid},{score:.6f},{pred_bin},{true_label},ensemble_{method_name}\n")

    return csv_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Discovery for MOF Conductivity Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--prediction_dirs', type=str, nargs='*', default=None,
                        help='Directories containing test_predictions.csv files')
    parser.add_argument('--models', type=str, nargs='*', default=None,
                        help='Short names: NN (exp364, exp362), ML (extra_trees, random_forest), knn. '
                             'Resolved under --base_dir. Use with regular ML and experiments together.')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base dir for resolving --models (default: current dir)')
    parser.add_argument('--auto_discover', action='store_true',
                        help='Auto-discover prediction directories')
    parser.add_argument('--nn_dirs', type=str, nargs='*', default=None,
                        help='NN experiment directories (for auto_discover)')
    parser.add_argument('--clf_dir', type=str, default=None,
                        help='Embedding classifiers base dir (for auto_discover)')
    parser.add_argument('--knn_dir', type=str, default=None,
                        help='k-NN results directory (for auto_discover)')
    parser.add_argument('--output_dir', type=str, default='./ensemble_results',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Bandgap threshold for positive class (eV)')
    parser.add_argument('--rrf_k', type=int, default=60,
                        help='RRF smoothing parameter k (default: 60)')
    parser.add_argument('--vote_k', type=int, default=200,
                        help='Top-K threshold for voting ensemble')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study over model subsets (slow)')
    parser.add_argument('--ablation_metric', type=str, default='recall@50',
                        choices=['recall@25', 'recall@50', 'recall@100', 'recall@200', 'composite'],
                        help='Metric to maximize in ablation (default: recall@50)')
    parser.add_argument('--max_models', type=int, default=None,
                        help='Max models to use (greedy selection of most complementary)')
    parser.add_argument('--recommend_metric', type=str, default='recall@50',
                        choices=['recall@25', 'recall@50', 'recall@100', 'recall@200', 'composite'],
                        help='Metric to maximize (used if --recommend_metrics not given)')
    parser.add_argument('--recommend_metrics', type=str, nargs='*', default=None,
                        help='Multiple metrics for recommendation, e.g. recall@25 recall@50 recall@100')
    parser.add_argument('--recommend_max_models', type=int, default=15,
                        help='Max subset size in greedy ensemble recommendation (default: 15)')
    parser.add_argument('--greedy_k', type=int, default=200,
                        help='Top-K cutoff for greedy complementary model selection')
    parser.add_argument('--subsampled', action='store_true', default=True,
                        help='Run subsampled + mini-split robustness evaluation')
    parser.add_argument('--n_subsample', type=int, default=1500,
                        help='Subsample size for robustness evaluation')
    parser.add_argument('--no_run_subfolder', action='store_true',
                        help='Do NOT create a per-run subfolder; write directly to --output_dir (use for exhaustive/selective)')
    parser.add_argument('--exhaustive_search_limits', action='store_true',
                        help='Try ALL 2/3/4-model combinations; save only those meeting goals (>=3 @25, >=4 @50, >=5 @100)')
    parser.add_argument('--search_min_exp', type=int, default=362,
                        help='For exhaustive search: only NNs with exp number >= this (default: 362)')
    parser.add_argument('--search_max_combo_size', type=int, default=4,
                        help='Max combination size in exhaustive search (default: 4)')
    args = parser.parse_args()

    print("=" * 100)
    print("  ENSEMBLE DISCOVERY FOR MOF CONDUCTIVITY")
    print("=" * 100)

    # =========================================================================
    # 1. COLLECT PREDICTIONS
    # =========================================================================
    pred_dirs = []
    if args.models:
        pred_dirs = resolve_models_to_dirs(args.base_dir, args.models)
        print(f"  Resolved --models to {len(pred_dirs)} dirs")
    if args.auto_discover:
        pred_dirs.extend(auto_discover_models(args.nn_dirs, args.clf_dir, args.knn_dir))
        if args.prediction_dirs:
            pred_dirs.extend(args.prediction_dirs)
    elif args.prediction_dirs:
        pred_dirs.extend(args.prediction_dirs)
    if not pred_dirs:
        print("ERROR: Provide --models, --prediction_dirs, or --auto_discover with --nn_dirs/--clf_dir/--knn_dir")
        sys.exit(1)
    # Deduplicate while preserving order
    seen = set()
    pred_dirs = [d for d in pred_dirs if d not in seen and not seen.add(d)]

    # Create per-run subfolder for custom runs (so multiple runs don't overwrite)
    if not args.auto_discover and not args.no_run_subfolder:
        run_slug = _run_slug_from_dirs(pred_dirs)
        base_out = args.output_dir
        candidate = os.path.join(base_out, run_slug)
        idx = 1
        while os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, 'ensemble_results.json')):
            idx += 1
            candidate = os.path.join(base_out, f"{run_slug}_{idx}")
        args.output_dir = candidate
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n  Run subfolder: {args.output_dir}")
        print(f"  (models: {run_slug})")
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n  Discovered {len(pred_dirs)} prediction directories:")
    for d in pred_dirs:
        print(f"    {d}")

    print(f"\n--- Loading predictions ---")
    models, true_labels, test_cids = collect_predictions(pred_dirs)

    # =========================================================================
    # 2. INDIVIDUAL MODEL METRICS
    # =========================================================================
    print(f"\n\n{'#'*100}")
    print(f"  INDIVIDUAL MODEL RESULTS")
    print(f"{'#'*100}")

    individual_metrics = {}
    for name in sorted(models.keys()):
        m = compute_ranking_metrics(test_cids, models[name], true_labels, args.threshold)
        individual_metrics[name] = m

    print_comparison_table(individual_metrics, "INDIVIDUAL MODELS")

    # =========================================================================
    # 3. PER-POSITIVE ANALYSIS
    # =========================================================================
    analysis = per_positive_analysis(models, test_cids, true_labels, args.threshold)
    print_per_positive_report(analysis, sorted(models.keys()))

    # =========================================================================
    # 4. COMPLEMENTARITY ANALYSIS
    # =========================================================================
    model_names_comp, comp_matrix, model_topk_pos = complementarity_analysis(
        models, test_cids, true_labels, args.threshold, top_k=200)
    n_pos = sum(1 for cid in test_cids if true_labels.get(cid, 99) < args.threshold)
    print_complementarity_report(model_names_comp, comp_matrix, model_topk_pos, n_pos)

    # =========================================================================
    # 4b. GREEDY COMPLEMENTARY MODEL SELECTION
    # =========================================================================
    print(f"\n\n{'#'*100}")
    print(f"  GREEDY COMPLEMENTARY MODEL SELECTION (top-{args.greedy_k} coverage)")
    print(f"{'#'*100}")

    selected, trace, target_cids, model_topk, model_ranks_all = select_complementary_models(
        models, test_cids, true_labels, args.threshold,
        top_k=args.greedy_k, max_models=min(len(models), 10))

    for t in trace:
        new_str = ', '.join(f"{c}(rank={t['new_target_ranks'][c]})" for c in sorted(t['new_targets']))
        print(f"  Step {t['step']}: {t['model']}")
        print(f"    +{len(t['new_targets'])} new: {new_str}")
        print(f"    Coverage: {t['cumulative_coverage']}/{t['total_targets']}")

    print(f"\n  Recommended model set ({len(selected)} models):")
    for i, name in enumerate(selected):
        n_found = len(model_topk[name])
        print(f"    {i+1}. {name} (finds {n_found}/{len(target_cids)} in top-{args.greedy_k})")

    uncovered = target_cids - set().union(*(model_topk[n] for n in selected)) if selected else target_cids
    if uncovered:
        print(f"\n  WARNING: {len(uncovered)} targets NOT in top-{args.greedy_k} of any model:")
        for cid in sorted(uncovered):
            best_rank = min(model_ranks_all[n].get(cid, 99999) for n in models)
            print(f"    {cid} (best rank across all models: {best_rank})")

    # =========================================================================
    # 4c. RECOMMENDED MODEL COMBINATIONS (greedy by ensemble performance)
    # =========================================================================
    metrics_to_recommend = (args.recommend_metrics or [args.recommend_metric])
    if not metrics_to_recommend:
        metrics_to_recommend = [args.recommend_metric]
    recommend_by_metric = {}
    for m in metrics_to_recommend:
        if m not in ('recall@25', 'recall@50', 'recall@100', 'recall@200', 'composite'):
            continue
        rec = greedy_ensemble_forward_selection(
            models, test_cids, true_labels, args.threshold,
            metric=m, max_models=min(args.recommend_max_models, len(models)),
            rrf_k=args.rrf_k)
        if rec:
            recommend_by_metric[m] = rec
    recommend_rec = recommend_by_metric.get(args.recommend_metric) or (list(recommend_by_metric.values())[0] if recommend_by_metric else None)
    if recommend_by_metric:
        n_pos = sum(1 for cid in test_cids if true_labels.get(cid, 99) < args.threshold)
        goal_25, goal_50, goal_100 = 3, 4, 5  # discovery goals: at least this many targets in top 25/50/100
        print(f"\n\n{'#'*100}")
        print(f"  RECOMMENDED MODEL COMBINATIONS (discovery: recall@25, @50, @100)")
        print(f"  Goals: >= %d targets in top 25, >= %d in top 50, >= %d in top 100 (of %d test positives)." % (goal_25, goal_50, goal_100, n_pos))
        print(f"{'#'*100}")
        for m in metrics_to_recommend:
            if m not in recommend_by_metric:
                continue
            rec = recommend_by_metric[m]
            print(f"\n  === Metric: {m} ===")
            for method_name in ['rrf', 'rank_avg']:
                order_key = f'order_{method_name}'
                if order_key not in rec:
                    continue
                order = rec[order_key]
                best_combo = rec.get(f'best_combo_{method_name}', order)
                best_size = rec.get(f'best_size_{method_name}', len(best_combo))
                best_m = rec.get(f'best_metrics_{method_name}', {})
                label = 'RRF (k=%d)' % args.rrf_k if method_name == 'rrf' else 'Rank averaging'
                print(f"  --- {label} ---")
                print(f"  Best size: %d  Combo: %s" % (best_size, best_combo if best_combo else "—"))
                if best_m:
                    h25 = best_m.get('hits@25', int(best_m.get('recall@25', 0) * n_pos))
                    h50 = best_m.get('hits@50', int(best_m.get('recall@50', 0) * n_pos))
                    h100 = best_m.get('hits@100', int(best_m.get('recall@100', 0) * n_pos))
                    print(f"  Hits: %d @25, %d @50, %d @100  (of %d targets)" % (h25, h50, h100, n_pos))
                    g25 = "OK" if h25 >= goal_25 else "MISS"
                    g50 = "OK" if h50 >= goal_50 else "MISS"
                    g100 = "OK" if h100 >= goal_100 else "MISS"
                    print(f"  Goals (>=%d @25, >=%d @50, >=%d @100): %s @25  %s @50  %s @100" % (goal_25, goal_50, goal_100, g25, g50, g100))
                    print(f"  R@25=%.3f R@50=%.3f R@100=%.3f R@200=%.3f FHR=%s" % (
                        best_m.get('recall@25', 0), best_m.get('recall@50', 0),
                        best_m.get('recall@100', 0), best_m.get('recall@200', 0),
                        best_m.get('first_hit_rank', '?')))
        print("")

    # =========================================================================
    # 4d. EXHAUSTIVE COMBINATION SEARCH (all 2/3/4 combos; save only those meeting limits)
    # =========================================================================
    search_found_ensembles = []
    if args.exhaustive_search_limits:
        pool = filter_models_exp_min(models, args.search_min_exp)
        n_nn = sum(1 for n in pool if n.startswith('experiments/'))
        n_ml = len(pool) - n_nn
        print(f"\n\n{'#'*100}")
        print(f"  EXHAUSTIVE SEARCH (all combinations of size 2..%d)" % args.search_max_combo_size)
        print(f"  Pool: exp >= %d (%d NN) + all ML/kNN (%d) = %d models" % (args.search_min_exp, n_nn, n_ml, len(pool)))
        pool_nn = sorted(n for n in pool if n.startswith('experiments/'))
        if pool_nn:
            print(f"  NN in pool (exp >= %d only): %s" % (args.search_min_exp, ', '.join(pool_nn)))
        print(f"  Goals: >= 3 @25, >= 4 @50, >= 5 @100. Saving only combos that meet all three.")
        print(f"{'#'*100}")
        search_found_ensembles = exhaustive_search_meeting_limits(
            pool, test_cids, true_labels, args.threshold, args.rrf_k,
            goal_25=3, goal_50=4, goal_100=5, max_combo_size=args.search_max_combo_size)
        print(f"\n  Found %d combinations meeting limits (RRF + rank_avg each counted separately)." % len(search_found_ensembles))
        for i, item in enumerate(search_found_ensembles[:15]):
            print(f"    %d. %s  size=%d  hits %d @25 %d @50 %d @100" % (
                i + 1, item['name'][:60], item['combo_size'],
                item['hits@25'], item['hits@50'], item['hits@100']))
        if len(search_found_ensembles) > 15:
            print(f"    ... and %d more" % (len(search_found_ensembles) - 15))
        print("")

    if args.max_models and args.max_models < len(models):
        selected_subset = selected[:args.max_models]
        print(f"\n  --max_models={args.max_models}: Using top-{args.max_models} complementary models:")
        for name in selected_subset:
            print(f"    - {name}")
        models = {name: models[name] for name in selected_subset}
        print(f"  Filtered to {len(models)} models for ensembling")

    # =========================================================================
    # 5. ENSEMBLE METHODS
    # =========================================================================
    ensemble_metrics = {}

    # 5a. Reciprocal Rank Fusion
    print(f"\n\n--- Ensemble: Reciprocal Rank Fusion (k={args.rrf_k}) ---")
    rrf_scores = reciprocal_rank_fusion(models, test_cids, k=args.rrf_k)
    m_rrf = compute_ranking_metrics(test_cids, rrf_scores, true_labels, args.threshold)
    print_metrics_report(f"RRF Ensemble (k={args.rrf_k})", m_rrf)
    ensemble_metrics[f'rrf_k{args.rrf_k}'] = m_rrf
    save_ensemble_predictions(test_cids, rrf_scores, true_labels, f'rrf_k{args.rrf_k}',
                              args.output_dir, args.threshold)

    # Try different k values
    for k_val in [10, 30, 60, 100, 200]:
        if k_val == args.rrf_k:
            continue
        rrf_k = reciprocal_rank_fusion(models, test_cids, k=k_val)
        m_k = compute_ranking_metrics(test_cids, rrf_k, true_labels, args.threshold)
        ensemble_metrics[f'rrf_k{k_val}'] = m_k

    # 5b. Rank Averaging
    print(f"\n--- Ensemble: Rank Averaging ---")
    ra_scores = rank_averaging(models, test_cids)
    m_ra = compute_ranking_metrics(test_cids, ra_scores, true_labels, args.threshold)
    print_metrics_report("Rank Averaging", m_ra)
    ensemble_metrics['rank_avg'] = m_ra
    save_ensemble_predictions(test_cids, ra_scores, true_labels, 'rank_avg',
                              args.output_dir, args.threshold)

    # 5c. Top-K Voting
    for vote_k in [50, 100, 200, 500]:
        print(f"\n--- Ensemble: Top-{vote_k} Voting ---")
        vote_scores = top_k_voting(models, test_cids, K=vote_k)
        m_vote = compute_ranking_metrics(test_cids, vote_scores, true_labels, args.threshold)
        print_metrics_report(f"Top-{vote_k} Voting", m_vote)
        ensemble_metrics[f'vote_top{vote_k}'] = m_vote
        if vote_k == args.vote_k:
            save_ensemble_predictions(test_cids, vote_scores, true_labels,
                                      f'vote_top{vote_k}', args.output_dir, args.threshold)

    # 5d. Score Averaging
    print(f"\n--- Ensemble: Score Averaging ---")
    sa_scores = score_averaging(models, test_cids)
    m_sa = compute_ranking_metrics(test_cids, sa_scores, true_labels, args.threshold)
    print_metrics_report("Score Averaging (normalized)", m_sa)
    ensemble_metrics['score_avg'] = m_sa
    save_ensemble_predictions(test_cids, sa_scores, true_labels, 'score_avg',
                              args.output_dir, args.threshold)

    # 5e. Weighted RRF (uniform weights for valid evaluation; test-derived weights would leak)
    print(f"\n--- Ensemble: Weighted RRF (uniform weights, no test leakage) ---")
    model_weights = {name: 1.0 for name in models}

    wrrf_scores = weighted_rrf(models, test_cids, model_weights, k=args.rrf_k)
    m_wrrf = compute_ranking_metrics(test_cids, wrrf_scores, true_labels, args.threshold)
    print_metrics_report("Weighted RRF", m_wrrf)
    ensemble_metrics['weighted_rrf'] = m_wrrf
    save_ensemble_predictions(test_cids, wrrf_scores, true_labels, 'weighted_rrf',
                              args.output_dir, args.threshold)

    # 5f. Stacking
    print(f"\n--- Ensemble: Stacking (LogReg meta-learner) ---")
    stack_scores = stacking_ensemble(models, test_cids, true_labels, args.threshold)
    m_stack = compute_ranking_metrics(test_cids, stack_scores, true_labels, args.threshold)
    print_metrics_report("Stacking (LogReg)", m_stack)
    ensemble_metrics['stacking'] = m_stack
    save_ensemble_predictions(test_cids, stack_scores, true_labels, 'stacking',
                              args.output_dir, args.threshold)

    # =========================================================================
    # 6. ABLATION (optional)
    # =========================================================================
    best_combo = None
    if args.ablation and len(models) <= 15:
        print(f"\n\n--- Ablation: Finding best model subset (maximize {args.ablation_metric}) ---")
        best_combo, best_m = ablation_rrf(models, test_cids, true_labels,
                                          args.threshold, k=args.rrf_k, metric=args.ablation_metric)
        if best_combo:
            print(f"  Best subset ({len(best_combo)} models):")
            for name in best_combo:
                print(f"    - {name}")
            print_metrics_report(f"Best Subset RRF ({len(best_combo)} models)", best_m)
            ensemble_metrics['ablation_best'] = best_m

    # =========================================================================
    # 7. FINAL COMPARISON
    # =========================================================================
    all_results = {}
    for name, m in individual_metrics.items():
        all_results[f"[single] {name}"] = m
    for name, m in ensemble_metrics.items():
        all_results[f"[ENSEMBLE] {name}"] = m

    print_comparison_table(all_results, "FINAL COMPARISON: Individual vs Ensemble")

    # =========================================================================
    # 7b. ROBUSTNESS EVALUATION (subsampled + mini-splits)
    # =========================================================================
    sub_results = {}
    mini_results = {}
    if args.subsampled:
        print(f"\n\n{'#'*100}")
        print(f"  ROBUSTNESS EVALUATION (subsampled + mini-splits)")
        print(f"{'#'*100}")

        best_ens_by_r200 = max(ensemble_metrics, key=lambda n: ensemble_metrics[n].get('recall@200', 0))

        methods_to_eval = {
            f'rrf_k{args.rrf_k}': rrf_scores,
            'rank_avg': ra_scores,
            'score_avg': sa_scores,
            'weighted_rrf': wrrf_scores,
            'stacking': stack_scores,
        }

        print(f"\n  --- Subsampled evaluation (N={args.n_subsample}, 30 resamples) ---")
        print(f"  {'Method':<30s}  {'R@25':>10s}  {'R@50':>10s}  {'R@100':>10s}  {'FHR':>12s}  {'MRR':>10s}")
        print(f"  {'-'*85}")
        sub_results = {}
        for method_name, method_scores in methods_to_eval.items():
            sub = evaluate_subsampled(test_cids, method_scores, true_labels,
                                      args.threshold, args.n_subsample)
            sub_results[method_name] = sub
            if sub:
                print(f"  {method_name:<30s}  "
                      f"{sub.get('recall@25_mean',0):>5.3f}+-{sub.get('recall@25_std',0):.3f}  "
                      f"{sub.get('recall@50_mean',0):>5.3f}+-{sub.get('recall@50_std',0):.3f}  "
                      f"{sub.get('recall@100_mean',0):>5.3f}+-{sub.get('recall@100_std',0):.3f}  "
                      f"{sub.get('first_hit_rank_mean',0):>5.0f}+-{sub.get('first_hit_rank_std',0):>4.0f}  "
                      f"{sub.get('mrr_mean',0):>5.4f}+-{sub.get('mrr_std',0):.4f}")

        print(f"\n  --- Mini-split evaluation (5 disjoint negative chunks) ---")
        print(f"  {'Method':<30s}  {'R@50':>10s}  {'R@100':>10s}  {'FHR':>12s}")
        print(f"  {'-'*65}")
        mini_results = {}
        for method_name, method_scores in methods_to_eval.items():
            mini = evaluate_mini_splits(test_cids, method_scores, true_labels,
                                        args.threshold)
            mini_results[method_name] = mini
            if mini:
                print(f"  {method_name:<30s}  "
                      f"{mini.get('recall@50_mean',0):>5.3f}+-{mini.get('recall@50_std',0):.3f}  "
                      f"{mini.get('recall@100_mean',0):>5.3f}+-{mini.get('recall@100_std',0):.3f}  "
                      f"{mini.get('first_hit_rank_mean',0):>5.0f}+-{mini.get('first_hit_rank_std',0):>4.0f}")

    # Best ensemble
    best_ens_name = max(ensemble_metrics, key=lambda n: ensemble_metrics[n].get('recall@200', 0))
    best_ens = ensemble_metrics[best_ens_name]
    best_single_name = max(individual_metrics, key=lambda n: individual_metrics[n].get('recall@200', 0))
    best_single = individual_metrics[best_single_name]

    print(f"\n  SUMMARY:")
    print(f"    Best single model:   {best_single_name}")
    print(f"      FHR={best_single.get('first_hit_rank','?')}, "
          f"R@100={best_single.get('recall@100',0):.3f}, "
          f"R@200={best_single.get('recall@200',0):.3f}")
    print(f"    Best ensemble:       {best_ens_name}")
    print(f"      FHR={best_ens.get('first_hit_rank','?')}, "
          f"R@100={best_ens.get('recall@100',0):.3f}, "
          f"R@200={best_ens.get('recall@200',0):.3f}")

    improvement = best_ens.get('recall@200', 0) - best_single.get('recall@200', 0)
    print(f"    Ensemble recall@200 improvement: {improvement:+.3f} "
          f"({improvement / max(best_single.get('recall@200', 0.001), 0.001) * 100:+.1f}%)")

    # =========================================================================
    # 7c. ADD ENSEMBLE RANKS TO PER-POSITIVE (so fig1 heatmap includes ensemble columns)
    # =========================================================================
    ensemble_score_dicts = [
        (f'rrf_k{args.rrf_k}', rrf_scores),
        ('rank_avg', ra_scores),
        ('score_avg', sa_scores),
        ('weighted_rrf', wrrf_scores),
        ('stacking', stack_scores),
    ]
    if args.ablation and best_combo:
        sub_models = {n: models[n] for n in best_combo}
        ablation_scores = reciprocal_rank_fusion(sub_models, test_cids, k=args.rrf_k)
        ensemble_score_dicts.append(('ablation_best', ablation_scores))

    for method_name, scores_dict in ensemble_score_dicts:
        rank_dict = _scores_to_rank_dict(scores_dict, test_cids, lower_is_better=True)
        for cid in analysis:
            analysis[cid]['ranks'][method_name] = rank_dict.get(cid, 9999)
        for cid in analysis:
            r = rank_dict.get(cid, 9999)
            if r < analysis[cid]['best_rank']:
                analysis[cid]['best_rank'] = r
                analysis[cid]['best_model'] = method_name
    for item in search_found_ensembles:
        name = item['name']
        rank_dict = item['per_positive_ranks']
        for cid in analysis:
            analysis[cid]['ranks'][name] = rank_dict.get(cid, 9999)
        for cid in analysis:
            r = rank_dict.get(cid, 9999)
            if r < analysis[cid]['best_rank']:
                analysis[cid]['best_rank'] = r
                analysis[cid]['best_model'] = name
    for cid in analysis:
        analysis[cid]['n_in_top100'] = sum(1 for r in analysis[cid]['ranks'].values() if r <= 100)
        analysis[cid]['n_in_top200'] = sum(1 for r in analysis[cid]['ranks'].values() if r <= 200)

    # =========================================================================
    # 8. SAVE RESULTS
    # =========================================================================
    results = {
        'individual_models': {},
        'ensemble_methods': {},
        'per_positive': {},
        'complementarity': {
            'model_names': model_names_comp,
            'top200_union_count': int(comp_matrix.max()),
        },
        'greedy_selection': {
            'selected_models': selected,
            'trace': [{k: v for k, v in t.items() if k != 'new_target_ranks'}
                      for t in trace],
        },
    }

    for name, m in individual_metrics.items():
        clean = {}
        for k, v in m.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                clean[k] = float(v)
            elif isinstance(v, list):
                try:
                    clean[k] = [(str(a), float(b), int(c)) for a, b, c in v]
                except (ValueError, TypeError):
                    pass
        results['individual_models'][name] = clean

    for name, m in ensemble_metrics.items():
        clean = {}
        for k, v in m.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                clean[k] = float(v)
            elif isinstance(v, list):
                try:
                    clean[k] = [(str(a), float(b), int(c)) for a, b, c in v]
                except (ValueError, TypeError):
                    pass
        results['ensemble_methods'][name] = clean

    for cid, a in analysis.items():
        results['per_positive'][cid] = {
            'bandgap': a['bandgap'],
            'best_rank': a['best_rank'],
            'best_model': a['best_model'],
            'mean_rank': a['mean_rank'],
            'n_in_top100': a['n_in_top100'],
            'n_in_top200': a['n_in_top200'],
            'ranks': a['ranks'],
        }

    if sub_results:
        results['subsampled_evaluation'] = sub_results
    if mini_results:
        results['mini_split_evaluation'] = mini_results

    if search_found_ensembles:
        results['search_found_ensembles'] = []
        for item in search_found_ensembles:
            rec = {
                'name': item['name'],
                'model_names': item['model_names'],
                'method': item['method'],
                'per_positive_ranks': {str(cid): int(r) for cid, r in item['per_positive_ranks'].items()},
                'hits@25': item['hits@25'], 'hits@50': item['hits@50'], 'hits@100': item['hits@100'],
                'recall@25': item['recall@25'], 'recall@50': item['recall@50'],
                'recall@100': item['recall@100'], 'recall@200': item['recall@200'],
                'first_hit_rank': item['first_hit_rank'], 'mrr': item['mrr'],
                'combo_size': item['combo_size'],
            }
            results['search_found_ensembles'].append(rec)

    results['run_metadata'] = {
        'model_paths': sorted(models.keys()),
        'model_slug': _run_slug_from_dirs(list(models.keys())),
        'n_models': len(models),
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    }

    if recommend_by_metric:
        def _serialize_rec(rec):
            out = {
                'order_rrf': rec.get('order_rrf', []),
                'order_rank_avg': rec.get('order_rank_avg', []),
                'curve_rrf': [[int(s), float(v)] for s, v in rec.get('curve_rrf', [])],
                'curve_rank_avg': [[int(s), float(v)] for s, v in rec.get('curve_rank_avg', [])],
                'best_size_rrf': rec.get('best_size_rrf'),
                'best_size_rank_avg': rec.get('best_size_rank_avg'),
                'best_combo_rrf': rec.get('best_combo_rrf', []),
                'best_combo_rank_avg': rec.get('best_combo_rank_avg', []),
            }
            for key in ['best_metrics_rrf', 'best_metrics_rank_avg']:
                if key in rec and rec[key]:
                    out[key] = {k: float(v) for k, v in rec[key].items()
                                if isinstance(v, (int, float, np.integer, np.floating))}
            return out

        by_metric_ser = {m: _serialize_rec(rec) for m, rec in recommend_by_metric.items()}
        primary = recommend_by_metric.get(args.recommend_metric) or next(iter(recommend_by_metric.values()))
        results['recommended_combinations'] = {
            'metrics': list(recommend_by_metric.keys()),
            'by_metric': by_metric_ser,
            'primary_metric': args.recommend_metric,
            **( _serialize_rec(primary) if primary else {} ),
        }
        rec_txt = os.path.join(args.output_dir, 'recommended_combinations.txt')
        with open(rec_txt, 'w') as f:
            f.write("# Recommended model combinations (discovery: recall@25, @50, @100)\n")
            f.write("# Goals: >= 3 targets in top 25, >= 4 in top 50, >= 5 in top 100.\n\n")
            for m in recommend_by_metric:
                rec = recommend_by_metric[m]
                f.write("## Metric: %s\n" % m)
                for method_name, label in [('rrf', 'RRF'), ('rank_avg', 'Rank avg')]:
                    combo = rec.get('best_combo_%s' % method_name, [])
                    size = rec.get('best_size_%s' % method_name, 0)
                    best_m = rec.get('best_metrics_%s' % method_name, {})
                    h25 = best_m.get('hits@25', 0)
                    h50 = best_m.get('hits@50', 0)
                    h100 = best_m.get('hits@100', 0)
                    f.write("%s: best size %d  %s\n" % (label, size, ', '.join(combo)))
                    f.write("  Hits: %d @25, %d @50, %d @100  Goals: %s @25 %s @50 %s @100\n" % (
                        h25, h50, h100,
                        'OK' if h25 >= 3 else 'MISS', 'OK' if h50 >= 4 else 'MISS', 'OK' if h100 >= 5 else 'MISS'))
                f.write("\n")
        print(f"  Recommended combinations: {rec_txt}")

    results_path = os.path.join(args.output_dir, 'ensemble_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    run_meta_path = os.path.join(args.output_dir, 'run_metadata.json')
    with open(run_meta_path, 'w') as f:
        json.dump(results['run_metadata'], f, indent=2, default=str)
    print(f"  Run metadata: {run_meta_path}")

    print(f"\n{'='*100}")
    print(f"  Ensemble pipeline complete.")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
