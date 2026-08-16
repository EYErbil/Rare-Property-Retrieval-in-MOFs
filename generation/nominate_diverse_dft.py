#!/usr/bin/env python3
"""
Diversity-aware candidate nomination from reciprocal-rank-fused predictions.

The matrix supplied by --embeddings_path is the diversity coordinate used for
PCA, clustering, cosine distances, MMR, exploration, and visualization. The
paper entrypoints supply SOAP descriptors here; PMTransformer embeddings are
used by the predictive models and for the labeled-data split, not as the
prospective nomination-diversity coordinate.

Canonical paper usage
---------------------
  python nominate_diverse_dft.py \
    --embeddings_path  /path/to/generated_soap_descriptors.npz \
    --embedding_key soap_descriptors \
    --embedding_label SOAP \
    --prediction_csvs  exp364=/path/to/exp364/inference_predictions.csv \
                       smote_extra_trees=/path/to/extra_trees/test_predictions.csv \
    --nn_models exp364 \
    --ml_models smote_extra_trees \
    --output_dir paper_results/nomination-SOAP \
    --pool_size 500 --pca_components 50 --n_clusters 20 \
    --kmeans_n_init 10 --max_per_cluster 1 \
    --mmr_lambdas 0.2 0.3 0.4 --budget 25 \
    --exploration_budget 5 --exploration_pool_lo 500 \
    --exploration_pool_hi 2000 --rrf_k 60 --seed 42

The paper wrappers pin the remaining strategy weights explicitly; see
``scripts/12_nominate_paper_candidates.sh`` in the generation module and
``scripts/07_nominate_candidates.sh`` in the screening module.
"""

import os
import sys
import csv
import json
import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import sparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Diversity-matrix loading
# ---------------------------------------------------------------------------


def _as_python_strings(values):
    """Return one-dimensional identifier data as ordinary Python strings."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"structure identifiers must be 1-D, got shape {arr.shape}")
    return [str(value) for value in arr.tolist()]


def _validate_diversity_matrix(cids, matrix, path):
    """Validate an embedding/descriptor matrix without converting sparse data."""
    if matrix.ndim != 2:
        raise ValueError(f"diversity matrix in {path} must be 2-D, got {matrix.ndim}-D")
    if matrix.shape[0] != len(cids):
        raise ValueError(
            f"row/identifier mismatch in {path}: {matrix.shape[0]} rows for "
            f"{len(cids)} identifiers"
        )
    if matrix.shape[1] == 0:
        raise ValueError(f"diversity matrix in {path} has zero columns")
    if len(set(cids)) != len(cids):
        raise ValueError(f"duplicate structure identifiers found in {path}")

    values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"diversity matrix in {path} must be numeric, got {values.dtype}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"diversity matrix in {path} contains NaN or infinite values")


def load_diversity_matrix(path, embedding_key="embeddings"):
    """Load identifiers and a dense or CSR diversity matrix from an NPZ file.

    Dense archives use ``cif_ids`` (or ``cids``) plus *embedding_key*. The
    custom sparse QMOF/SOAP archives use ``sp_data``, ``sp_indices``,
    ``sp_indptr``, and ``sp_shape``. SciPy-style unprefixed CSR keys are also
    accepted when identifiers are stored in the same archive.

    Returns ``(cids, matrix, resolved_key, storage_format)``. Sparse inputs stay
    CSR throughout; this function never materializes the full dense matrix.
    """
    path = os.path.abspath(path)
    with np.load(path, allow_pickle=True) as archive:
        keys = set(archive.files)
        id_key = next((key for key in ("cif_ids", "cids") if key in keys), None)
        if id_key is None:
            raise ValueError(
                f"{path} has no structure-ID array (expected 'cif_ids' or 'cids')"
            )
        cids = _as_python_strings(archive[id_key])

        prefixed_csr = {"sp_data", "sp_indices", "sp_indptr", "sp_shape"}
        scipy_csr = {"data", "indices", "indptr", "shape"}
        if prefixed_csr <= keys:
            prefix = "sp_"
            matrix = sparse.csr_matrix(
                (
                    archive[f"{prefix}data"],
                    archive[f"{prefix}indices"],
                    archive[f"{prefix}indptr"],
                ),
                shape=tuple(int(value) for value in archive[f"{prefix}shape"]),
            )
            resolved_key = "sp_*"
            storage_format = "custom CSR"
        elif scipy_csr <= keys:
            matrix = sparse.csr_matrix(
                (archive["data"], archive["indices"], archive["indptr"]),
                shape=tuple(int(value) for value in archive["shape"]),
            )
            resolved_key = "scipy-csr"
            storage_format = "CSR"
        else:
            dense_key = embedding_key
            if dense_key not in keys and embedding_key == "embeddings":
                dense_key = next(
                    (key for key in ("descriptors", "soap_descriptors") if key in keys),
                    dense_key,
                )
            if dense_key not in keys:
                raise ValueError(
                    f"key '{embedding_key}' not found in {path}; available keys: "
                    f"{sorted(keys)}"
                )
            matrix = archive[dense_key]
            if matrix.dtype == object and matrix.shape == ():
                matrix = matrix.item()
            if sparse.issparse(matrix):
                matrix = matrix.tocsr()
                storage_format = "CSR object"
            else:
                matrix = np.asarray(matrix)
                storage_format = "dense"
            resolved_key = dense_key

    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=False)
        matrix.sum_duplicates()
        matrix.sort_indices()
    _validate_diversity_matrix(cids, matrix, path)
    return cids, matrix, resolved_key, storage_format


def take_matrix_rows(matrix, row_indices):
    """Select rows while preserving CSR storage for sparse descriptors."""
    rows = np.asarray(row_indices, dtype=np.intp)
    selected = matrix[rows]
    return selected.tocsr() if sparse.issparse(selected) else np.asarray(selected)


def common_prediction_universe(embedding_cids, prediction_models):
    """Embedding IDs present in every required prediction model, in NPZ order."""
    if not prediction_models:
        return []
    common = set.intersection(*(set(scores) for scores in prediction_models.values()))
    return [cid for cid in embedding_cids if cid in common]


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


def infer_score_direction(mode_str):
    """regression / knn → lower = better;  multiclass → higher = better."""
    if not mode_str:
        return True
    mode_lower = str(mode_str).lower().strip()
    if any(tag in mode_lower for tag in ("regression", "knn", "sim_to_pos", "ensemble")):
        return True
    return False


def load_predictions_from_csv(csv_path, normalize_lower_is_better=True):
    """Load {cif_id: score} from test_predictions.csv / inference_predictions.csv.
    If *normalize_lower_is_better*, multiclass scores are inverted so that
    lower = better everywhere (consistent with NN regression).
    """
    preds = {}
    mode_str = "regression"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("cif_id", "").strip()
            if not cid:
                continue
            try:
                score = float(row["score"])
                if not np.isfinite(score):
                    continue
                preds[cid] = score
                if "mode" in row and row["mode"]:
                    mode_str = row["mode"]
            except (KeyError, ValueError):
                continue
    if not preds:
        return preds
    if normalize_lower_is_better and not infer_score_direction(mode_str):
        max_s = max(preds.values())
        preds = {cid: max_s - s for cid, s in preds.items()}
    return preds


# ---------------------------------------------------------------------------
# Rank / fusion helpers
# ---------------------------------------------------------------------------


def score_to_rank(scores, lower_is_better=True):
    """Ordinal ranks (1 = best)."""
    arr = np.asarray(scores, dtype=float)
    order = np.argsort(arr) if lower_is_better else np.argsort(-arr)
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks


def _fill_missing(scores_arr, lower_is_better):
    arr = np.asarray(scores_arr, dtype=float)
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    fill = (np.nanmax(arr) + 1e9) if lower_is_better else (np.nanmin(arr) - 1e9)
    arr[mask] = fill
    return arr


def reciprocal_rank_fusion(models, cids, k=60):
    """RRF over *models* (each {cid: score}, lower = better).
    Returns {cid: score} where **lower = better**.
    """
    rrf = defaultdict(float)
    for _name, scores in models.items():
        arr = np.array([scores.get(c, np.nan) for c in cids])
        arr = _fill_missing(arr, lower_is_better=True)
        ranks = score_to_rank(arr, lower_is_better=True)
        for i, c in enumerate(cids):
            rrf[c] += 1.0 / (k + ranks[i])
    # Invert so lower = better
    mx = max(rrf.values()) if rrf else 0.0
    return {c: mx - rrf[c] for c in cids}


# ---------------------------------------------------------------------------
# Uncertainty / disagreement
# ---------------------------------------------------------------------------


def compute_uncertainty(models, cids, nn_names, ml_names):
    """For every CID compute rank-based uncertainty signals.

    Returns dict-of-dicts with keys:
      rank_std          – std of ranks across all models
      rank_range        – max rank – min rank
      nn_ml_disagreement – |mean NN rank – mean ML rank|

    All values are raw (not normalised).
    """
    # Pre-compute per-model rank arrays  (each array has len == len(cids))
    model_ranks = {}
    for name, scores in models.items():
        arr = np.array([scores.get(c, np.nan) for c in cids])
        arr = _fill_missing(arr, lower_is_better=True)
        model_ranks[name] = score_to_rank(arr, lower_is_better=True)

    rank_matrix = np.column_stack([model_ranks[n] for n in models])  # (N, M)
    std = rank_matrix.std(axis=1)
    rng = rank_matrix.max(axis=1) - rank_matrix.min(axis=1)

    nn_idx = [i for i, n in enumerate(models) if n in nn_names]
    ml_idx = [i for i, n in enumerate(models) if n in ml_names]
    if nn_idx and ml_idx:
        nn_mean = rank_matrix[:, nn_idx].mean(axis=1)
        ml_mean = rank_matrix[:, ml_idx].mean(axis=1)
        disagree = np.abs(nn_mean - ml_mean)
    else:
        disagree = np.zeros(len(cids))

    out = {}
    for i, c in enumerate(cids):
        out[c] = {
            "rank_std": float(std[i]),
            "rank_range": float(rng[i]),
            "nn_ml_disagreement": float(disagree[i]),
        }
    return out


def normalize_01(values):
    """Min-max normalise a 1-D array to [0, 1]."""
    arr = np.asarray(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_pool(embeddings, n_clusters, seed, pca_components=50,
                 kmeans_n_init=10):
    """Centered PCA followed by k-means, matching the historical paper run.

    A CSR input is converted to dense only after shortlist row selection. Thus
    the bounded pool (200 rows by default; 500 in the paper run) receives the
    same centered PCA as the historical dense implementation, while the full
    20k-by-244k QMOF SOAP cache is never densified.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    n_samples, n_features = embeddings.shape
    if n_samples == 0:
        raise ValueError("cannot cluster an empty shortlist pool")
    if n_samples == 1:
        return np.zeros(1, dtype=int), float("nan")

    pca_input = embeddings.toarray() if sparse.issparse(embeddings) else embeddings
    reduced_dim = max(1, min(pca_components, n_features, n_samples))
    reducer = PCA(n_components=reduced_dim, random_state=seed)
    reduced = reducer.fit_transform(pca_input)

    n_clusters = max(1, min(n_clusters, n_samples))
    km = KMeans(n_clusters=n_clusters, n_init=kmeans_n_init,
                random_state=seed)
    labels = km.fit_predict(reduced)

    sil = float("nan")
    n_labels = len(set(labels))
    if 1 < n_labels < n_samples:
        from sklearn.metrics import silhouette_score as sil_fn
        sil = float(sil_fn(reduced, labels))
    return labels, sil


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def strategy_cluster_quota(pool_cids, pool_quality, cluster_labels,
                           budget, max_per_cluster):
    """Round-robin: best quality per cluster, cycle through clusters
    sorted by best quality, cap at *max_per_cluster*."""
    # Group pool indices by cluster
    cluster_members = defaultdict(list)
    for i, c in enumerate(pool_cids):
        cluster_members[cluster_labels[i]].append(i)

    # Sort each cluster's members by quality (ascending = better)
    for cl in cluster_members:
        cluster_members[cl].sort(key=lambda i: pool_quality[i])

    # Sort clusters by best (lowest) quality score in each
    cluster_order = sorted(cluster_members.keys(),
                           key=lambda cl: pool_quality[cluster_members[cl][0]])

    selected = []
    selected_set = set()
    picked_per_cluster = defaultdict(int)
    pointer = {cl: 0 for cl in cluster_order}

    while len(selected) < budget:
        progress = False
        for cl in cluster_order:
            if len(selected) >= budget:
                break
            if picked_per_cluster[cl] >= max_per_cluster:
                continue
            members = cluster_members[cl]
            while pointer[cl] < len(members):
                idx = members[pointer[cl]]
                pointer[cl] += 1
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    picked_per_cluster[cl] += 1
                    progress = True
                    break
            else:
                continue
        if not progress:
            # Relax cap
            max_per_cluster += 1

    return [pool_cids[i] for i in selected[:budget]]


def strategy_mmr(pool_cids, pool_quality_norm, dist_matrix, lam, budget):
    """Maximal Marginal Relevance.

    score_i = lambda * quality_i  +  (1-lambda) * min_dist(i, selected)
    quality_norm: 1 = best, 0 = worst  (inverted from rank).
    dist_matrix: pairwise cosine distance (higher = more diverse).
    """
    n = len(pool_cids)
    selected_idx = []
    remaining = set(range(n))

    # Seed with best quality
    first = int(np.argmax(pool_quality_norm))
    selected_idx.append(first)
    remaining.discard(first)

    for _ in range(budget - 1):
        if not remaining:
            break
        best_score = -np.inf
        best_idx = -1
        for i in sorted(remaining):
            min_d = min(dist_matrix[i, j] for j in selected_idx)
            score = lam * pool_quality_norm[i] + (1.0 - lam) * min_d
            if score > best_score:
                best_score = score
                best_idx = i
        selected_idx.append(best_idx)
        remaining.discard(best_idx)

    return [pool_cids[i] for i in selected_idx[:budget]]


def strategy_uncertainty_quota(pool_cids, pool_combined_score,
                               cluster_labels, budget, max_per_cluster):
    """Like cluster_quota but ranks within each cluster by a combined
    quality + uncertainty + disagreement score (lower = better)."""
    cluster_members = defaultdict(list)
    for i, c in enumerate(pool_cids):
        cluster_members[cluster_labels[i]].append(i)

    for cl in cluster_members:
        cluster_members[cl].sort(key=lambda i: pool_combined_score[i])

    cluster_order = sorted(cluster_members.keys(),
                           key=lambda cl: pool_combined_score[cluster_members[cl][0]])

    selected = []
    selected_set = set()
    picked_per_cluster = defaultdict(int)
    pointer = {cl: 0 for cl in cluster_order}

    while len(selected) < budget:
        progress = False
        for cl in cluster_order:
            if len(selected) >= budget:
                break
            if picked_per_cluster[cl] >= max_per_cluster:
                continue
            members = cluster_members[cl]
            while pointer[cl] < len(members):
                idx = members[pointer[cl]]
                pointer[cl] += 1
                if idx not in selected_set:
                    selected.append(idx)
                    selected_set.add(idx)
                    picked_per_cluster[cl] += 1
                    progress = True
                    break
            else:
                continue
        if not progress:
            max_per_cluster += 1

    return [pool_cids[i] for i in selected[:budget]]


# ---------------------------------------------------------------------------
# SOAP diversity check
# ---------------------------------------------------------------------------


def soap_diversity(soap_emb, cids, selected_cids, label=""):
    """Compute pairwise cosine distances for selected set in SOAP space.
    Returns (min, mean, max) distance and the distance matrix.
    """
    from sklearn.metrics.pairwise import cosine_distances
    idx = [i for i, c in enumerate(cids) if c in set(selected_cids)]
    if len(idx) < 2:
        return None, None, None, None
    sub = soap_emb[idx]
    D = cosine_distances(sub)
    np.fill_diagonal(D, np.nan)
    vals = D[~np.isnan(D)]
    return float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)), D


# ---------------------------------------------------------------------------
# UMAP visualisation
# ---------------------------------------------------------------------------


def plot_umap(diversity_matrix, diversity_cids, pool_set, nominees_dict,
              old_nominees, output_dir, umap_cache_path=None, seed=42,
              emb_label="SOAP"):
    """2-D UMAP of the prediction-complete nomination universe with overlays.

    nominees_dict: {strategy_name: [cid_list]}
    old_nominees: set of cids
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from umap import UMAP
    except ImportError:
        print("  [UMAP] matplotlib or umap-learn not installed — skipping plot.")
        return

    # Try cache
    coords = None
    if umap_cache_path and os.path.isfile(umap_cache_path):
        try:
            cache = np.load(umap_cache_path, allow_pickle=True)
            cached_ids = list(cache.get("cif_ids", cache.get("cids", [])))
            cached_coords = cache.get("coords", cache.get("coordinates", None))
            cached_ids = [str(cid) for cid in cached_ids]
            if cached_coords is not None and len(cached_ids) == len(diversity_cids):
                if set(cached_ids) == set(diversity_cids):
                    # Reorder to match diversity_cids
                    id2idx = {c: i for i, c in enumerate(cached_ids)}
                    reorder = [id2idx[c] for c in diversity_cids]
                    coords = cached_coords[reorder]
                    print("  [UMAP] Loaded from cache.")
        except Exception:
            pass

    if coords is None:
        print("  [UMAP] Fitting UMAP (may take a minute)...")
        reducer = UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=seed)
        coords = reducer.fit_transform(diversity_matrix)
        # Save cache
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        np.savez_compressed(os.path.join(plots_dir, "umap_cache.npz"),
                            cif_ids=np.array(diversity_cids), coords=coords)

    cid2idx = {c: i for i, c in enumerate(diversity_cids)}
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 1: Pool + nominees ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=1, c="lightgrey", alpha=0.3,
               label="Prediction-complete nomination universe")

    pool_idx = [cid2idx[c] for c in pool_set if c in cid2idx]
    if pool_idx:
        ax.scatter(coords[pool_idx, 0], coords[pool_idx, 1],
                   s=8, c="steelblue", alpha=0.4, label="Shortlist pool")

    colours = ["red", "green", "purple", "orange", "cyan"]
    for ci, (sname, scids) in enumerate(nominees_dict.items()):
        idx = [cid2idx[c] for c in scids if c in cid2idx]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       s=60, c=colours[ci % len(colours)], edgecolors="black",
                       linewidth=0.5, alpha=0.9, label=sname, zorder=5)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Diversity-aware nominees — UMAP of {emb_label} embeddings")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "umap_diverse_nominees.png"), dpi=200)
    plt.close(fig)

    # --- Plot 2: Old vs new comparison ---
    combined_cids = set()
    for scids in nominees_dict.values():
        combined_cids.update(scids)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax_i, (title, cids_set, colour) in enumerate([
        ("Old nominees (consensus)", old_nominees, "red"),
        ("New nominees (diversity-aware)", combined_cids, "green"),
    ]):
        ax = axes[ax_i]
        ax.scatter(coords[:, 0], coords[:, 1], s=1, c="lightgrey", alpha=0.3)
        idx = [cid2idx[c] for c in cids_set if c in cid2idx]
        if idx:
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       s=80, c=colour, edgecolors="black", linewidth=0.7, zorder=5)
        ax.set_title(title, fontsize=13)
    fig.suptitle(f"Old vs New nominees on {emb_label} UMAP", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "umap_comparison_old_vs_new.png"), dpi=200)
    plt.close(fig)

    print(f"  [UMAP] Saved plots to {plots_dir}/")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(output_dir, args_dict, pool_size_actual, n_clusters, sil_score,
                 strategy_results, combined_cids, old_nominees,
                 verification_soap_old, verification_soap_new,
                 diversity_old_stats, diversity_new_stats,
                 uncertainty_info, pool_cids, unc_data):
    """Write a Markdown diversity report."""
    path = os.path.join(output_dir, "diversity_report.md")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    overlap = set(combined_cids) & set(old_nominees)

    with open(path, "w", encoding="utf-8") as f:
        emb_label = args_dict.get('embedding_label', 'SOAP')
        f.write(f"# Diversity-Aware DFT Nomination Report ({emb_label})\n\n")
        f.write(f"**Generated**: {ts}\n")
        resolved_key = args_dict.get('resolved_embedding_key',
                                     args_dict.get('embedding_key', 'embeddings'))
        f.write(f"**Diversity space**: {emb_label} (key: `{resolved_key}`)\n")
        f.write(f"**Budget**: {args_dict['budget']}\n")
        f.write(f"**Pool size**: {pool_size_actual} (requested {args_dict['pool_size']})\n")
        f.write(f"**Clusters**: {n_clusters} (silhouette = {sil_score:.3f})\n")
        f.write(f"**Seed**: {args_dict['seed']}\n\n")

        f.write("## Strategy Comparison\n\n")
        f.write("| Strategy | Description |\n|---|---|\n")
        for sname, scids in strategy_results.items():
            if sname == "COMBINED":
                continue
            f.write(f"| {sname} | {len(scids)} nominees |\n")
        f.write(f"| **Combined** | {len(combined_cids)} nominees |\n\n")

        # Overlap with old
        f.write("## Old vs New Overlap\n\n")
        f.write(f"- Old nominees: {len(old_nominees)}\n")
        f.write(f"- New nominees (combined): {len(combined_cids)}\n")
        f.write(f"- Overlap: {len(overlap)}\n")
        if overlap:
            f.write(f"- Shared: {', '.join(sorted(overlap))}\n")
        f.write("\n")

        # Diversity metrics
        f.write("## Diversity Metrics (pairwise cosine distance)\n\n")
        f.write("| Set | Embedding | Min | Mean | Max |\n|---|---|---|---|---|\n")
        if diversity_old_stats:
            f.write(f"| Old 25 | {emb_label} | {diversity_old_stats[0]:.4f} | {diversity_old_stats[1]:.4f} | {diversity_old_stats[2]:.4f} |\n")
        if diversity_new_stats:
            f.write(f"| New 25 | {emb_label} | {diversity_new_stats[0]:.4f} | {diversity_new_stats[1]:.4f} | {diversity_new_stats[2]:.4f} |\n")
        if verification_soap_old and emb_label.casefold() != "soap":
            f.write(f"| Old 25 | SOAP | {verification_soap_old[0]:.4f} | {verification_soap_old[1]:.4f} | {verification_soap_old[2]:.4f} |\n")
        if verification_soap_new and emb_label.casefold() != "soap":
            f.write(f"| New 25 | SOAP | {verification_soap_new[0]:.4f} | {verification_soap_new[1]:.4f} | {verification_soap_new[2]:.4f} |\n")
        f.write("\n")

        # Per-strategy lists
        for sname, scids in strategy_results.items():
            if sname == "COMBINED":
                continue
            f.write(f"## {sname}\n\n")
            f.write("| # | CIF ID |\n|---|---|\n")
            for j, cid in enumerate(scids, 1):
                f.write(f"| {j} | {cid} |\n")
            f.write("\n")

        # Combined final
        f.write("## Combined Final 25\n\n")
        f.write("| # | CIF ID | Strategies nominating | Avg RRF rank |\n|---|---|---|---|\n")
        # Build nomination counts
        nom_count = defaultdict(int)
        for sname, scids in strategy_results.items():
            if sname == "COMBINED":
                continue
            for c in scids:
                nom_count[c] += 1
        for j, cid in enumerate(combined_cids, 1):
            unc = unc_data.get(cid, {})
            f.write(f"| {j} | {cid} | {nom_count.get(cid, 0)} | - |\n")
        f.write("\n")

        # Uncertainty profile of nominees
        f.write("## Uncertainty Profile of Combined Nominees\n\n")
        f.write("| CIF ID | Rank Std | Rank Range | NN-ML Disagreement |\n|---|---|---|---|\n")
        for cid in combined_cids:
            u = unc_data.get(cid, {})
            f.write(f"| {cid} | {u.get('rank_std', 0):.1f} | {u.get('rank_range', 0):.0f} | {u.get('nn_ml_disagreement', 0):.0f} |\n")
        f.write("\n")

    print(f"  Report saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SOAP-diversity-aware DFT candidate nomination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--embeddings_path", required=True,
                        help="Diversity descriptors .npz: dense cif_ids + "
                             "<embedding_key>, or custom CSR cif_ids + sp_* keys")
    parser.add_argument("--embedding_key", default="embeddings",
                        help="Key inside the .npz for the embedding matrix "
                             "(default: 'embeddings'; use 'soap_descriptors' for SOAP)")
    parser.add_argument("--embedding_label", default="SOAP",
                        help="Human-readable label for plots and report "
                             "(default: 'SOAP')")
    parser.add_argument("--prediction_csvs", nargs="+", required=True,
                        help="model=path pairs, e.g. exp364=/path/to/inference_predictions.csv")
    parser.add_argument("--soap_embeddings_path", default=None,
                        help="Optional independent SOAP descriptors for legacy "
                             "cross-space verification; not needed when the "
                             "primary diversity matrix is SOAP")
    parser.add_argument("--old_nominees", default=None,
                        help="Path to old FINAL_DFT_TOP25.txt for comparison")
    parser.add_argument("--umap_cache", default=None,
                        help="Path to existing umap_cache.npz (optional, speeds up plotting)")
    parser.add_argument("--output_dir", default="DFT-subset-Nomination-v2",
                        help="Output directory (default: DFT-subset-Nomination-v2)")
    parser.add_argument("--pool_size", type=int, default=200,
                        help="Shortlist pool size (default: 200)")
    parser.add_argument("--pca_components", type=int, default=50,
                        help="PCA dimensions before k-means (default: 50)")
    parser.add_argument("--n_clusters", type=int, default=15,
                        help="Number of k-means clusters (default: 15)")
    parser.add_argument("--kmeans_n_init", type=int, default=10,
                        help="Number of k-means initializations (default: 10)")
    parser.add_argument("--max_per_cluster", type=int, default=2,
                        help="Max picks per cluster in quota strategies (default: 2)")
    parser.add_argument("--mmr_lambdas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                        help="Lambda values for MMR (default: 0.3 0.5 0.7)")
    parser.add_argument("--budget", type=int, default=25,
                        help="Number of structures to nominate (default: 25)")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF smoothing parameter (default: 60)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    # Uncertainty weights for strategy C
    parser.add_argument("--alpha", type=float, default=0.50,
                        help="Weight for quality rank in strategy C (default: 0.50)")
    parser.add_argument("--beta", type=float, default=0.30,
                        help="Weight for uncertainty in strategy C (default: 0.30)")
    parser.add_argument("--gamma", type=float, default=0.20,
                        help="Weight for NN-ML disagreement in strategy C (default: 0.20)")
    # Model type annotations
    parser.add_argument("--nn_models", nargs="+",
                        default=["exp364"],
                        help="Which model names are NN (paper default: exp364)")
    parser.add_argument("--ml_models", nargs="+",
                        default=["smote_extra_trees"],
                        help="Which model names are ML (paper default: smote_extra_trees)")
    # Long-tail exploration
    parser.add_argument("--exploration_budget", type=int, default=5,
                        help="How many of the --budget slots to reserve for long-tail "
                             "exploration picks from OUTSIDE the pool (default: 5). "
                             "Set to 0 to disable.")
    parser.add_argument("--exploration_pool_lo", type=int, default=None,
                        help="Long-tail exploration zone start rank (default: pool_size+1)")
    parser.add_argument("--exploration_pool_hi", type=int, default=1000,
                        help="Long-tail exploration zone end rank (default: 1000)")
    parser.add_argument("--exploration_disagreement_weight", type=float,
                        default=0.60,
                        help="Rank-disagreement weight in exploration relevance "
                             "(default: 0.60)")
    parser.add_argument("--exploration_rank_std_weight", type=float,
                        default=0.40,
                        help="Rank-standard-deviation weight in exploration "
                             "relevance (default: 0.40)")
    parser.add_argument("--exploration_mmr_lambda", type=float, default=0.40,
                        help="Relevance weight for exploration MMR (default: 0.40)")
    args = parser.parse_args()

    if not np.isclose(args.alpha + args.beta + args.gamma, 1.0):
        parser.error("--alpha, --beta, and --gamma must sum to 1.")
    if not np.isclose(args.exploration_disagreement_weight
                      + args.exploration_rank_std_weight, 1.0):
        parser.error("The two exploration relevance weights must sum to 1.")

    np.random.seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load the diversity representation
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("  Diversity-aware DFT nomination")
    print("=" * 70)

    try:
        embedding_cids, full_diversity_matrix, resolved_key, storage_format = (
            load_diversity_matrix(args.embeddings_path, args.embedding_key)
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"  ERROR: could not load diversity descriptors: {exc}")
        return 1
    embedding_id_to_row = {cid: i for i, cid in enumerate(embedding_cids)}
    args.resolved_embedding_key = resolved_key
    print(f"  Loaded {len(embedding_cids)} structures, diversity dim = "
          f"{full_diversity_matrix.shape[1]} ({storage_format})")
    print(f"  Diversity key: '{resolved_key}'  label: {args.embedding_label}")
    print(f"  Example CIF IDs: {embedding_cids[:3]}")

    # Optional independent SOAP verification representation (legacy mode).
    verification_soap_matrix = None
    verification_soap_cids = None
    if args.soap_embeddings_path and os.path.isfile(args.soap_embeddings_path):
        try:
            (verification_soap_cids, verification_soap_matrix,
             soap_key, soap_storage) = load_diversity_matrix(
                 args.soap_embeddings_path, "embeddings"
             )
        except (OSError, ValueError, KeyError) as exc:
            print(f"  ERROR: could not load optional SOAP descriptors: {exc}")
            return 1
        print(f"  Loaded optional SOAP verification descriptors: "
              f"{verification_soap_matrix.shape} ({soap_storage}, key={soap_key})")
    elif args.soap_embeddings_path:
        print(f"  Warning: SOAP file not found: {args.soap_embeddings_path}")

    # Old nominees
    old_nominees = set()
    if args.old_nominees and os.path.isfile(args.old_nominees):
        with open(args.old_nominees, "r", encoding="utf-8") as f:
            old_nominees = set(line.strip() for line in f if line.strip())
        print(f"  Loaded {len(old_nominees)} old nominees for comparison")

    # -------------------------------------------------------------------------
    # 2. Load predictions
    # -------------------------------------------------------------------------
    prediction_models = {}  # {name: {cid: score}}; lower = better
    prediction_errors = []
    for spec in args.prediction_csvs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(spec))[0]
            path = spec
        name = name.strip()
        path = path.strip()
        if not name or not path:
            prediction_errors.append(f"invalid prediction specification: {spec!r}")
            continue
        if name in prediction_models:
            prediction_errors.append(f"duplicate prediction-model name: {name}")
            continue
        path = os.path.abspath(path)
        try:
            preds = load_predictions_from_csv(path, normalize_lower_is_better=True)
        except OSError as exc:
            prediction_errors.append(f"{name}: {exc}")
            continue
        if preds:
            prediction_models[name] = preds
            print(f"  Loaded {name}: {len(preds)} predictions from {os.path.basename(path)}")
        else:
            prediction_errors.append(f"{name}: no valid predictions in {path}")

    if prediction_errors:
        print("ERROR: Every --prediction_csvs input is required and must load:")
        for message in prediction_errors:
            print(f"  - {message}")
        return 1

    nn_names = set(args.nn_models) & set(prediction_models)
    ml_names = set(args.ml_models) & set(prediction_models)
    print(f"  NN models: {sorted(nn_names)}  |  ML models: {sorted(ml_names)}")

    # -------------------------------------------------------------------------
    # DIAGNOSTIC: verify score directions & CID overlap
    # -------------------------------------------------------------------------
    print("\n  === DIAGNOSTICS (verify these!) ===")
    embedding_id_set = set(embedding_cids)
    for mname, mscores in prediction_models.items():
        vals = list(mscores.values())
        n_overlap = len(set(mscores) & embedding_id_set)
        sorted_cids_m = sorted(mscores.keys(), key=lambda c: mscores[c])
        print(f"  [{mname}] {len(vals)} predictions, "
              f"overlap with diversity descriptors: {n_overlap}/{len(embedding_cids)}, "
              f"score range: [{min(vals):.4f}, {max(vals):.4f}], "
              f"median: {np.median(vals):.4f}")
        print(f"    Convention: lower = better (after normalisation)")
        print(f"    Top-5 (best):  {sorted_cids_m[:5]}")
        print(f"    Top-5 scores:  {[f'{mscores[c]:.4f}' for c in sorted_cids_m[:5]]}")
        print(f"    Bot-5 (worst): {sorted_cids_m[-5:]}")
        if n_overlap == 0:
            print(f"    *** WARNING: ZERO overlap between {mname} CIDs and embedding CIDs! ***")
            print(f"    CSV CID examples: {sorted_cids_m[:3]}")
            print(f"    Descriptor CID examples: {embedding_cids[:3]}")
            print(f"    This likely means CID format mismatch — check your files!")
    print("  === END DIAGNOSTICS ===\n")

    # Only IDs scored by every required model are eligible. This is essential
    # when --embeddings_path is the full QMOF SOAP cache: labeled structures in
    # that cache must never enter RRF, the shortlist, exploration, or UMAP.
    candidate_cids = common_prediction_universe(embedding_cids, prediction_models)
    if not candidate_cids:
        print("ERROR: No structure ID is shared by the diversity matrix and every "
              "required prediction CSV.")
        return 1
    excluded = len(embedding_cids) - len(candidate_cids)
    print(f"  Prediction-complete nomination universe: {len(candidate_cids)} "
          f"structures ({excluded} descriptor rows excluded)")
    candidate_rows = [embedding_id_to_row[cid] for cid in candidate_cids]
    diversity_matrix = take_matrix_rows(full_diversity_matrix, candidate_rows)
    del full_diversity_matrix
    diversity_id_to_row = {cid: i for i, cid in enumerate(candidate_cids)}

    # -------------------------------------------------------------------------
    # 3. Ensemble RRF → ranked list
    # -------------------------------------------------------------------------
    rrf_scores = reciprocal_rank_fusion(
        prediction_models, candidate_cids, k=args.rrf_k
    )
    rrf_sorted = sorted(candidate_cids, key=lambda c: rrf_scores[c])
    rrf_rank = {c: i + 1 for i, c in enumerate(rrf_sorted)}
    print(f"  RRF ensemble computed over {len(prediction_models)} models")

    # -------------------------------------------------------------------------
    # 4. Uncertainty signals
    # -------------------------------------------------------------------------
    if len(candidate_cids) < args.budget:
        print(f"ERROR: nomination budget {args.budget} exceeds the "
              f"{len(candidate_cids)} eligible structures.")
        return 1
    if args.pool_size < args.budget:
        print(f"ERROR: --pool_size ({args.pool_size}) must be at least "
              f"--budget ({args.budget}).")
        return 1

    unc_data = compute_uncertainty(
        prediction_models, candidate_cids, nn_names, ml_names
    )

    # -------------------------------------------------------------------------
    # 5. Build shortlist pool (top-N by RRF)
    # -------------------------------------------------------------------------
    pool_size = min(args.pool_size, len(candidate_cids))
    pool_cids = rrf_sorted[:pool_size]
    pool_set = set(pool_cids)
    pool_rows = [diversity_id_to_row[cid] for cid in pool_cids]
    pool_diversity_matrix = take_matrix_rows(diversity_matrix, pool_rows)
    pool_quality = np.array([rrf_scores[c] for c in pool_cids])  # lower = better
    print(f"  Shortlist pool: top {pool_size} by RRF")

    # Normalised quality for MMR (1 = best, 0 = worst)
    pool_quality_norm = 1.0 - normalize_01(pool_quality)

    # -------------------------------------------------------------------------
    # 6. Cluster the pool
    # -------------------------------------------------------------------------
    labels, sil_score = cluster_pool(
        pool_diversity_matrix, args.n_clusters, args.seed,
        pca_components=args.pca_components,
        kmeans_n_init=args.kmeans_n_init)
    n_unique_clusters = len(set(labels))
    print(f"  Clustered into {n_unique_clusters} clusters (silhouette = {sil_score:.3f})")

    # -------------------------------------------------------------------------
    # 7. Pairwise cosine distance for MMR
    # -------------------------------------------------------------------------
    from sklearn.metrics.pairwise import cosine_distances
    dist_matrix = cosine_distances(pool_diversity_matrix)
    # Normalise distances to [0,1] for MMR
    dmax = dist_matrix.max()
    if dmax > 0:
        dist_norm = dist_matrix / dmax
    else:
        dist_norm = dist_matrix

    # -------------------------------------------------------------------------
    # 8. Run selection strategies
    # -------------------------------------------------------------------------
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    strategy_results = {}

    # --- Strategy A: Cluster Quota ---
    print("\n  Strategy A: Cluster-quota round-robin")
    sel_a = strategy_cluster_quota(pool_cids, pool_quality, labels,
                                   args.budget, args.max_per_cluster)
    strategy_results["A_cluster_quota"] = sel_a
    cid2pool = {c: i for i, c in enumerate(pool_cids)}
    sel_a_clusters = set(labels[cid2pool[c]] for c in sel_a if c in cid2pool)
    print(f"    Selected {len(sel_a)} nominees across {len(sel_a_clusters)} clusters")

    # --- Strategy B: MMR (multiple lambdas) ---
    for lam in args.mmr_lambdas:
        sname = f"B_mmr_lambda{lam:.1f}"
        print(f"\n  Strategy B: MMR (lambda={lam})")
        sel_b = strategy_mmr(pool_cids, pool_quality_norm, dist_norm, lam, args.budget)
        strategy_results[sname] = sel_b
        print(f"    Selected {len(sel_b)} nominees")

    # --- Strategy C: Uncertainty-weighted cluster quota ---
    print(f"\n  Strategy C: Uncertainty-weighted cluster quota (alpha={args.alpha}, beta={args.beta}, gamma={args.gamma})")
    pool_unc_std = normalize_01([unc_data[c]["rank_std"] for c in pool_cids])
    pool_unc_dis = normalize_01([unc_data[c]["nn_ml_disagreement"] for c in pool_cids])
    # Combined: lower = better (quality is already lower=better, uncertainty we WANT high → invert)
    pool_combined = (
        args.alpha * normalize_01(pool_quality)                  # quality: lower rank = better
        + args.beta * (1.0 - pool_unc_std)                      # prefer HIGH uncertainty (inverted)
        + args.gamma * (1.0 - pool_unc_dis)                     # prefer HIGH disagreement (inverted)
    )
    sel_c = strategy_uncertainty_quota(pool_cids, pool_combined, labels,
                                       args.budget, args.max_per_cluster)
    strategy_results["C_uncertainty_quota"] = sel_c
    print(f"    Selected {len(sel_c)} nominees")

    # --- Strategy D: Long-tail exploration (hedge against pool being wrong) ---
    exploration_budget = min(args.exploration_budget, args.budget - 1)
    if exploration_budget > 0:
        explore_lo = args.exploration_pool_lo if args.exploration_pool_lo is not None else pool_size
        explore_hi = min(args.exploration_pool_hi, len(rrf_sorted))
        print(f"\n  Strategy D: Long-tail exploration ({exploration_budget} picks from ranks {explore_lo+1}–{explore_hi})")
        # Candidates: structures ranked explore_lo..explore_hi that are NOT in the pool
        tail_cids = [c for c in rrf_sorted[explore_lo:explore_hi] if c not in pool_set]
        if tail_cids:
            # Rank by NN-ML disagreement (highest first = most uncertain)
            tail_disagree = np.array([unc_data[c]["nn_ml_disagreement"] for c in tail_cids])
            tail_unc_std = np.array([unc_data[c]["rank_std"] for c in tail_cids])
            # Both terms are higher-is-more-informative.
            tail_score = (
                args.exploration_disagreement_weight
                * normalize_01(tail_disagree)
                + args.exploration_rank_std_weight
                * normalize_01(tail_unc_std)
            )
            # Also enforce diversity: pick via mini-MMR within tail embeddings
            tail_rows = [diversity_id_to_row[cid] for cid in tail_cids]
            tail_diversity_matrix = take_matrix_rows(diversity_matrix, tail_rows)
            tail_dist = cosine_distances(tail_diversity_matrix)
            td_max = tail_dist.max()
            if td_max > 0:
                tail_dist_norm = tail_dist / td_max
            else:
                tail_dist_norm = tail_dist

            sel_d = strategy_mmr(tail_cids, tail_score, tail_dist_norm,
                                 lam=args.exploration_mmr_lambda,
                                 budget=exploration_budget)
            strategy_results["D_longtail_exploration"] = sel_d
            print(f"    Selected {len(sel_d)} exploration picks from rank {explore_lo+1}–{explore_hi}")
            for j, cid in enumerate(sel_d, 1):
                print(f"      {j}. {cid}  (RRF rank={rrf_rank[cid]}, "
                      f"disagree={unc_data[cid]['nn_ml_disagreement']:.0f}, "
                      f"rank_std={unc_data[cid]['rank_std']:.0f})")
        else:
            print("    Warning: no tail candidates available — skipping.")
            exploration_budget = 0

    # -------------------------------------------------------------------------
    # 9. Combined "best-of-all" list
    # -------------------------------------------------------------------------
    print("\n  Combining strategies...")
    nom_count = defaultdict(int)
    nom_quality = {}
    # Count nominations from pool-based strategies (A, B, C) only
    pool_strategies = {k: v for k, v in strategy_results.items() if not k.startswith("D_")}
    for sname, scids in pool_strategies.items():
        for c in scids:
            nom_count[c] += 1
            if c not in nom_quality:
                nom_quality[c] = rrf_rank[c]

    # Reserve slots for long-tail exploration picks (guaranteed inclusion)
    exploration_picks = strategy_results.get("D_longtail_exploration", [])
    reserved = set(exploration_picks)
    main_budget = args.budget - len(exploration_picks)

    # Fill main slots from pool-based strategies (excluding exploration picks)
    pool_nominated = sorted(
        [c for c in nom_count if c not in reserved],
        key=lambda c: (-nom_count[c], nom_quality[c])
    )
    main_picks = pool_nominated[:main_budget]
    combined_cids = main_picks + list(exploration_picks)

    print(f"  Combined: {len(main_picks)} pool picks + {len(exploration_picks)} exploration picks = {len(combined_cids)}")
    if main_picks:
        print(f"  Pool pick nomination counts: max={max(nom_count[c] for c in main_picks)}, "
              f"min={min(nom_count[c] for c in main_picks)}")

    # -------------------------------------------------------------------------
    # 10. Save outputs
    # -------------------------------------------------------------------------
    print("\n  Saving outputs...")
    for sname, scids in strategy_results.items():
        fname = f"{sname}_top{args.budget}.txt"
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            for c in scids:
                f.write(c + "\n")

    # Add COMBINED after writing per-strategy files (avoids double-counting in nom_count)
    strategy_results["COMBINED"] = combined_cids
    with open(os.path.join(output_dir, f"COMBINED_top{args.budget}.txt"), "w", encoding="utf-8") as f:
        for c in combined_cids:
            f.write(c + "\n")

    # Final CSV
    csv_path = os.path.join(output_dir, "FINAL_TOP25_diverse.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "cif_id", "rrf_rank", "strategies_nominating",
                         "rank_std", "rank_range", "nn_ml_disagreement", "cluster"])
        for j, cid in enumerate(combined_cids, 1):
            u = unc_data.get(cid, {})
            cl = labels[pool_cids.index(cid)] if cid in pool_set else -1
            writer.writerow([j, cid, rrf_rank.get(cid, -1), nom_count.get(cid, 0),
                             f"{u.get('rank_std', 0):.1f}",
                             f"{u.get('rank_range', 0):.0f}",
                             f"{u.get('nn_ml_disagreement', 0):.0f}",
                             cl])

    # Simple text list
    txt_path = os.path.join(output_dir, "FINAL_TOP25_diverse.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for c in combined_cids:
            f.write(c + "\n")

    # Shortlist pool CSV
    pool_csv = os.path.join(output_dir, "shortlist_pool.csv")
    with open(pool_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pool_rank", "cif_id", "rrf_rank", "cluster",
                         "rank_std", "nn_ml_disagreement"])
        for j, cid in enumerate(pool_cids, 1):
            u = unc_data.get(cid, {})
            writer.writerow([j, cid, rrf_rank.get(cid, -1), labels[j - 1],
                             f"{u.get('rank_std', 0):.1f}",
                             f"{u.get('nn_ml_disagreement', 0):.0f}"])

    print(f"  Saved {csv_path}")
    print(f"  Saved {txt_path}")

    # -------------------------------------------------------------------------
    # 11. Diversity verification
    # -------------------------------------------------------------------------
    print("\n  Computing diversity metrics...")

    # Pairwise distances in the nomination diversity representation.
    def diversity_stats(cids_set):
        idx = [diversity_id_to_row[c] for c in cids_set
               if c in diversity_id_to_row]
        if len(idx) < 2:
            return None
        sub = take_matrix_rows(diversity_matrix, idx)
        D = cosine_distances(sub)
        np.fill_diagonal(D, np.nan)
        vals = D[~np.isnan(D)]
        return (float(np.min(vals)), float(np.mean(vals)), float(np.max(vals)))

    diversity_old_stats = diversity_stats(old_nominees) if old_nominees else None
    diversity_new_stats = diversity_stats(combined_cids)

    if diversity_old_stats:
        print(f"  {args.embedding_label} distances — Old: min={diversity_old_stats[0]:.4f} mean={diversity_old_stats[1]:.4f} max={diversity_old_stats[2]:.4f}")
    if diversity_new_stats:
        print(f"  {args.embedding_label} distances — New: min={diversity_new_stats[0]:.4f} mean={diversity_new_stats[1]:.4f} max={diversity_new_stats[2]:.4f}")

    # Optional independent SOAP verification for legacy non-SOAP runs.
    verification_soap_old_stats = None
    verification_soap_new_stats = None
    if verification_soap_matrix is not None and verification_soap_cids is not None:
        mn, me, mx, _ = soap_diversity(
            verification_soap_matrix, verification_soap_cids, list(old_nominees)
        )
        verification_soap_old_stats = (mn, me, mx) if mn is not None else None
        mn, me, mx, _ = soap_diversity(
            verification_soap_matrix, verification_soap_cids, combined_cids
        )
        verification_soap_new_stats = (mn, me, mx) if mn is not None else None
        if verification_soap_old_stats:
            print(f"  SOAP verification distances — Old: min={verification_soap_old_stats[0]:.4f} mean={verification_soap_old_stats[1]:.4f} max={verification_soap_old_stats[2]:.4f}")
        if verification_soap_new_stats:
            print(f"  SOAP verification distances — New: min={verification_soap_new_stats[0]:.4f} mean={verification_soap_new_stats[1]:.4f} max={verification_soap_new_stats[2]:.4f}")

    # -------------------------------------------------------------------------
    # 12. UMAP plots
    # -------------------------------------------------------------------------
    plot_umap(diversity_matrix, candidate_cids, pool_set,
              {k: v for k, v in strategy_results.items() if k == "COMBINED"},
              old_nominees, output_dir,
              umap_cache_path=args.umap_cache, seed=args.seed,
              emb_label=args.embedding_label)

    # -------------------------------------------------------------------------
    # 13. Write full report
    # -------------------------------------------------------------------------
    write_report(output_dir, vars(args), pool_size, n_unique_clusters, sil_score,
                 strategy_results, combined_cids, old_nominees,
                 verification_soap_old_stats, verification_soap_new_stats,
                 diversity_old_stats, diversity_new_stats,
                 {}, pool_cids, unc_data)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DONE — Diversity-aware nomination complete")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print(f"  Combined top-{args.budget}: {txt_path}")
    overlap = set(combined_cids) & set(old_nominees)
    if old_nominees:
        print(f"  Overlap with old nominees: {len(overlap)}/{len(old_nominees)}")
    if diversity_new_stats and diversity_old_stats:
        delta = diversity_new_stats[1] - diversity_old_stats[1]
        print(f"  {args.embedding_label} mean distance: {diversity_old_stats[1]:.4f} (old) → {diversity_new_stats[1]:.4f} (new)  "
              f"({'↑' if delta > 0 else '↓'} {abs(delta):.4f})")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
