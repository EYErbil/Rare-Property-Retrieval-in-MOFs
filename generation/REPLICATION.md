# Replication guide — generation, screening, and nomination (Steps 1–12)

Step-by-step manual for reproducing the generated-candidate arm: reference-database analysis,
PORMAKE candidate construction, MOFTransformer preprocessing, chemical-space analyses,
inference with the trained models, and diversity-aware nomination of the 25 DFT candidates.
Continue with [DFT_WORKFLOW.md](DFT_WORKFLOW.md) (Steps 13–26) after nomination; common
failure modes are collected in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

Read sections **in order**. Replace every `REPO_ROOT` with the absolute path to the
**`generation/` module** of your clone (the folder that contains `bulk_pormake_generation/`,
`scripts/`, `nominate_diverse_dft.py`, etc.).

---

## Table of contents

1. [What this pipeline produces](#what-this-pipeline-produces)
2. [Prerequisites (hardware, data, software)](#prerequisites-hardware-data-software)
3. [Path and convention](#path-and-convention)
4. [Replication workflow — follow these steps in order](#replication-workflow--follow-these-steps-in-order)
5. [Appendix A — `.npz` keys (`embedding_key`)](#appendix-a--npz-keys-embedding_key)
6. Post-nomination DFT workflow → [DFT_WORKFLOW.md](DFT_WORKFLOW.md)
7. Optional Slurm wrappers and troubleshooting → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## What this pipeline produces

| Stage | Main artifacts |
|-------|----------------|
| QMOF-aligned PORMAKE libraries | `qmof_bb_dir/`, `qmof_topo_dir/`, `qmof_analysis/` |
| Generated CIFs | `generated_cifs/small_30A_200atom/*.cif` (and paired `large/` tree) |
| MOFTransformer-ready generated data | `generated_cifs/PMtransformer_Files/` (`test/` + JSONs) |
| SOAP vs QMOF | `soap_umap_generated_vs_qmof.*`, `qmof_soap_descriptors.npz`, `generated_soap_descriptors.npz`, summaries |
| SOAP split UMAP | `soap_umap_generated_vs_qmof_splits.*`, split summary JSON |
| PMTransformer vs QMOF | `pmt_umap_generated_vs_qmof.*`, `qmof_pmt_embeddings.npz`, `generated_pmt_embeddings.npz`, `pmt_comparison_summary.json` |
| NN inference (trained regressor) | `inference_predictions.csv` |
| ML inference on embeddings | `<method>/test_predictions.csv` |
| DFT nomination | `FINAL_TOP25_diverse.txt`, `COMBINED_top25.txt`, `diversity_report.md`, UMAP plots |
| Post-nomination VASP prep | selected CIFs, POSCAR folders, `conversion_report.csv`, validation CSVs, `KPOINTS`, `PBED3-PreRelax` / `PBED3-Relax` / `PBED3-Single` / `HSE-single` folders with stage `INCAR` and MAGMOM-ready inputs |

---

## Prerequisites (hardware, data, software)

### Hardware

- **CIF generation**: CPU; use batched build if you hit OOM.
- **SOAP + UMAP**: often **RAM-bound** (64G+ typical for large sets); GPU rarely helps.
- **PMTransformer forward passes / NN inference**: **GPU strongly recommended** (CUDA).
- **DFT preparation / VASP execution**: POSCAR/KPOINTS/MAGMOM preparation is CPU-light; actual VASP runs require your cluster's VASP, POTCAR, MPI, and scheduler setup.

### Data you must obtain yourself

1. **`qmof.csv`** — QMOF database release table, read by `scripts/analyze_reference_db.py` (see its
   docstring). Download the release table from the
   [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF) and place it at
   `REPO_ROOT/qmof.csv`; it is not committed or included in Globus.

2. **QMOF CIFs for SOAP** — A directory of `.cif` files for structures you treat as the QMOF reference set (subset or full). Used only when computing SOAP from CIFs (`--qmof-cif-dir`). Same basename conventions as your workflow expects.

3. **Train/validation/test partition JSONs** — For split-colored SOAP/PMT figures and nomination alignment, directories containing:

   `train_bandgaps_regression.json`, `val_bandgaps_regression.json`, `test_bandgaps_regression.json`

   (paths passed as `--labeled-splits-dir` / used by analysis scripts as documented).
   **The paper's exact split is committed in the screening module** at
   [`../screening/data/splits/strategy_d_farthest_point/`](../screening/data/splits/strategy_d_farthest_point/) — point `--labeled-splits-dir` there.

4. **Trained NN experiment** — Checkpoint(s) under `TRAIN_ROOT/experiments/<exp_name>/` where `TRAIN_ROOT` is a directory that contains **`train_regressor.py`** next to `experiments/` — normally the [`../screening/`](../screening/) module after its Step 2 has been run. Used by `scripts/re_inference/run_inference.py` via `--base_dir`.

5. **Trained ML embedding classifiers** — Directory tree with one subfolder per method (`extra_trees/`, `random_forest/`, …), each containing `model.joblib` and/or `artifacts.joblib`. You pass this tree as **`--clf_dir`** to `scripts/re_inference/reinfer_ml.py` (see Step 10); its built-in default `--clf_dir` is only a placeholder, so supply your own path.

6. **Representation `.npz` files** — PMTransformer embeddings (`cif_ids` + 768-d
   `embeddings`) are used for ML scoring and representation comparisons. SOAP is the sole
   structural-diversity matrix for the generated-pool nomination; the second-phase QMOF
   acquisition ran the same procedure separately in PMTransformer-embedding and SOAP spaces, so
   both matrices are diversity inputs there. The generated PMTransformer cache is released
   on Globus as `embeddings/generated_pmtransformer_embeddings.npz` (13,802 rows, identifiers and
   row order identical to the generated SOAP archive); it can alternatively be regenerated in
   Step 7.

7. **DFT runtime inputs** — For the post-nomination VASP workflow you must supply any licensed or site-specific files yourself, especially **`POTCAR`** files and cluster job scripts/modules. This repo provides helper templates and preparation scripts, but it does **not** distribute POTCARs or run VASP.

### Software environment

From repo root, typical cluster setup:

```bash
module purge
module load cuda/12.3 cudnn/8.9.5/cuda-12.x python/3.9.5
source /path/to/your/venv/bin/activate
```

Install Python packages as needed for each stage:

| Stage | Packages |
|-------|----------|
| `analyze_reference_db.py` | `pandas` |
| PORMAKE generation | `pormake` (see upstream repo) |
| MOFTransformer prep + PMTransformer | `moftransformer`, GRIDAY (`moftransformer install-griday`) |
| SOAP scripts | `numpy`, `scikit-learn`, `umap-learn`, `matplotlib`, ASE/`dscribe` as required by your SOAP stack |
| NN inference | `torch`, `pytorch_lightning`, `moftransformer` |
| ML inference | `numpy`, `joblib`, `scikit-learn` |
| Nomination | `numpy`, `scikit-learn`, `matplotlib`, `umap-learn` |
| Post-nomination DFT prep | `pymatgen`, `numpy`; shell wrappers assume Bash + Slurm + VASP on a cluster |

---

## Path and convention

- **`REPO_ROOT`** — the `generation/` module directory (contains `README.md`, `scripts/`, `nominate_diverse_dft.py`).
- **`TRAIN_ROOT`** — Directory that contains **`train_regressor.py`** and **`experiments/`** (passed as `--base_dir` to NN inference) — normally the `screening/` module of this repository.
- **`DFT_WORK_ROOT`** — Your scratch/work directory for VASP folders after nomination. A typical layout is one MOF folder per nominee, with stage subfolders such as `PBED3-PreRelax`, `PBED3-Relax`, `PBED3-Single`, and `HSE-single`.
- **Working directory**: Run all `python …` commands from **`REPO_ROOT`** unless a script docstring says otherwise (so relative paths like `data/…`, `qmof_bb_dir/` resolve correctly).

---

## Replication workflow — follow these steps in order

### Step 1 — QMOF-aligned topology and building-block libraries

Requires `REPO_ROOT/qmof.csv` in place.

```bash
cd REPO_ROOT
python scripts/analyze_reference_db.py
python scripts/build_custom_dirs.py
python scripts/verify_custom_dirs.py
```

**Outputs:**

- `qmof_analysis/` (counts, selected topology/metal/linker lists)
- `qmof_topo_dir/` (`.cgd` topologies)
- `qmof_bb_dir/` (filtered `.xyz` building blocks)

---

### Step 2 — Generate candidate strings

**Prerequisite — RMSD node/topology compatibility table.** `make_candidates.py` draws nodes from a
precomputed pickle that lists, per topology, the building-block nodes whose connection points fit
within an RMSD tolerance (≤ 0.3). Build it once with `rmsd_calculated_node.py` (wrapped by the
top-level `build_rmsd_table.sh`), which writes `data/rmsd_qmof.pickle`.

```bash
cd REPO_ROOT
python bulk_pormake_generation/make_candidates.py \
  -n 20000 \
  --max-n-atoms 200 \
  --has-metal True \
  --pre-defined-list data/rmsd_qmof.pickle \
  --bb-dir qmof_bb_dir \
  --topo-dir qmof_topo_dir \
  --save data/candidates_qmof_13k_200atom.txt
```

> `-n` is the number of **candidate strings** drawn, not the number of structures that survive.
> Candidate sampling is stochastic (not seeded) and the build stage (Step 3) discards
> candidates that fail assembly or the cell-size filters, so 20,000 candidates reduced to the
> paper's 13,802 successfully built, screenable structures. Exact counts will vary between
> runs. This random-generation route constructs a new pool; it does not claim to recreate the
> exact paper pool. The paper-pool identities and descriptor-row order are recoverable from the
> released generated SOAP archive as shown below. The full paper-pool CIF database is not
> distributed.

> The output flag is `-s` / `--save` (there is **no** `--output`). `--has-metal` defaults to `True`;
> due to an argparse quirk (`type=bool`) passing `--has-metal False` does *not* disable the metal
> requirement, so leave it at the default.

**Output:** `data/candidates_qmof_13k_200atom.txt`

#### Exact paper-pool identity manifest

To materialize the exact 13,802 paper-pool IDs, preserving their order in the released SOAP
descriptor matrix, run:

```bash
python scripts/materialize_generated_pool_manifest.py \
  --descriptor-npz soap_analysis/generated_vs_qmof/generated_soap_descriptors.npz \
  --output data/paper_generated_pool_manifest.txt
```

This mode requires no generated-pool CSV. It validates the expected count and uniqueness and prints
a SHA-256 checksum. The manifest fixes the identity and row order of the paper pool; it does not
reconstruct undistributed CIF coordinate files. Use the stochastic route above to build a new pool.

---

### Step 3 — Build generated CIFs (OOM-safe)

```bash
cd REPO_ROOT
python scripts/build_materials_batched.py \
  --candidates data/candidates_qmof_13k_200atom.txt \
  --bb-dir qmof_bb_dir \
  --topo-dir qmof_topo_dir \
  --save-dir generated_cifs/small_30A_200atom \
  --large-dir generated_cifs/large_30A_200atom \
  --cutoff 30.0 \
  --chunk-size 200
```

**Outputs:**

- `generated_cifs/small_30A_200atom/*.cif`
- `generated_cifs/large_30A_200atom/*.cif` (structures routed to large cell)

Cell-size handling (in `build_materials.py`): structures whose **smallest** lattice length is below a
hardcoded **4.5 Å** are skipped, and the rest are routed to `--save-dir` when their **largest** lattice
length is `< --cutoff` (here 30 Å), otherwise to `--large-dir`. Resume: existing names are skipped
unless you force rebuild (see `build_materials_batched.py --help`).

---

### Step 4 — Build MOFTransformer-ready dataset from generated CIFs

This produces grid/graphdata and split folders used by PMTransformer **embedding extraction** and by **NN inference** on the generated set.

```bash
cd REPO_ROOT
python scripts/prepare_moftransformer_test_only.py \
  --cif-dir REPO_ROOT/generated_cifs/small_30A_200atom \
  --output-dataset-dir REPO_ROOT/generated_cifs/PMtransformer_Files \
  --downstream bandgaps \
  --default-target-value 0.0 \
  --overwrite-raw-json
```

**Outputs (under `--output-dataset-dir`):**

- `total/` — preprocessed tensors + copied `.cif`
- `train/`, `val/`, `test/` — split layout (`test/` holds structures for test-only routing)
- `train_bandgaps.json`, `val_bandgaps.json`, `test_bandgaps.json`

For downstream scripts, align **`--downstream`** with this preparation (`bandgaps` here ⇒ pass **`--downstream bandgaps`** to PMTransformer compare / NN inference when those scripts default to `bandgaps_regression`).

---

### Step 5 — SOAP comparison vs QMOF (figures + caches)

Script: `scripts/soap_analysis/compare_generated_vs_qmof.py`

You must supply **either** cached SOAP `.npz` **or** a CIF directory for each side (QMOF and generated). See `--help`. Typical modes:

#### 5A — Cold start (compute SOAP from CIF directories)

Writes `qmof_soap_descriptors.npz` and `generated_soap_descriptors.npz` **inside `--output_dir`**.

```bash
cd REPO_ROOT
python scripts/soap_analysis/compare_generated_vs_qmof.py \
  --qmof-cif-dir /path/to/qmof/cifs \
  --generated-cif-dir REPO_ROOT/generated_cifs/small_30A_200atom \
  --output_dir REPO_ROOT/soap_analysis/generated_vs_qmof \
  --save_umap_cache
```

#### 5B — Cached QMOF SOAP + generated from CIF (common after first run)

```bash
cd REPO_ROOT
python scripts/soap_analysis/compare_generated_vs_qmof.py \
  --qmof-cache REPO_ROOT/soap_analysis/generated_vs_qmof/qmof_soap_descriptors.npz \
  --generated-cif-dir REPO_ROOT/generated_cifs/small_30A_200atom \
  --output_dir REPO_ROOT/soap_analysis/generated_vs_qmof \
  --save_umap_cache
```

#### 5C — Both sides cached (fastest)

```bash
cd REPO_ROOT
python scripts/soap_analysis/compare_generated_vs_qmof.py \
  --qmof-cache REPO_ROOT/soap_analysis/generated_vs_qmof/qmof_soap_descriptors.npz \
  --generated-cache REPO_ROOT/soap_analysis/generated_vs_qmof/generated_soap_descriptors.npz \
  --output_dir REPO_ROOT/soap_analysis/generated_vs_qmof \
  --save_umap_cache
```

**Key outputs:**

- `soap_umap_generated_vs_qmof.(png|svg|pdf)`
- `soap_umap_density_generated_vs_qmof.(png|svg|pdf)`
- `soap_comparison_summary.json`
- `qmof_soap_descriptors.npz` / `generated_soap_descriptors.npz` (under `--output_dir` when computed)
- Optional `soap_umap_cache.npz` if `--save_umap_cache`

**Note:** Very large SOAP vectors may trigger automatic PCA before UMAP unless `--full-soap-umap` (RAM-heavy). Tunables: `--pca-dim`, `--pca-auto-dim`, `--pca-batch`.

---

### Step 6 — SOAP split-colored UMAP (QMOF train/val/test vs all generated)

Script: `scripts/soap_analysis/compare_generated_vs_qmof_splits.py`

Requires **`--labeled-splits-dir`** with `{train,val,test}_bandgaps_regression.json`. Prefer **cached** `--qmof-cache` + `--generated-cache` on clusters (full SOAP for both sides is memory-heavy).

```bash
cd REPO_ROOT
python scripts/soap_analysis/compare_generated_vs_qmof_splits.py \
  --qmof-cache REPO_ROOT/soap_analysis/generated_vs_qmof/qmof_soap_descriptors.npz \
  --generated-cache REPO_ROOT/soap_analysis/generated_vs_qmof/generated_soap_descriptors.npz \
  --labeled-splits-dir /path/to/new_splits/strategy_d_farthest_point \
  --output_dir REPO_ROOT/soap_analysis/generated_vs_qmof_splits
```

**Outputs:**

- `soap_umap_generated_vs_qmof_splits.(png|svg|pdf)`
- `soap_umap_cache_generated_vs_qmof_splits.npz`
- `soap_generated_vs_qmof_splits_summary.json`

---

### Step 7 — PMTransformer embedding-space comparison (QMOF vs generated)

Script: `scripts/pmtransformer_analysis/compare_generated_vs_qmof.py`

This extracts **768-dim CLS embeddings** with the PMTransformer backbone (`load_path=pmtransformer`) and fits **one** UMAP on **all QMOF + generated**. Optional **`--labeled-splits-dir`** adds split-colored panels using the **same** 2D coordinates (no second UMAP).

**Important:**

- Use **`--generated-cache`** if `generated_pmt_embeddings.npz` already exists to **avoid** redoing the generated forward pass.
- Match **`--downstream`** to how you built `PMtransformer_Files` (Step 4). If you used `--downstream bandgaps`, pass **`--downstream bandgaps`** here (the argparse default in code is `bandgaps_regression`).

```bash
cd REPO_ROOT
python scripts/pmtransformer_analysis/compare_generated_vs_qmof.py \
  --qmof-cache REPO_ROOT/embeddings/pmt_embeddings_qmof_all.npz \
  --generated-data-dir REPO_ROOT/generated_cifs/PMtransformer_Files \
  --generated-split test \
  --downstream bandgaps \
  --labeled-splits-dir /path/to/new_splits/strategy_d_farthest_point \
  --output_dir REPO_ROOT/PmTransformer_analysis/results \
  --save_umap_cache
```

After the first successful run, rerun with caches only:

```bash
cd REPO_ROOT
python scripts/pmtransformer_analysis/compare_generated_vs_qmof.py \
  --qmof-cache REPO_ROOT/embeddings/pmt_embeddings_qmof_all.npz \
  --generated-cache REPO_ROOT/PmTransformer_analysis/results/generated_pmt_embeddings.npz \
  --downstream bandgaps \
  --labeled-splits-dir /path/to/new_splits/strategy_d_farthest_point \
  --output_dir REPO_ROOT/PmTransformer_analysis/results \
  --save_umap_cache
```

**Outputs:**

- `qmof_pmt_embeddings.npz`, `generated_pmt_embeddings.npz` (when extracted from dirs; unchanged if only caches passed)
- `pmt_umap_generated_vs_qmof.(png|svg|pdf)`, `pmt_umap_density_generated_vs_qmof.(png|svg|pdf)`
- If `--labeled-splits-dir`: `pmt_umap_generated_vs_qmof_splits.(png|svg|pdf)`, density variant
- `pmt_comparison_summary.json`
- Optional: `pmt_umap_cache.npz`, `pmt_umap_cache_splits.npz` with `--save_umap_cache`

**NumPy:** On older NumPy, if you see `TypeError: asarray() got an unexpected keyword argument 'copy'`, upgrade NumPy or use the current script from this repo (fixed).

---

### Step 8 — Unified PMTransformer embeddings for **all** structures (optional but recommended for aligned spaces)

If you need **one** `.npz` covering labeled + unlabeled MOFs in the same embedding space, you can adapt the reference implementation:

- [`scripts/extract_all_embeddings_unified.py`](scripts/extract_all_embeddings_unified.py)

This single-forward-pass extractor produced the aligned `pmt_embeddings_qmof_all.npz` reference
cache (distributed via Globus). Copy it next to your `train_regressor` project or adjust `sys.path` as in
your cluster workflow. See the script docstring for full CLI.

---

### Step 9 — NN inference with **your trained regressor** (band-gap predictions CSV)

Script: `scripts/re_inference/run_inference.py`

**Requirements:**

- **`--base_dir`** = directory containing **`train_regressor.py`** and **`experiments/`** (e.g. `TRAIN_ROOT`).
- **`--data_dir`** = MOFTransformer-ready folder with populated **`test/`** and a test JSON (`test_bandgaps.json` or `test_bandgaps_regression.json`; script supports `--downstream auto`).
- **`--output_dir`** = where you want `inference_predictions.csv`.

```bash
cd REPO_ROOT
python scripts/re_inference/run_inference.py \
  --base_dir TRAIN_ROOT \
  --data_dir REPO_ROOT/generated_cifs/PMtransformer_Files \
  --experiments YOUR_EXPERIMENT_NAME \
  --downstream auto \
  --output_dir REPO_ROOT/re_infer/nn/YOUR_EXPERIMENT_NAME
```

**Outputs:**

- `inference_predictions.csv` (columns: `cif_id`, `score`, `predicted_binary`, `true_label`, `mode`)
- `inference_ranked.csv`
- `topK_for_DFT.txt`, `topK_for_DFT.csv` (`--top_k`, default 25)

---

### Step 10 — ML re-inference on an embeddings `.npz`

Script: `scripts/re_inference/reinfer_ml.py`

**Inputs:**

- **`cif_ids`** + **`embeddings`** in one `.npz` file (same format as `predict_with_embedding_classifier.py`).
- Provide **either**:
  - **`--embeddings_path`** — path to that `.npz` (recommended), **or**
  - **`--npz_dir`** — a directory where the script falls back to `<npz_dir>/pmt_embeddings_qmof_unlabeled.npz` (legacy `Phase6_embeddings.npz` also accepted; run `python scripts/re_inference/reinfer_ml.py --help` for the exact lookup).
- **`--clf_dir`** — parent directory of per-method folders (`extra_trees/`, …). Its built-in default is only a non-functional placeholder, so in practice **always pass `--clf_dir`** pointing at your embedding-classifier root.

```bash
cd REPO_ROOT
python scripts/re_inference/reinfer_ml.py \
  --embeddings_path REPO_ROOT/PmTransformer_analysis/results/generated_pmt_embeddings.npz \
  --clf_dir /path/to/embedding_classifiers/strategy_d_farthest_point \
  --output_dir REPO_ROOT/re_infer/ml \
  --methods smote_extra_trees
```

**Alternative (directory-only embeddings discovery):**

```bash
cd REPO_ROOT
python scripts/re_inference/reinfer_ml.py \
  --npz_dir /path/to/dir_containing_embeddings_npz \
  --clf_dir /path/to/embedding_classifiers/strategy_d_farthest_point \
  --output_dir REPO_ROOT/re_infer/ml
```

**Outputs:**

- `REPO_ROOT/re_infer/ml/<method>/test_predictions.csv`
- no validation-metric JSON is synthesized during re-inference; training-time
  `final_results.json` files remain with the trained artifacts

If `--output_dir` is omitted, files are written **in place** under each method folder in `--clf_dir`.

---

### Step 11 — Optional: plot NN score distribution

```bash
cd REPO_ROOT
python scripts/re_inference/plot_inference_bandgap_distribution.py \
  --csv REPO_ROOT/re_infer/nn/YOUR_EXPERIMENT_NAME/inference_predictions.csv \
  --out REPO_ROOT/re_infer/nn/YOUR_EXPERIMENT_NAME/inference_bandgap_distribution.png
```

---

### Step 12 — Diversity-aware DFT nomination

Canonical entrypoint: **`REPO_ROOT/scripts/12_nominate_paper_candidates.sh`**; it calls
`REPO_ROOT/nominate_diverse_dft.py` with every paper parameter explicitly pinned.

> **Two copies exist by design.** Each module carries its own copy so that each is
> self-contained: this copy was used for the **generated-MOF pool**, the screening module's
> [`discovery/nominate_diverse_dft.py`](../screening/discovery/nominate_diverse_dft.py)
> for the **unlabelled QMOF pool**. Both expose the same CLI and strategy set — cluster quota,
> MMR, uncertainty quota, long-tail exploration — differing only in pool-specific reporting.
> For the paper, `--embeddings_path` always points to SOAP descriptors and
> `--embedding_key soap_descriptors`: for the generated pool, SOAP is the sole structural-diversity
> coordinate in both the main and exploration tiers. (The second-phase QMOF acquisition in the
> screening module additionally ran this procedure in PMTransformer-embedding space.) RRF and
> NN–ML disagreement are priority scores, not geometry.

Run from **`REPO_ROOT`** so imports resolve predictably:

```bash
cd REPO_ROOT
```

Run the paper nomination **once, in SOAP space**. PMTransformer embeddings are not a nomination
diversity alternative in the paper workflow.

#### Canonical paper run — SOAP diversity for both tiers

```bash
bash scripts/12_nominate_paper_candidates.sh

# Equivalent explicit invocation:
python nominate_diverse_dft.py \
  --embeddings_path REPO_ROOT/soap_analysis/generated_vs_qmof/generated_soap_descriptors.npz \
  --embedding_key soap_descriptors \
  --embedding_label SOAP \
  --prediction_csvs \
    exp364=REPO_ROOT/re_infer/nn/exp364/inference_predictions.csv \
    smote_extra_trees=REPO_ROOT/re_infer/ml/smote_extra_trees/test_predictions.csv \
  --nn_models exp364 \
  --ml_models smote_extra_trees \
  --output_dir REPO_ROOT/paper_results/nomination-SOAP \
  --pool_size 500 \
  --pca_components 50 \
  --n_clusters 20 \
  --kmeans_n_init 10 \
  --max_per_cluster 1 \
  --mmr_lambdas 0.2 0.3 0.4 \
  --alpha 0.50 \
  --beta 0.30 \
  --gamma 0.20 \
  --budget 25 \
  --exploration_budget 5 \
  --exploration_pool_lo 500 \
  --exploration_pool_hi 2000 \
  --exploration_disagreement_weight 0.60 \
  --exploration_rank_std_weight 0.40 \
  --exploration_mmr_lambda 0.40 \
  --rrf_k 60 \
  --seed 42
```

**Outputs (`REPO_ROOT/paper_results/nomination-SOAP`):**

- `FINAL_TOP25_diverse.txt`, `FINAL_TOP25_diverse.csv` (columns: `rank, cif_id, rrf_rank, strategies_nominating, rank_std, rank_range, nn_ml_disagreement, cluster`)
- `COMBINED_top25.txt` and one file per strategy — `A_cluster_quota_top25.txt`, `B_mmr_lambda{λ}_top25.txt` (one per `--mmr_lambdas` value), `C_uncertainty_quota_top25.txt`, `D_longtail_exploration_top25.txt`
- `shortlist_pool.csv` (the clustered top-`pool_size` shortlist)
- `diversity_report.md`
- `plots/umap_diverse_nominees.png`, `plots/umap_comparison_old_vs_new.png`, `plots/umap_cache.npz`

> Additional optional flags (defaults shown in the runs above): `--alpha 0.5 --beta 0.3 --gamma 0.2`
> weight strategy C's quality / diversity / disagreement terms; `--exploration_pool_lo` (defaults to
> `--pool_size`) and `--exploration_pool_hi` bound the long-tail pool; `--old_nominees` and
> `--umap_cache` enable the old-vs-new comparison plot and faster re-runs.
> In every strategy, cluster membership, MMR distance, diversity weighting, and exploration-tier
> spread come from the same SOAP matrix. The prediction CSVs supply RRF/disagreement priorities
> only.

**Next:** carry the nominated CIFs into the four-stage VASP cascade —
[DFT_WORKFLOW.md](DFT_WORKFLOW.md) (Steps 13–26).


---

## Appendix A — `.npz` keys (`embedding_key`)

| File role | Typical filename | Matrix key for `--embedding_key` |
|-----------|------------------|-----------------------------------|
| PMTransformer / unified embeddings (ML scoring and representation analysis, not paper nomination geometry) | `generated_pmt_embeddings.npz`, `pmt_embeddings_qmof_all.npz` | `embeddings` |
| SOAP descriptors (sole paper nomination geometry) | `*_soap_descriptors.npz` from `compare_generated_vs_qmof.py` | `soap_descriptors` |

Always verify keys:

`python -c "import numpy as np; d=np.load('file.npz'); print(d.files)"`
