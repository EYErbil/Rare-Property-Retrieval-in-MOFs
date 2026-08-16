# Data Directory

This directory should contain the MOF datasets used by the pipeline.
The MOF structure files are **not included** in this repository due to their size.

**Included: the paper's exact partitions.** The final train/validation/test membership lists used in the
manuscript are committed at `splits/strategy_d_farthest_point/{train,val,test}_bandgaps_regression.json`
(1,136 / 524 / 9,150 structures; 60 / 5 / 9 low-band-gap positives). These JSONs are the ground
truth for reproducing the paper. Restore the labeled Globus archive to
`embeddings/embeddings_pretrained.npz`; the default Step 1 verifies the committed JSONs against its
recorded split labels and never overwrites them. The structure files
(`.grid`/`.griddata16`/`.graphdata`) still have to be prepared (Step 0) and placed/symlinked per
split as described below. Use `SPLIT_MODE=fresh` only for a new dataset or target; it writes to a
separate `noncanonical/<run-id>/` tree.

Partition membership was designed before any task-specific training from distances in the frozen
pretrained PMTransformer embedding space. The `data/raw/test/` name used below is merely a
MOFTransformer input-layout convention for an unsplit pool; it is not the paper's labeled test
partition.

## Expected Structure

After Step 0 (preprocessing) and Step 1 (canonical materialization/verification), the directory
will look like this:

### Labeled data before Strategy D (`data/raw/`)

Only explicit `SPLIT_MODE=fresh` calls `analyze_embeddings.py` with `--data_dir` pointing at
`data/raw/`. That script walks **train**, **val**, and **test** inputs: for each one it expects
`data/raw/<split>/` (MOFTransformer files) and a matching
`data/raw/<split>_bandgaps_regression.json` when that split exists.

**Layout A — single pool (simplest):** put **all** labeled structures under `data/raw/test/` and **all** bandgaps in `data/raw/test_bandgaps_regression.json` only. Train/val JSONs and folders may be absent; embeddings are extracted from the test pool only, then `embedding_split.py` (still reading labels from `data/raw/`) builds Strategy D under `data/splits/`.

**Layout B — initial train/val/test:** use three subfolders `data/raw/train|val|test/` and three JSON files `train_bandgaps_regression.json`, etc., if you already have a split (e.g. from QMOF) before re-splitting in embedding space.

Optional: keep original **CIF** files under `data/raw/cif/` for SOAP (figures F4, Step 7).

```
data/
├── raw/
│   ├── test/                           # MOFTransformer preprocessed files (see layouts A/B above)
│   │   ├── <CIF_ID>.grid
│   │   ├── <CIF_ID>.griddata16
│   │   └── <CIF_ID>.graphdata
│   ├── test_bandgaps_regression.json   # Required for layout A (all labels); also part of layout B
│   └── cif/                            # (optional) Original .cif for SOAP
│       └── <CIF_ID>.cif
│
├── splits/
│   └── strategy_d_farthest_point/      # JSONs COMMITTED (the paper's exact split)
│       ├── train/                      #   symlinks to raw/ files (create locally)
│       ├── val/
│       ├── test/
│       ├── train_bandgaps_regression.json   # committed — 1,136 structures, 60 positives
│       ├── val_bandgaps_regression.json     # committed —   524 structures,  5 positives
│       └── test_bandgaps_regression.json    # committed — 9,150 structures,  9 positives
│
├── embeddings/                         # Created by Step 1
│   └── embeddings_pretrained.npz       #   768-dim pretrained CLS embeddings
│
├── embedding_classifiers/              # Created by Step 3
│   └── strategy_d_farthest_point/      #   saved sklearn models
│
├── ensemble_results/                   # Created by Step 4
├── final_results/                      # Created by Step 5
│
├── unlabeled/                          # Separate unlabeled MOFs for Steps 6-7
│   ├── test/
│   │   ├── <CIF_ID>.grid
│   │   ├── <CIF_ID>.griddata16
│   │   └── <CIF_ID>.graphdata
│   ├── test_bandgaps_regression.json   #   CIF IDs → 0.0 (placeholder)
│   └── embedding_analysis/             # Created by Step 6
│       └── unlabeled_embeddings.npz
│
└── qmof.csv                           # (optional) QMOF Database metadata
                                        #   needed for metal center UMAP panels (F2, F3)
```

## File Formats

Each MOF structure is represented by **three files** sharing the same CIF ID stem:

| File              | Description                                           |
|-------------------|-------------------------------------------------------|
| `<CIF_ID>.grid`        | 3-D energy grid metadata (dimensions, spacing)   |
| `<CIF_ID>.griddata16`  | Flattened energy grid values (float16)            |
| `<CIF_ID>.graphdata`   | Graph representation (atoms, bonds, unit cell)    |

These are the input formats expected by **MOFTransformer**. Generate them from raw CIF files using MOFTransformer's `prepare_data` utility (see Step 0 in the main [README](../README.md)).

## Label Files

Label JSON files map CIF IDs to bandgap values in eV. When repurposing the pipeline for a
different scalar property, keep the `*_bandgaps_regression.json` filenames and the
`bandgaps_regression` downstream identifier — they are fixed pipeline identifiers used
throughout both repositories — and simply store the new property's values:

```json
{
  "QMOF-a1b2c3": 0.42,
  "QMOF-d4e5f6": 2.15
}
```

**Classification threshold:** bandgap <= 1.0 eV → "positive" (potentially conductive).

For the **unlabeled** set, create `test_bandgaps_regression.json` with placeholder values:

```json
{
  "QMOF-x1y2z3": 0.0,
  "QMOF-a9b8c7": 0.0
}
```

The pipeline only uses the keys (CIF IDs) for inference — the values are ignored.

## `qmof.csv` (optional)

This file contains QMOF Database metadata, particularly the `name` and `info.formula` columns used to extract metal center information for UMAP panels in figures F2 and F3. Download it from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF). If missing, metal center panels will display "Unknown" for all MOFs.

## Obtaining the Data

The raw data originates from the **QMOF Database**:

1. Download structures and bandgap values from the [QMOF Database](https://github.com/Andrew-S-Rosen/QMOF).
2. Convert CIF files to MOFTransformer format using `moftransformer.utils.prepare_data` (Step 0).
3. Place all preprocessed files in `data/raw/test/`.
4. Create `data/raw/test_bandgaps_regression.json` mapping every CIF ID to its bandgap value in eV (this is the initial label file before splitting):
   ```json
   {"QMOF-a1b2c3": 0.42, "QMOF-d4e5f6": 2.15, ...}
   ```
5. Restore the labeled PMTransformer archive and run Step 1
   (`scripts/01_extract_embeddings.sh`); paper mode verifies/materializes its recorded partition and
   links the preprocessed structures. Only explicit `SPLIT_MODE=fresh` extracts new embeddings and
   designs a noncanonical split.
6. For discovery (Steps 6-7), prepare a separate unlabeled set in `data/unlabeled/test/` with a placeholder `data/unlabeled/test_bandgaps_regression.json`.
