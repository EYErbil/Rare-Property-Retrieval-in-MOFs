# Generation module — hypothetical-framework assembly and DFT validation

### Assembling new candidate structures with PORMAKE and confirming rare-property hits with hybrid-functional DFT

This is the **generation module** of the [rare-property-retrieval](../) repository — the
generative half of the enrichment-driven workflow, demonstrated on low-band-gap MOF discovery.
Structure assembly builds on [PORMAKE](https://github.com/Sangwon91/PORMAKE) (rule-based
reticular construction); the scoring models are trained in the screening module:

> | Module | Role |
> |---|---|
> | [`screening/`](../screening/) | Trains the PMTransformer regressor + ExtraTrees classifier and screens *known* structures |
> | **`generation/`** *(this module)* | **Generates** new candidates, screens them with the trained models, and runs the **DFT** validation |
>
> Paper: *Enrichment-driven discovery of low-band-gap metal–organic frameworks with pretrained
> porous-material representations* (see [Citation](#citation)). Author: Ege Yiğit Erbil, Koç University.

---

## What problem does this solve?

Metal–organic frameworks (MOFs) with **narrow electronic band gaps** (HSE06 $E_\mathrm{g} \le 1$ eV)
are valuable for optoelectronics, photocatalysis, sensing, and energy conversion — but they are
**extremely rare** (under 1% of the labelled [QMOF](https://github.com/Andrew-S-Rosen/QMOF)
database) and the reliable hybrid-functional (HSE06) calculations needed to confirm them are
expensive. Searching only *known* frameworks is therefore limiting.

This repository takes the complementary route: it **invents new, chemically realistic MOFs** and
funnels them down to a handful worth the cost of DFT. Concretely, it

1. **mines QMOF** for the topologies, metal nodes, and organic linkers that real MOFs are made of,
2. **assembles brand-new hypothetical frameworks** from those building blocks with
   [PORMAKE](https://github.com/Sangwon91/PORMAKE) (rule-based reticular construction — no random
   atom placement),
3. **scores and ranks** every generated structure with the trained models from
   the [screening module](../screening/) (fine-tuned PMTransformer regressor +
   ExtraTrees classifier, fused by reciprocal-rank fusion),
4. **nominates a small, chemically diverse subset** for validation, and
5. **runs the full DFT cascade** (PBE-D3(BJ) → HSE06) and analyses the electronic structure of the
   confirmed hits.

## How it works (conceptual pipeline)

```mermaid
flowchart LR
    QMOF[("QMOF database<br/>known MOFs + DFT gaps")] --> A["Analyze QMOF<br/>frequent topologies, metals, linkers"]
    A --> B["QMOF-aligned PORMAKE libraries<br/>building blocks + topologies"]
    B --> C["Assemble new MOFs<br/>PORMAKE reticular construction"]
    C --> D["Screen & rank<br/>PMTransformer + ExtraTrees + RRF<br/>(models from the screening module)"]
    D --> E["Diversity-aware nomination<br/>SOAP clustering + MMR"]
    E --> F["DFT validation<br/>PBE-D3(BJ) → HSE06 cascade"]
    F --> G(["5 novel low-band-gap MOFs<br/>confirmed at HSE06"])
```

## Results at a glance

Applied to **13,802** generated frameworks, the workflow nominated **25** candidates for
hybrid-functional DFT. Five did not converge within the protocol's iteration limits and were set
aside; the remaining **20** completed the full PBE-D3(BJ) → HSE06 cascade, and **5** were confirmed
with HSE06 band gaps at or below 1 eV — a **25% hit rate** among completed validations (20% if all
selected candidates are counted), versus the sub-1% prevalence in labelled QMOF. **None** of the five matches any structure
(or even a reduced composition) in **QMOF**, **CoRE MOF 2019**, **hMOF**, or **ToBaCCo**, so all
five are novel against the databases searched.

| Label | Net | Node | Linker (neutral acid) | HSE06 $E_\mathrm{g}$ (eV) | In databases? |
|-------|-----|------|------------------------|:---:|:---:|
| `hex+N199+E185` | hex | Cu₄ (8-c) | C₈H₂Br₄O₄ | 0.25 | no |
| `pcu+N273+E44`  | pcu | Mn₃ (6-c) | C₄H₂O₄ | 0.54 | no |
| `hex+N67+E151`  | hex | Cd₃ (8-c) | C₈H₁₂O₄ | 0.56 | no |
| `pcu+N273+E128` | pcu | Mn₃ (6-c) | C₈H₇NO₄ | 0.81 | no |
| `cds+N29+E128`  | cds | Mn₂ (4-c) | C₈H₇NO₄ | 0.83 | no |

Each hit is a force-converged PBE-D3(BJ) local minimum (a metastable structure, not a generator
artefact); thermodynamic stability and experimental synthesizability are beyond this computational
screen.

## How this repository maps to the paper

| Paper section (Methods/Results) | What this repository provides |
|---|---|
| *Generated MOF database construction* | `scripts/analyze_reference_db.py`, `scripts/build_custom_dirs.py`, `bulk_pormake_generation/make_candidates.py`, `scripts/build_materials_batched.py` |
| *Candidate selection for validation* | `nominate_diverse_dft.py` (SOAP clustering + MMR + uncertainty) |
| *Selected candidates sample diverse regions …* (chemical-space figures) | `scripts/soap_analysis/`, `scripts/pmtransformer_analysis/`, `qmof_pool_analysis/` |
| *DFT protocol* (PBE-D3(BJ) → HSE06) | `scripts/Dft-After-nomination/` (four-stage VASP cascade + MAGMOM manager) |
| *Electronic-structure analysis* (projected DOS) | `DOS_analysis/` |
| *Building-block and linker identification* | [`scripts/resolve_linker_smiles.py`](scripts/resolve_linker_smiles.py) — Open Babel SMILES/InChI/InChIKey from the PORMAKE building-block bond graphs |
| *Stability and novelty assessment* | [`scripts/novelty/novelty_check_multidb.py`](scripts/novelty/novelty_check_multidb.py) — `pymatgen` `StructureMatcher` vs QMOF, CoRE MOF 2019, hMOF, ToBaCCo |
| Main-text hit tables + the five confirmed-hit CIFs | results archive — see the paper's Data availability |

## What's already included (you don't rebuild the QMOF generation space)

The QMOF-informed assembly space is **committed to this repository**, so you can start generating
immediately without re-mining QMOF:

| Provided | Contents |
|----------|----------|
| `qmof.csv` | The QMOF release table — source of the topology / metal / linker distributions. |
| `qmof_bb_dir/` | The building-block library: **506 metal-node SBUs** (`N*.xyz`) and **229 organic edges/linkers** (`E*.xyz`, including augmented `E9xxx` linkers). |
| `qmof_topo_dir/` | **42 net topologies** (`.cgd`: `pcu`, `hex`, `cds`, `dia`, `fcu`, `sod`, `nbo`, `acs`, …). |
| `qmof_analysis/` | The QMOF-derived whitelists (`selected_{topologies,metals,linkers}.txt`), frequency counts, and `coverage_report.md`. |

The **only** generation input you must create yourself is the RMSD node↔topology compatibility table
**`data/rmsd_qmof.pickle`** (not committed — it is large and fully derived). Build it once with
`build_rmsd_table.sh` before Step 2. Rebuilding the libraries above (Step 1) is **optional** — do it only if
you want to change the QMOF coverage thresholds.

## Reproducing the paper

A full reproduction uses **both modules of this repository**. Train the models in
[`screening/`](../screening/) first (see its “Reproducing the paper”), then run this module:

1. **Install** this module's `requirements.txt`. Set `TRAIN_ROOT` to the `screening/` module
   directory (it holds `train_regressor.py` and the trained `experiments/`).
2. **Steps 1–12** — follow [REPLICATION.md](REPLICATION.md): (optionally) rebuild the
   QMOF-aligned libraries, generate the candidate database (≈13,802 structures), preprocess it,
   screen it with the trained models, and nominate the **25** diversity-aware DFT candidates.
3. **Steps 13–26** — follow [DFT_WORKFLOW.md](DFT_WORKFLOW.md): the four-stage PBE-D3(BJ) →
   HSE06 cascade (20 of the 25 complete within the protocol's iteration limits) → the
   **5 confirmed low-band-gap hits**.

## Beyond low-band-gap MOFs

The generate–screen–validate loop in this repository is property- and dataset-agnostic:

- **Generation** imitates whatever reference chemistry the whitelists define (next section) —
  PORMAKE assembly makes no reference to band gaps or any other target property.
- **Screening** consumes any scoring model that emits one score per structure; the ranking,
  reciprocal-rank fusion, and diversity-aware nomination make no assumption about the property
  being screened.
- **Validation** is a staged VASP protocol (PBE-D3(BJ) relaxation → hybrid single point) whose
  acceptance criterion is read from the converged calculation. Band gaps are one choice; any
  observable computable from the same cascade — formation energetics, magnetic order, projected
  densities of states — can serve instead.

Together with the [`screening/`](../screening/) module, this constitutes a general pipeline for
rare-event materials discovery under a fixed budget of expensive validations.

> **Naming note for other properties.** The `*_bandgaps_regression.json` label filenames and
> the `bandgaps` / `bandgaps_regression` downstream tags are fixed pipeline identifiers shared
> by both repositories. When screening a different property, keep these names and replace the
> stored values — do not rename the files.

## Generate from your own reference database

QMOF is only the **default** reference set — the generator itself is dataset-agnostic. Point
[`scripts/analyze_reference_db.py`](scripts/analyze_reference_db.py) at any reference database
table and the pipeline extracts its frequent **topologies, metal nodes, and organic linkers**,
builds PORMAKE libraries restricted to that chemistry, and generates new candidates from it —
exactly the procedure the paper applied to QMOF. The chemistry it imitates is defined entirely
by three plain-text whitelists (`selected_topologies.txt`, `selected_metals.txt`,
`selected_linkers.txt`), and the construction step (`make_candidates.py` / `build_materials.py`)
simply consumes whatever `--bb-dir` / `--topo-dir` you give it. There are three entry points,
by decreasing level of automation.

### Option A — your database as a decomposed table (recommended)

`analyze_reference_db.py` accepts **any** CSV; the column names default to QMOF's MOFid columns but are fully
overridable. Each row needs a topology code, a list of node SMILES (with bracketed metals, e.g.
`"['[Zn]', '[Cu]']"`), and a list of organic-linker SMILES; the pore-geometry columns are optional.

> **Starting from raw CIFs?** Deriving topology / node / linker identifiers from crystal
> structures is exactly what the published [MOFid](https://github.com/snurr-group/mofid) tool
> does. Run MOFid over your CIFs, collect its topology and node/linker SMILES output into a CSV,
> and proceed below. (QMOF ships these columns precomputed, which is why the paper's run needs
> no extra step.)

```bash
cd REPO_ROOT
# 1. Mine YOUR table for the frequent topologies / metals / linkers
python scripts/analyze_reference_db.py \
  --csv /path/to/your_dataset.csv \
  --out my_analysis \
  --topology-col your_topology_column \
  --nodes-col    your_node_smiles_column \
  --linkers-col  your_linker_smiles_column \
  --pld-col your_pld_column --lcd-col your_lcd_column      # omit these two if you have no pore data

# 2. Build PORMAKE libraries restricted to YOUR chemistry
python scripts/build_custom_dirs.py \
  --analysis-dir my_analysis \
  --topo-out my_topo_dir \
  --bb-out   my_bb_dir

# 3. Generate exactly as in Steps 2-3, pointing at your own dirs
python bulk_pormake_generation/rmsd_calculated_node.py \
  --save data/rmsd_mine.pickle --bb-dir my_bb_dir --topo-dir my_topo_dir
python bulk_pormake_generation/make_candidates.py -n 20000 --max-n-atoms 200 \
  --pre-defined-list data/rmsd_mine.pickle --bb-dir my_bb_dir --topo-dir my_topo_dir \
  --save data/candidates_mine.txt   # -n = candidate strings drawn; build filters reduce this
python scripts/build_materials_batched.py --candidates data/candidates_mine.txt \
  --bb-dir my_bb_dir --topo-dir my_topo_dir \
  --save-dir generated_cifs/mine_small --large-dir generated_cifs/mine_large \
  --cutoff 30.0 --chunk-size 200
```

If a column name is wrong, `analyze_reference_db.py` stops and prints the columns it actually found.

### Option B — supply the whitelists directly (any source format)

If your reference data isn't a tidy CSV, skip `analyze_reference_db.py` and write the three whitelist files
yourself, then run `build_custom_dirs.py` as above. The formats are trivial — **one item per line**:

| File | Contents |
|------|----------|
| `selected_topologies.txt` | RCSR-style net codes that exist in PORMAKE (e.g. `pcu`, `dia`, `cds`). |
| `selected_metals.txt` | Element symbols of the metals to keep in node SBUs (e.g. `Zn`, `Cu`, `Mn`). |
| `selected_linkers.txt` | Canonical organic-linker SMILES used to (optionally) augment the edge library. |

### Option C — bring your own building blocks

If you already have PORMAKE-format node/edge `.xyz` files and `.cgd` topologies, skip the analysis
entirely and pass your own `--bb-dir` / `--topo-dir` straight to `make_candidates.py` /
`build_materials.py`.

> **What never changes:** the node↔topology RMSD matching, atom-count limits, the cell-size filter, the
> ML screening, and the DFT cascade are all reference-agnostic — only the *whitelists* (i.e. the
> building-block and topology libraries) differ.

## Code, not results

This repository contains the **code and workflow only** — no result files. The exact package
versions used for the paper are frozen in [`env/`](env/) (`requirements_finetuning.txt` and
`requirements_analysis.txt`); the top-level `requirements.txt` remains the permissive
quick-install list; the exact package versions used for the paper are frozen at the repository
root in [`../env/`](../env/). The paper's exact train/val/test split JSONs are committed in the
screening module, [`../screening/data/splits/`](../screening/data/splits/strategy_d_farthest_point/).

All result artifacts — the five HSE06-confirmed hit CIFs, curated hit tables, full ranked
screening tables, embedding archives, and the complete DFT inputs and outputs for every
completed validation — are distributed separately (see the paper's Data availability).

## Large files via Globus (and the `PhaseN` names)

The embedding/descriptor archives and all raw DFT calculation files are **not stored in this
repository** — they are shared via Globus (see the paper's Data availability). After downloading,
restore each archive to the path below; a `README.md` placeholder marks each location.

| Globus file | Restore to | Meaning |
|------|------|---------|
| `pmt_embeddings_qmof_labeled.npz` | `embeddings/` | PMTransformer embeddings of the **labelled** QMOF set |
| `pmt_embeddings_qmof_unlabeled.npz` | `embeddings/` | PMTransformer embeddings of the **unlabelled** QMOF screening pool |
| `pmt_embeddings_qmof_all.npz` | `embeddings/` | Labelled + unlabelled QMOF embedded in **one** aligned forward pass (reference cache) |
| `soap_descriptors_sparse.npz` | `soap_analysis/` | Cached SOAP descriptors (96 MB — too close to GitHub's 100 MB hard limit) |
| DFT stage folders (per validated MOF) | your `DFT_WORK_ROOT` | Complete PBED3-PreRelax / PBED3-Relax / PBED3-Single / HSE-single inputs and outputs |

(Historical provenance strings may refer to the embedding archives by their original working
names `Phase5_embeddings.npz`, `Phase6_embeddings.npz`, and `all_embeddings.npz`.)

---

## Detailed guides

The complete, step-numbered manuals live in three companion documents:

| Guide | Covers |
|---|---|
| [REPLICATION.md](REPLICATION.md) | Steps 1–12: reference-database analysis, PORMAKE construction, preprocessing, chemical-space analyses, inference, and nomination of the 25 DFT candidates |
| [DFT_WORKFLOW.md](DFT_WORKFLOW.md) | Steps 13–26: the four-stage PBE-D3(BJ) → HSE06 VASP cascade with MAGMOM management |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Optional Slurm wrappers and a symptom → fix table for every step |

## Related module

The machine-learning models used here for scoring and ranking are developed, trained, and
benchmarked in the [`screening/`](../screening/) module — diversity-aware ensemble screening of
*known* structures (PMTransformer regressor + ExtraTrees classifier + RRF).

## Acknowledgements & upstream tools

- **[PORMAKE](https://github.com/Sangwon91/PORMAKE)** — rule-based reticular MOF construction.
- **[bulk_pormake_generation](https://github.com/Yeonghun1675/bulk_pormake_generation)** — upstream
  bulk-generation utilities that this repository builds on.
- **[MOFTransformer / PMTransformer](https://github.com/hspark1212/MOFTransformer)** — pretrained
  porous-material representation used for embeddings and band-gap regression.
- **[QMOF Database](https://github.com/Andrew-S-Rosen/QMOF)** — reference MOFs and DFT band gaps.
- **[DScribe](https://github.com/SINGROUP/dscribe)** (SOAP), **[pymatgen](https://pymatgen.org/)**,
  **[Open Babel](https://openbabel.org/)**, **[UMAP](https://github.com/lmcinnes/umap)**, and **VASP**
  for descriptors, structure handling, cheminformatics, visualisation, and DFT.

## Citation

If you use this software, please cite the accompanying paper:

> Erbil, E. Y. *et al.* Enrichment-driven discovery of low-band-gap metal–organic frameworks with
> pretrained porous-material representations. *Manuscript in preparation* (2026).

```bibtex
@article{erbil2026lowgapmof,
  title   = {Enrichment-driven discovery of low-band-gap metal--organic frameworks
             with pretrained porous-material representations},
  author  = {Erbil, Ege Yi{\u{g}}it and others},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```
<!-- TODO: update author list, journal/DOI, and year once the paper is published. -->

To cite this software repository specifically, see the [repository-level README](../README.md).

## License

Released under the [MIT License](../LICENSE) © 2025 Ege Yiğit Erbil, Koç University.
