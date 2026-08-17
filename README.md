<div align="center">

# rare-property-retrieval

**Find the rare ones — and spend your expensive validations where they actually pay off.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21978025.svg)](https://doi.org/10.5281/zenodo.21978025)
[![Tests](https://github.com/EYErbil/Rare-Property-Retrieval-in-MOFs/actions/workflows/tests.yml/badge.svg)](https://github.com/EYErbil/Rare-Property-Retrieval-in-MOFs/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](DATA_AND_THIRD_PARTY_NOTICES.md)

[**Method**](#how-it-works) · [**Install**](#installation) · [**Quickstart**](#quickstart) · [**Your own property**](#use-it-on-your-own-property) · [**Reproduce the paper**](#reproducing-the-paper) · [**Cite**](#citation)

</div>

---

An **enrichment-driven machine-learning workflow** for discovering materials whose target property
is both **rare** and **expensive to confirm**. When positives are a fraction of a percent and every
validation costs hours of compute or weeks of lab time, accurate average prediction is the wrong
objective — what matters is *concentrating the few true positives at the top of a short queue*.

The pipeline is **property- and dataset-agnostic**. It is demonstrated here on **low-band-gap
metal–organic frameworks** (HSE06 $E_\mathrm{g} \le 1$ eV — under 1% of labelled data), confirmed
with hybrid-functional DFT.

![Enrichment-driven discovery workflow for low-band-gap MOFs](docs/workflow.png)

<sub>The workflow as applied in the paper. A fine-tuned PMTransformer regressor and an ExtraTrees
classifier on frozen embeddings independently rank 23,363 prospective candidates; their rankings are
combined by reciprocal-rank fusion, and diversity- and disagreement-aware nomination spends a fixed
budget of 25 DFT validations per pool. <em>(Figure 1 of the paper; vector version in
<a href="docs/workflow.pdf"><code>docs/workflow.pdf</code></a>.)</em></sub>

## Results

Demonstration task: retrieve MOFs with HSE06 $E_\mathrm{g} \le 1$ eV. Only **74 of 10,810**
HSE06-labelled QMOF structures qualify (**0.68%**).

| | Unlabelled QMOF pool | Generated pool | **Total** |
|---|---:|---:|---:|
| Structures screened | 9,561 | 13,802 | **23,363** |
| Nominated for DFT | 25 | 25 | **50** |
| Submitted to DFT | 25 | 23 | **48** |
| **Confirmed $E_\mathrm{g} \le 1$ eV** | **3** | **6** | **9** |

- **41** of the 48 submissions yielded reportable HSE06 results.
- **~122× enrichment** over random screening at a 25-structure budget, measured retrospectively on
  the labelled test partition. That partition was excluded from model fitting but was used
  retrospectively to compare models and fusion choices — it is not an untouched external benchmark.
- **Rank rescue:** the regressor buries one true positive at rank 5,942 and the classifier buries a
  *different* one at rank 8,140, yet reciprocal-rank fusion keeps its worst positive at rank 1,793 —
  robust precisely where either model alone fails catastrophically.

## How it works

Two complementary signals are extracted from a single pretrained foundation model, fused by
**rank** rather than by score, and then spent through a diversity-aware budget allocator:

```mermaid
flowchart LR
    L[("Labelled data<br/>(rare positives)")] --> S["screening/<br/>regressor + rare-class classifier<br/>reciprocal-rank fusion"]
    G["generation/<br/>PORMAKE assembly from<br/>reference-informed building blocks"] --> S
    S --> N["Diversity-aware nomination<br/>(fixed budget per pool)"]
    N --> V["Staged validation<br/>PBE-D3(BJ) → HSE06"]
    V --> H(["Confirmed rare-property hits"])
```

1. **Fine-tuned regressor** — a property model adapted end-to-end to the target property.
2. **Rare-class classifier** — lightweight trees (ExtraTrees / SMOTE) on *frozen* pretrained
   embeddings, leveraging general structural knowledge without touching the encoder.
3. **Reciprocal-rank fusion** — fusing *ranks* inherits each model's strong placements while
   discarding its catastrophic ones; model disagreement is used as a bounded exploration proxy,
   not as calibrated uncertainty.
4. **Diversity-aware nomination** — cluster-quota round-robin plus Maximal Marginal Relevance,
   measured in a representation space **separate from the ranking**, so geometry buys coverage
   while fusion and disagreement set priority.

Optionally, the **generation arm** invents new candidates: it mines a reference database for its
frequent topologies, metal nodes and organic linkers, assembles brand-new frameworks from those
building blocks with rule-based reticular construction (PORMAKE), and feeds them into the same
ranking and nomination machinery.

## Installation

Python **3.10+**. The two arms have separate dependency sets — install only what you need.

```bash
git clone https://github.com/EYErbil/Rare-Property-Retrieval-in-MOFs.git
cd Rare-Property-Retrieval-in-MOFs
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

| You want to… | Install |
|---|---|
| Train / screen / fuse / nominate | `pip install -r screening/requirements.txt` |
| Generate new candidates + run the DFT cascade | `pip install -r generation/requirements.txt` |
| Just run the test suite | `pip install numpy scipy scikit-learn pytest` |

> **PyTorch first.** `moftransformer` needs Torch and PyTorch Geometric built for *your* CUDA
> version. Install those from [pytorch.org](https://pytorch.org/get-started) before the
> requirements file, then run `moftransformer install-griday` for the energy-grid features.
>
> **Not on PyPI:** Open Babel (linker identification, `conda install -c conda-forge openbabel`)
> and VASP (licensed DFT engine, with its POTCAR pseudopotentials).

The exact paper environments are recorded as `pip freeze` files in [`env/`](env/) — see
[`env/README.md`](env/README.md) for precisely what they do and do not pin.

## Quickstart

Verify the install and see the core logic run end-to-end on synthetic fixtures — no downloads,
no GPU, a few seconds:

```bash
pip install numpy scipy scikit-learn pytest
pytest tests/ -q
```

This exercises the rank-fusion and nomination path, the paper-split materialization, the generated-pool
manifest, and the novelty-neutralization logic.

Then pick an arm and follow its step-by-step guide:

```bash
# Screening arm — train, fuse, evaluate, nominate on known structures
less screening/README.md      # Steps 0-7

# Generation arm — assemble candidates, screen, nominate, run the DFT cascade
less generation/README.md     # Steps 1-26
```

## Use it on your own property

Nothing in the method is specific to band gaps or to MOFs. **Four things change:**

| What you swap | Format | Where |
|---|---|---|
| **Label files** | plain JSON, `structure_id → scalar` | [`screening/`](screening/README.md#applying-the-pipeline-to-other-rare-event-discovery-tasks) |
| **Positive-class threshold** | one number | same |
| **Representation** | any per-structure embedding `.npz` | same |
| **Validation oracle** | any scarce, costly measurement | your call |

To generate *new* candidates in a different chemistry, point
[`scripts/analyze_reference_db.py`](generation/scripts/analyze_reference_db.py) at **any** reference
table — the column names default to QMOF's but are fully overridable — or supply the three
plain-text whitelists (topologies, metals, linkers) directly, or bring your own PORMAKE-format
building blocks. The node↔topology RMSD matching, atom-count limits, cell-size filter, ML screening
and DFT cascade are all reference-agnostic; only the whitelists differ. Full recipe:
[generation/README.md → *Beyond low-band-gap MOFs*](generation/README.md#beyond-low-band-gap-mofs).

## Repository layout

| Path | Contents | Guide |
|---|---|---|
| [`screening/`](screening/) | Model training (PMTransformer fine-tuning, ExtraTrees on frozen embeddings), rank fusion, evaluation, and diversity-aware nomination on known structures | [`screening/README.md`](screening/README.md) |
| [`generation/`](generation/) | Reference-informed PORMAKE assembly of new candidates, screening with the trained models, nomination, and the four-stage VASP validation cascade | [`generation/README.md`](generation/README.md) |
| [`env/`](env/) | Recorded pip freezes for model fine-tuning and general analysis | [`env/README.md`](env/README.md) |
| [`tests/`](tests/) | Regression tests for fusion, nomination, split materialization and novelty logic | — |
| [`docs/`](docs/) | Workflow figure (PNG + vector PDF) | — |

## Reproducing the paper

Every procedure described in the paper's Methods maps to a documented, runnable step:

1. **Screening arm** — [`screening/`](screening/README.md#reproducing-the-paper), Steps 0–7:
   preprocess the labelled set, train models on the committed exact split, repeat the
   retrospective test-partition analysis, and nominate candidates from the unlabelled pool.
2. **Generation arm** — [`generation/`](generation/README.md#reproducing-the-paper),
   Steps 1–26: assemble the candidate database from the committed building-block libraries,
   screen it with the models trained in step 1, nominate 25 diverse candidates, and run the
   PBE-D3(BJ) → HSE06 cascade.

A fresh training run is a computational replication, not a bitwise reconstruction of the paper
rankings: trained checkpoints and prediction CSVs are not distributed, and stochastic training can
change numerical scores and ranks. Likewise, random PORMAKE sampling constructs a new generated
pool. The exact 13,802 paper-pool identities can instead be materialized from the released SOAP
descriptor archive with `generation/scripts/materialize_generated_pool_manifest.py`.

## Code, not results

This repository contains **code and workflow only**. The exact train/validation/test partition membership
lists and the retained fine-tuning and general-analysis environment records are committed. Derived
CSV files, ranked tables, curated result tables, plots, trained checkpoints, and legacy nomination
archives are not distributed. The code recomputes these artifacts for a fresh run, but without the
paper checkpoints and prediction CSVs it does not promise numerically identical scores or ranks.
The Zenodo deposit (https://doi.org/10.5281/zenodo.21978025) contains
only the SOAP and frozen pretrained PMTransformer embedding archives, the PORMAKE structure files
of the 13,802-member generated pool, and the DFT calculation directories for the 25 QMOF-pool and
23 generated submissions (without licensed `POTCAR` files).

The labeled split was designed **before any task-specific training** from distances in the frozen
pretrained PMTransformer embedding space, whose fixed local-and-global geometry distributes the
rare positive chemistries without label-trained leakage. The test partition was excluded from
model fitting, but it was used retrospectively to compare model and fusion choices and is therefore
not described as an untouched external benchmark.

## Built on

[PORMAKE](https://github.com/Sangwon91/PORMAKE) and
[bulk_pormake_generation](https://github.com/Yeonghun1675/bulk_pormake_generation) (reticular
structure assembly), [MOFTransformer / PMTransformer](https://github.com/hspark1212/MOFTransformer)
(pretrained porous-material representation), the
[QMOF Database](https://github.com/Andrew-S-Rosen/QMOF) (HSE06 band gaps),
[DScribe](https://github.com/SINGROUP/dscribe), [pymatgen](https://pymatgen.org/),
[Open Babel](https://openbabel.org/), [scikit-learn](https://scikit-learn.org/), and VASP.

## Data licensing and provenance

The [MIT License](LICENSE) covers the author-written software. Data, derived metadata, bundled
third-party materials, and their source-specific terms are documented separately in
[`DATA_AND_THIRD_PARTY_NOTICES.md`](DATA_AND_THIRD_PARTY_NOTICES.md). The large archival data
release also carries a collection-level data license, provenance table, and SHA-256 manifest.

## Contributing

Issues and pull requests are welcome — bug reports, portability fixes, and adaptations to new
rare-property tasks especially. Please run `pytest tests/ -q` before opening a pull request.

## Citation

If you use this workflow, please cite the paper and the dataset. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

> Erbil, E. Y., Çağatan, Ö. V. & Dereli, B. Enrichment-driven discovery of low-band-gap
> metal–organic frameworks. *Manuscript in preparation* (2026).

```bibtex
@article{erbil2026lowgapmof,
  title   = {Enrichment-driven discovery of low-band-gap metal--organic frameworks},
  author  = {Erbil, Ege Yi{\u{g}}it and \c{C}a\u{g}atan, {\"O}mer Veysel and Dereli, B{\"u}\c{s}ra},
  year    = {2026},
  note    = {Manuscript in preparation}
}

@dataset{erbil2026dataset,
  title     = {Data for Enrichment-driven discovery of low-band-gap metal--organic frameworks},
  author    = {Erbil, Ege Yi{\u{g}}it and \c{C}a\u{g}atan, {\"O}mer Veysel and Dereli, B{\"u}\c{s}ra},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21978025}
}
```
<!-- TODO: update journal, volume and DOI once the paper is published. -->

## License

Released under the [MIT License](LICENSE) © 2025 Ege Yiğit Erbil, Koç University.
Code author: Ege Yiğit Erbil, Koç University.
