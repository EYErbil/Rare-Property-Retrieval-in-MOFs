# Embedding archives (fetched via Globus)

The PMTransformer embedding archives are distributed via Globus (see the paper's Data
availability and the repository README, section "Large files via Globus"). Download them into
this folder, keeping the filenames:

| File | Contents | Format |
|---|---|---|
| `pmt_embeddings_qmof_labeled.npz` | Pretrained PMTransformer CLS embeddings of the **HSE06-labelled** QMOF set (10,810 structures) | keys: `cif_ids`, `embeddings` (768-d) |
| `pmt_embeddings_qmof_unlabeled.npz` | Embeddings of the **unlabelled** QMOF screening pool (9,561 structures) | keys: `cif_ids`, `embeddings` (768-d) |
| `pmt_embeddings_qmof_all.npz` | Labelled + unlabelled QMOF embedded in **one aligned forward pass** (reference cache for chemical-space comparisons) | keys: `cif_ids`, `embeddings` (768-d) |

Historical note: older logs and provenance strings refer to these archives by their original
working names `Phase5_embeddings.npz`, `Phase6_embeddings.npz`, and `all_embeddings.npz`,
respectively.
