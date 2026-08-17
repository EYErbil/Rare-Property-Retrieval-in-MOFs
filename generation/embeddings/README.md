# Frozen pretrained PMTransformer embedding archives

These large inputs are restored from the paper's Zenodo deposit into this folder. They are used
for split recovery, ML scoring, and representation analyses. The labeled train/validation/test
partitions were designed before any task-specific training from distances in the fixed pretrained
PMTransformer space. SOAP is the sole diversity coordinate for the generated-pool nomination; the
second-phase QMOF acquisition ran the nomination procedure separately in PMTransformer-embedding
and SOAP spaces, so these embeddings are a diversity coordinate there as well.

| File (after restore) | Zenodo path | Verified contents |
|---|---|---|
| `pmt_embeddings_qmof_labeled.npz` | `embeddings/qmof_labeled_pmtransformer_embeddings.npz` | 10,810 labeled QMOF structures; includes band gaps and the paper partition labels. Copy to `../../screening/data/embeddings/embeddings_pretrained.npz` for the canonical split verifier |
| `pmt_embeddings_qmof_all.npz` | `embeddings/qmof_pmtransformer_embeddings.npz` | authoritative aligned QMOF cache: 20,371 rows = 10,810 labeled + 9,561 unlabeled; canonical source for the complete unlabeled pool |
| `generated_pmt_embeddings.npz` | `embeddings/generated_pmtransformer_embeddings.npz` | 13,802 generated structures; identifiers and row order identical to the generated SOAP archive |

The historical standalone unlabeled archive contains only 9,527 structures. It has been moved to
the umbrella workspace's local, non-release `RESULTS/local_only_archive/` area and must not be used as the paper cache or
deposited on Zenodo. Select the 9,561 unlabeled IDs from `pmt_embeddings_qmof_all.npz` instead.

Historical logs may refer to `pmt_embeddings_qmof_labeled.npz`, the incomplete standalone archive,
and `pmt_embeddings_qmof_all.npz` as `Phase5_embeddings.npz`, `Phase6_embeddings.npz`, and
`all_embeddings.npz`, respectively.

The released generated SOAP archive independently carries all 13,802 `cif_ids`. Use
`../scripts/materialize_generated_pool_manifest.py` to recover the exact paper-pool identity and
descriptor-row order without a separately distributed CSV. Zenodo also ships the corresponding
PORMAKE structure files under `generated_structures/`, so the pool can be used directly instead of
being reconstructed.
