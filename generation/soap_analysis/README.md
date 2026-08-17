# SOAP descriptor archives

SOAP is the **sole structural-diversity coordinate for the generated-pool nomination**, in both the
main RRF-prioritized tier and the disagreement-prioritized exploration tier. The second-phase QMOF
acquisition ran the same procedure separately in PMTransformer-embedding and SOAP spaces, so there
SOAP is one of two diversity coordinates. RRF and model disagreement affect priority only. This
training-free local-environment geometry keeps structural selection independent of prediction
scores.

Restore these Zenodo archives to the paths shown:

| File | Verified contents |
|---|---|
| `soap_descriptors_sparse.npz` | QMOF SOAP matrix with 20,370 rows; `core_ERIWAF_freeONLY` is the one absent QMOF ID |
| `generated_vs_qmof/generated_soap_descriptors.npz` | generated-pool SOAP matrix with 13,802 rows |

The QMOF archive is a custom sparse-CSR NPZ and the generated archive is a dense, RAM-heavy
matrix; the nomination loaders support both formats. Derived UMAP caches, CSVs, ranking tables, and
legacy nomination folders are not distributed.
