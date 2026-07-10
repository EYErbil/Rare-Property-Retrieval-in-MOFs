# QMOF coverage report

- Total rows: **20372**
- Rows with topology label: 7833 (38.4%)
- Rows with a metal node: 17677 (86.8%)
- Rows with a linker SMILES: 17036 (83.6%)
- Linker SMILES that RDKit could not canonicalize: 984/27693

## Topology whitelist
- Target item-count coverage: 95%
- Achieved item-count coverage: 95.0%
- Selected 52 topologies
- Rows whose topology set intersects the whitelist: 36.5%
- Rows whose topology set is fully contained: 36.5%

## Metal whitelist
- Target item-count coverage: 99%
- Achieved item-count coverage: 99.0%
- Selected 45 metals
- Rows whose metal set intersects the whitelist: 85.9%
- Rows whose metal set is fully contained: 85.6%

## Linker whitelist
- Top-N cap: 200
- Selected 200 canonical linker SMILES (out of 8357 unique)
- Rows whose linker set intersects the whitelist: 46.4%
- Rows whose linker set is fully contained: 28.2%

## Pore size reference (for downstream screening)
```json
{
  "pld": {
    "count": 20372,
    "min": 0.0,
    "p10": 0.8578129999999999,
    "p50": 1.327005,
    "p90": 7.455698000000001,
    "max": 44.41671
  },
  "lcd": {
    "count": 20372,
    "min": 0.78371,
    "p10": 1.730181,
    "p50": 2.716785,
    "p90": 9.79263,
    "max": 44.89372
  }
}
```
