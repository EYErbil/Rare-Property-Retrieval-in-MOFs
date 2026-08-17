# Data licensing, provenance, and third-party notices

## Scope

The repository-level [MIT License](LICENSE) covers the author-written software unless a nested
directory carries its own license. Unless otherwise noted below, author-created data and derived
metadata committed to this repository are released under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0). Source-derived material remains subject to its source-specific terms.

The large calculation, descriptor, embedding, and generated-structure release is distributed
separately. That archival collection carries its own `LICENSE_DATA.md`, `PROVENANCE.md`, and
SHA-256 manifest.

## QMOF-derived material

The split identifiers, QMOF-aligned whitelists, selected building-block metadata, labels, and
other QMOF-derived records originate from the public
[QMOF Database](https://github.com/Andrew-S-Rosen/QMOF), whose underlying data are available under
CC BY 4.0. Repository paths containing QMOF-derived material include `screening/data/splits/`,
`generation/qmof_analysis/`, `generation/qmof_bb_dir/`, and `generation/qmof_pool_analysis/`.
Selections, filtering, representation generation, and identifier alignment performed for this
study constitute changes to the source data.

Relevant QMOF citations:

- Rosen, A. S. et al. *Matter* **4**, 1578–1597 (2021).
  https://doi.org/10.1016/j.matt.2021.02.015
- Rosen, A. S. et al. *npj Computational Materials* **8**, 112 (2022).
  https://doi.org/10.1038/s41524-022-00796-6

The repository does not intentionally redistribute the license-restricted initial CSD structures
from which some public QMOF DFT-optimized structures were derived.

## PORMAKE and bulk-generation material

Structure construction uses [PORMAKE](https://github.com/Sangwon91/PORMAKE), version 0.2.2 in the
paper environment. PORMAKE is distributed under the MIT License. The bundled
`generation/bulk_pormake_generation/` utilities derive from
[Yeonghun1675/bulk_pormake_generation](https://github.com/Yeonghun1675/bulk_pormake_generation)
and retain the upstream MIT license at `generation/bulk_pormake_generation/LICENSE`.

The QMOF-selected `.xyz` building-block and `.cgd` topology libraries are provided to materialize
the documented generation space. Their selection is study-specific; the underlying PORMAKE and
source-resource attributions are not replaced by this repository's licenses.

PORMAKE method citation:

- Lee, S. et al. *ACS Applied Materials & Interfaces* **13**, 23647–23654 (2021).
  https://doi.org/10.1021/acsami.1c02471

## RCSR topology attribution

The topology codes and topology records used through PORMAKE follow the Reticular Chemistry
Structure Resource (RCSR). No ownership of RCSR nomenclature or source records is claimed. Users
redistributing source-derived topology collections should preserve RCSR attribution and follow any
terms specified by the source resource.

- O'Keeffe, M., Peskov, M. A., Ramsden, S. J. & Yaghi, O. M. *Accounts of Chemical Research*
  **41**, 1782–1789 (2008). https://doi.org/10.1021/ar800124u

## VASP

This repository and the associated data release contain no VASP executable, source code, or
licensed `POTCAR` files. Reproducing VASP calculations requires users to obtain their own VASP
license and construct the required `POTCAR` files from their licensed potential library.
