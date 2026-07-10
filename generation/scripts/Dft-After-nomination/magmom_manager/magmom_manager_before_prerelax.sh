#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/vasp_magmom_manager.py" seed \
  --root /path/to/DFT_WORK_ROOT \
  --stage PBED3-PreRelax \
  --afm \
  --backup \
  --write
