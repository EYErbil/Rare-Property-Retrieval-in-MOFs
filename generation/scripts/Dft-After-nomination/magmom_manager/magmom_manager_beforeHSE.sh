#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/vasp_magmom_manager.py" extract \
  --root /path/to/DFT_WORK_ROOT \
  --source-stage PBED3-Single \
  --target-stage HSE-single \
  --backup \
  --write
