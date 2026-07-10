#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"

ROOT="$ROOT" STAGE="PBED3-Single" MAX_NELM_ITER="${MAX_NELM_ITER:-150}" bash "$SCRIPT_DIR/status_single.sh"
