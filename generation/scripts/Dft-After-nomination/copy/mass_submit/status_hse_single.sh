#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"

ROOT="$ROOT" STAGE="HSE-single" MAX_NELM_ITER="${MAX_NELM_ITER:-100}" bash "$SCRIPT_DIR/status_single.sh"
