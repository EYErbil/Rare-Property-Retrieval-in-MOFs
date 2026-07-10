#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"

ROOT="$ROOT" STAGE="PBED3-PreRelax" bash "$SCRIPT_DIR/status_relax.sh"
