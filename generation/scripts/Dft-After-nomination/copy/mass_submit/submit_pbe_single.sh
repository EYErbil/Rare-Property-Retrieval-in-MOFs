#!/bin/bash

ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"
STAGE="${STAGE:-PBED3-Single}"
MAX_NELM_ITER="${MAX_NELM_ITER:-150}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "BURNER MODE: no sbatch"
  echo "SCRIPT       : ${BASH_SOURCE[0]}"
  echo "ROOT         : $ROOT"
  echo "STAGE        : $STAGE"
  echo "MAX NELM ITER: $MAX_NELM_ITER"
fi

has_timing_block() {
  outcar="$1"
  if [ -s "$outcar" ] && grep -q "General timing and accounting informations" "$outcar"; then
    return 0
  fi
  return 1
}

has_max_nelm_iteration() {
  outcar="$1"
  grep -Eq "Iteration[[:space:]]+[0-9]+[[:space:]]*\\([[:space:]]*$MAX_NELM_ITER[[:space:]]*\\)" "$outcar"
}

job_already_in_queue() {
  jobname="$1"
  stagedir="$2"

  if ! command -v squeue >/dev/null 2>&1; then
    return 1
  fi

  if squeue -h -u "$USER" -n "$jobname" 2>/dev/null | grep -q .; then
    return 0
  fi

  if squeue -h -u "$USER" -o "%Z" 2>/dev/null | grep -Fxq "$stagedir"; then
    return 0
  fi

  return 1
}

find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "copy" | sort | while IFS= read -r mofdir; do
  mofname="$(basename "$mofdir")"
  stagedir="$mofdir/$STAGE"
  jobname="${mofname}_${STAGE}"

  if [ ! -d "$stagedir" ]; then
    echo "SKIP $mofname: no $STAGE folder"
    continue
  fi

  if job_already_in_queue "$jobname" "$stagedir"; then
    echo "SKIP $mofname: job already in queue or running for this folder"
    continue
  fi

  has_timing=0
  has_maxnelm=0
  if [ -e "$stagedir/OUTCAR" ] && [ -s "$stagedir/OUTCAR" ]; then
    if has_timing_block "$stagedir/OUTCAR"; then
      has_timing=1
    fi
    if has_max_nelm_iteration "$stagedir/OUTCAR"; then
      has_maxnelm=1
    fi
  fi

  if [ "$has_timing" -eq 1 ] && [ "$has_maxnelm" -eq 0 ]; then
    echo "SKIP $mofname: successful completion (timing present, no electronic Iteration $MAX_NELM_ITER)"
    continue
  fi

  for f in INCAR POSCAR POTCAR KPOINTS; do
    if [ ! -s "$stagedir/$f" ]; then
      echo "SKIP $mofname: missing/empty $f"
      continue 2
    fi
  done

  n_scripts=$(find "$stagedir" -maxdepth 1 -type f -name "*.sh" | wc -l)

  if [ "$n_scripts" -ne 1 ]; then
    echo "SKIP $mofname: expected 1 .sh file, found $n_scripts"
    find "$stagedir" -maxdepth 1 -type f -name "*.sh"
    continue
  fi

  script=$(find "$stagedir" -maxdepth 1 -type f -name "*.sh" | head -n 1)

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "BURN $mofname: would submit $jobname"
  else
    echo "SUBMIT $jobname"
    sbatch --job-name="$jobname" --chdir="$stagedir" "$script"
  fi
done
