#!/bin/bash
#
# Mass submit for PBED3-Relax (same OUTCAR buckets as status.sh).
# Echo + sbatch naming matches submit_relax.sh: "SKIP $mofname: ..." and "SUBMIT $jobname".
#
#   - SKIP if in squeue (name or stage dir).
#   - SKIP if timing present and no Iteration 249 (successful completion).
#   - Max-step (timing + Iteration 249): require CONTCAR; mv CONTCAR POSCAR; then sbatch.
#   - Other submit paths: no CONTCAR rename.
#
# MPI memory summary lines are ignored (they can appear mid-OUTCAR).
#
# Usage: bash mass_submit_better_relax.sh
# Override: ROOT=... STAGE=... bash mass_submit_better_relax.sh

ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"
STAGE="${STAGE:-PBED3-Relax}"

has_timing_block() {
  outcar="$1"
  if [ -s "$outcar" ] && grep -q "General timing and accounting informations" "$outcar"; then
    return 0
  fi
  return 1
}

has_iteration_249() {
  outcar="$1"
  grep -Eq "Iteration[[:space:]]+249[[:space:]]*\\(" "$outcar"
}

job_already_in_queue() {
  jobname="$1"
  stagedir="$2"

  if squeue -h -u "$USER" -n "$jobname" | grep -q .; then
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
  outcar="$stagedir/OUTCAR"
  contcar="$stagedir/CONTCAR"

  if [ ! -d "$stagedir" ]; then
    echo "SKIP $mofname: no $STAGE folder"
    continue
  fi

  if job_already_in_queue "$jobname" "$stagedir"; then
    echo "SKIP $mofname: job already in queue or running for this folder"
    continue
  fi

  has_timing=0
  has_i249=0
  if [ -e "$outcar" ] && [ -s "$outcar" ]; then
    if has_timing_block "$outcar"; then
      has_timing=1
    fi
    if has_iteration_249 "$outcar"; then
      has_i249=1
    fi
  fi

  if [ "$has_timing" -eq 1 ] && [ "$has_i249" -eq 0 ]; then
    echo "SKIP $mofname: successful completion (timing present, no Iteration 249)"
    continue
  fi

  max_step=0
  if [ "$has_timing" -eq 1 ] && [ "$has_i249" -eq 1 ]; then
    max_step=1
  fi

  if [ "$max_step" -eq 1 ]; then
    if [ ! -s "$contcar" ]; then
      echo "SKIP $mofname: max-step restart needs non-empty CONTCAR"
      continue
    fi
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

  if [ "$max_step" -eq 1 ]; then
    if ! mv "$contcar" "$stagedir/POSCAR"; then
      echo "SKIP $mofname: mv CONTCAR to POSCAR failed"
      continue
    fi
  fi

  echo "SUBMIT $jobname"
  sbatch --job-name="$jobname" --chdir="$stagedir" "$script"
done
