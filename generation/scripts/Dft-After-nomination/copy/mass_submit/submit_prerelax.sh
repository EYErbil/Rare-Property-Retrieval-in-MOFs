#!/bin/bash
#
# Mass submit for PBED3-PreRelax using the same decision logic as
# mass submit for PBED3-PreRelax only.
#
#   - SKIP if in squeue (name or stage dir).
#   - SKIP if the next stage is in squeue or already has run-output files.
#   - SKIP if timing present and no Iteration 149 (successful completion).
#   - Max-step (timing + Iteration 149): require CONTCAR; backup POSCAR;
#     mv CONTCAR -> POSCAR; verify; then sbatch (restore POSCAR on failure).
#   - Other submit paths: no CONTCAR rename.
#
# MPI memory summary lines are ignored (they can appear mid-OUTCAR).
#
# Usage: bash submit_prerelax.sh
# Burner: DRY_RUN=1 bash submit_prerelax.sh
# Override: ROOT=... NEXT_STAGE=PBED3-Relax bash submit_prerelax.sh

ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"
readonly SUBMIT_STAGE="PBED3-PreRelax"
NEXT_STAGE="${NEXT_STAGE:-PBED3-Relax}"
MAX_STEP_ITER="${MAX_STEP_ITER:-149}"
DRY_RUN="${DRY_RUN:-0}"
NEXT_STAGE_GUARD_FILES="${NEXT_STAGE_GUARD_FILES:-OUTCAR CONTCAR WAVECAR CHGCAR OSZICAR XDATCAR vasprun.xml}"

if [ "$SUBMIT_STAGE" != "PBED3-PreRelax" ]; then
  echo "ERROR: submit_prerelax.sh safety check failed: SUBMIT_STAGE=$SUBMIT_STAGE"
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "BURNER MODE: no sbatch, no CONTCAR -> POSCAR moves"
  echo "SCRIPT     : ${BASH_SOURCE[0]}"
  echo "ROOT       : $ROOT"
  echo "SUBMIT     : $SUBMIT_STAGE"
  echo "NEXT_STAGE : $NEXT_STAGE"
  echo "NEXT GUARD : $NEXT_STAGE_GUARD_FILES"
fi

has_timing_block() {
  outcar="$1"
  if [ -s "$outcar" ] && grep -q "General timing and accounting informations" "$outcar"; then
    return 0
  fi
  return 1
}

has_max_step_iteration() {
  outcar="$1"
  grep -Eq "Iteration[[:space:]]+$MAX_STEP_ITER[[:space:]]*\\(" "$outcar"
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

next_stage_has_work() {
  nextdir="$1"

  [ -d "$nextdir" ] || return 1

  for f in $NEXT_STAGE_GUARD_FILES; do
    if [ -s "$nextdir/$f" ]; then
      echo "$f"
      return 0
    fi
  done

  return 1
}

find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "copy" | sort | while IFS= read -r mofdir; do
  mofname="$(basename "$mofdir")"
  stagedir="$mofdir/PBED3-PreRelax"
  nextdir="$mofdir/$NEXT_STAGE"

  case "$stagedir" in
    */PBED3-PreRelax) ;;
    *)
      echo "ERROR $mofname: submit_prerelax.sh resolved non-prerelax stagedir: $stagedir"
      exit 1
      ;;
  esac

  jobname="${mofname}_${SUBMIT_STAGE}"
  next_jobname="${mofname}_${NEXT_STAGE}"
  outcar="$stagedir/OUTCAR"
  contcar="$stagedir/CONTCAR"
  poscar="$stagedir/POSCAR"
  poscar_bak="$stagedir/POSCAR.bak_before_maxstep_contcar"

  if [ ! -d "$stagedir" ]; then
    echo "SKIP $mofname: no $SUBMIT_STAGE folder"
    continue
  fi

  if job_already_in_queue "$jobname" "$stagedir"; then
    echo "SKIP $mofname: job already in queue or running for this folder"
    continue
  fi

  if job_already_in_queue "$next_jobname" "$nextdir"; then
    echo "SKIP $mofname: $NEXT_STAGE job already in queue or running"
    continue
  fi

  next_work_file="$(next_stage_has_work "$nextdir")"
  if [ -n "$next_work_file" ]; then
    echo "SKIP $mofname: $NEXT_STAGE already has non-empty $next_work_file"
    continue
  fi

  has_timing=0
  has_maxiter=0
  if [ -e "$outcar" ] && [ -s "$outcar" ]; then
    if has_timing_block "$outcar"; then
      has_timing=1
    fi
    if has_max_step_iteration "$outcar"; then
      has_maxiter=1
    fi
  fi

  if [ "$has_timing" -eq 1 ] && [ "$has_maxiter" -eq 0 ]; then
    echo "SKIP $mofname: successful completion (timing present, no Iteration $MAX_STEP_ITER)"
    continue
  fi

  max_step=0
  if [ "$has_timing" -eq 1 ] && [ "$has_maxiter" -eq 1 ]; then
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
      echo "SKIP $mofname: missing/empty $f at $stagedir/$f"
      continue 2
    fi
  done

  if [ "$max_step" -eq 1 ]; then
    if [ "$contcar" -ef "$poscar" ]; then
      echo "SKIP $mofname: CONTCAR and POSCAR are the same file; unsafe to mv"
      continue
    fi
  fi

  n_scripts=$(find "$stagedir" -maxdepth 1 -type f -name "*.sh" | wc -l)

  if [ "$n_scripts" -ne 1 ]; then
    echo "SKIP $mofname: expected 1 .sh file, found $n_scripts"
    find "$stagedir" -maxdepth 1 -type f -name "*.sh"
    continue
  fi

  script=$(find "$stagedir" -maxdepth 1 -type f -name "*.sh" | head -n 1)

  if [ "$max_step" -eq 1 ]; then
    contcar_bytes="$(wc -c <"$contcar" | tr -d ' ')"
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "BURN $mofname: would backup POSCAR, move CONTCAR -> POSCAR, then submit $jobname"
      continue
    fi
    if ! cp "$poscar" "$poscar_bak"; then
      echo "SKIP $mofname: backup POSCAR to POSCAR.bak_before_maxstep_contcar failed; not restarting"
      continue
    fi
    if ! mv "$contcar" "$poscar"; then
      echo "SKIP $mofname: mv CONTCAR to POSCAR failed"
      cp "$poscar_bak" "$poscar" 2>/dev/null || true
      continue
    fi
    if [ -e "$contcar" ] || [ ! -s "$poscar" ]; then
      echo "SKIP $mofname: after mv, CONTCAR path still exists or POSCAR empty; restoring POSCAR from backup"
      cp "$poscar_bak" "$poscar" 2>/dev/null || true
      continue
    fi
    poscar_bytes="$(wc -c <"$poscar" | tr -d ' ')"
    if [ "$poscar_bytes" != "$contcar_bytes" ]; then
      echo "SKIP $mofname: after mv, POSCAR size ($poscar_bytes) != CONTCAR size before mv ($contcar_bytes); restoring POSCAR from backup"
      cp "$poscar_bak" "$poscar" 2>/dev/null || true
      continue
    fi
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "BURN $mofname: would submit $jobname"
  else
    echo "SUBMIT $jobname"
    sbatch --job-name="$jobname" --chdir="$stagedir" "$script"
  fi
done
