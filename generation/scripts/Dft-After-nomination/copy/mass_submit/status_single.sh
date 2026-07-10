#!/bin/bash

# Compact status report for singlepoint OUTCAR files.
# Completion logic matches the singlepoint copiers and submitters:
# timing present and no electronic Iteration *(MAX_NELM_ITER).
#
# Usage:
#   bash status_single.sh
#   ROOT=/scratch/.../Generated-MOFs STAGE=PBED3-Single MAX_NELM_ITER=150 bash status_single.sh
#   ROOT=/scratch/.../Generated-MOFs STAGE=HSE-single MAX_NELM_ITER=100 bash status_single.sh

set -u

ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"
STAGE="${STAGE:-PBED3-Single}"

if [[ -z "${MAX_NELM_ITER:-}" ]]; then
  case "$STAGE" in
    HSE-single) MAX_NELM_ITER=100 ;;
    *) MAX_NELM_ITER=150 ;;
  esac
fi

has_timing_block() {
  local outcar="$1"
  [[ -s "$outcar" ]] && grep -q "General timing and accounting informations" "$outcar"
}

is_max_nelm_ended() {
  local outcar="$1"
  # Singlepoint geometry does not step, so VASP reports electronic iterations as:
  # --------------------------------------- Iteration      1(   9)  ---------------------------------------
  # A line ending at NELM means electronic max iteration was reached.
  grep -Eq "Iteration[[:space:]]+[0-9]+[[:space:]]*\\([[:space:]]*$MAX_NELM_ITER[[:space:]]*\\)" "$outcar"
}

has_small_distance_warning() {
  local outcar="$1"
  grep -q "The distance between some ions is very small" "$outcar"
}

outcar_flags_summary() {
  local outcar="$1"
  local timing=0 mem=0 nelm=0 bytes=0
  if [[ -e "$outcar" ]]; then
    bytes="$(wc -c <"$outcar" 2>/dev/null | tr -d ' ')"
  fi
  if [[ -s "$outcar" ]]; then
    grep -q "General timing and accounting informations" "$outcar" && timing=1
    grep -q "total amount of memory used by VASP MPI-rank0" "$outcar" && mem=1
    grep -Eq "Iteration[[:space:]]+[0-9]+[[:space:]]*\\([[:space:]]*$MAX_NELM_ITER[[:space:]]*\\)" "$outcar" && nelm=1
  fi
  echo "timing=$timing memline=$mem maxnelm=$nelm max_nelm_iter=$MAX_NELM_ITER bytes=$bytes"
}

job_already_in_queue() {
  local jobname="$1"
  local stagedir="$2"

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

print_list_uncapped() {
  local title="$1"
  shift
  local items=("$@")
  local n="${#items[@]}"
  echo
  echo "$title ($n total, uncapped):"
  if (( n == 0 )); then
    echo "  (none)"
    return
  fi
  for x in "${items[@]}"; do
    echo "  - $x"
  done
}

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: ROOT not found: $ROOT"
  exit 1
fi

total_mofs=0
with_stage=0
in_queue_skip=0
no_outcar=0
empty_outcar=0
successful_complete=0
nelm_limit_reached=0
crash_bucket=0
maxnelm_without_timing=0
resubmit_total=0
small_dist_warn=0

warn_list=()
in_queue_list=()
successful_complete_list=()
successful_complete_diag=()
nelm_limit_list=()
crash_bucket_list=()
maxnelm_only_list=()
no_outcar_list=()
empty_outcar_list=()

while IFS= read -r mofdir; do
  ((total_mofs+=1))
  mofname="$(basename "$mofdir")"
  stagedir="$mofdir/$STAGE"
  outcar="$stagedir/OUTCAR"
  jobname="${mofname}_${STAGE}"

  if [[ ! -d "$stagedir" ]]; then
    continue
  fi
  ((with_stage+=1))

  if [[ -e "$outcar" && -s "$outcar" ]] && has_small_distance_warning "$outcar"; then
    ((small_dist_warn+=1))
    warn_list+=("$mofname")
  fi

  if job_already_in_queue "$jobname" "$stagedir"; then
    ((in_queue_skip+=1))
    in_queue_list+=("$mofname")
    continue
  fi

  has_timing=0
  has_maxnelm=0
  if [[ -e "$outcar" && -s "$outcar" ]]; then
    if has_timing_block "$outcar"; then
      has_timing=1
    fi
    if is_max_nelm_ended "$outcar"; then
      has_maxnelm=1
    fi
  fi

  if [[ "$has_timing" -eq 1 && "$has_maxnelm" -eq 1 ]]; then
    ((nelm_limit_reached+=1))
    ((resubmit_total+=1))
    nelm_limit_list+=("$mofname")
  elif [[ "$has_timing" -eq 1 && "$has_maxnelm" -eq 0 ]]; then
    ((successful_complete+=1))
    successful_complete_list+=("$mofname")
    successful_complete_diag+=("$(outcar_flags_summary "$outcar")")
  elif [[ "$has_timing" -eq 0 && "$has_maxnelm" -eq 0 ]]; then
    ((crash_bucket+=1))
    ((resubmit_total+=1))
    crash_bucket_list+=("$mofname")
    if [[ ! -e "$outcar" ]]; then
      ((no_outcar+=1))
      no_outcar_list+=("$mofname")
    elif [[ ! -s "$outcar" ]]; then
      ((empty_outcar+=1))
      empty_outcar_list+=("$mofname")
    fi
  else
    ((maxnelm_without_timing+=1))
    ((resubmit_total+=1))
    maxnelm_only_list+=("$mofname")
  fi
done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "copy" | sort)

echo "============================================================"
echo "Singlepoint Status Summary"
echo "============================================================"
echo "ROOT              : $ROOT"
echo "STAGE             : $STAGE"
echo "Max-NELM Iteration: $MAX_NELM_ITER"
echo "Total MOF folders : $total_mofs"
echo "With stage folder : $with_stage"
echo
echo "Queue-first gate:"
echo "  In squeue (skip duplicate submit) : $in_queue_skip"
echo
echo "When NOT in squeue - OUTCAR flags:"
echo "  Successful completion"
echo "    (timing present, no electronic Iteration $MAX_NELM_ITER)    : $successful_complete"
echo "  NELM limit reached - inspect/resubmit"
echo "    (timing present + electronic Iteration $MAX_NELM_ITER)      : $nelm_limit_reached"
echo "  Crash/unfinished bucket - inspect/resubmit"
echo "    (no timing, no electronic Iteration $MAX_NELM_ITER)         : $crash_bucket"
echo "  Electronic Iteration $MAX_NELM_ITER without timing            : $maxnelm_without_timing"
echo "  -------------------------------------------------"
echo "  Inspect/resubmit targets (sum of last three lines): $resubmit_total"
echo
echo "Extra diagnostics:"
echo "  No OUTCAR (subset of crash bucket)               : $no_outcar"
echo "  Empty OUTCAR (subset of crash bucket)            : $empty_outcar"
echo "  Small-distance warnings (all staged OUTCARs)     : $small_dist_warn"
echo "============================================================"

print_list_uncapped "In squeue (skip)" "${in_queue_list[@]}"
print_list_uncapped "Successful completion (timing yes, electronic Iteration $MAX_NELM_ITER no)" "${successful_complete_list[@]}"
print_list_uncapped "NELM limit reached - inspect/resubmit (timing + electronic Iteration $MAX_NELM_ITER)" "${nelm_limit_list[@]}"
print_list_uncapped "Crash/unfinished bucket - inspect/resubmit (no timing, no electronic Iteration $MAX_NELM_ITER)" "${crash_bucket_list[@]}"
print_list_uncapped "Electronic Iteration $MAX_NELM_ITER without timing - inspect/resubmit" "${maxnelm_only_list[@]}"
print_list_uncapped "No OUTCAR (under crash bucket)" "${no_outcar_list[@]}"
print_list_uncapped "Empty OUTCAR (under crash bucket)" "${empty_outcar_list[@]}"

echo
echo "ALL small-distance-warning MOFs ($small_dist_warn total, uncapped):"
if (( small_dist_warn == 0 )); then
  echo "  (none)"
else
  for x in "${warn_list[@]}"; do
    echo "  - $x"
  done
fi

echo
echo "Successful completion - OUTCAR flags (reference only):"
echo "  (memline = MPI memory text; can be 1 mid-OUTCAR - not used for classification)"
if (( successful_complete == 0 )); then
  echo "  (none)"
else
  for ((i = 0; i < ${#successful_complete_list[@]}; i++)); do
    echo "  - ${successful_complete_list[$i]}  (${successful_complete_diag[$i]:-diag_missing})"
  done
fi
