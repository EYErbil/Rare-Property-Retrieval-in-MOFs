#!/bin/bash



ROOT="${ROOT:-/path/to/DFT_WORK_ROOT}"
SOURCE_STAGE="${SOURCE_STAGE:-PBED3-Relax}"
TARGET_STAGE="${TARGET_STAGE:-PBED3-Single}"
SOURCE_MAX_STEP_ITER="${SOURCE_MAX_STEP_ITER:-249}"
TARGET_MAX_NELM_ITER="${TARGET_MAX_NELM_ITER:-150}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCAR_TEMPLATE="$SCRIPT_DIR/INCAR"

if [ ! -s "$INCAR_TEMPLATE" ]; then
  echo "ERROR: missing INCAR template: $INCAR_TEMPLATE"
  exit 1
fi

has_timing_block() {
  outcar="$1"
  [ -s "$outcar" ] && grep -q "General timing and accounting informations" "$outcar"
}

has_max_step_iteration() {
  outcar="$1"
  max_step_iter="$2"
  grep -Eq "Iteration[[:space:]]+$max_step_iter[[:space:]]*\\(" "$outcar"
}

has_max_nelm_iteration() {
  outcar="$1"
  max_nelm_iter="$2"
  grep -Eq "Iteration[[:space:]]+[0-9]+[[:space:]]*\\([[:space:]]*$max_nelm_iter[[:space:]]*\\)" "$outcar"
}

is_relax_successfully_complete() {
  outcar="$1"
  max_step_iter="$2"
  has_timing_block "$outcar" && ! has_max_step_iteration "$outcar" "$max_step_iter"
}

is_single_successfully_complete() {
  outcar="$1"
  max_nelm_iter="$2"
  has_timing_block "$outcar" && ! has_max_nelm_iteration "$outcar" "$max_nelm_iter"
}

mof_already_in_queue() {
  mofname="$1"
  source="$2"
  target="$3"

  if ! command -v squeue >/dev/null 2>&1; then
    return 1
  fi

  if squeue -h -u "$USER" -o "%j" 2>/dev/null | awk -v mof="$mofname" '$0 == mof || index($0, mof "_") == 1 { found = 1 } END { exit !found }'; then
    return 0
  fi

  if squeue -h -u "$USER" -o "%Z" 2>/dev/null | grep -Fxq "$source"; then
    return 0
  fi

  if squeue -h -u "$USER" -o "%Z" 2>/dev/null | grep -Fxq "$target"; then
    return 0
  fi

  return 1
}

copy_required() {
  src="$1"
  dst="$2"

  if [ -s "$src" ]; then
    cp "$src" "$dst"
    echo "  COPIED: $src -> $dst"
  else
    echo "  WARNING missing/empty: $src"
  fi
}

copy_magmom_from_source_incar() {
  src_incar="$1"
  dst_incar="$2"

  if [ ! -s "$src_incar" ]; then
    echo "  WARNING missing/empty source INCAR for MAGMOM: $src_incar"
    return
  fi

  magmom_line="$(grep -m 1 '^[[:space:]]*MAGMOM[[:space:]]*=' "$src_incar" || true)"
  if [ -z "$magmom_line" ]; then
    echo "  WARNING no MAGMOM line found in source INCAR: $src_incar"
    return
  fi

  if grep -q '^[[:space:]]*MAGMOM[[:space:]]*=' "$dst_incar"; then
    escaped_magmom_line="$(printf '%s\n' "$magmom_line" | sed 's/[\/&]/\\&/g')"
    sed -i "0,/^[[:space:]]*MAGMOM[[:space:]]*=.*/s//$escaped_magmom_line/" "$dst_incar"
  else
    printf '\n%s\n' "$magmom_line" >> "$dst_incar"
  fi

  echo "  COPIED MAGMOM: $src_incar -> $dst_incar"
}

find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "copy" | sort | while IFS= read -r mofdir; do
  mofname="$(basename "$mofdir")"
  relax="$mofdir/$SOURCE_STAGE"
  single="$mofdir/$TARGET_STAGE"

  echo "=============================="
  echo "MOF: $mofname"

  if mof_already_in_queue "$mofname" "$relax" "$single"; then
    echo "  SKIP: job already in queue/running for this MOF"
    continue
  fi

  if [ ! -d "$relax" ]; then
    echo "  SKIP: no $SOURCE_STAGE folder"
    continue
  fi

  if ! is_relax_successfully_complete "$relax/OUTCAR" "$SOURCE_MAX_STEP_ITER"; then
    echo "  SKIP: $SOURCE_STAGE not complete by status logic (timing present, no Iteration $SOURCE_MAX_STEP_ITER)"
    continue
  fi

  if is_single_successfully_complete "$single/OUTCAR" "$TARGET_MAX_NELM_ITER"; then
    echo "  SKIP: $TARGET_STAGE already complete by status logic; not copying from $SOURCE_STAGE again"
    continue
  fi

  if [ ! -s "$relax/CONTCAR" ]; then
    echo "  SKIP: $SOURCE_STAGE has no non-empty CONTCAR"
    continue
  fi

  mkdir -p "$single"

  copy_required "$relax/CONTCAR" "$single/POSCAR"
  copy_required "$relax/CHGCAR"  "$single/CHGCAR"
  copy_required "$relax/WAVECAR" "$single/WAVECAR"
  copy_required "$relax/KPOINTS" "$single/KPOINTS"
  copy_required "$relax/POTCAR"  "$single/POTCAR"

  for shfile in "$relax"/*.sh; do
    if [ -f "$shfile" ]; then
      cp "$shfile" "$single/$(basename "$shfile")"
      echo "  COPIED SH: $shfile -> $single/$(basename "$shfile")"
    fi
  done

  if [ -f "$single/INCAR" ]; then
    cp "$single/INCAR" "$single/INCAR.bak_before_pbe_single_copy"
  fi

  cp "$INCAR_TEMPLATE" "$single/INCAR"
  echo "  COPIED INCAR TEMPLATE: $INCAR_TEMPLATE -> $single/INCAR"
  copy_magmom_from_source_incar "$relax/INCAR" "$single/INCAR"
done
