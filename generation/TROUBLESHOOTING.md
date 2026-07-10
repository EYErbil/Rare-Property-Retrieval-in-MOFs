# Troubleshooting and optional Slurm wrappers

Companion to [REPLICATION.md](REPLICATION.md) and [DFT_WORKFLOW.md](DFT_WORKFLOW.md).

---

## Appendix B — Optional Slurm wrappers

SOAP jobs:

```bash
cd REPO_ROOT
sbatch soap_analysis/Soap-analysis-compare.sh
sbatch soap_analysis/Soap-analysis-compare-cached.sh
```

Create `REPO_ROOT/logs/` before submitting jobs that write `%j.out` / `%j.err` under `logs/`.

DFT helper wrappers:

```bash
cd REPO_ROOT
bash scripts/Dft-After-nomination/copy/to_prerelax_step0/starter.sh
bash scripts/Dft-After-nomination/copy/mass_submit/submit_prerelax.sh
bash scripts/Dft-After-nomination/copy/mass_submit/status_prerelax.sh
bash scripts/Dft-After-nomination/copy/to_relax_step1/copy_completed_prerelax_to_relax.sh
bash scripts/Dft-After-nomination/copy/to_pbe_single_Step2/copy_completed_relax_to_pbe_single.sh
bash scripts/Dft-After-nomination/copy/to_hse_step3/copy_completed_pbe_single_to_hse_single.sh
bash scripts/Dft-After-nomination/magmom_manager/magmom_manager_before_prerelax.sh
bash scripts/Dft-After-nomination/magmom_manager/magmom_manager_before_relax_from_prerelax.sh
bash scripts/Dft-After-nomination/magmom_manager/magmom_manager_beforePBE-single.sh
bash scripts/Dft-After-nomination/magmom_manager/magmom_manager_beforeHSE.sh
bash scripts/Dft-After-nomination/copy/mass_submit/status_relax.sh
bash scripts/Dft-After-nomination/copy/mass_submit/mass_submit_better_relax.sh
bash scripts/Dft-After-nomination/copy/mass_submit/submit_pbe_single.sh
bash scripts/Dft-After-nomination/copy/mass_submit/submit_hse_single.sh
```

Before running any DFT wrapper, edit hard-coded paths and review module, account, partition, and job-script assumptions for your cluster. Prefer the Python entry points in Steps 13-17 for portable CIF/POSCAR/KPOINTS/MAGMOM preparation.

---

## Appendix C — Troubleshooting

| Symptom | What to do |
|---------|------------|
| `qmof.csv` not found | Place QMOF release CSV at `REPO_ROOT/qmof.csv` before Step 1. |
| CIF build OOM | Lower `--chunk-size` in `build_materials_batched.py`. |
| SOAP / UMAP OOM | Use `--qmof-cache` + `--generated-cache`; enable PCA defaults; avoid `--full-soap-umap` on large dims without huge RAM. |
| Split figures show zero matches | Check `--labeled-splits-dir` and **ID strings** vs embedding/QMOF IDs. |
| PMTransformer compare recomputes generated forever | Pass **`--generated-cache`** pointing at existing `generated_pmt_embeddings.npz`. |
| `downstream` / JSON name mismatch | Align `prepare_moftransformer_test_only.py --downstream` with `compare_generated_vs_qmof.py --downstream` and NN inference (`--downstream auto` helps). |
| `TypeError: asarray() ... copy` | Upgrade NumPy or use current `compare_generated_vs_qmof.py` from this repo. |
| NN inference cannot import `train_regressor` | Point `--base_dir` at the project that **contains** `train_regressor.py` (not necessarily `REPO_ROOT`). |
| ML re-inference finds no methods or writes to wrong place | Pass explicit **`--clf_dir`** to your embedding-classifier root; omit **`--output_dir`** only if you intend to overwrite files inside each method folder. |
| `FileNotFoundError` after `--npz_dir` | Use **`--embeddings_path`** to an explicit `.npz`, or ensure `<npz_dir>/pmt_embeddings_qmof_unlabeled.npz` exists (see `python scripts/re_inference/reinfer_ml.py --help`). |
| GRIDAY errors | `moftransformer install-griday` in the active environment. |
| Step 13: nominated CIF not copied | Confirm `FINAL_TOP25_diverse.txt` / `COMBINED_top25.txt` lists exact CIF stems; `select_cifs_from_list.py` exits nonzero on missing or ambiguous partial matches. |
| Step 14: CIF to POSCAR conversion fails | Inspect `conversion_report.csv`; malformed CIFs and occupancy issues are raised by `pymatgen.CifParser` (kept faithful — no auto-fix). |
| Step 15: validation `[FAIL]` rows | Look at `same_formula` / `same_num_sites` first, then lattice and fractional-coord tolerances. The validator assumes preserved atom order between CIF and POSCAR. |
| Step 17: `Kpoint-generator.sh` uses pinned absolute paths | It already calls `kpoint_maker.py`; edit its `--root`/`--report` to your `DFT_WORK_ROOT`, or run `python scripts/Dft-After-nomination/kpoint_maker.py --root DFT_WORK_ROOT ...` directly. |
| Step 17: KPOINTS appear in unexpected folders | `kpoint_maker.py` recursively writes `KPOINTS` next to **every** `POSCAR` under `--root`; aim `--root` at only the intended tree (run it once before Step 21). |
| Step 18: helper does nothing | The bash wrapper iterates `ROOT`'s immediate children excluding `copy/`; edit `ROOT="..."` to your `DFT_WORK_ROOT`, and ensure each MOF has a `PBED3-PreRelax` subfolder already. |
| Step 19: MAGMOM manager rejects POSCAR | `vasp_magmom_manager.py` requires VASP5 POSCAR with element symbols on line 6; old VASP4 count-only POSCARs are explicitly rejected. |
| Step 19 / 19d / 22 / 25: MAGMOM wrapper fails on Python 3.6 | This repo's manager avoids `argparse.add_subparsers(required=True)` for Python 3.6 compatibility. The wrappers call `python3`; edit the command if your cluster needs a specific interpreter path. |
| Steps 22 / 25: extract fails with ion-count mismatch | The OUTCAR moment count must equal the number of atoms in `target_stage/POSCAR`. Make sure Step 21 / Step 24 ran before Step 22 / Step 25, and that `CONTCAR -> POSCAR` actually copied. |
| Steps 20 / 23 / 26: submit helper skips a folder for missing files | Ensure the stage folder has non-empty `INCAR`, `POSCAR`, `POTCAR`, `KPOINTS` and exactly one `.sh` job script. |
| Steps 20 / 23 / 26: submit helper skips a folder with `AECCAR0` | `AECCAR0` plus an incomplete `OUTCAR` is treated as a manual-inspection case before any resubmission; clean up the folder yourself. |
| Steps 20 / 23 / 26: submit helper does not detect a finished job | Completion is detected by the literal line `General timing and accounting informations` in `OUTCAR`; jobs killed before that point are re-submitted. |

