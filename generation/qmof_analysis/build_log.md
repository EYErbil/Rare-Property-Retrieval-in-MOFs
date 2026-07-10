# QMOF bb-dir / topo-dir build log

- topo_out = `<REPO_ROOT>/qmof_topo_dir`
- bb_out   = `<REPO_ROOT>/qmof_bb_dir`

## Topology copy
- Requested: 52
- Copied:    42
- Missing from PORMAKE: 10

### Missing topology codes
- `sql`
- `hcb`
- `kgd`
- `fes`
- `hxl`
- `bey`
- `bex`
- `kgm`
- `met`
- `cpr`

## Building-block filtering
- Kept metal nodes:  506
- Skipped nodes (metal not whitelisted / no metal): 142
- Kept organic edges: 219

## Linker augmentation
- Added: 20
- Failed: 18
- Skipped: 0
