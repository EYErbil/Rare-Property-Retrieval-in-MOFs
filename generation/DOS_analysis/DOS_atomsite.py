from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

VASPRUN_FILE = "vasprun.xml"

XMIN = -1.2
EXTRA_AFTER_CBM = 1.5

BAND_EDGE_HALF_WIDTH = 0.15
SMOOTH_SIGMA_POINTS = 2.0

TOP_N_PRINT = 20
TOP_N_PLOTS = 10

OUTPUT_DIR = Path("site_localization_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# READ DATA
# ============================================================

print(f"Reading {VASPRUN_FILE} ...")

vr = Vasprun(
    VASPRUN_FILE,
    parse_dos=True,
    parse_projected_eigen=True
)

cdos = vr.complete_dos
structure = vr.final_structure

band_gap, cbm, vbm, is_direct = vr.eigenvalue_band_properties

energies = cdos.energies - vbm
XMAX = band_gap + EXTRA_AFTER_CBM

print("\n=== Band info ===")
print(f"Band gap:   {band_gap:.6f} eV")
print(f"VBM:        {vbm:.6f} eV")
print(f"CBM:        {cbm:.6f} eV")
print(f"Direct gap: {is_direct}")
print(f"Atoms:      {len(structure)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def smooth(y, sigma_points=0):
    if sigma_points is None or sigma_points <= 0:
        return y

    radius = int(4 * sigma_points)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma_points**2))
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def mirrored_spin(dos_obj):
    """
    Returns spin-up positive and spin-down negative for plotting.
    For non-spin-polarized runs, spin-down is zero.
    """
    up = np.zeros_like(energies)
    down = np.zeros_like(energies)

    if dos_obj is None:
        return up, down

    if Spin.up in dos_obj.densities:
        up = np.array(dos_obj.densities[Spin.up], dtype=float)

    if Spin.down in dos_obj.densities:
        down = -np.array(dos_obj.densities[Spin.down], dtype=float)

    up = smooth(up, SMOOTH_SIGMA_POINTS)
    down = smooth(down, SMOOTH_SIGMA_POINTS)

    return up, down


def edge_weight(up, down, center):
    """
    Integrated absolute DOS weight inside an energy window.
    """
    mask = (energies >= center - BAND_EDGE_HALF_WIDTH) & (
        energies <= center + BAND_EDGE_HALF_WIDTH
    )

    return float(np.sum(np.abs(up[mask])) + np.sum(np.abs(down[mask])))


def get_site_total_dos(site):
    try:
        dos_obj = cdos.get_site_dos(site)
        return mirrored_spin(dos_obj)
    except Exception:
        return np.zeros_like(energies), np.zeros_like(energies)


def get_site_spd_dos(site):
    """
    Returns:
    {
        "s": (up, down),
        "p": (up, down),
        "d": (up, down),
        ...
    }
    """
    out = {}

    try:
        spd = cdos.get_site_spd_dos(site)
    except Exception:
        return out

    for orb_type, dos_obj in spd.items():
        key = str(orb_type).lower()

        if key.endswith(".s") or key == "s":
            label = "s"
        elif key.endswith(".p") or key == "p":
            label = "p"
        elif key.endswith(".d") or key == "d":
            label = "d"
        elif key.endswith(".f") or key == "f":
            label = "f"
        else:
            label = str(orb_type)

        out[label] = mirrored_spin(dos_obj)

    return out


def localization_metrics(weights):
    """
    weights: list/array of nonnegative site weights.
    Returns IPR-like localization diagnostics.

    If one atom dominates, max_fraction is large and N_eff is small.
    If many atoms contribute, max_fraction is small and N_eff is large.
    """
    weights = np.array(weights, dtype=float)
    total = weights.sum()

    if total <= 1e-14:
        return {
            "total_weight": total,
            "max_fraction": 0.0,
            "top3_fraction": 0.0,
            "top5_fraction": 0.0,
            "ipr": 0.0,
            "n_eff": 0.0,
        }

    p = weights / total
    p_sorted = np.sort(p)[::-1]

    ipr = float(np.sum(p**2))
    n_eff = float(1.0 / ipr) if ipr > 1e-14 else 0.0

    return {
        "total_weight": float(total),
        "max_fraction": float(p_sorted[0]),
        "top3_fraction": float(p_sorted[:3].sum()),
        "top5_fraction": float(p_sorted[:5].sum()),
        "ipr": ipr,
        "n_eff": n_eff,
    }


def style_axis(ax):
    ax.axvline(0, linestyle="--", linewidth=1.5, color="#1F77D0", alpha=0.9)
    ax.axvline(band_gap, linestyle="--", linewidth=1.5, color="#D9534F", alpha=0.9)
    ax.axhline(0, linestyle="-", linewidth=0.8, color="#666666", alpha=0.6)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)
    ax.set_xlim(XMIN, XMAX)


def autoscale(ax, arrays, margin=1.15):
    mask = (energies >= XMIN) & (energies <= XMAX)
    vals = []

    for arr in arrays:
        arr = np.asarray(arr)
        if arr.shape == energies.shape:
            vals.append(arr[mask])

    if not vals:
        ax.set_ylim(-1, 1)
        return

    vals = np.concatenate(vals)
    max_abs = np.max(np.abs(vals)) if vals.size else 1.0

    if max_abs < 1e-8:
        max_abs = 1.0

    ax.set_ylim(-margin * max_abs, margin * max_abs)


# ============================================================
# TOTAL DOS FOR REFERENCE
# ============================================================

total_up, total_down = mirrored_spin(cdos)

total_vbm_weight = edge_weight(total_up, total_down, 0.0)
total_cbm_weight = edge_weight(total_up, total_down, band_gap)

print("\n=== Total DOS window weights ===")
print(f"Total near VBM: {total_vbm_weight:.6f}")
print(f"Total near CBM: {total_cbm_weight:.6f}")


# ============================================================
# SITE-PROJECTED RANKING FOR ALL ATOMS
# ============================================================

site_rows = []
site_curves = {}

for atom_number, site in enumerate(structure, start=1):
    elem = site.species_string

    up, down = get_site_total_dos(site)

    vbm_w = edge_weight(up, down, 0.0)
    cbm_w = edge_weight(up, down, band_gap)

    site_rows.append({
        "atom": atom_number,
        "element": elem,
        "vbm_weight": vbm_w,
        "cbm_weight": cbm_w,
        "frac_a": site.frac_coords[0],
        "frac_b": site.frac_coords[1],
        "frac_c": site.frac_coords[2],
    })

    site_curves[atom_number] = {
        "element": elem,
        "up": up,
        "down": down,
    }

df = pd.DataFrame(site_rows)

# Normalize by sum of all site-projected weights.
# This is better for localization than comparing directly to total DOS,
# because PAW site projections do not always sum exactly to total DOS.
vbm_site_sum = df["vbm_weight"].sum()
cbm_site_sum = df["cbm_weight"].sum()

df["vbm_pct_of_site_projected"] = (
    100.0 * df["vbm_weight"] / vbm_site_sum if vbm_site_sum > 1e-14 else 0.0
)
df["cbm_pct_of_site_projected"] = (
    100.0 * df["cbm_weight"] / cbm_site_sum if cbm_site_sum > 1e-14 else 0.0
)

# Also report percentage vs total DOS window, but this may not sum to 100.
df["vbm_pct_vs_total_dos"] = (
    100.0 * df["vbm_weight"] / total_vbm_weight if total_vbm_weight > 1e-14 else 0.0
)
df["cbm_pct_vs_total_dos"] = (
    100.0 * df["cbm_weight"] / total_cbm_weight if total_cbm_weight > 1e-14 else 0.0
)

df_sorted_vbm = df.sort_values("vbm_pct_of_site_projected", ascending=False)
df_sorted_cbm = df.sort_values("cbm_pct_of_site_projected", ascending=False)

df.to_csv(OUTPUT_DIR / "all_sites_band_edge_weights.csv", index=False)
df_sorted_vbm.to_csv(OUTPUT_DIR / "ranked_by_vbm_site_contribution.csv", index=False)
df_sorted_cbm.to_csv(OUTPUT_DIR / "ranked_by_cbm_site_contribution.csv", index=False)

print("\n=== Top atoms near VBM ===")
print(
    df_sorted_vbm[
        ["atom", "element", "vbm_weight", "vbm_pct_of_site_projected", "vbm_pct_vs_total_dos"]
    ].head(TOP_N_PRINT).to_string(index=False)
)

print("\n=== Top atoms near CBM ===")
print(
    df_sorted_cbm[
        ["atom", "element", "cbm_weight", "cbm_pct_of_site_projected", "cbm_pct_vs_total_dos"]
    ].head(TOP_N_PRINT).to_string(index=False)
)


# ============================================================
# LOCALIZATION METRICS
# ============================================================

vbm_loc = localization_metrics(df["vbm_weight"].values)
cbm_loc = localization_metrics(df["cbm_weight"].values)

loc_df = pd.DataFrame([
    {"edge": "VBM", **vbm_loc},
    {"edge": "CBM", **cbm_loc},
])

loc_df.to_csv(OUTPUT_DIR / "localization_metrics.csv", index=False)

print("\n=== Localization metrics ===")
print(loc_df.to_string(index=False))

print("\nInterpretation:")
print("  max_fraction: fraction of edge weight on the single strongest atom")
print("  top3_fraction: fraction on top 3 atoms")
print("  n_eff: effective number of atoms carrying the edge state")
print("  Smaller n_eff = more localized")
print("  Larger n_eff = more delocalized")


# ============================================================
# ORBITAL-RESOLVED SITE RANKING
# ============================================================

orbital_rows = []

for atom_number, site in enumerate(structure, start=1):
    elem = site.species_string

    spd = get_site_spd_dos(site)

    for orb, (up, down) in spd.items():
        vbm_w = edge_weight(up, down, 0.0)
        cbm_w = edge_weight(up, down, band_gap)

        orbital_rows.append({
            "atom": atom_number,
            "element": elem,
            "orbital": orb,
            "vbm_weight": vbm_w,
            "cbm_weight": cbm_w,
        })

orb_df = pd.DataFrame(orbital_rows)

if len(orb_df) > 0:
    orb_df["vbm_pct_of_site_projected"] = (
        100.0 * orb_df["vbm_weight"] / vbm_site_sum if vbm_site_sum > 1e-14 else 0.0
    )
    orb_df["cbm_pct_of_site_projected"] = (
        100.0 * orb_df["cbm_weight"] / cbm_site_sum if cbm_site_sum > 1e-14 else 0.0
    )

    orb_df.to_csv(OUTPUT_DIR / "all_site_orbital_band_edge_weights.csv", index=False)

    print("\n=== Top site-orbital contributors near VBM ===")
    print(
        orb_df.sort_values("vbm_pct_of_site_projected", ascending=False)
        [["atom", "element", "orbital", "vbm_weight", "vbm_pct_of_site_projected"]]
        .head(TOP_N_PRINT)
        .to_string(index=False)
    )

    print("\n=== Top site-orbital contributors near CBM ===")
    print(
        orb_df.sort_values("cbm_pct_of_site_projected", ascending=False)
        [["atom", "element", "orbital", "cbm_weight", "cbm_pct_of_site_projected"]]
        .head(TOP_N_PRINT)
        .to_string(index=False)
    )


# ============================================================
# PLOT TOP CONTRIBUTOR ATOMS AUTOMATICALLY
# ============================================================

top_vbm_atoms = list(df_sorted_vbm["atom"].head(TOP_N_PLOTS))
top_cbm_atoms = list(df_sorted_cbm["atom"].head(TOP_N_PLOTS))

top_atoms = []
for a in top_vbm_atoms + top_cbm_atoms:
    if a not in top_atoms:
        top_atoms.append(a)

print(f"\nPlotting top atoms: {top_atoms}")

for atom_number in top_atoms:
    row = df[df["atom"] == atom_number].iloc[0]
    elem = row["element"]

    up = site_curves[atom_number]["up"]
    down = site_curves[atom_number]["down"]

    fig, ax = plt.subplots(figsize=(6.2, 3.7))

    ax.plot(energies, up, color="#222222", linewidth=2.0, label=f"Atom {atom_number} {elem} up")
    ax.plot(energies, down, color="#222222", linewidth=2.0, linestyle="--", label=f"Atom {atom_number} {elem} down")

    style_axis(ax)
    autoscale(ax, [up, down])

    ax.set_title(f"Site DOS: atom {atom_number} {elem}")
    ax.set_xlabel("Energy - VBM (eV)")
    ax.set_ylabel("DOS")

    ylim = ax.get_ylim()
    ax.text(0, ylim[1] * 0.86, "VBM", color="#1F77D0", ha="center", fontsize=10, fontweight="bold")
    ax.text(band_gap, ylim[1] * 0.86, "CBM", color="#D9534F", ha="center", fontsize=10, fontweight="bold")

    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")

    out_png = OUTPUT_DIR / f"site_dos_atom_{atom_number}_{elem}.png"
    out_svg = OUTPUT_DIR / f"site_dos_atom_{atom_number}_{elem}.svg"

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

print("\nSaved outputs in:", OUTPUT_DIR.resolve())
print("Done.")