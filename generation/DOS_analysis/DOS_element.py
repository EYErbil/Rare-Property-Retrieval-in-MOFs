from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin, OrbitalType
from pymatgen.core import Element
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path


# ============================================================
# USER SETTINGS
# ============================================================

VASPRUN_FILE = "vasprun.xml"

# Energy window: VBM = 0 eV, CBM = band_gap
XMIN = -1.2
EXTRA_AFTER_CBM = 1.5

# Smoothing in number of energy-grid points
SMOOTH_SIGMA_POINTS = 2.0

# Band-edge window for contribution summary
BAND_EDGE_HALF_WIDTH = 0.15

# Output prefix
OUTPUT_PREFIX = "auto_pdos"


# ============================================================
# COLORS
# ============================================================

NAVY   = "#0B2A66"
BLUE   = "#1F77D0"
CYAN   = "#2EC4D6"
PURPLE = "#8E5BD9"
TEAL   = "#2A9D8F"
GREEN  = "#4CAF50"
RED    = "#D9534F"
GRAY   = "#666666"
BLACK  = "#222222"

COLOR_CYCLE = [
    NAVY, BLUE, CYAN, PURPLE, TEAL, GREEN, RED,
    "#6A51A3", "#D65DB1", "#FFB000", "#8C564B",
    "#7F7F7F", "#17BECF", "#BCBD22"
]

ELEMENT_COLORS = {
    "H": "#BBBBBB",
    "C": "#222222",
    "N": "#4169E1",
    "O": "#D9534F",
    "S": "#FFB000",
    "P": "#FF7F0E",
    "F": "#2A9D8F",
    "Cl": "#4CAF50",
    "Br": "#8C564B",
    "I": "#6A51A3",
    "Mn": "#8E5BD9",
    "Fe": "#B22222",
    "Co": "#1F77B4",
    "Ni": "#2E8B57",
    "Cu": "#B87333",
    "Zn": "#708090",
    "Cd": "#9467BD",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.2,
    "font.family": "DejaVu Sans",
})


# ============================================================
# READ VASP DATA
# ============================================================

print(f"Reading {VASPRUN_FILE} ...")

vr = Vasprun(
    VASPRUN_FILE,
    parse_projected_eigen=True,
    parse_dos=True
)

cdos = vr.complete_dos

band_gap, cbm, vbm, is_direct = vr.eigenvalue_band_properties

print("\n=== Band information ===")
print(f"Band gap:    {band_gap:.6f} eV")
print(f"VBM:         {vbm:.6f} eV")
print(f"CBM:         {cbm:.6f} eV")
print(f"Direct gap?  {is_direct}")

energies = cdos.energies - vbm
XMAX = band_gap + EXTRA_AFTER_CBM


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


def get_spin_up_down(dos_obj):
    """
    Return spin-up and mirrored spin-down DOS.
    For non-spin-polarized calculations, spin-down is zero.
    """
    up = np.zeros_like(energies)
    down = np.zeros_like(energies)

    if dos_obj is None:
        return up, down

    if Spin.up in dos_obj.densities:
        up = dos_obj.densities[Spin.up]

    if Spin.down in dos_obj.densities:
        down = -dos_obj.densities[Spin.down]

    up = smooth(up, SMOOTH_SIGMA_POINTS)
    down = smooth(down, SMOOTH_SIGMA_POINTS)

    return up, down


def add_components(component_list):
    """
    Sum several (up, down) components.
    """
    up_total = np.zeros_like(energies)
    down_total = np.zeros_like(energies)

    for up, down in component_list:
        up_total += up
        down_total += down

    return up_total, down_total


def get_element_dos(element_dos_dict, symbol):
    """
    Robustly fetch element DOS whether keys are Element objects or strings.
    """
    target = Element(symbol)

    for key, dos_obj in element_dos_dict.items():
        if str(key) == symbol:
            return dos_obj
        if key == target:
            return dos_obj

    return None


def get_element_spd_dos_safe(cdos, symbol):
    """
    Return {orbital_type_string: dos_obj} for one element.
    Example keys: 's', 'p', 'd', 'f'
    """
    try:
        spd = cdos.get_element_spd_dos(Element(symbol))
    except Exception:
        try:
            spd = cdos.get_element_spd_dos(symbol)
        except Exception:
            return {}

    out = {}

    for orb_type, dos_obj in spd.items():
        key = str(orb_type).lower()

        # pymatgen may print OrbitalType.d or just d depending on version
        if "s" == key or key.endswith(".s"):
            out["s"] = dos_obj
        elif "p" == key or key.endswith(".p"):
            out["p"] = dos_obj
        elif "d" == key or key.endswith(".d"):
            out["d"] = dos_obj
        elif "f" == key or key.endswith(".f"):
            out["f"] = dos_obj

    return out


def window_sum(y, center, half_width=BAND_EDGE_HALF_WIDTH):
    mask = (energies >= center - half_width) & (energies <= center + half_width)
    return float(np.sum(np.abs(y[mask])))


def component_weight(up, down, center):
    return window_sum(up, center) + window_sum(down, center)


def style_axis(ax):
    ax.axvline(0, linestyle="--", linewidth=1.6, color=BLUE, alpha=0.95)
    ax.axvline(band_gap, linestyle="--", linewidth=1.6, color=RED, alpha=0.90)
    ax.axhline(0, linestyle="-", linewidth=0.8, color=GRAY, alpha=0.7)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)
    ax.set_xlim(XMIN, XMAX)


def autoscale_symmetric_y(ax, arrays, margin=1.15):
    mask = (energies >= XMIN) & (energies <= XMAX)

    visible_values = []

    for arr in arrays:
        arr = np.asarray(arr)
        if arr.shape[0] == energies.shape[0]:
            visible_values.append(arr[mask])

    if not visible_values:
        ax.set_ylim(-1, 1)
        return

    vals = np.concatenate(visible_values)
    max_abs = np.max(np.abs(vals)) if vals.size else 1.0

    if max_abs < 1e-8:
        max_abs = 1.0

    ax.set_ylim(-margin * max_abs, margin * max_abs)


def savefig(base):
    fig = plt.gcf()
    fig.subplots_adjust(hspace=0.14)
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    print(f"Saved {base}.png and {base}.svg")
    plt.close(fig)


def element_color(symbol, fallback_index=0):
    return ELEMENT_COLORS.get(symbol, COLOR_CYCLE[fallback_index % len(COLOR_CYCLE)])


def is_transition_or_d_metal(symbol):
    """
    For PDOS purposes, treat these as important metal centers.
    This avoids relying only on Element.is_metal, which includes alkali/alkaline earth metals.
    """
    d_metals = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
    }
    return symbol in d_metals


def is_halogen(symbol):
    return symbol in {"F", "Cl", "Br", "I"}


def is_linker_element(symbol):
    """
    Organic/linker elements commonly present in MOFs.
    H is included but can be excluded from plotting if too small.
    """
    return symbol in {"H", "B", "C", "N", "O", "P", "S", "Si", "Se"}


# ============================================================
# DETECT ELEMENTS AND BUILD DOS DICTIONARIES
# ============================================================

element_dos_raw = cdos.get_element_dos()

present_elements = sorted(
    [str(k) for k in element_dos_raw.keys()],
    key=lambda x: Element(x).Z
)

print("\n=== Elements detected from vasprun.xml ===")
print(present_elements)

element_total = {}
element_spd = {}

for sym in present_elements:
    dos_obj = get_element_dos(element_dos_raw, sym)
    up, down = get_spin_up_down(dos_obj)
    element_total[sym] = (up, down)
    element_spd[sym] = {}

    spd = get_element_spd_dos_safe(cdos, sym)
    for orb, dos_obj_orb in spd.items():
        element_spd[sym][orb] = get_spin_up_down(dos_obj_orb)


total_up, total_down = get_spin_up_down(cdos)

metal_elements = [s for s in present_elements if is_transition_or_d_metal(s)]
halogen_elements = [s for s in present_elements if is_halogen(s)]
linker_elements = [s for s in present_elements if is_linker_element(s) and not is_halogen(s)]
other_elements = [
    s for s in present_elements
    if s not in metal_elements and s not in halogen_elements and s not in linker_elements
]

print("\n=== Automatically classified groups ===")
print("Metals / d-block:", metal_elements)
print("Linker-like:", linker_elements)
print("Halogens:", halogen_elements)
print("Other:", other_elements)


# ============================================================
# BUILD COMPONENTS FOR SUMMARY
# ============================================================

components = {}
components["Total"] = (total_up, total_down)

# Individual elements
for sym in present_elements:
    components[sym] = element_total[sym]

# Metal d orbitals, if available
for sym in metal_elements:
    if "d" in element_spd[sym]:
        components[f"{sym} d"] = element_spd[sym]["d"]

# Linker group
if linker_elements:
    linker_parts = [element_total[s] for s in linker_elements if s in element_total]
    components["Linker total"] = add_components(linker_parts)

# Linker p group, often useful for band edges
linker_p_parts = []
for sym in linker_elements:
    if "p" in element_spd[sym]:
        linker_p_parts.append(element_spd[sym]["p"])

if linker_p_parts:
    components["Linker p"] = add_components(linker_p_parts)

# Halogen group
if halogen_elements:
    halogen_parts = [element_total[s] for s in halogen_elements]
    components["Halogen total"] = add_components(halogen_parts)

# Other group
if other_elements:
    other_parts = [element_total[s] for s in other_elements]
    components["Other total"] = add_components(other_parts)


# ============================================================
# PRINT AND SAVE BAND-EDGE CONTRIBUTION SUMMARY
# ============================================================

print("\n=== Approximate DOS weight near VBM and CBM ===")
print(f"Window: ±{BAND_EDGE_HALF_WIDTH:.2f} eV")
print(f"{'Component':18s} | {'near VBM':>12s} | {'near CBM':>12s} | {'VBM %':>8s} | {'CBM %':>8s}")

total_vbm = component_weight(total_up, total_down, 0.0)
total_cbm = component_weight(total_up, total_down, band_gap)

summary_rows = []

for name, (up, down) in components.items():
    vbm_w = component_weight(up, down, 0.0)
    cbm_w = component_weight(up, down, band_gap)

    vbm_pct = 100 * vbm_w / total_vbm if total_vbm > 1e-12 else 0.0
    cbm_pct = 100 * cbm_w / total_cbm if total_cbm > 1e-12 else 0.0

    print(f"{name:18s} | {vbm_w:12.4f} | {cbm_w:12.4f} | {vbm_pct:7.1f}% | {cbm_pct:7.1f}%")

    summary_rows.append({
        "component": name,
        "vbm_weight": vbm_w,
        "cbm_weight": cbm_w,
        "vbm_percent_vs_total": vbm_pct,
        "cbm_percent_vs_total": cbm_pct,
    })

with open(f"{OUTPUT_PREFIX}_band_edge_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "component",
            "vbm_weight",
            "cbm_weight",
            "vbm_percent_vs_total",
            "cbm_percent_vs_total",
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nSaved {OUTPUT_PREFIX}_band_edge_summary.csv")


# ============================================================
# WARNINGS FOR SUSPICIOUS BAND-EDGE DOMINATION
# ============================================================

print("\n=== Quick diagnostic warnings ===")

for sym in halogen_elements:
    up, down = element_total[sym]
    vbm_pct = 100 * component_weight(up, down, 0.0) / total_vbm if total_vbm > 1e-12 else 0
    cbm_pct = 100 * component_weight(up, down, band_gap) / total_cbm if total_cbm > 1e-12 else 0

    if vbm_pct > 25 or cbm_pct > 25:
        print(
            f"WARNING: {sym} contributes strongly near band edge "
            f"(VBM {vbm_pct:.1f}%, CBM {cbm_pct:.1f}%). "
            "Check whether this is a real ligand/counterion or a suspicious dangling atom."
        )

for sym in ["O", "S", "N", "Cl", "Br", "I"]:
    if sym in element_total:
        up, down = element_total[sym]
        vbm_pct = 100 * component_weight(up, down, 0.0) / total_vbm if total_vbm > 1e-12 else 0
        cbm_pct = 100 * component_weight(up, down, band_gap) / total_cbm if total_cbm > 1e-12 else 0

        if vbm_pct > 50 or cbm_pct > 50:
            print(
                f"NOTE: {sym} dominates one band edge "
                f"(VBM {vbm_pct:.1f}%, CBM {cbm_pct:.1f}%). "
                "This may be chemically real, but inspect local environment and orbital character."
            )


# ============================================================
# PLOT 1: TOTAL DOS
# ============================================================

fig, ax = plt.subplots(figsize=(6.2, 3.7))

ax.plot(energies, total_up, color=NAVY, linewidth=2.0, label="Spin up")
ax.plot(energies, total_down, color=CYAN, linewidth=2.0, label="Spin down")

style_axis(ax)
autoscale_symmetric_y(ax, [total_up, total_down])

ax.set_title("Total DOS")
ax.set_xlabel("Energy - VBM (eV)")
ax.set_ylabel("DOS")

ylim = ax.get_ylim()
ax.text(0, ylim[1] * 0.86, "VBM", color=BLUE, ha="center", fontsize=10, fontweight="bold")
ax.text(band_gap, ylim[1] * 0.86, "CBM", color=RED, ha="center", fontsize=10, fontweight="bold")

ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")

savefig(f"{OUTPUT_PREFIX}_total_dos_vbm_aligned")


# ============================================================
# PLOT 2: AUTOMATIC STACKED PDOS
# ============================================================

panels = []

# Always include total
panels.append({
    "ylabel": "Total",
    "curves": [("Total", total_up, total_down, BLACK)]
})

# Metal d panels
for idx, sym in enumerate(metal_elements):
    if "d" in element_spd[sym]:
        up, down = element_spd[sym]["d"]
        panels.append({
            "ylabel": f"{sym} d",
            "curves": [(f"{sym} d", up, down, element_color(sym, idx))]
        })
    else:
        up, down = element_total[sym]
        panels.append({
            "ylabel": f"{sym}",
            "curves": [(sym, up, down, element_color(sym, idx))]
        })

# Linker panel: individual linker elements, excluding H if desired
linker_plot_elements = [s for s in linker_elements if s != "H"]

if linker_plot_elements:
    curves = []
    for idx, sym in enumerate(linker_plot_elements):
        up, down = element_total[sym]
        curves.append((sym, up, down, element_color(sym, idx)))
    panels.append({
        "ylabel": "/".join(linker_plot_elements),
        "curves": curves
    })

# Halogen panel
if halogen_elements:
    curves = []
    for idx, sym in enumerate(halogen_elements):
        up, down = element_total[sym]
        curves.append((sym, up, down, element_color(sym, idx)))
    panels.append({
        "ylabel": "/".join(halogen_elements),
        "curves": curves
    })

# Other panel
if other_elements:
    curves = []
    for idx, sym in enumerate(other_elements):
        up, down = element_total[sym]
        curves.append((sym, up, down, element_color(sym, idx)))
    panels.append({
        "ylabel": "/".join(other_elements),
        "curves": curves
    })


n_panels = len(panels)
height = max(3.5, 1.55 * n_panels + 1.0)

fig, axes = plt.subplots(
    n_panels,
    1,
    figsize=(6.6, height),
    sharex=True,
    gridspec_kw={"hspace": 0.08}
)

if n_panels == 1:
    axes = [axes]

fig.suptitle("Automatic Projected DOS", fontsize=20, y=0.995)

for ax, panel in zip(axes, panels):
    arrays_for_scale = []

    for label, up, down, color in panel["curves"]:
        ax.plot(energies, up, color=color, linewidth=1.4, label=label)
        ax.plot(energies, down, color=color, linewidth=1.4, linestyle="--")
        arrays_for_scale.extend([up, down])

    ax.set_ylabel(panel["ylabel"])
    ax.legend(loc="upper right", frameon=False, ncol=min(4, len(panel["curves"])))
    autoscale_symmetric_y(ax, arrays_for_scale)
    style_axis(ax)

axes[-1].set_xlabel("Energy - VBM (eV)")

ylim0 = axes[0].get_ylim()
axes[0].text(0, ylim0[1] * 0.78, "VBM", color=BLUE, ha="center", fontsize=9, fontweight="bold")
axes[0].text(band_gap, ylim0[1] * 0.78, "CBM", color=RED, ha="center", fontsize=9, fontweight="bold")

axes[0].text(
    0.02, 0.86,
    "solid = spin up\ndashed = spin down",
    transform=axes[0].transAxes,
    fontsize=7.5,
    ha="left",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.18",
        facecolor="white",
        edgecolor="#CCCCCC",
        alpha=0.88
    )
)

savefig(f"{OUTPUT_PREFIX}_stacked_pdos_vbm_aligned")


# ============================================================
# PLOT 3: INDIVIDUAL METAL d DOS PLOTS
# ============================================================

for idx, sym in enumerate(metal_elements):
    if "d" not in element_spd[sym]:
        continue

    up, down = element_spd[sym]["d"]

    fig, ax = plt.subplots(figsize=(6.2, 3.7))

    color = element_color(sym, idx)
    ax.plot(energies, up, color=color, linewidth=2.0, label=f"{sym} d spin up")
    ax.plot(energies, down, color=color, linewidth=2.0, linestyle="--", label=f"{sym} d spin down")

    style_axis(ax)
    autoscale_symmetric_y(ax, [up, down])

    ax.set_title(f"{sym} d-projected DOS")
    ax.set_xlabel("Energy - VBM (eV)")
    ax.set_ylabel("DOS")

    ylim = ax.get_ylim()
    ax.text(0, ylim[1] * 0.86, "VBM", color=BLUE, ha="center", fontsize=10, fontweight="bold")
    ax.text(band_gap, ylim[1] * 0.86, "CBM", color=RED, ha="center", fontsize=10, fontweight="bold")

    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")

    savefig(f"{OUTPUT_PREFIX}_{sym}_d_pdos_vbm_aligned")


print("\nDone.")