from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin, OrbitalType
from pymatgen.core import Element
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# COLORS: poster palette
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
# USER SETTINGS
# ============================================================

VASPRUN_FILE = "vasprun.xml"

# Use a smaller window around the gap for interpretation.
# VBM = 0 eV, CBM = band_gap.
XMIN = -1.2
EXTRA_AFTER_CBM = 1.5

# Set to 0 for no smoothing.
# For poster visuals, 2-4 is usually okay.
SMOOTH_SIGMA_POINTS = 2.0


# ============================================================
# READ VASP DATA
# ============================================================

vr = Vasprun(VASPRUN_FILE, parse_projected_eigen=True)
cdos = vr.complete_dos

band_gap, cbm, vbm, is_direct = vr.eigenvalue_band_properties

print("Band gap:", band_gap, "eV")
print("VBM:", vbm, "eV")
print("CBM:", cbm, "eV")
print("Direct gap?", is_direct)

energies = cdos.energies - vbm
XMAX = band_gap + EXTRA_AFTER_CBM


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def smooth(y, sigma_points=0):
    """
    Simple Gaussian smoothing without scipy.
    sigma_points = 0 means no smoothing.
    """
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
    """
    up = np.zeros_like(dos_obj.energies)
    down = np.zeros_like(dos_obj.energies)

    if Spin.up in dos_obj.densities:
        up = dos_obj.densities[Spin.up]

    if Spin.down in dos_obj.densities:
        down = -dos_obj.densities[Spin.down]

    up = smooth(up, SMOOTH_SIGMA_POINTS)
    down = smooth(down, SMOOTH_SIGMA_POINTS)

    return up, down


def sum_dos_objects(dos_objects):
    """
    Sum several DOS objects.
    """
    up_total = np.zeros_like(energies)
    down_total = np.zeros_like(energies)

    for dos_obj in dos_objects:
        if dos_obj is None:
            continue

        up, down = get_spin_up_down(dos_obj)
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


def get_mn_d_dos(cdos):
    """
    Robustly fetch Mn d-projected DOS.
    """
    try:
        spd = cdos.get_element_spd_dos(Element("Mn"))
    except Exception:
        spd = cdos.get_element_spd_dos("Mn")

    for orb_type, dos_obj in spd.items():
        if orb_type == OrbitalType.d or str(orb_type).lower() == "d":
            return dos_obj

    raise ValueError("Could not find Mn d DOS. Check LORBIT/projected DOS.")


def style_axis(ax):
    ax.axvline(0, linestyle="--", linewidth=1.8, color=BLUE, alpha=0.95)
    ax.axvline(band_gap, linestyle="--", linewidth=1.8, color=RED, alpha=0.90)
    ax.axhline(0, linestyle="-", linewidth=0.8, color=GRAY, alpha=0.7)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)
    ax.set_xlim(XMIN, XMAX)


def autoscale_symmetric_y(ax, arrays, margin=1.15):
    """
    Give each stacked panel a reasonable symmetric y-scale
    using only the visible energy window.
    """
    mask = (energies >= XMIN) & (energies <= XMAX)

    visible_values = []

    for arr in arrays:
        arr = np.asarray(arr)

        if arr.shape[0] != energies.shape[0]:
            raise ValueError(
                f"Array length {arr.shape[0]} does not match energy length {energies.shape[0]}"
            )

        visible_values.append(arr[mask])

    vals = np.concatenate(visible_values)

    max_abs = np.max(np.abs(vals)) if vals.size else 1.0

    if max_abs < 1e-8:
        max_abs = 1.0

    ax.set_ylim(-margin * max_abs, margin * max_abs)


def savefig(base):
    fig = plt.gcf()
    fig.subplots_adjust(hspace=0.12)
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.show()


# ============================================================
# GET DOS COMPONENTS
# ============================================================

element_dos = cdos.get_element_dos()

print("Element DOS keys found:", [str(k) for k in element_dos.keys()])

mn_dos = get_element_dos(element_dos, "Mn")
c_dos  = get_element_dos(element_dos, "C")
n_dos  = get_element_dos(element_dos, "N")
o_dos  = get_element_dos(element_dos, "O")
cl_dos = get_element_dos(element_dos, "Cl")
h_dos  = get_element_dos(element_dos, "H")

mn_d_dos = get_mn_d_dos(cdos)

total_up, total_down = get_spin_up_down(cdos)
mn_d_up, mn_d_down = get_spin_up_down(mn_d_dos)

c_up, c_down = get_spin_up_down(c_dos) if c_dos else (np.zeros_like(energies), np.zeros_like(energies))
n_up, n_down = get_spin_up_down(n_dos) if n_dos else (np.zeros_like(energies), np.zeros_like(energies))
o_up, o_down = get_spin_up_down(o_dos) if o_dos else (np.zeros_like(energies), np.zeros_like(energies))
cl_up, cl_down = get_spin_up_down(cl_dos) if cl_dos else (np.zeros_like(energies), np.zeros_like(energies))

linker_up = c_up + n_up + o_up
linker_down = c_down + n_down + o_down


# ============================================================
# PRINT BAND-EDGE CONTRIBUTION SUMMARY
# ============================================================

def window_sum(y, center, half_width=0.15):
    mask = (energies >= center - half_width) & (energies <= center + half_width)
    return float(np.sum(np.abs(y[mask])))

print("\nApproximate DOS weight near VBM and CBM")
print("Window: ±0.15 eV")

components = {
    "Total": (total_up, total_down),
    "Mn d": (mn_d_up, mn_d_down),
    "C": (c_up, c_down),
    "N": (n_up, n_down),
    "O": (o_up, o_down),
    "C/N/O": (linker_up, linker_down),
    "Cl": (cl_up, cl_down),
}

for name, (up, down) in components.items():
    vbm_weight = window_sum(up, 0.0) + window_sum(down, 0.0)
    cbm_weight = window_sum(up, band_gap) + window_sum(down, band_gap)
    print(f"{name:8s} | near VBM: {vbm_weight:10.3f} | near CBM: {cbm_weight:10.3f}")


# ============================================================
# 1. CLEAN TOTAL DOS, VBM-ALIGNED
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

savefig("total_dos_vbm_aligned_fixed")


# ============================================================
# 2. STACKED PDOS, ROSEN-LIKE
# ============================================================

fig, axes = plt.subplots(
    4, 1,
    figsize=(6.4, 7.2),
    sharex=True,
    gridspec_kw={"hspace": 0.08}
)

# Panel 1: total
axes[0].plot(energies, total_up, color=BLACK, linewidth=1.5, label="Total")
axes[0].plot(energies, total_down, color=BLACK, linewidth=1.5, linestyle="--")
fig.suptitle("Projected DOS of Generated MOF", fontsize=24, y=0.985)
axes[0].set_ylabel("Total")
axes[0].legend(loc="upper right", frameon=False)
autoscale_symmetric_y(axes[0], [total_up, total_down])

# Panel 2: Mn d
axes[1].plot(energies, mn_d_up, color=PURPLE, linewidth=1.7, label="Mn d")
axes[1].plot(energies, mn_d_down, color=PURPLE, linewidth=1.7, linestyle="--")
axes[1].set_ylabel("Mn d")
axes[1].legend(loc="upper right", frameon=False)
autoscale_symmetric_y(axes[1], [mn_d_up, mn_d_down])

# Panel 3: linker C/N/O
axes[2].plot(energies, c_up, color="#6A51A3", linewidth=1.3, label="C")
axes[2].plot(energies, c_down, color="#6A51A3", linewidth=1.3, linestyle="--")

axes[2].plot(energies, n_up, color="#D65DB1", linewidth=1.3, label="N")
axes[2].plot(energies, n_down, color="#D65DB1", linewidth=1.3, linestyle="--")

axes[2].plot(energies, o_up, color=CYAN, linewidth=1.3, label="O")
axes[2].plot(energies, o_down, color=CYAN, linewidth=1.3, linestyle="--")

axes[2].set_ylabel("C/N/O")
axes[2].legend(loc="upper right", frameon=False, ncol=3)
autoscale_symmetric_y(axes[2], [c_up, c_down, n_up, n_down, o_up, o_down])

# Panel 4: Cl
axes[3].plot(energies, cl_up, color=GREEN, linewidth=1.6, label="Cl")
axes[3].plot(energies, cl_down, color=GREEN, linewidth=1.6, linestyle="--")
axes[3].set_ylabel("Cl")
axes[3].set_xlabel("Energy - VBM (eV)")
axes[3].legend(loc="upper right", frameon=False)
autoscale_symmetric_y(axes[3], [cl_up, cl_down])

for ax in axes:
    style_axis(ax)

# labels
ylim0 = axes[0].get_ylim()
axes[0].text(0, ylim0[1] * 0.78, "VBM", color=BLUE, ha="center", fontsize=9, fontweight="bold")
axes[0].text(band_gap, ylim0[1] * 0.78, "CBM", color=RED, ha="center", fontsize=9, fontweight="bold")

axes[0].text(
    0.02, 0.86,   # same x, slightly higher y
    "solid = spin up\ndashed = spin down",
    transform=axes[0].transAxes,
    fontsize=7.5,  # smaller
    ha="left",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.18",  # smaller box padding
        facecolor="white",
        edgecolor="#CCCCCC",
        alpha=0.88
    )
)

savefig("stacked_pdos_vbm_aligned_fixed")


# ============================================================
# 3. Mn d ONLY, VBM-ALIGNED
# ============================================================

fig, ax = plt.subplots(figsize=(6.2, 3.7))

ax.plot(energies, mn_d_up, color=PURPLE, linewidth=2.0, label="Mn d spin up")
ax.plot(energies, mn_d_down, color=PURPLE, linewidth=2.0, linestyle="--", label="Mn d spin down")

style_axis(ax)
autoscale_symmetric_y(ax, [mn_d_up, mn_d_down])

ax.set_title("Mn d-projected DOS")
ax.set_xlabel("Energy - VBM (eV)")
ax.set_ylabel("DOS")

ylim = ax.get_ylim()
ax.text(0, ylim[1] * 0.86, "VBM", color=BLUE, ha="center", fontsize=10, fontweight="bold")
ax.text(band_gap, ylim[1] * 0.86, "CBM", color=RED, ha="center", fontsize=10, fontweight="bold")

ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")

savefig("mn_d_pdos_vbm_aligned_fixed")