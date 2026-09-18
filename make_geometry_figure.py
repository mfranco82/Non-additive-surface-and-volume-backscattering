#!/usr/bin/env python3
"""
Figure 1: the coordinate change z = zeta + h(x) f(zeta).

Left  -- physical coordinates (x, z): rough interface z = h(x), and below it a
         genuinely inhomogeneous medium, a correlated Gaussian realisation of
         eps_1(r).
Right -- transformed coordinates (x, zeta): the interface is the plane zeta = 0.
         The medium shown is the SAME realisation pulled back through the map,
         eps_1(x, zeta + h(x) f(zeta)), so the fluctuations are dragged with the
         interface near zeta = 0 and are untouched far from it, where f -> 0.

Run:  python3 make_geometry_figure.py
Out:  figures/chg_coord.png
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})

HALO = pe.withStroke(linewidth=3.0, foreground="#1d5fae", alpha=0.35)

rng = np.random.default_rng(20240917)

# ------------------------------------------------------------------ geometry
XMIN, XMAX = 0.0, 1.0
ZMIN, ZMAX = -0.60, 0.30
NX, NZ = 1400, 1400

x = np.linspace(XMIN, XMAX, NX)
z = np.linspace(ZMIN, ZMAX, NZ)
dx, dz = x[1] - x[0], z[1] - z[0]

# --- rough surface: smooth, small slope (the SPM regime the paper works in)
L_SURF = 0.072                     # lateral correlation length of h
S_SURF = 0.042                     # rms height (exaggerated for legibility)
h = gaussian_filter(rng.standard_normal(NX), L_SURF / dx, mode="wrap")
h *= S_SURF / h.std()
h -= h.mean()

# --- permittivity fluctuation: correlated 2D Gaussian field, anisotropic
LX_EPS, LZ_EPS = 0.075, 0.045
eps = gaussian_filter(rng.standard_normal((NZ, NX)),
                      (LZ_EPS / dz, LX_EPS / dx), mode="wrap")
eps /= eps.std()

# the field is only defined in the lower medium; extend it below the box so the
# pull-back never has to extrapolate
field = RegularGridInterpolator((z, x), eps, bounds_error=False,
                                fill_value=None, method="linear")


def f_profile(zz, d=0.17):
    """Gauge profile: f(0)=1, f'(0)=f''(0)=0, f(+-inf)=0."""
    return np.exp(-(zz / d) ** 4)


ZETA_LINES = (-0.045, -0.11, -0.20, -0.33)

X, Z = np.meshgrid(x, z)
H = np.broadcast_to(h, X.shape)

# left panel: the field as it is, masked above the surface
eps_phys = np.ma.masked_where(Z > H, eps)

# right panel: same realisation, pulled back through z = zeta + h f(zeta)
Zmap = Z + H * f_profile(Z)
eps_tran = field(np.stack([Zmap.ravel(), X.ravel()], axis=-1)).reshape(X.shape)
eps_tran = np.ma.masked_where(Z > 0, eps_tran)

VMIN, VMAX = -3.1, 3.1

# -------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.15), constrained_layout=True)

for ax, data, flat in zip(axes, (eps_phys, eps_tran), (False, True)):
    ax.imshow(data, origin="lower", extent=[XMIN, XMAX, ZMIN, ZMAX],
              cmap=SEQ, vmin=VMIN, vmax=VMAX, aspect="auto",
              interpolation="bilinear", rasterized=True)

    # vacuum half-space
    if flat:
        ax.axhspan(0, ZMAX, color="white", zorder=2)
        ax.plot([XMIN, XMAX], [0, 0], color=INK, lw=2.0, ls=(0, (6, 3)), zorder=3)
    else:
        ax.fill_between(x, h, ZMAX, color="white", zorder=2)
        ax.plot(x, h, color=INK, lw=2.0, zorder=3, solid_capstyle="round")

    # surfaces of constant zeta: wavy in (x, z), horizontal in (x, zeta).
    # They flatten with depth because f -> 0, which is what the map does.
    for zc in ZETA_LINES:
        zz = np.full_like(x, zc) if flat else zc + h * f_profile(zc)
        ax.plot(x, zz, color="#ffffff", lw=1.0, ls=(0, (5, 4)),
                alpha=0.75, zorder=3.5)

    # incident wave
    ax.annotate("", xy=(0.40, 0.10), xytext=(0.23, 0.255), zorder=4,
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=INK,
                                shrinkA=0, shrinkB=0))
    ax.text(0.215, 0.258, r"$\vec{\kappa}_i$", color=INK, fontsize=14,
            ha="right", va="bottom", zorder=4)

    # vertical axis
    ax.annotate("", xy=(0.80, 0.235), xytext=(0.80, -0.36), zorder=4,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK2,
                                shrinkA=0, shrinkB=0))

    ax.text(0.045, 0.165, r"$\epsilon_0 = 1$", fontsize=15, color=INK, zorder=4)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(ZMIN, ZMAX)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#d8d7d3")
        sp.set_linewidth(0.9)

# panel-specific annotation
axes[0].text(0.815, 0.245, r"$z$", fontsize=15, color=INK2, va="bottom")
axes[0].text(0.985, 0.075, r"$z = h(\vec{x})$", fontsize=13.5, color=INK,
             ha="right", va="bottom", zorder=4)
axes[0].text(0.045, -0.50, r"$\epsilon_1(\vec{x},z)$", fontsize=15,
             color="white", zorder=4,
             path_effects=[HALO])
axes[0].set_title("physical coordinates", fontsize=13, color=INK,
                  loc="left", pad=7, fontweight="semibold")

axes[1].text(0.815, 0.245, r"$\zeta$", fontsize=15, color=INK2, va="bottom")
axes[1].text(0.985, 0.035, r"$\zeta = 0$", fontsize=13.5, color=INK,
             ha="right", va="bottom", zorder=4)
axes[1].text(0.045, -0.50, r"$\epsilon_1(\vec{x},\zeta)$", fontsize=15,
             color="white", zorder=4, path_effects=[HALO])
axes[1].set_title(r"transformed coordinates,   $z = \zeta + h(\vec{x})\,f(\zeta)$",
                  fontsize=13, color=INK, loc="left", pad=7, fontweight="semibold")

fig.savefig(os.path.join(OUT, "chg_coord.png"))
plt.close(fig)
print("wrote", os.path.join(OUT, "chg_coord.png"))
print(f"  surface: rms = {h.std():.4f}, max |dh/dx| = {np.abs(np.gradient(h, dx)).max():.3f}")
