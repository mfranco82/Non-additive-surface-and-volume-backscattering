#!/usr/bin/env python3
"""
Figures for Section 'Results' of the manuscript.

Every quantity plotted here is a *ratio of contributions within one channel*,
so all of them are independent of the overall normalisation convention chosen
for sigma^0 in eq. (cross_section).

Physics is identical to results_visualization.ipynb (same expressions, same
default parameters); only the labels are in English and the styling is the
print one.  Lengths are in units of the free-space wavelength.

Run:  python3 make_paper_figures.py
Out:  figures/*.png
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- palette
SER = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e2e1dd"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_br", [
    "#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
    "#f7b9b8", "#e87675", "#d13a39", "#8b1f1f"])

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 8, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "font.size": 10, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ------------------------------------------------------------------ model
LAM = 1.0
Ki = 2 * np.pi / LAM


def fresnel(theta, eps1, ki=Ki):
    """k, K0, K1, R_V, T_V, R_H, T_H.  theta in radians, branch Im(K1) > 0."""
    theta = np.asarray(theta, dtype=float)
    eps1 = np.asarray(eps1, dtype=complex)
    k = ki * np.sin(theta)
    K0 = ki * np.cos(theta)
    K1 = np.sqrt(eps1 * ki**2 - k**2 + 0j)
    K1 = np.where(K1.imag < 0, -K1, K1)
    RV = (eps1 * K0 - K1) / (eps1 * K0 + K1)
    TV = 1 - RV
    RH = (K0 - K1) / (K0 + K1)
    TH = 1 + RH
    return k, K0, K1, RV, TV, RH, TH


# Default: L-band moist soil, roughness inside the SPM regime (ki s << 1).
# s and s_eps are chosen so that the two mechanisms are *balanced*: the cross
# term is a geometric mean, hence largest when I_rough ~ I_diel.
P0 = dict(eps1=15 + 3j, s=0.015, l=0.5, seps=4.0, lr=0.5, lv=0.20,
          rho0=0.5, Lr=0.5, Lv=0.03)


def contributions(theta, ki=Ki, **kw):
    """Roughness, dielectric and cross contributions to <|chi^(1)|^2>/A."""
    p = {**P0, **kw}
    eps1 = np.asarray(p["eps1"], dtype=complex)
    s, l, seps, lr, lv = p["s"], p["l"], p["seps"], p["lr"], p["lv"]
    rho0, Lr, Lv = p["rho0"], p["Lr"], p["Lv"]

    k, K0, K1, RV, TV, RH, TH = fresnel(theta, eps1, ki)
    K1p, K1pp = K1.real, K1.imag
    IVV = 4 * RV / TV * k**2 + 2 * eps1 * (eps1 - 1) * k**4 / K1**2 * TV

    Wh = s**2 * l**2 / (4 * np.pi) * np.exp(-(k * l)**2)
    Om_e = seps**2 * lr**2 / (4 * np.pi) * np.exp(-(k * lr)**2) * lv / (1 + 2 * K1p * lv)
    Om_he = (rho0 * seps * s * Lr**2 / (4 * np.pi) * np.exp(-(k * Lr)**2)
             * Lv / (1 + 2 * Lv * (K1pp + 1j * K1p)))

    hh_r = 8 * K0**3 * np.abs(RH)**2 * Wh
    hh_d = ki**4 * np.abs(TH)**4 / (4 * K0 * K1pp) * Om_e
    hh_c = -2 * np.real(2 * ki**2 * K0 * RH * np.conj(TH)**2  * Om_he)

    vv_r = np.abs(IVV)**2 * Wh
    vv_d = np.abs(eps1)**2 * ki**4 * k**4 * np.abs(TV)**2 / np.abs(K1)**4 * Om_e / (2 * K1pp)
    vv_c = 2 * np.real(IVV * np.conj(eps1) * ki**2 * k**2 * np.conj(TV) / np.conj(K1)**2 * Om_he)

    return dict(HH=(hh_r, hh_d, hh_c), VV=(vv_r, vv_d, vv_c))


def eta(theta, channel="VV", **kw):
    """Fractional departure from additivity, I_cross / (I_rough + I_diel)."""
    r, d, c = contributions(theta, **kw)[channel]
    return c / (r + d)


def bias(theta, channel="VV", **kw):
    """s_eps^eff / s_eps when the data are inverted with an additive model."""
    _, d, c = contributions(theta, **kw)[channel]
    return np.sqrt(np.maximum(1 + c / d, 0))


def sigma0(theta, channel="VV", ki=Ki, **kw):
    """sigma0_QQ up to the common factor 4*pi/A of eq. (sigma0_working), which
    is angle- and channel-independent and therefore irrelevant to any
    comparison drawn from this function (HH vs VV, coupled vs additive, ...).

    From (Ehat_def): Ehat_HH = phi^(1)/sqrt(2K), Ehat_VV = (K0*TV/2k^2)*psi^(1),
    with phi^(1), psi^(1) the (HH_O1), (VV_O1) mean squares returned by
    contributions().  Substituting into sigma0 = 4*pi*K0^2*<|Ehat|^2> gives the
    two prefactors below; both reduce to (spm_sigma0) exactly when rho0 =
    s_eps = 0 (checked numerically further down).
    """
    p = {**P0, **kw}
    eps1 = np.asarray(p["eps1"], dtype=complex)
    k, K0, K1, RV, TV, RH, TH = fresnel(theta, eps1, ki)
    r, d, c = contributions(theta, ki=ki, **kw)[channel]
    tot = r + d + c
    if channel == "HH":
        return 2 * np.pi * K0 * tot
    return np.pi * K0**4 * np.abs(TV)**2 / k**4 * tot


def heatmap(ax, X, Y, Z, xlabel, ylabel, title, levels=(0.05, 0.10, 0.20), fmt=None):
    vmax = np.nanpercentile(np.abs(Z), 99.5)
    if Z.min() < 0 < Z.max():
        im = ax.pcolormesh(X, Y, Z, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax),
                           shading="auto", rasterized=True)
        lc = INK2
    else:
        flip = Z.max() <= 0
        im = ax.pcolormesh(X, Y, -Z if flip else Z, cmap=SEQ, vmin=0, vmax=vmax,
                           shading="auto", rasterized=True)
        lc = "#ffffff"
    if fmt is None:
        fmt = lambda v: f"{v:.0%}"
    cs = ax.contour(X, Y, np.abs(Z), levels=levels, colors=[lc], linewidths=0.8)
    ax.clabel(cs, fmt=fmt, fontsize=8, colors=[INK2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im


ANG = "incidence angle  $\\theta$  [deg]"

# ================================================== Fig. 1 -- composition
th_d = np.linspace(5, 75, 400)
th = np.radians(th_d)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True, sharey=True)
for ax, ch in zip(axes, ["HH", "VV"]):
    r, d, c = contributions(th)[ch]
    tot = r + d + c
    ax.plot(th_d, r / tot, color=SER[0], label="roughness")
    ax.plot(th_d, d / tot, color=SER[1], label="dielectric")
    ax.plot(th_d, c / tot, color=SER[2], label="cross")
    ax.axhline(0, color=INK3, lw=0.7)
    ax.set_xlabel(ANG)
    ax.set_title(f"{ch} channel")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
axes[0].set_ylabel("fraction of total backscattered power")
axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
axes[1].legend(loc="center right")
fig.savefig(os.path.join(OUT, "fractional_composition.png"))
plt.close(fig)

# ======================================= Fig. 1b -- HH flatness needs l=l_r
# Eq. (HH_ratio): the Gaussian factors of W_h and Omega_eps cancel only when
# l = l_r, which is why the HH curves of Fig. 1 are flat.  Away from that
# matched case the cancellation fails and HH acquires the same kind of
# crossover already seen in VV.
th_d = np.linspace(5, 75, 400)
th = np.radians(th_d)
fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
lr_vals = [0.25, 0.5, 1.0]
cols = [SER[2], SER[0], SER[1]]
for lrv, col in zip(lr_vals, cols):
    r, d, c = contributions(th, lr=lrv)["HH"]
    tot = r + d + c
    tag = f"$l_r={lrv:g}\\lambda$ (matched to $l$)" if np.isclose(lrv, P0["l"]) \
        else f"$l_r={lrv:g}\\lambda$"
    ax.plot(th_d, r / tot, color=col, ls="-", label=f"roughness, {tag}")
    ax.plot(th_d, d / tot, color=col, ls="--")
ax.plot([], [], color=INK3, ls="-", label="roughness")
ax.plot([], [], color=INK3, ls="--", label="dielectric")
ax.axhline(0, color=INK3, lw=0.7)
ax.set_xlabel(ANG)
ax.set_ylabel("fraction of total backscattered power  (HH)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title("HH channel: the flat ratio is special to $l_r=l$")
ax.grid(True, axis="y")
ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5,
          borderaxespad=0.0, frameon=False)
fig.savefig(os.path.join(OUT, "hh_lr_mismatch.png"))
plt.close(fig)

# ========================================================= Fig. 2 -- eta
th_d = np.linspace(5, 75, 260)
rho = np.linspace(-1, 1, 260)
TH, RHO = np.meshgrid(th_d, rho)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)
for ax, ch in zip(axes, ["HH", "VV"]):
    Z = eta(np.radians(TH), ch, rho0=RHO)
    im = heatmap(ax, TH, RHO, Z, ANG, "correlation coefficient  $\\rho_0$",
                 f"{ch} channel")
    ax.axhline(0, color=INK3, lw=0.7, ls=(0, (4, 3)))
fig.colorbar(im, ax=axes, label="$\\eta = I_{\\rm cross}/(I_{\\rm rough}+I_{\\rm diel})$",
             shrink=0.92, pad=0.02, format=lambda v, _: f"{v:.0%}")
fig.savefig(os.path.join(OUT, "eta_angle_rho.png"))
plt.close(fig)

# ============================================ Fig. 2b -- sigma0(theta), dB
# Same information as eta_angle_rho.png, but as the actual cross-section
# curves a reader would recognise, rather than the abstract ratio eta.
th_d = np.linspace(5, 75, 400)
th = np.radians(th_d)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True, sharex=True)
for ax, ch in zip(axes, ["HH", "VV"]):
    for rv, col, ls in zip([-1.0, 0.0, 1.0], [SER[0], INK3, SER[1]], ["-", (0, (4, 3)), "-"]):
        s0 = sigma0(th, ch, rho0=rv)
        lbl = "additive  ($\\rho_0=0$)" if rv == 0 else f"$\\rho_0={rv:+.0f}$"
        ax.plot(th_d, 10 * np.log10(s0), color=col, ls=ls, label=lbl)
    ax.set_xlabel(ANG)
    ax.set_title(f"{ch} channel")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
axes[0].set_ylabel("$\\sigma^0$  [dB, common arbitrary reference]")
axes[1].legend(loc="best", fontsize=8.5)
fig.savefig(os.path.join(OUT, "sigma0_curves.png"))
plt.close(fig)

# ============================================== Fig. 3 -- two conditions
th_d = np.linspace(5, 75, 260)
epp = np.logspace(-1, 1.4, 260)
Lvv = np.logspace(-2.7, 0.0, 260)
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.14)

TH, EPP = np.meshgrid(th_d, epp)
Z1 = eta(np.radians(TH), "VV", eps1=15 + 1j * EPP)
im = heatmap(axes[0], TH, EPP, Z1, ANG, "dielectric loss  $\\epsilon_1''$",
             "(a)  angle vs loss")
axes[0].set_yscale("log")

# (b) a line cut is far clearer than a heatmap here: it shows the rise, the
# ridge and the plateau, none of which a colour map resolves unambiguously.
axes[1].set_title("(b)  the ridge in $L_v$   ($\\theta=40^\\circ$)")
Lv = np.logspace(-3.2, 1.0, 600)
for eppv, col in zip([1.0, 3.0, 10.0], SER):
    ev = eta(np.radians(40.0), "VV", Lv=Lv, eps1=15 + 1j * eppv)
    axes[1].semilogx(Lv, ev, color=col, label=f"$\\epsilon_1''={eppv:g}$")
    axes[1].plot(Lv[np.argmax(ev)], ev.max(), "o", ms=4.5, color=col)
_, _, K1_40, _, _, _, _ = fresnel(np.radians(40.0), 15 + 3j)
axes[1].axvline(1 / (2 * np.abs(K1_40)), color=INK3, lw=1.0, ls=(0, (4, 3)))
axes[1].text(1 / (2 * np.abs(K1_40)) * 1.15, 0.012, "$2L_v|K_1|=1$",
             color=INK2, fontsize=8.5)
axes[1].set_xlabel("correlation depth  $L_v\\ [\\lambda]$")
axes[1].set_ylabel("$\\eta$")
axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
axes[1].grid(True, axis="y")
axes[1].set_axisbelow(True)
axes[1].legend(loc="upper left")

fig.colorbar(im, ax=axes[0], label="$\\eta$", shrink=0.92, pad=0.02,
             format=lambda v, _: f"{v:.0%}")
fig.savefig(os.path.join(OUT, "coupling_conditions.png"))
plt.close(fig)

# ========================================== Fig. 3b -- a fixed layer, three bands
# The ridge 2*L_v*|K_1|=1 is universal in L_v/lambda (same eps1, same angle),
# but a correlated layer of *fixed physical thickness* sits at a different
# point on that universal curve for every radar band.  This is the concrete
# version of the "second frequency" argument closing Sec. 6.4/6.6: a coupling
# invisible at one band can become visible at another.
LAM_BAND = {"L": 23.5, "C": 5.6, "X": 3.1}  # cm, representative centre wavelengths
LV_PHYS = 0.2  # cm, a thin correlated skin (thinner than the 7 mm default)

fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
Lv_lambda = np.logspace(-3.2, 1.0, 600)
ev = eta(np.radians(40.0), "VV", Lv=Lv_lambda)
ax.semilogx(Lv_lambda, ev, color=INK3, lw=1.6)
for (band, lam), col in zip(LAM_BAND.items(), SER):
    x = LV_PHYS / lam
    y = float(eta(np.radians(40.0), "VV", Lv=x))
    ax.plot(x, y, "o", ms=7, color=col, zorder=5)
    ax.annotate(f"{band}-band\n($\\lambda={lam:g}$ cm)", (x, y), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=col)
ax.set_xlabel("correlation depth  $L_v/\\lambda$")
ax.set_ylabel("$\\eta$  (VV, $\\theta=40^\\circ$, $\\rho_0=0.5$)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_title(f"a fixed $L_v={LV_PHYS:g}$ cm layer sits differently on the ridge by band")
ax.grid(True, axis="y")
ax.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "band_dependence.png"))
plt.close(fig)

# ==================================== Fig. 4 -- polarisation-ratio shift, dB
th_d = np.linspace(5, 75, 260)
rho = np.linspace(-1, 1, 260)
TH, RHO = np.meshgrid(th_d, rho)
fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
Zvv = eta(np.radians(TH), "VV", rho0=RHO)
Zhh = eta(np.radians(TH), "HH", rho0=RHO)
Zdb = 10 * np.log10((1 + Zvv) / (1 + Zhh))
im = heatmap(ax, TH, RHO, Zdb, ANG, "correlation coefficient  $\\rho_0$",
             "shift of the co-polarised ratio  $\\sigma^0_{VV}/\\sigma^0_{HH}$",
             levels=(0.02, 0.04, 0.06), fmt=lambda v: f"{v:.2f} dB")
ax.axhline(0, color=INK3, lw=0.7, ls=(0, (4, 3)))
fig.colorbar(im, ax=ax, label="$\\Delta[\\sigma^0_{VV}/\\sigma^0_{HH}]$  [dB]",
             shrink=0.92, pad=0.03)
fig.savefig(os.path.join(OUT, "pol_ratio_shift.png"))
plt.close(fig)

# =================================================== Fig. 5 -- inversion bias
th_d = np.linspace(5, 75, 260)
rho = np.linspace(-1, 1, 260)
se = np.logspace(-0.5, 1.0, 260)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)

TH, RHO = np.meshgrid(th_d, rho)
B1 = bias(np.radians(TH), "VV", rho0=RHO)
im = axes[0].pcolormesh(TH, RHO, B1, cmap=DIV,
                        norm=TwoSlopeNorm(1, B1.min(), B1.max()),
                        shading="auto", rasterized=True)
cs = axes[0].contour(TH, RHO, B1, levels=[0.7, 0.85, 1.15, 1.3], colors=[INK2], linewidths=0.8)
axes[0].clabel(cs, fmt=lambda x: f"{x:.2f}", fontsize=8, colors=[INK2])
axes[0].axhline(0, color=INK3, lw=0.7, ls=(0, (4, 3)))
axes[0].set_xlabel(ANG)
axes[0].set_ylabel("correlation coefficient  $\\rho_0$")
axes[0].set_title("(a)  angle vs correlation")
fig.colorbar(im, ax=axes[0], label="$s_\\varepsilon^{\\rm eff}/s_\\varepsilon$",
             shrink=0.92, pad=0.02)

SE, TH2 = np.meshgrid(se, th_d)
B2 = bias(np.radians(TH2), "VV", seps=SE, rho0=1.0)
im2 = axes[1].pcolormesh(SE, TH2, B2, cmap=SEQ, vmin=1, vmax=np.nanmax(B2),
                         shading="auto", rasterized=True)
cs2 = axes[1].contour(SE, TH2, B2, levels=[1.1, 1.25, 1.5, 2.0], colors=["#ffffff"],
                      linewidths=0.9)
axes[1].clabel(cs2, fmt=lambda x: f"{x:.2f}", fontsize=8, colors=["#ffffff"])
axes[1].set_xscale("log")
axes[1].axvline(P0["seps"], color="#ffffff", lw=1.0, ls=(0, (4, 3)))
axes[1].set_xlabel("dielectric fluctuation amplitude  $s_\\varepsilon$")
axes[1].set_ylabel(ANG)
axes[1].set_title("(b)  the bias grows when the volume term is weak  ($\\rho_0=1$)")
fig.colorbar(im2, ax=axes[1], label="$s_\\varepsilon^{\\rm eff}/s_\\varepsilon$",
             shrink=0.92, pad=0.02)
fig.savefig(os.path.join(OUT, "retrieval_bias.png"))
plt.close(fig)

# ======================================================= numbers for the text
print("figures written to", OUT)
th_d = np.linspace(5, 75, 701)
th = np.radians(th_d)

# SPM closure residual
for e in [4 + 0j, 15 + 3j, 30 + 10j, 2.5 + 0.1j]:
    k, K0, K1, RV, TV, _, _ = fresnel(th, e)
    IVV = 4 * RV / TV * k**2 + 2 * e * (e - 1) * k**4 / K1**2 * TV
    EVV_hat = -(K0 * TV / (2 * k**2)) * IVV
    EVV_spm = 2 * K0 * (e - 1) / (e * K0 + K1)**2 * (K1**2 + e * k**2)
    print(f"  SPM closure  eps1={e!s:>12}  max|Ehat/SPM+1| = {np.max(np.abs(EVV_hat / EVV_spm + 1)):.2e}")

# sigma0() normalisation check against (spm_sigma0), both channels, no free
# constant.  Restricted to lossy eps1: the dielectric term of contributions()
# carries a 1/K1'' that is only well-defined once K1'' > 0.
for e in [15 + 3j, 30 + 10j, 2.5 + 0.1j]:
    k, K0, K1, RV, TV, RH, TH_ = fresnel(th, e)
    Wh = P0["s"]**2 * P0["l"]**2 / (4 * np.pi) * np.exp(-(k * P0["l"])**2)
    aHH = (e - 1) / (np.cos(th) + np.sqrt(e - np.sin(th)**2))**2
    aVV = ((e - 1) * (e * (1 + np.sin(th)**2) - np.sin(th)**2)
           / (e * np.cos(th) + np.sqrt(e - np.sin(th)**2))**2)
    spm_HH = 16 * np.pi * Ki**4 * np.cos(th)**4 * np.abs(aHH)**2 * Wh
    spm_VV = 16 * np.pi * Ki**4 * np.cos(th)**4 * np.abs(aVV)**2 * Wh
    ratio_HH = sigma0(th, "HH", eps1=e, seps=0, rho0=0) / spm_HH
    ratio_VV = sigma0(th, "VV", eps1=e, seps=0, rho0=0) / spm_VV
    print(f"  sigma0() vs spm_sigma0  eps1={e!s:>9}  "
          f"HH ratio in [{ratio_HH.min():.6f},{ratio_HH.max():.6f}]  "
          f"VV ratio in [{ratio_VV.min():.6f},{ratio_VV.max():.6f}]")

# a second, independent check on the channel prefactors of sigma0(): at normal
# incidence there is no distinction between polarizations, so sigma0_HH must
# equal sigma0_VV exactly.  (The unnormalised contributions() building blocks
# do not satisfy this on their own -- VV vanishes as theta -> 0 there, purely
# from the missing 1/k^4 -- which is precisely why contributions() must never
# be compared across channels without going through sigma0() first.)
th0 = np.radians([1.0, 0.1, 0.01])
ratio_normal = sigma0(th0, "HH") / sigma0(th0, "VV")
print(f"  sigma0_HH/sigma0_VV at theta = 1, 0.1, 0.01 deg: {np.round(ratio_normal, 6)}")

# eta extremes at rho0 = +-1
for ch in ("HH", "VV"):
    ep = eta(th, ch, rho0=1.0)
    print(f"  {ch}: eta(rho0=+1) in [{ep.min():+.3f}, {ep.max():+.3f}]"
          f"  at theta = {th_d[np.argmax(np.abs(ep))]:.0f} deg")

# polarization-ratio shift, in dB
sh = 10 * np.log10((1 + eta(th, "VV", rho0=1.0)) / (1 + eta(th, "HH", rho0=1.0)))
print(f"  polarization-ratio shift (rho0=+1): max |delta| = {np.max(np.abs(sh)):.3f} dB")

# polarization-ratio shift over the full (theta, rho0) grid of pol_ratio_shift.png
rho_full = np.linspace(-1, 1, 401)
TH_f, RHO_f = np.meshgrid(th, rho_full)
sh_full = 10 * np.log10((1 + eta(TH_f, "VV", rho0=RHO_f)) / (1 + eta(TH_f, "HH", rho0=RHO_f)))
j = np.unravel_index(np.argmax(np.abs(sh_full)), sh_full.shape)
print(f"  polarization-ratio shift, full grid: max |delta| = {np.abs(sh_full[j]):.3f} dB"
      f"  at theta={th_d[j[1]]:.0f} deg, rho0={rho_full[j[0]]:+.2f}")

# HH roughness/dielectric ratio: relative angular swing when l_r != l
for lrv in [0.25, 0.5, 1.0]:
    r, d, c = contributions(th, lr=lrv)["HH"]
    frac = r / (r + d + c)
    print(f"  HH roughness fraction, l_r={lrv:g}lambda: "
          f"[{frac.min():.3f},{frac.max():.3f}]  swing={frac.max()-frac.min():.3f}")

# a fixed physical L_v across bands
for band, lam in LAM_BAND.items():
    x = LV_PHYS / lam
    y = float(eta(np.radians(40.0), "VV", Lv=x))
    print(f"  {band}-band (lambda={lam:g} cm): L_v/lambda={x:.4f}  eta={y:.3f}")

# retrieval bias at the default point
b = bias(th, "VV", rho0=1.0)
print(f"  s_eps bias (rho0=+1, default): [{b.min():.2f}, {b.max():.2f}]")
b2 = bias(th, "VV", rho0=1.0, seps=1.0)
print(f"  s_eps bias (rho0=+1, s_eps=1): [{b2.min():.2f}, {b2.max():.2f}]")

# optimum L_v
_, _, K1_40, _, _, _, _ = fresnel(np.radians(40.0), 15 + 3j)
print(f"  coherence condition at 40 deg: L_v = 1/(2|K1|) = {1/(2*abs(K1_40)):.4f} lambda"
      f"  = {1/(2*abs(K1_40))*0.235*100:.2f} cm at L band")

# where roughness and dielectric cross over, VV
r, d, c = contributions(th)["VV"]
i = np.argmin(np.abs(r - d))
print(f"  VV roughness/dielectric crossover at theta = {th_d[i]:.0f} deg")

# --------------------------------------------------------------------------------
# Convenience for the authors: keep the manuscript's copy of the figures in sync, so
# that editing a figure here cannot leave a stale one in the paper.  The repository
# is self-contained without this; the block is a no-op when the manuscript tree is
# not present alongside it.
# --------------------------------------------------------------------------------
_MS = os.path.normpath(os.path.join(OUT, os.pardir, os.pardir, "manuscript", "figures"))
if os.path.isdir(_MS):
    import shutil
    _n = 0
    for _f in sorted(os.listdir(OUT)):
        if _f.endswith(".png"):
            shutil.copy2(os.path.join(OUT, _f), os.path.join(_MS, _f))
            _n += 1
    print(f"  {_n} figures mirrored to {_MS}")
