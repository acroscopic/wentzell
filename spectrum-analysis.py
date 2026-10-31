import numpy as np
import matplotlib.pyplot as plt
from contourpy import contour_generator
from dataclasses import dataclass, replace
from functools import lru_cache
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gamma, hyp2f1, rgamma
"""Spectrum and stability of a Klein-Gordon field with dynamical Wentzell boundary conditions in global AdS.
Quantization condition, Eq. (general-quantization) of the preprint:
    kappa (w^2 - Lambda) = -A/B = -Gamma(-nu)/Gamma(nu) * Gamma(s - w/2) Gamma(s + w/2) / [Gamma(t - w/2) Gamma(t + w/2)]
    nu = sqrt(1 + 4 M^2)/2,  s = (2 ell + d + 2 nu)/4,  t = (2 ell + d - 2 nu)/4,  Lambda = ell(ell + d - 2) + mb2 + xi_b (d - 1)(d - 2)
The condition is linear in kappa, so every complex w is a mode of exactly one coupling kappa(w) = -A / [B (w^2 - Lambda)].
Edit the parameters in P and run: python wentzell.py
"""
OUT = Path(__file__).resolve().parent / "generated_graphs"
KAPPAS = (-2.5, 2.5)  # kappa range for the sweeps
BLUE, RED, GREEN, PINK, GOLD = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"  # Okabe-Ito, color-blind safe
LINES, DOTS = ["-", "--", "-.", ":"], "osD^"  # series also differ by line style or marker, never color alone
W1 = (5.5, 3.6)  # one panel at 0.85\textwidth, so text prints at true size
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11, "legend.fontsize": 10.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.linewidth": 2, "axes.linewidth": 0.9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
@dataclass(frozen=True)
class P:
    d: int = 3
    ell: int = 0
    kappa: float = -1.0
    mb2: float = 1.0
    xi_b: float = 0.0
    M2: float = 0.25
    @property
    def nu(self): return np.sqrt(1 + 4 * self.M2) / 2
    @property
    def s(self): return (2 * self.ell + self.d + 2 * self.nu) / 4
    @property
    def t(self): return (2 * self.ell + self.d - 2 * self.nu) / 4
    @property
    def Lam(self): return self.ell * (self.ell + self.d - 2) + self.mb2 + self.xi_b * (self.d - 1) * (self.d - 2)
# --- quantization condition
def ab(w, p):
    """Entire functions a, b with A/B = a/b."""
    a = gamma(-p.nu) / gamma(p.nu) * rgamma(p.t - w / 2) * rgamma(p.t + w / 2)
    return a, rgamma(p.s - w / 2) * rgamma(p.s + w / 2)
def residual(w, p, kappa):
    """Pole-free residual, zero exactly at the modes."""
    a, b = ab(w, p)
    return kappa * (w * w - p.Lam) * b + a
def kappa_of(w, p):
    """The coupling for which w is a mode."""
    a, b = ab(w, p)
    return -a / ((w * w - p.Lam) * b)
def real_roots(p, kappa=None, wmax=14, n=6000):
    """Positive real modes below wmax."""
    k = p.kappa if kappa is None else kappa
    w = np.linspace(1e-6, wmax, n)
    f = residual(w, p, k).real
    return np.array([brentq(lambda x: residual(x, p, k).real, w[j], w[j + 1]) for j in np.flatnonzero(f[:-1] * f[1:] < 0)])
# --- growing modes
@lru_cache(maxsize=None)
def locus(p, re_max=3.0, im_max=8.0, n=500):
    """Growing modes for every real kappa, as branches (w, kappa) with Im w > 0."""
    y = np.linspace(1e-4, im_max, 2 * n)
    branches = [(1j * y, kappa_of(1j * y, p).real)]  # kappa is real on the imaginary axis
    x, y = np.linspace(1e-3, re_max, n), np.linspace(1e-3, im_max, n)
    W = x + 1j * y[:, None]
    a, b = ab(W, p)
    im = (-a * np.conj((W * W - p.Lam) * b)).imag  # |den|^2 Im kappa: same zeros, no poles
    for seg in contour_generator(x, y, im).lines(0):  # off-axis modes are the curves Im kappa = 0
        w = seg[:, 0] + 1j * seg[:, 1]
        branches.append((w, kappa_of(w, p).real))
    return branches
def kappa_c(p):
    """Coupling where a mode passes through w = 0."""
    return gamma(-p.nu) / gamma(p.nu) * (gamma(p.s) / gamma(p.t)) ** 2 / p.Lam
def kappa_min(p, im_max=8.0):
    """Most negative coupling with a purely imaginary mode, and its Im w."""
    y = np.linspace(1e-4, im_max, 4000)
    j = np.argmin(kappa_of(1j * y, p).real)
    if j == 0:  # no interior minimum: the branch ends at w = 0
        return kappa_c(p), 0.0
    r = minimize_scalar(lambda v: kappa_of(1j * v, p).real, bounds=(y[j - 1], y[j + 1]), method="bounded")
    return r.fun, r.x
def collisions(p):
    """Krein collisions, where off-axis branches meet the real axis: [(kappa_s, w_s)]."""
    dk = lambda x: (kappa_of(x + 1e-6, p) - kappa_of(x - 1e-6, p)).real  # kappa(w) is extremal there
    ends = [w[j].real for w, _ in locus(p)[1:] for j in (0, -1) if w[j].imag < 1e-2]
    ws = [brentq(dk, x - 0.05, x + 0.05) for x in ends if dk(x - 0.05) * dk(x + 0.05) < 0]
    return [(kappa_of(x, p).real, x) for x in ws if abs(kappa_of(x, p)) < 1e3]  # skip poles
def window(p):
    """Lowest coupling with a growing mode; the unstable window is (window, 0)."""
    ends = [k[np.abs(k) < 1e3].min(initial=0) for _, k in locus(p)]  # grid estimate of each branch's end
    return min(ends + [ks for ks, _ in collisions(p)])  # a Krein collision is the exact end of the pair branch
def pair_range(p):
    """Couplings spanned by the complex pair (NaNs if there is none)."""
    k = np.concatenate([k[np.abs(k) < 1e3] for _, k in locus(p)[1:]] + [np.empty(0)])
    return (k.min(), k.max()) if k.size else (np.nan, np.nan)
# --- figures
def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved {name}.png")
def below(ax, ncol, **kw):
    """Legend under the axes."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=ncol, frameon=False, **kw)
def tint(c, a):
    """Opaque mix of color c with white, so fills never blend."""
    return 1 - a * (1 - np.array(to_rgb(c)))
def plot_intersections(p, name, wmax=8.5):
    """Both sides of the condition against w; modes are the crossings. A close pair gets a zoomed inset."""
    w = np.linspace(0, wmax, 6000)
    a, b = ab(w, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs = -a / b
    rhs[(np.abs(b) < 1e-3) | (np.abs(rhs) > 300)] = np.nan  # hide the poles
    roots = real_roots(p, wmax=wmax)
    x = p.kappa * (roots ** 2 - p.Lam)
    H, xlim, ylim = 4.6, (-70, 12), (0, roots[-1] + 0.4)
    fig, ax = plt.subplots(figsize=(W1[0], H))
    close = np.flatnonzero(np.diff(roots) < 0.6)  # pairs too close to read at full scale
    axes = [ax]
    if close.size:
        i, box = close[0], [0.14, 0.03, 0.42, 0.27]
        ins = ax.inset_axes(box, zorder=6)  # drawn over the lowest branch
        y0, y1 = roots[i] - 0.25, roots[i + 1] + 0.3
        pos = ax.get_position()
        shape = box[2] * pos.width * W1[0] / (box[3] * pos.height * H)  # width/height of the inset in inches
        dx = (y1 - y0) * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * shape  # zoom region with the same shape
        x0 = x[i:i + 2].mean() - 0.4 * dx  # modes left of center, labels to their right
        ins.set(xlim=(x0, x0 + dx), ylim=(y0, y1), xticks=[], yticks=[])
        ins.spines[["top", "right"]].set_visible(True)
        ax.indicate_inset([x0, y0, dx, y1 - y0], edgecolor="black", alpha=1, lw=0.8)
        ax.annotate("", (x0, (y0 + y1) / 2), (box[0] + box[2], box[1] + box[3] / 2), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", lw=0.8, shrinkA=0, shrinkB=0))
        axes.append(ins)
    lhs = r"$\omega\cot(\pi\omega/2)$" if (p.M2, p.d, p.ell) == (0, 3, 0) else r"$-A(\omega)/B(\omega)$"
    for a_ in axes:
        main = a_ is ax
        a_.axvline(0, color="gray", lw=0.6)
        a_.plot(rhs, w, color=BLUE, label=lhs if main else None)
        a_.plot(p.kappa * (w * w - p.Lam), w, "--", color=RED, label=r"$\kappa(\omega^2-\Lambda)$" if main else None)
        a_.plot(x, roots, "o", color=GOLD, mec="black", ms=7, zorder=5, label="modes" if main else None)
    for xi, ri in zip(x, roots):  # the close pair is labeled in the inset only
        on = axes[-1] if close.size and ri < roots[close[0]] + 0.7 else ax
        on.annotate(rf"$\omega={ri:.3f}$", (xi, ri), (8, 9) if on is ax else (14, 9), textcoords="offset points", va="center", fontsize=9)
    ax.set(xlim=xlim, ylim=ylim, xlabel="value of each side", ylabel=r"frequency $\omega$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    save(fig, name)
def spectrum_curves(ax, p, wmax=13, color=BLUE, lw=2):
    """Real modes against kappa as smooth curves, the zero contour of the normalized residual."""
    k, w = np.linspace(*KAPPAS, 600), np.linspace(1e-4, wmax, 1200)
    a, b = ab(w, p)
    f = (k[None, :] * ((w * w - p.Lam) * b)[:, None] + a[:, None]) / (np.abs((w * w + p.Lam) * b) + np.abs(a))[:, None]
    for seg in contour_generator(k, w, f).lines(0):
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw)
    ax.set(xlim=KAPPAS, ylim=(0, wmax), xlabel=r"Wentzell coupling $\kappa$", ylabel=r"real frequency $\omega_n$")
def spectrum_panel(ax, p, tag):
    """Real modes against kappa; the curves turn orange inside the unstable window."""
    lo, ks = window(p), collisions(p)
    ax.axvspan(lo, 0, color=to_rgba(RED, 0.09), lw=0)
    spectrum_curves(ax, p, wmax=8, lw=1.8)
    n = len(ax.lines)
    spectrum_curves(ax, p, wmax=8, color=RED, lw=1.8)  # the same curves in orange, clipped to the window
    for line in ax.lines[n:]:
        line.set_clip_path(Rectangle((lo, 0), -lo, 8, transform=ax.transData))
    for k, tex in [(0, "$0$"), (kappa_c(p), r"$\kappa_c$")] + [(k, r"$\kappa_s$") for k, _ in ks]:
        ax.axvline(k, ls=":", color="black", lw=1.3)
        ax.text(k, 1.02, tex, transform=ax.get_xaxis_transform(), ha="center")
    for k, w in ks:  # two real modes merge here and leave the real axis
        ax.plot(k, w, "o", ms=3.5, color="black", zorder=5)
        ax.text(k + 0.05, w, "roots\nmerge", va="center", fontsize=9)
    ax.plot([lo, 0], [-0.045, -0.045], transform=ax.get_xaxis_transform(), color=RED, lw=2, marker="|", ms=8, mew=2, clip_on=False)  # bracket marks the window without color
    ax.tick_params(axis="x", pad=9)
    ax.set_title(rf"({tag}) $M^2={p.M2:g}$", loc="left", pad=16)
def plot_spectrum_vs_kappa(massless, massive, name):
    """Massless and massive real spectra side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(W1[0], 3.5), sharey=True)
    for ax, p, tag in zip(axes, [massless, massive], "ab"):
        spectrum_panel(ax, p, tag)
        ax.set(xlim=KAPPAS, ylim=(0, 8), xlabel=r"Wentzell coupling $\kappa$", ylabel="")
    axes[0].set_ylabel(r"real frequency $\omega_n$")
    h = [plt.Line2D([], [], color=BLUE, lw=1.8), plt.Line2D([0, 1], [0, 0], color=RED, lw=2, marker="|", ms=8, mew=2), plt.Line2D([], [], ls=":", color="black", lw=1.3)]
    fig.legend(h, ["real roots", "unstable window", r"$0$, $\kappa_c$, $\kappa_s$"], loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=10, handlelength=2.2)
    save(fig, name)
def plot_root_sweep(ax, p, field, values):
    """n-th mode against n for several values of one parameter."""
    for c, ls, m, v in zip([BLUE, RED, GREEN, PINK], LINES, DOTS, values):
        r = real_roots(replace(p, **{field: v}))
        ax.plot(np.arange(1, len(r) + 1), r, ls, marker=m, ms=5, color=c, label=f"${v:g}$")
    ax.set(xlabel="mode index $n$", ylabel=r"real frequency $\omega_n$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
def plot_sweeps(massless, massive):
    """Boundary-parameter sweeps (massless) and bulk mass and ell sweeps (massive)."""
    fig, (l, r) = plt.subplots(1, 2, figsize=(W1[0], 3.2), sharey=True)
    plot_root_sweep(l, massless, "mb2", [0, 0.5, 1, 2])
    plot_root_sweep(r, massless, "xi_b", [-0.5, 0, 0.5, 1])
    r.set_ylabel("")
    below(l, 2, title=r"$m_b^2$")
    below(r, 2, title=r"$\xi_b$")
    save(fig, "03_massless_boundary_sweeps")
    for name, field, values, title in [("06_massive_bulk_mass_sweep", "M2", [-0.2, 0, 0.25, 0.7], r"$M^2$"),
                                       ("07_massive_ell_sweep", "ell", [0, 1, 2, 3], r"$\ell$")]:
        fig, ax = plt.subplots(figsize=W1)
        plot_root_sweep(ax, massive, field, values)
        ax.legend(loc="upper left", frameon=False, title=title)
        save(fig, name)
def plot_growth_rate(massless, massive, name):
    """Im w of every growing branch against kappa."""
    fig, ax = plt.subplots(figsize=W1)
    for p, c, ls in [(massless, BLUE, "-"), (massive, RED, "--")]:
        for i, (w, k) in enumerate(locus(p)):
            ax.plot(k, w.imag, ls, color=c, label=rf"$M^2={p.M2:g}$" if i == 0 else None)
    ax.set(xlim=(KAPPAS[0], 0.1), ylim=(0, 6), xlabel=r"Wentzell coupling $\kappa$", ylabel=r"growth rate $\mathrm{Im}\,\omega$")
    ax.legend(frameon=False)
    save(fig, name)
def colored_line(ax, w, k, norm, cmap="viridis", lw=2.6):
    """A branch in the complex w plane, colored by kappa."""
    pts = np.column_stack([w.real, w.imag])
    lc = LineCollection(np.stack([pts[:-1], pts[1:]], 1), cmap=cmap, norm=norm, lw=lw, capstyle="round")
    lc.set_array((k[:-1] + k[1:]) / 2)
    ax.add_collection(lc)
    return lc
def plot_complex_plane(massless, massive, name, kmin=-2.0):
    """Growing modes in the complex w plane, colored by kappa (viridis also reads in grayscale)."""
    norm = plt.Normalize(kmin, 0)
    fig, axes = plt.subplots(1, 2, figsize=W1, sharey=True)
    for ax, p, tag in zip(axes, [massless, massive], "ab"):
        for w, k in locus(p):
            m = (k >= kmin) & (k < 0)
            for sign in (1, -1):  # modes come in pairs w, -conj(w)
                lc = colored_line(ax, np.where(m, sign * w.real + 1j * w.imag, np.nan), np.where(m, k, np.nan), norm)
        dot = dict(ms=7, mfc="white", mec="black", zorder=5, clip_on=False)
        ax.plot(0, 0, "o", **dot)  # circle: kappa_c
        ax.annotate(r"$\kappa_c$", (0, 0), xytext=(7, 4), textcoords="offset points")
        km, ym = kappa_min(p)
        if ym > 0:  # square: the imaginary modes collide and leave the axis
            ax.plot(0, ym, "s", **dot)
            ax.annotate(r"$\kappa_{\min}$", (0, ym), xytext=(7, 5), textcoords="offset points")
        for ks, ws in collisions(p):  # diamond: Krein collision on the real axis
            ax.plot([ws, -ws], [0, 0], "D", **dot)
            ax.annotate(r"$\kappa_s$", (ws, 0), xytext=(5, 7), textcoords="offset points")
        ax.set(xlim=(-2, 2), ylim=(0, 4), xlabel=r"$\mathrm{Re}\,\omega$")
        ax.set_title(rf"({tag}) $M^2={p.M2:g}$", loc="left")
    axes[0].set_ylabel(r"$\mathrm{Im}\,\omega$")
    fig.colorbar(lc, ax=axes, orientation="horizontal", pad=0.24, aspect=35, shrink=0.8, ticks=[-2, -1.5, -1, -0.5, 0], label=r"Wentzell coupling $\kappa$")
    save(fig, name)
def stability_diagram(p, name="10_stability_diagram"):
    """Stable (blue) and unstable (orange) couplings against the bulk mass. The left edge is the BF bound."""
    m2 = np.r_[np.linspace(0.02, 0.5, 15)[:-1] ** 2 - 0.25, np.linspace(0, 0.15, 150)[1:], np.linspace(0.15, 0.5, 60)[1:]]  # dense near the BF bound and where the pair opens
    qs = [replace(p, M2=m) for m in m2]
    km = np.array([kappa_min(q)[0] for q in qs])  # edge of the imaginary-mode region
    lo, hi = np.array([pair_range(q) for q in qs]).T  # edges of the complex-pair region
    i0 = np.flatnonzero(np.isfinite(lo))[0]
    m2p, lo, hi = np.r_[0, m2[i0:]], np.r_[-np.pi / 4, lo[i0:]], np.r_[-np.pi / 4, hi[i0:]]  # the pair opens at M^2 = 0, kappa = -pi/4
    fig, ax = plt.subplots(figsize=W1)
    ax.set_facecolor(tint(BLUE, 0.16))  # stable background
    ax.fill_between(m2, km, 0, color=tint(RED, 0.28), lw=0)
    ax.fill_between(m2p, lo, hi, color=tint(RED, 0.5), lw=0)  # darker, so the two regions differ in lightness too
    for xs, ys in ((m2, km), (m2p, lo), (m2p, hi)):
        ax.plot(xs, ys, color="black", lw=1)
    ax.axhline(0, color="black", lw=1)
    stable, unstable = dict(color="#004A73", ha="center", va="center"), dict(color="#7A3300", ha="center", va="center")
    ax.text(0.125, 0.225, "stable", **stable)
    ax.text(-0.03, -2.3, "stable", **stable)
    ax.text(0.37, -0.45, "unstable,\nimaginary $\\omega$", **unstable)
    ax.text(0.31, -1.4, "unstable,\ncomplex $\\omega$", **unstable)
    ax.plot(0, -np.pi / 4, "o", ms=5, color="black", zorder=5)
    ax.annotate(r"$\kappa=-\pi/4$", (0, -np.pi / 4), (-0.07, -1.3), arrowprops=dict(arrowstyle="-", lw=0.8, shrinkA=2, shrinkB=2), ha="center", va="center")
    ax.set(xlim=(-0.25, 0.5), ylim=(-3.5, 0.45), xlabel=r"effective bulk mass $M^2$", ylabel=r"Wentzell coupling $\kappa$")
    ax.set_xticks([-0.25, 0, 0.25, 0.5], ["$-0.25$", "0", "0.25", "0.5"])
    ax.set_yticks([-3, -2, -1, 0])
    save(fig, name)
def bifurcation_diagram(p, name="11_bifurcation_diagram"):
    """Re w and Im w of the modes against kappa; the Krein collision is the fold at kappa_s."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(W1[0], 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.1})
    spectrum_curves(top, p, wmax=3.2)
    for i, (w, k) in enumerate(locus(p)):
        if i:  # branch 0 is the imaginary axis, where Re w = 0
            top.plot(k, w.real, "--", color=RED)
        bot.plot(k, w.imag, "--", color=RED)
    ks, ws = collisions(p)[0]
    km = kappa_min(p)[0]
    top.plot(ks, ws, "D", ms=8, mfc="white", mec="black", zorder=5)
    for ax in (top, bot):
        for k in (ks, km):
            ax.axvline(k, color="gray", lw=0.9, ls=":")
    for k, tex in ((ks, r"$\kappa_s$"), (km, r"$\kappa_{\min}$")):
        top.text(k, 1.03, tex, transform=top.get_xaxis_transform(), ha="center")
    top.plot([], [], color=BLUE, label="real modes")
    top.plot([], [], "--", color=RED, label="growing modes")
    top.set(xlim=(-2.5, 0.5), xlabel="", ylabel=r"$\mathrm{Re}\,\omega$")
    bot.set(ylim=(0, 3), xlabel=r"Wentzell coupling $\kappa$", ylabel=r"$\mathrm{Im}\,\omega$")
    fig.legend(*top.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=False)
    save(fig, name)
def mode_profiles(p, name="12_mode_profiles", n=3):
    """Normalized radial profiles of the first real modes; dots are the boundary values F."""
    r = np.linspace(0, np.pi / 2, 600)[:-1]
    fig, ax = plt.subplots(figsize=W1)
    for c, ls, w in zip([BLUE, RED, GREEN], LINES, real_roots(p)[:n]):
        f = np.sin(r) ** p.ell * np.cos(r) ** (2 * p.nu) * hyp2f1(p.s - w / 2, p.s + w / 2, p.ell + p.d / 2, np.sin(r) ** 2)
        f /= np.max(np.abs(f))
        ax.plot(r, f, ls, color=c, label=rf"$\omega={w:.3f}$")
        ax.plot(np.pi / 2, f[-1], "o", ms=7, color=c, mec="black", clip_on=False)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set(xlim=(0, np.pi / 2), xlabel=r"radial coordinate $r$", ylabel="radial profile (normalized)")
    ax.set_xticks([0, np.pi / 4, np.pi / 2], ["$0$", r"$\pi/4$", r"$\pi/2$"])
    ax.legend(loc="upper right", frameon=False)
    save(fig, name)
# --- run
def report(p):
    """Print the numbers quoted in the paper."""
    km, ym = kappa_min(p)
    print(f"  M^2 = {p.M2:g} (nu = {p.nu:.6f}), kappa = {p.kappa:g}")
    print("    roots:     " + ", ".join(f"{r:.6f}" for r in real_roots(p)[:5]))
    print(f"    kappa_c   = {kappa_c(p):.6f}")
    print(f"    kappa_min = {km:.6f} at Im w = {ym:.4f}")
    for ks, ws in collisions(p):
        print(f"    Krein collision: kappa_s = {ks:.6f} at w = {ws:.6f}")
    print(f"    growing modes for {window(p):.6f} < kappa < 0")
def main():
    massive = P()
    massless = replace(massive, M2=0.0)
    report(massless)
    report(massive)
    plot_intersections(massless, "01_massless_intersections")
    plot_spectrum_vs_kappa(massless, massive, "02_spectrum_vs_kappa")
    plot_intersections(massive, "04_massive_intersections")
    plot_sweeps(massless, massive)
    plot_growth_rate(massless, massive, "08_growth_rate_vs_kappa")
    plot_complex_plane(massless, massive, "09_growing_modes_complex_plane")
    stability_diagram(massive)
    bifurcation_diagram(massive)
    mode_profiles(massive)
if __name__ == "__main__":
    main()
