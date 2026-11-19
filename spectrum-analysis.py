"""Spectrum and stability of a Klein-Gordon field with Wentzell boundary conditions in AdS_{d+1}.

Quantization condition (w here is omega):
    kappa (w^2 - Lambda) = -A/B
        = -Gamma(-nu)/Gamma(nu) * Gamma(s - w/2) Gamma(s + w/2) / [Gamma(t - w/2) Gamma(t + w/2)]
    nu = sqrt(1 + 4 M^2)/2,   s = (2 ell + d + 2 nu)/4,   t = (2 ell + d - 2 nu)/4,
    Lambda = ell (ell + d - 2) + mb2 + xi_b (d - 1)(d - 2).

Roots are found from D(w) = kappa (w^2 - Lambda) b(w) + a(w), with
    a = Gamma(-nu)/Gamma(nu) / [Gamma(t - w/2) Gamma(t + w/2)],
    b = 1 / [Gamma(s - w/2) Gamma(s + w/2)],
both entire since 1/Gamma is entire.

Searched: real roots in 0 < w < 14; complex roots in 0 < Re w <= 3, 0 < Im w <= 8, enlarged
until no branch reaches the edge. Requires -1/4 < M^2 < 3/4.

Run:  python spectrum_analysis_v2.py   (figures to generated_graphs/, tables to generated_data/)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from contourpy import contour_generator
from scipy.optimize import brentq
from scipy.special import gamma, hyp2f1, rgamma

# ------------------------------------------------------------------ parameters

PAR = dict(d=3, ell=0, kappa=-1.0, mb2=1.0, xi_b=0.0, M2=0.25)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "generated_graphs")
DATA = os.path.join(HERE, "generated_data")

KAPPAS = (-2.5, 2.5)  # kappa range of the sweeps
# M^2 values for the stability diagram, denser near both ends and near 0
M2_GRID = np.r_[np.linspace(0.02, 0.5, 15)[:-1] ** 2 - 0.25,
                np.linspace(0, 0.15, 150)[1:],
                np.linspace(0.15, 0.7, 90)[1:],
                0.75 - np.linspace(np.sqrt(0.05), 0.02, 10)[1:] ** 2]
FINE_BOX = (3.0, 8.0)  # Re w and Im w covered by the fine grid of locus

# Okabe-Ito colors
BLUE, RED, GREEN, PINK, GOLD = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"
COLORS, LINES, DOTS = [BLUE, RED, GREEN, PINK], ["-", "--", "-.", ":"], "osD^"
W1 = (5.5, 3.6)  # one panel at 0.85 textwidth, so text prints at true size
plt.rcParams.update({"font.size": 11, "savefig.dpi": 300})

def derived(par):
    """nu, s, t, Lambda."""
    d, ell, M2 = par["d"], par["ell"], par["M2"]
    if not -0.25 < M2 < 0.75:
        raise ValueError(f"M^2 = {M2} is outside (-1/4, 3/4), so nu is outside (0, 1)")
    nu = np.sqrt(1 + 4 * M2) / 2
    s = (2 * ell + d + 2 * nu) / 4
    t = (2 * ell + d - 2 * nu) / 4
    lam = ell * (ell + d - 2) + par["mb2"] + par["xi_b"] * (d - 1) * (d - 2)
    return nu, s, t, lam


# ------------------------------------------------------- quantization condition

def ab(w, par):
    """a(w), b(w), with A/B = a/b."""
    nu, s, t, _ = derived(par)
    a = gamma(-nu) / gamma(nu) * rgamma(t - w / 2) * rgamma(t + w / 2)
    b = rgamma(s - w / 2) * rgamma(s + w / 2)
    return a, b

def residual(w, par, kappa):
    """D(w)."""
    a, b = ab(w, par)
    return kappa * (w * w - derived(par)[3]) * b + a

def kappa_of(w, par):
    """kappa(w) = -a / [(w^2 - Lambda) b]."""
    a, b = ab(w, par)
    return -a / ((w * w - derived(par)[3]) * b)

def slope(f, x, h=1e-6):
    """Central difference f'(x)."""
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------- root finding

def real_roots(par, kappa=None, wmax=14.0, n=6000, mult=False):
    """Real roots of D in (0, wmax); mult=True also returns multiplicities (1 or 2)."""
    kappa = par["kappa"] if kappa is None else kappa
    def f(x):
        return residual(x, par, kappa).real
    w = np.linspace(1e-6, wmax, n)
    fw = f(w)
    found = [(brentq(f, w[j], w[j + 1]), 1) for j in np.flatnonzero(fw[:-1] * fw[1:] < 0)]
    for j in np.flatnonzero(fw == 0):  # exact zeros on the grid
        touches = 0 < j < n - 1 and fw[j - 1] * fw[j + 1] > 0
        found.append((w[j], 2 if touches else 1))
    # local minima of |D| without a sign change
    same_sign = (fw[:-2] * fw[1:-1] > 0) & (fw[1:-1] * fw[2:] > 0)
    low = (abs(fw[1:-1]) <= abs(fw[:-2])) & (abs(fw[1:-1]) <= abs(fw[2:]))
    for j in np.flatnonzero(same_sign & low) + 1:
        found += split_dip(par, kappa, w[j - 1], w[j + 1])
    kd = sqrt_lambda_kappa(par)
    if kd is not None and np.sqrt(derived(par)[3]) < wmax:
        found.append((np.sqrt(derived(par)[3]), 2 if abs(kappa - kd) < 1e-8 * abs(kd) else 1))
    roots, mults = [], []  # merge roots closer than 1e-8
    for r, m in sorted(found):
        if roots and r - roots[-1] < 1e-8:
            mults[-1] = max(mults[-1], m)
        else:
            roots.append(r)
            mults.append(m)
    roots, mults = np.array(roots), np.array(mults, dtype=int)
    return (roots, mults) if mult else roots

def split_dip(par, kappa, lo, hi):
    """Roots of D on (lo, hi) around a minimum of |D|: two, one double, or none."""
    def f(x):
        return residual(x, par, kappa).real
    def df(x):
        return slope(f, x)

    if df(lo) * df(hi) >= 0:
        return []
    x = brentq(df, lo, hi)  # extremum of D
    if f(x) * f(lo) < 0:
        return [(brentq(f, lo, x), 1), (brentq(f, x, hi), 1)]
    a, b = ab(x, par)
    # scale of the two terms of D
    size = abs(kappa * b) * (x * x + abs(derived(par)[3])) + abs(a)
    return [(x, 2)] if abs(f(x)) < 1e-10 * size else []

def sqrt_lambda_kappa(par):
    """kappa where dD/dw = 0 at w = sqrt(Lambda), if a(sqrt(Lambda)) = 0; else None.

    a(sqrt(Lambda)) = 0 when t - sqrt(Lambda)/2 is a non-positive integer (1/Gamma vanishes there).
    """
    _, _, t, lam = derived(par)
    if lam <= 0:
        return None
    r = np.sqrt(lam)
    m = r / 2 - t
    if m < -1e-9 or abs(m - round(m)) > 1e-9:
        return None
    da = slope(lambda x: ab(x, par)[0], r)
    return -da / (2 * r * ab(r, par)[1])  # dD/dw = 2 kappa w b + a' at w^2 = Lambda


# --------------------------------------------------------- instability windows

def locus(par, n=500, max_side=200.0):
    """Branches (w, kappa(w)) with Im w > 0 and kappa real.

    Branch 0 is the imaginary axis, the rest the zero contour of Im kappa. The box starts at
    FINE_BOX (n points per side), at least twice the Im w of kappa_min high; a side doubles
    (n/2 more points) while a branch reaches it.
    """
    def grid(fine_end, end, points):
        g = np.linspace(1e-3, fine_end, points)
        return g if end <= fine_end else np.r_[g, np.linspace(fine_end, end, points // 2 + 1)[1:]]
    re_max, im_max = FINE_BOX[0], max(FINE_BOX[1], 2 * kappa_min(par)[1])
    while True:
        x, y = grid(FINE_BOX[0], re_max, n), grid(FINE_BOX[1], im_max, n)
        axis = np.r_[1e-4, grid(FINE_BOX[1], im_max, 2 * n)]
        branches = [(1j * axis, kappa_of(1j * axis, par).real)]
        W = x + 1j * y[:, None]
        a, b = ab(W, par)
        im = (-a * np.conj((W * W - derived(par)[3]) * b)).imag  # |den|^2 Im kappa
        for seg in contour_generator(x, y, im).lines(0):
            w = seg[:, 0] + 1j * seg[:, 1]
            branches.append((w, kappa_of(w, par).real))
        right = any(np.any(w.real > x[-2]) for w, _ in branches[1:])  # within one grid step
        top = any(np.any(w.imag > y[-2]) for w, _ in branches[1:])
        if not (right or top):
            return branches
        re_max, im_max = re_max * (2 if right else 1), im_max * (2 if top else 1)
        if max(re_max, im_max) > max_side:
            raise ValueError(f"a branch still leaves the box {re_max:g} x {im_max:g} "
                             f"at M^2 = {par['M2']:g}")

def kappa_c(par):
    """kappa(0). Needs Lambda != 0."""
    nu, s, t, lam = derived(par)
    return gamma(-nu) / gamma(nu) * (gamma(s) / gamma(t)) ** 2 / lam

def kappa_min(par, im_max=8.0, n=4000):
    """Minimum of kappa(iy) over y > 0, and the y where it occurs."""
    def k_im(v):
        return kappa_of(1j * v, par).real
    while True:
        y = np.linspace(1e-4, im_max, n)
        j = np.argmin(k_im(y))
        if j < n - 1:
            break
        if im_max > 200:  # safety stop
            raise ValueError(f"no minimum of kappa(i y) below y = {im_max:g}")
        im_max *= 2
    if j == 0:  # minimum at y -> 0
        return kappa_c(par), 0.0
    v = brentq(lambda u: slope(k_im, u), y[j - 1], y[j + 1])
    return k_im(v), v

def pole_cut(par):
    """|kappa| above which a point is treated as a pole."""
    nu = derived(par)[0]
    return 1e3 * abs(gamma(-nu) / gamma(nu))

def collisions(par, branches):
    """[(kappa_s, w_s)]: real w within 0.05 of a branch end where dkappa/dw = 0."""
    def k_re(x):
        return kappa_of(x, par).real
    out = []
    for x in [w[j].real for w, _ in branches[1:] for j in (0, -1) if w[j].imag < 1e-2]:
        if slope(k_re, x - 0.05) * slope(k_re, x + 0.05) < 0:
            ws = brentq(lambda u: slope(k_re, u), x - 0.05, x + 0.05)
            if abs(k_re(ws)) < pole_cut(par):
                out.append((k_re(ws), ws))
    return out

def pair_range(par, branches):
    """Min and max of kappa on the off-axis branches (NaN if none)."""
    ks = np.concatenate([np.empty(0)] + [k[np.abs(k) < pole_cut(par)] for _, k in branches[1:]])
    return (ks.min(), ks.max()) if ks.size else (np.nan, np.nan)

def unstable_intervals(par, branches, tol=2e-3):
    """Sorted disjoint intervals (lo, hi): (kappa_min, 0) joined with pair_range.

    Ends of pair_range within relative tol of kappa_s or kappa_min are set to them.
    """
    km = kappa_min(par)[0]
    spans = [(km, 0.0)]
    lo, hi = pair_range(par, branches)
    if np.isfinite(lo):
        exact = np.array([ks for ks, _ in collisions(par, branches)] + [km])
        snapped = []
        for k in (lo, hi):
            near = exact[np.argmin(np.abs(exact - k))]
            snapped.append(near if abs(near - k) < tol * max(1.0, abs(near)) else k)
        spans.append(tuple(snapped))
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# ----------------------------------------------------------------- data tables

def first(values, n):
    """The first n entries, padded with NaN."""
    values = np.asarray(values, dtype=float)[:n]
    return np.r_[values, np.full(n - len(values), np.nan)]

def as_table(names, rows, ints=()):
    """Rows of numbers as a dict of named columns; the columns in ints become integers."""
    rows = np.array(rows, dtype=float)
    return {c: rows[:, i].astype(int) if c in ints else rows[:, i] for i, c in enumerate(names)}

def roots_vs_kappa(par, kappas, n=5):
    """First n real roots at each kappa, and the number of double roots among them."""
    rows = []
    for k in kappas:
        r, m = real_roots(par, kappa=k, mult=True)
        rows.append(np.r_[k, first(r, n), np.sum(m[:n] == 2)])
    return as_table(["kappa"] + [f"w{i + 1}" for i in range(n)] + ["n_double"], rows,
                    ints=["n_double"])

def thresholds_vs_ell(par, ells, n=3):
    """Lambda, kappa_c, kappa_min and the first n real roots for each ell."""
    rows = []
    for ell in ells:
        q = {**par, "ell": ell}
        km, ym = kappa_min(q)
        rows.append(np.r_[ell, derived(q)[3], kappa_c(q), km, ym, first(real_roots(q), n)])
    names = ["ell", "Lambda", "kappa_c", "kappa_min", "Im_w_min"] + [f"w{i + 1}" for i in range(n)]
    return as_table(names, rows, ints=["ell"])

def stability_table(par, m2_values):
    """kappa_c, kappa_min, pair_range, kappa_s and number of intervals for each M^2."""
    rows = []
    for m2 in m2_values:
        q = {**par, "M2": m2}
        branches = locus(q)
        km, ym = kappa_min(q)
        lo, hi = pair_range(q, branches)
        ks, ws = min(collisions(q, branches), default=(np.nan, np.nan))  # the lowest kappa_s
        n_int = len(unstable_intervals(q, branches))
        rows.append([m2, derived(q)[0], kappa_c(q), km, ym, lo, hi, ks, ws, n_int])
    names = ["M2", "nu", "kappa_c", "kappa_min", "Im_w_min",
             "pair_lo", "pair_hi", "kappa_s", "w_s", "n_intervals"]
    return as_table(names, rows, ints=["n_intervals"])

def save_dat(name, table, notes):
    """Write a dict of equal-length columns as an aligned text table with a '#' header."""
    os.makedirs(DATA, exist_ok=True)
    fmt = ["%15d" if np.issubdtype(col.dtype, np.integer) else "%15.8f" for col in table.values()]
    header_row = " ".join(f"{c:>15s}" for c in table)[2:]  # savetxt prefixes '# '
    np.savetxt(os.path.join(DATA, name + ".dat"), np.column_stack(list(table.values())),
               fmt=fmt, delimiter=" ", header="\n".join(notes + [header_row]))
    print(f"  saved {name}.dat")

def describe(par, skip=()):
    """One header line listing the parameters, leaving out the ones a table varies."""
    return "parameters: " + ", ".join(f"{k} = {v:g}" for k, v in par.items() if k not in skip)

def write_tables(massless, massive, stab):
    kappas = np.unique(np.r_[np.linspace(*KAPPAS, 101), -np.pi / 4])  # -pi/4: massless double root
    for par, tag in [(massless, "massless"), (massive, "massive")]:
        save_dat(f"roots_vs_kappa_{tag}", roots_vs_kappa(par, kappas), [
            f"First five real modes (0 < w < 14) against kappa, M^2 = {par['M2']:g}.",
            describe(par, skip=("kappa",)),
            "A double root is listed once; n_double counts them. nan: fewer than five modes."])
    save_dat("thresholds_vs_ell", thresholds_vs_ell(massive, range(7)), [
        "Thresholds against angular momentum ell.", describe(massive, skip=("ell",)),
        "kappa_c: a mode passes through w = 0.  kappa_min: lowest coupling with an imaginary",
        "mode, which has Im w = Im_w_min (0 if the branch ends at w = 0).  w1..w3: real modes."])
    save_dat("stability_vs_M2", stab, [
        "Stability data against M^2 (the data of 10_stability_diagram).",
        describe(massive, skip=("M2", "kappa")),
        "pair_lo, pair_hi: couplings spanned by the complex pair (nan if none).  kappa_s, w_s:",
        "lowest Krein collision.  n_intervals: number of separate unstable kappa intervals."])


# --------------------------------------------------------------------- figures

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png")

def legend_below(ax, ncol, **kw):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=ncol, frameon=False, **kw)

def tint(hex_color, a):
    """Mix of a color with white; a = 1 is the color."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(1 - a * (1 - rgb))

def plot_intersections(par, name, wmax=8.5):
    """Both sides of the quantization condition against w."""
    kappa, lam = par["kappa"], derived(par)[3]
    roots = real_roots(par, wmax=wmax)
    x = kappa * (roots ** 2 - lam)
    top = roots[-1] + 0.9  # headroom for the top label
    w = np.linspace(0, top, 6000)  # curves run to the top edge
    a, b = ab(w, par)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs = -a / b
    rhs[(np.abs(b) < 1e-3) | (np.abs(rhs) > 300)] = np.nan  # hide the poles
    simple_case = (par["M2"], par["d"], par["ell"]) == (0, 3, 0)

    fig, ax = plt.subplots(figsize=(W1[0], 4.6))
    ax.axvline(0, color="gray", lw=0.6)
    ax.plot(rhs, w, color=BLUE,
            label=r"$\omega\cot(\pi\omega/2)$" if simple_case else r"$-A(\omega)/B(\omega)$")
    ax.plot(kappa * (w * w - lam), w, "--", color=RED, label=r"$\kappa(\omega^2-\Lambda)$")
    ax.plot(x, roots, "o", color=GOLD, mec="black", ms=7, zorder=5, label="modes")
    # labels above-right, or below-left in the right half; roots within 0.5 share one label
    xlim = (-70, 12)
    for group in np.split(np.arange(len(roots)), np.flatnonzero(np.diff(roots) >= 0.5) + 1):
        text = r"$\omega=" + ", ".join(f"{roots[i]:.3f}" for i in group) + "$"
        if x[group[0]] > (xlim[0] + xlim[1]) / 2:
            i, offset, ha, va = group[0], (-6, -6), "right", "top"
        else:
            i, offset, ha, va = group[-1], (8, 9), "left", "bottom"
        ax.annotate(text, (x[i], roots[i]), offset, textcoords="offset points",
                    ha=ha, va=va, fontsize=9)
    ax.set(xlim=xlim, ylim=(0, top), xlabel="value of each side", ylabel=r"frequency $\omega$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    save(fig, name)

def spectrum_curves(ax, par, wmax=13, color=BLUE, lw=2, only=None):
    """Real roots against kappa. With only = [(lo, hi), ...], draw just those kappa."""
    k, w = np.linspace(*KAPPAS, 600), np.linspace(1e-4, wmax, 1200)
    a, b = ab(w, par)
    lam = derived(par)[3]
    # D divided by the scale of its two terms
    norm = np.abs((w * w + lam) * b) + np.abs(a)
    f = (k[None, :] * ((w * w - lam) * b)[:, None] + a[:, None]) / norm[:, None]
    for seg in contour_generator(k, w, f).lines(0):
        kk, ww = seg[:, 0], seg[:, 1].copy()
        if only is not None:
            ww[~np.any([(kk > lo) & (kk < hi) for lo, hi in only], axis=0)] = np.nan
        ax.plot(kk, ww, color=color, lw=lw)
    ax.set(xlim=KAPPAS, ylim=(0, wmax),
           xlabel=r"Wentzell coupling $\kappa$", ylabel=r"real frequency $\omega_n$")

def plot_spectrum_vs_kappa(cases, name):
    """Real roots against kappa for two masses; red inside unstable_intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(W1[0], 3.5), sharey=True)
    for ax, (par, branches), tag in zip(axes, cases, "ab"):
        spans, ks = unstable_intervals(par, branches), collisions(par, branches)
        for lo, hi in spans:
            ax.axvspan(lo, hi, color=RED, alpha=0.09, lw=0)
            # bracket under the axis
            ax.plot([lo, hi], [-0.045, -0.045], transform=ax.get_xaxis_transform(), color=RED,
                    lw=2, marker="|", ms=8, mew=2, clip_on=False)
        spectrum_curves(ax, par, wmax=8, lw=1.8)
        spectrum_curves(ax, par, wmax=8, color=RED, lw=1.8, only=spans)
        marks = [(0, "$0$"), (kappa_c(par), r"$\kappa_c$")] + [(k, r"$\kappa_s$") for k, _ in ks]
        for k, tex in marks:
            ax.axvline(k, ls=":", color="black", lw=1.3)
            ax.text(k, 1.02, tex, transform=ax.get_xaxis_transform(), ha="center")
        for k, w in ks:
            ax.plot(k, w, "o", ms=3.5, color="black", zorder=5)
        ax.tick_params(axis="x", pad=9)  # room for the brackets
        ax.set_title(rf"({tag}) $M^2={par['M2']:g}$", loc="left", pad=16)
        ax.set(xlim=KAPPAS, ylim=(0, 8), ylabel="")
    axes[0].set_ylabel(r"real frequency $\omega_n$")
    handles = [plt.Line2D([], [], color=BLUE, lw=1.8),
               plt.Line2D([0, 1], [0, 0], color=RED, lw=2, marker="|", ms=8, mew=2),
               plt.Line2D([], [], ls=":", color="black", lw=1.3),
               plt.Line2D([], [], ls="", marker="o", ms=3.5, color="black")]
    labels = ["real roots", "unstable intervals", r"$0$, $\kappa_c$, $\kappa_s$", "roots merge"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2,
               frameon=False, fontsize=10)
    save(fig, name)

def plot_root_sweep(ax, par, field, values):
    """n-th real root against n for several values of one parameter."""
    for c, ls, m, v in zip(COLORS, LINES, DOTS, values):
        r = real_roots({**par, field: v})
        ax.plot(np.arange(1, len(r) + 1), r, ls, marker=m, ms=5, color=c, label=f"${v:g}$")
    ax.set(xlabel="mode index $n$", ylabel=r"real frequency $\omega_n$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

def plot_sweeps(massless, massive):
    """Sweeps of mb2 and xi_b (massless), and of M2 and ell (massive)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(W1[0], 3.2), sharey=True)
    plot_root_sweep(left, massless, "mb2", [0, 0.5, 1, 2])
    plot_root_sweep(right, massless, "xi_b", [-0.5, 0, 0.5, 1])
    right.set_ylabel("")
    legend_below(left, 2, title=r"$m_b^2$")
    legend_below(right, 2, title=r"$\xi_b$")
    save(fig, "04_massless_boundary_sweeps")
    for name, field, values, title in [
            ("06_massive_bulk_mass_sweep", "M2", [-0.2, 0, 0.25, 0.7], r"$M^2$"),
            ("07_massive_ell_sweep", "ell", [0, 1, 2, 3], r"$\ell$")]:
        fig, ax = plt.subplots(figsize=W1)
        plot_root_sweep(ax, massive, field, values)
        ax.legend(loc="upper left", frameon=False, title=title)
        save(fig, name)

def plot_growth_rate(cases, name):
    """Im w against kappa on each branch."""
    fig, ax = plt.subplots(figsize=W1)
    for (par, branches), c, ls in zip(cases, [BLUE, RED], ["-", "--"]):
        for i, (w, k) in enumerate(branches):
            ax.plot(k, w.imag, ls, color=c, label=rf"$M^2={par['M2']:g}$" if i == 0 else None)
    ax.set(xlim=(KAPPAS[0], 0.1), ylim=(0, 6), xlabel=r"Wentzell coupling $\kappa$",
           ylabel=r"growth rate $\mathrm{Im}\,\omega$")
    ax.legend(frameon=False)
    save(fig, name)

def plot_complex_plane(cases, name, kmin=-2.0):
    """Branches in the complex w plane, colored by kappa."""
    fig, axes = plt.subplots(1, 2, figsize=W1, sharey=True)
    dot = dict(ms=7, mfc="white", mec="black", zorder=5, clip_on=False)
    for ax, (par, branches), tag in zip(axes, cases, "ab"):
        for w, k in branches:
            m = (k >= kmin) & (k < 0)
            for sign in (1, -1):  # w and -conj(w)
                points = ax.scatter(sign * w.real[m], w.imag[m], c=k[m], s=6, lw=0,
                                    cmap="viridis", vmin=kmin, vmax=0)
        ax.plot(0, 0, "o", **dot)
        ax.annotate(r"$\kappa_c$", (0, 0), xytext=(7, 4), textcoords="offset points")
        ym = kappa_min(par)[1]
        if ym > 0:
            ax.plot(0, ym, "s", **dot)
            ax.annotate(r"$\kappa_{\min}$", (0, ym), xytext=(7, 5), textcoords="offset points")
        for _, ws in collisions(par, branches):
            ax.plot([ws, -ws], [0, 0], "D", **dot)
            ax.annotate(r"$\kappa_s$", (ws, 0), xytext=(5, 7), textcoords="offset points")
        ax.set(xlim=(-2, 2), ylim=(0, 4), xlabel=r"$\mathrm{Re}\,\omega$")
        ax.set_title(rf"({tag}) $M^2={par['M2']:g}$", loc="left")
    axes[0].set_ylabel(r"$\mathrm{Im}\,\omega$")
    fig.colorbar(points, ax=axes, orientation="horizontal", pad=0.24, aspect=35, shrink=0.8,
                 ticks=[-2, -1.5, -1, -0.5, 0], label=r"Wentzell coupling $\kappa$")
    save(fig, name)

def stability_diagram(stab, name="10_stability_diagram"):
    """unstable_intervals in the (M^2, kappa) plane."""
    m2, km, lo, hi = stab["M2"], stab["kappa_min"], stab["pair_lo"], stab["pair_hi"]
    i0 = np.flatnonzero(np.isfinite(lo))[0]
    # pair_range curves start at the added point (0, -pi/4)
    m2p = np.r_[0, m2[i0:]]
    lop, hip = np.r_[-np.pi / 4, lo[i0:]], np.r_[-np.pi / 4, hi[i0:]]

    fig, ax = plt.subplots(figsize=W1)
    ax.fill_between([-0.25, 0.75], -3.5, 0.45, color=tint(BLUE, 0.16), lw=0)
    ax.fill_between(m2, km, 0, color=tint(RED, 0.28), lw=0)
    ax.fill_between(m2p, lop, hip, color=tint(RED, 0.5), lw=0)
    for xs, ys in ((m2, km), (m2p, lop), (m2p, hip)):
        ax.plot(xs, ys, color="black", lw=1)
    ax.axhline(0, color="black", lw=1)
    for x, y, text, color in [(0.25, 0.225, "stable", "#004A73"),
                              (-0.03, -2.3, "stable", "#004A73"),
                              (0.25, -0.36, "unstable,\nimaginary $\\omega$", "#7A3300"),
                              (0.31, -1.33, "unstable,\ncomplex $\\omega$", "#7A3300")]:
        ax.text(x, y, text, color=color, ha="center", va="center")
    ax.plot(0, -np.pi / 4, "o", ms=5, color="black", zorder=5, label=r"$\kappa=-\pi/4$")
    ax.set(xlim=(-0.25, 0.75), ylim=(-3.5, 0.45), xlabel=r"effective bulk mass $M^2$",
           ylabel=r"Wentzell coupling $\kappa$")
    ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75])
    ax.set_yticks([-3, -2, -1, 0])
    legend_below(ax, 1)
    save(fig, name)

def bifurcation_diagram(par, branches, name="11_bifurcation_diagram"):
    """Re w and Im w against kappa."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(W1[0], 4.8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.1})
    spectrum_curves(top, par, wmax=3.2)
    for i, (w, k) in enumerate(branches):
        if i > 0:  # skip the imaginary axis
            top.plot(k, w.real, "--", color=RED)
        bot.plot(k, w.imag, "--", color=RED)
    ks, ws = collisions(par, branches)[0]
    km = kappa_min(par)[0]
    top.plot(ks, ws, "D", ms=8, mfc="white", mec="black", zorder=5)
    for k, tex in ((ks, r"$\kappa_s$"), (km, r"$\kappa_{\min}$")):
        top.axvline(k, color="gray", lw=0.9, ls=":")
        bot.axvline(k, color="gray", lw=0.9, ls=":")
        top.text(k, 1.03, tex, transform=top.get_xaxis_transform(), ha="center")
    top.plot([], [], color=BLUE, label="real modes")
    top.plot([], [], "--", color=RED, label="growing modes")
    top.set(xlim=(-2.5, 0.5), xlabel="", ylabel=r"$\mathrm{Re}\,\omega$")
    bot.set(ylim=(0, 3), xlabel=r"Wentzell coupling $\kappa$", ylabel=r"$\mathrm{Im}\,\omega$")
    fig.legend(*top.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=False)
    save(fig, name)

def mode_profiles(par, name="12_mode_profiles", n=3):
    """sin^ell(r) cos^(2 nu)(r) 2F1(s - w/2, s + w/2; ell + d/2; sin^2 r), scaled to max 1."""
    nu, s, _, _ = derived(par)
    d, ell = par["d"], par["ell"]
    r = np.linspace(0, np.pi / 2, 600)[:-1]
    fig, ax = plt.subplots(figsize=W1)
    for c, ls, w in zip(COLORS, LINES, real_roots(par)[:n]):
        f = (np.sin(r) ** ell * np.cos(r) ** (2 * nu)
             * hyp2f1(s - w / 2, s + w / 2, ell + d / 2, np.sin(r) ** 2))
        f /= np.max(np.abs(f))
        ax.plot(r, f, ls, color=c, label=rf"$\omega={w:.3f}$")
        ax.plot(np.pi / 2, f[-1], "o", ms=7, color=c, mec="black", clip_on=False)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set(xlim=(0, np.pi / 2), xlabel=r"radial coordinate $r$",
           ylabel="radial profile (normalized)")
    ax.set_xticks([0, np.pi / 4, np.pi / 2], ["$0$", r"$\pi/4$", r"$\pi/2$"])
    ax.legend(loc="upper right", frameon=False)
    save(fig, name)


# ------------------------------------------------------------------------ main

def report(par, branches):
    """Print the numbers quoted in the paper."""
    roots, mults = real_roots(par, mult=True)
    km, ym = kappa_min(par)
    listed = [f"{r:.6f}" + (" (x2)" if m == 2 else "") for r, m in zip(roots[:5], mults[:5])]
    print(f"  M^2 = {par['M2']:g} (nu = {derived(par)[0]:.6f}), kappa = {par['kappa']:g}")
    print("    roots:     " + ", ".join(listed))
    print(f"    kappa_c   = {kappa_c(par):.6f}")
    print(f"    kappa_min = {km:.6f} at Im w = {ym:.4f}")
    for ks, ws in collisions(par, branches):
        print(f"    Krein collision: kappa_s = {ks:.6f} at w = {ws:.6f}")
    spans = unstable_intervals(par, branches)
    print("    growing modes for " + " and ".join(f"{a:.6f} < kappa < {b:.6f}" for a, b in spans))

def main():
    massive = dict(PAR)
    massless = {**massive, "M2": 0.0}
    cases = [(massless, locus(massless)), (massive, locus(massive))]
    for par, branches in cases:
        report(par, branches)
    plot_intersections(massless, "01_massless_intersections")
    plot_spectrum_vs_kappa(cases, "02_spectrum_vs_kappa")
    plot_intersections(massive, "03_massive_intersections")
    plot_sweeps(massless, massive)
    plot_growth_rate(cases, "08_growth_rate_vs_kappa")
    plot_complex_plane(cases, "09_growing_modes_complex_plane")
    stab = stability_table(massive, M2_GRID)  # one contour per mass: the slowest step
    stability_diagram(stab)
    bifurcation_diagram(*cases[1])
    mode_profiles(massive)
    write_tables(massless, massive, stab)

if __name__ == "__main__":
    main()
