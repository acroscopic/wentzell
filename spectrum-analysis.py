#!/usr/bin/env python3
"""
Spectrum analysis for a Klein-Gordon field with dynamical Wentzell boundary
conditions in global AdS.

General quantization condition, Eq. (37) of the preprint:

    kappa (omega^2 - Lambda) = 2 nu Gamma(-nu) / Gamma(nu)
                               * Gamma(s - omega/2) Gamma(s + omega/2)
                               / [Gamma(t - omega/2) Gamma(t + omega/2)]

where
    nu     = sqrt(1 + 4 M^2) / 2
    s      = (2 ell + d + 2 nu) / 4
    t      = (2 ell + d - 2 nu) / 4
    Lambda = ell (ell + d - 2) + m_b^2 + xi_b R_b,   with R_b = (d - 1)(d - 2)

For M^2 = 0, d = 3 and ell = 0 this reduces to the cotangent form, Eq. (40):

    kappa (omega^2 - Lambda) = -omega cot(pi omega / 2)

Run the script and press Enter at every prompt to reproduce the figures and
roots in the preprint. All figures are saved to the "generated_graphs" folder.
"""

import cmath
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

OUTPUT_FOLDER = Path(__file__).resolve().parent / "generated_graphs"

# Real frequencies are searched on 0 < omega < OMEGA_MAX.
OMEGA_MAX = 14.0
SAMPLES_PER_UNIT = 280

# Values used by the parameter sweeps.
KAPPA_SWEEP = [-2.5 + 0.1 * i for i in range(51)]
BOUNDARY_MASS_SWEEP = [0.0, 0.5, 1.0, 2.0]
XI_B_SWEEP = [-0.5, 0.0, 0.5, 1.0]
BULK_MASS_SWEEP = [0.0, 0.10, 0.24, 0.50]
ELL_SWEEP = [0, 1, 2, 3]

# Colors match the figures in the preprint.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GOLD = "#E69F00"
GREEN = "#009E73"
PALETTE = [BLUE, ORANGE, GREEN, "#CC79A7"]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# -----------------------------------------------------------------------------
# Physical parameters
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameters:
    d: int = 3
    ell: int = 0
    kappa: float = 1.0
    mb2: float = 1.0
    xi_b: float = 0.0
    M2: float = 0.24

    @property
    def nu(self):
        return math.sqrt(1 + 4 * self.M2) / 2

    @property
    def s(self):
        return (2 * self.ell + self.d + 2 * self.nu) / 4

    @property
    def t(self):
        return (2 * self.ell + self.d - 2 * self.nu) / 4

    @property
    def Lambda(self):
        R_b = (self.d - 1) * (self.d - 2)
        return self.ell * (self.ell + self.d - 2) + self.mb2 + self.xi_b * R_b

    def label(self):
        return (rf"$d={self.d}$, $\ell={self.ell}$, $\kappa={self.kappa:g}$, "
                rf"$m_b^2={self.mb2:g}$, $\xi_b={self.xi_b:g}$, $M^2={self.M2:g}$")


# -----------------------------------------------------------------------------
# The Gamma function
# -----------------------------------------------------------------------------

# Lanczos coefficients for log Gamma(z). They let us evaluate Gamma at complex
# arguments, which we need when searching for growing (imaginary) modes.
LANCZOS_G = 7.0
LANCZOS_COEFFICIENTS = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
]


def log_gamma(z):
    """log Gamma(z) using the Lanczos approximation. Valid for Re(z) >= 0.5."""
    z -= 1
    series = LANCZOS_COEFFICIENTS[0]
    for k in range(1, len(LANCZOS_COEFFICIENTS)):
        series += LANCZOS_COEFFICIENTS[k] / (z + k)
    w = z + LANCZOS_G + 0.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * cmath.log(w) - w + cmath.log(series)


def rgamma(z):
    """1 / Gamma(z) for any complex z.

    The reciprocal has no poles. It is simply zero at z = 0, -1, -2, ...
    For Re(z) < 0.5 we step z up with 1/Gamma(z) = z / Gamma(z + 1).
    """
    z = complex(z)
    product = 1
    while z.real < 0.5:
        product *= z
        z += 1
    return product * cmath.exp(-log_gamma(z))


# -----------------------------------------------------------------------------
# The quantization condition
# -----------------------------------------------------------------------------

def gamma_terms(omega, p):
    """The two Gamma pieces of Eq. (37), written with 1/Gamma so they never blow up.

    a = 2 nu Gamma(-nu)/Gamma(nu) / [Gamma(t - omega/2) Gamma(t + omega/2)]
    b = 1 / [Gamma(s - omega/2) Gamma(s + omega/2)]

    The right-hand side of Eq. (37) is then a / b.
    """
    prefactor = 2 * p.nu * math.gamma(-p.nu) / math.gamma(p.nu)
    a = prefactor * rgamma(p.t - omega / 2) * rgamma(p.t + omega / 2)
    b = rgamma(p.s - omega / 2) * rgamma(p.s + omega / 2)
    return a, b


def boundary_side(omega, p):
    """Left-hand side of Eq. (37): kappa (omega^2 - Lambda)."""
    return p.kappa * (omega * omega - p.Lambda)


def gamma_side(omega, p):
    """Right-hand side of Eq. (37). Returns NaN at a pole so plots break there."""
    a, b = gamma_terms(omega, p)
    if b == 0:
        return math.nan
    return (a / b).real


def residual(omega, p):
    """Eq. (37) multiplied through by b, so that it has no poles.

    The roots of this function are exactly the quantized frequencies. Because
    it is smooth everywhere, a sign change always means a real root and never
    a pole. It also works for complex omega.
    """
    a, b = gamma_terms(omega, p)
    return boundary_side(omega, p) * b - a


# -----------------------------------------------------------------------------
# Real roots
# -----------------------------------------------------------------------------

def bisect(p, left, right):
    """Shrink a sign-changing bracket [left, right] down to the root."""
    f_left = residual(left, p).real
    for _ in range(60):
        middle = (left + right) / 2
        f_middle = residual(middle, p).real
        if f_left * f_middle <= 0:
            right = middle
        else:
            left, f_left = middle, f_middle
    return (left + right) / 2


def real_roots(p, omega_max=OMEGA_MAX):
    """All positive real roots below omega_max, smallest first.

    We sample the residual on a fine grid and bisect every interval where it
    changes sign. Only positive roots are needed since Eq. (37) is even in omega.
    """
    count = int(omega_max * SAMPLES_PER_UNIT)
    omegas = [1e-6 + omega_max * i / count for i in range(count + 1)]
    values = [residual(w, p).real for w in omegas]

    roots = []
    for i in range(count):
        if values[i + 1] == 0:
            roots.append(omegas[i + 1])
        elif values[i] * values[i + 1] < 0:
            roots.append(bisect(p, omegas[i], omegas[i + 1]))
    return roots


# -----------------------------------------------------------------------------
# Stability: growing modes and the critical coupling
# -----------------------------------------------------------------------------

def critical_kappa(p):
    """kappa_c from Eq. (57). A mode passes through omega = 0 at this coupling."""
    if p.Lambda <= 0:
        return math.inf
    ratio = math.gamma(p.s) / math.gamma(p.t)
    return 2 / p.Lambda * math.gamma(1 - p.nu) / math.gamma(p.nu) * ratio ** 2


def secant(p, z0, z1):
    """Secant method in the complex plane. Returns None if it does not converge."""
    f0, f1 = residual(z0, p), residual(z1, p)
    for _ in range(60):
        if f1 == f0 or abs(z1) > 1e3:
            return None
        z0, z1 = z1, z1 - f1 * (z1 - z0) / (f1 - f0)
        if abs(z1 - z0) < 1e-12 * max(1, abs(z1)):
            return z1
        f0, f1 = f1, residual(z1, p)
    return None


def growing_modes(p):
    """Roots with Im(omega) > 0, most unstable first.

    A mode e^{-i omega t} with Im(omega) > 0 grows exponentially in time.
    We start the secant method from a grid of points in the upper half plane
    and keep every distinct root it lands on.
    """
    modes = []
    for x in [0.0, 0.8, 1.6, 2.4, 3.2, 4.0]:
        for y in [0.2, 0.7, 1.5, 3.0, 6.0, 12.0, 24.0]:
            try:
                root = secant(p, complex(x, y), complex(x, y + 0.05))
            except (OverflowError, ZeroDivisionError):
                continue
            if root is None or not 1e-7 < root.imag <= 24:
                continue

            # Make sure the secant method found a real root and did not just stall.
            a, b = gamma_terms(root, p)
            lhs = boundary_side(root, p) * b
            if abs(lhs - a) > 1e-8 * (abs(lhs) + abs(a)):
                continue

            # Roots come in pairs omega and -conj(omega), so keep Re(omega) >= 0.
            root = complex(abs(root.real) if abs(root.real) > 1e-9 else 0.0, root.imag)
            if all(abs(root - known) > 1e-6 for known in modes):
                modes.append(root)

    return sorted(modes, key=lambda z: -z.imag)


def is_unstable(p):
    """True if there is a growing mode at this coupling.

    For 0 < kappa < kappa_c an imaginary root is guaranteed, even if the
    secant search happens to miss it.
    """
    return bool(growing_modes(p)) or 0 < p.kappa < critical_kappa(p)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def sample(function, lo, hi, clip=None, points_per_unit=180):
    """Evaluate function on [lo, hi]. Values beyond +-clip become NaN so the
    curve breaks cleanly at poles instead of drawing a vertical line."""
    count = int((hi - lo) * points_per_unit)
    xs = [lo + (hi - lo) * i / count for i in range(count + 1)]
    ys = [function(x) for x in xs]
    if clip is not None:
        ys = [y if abs(y) < clip else math.nan for y in ys]
    return xs, ys


def save(fig, name):
    fig.tight_layout()
    path = OUTPUT_FOLDER / f"{name}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"  saved {path.name}")


def legend_above(ax, title=None, ncol=3):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol,
              frameon=False, title=title, alignment="left")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def plot_intersections(p, name, omega_max=8.5):
    """Both sides of the quantization condition. They cross at the roots (Figure 1)."""
    if p.M2 == 0 and p.d == 3 and p.ell == 0:
        gamma_label = r"$-\omega\cot(\pi\omega/2)$"
    else:
        gamma_label = r"$2\nu\,A(\omega)/B(\omega)$"

    xs, gamma_ys = sample(lambda w: gamma_side(w, p), 0, omega_max, clip=300)
    boundary_ys = [boundary_side(w, p) for w in xs]
    roots = real_roots(p, omega_max)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="gray", lw=0.6)
    ax.plot(xs, gamma_ys, color=BLUE, label=gamma_label)
    ax.plot(xs, boundary_ys, "--", color=ORANGE,
            label=rf"$\kappa(\omega^2-\Lambda)$, $\Lambda={p.Lambda:g}$")
    ax.plot(roots, [boundary_side(r, p) for r in roots], "o", color=GOLD,
            mec="black", zorder=5, label=r"quantized frequencies $\omega_n$")
    for n, r in enumerate(roots, start=1):
        ax.axvline(r, color="gray", ls=":", lw=0.7)
        ax.annotate(rf"$\omega_{n}$", (r, boundary_side(r, p)), xytext=(4, -12),
                    textcoords="offset points", fontsize=8)

    ax.set_xlim(0, omega_max)
    ax.set_ylim(-15, 70)
    ax.set_xlabel(r"frequency $\omega$")
    ax.set_ylabel("both sides of the condition")
    legend_above(ax, p.label())
    save(fig, name)


def plot_spectrum_vs_kappa(p, name):
    """Real roots as kappa varies, with the growing-mode window shaded (Figure 2)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="gray", lw=0.6)

    unstable_kappas = []
    for kappa in KAPPA_SWEEP:
        q = replace(p, kappa=kappa)
        unstable = is_unstable(q)
        if unstable:
            unstable_kappas.append(kappa)
        roots = real_roots(q, omega_max=13)
        ax.plot([kappa] * len(roots), roots, "D" if unstable else "o", ms=3,
                color=ORANGE if unstable else BLUE)

    # Growing modes need kappa > 0, and always exist up to kappa_c,
    # so the window runs from 0 to the larger of kappa_c and the last unstable kappa.
    kappa_c = critical_kappa(p)
    if unstable_kappas:
        kappa_max = max(unstable_kappas + [kappa_c])
        ax.axvspan(0, kappa_max, facecolor="none", edgecolor=ORANGE, hatch="///",
                   alpha=0.4, label=r"growing mode, $0<\kappa<\kappa_{\max}$")

    if KAPPA_SWEEP[0] < kappa_c < KAPPA_SWEEP[-1]:
        ax.axvline(kappa_c, ls="--", color=ORANGE, label=rf"$\kappa_c={kappa_c:.3f}$")

    ax.set_xlabel(r"Wentzell coupling $\kappa$")
    ax.set_ylabel(r"positive root $\omega_n$")
    legend_above(ax, rf"$d={p.d}$, $\ell={p.ell}$, $m_b^2={p.mb2:g}$, "
                     rf"$\xi_b={p.xi_b:g}$, $M^2={p.M2:g}$", ncol=2)
    save(fig, name)


def plot_root_sweep(ax, p, field, values, label):
    """Plot the n-th root against n, once for each value of one parameter."""
    for color, value in zip(PALETTE, values):
        roots = real_roots(replace(p, **{field: value}))
        ax.plot(range(1, len(roots) + 1), roots, "o-", ms=3.5, color=color,
                label=label.format(value))
    ax.set_xlabel("root index $n$")
    ax.set_ylabel(r"positive root $\omega_n$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left")


def plot_boundary_sweeps(p, name):
    """How the roots move with the boundary mass and the boundary curvature coupling."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    plot_root_sweep(left, p, "mb2", BOUNDARY_MASS_SWEEP, r"$m_b^2={:g}$")
    plot_root_sweep(right, p, "xi_b", XI_B_SWEEP, r"$\xi_b={:g}$")
    save(fig, name)


def plot_bulk_mass_sweep(p, name):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_root_sweep(ax, p, "M2", BULK_MASS_SWEEP, r"$M^2={:g}$")
    save(fig, name)


def plot_ell_sweep(p, name):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_root_sweep(ax, p, "ell", ELL_SWEEP, r"$\ell={}$")
    save(fig, name)


def plot_growth_rate(massless, massive, name):
    """Im(omega) of the most unstable mode as kappa varies."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, color, style in [(massless, BLUE, "-"), (massive, ORANGE, "--")]:
        rates = []
        for kappa in KAPPA_SWEEP:
            modes = growing_modes(replace(p, kappa=kappa))
            rates.append(modes[0].imag if modes else 0.0)
        ax.plot(KAPPA_SWEEP, rates, style, color=color, label=rf"$M^2={p.M2:g}$")

    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Wentzell coupling $\kappa$")
    ax.set_ylabel(r"growth rate $\mathrm{Im}\,\omega$")
    legend_above(ax, rf"$d={massive.d}$, $\ell={massive.ell}$, "
                     rf"$m_b^2={massive.mb2:g}$, $\xi_b={massive.xi_b:g}$", ncol=2)
    save(fig, name)


def plot_complex_plane(massless, massive, name):
    """Growing modes in the complex omega plane, colored by kappa.

    Purely imaginary modes sit on the vertical axis. Modes off the axis come in
    pairs omega and -conj(omega), so we draw both.
    """
    kappas = [0.02 * i for i in range(1, 131)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)

    for ax, p in zip(axes, [massless, massive]):
        xs, ys, ks = [], [], []
        for kappa in kappas:
            for mode in growing_modes(replace(p, kappa=kappa)):
                for re in {mode.real, -mode.real}:
                    xs.append(re)
                    ys.append(mode.imag)
                    ks.append(kappa)

        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        points = ax.scatter(xs, ys, c=ks, s=10, cmap="viridis")
        ax.set_ylim(0, 6)
        ax.tick_params(labelleft=True)
        ax.set_xlabel(r"$\mathrm{Re}\,\omega$")
        ax.set_title(rf"$M^2={p.M2:g}$, $\Lambda={p.Lambda:g}$", loc="left", fontsize=10)

    axes[0].set_ylabel(r"$\mathrm{Im}\,\omega$")
    axes[1].set_ylabel(r"$\mathrm{Im}\,\omega$")
    fig.colorbar(points, ax=axes, label=r"Wentzell coupling $\kappa$")
    fig.savefig(OUTPUT_FOLDER / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png")


# -----------------------------------------------------------------------------
# User input
# -----------------------------------------------------------------------------

def ask(prompt, default, kind=float):
    """Ask for one value. Pressing Enter keeps the default."""
    while True:
        try:
            text = input(f"  {prompt} [{default}]: ").strip()
        except EOFError:
            return default
        if not text:
            return default
        try:
            return kind(text)
        except ValueError:
            print("    Please enter a number, or press Enter for the default.")


def ask_parameters():
    print("Please enter a number, or press Enter for the default.\n")
    while True:
        p = Parameters(
            d=ask("Boundary dimension d", 3, int),
            ell=ask("Angular number ell", 0, int),
            kappa=ask("Wentzell coupling kappa", 1.0),
            mb2=ask("Boundary mass m_b^2", 1.0),
            xi_b=ask("Boundary curvature coupling xi_b", 0.0),
            M2=ask("Effective bulk mass M^2", 0.24),
        )

        # nu must be real (the BF bound) and not an integer, where Gamma(-nu) blows up.
        if p.M2 < -0.25:
            print("\n  M^2 is below the BF bound M^2 >= -1/4. Try again.\n")
        elif p.nu == round(p.nu):
            print(f"\n  nu = {p.nu:g} is an integer, where Gamma(-nu) is singular. Try again.\n")
        else:
            return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def format_roots(roots):
    return ", ".join(f"{r:.6f}" for r in roots[:5])


def format_modes(modes):
    return ", ".join(f"{z.real:.6f}{z.imag:+.6f}i" for z in modes) or "none"


def main():
    massive = ask_parameters()
    massless = replace(massive, M2=0.0)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    print("\nRoots of the quantization condition")
    print(f"  Massless (M^2 = 0):      {format_roots(real_roots(massless))}")
    print(f"  Massive  (M^2 = {massive.M2:g}): {format_roots(real_roots(massive))}")

    print("\nStability")
    print(f"  Massless kappa_c = {critical_kappa(massless):.6f}")
    print(f"  Massive  kappa_c = {critical_kappa(massive):.6f}")
    print(f"  Massless growing modes: {format_modes(growing_modes(massless))}")
    print(f"  Massive  growing modes: {format_modes(growing_modes(massive))}")

    print(f"\nWriting figures to {OUTPUT_FOLDER}")
    plot_intersections(massless, "01_massless_intersections")
    plot_spectrum_vs_kappa(massless, "02_massless_spectrum_vs_kappa")
    plot_boundary_sweeps(massless, "03_massless_boundary_sweeps")
    plot_intersections(massive, "04_massive_intersections")
    plot_spectrum_vs_kappa(massive, "05_massive_spectrum_vs_kappa")
    plot_bulk_mass_sweep(massive, "06_massive_bulk_mass_sweep")
    plot_ell_sweep(massive, "07_massive_ell_sweep")
    plot_growth_rate(massless, massive, "08_growth_rate_vs_kappa")
    plot_complex_plane(massless, massive, "09_growing_modes_complex_plane")
    print("Done.")


if __name__ == "__main__":
    main()
