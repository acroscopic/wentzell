#!/usr/bin/env python3
"""Generate publication-style graphs for Wentzell quantization conditions.

The script follows the conventions in the accompanying thesis notes:

Massless bulk, d=3 and ell=0
    kappa * (omega**2 - mb2 - 2*xi_b)
        = -omega*cot(pi*omega/2)

General massive bulk
    kappa * (omega**2 - ell*(ell+d-2) - mb2 - xi_b*R_b)
        = Gamma(lambda+nu/2-omega/2)
          Gamma(lambda+nu/2+omega/2)
          Gamma(-nu)
          -------------------------------------------------
          Gamma(lambda-nu/2-omega/2)
          Gamma(lambda-nu/2+omega/2)
          Gamma(nu)

where
    lambda = ell/2 + d/4,
    nu     = sqrt(1 + 4*M2)/2,
    R_b    = (d-1)*(d-2) for a unit boundary sphere.

Growing modes (Im omega > 0) are found in the complex plane from the entire
form kappa*(omega**2 - Lambda_b)*B(omega) - A(omega) = 0.

Only Matplotlib is required beyond the Python standard library.  NumPy and
SciPy are deliberately not used.  The non-interactive Agg backend is selected
before pyplot is imported, so running the script never opens a window.

Primary references used for mathematical and plotting conventions:
    https://arxiv.org/abs/1802.00283
    https://arxiv.org/abs/1805.03135
    https://arxiv.org/abs/2004.02704
    https://dlmf.nist.gov/5.2
    https://dlmf.nist.gov/5.5
"""

from __future__ import annotations

import argparse
import cmath
import csv
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Keep Matplotlib's cache in a writable temporary location.  This suppresses a
# harmless warning on read-only or managed systems and makes repeated runs faster.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wentzell_matplotlib_cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logging.getLogger("fontTools").setLevel(logging.ERROR)  # cmr10 embedding noise


# =============================================================================
# USER SETTINGS
#
# These are the main values to edit when making a new batch of graphs.
# Command-line options for the output folder, formats, DPI, and omega range are
# also available; run `python wentzell_spectrum_plots.py --help`.
# =============================================================================

D = 3
ELL = 0
KAPPA = 1.0
BOUNDARY_MASS_SQUARED = 1.0       # m_b^2 in the equations
XI_B = 0.0
BULK_MASS_SQUARED = 0.24          # M^2; gives nu = 0.7

OMEGA_MIN = 0.0
OMEGA_MAX = 14.0
PLOT_Y_LIMIT = 18.0
POINTS_PER_UNIT = 180
ROOT_SAMPLES_PER_UNIT = 280

# Parameter values used by the batch/sweep figures.
KAPPA_CURVES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
KAPPA_SPECTRUM_MIN = -2.5
KAPPA_SPECTRUM_MAX = 2.5
KAPPA_SPECTRUM_COUNT = 51
BOUNDARY_MASS_CURVES = (0.0, 0.5, 1.0, 2.0)
XI_B_CURVES = (-0.5, 0.0, 0.5, 1.0)
BULK_MASS_SQUARED_CURVES = (0.0, 0.10, 0.24, 0.50)
ELL_CURVES = (0, 1, 2, 3)

# Complex-plane search for growing modes.
GROWTH_KAPPA_COUNT = 201
GROWTH_RE_MAX = 4.0
GROWTH_IM_MAX = 24.0

DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 240
DEFAULT_OUTPUT_FOLDER = "generated_graphs"


# Restrained colors and line styles similar to the cited AdS papers, while
# remaining distinguishable when several curves share an axis.
BLACK = "#161616"
BLUE = "#315A7D"
BRICK = "#9A3D35"
GREEN = "#4D7358"
GOLD = "#A07828"
PURPLE = "#66527C"
GRAY = "#777777"
LIGHT_GRAY = "#D4D4D4"
PALETTE = (BLACK, BLUE, BRICK, GREEN, GOLD, PURPLE)
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1)))

# Lanczos coefficients (g=7) for the complex Gamma function.
LANCZOS_G = 7.0
LANCZOS_COEFFICIENTS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


@dataclass(frozen=True)
class Parameters:
    """Physical parameters appearing in the quantization conditions."""

    d: int = D
    ell: int = ELL
    kappa: float = KAPPA
    mb2: float = BOUNDARY_MASS_SQUARED
    xi_b: float = XI_B
    bulk_m2: float = BULK_MASS_SQUARED

    @property
    def boundary_curvature(self) -> float:
        """Ricci scalar R_b for R_t x S^(d-1) with unit sphere radius."""

        return float((self.d - 1) * (self.d - 2))

    @property
    def lam(self) -> float:
        """lambda = ell/2 + d/4."""

        return self.ell / 2.0 + self.d / 4.0

    @property
    def nu(self) -> float:
        """nu = sqrt(1 + 4*M^2)/2, restricted here to real nu."""

        radicand = 1.0 + 4.0 * self.bulk_m2
        if radicand < 0.0:
            raise ValueError(
                "The script currently plots real frequencies with real nu, so "
                "BULK_MASS_SQUARED must be at least -1/4."
            )
        return 0.5 * math.sqrt(radicand)

    @property
    def angular_eigenvalue(self) -> float:
        return float(self.ell * (self.ell + self.d - 2))

    @property
    def boundary_offset(self) -> float:
        """ell(ell+d-2) + m_b^2 + xi_b R_b."""

        return (
            self.angular_eigenvalue
            + self.mb2
            + self.xi_b * self.boundary_curvature
        )


@dataclass(frozen=True)
class RunSettings:
    output_dir: Path
    formats: tuple[str, ...]
    dpi: int
    omega_min: float
    omega_max: float
    y_limit: float


class Reporter:
    """Collect generated paths and print one clear line per output file."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.generated: list[Path] = []

    def record(self, path: Path, kind: str) -> None:
        resolved = path.resolve()
        self.generated.append(resolved)
        print(f"  [{kind:<8}] {path.name}")
        print(f"             {resolved}")


def configure_matplotlib() -> None:
    """Set a calm, paper-like Matplotlib style."""

    plt.rcParams.update(
        {
            "figure.figsize": (5.8, 3.7),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "legend.fontsize": 9.0,
            "legend.title_fontsize": 9.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Numerical functions
# =============================================================================


def linspace(start: float, stop: float, count: int) -> list[float]:
    """Small standard-library replacement for numpy.linspace."""

    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def is_nonpositive_integer(value: float, tolerance: float = 2.0e-12) -> bool:
    """True at the real poles 0, -1, -2, ... of Gamma."""

    if value > tolerance:
        return False
    nearest = round(value)
    return nearest <= 0 and abs(value - nearest) <= tolerance


def gamma_logabs_and_sign(value: float) -> tuple[float, int] | None:
    """Return (log(abs(Gamma(value))), sign(Gamma(value))).

    math.lgamma is stable for very large products but discards the sign.  For a
    negative, noninteger real argument, the reflection formula shows that the
    sign of Gamma(x) equals the sign of sin(pi*x).  None denotes a Gamma pole.
    """

    if is_nonpositive_integer(value):
        return None

    try:
        log_abs = math.lgamma(value)
    except (OverflowError, ValueError):
        return None

    if value > 0.0:
        sign = 1
    else:
        sine = math.sin(math.pi * value)
        if sine == 0.0:
            return None
        sign = 1 if sine > 0.0 else -1
    return log_abs, sign


def signed_gamma_ratio(
    numerator_arguments: Sequence[float],
    denominator_arguments: Sequence[float],
) -> float:
    """Evaluate a real Gamma product ratio using signs and logarithms.

    A denominator Gamma pole is a zero of the reciprocal Gamma function, so the
    ratio is returned as zero when the numerator is finite there.  A numerator
    pole is represented by NaN so Matplotlib breaks the curve at that point.
    """

    numerator_data = [gamma_logabs_and_sign(x) for x in numerator_arguments]
    denominator_data = [gamma_logabs_and_sign(x) for x in denominator_arguments]

    numerator_has_pole = any(item is None for item in numerator_data)
    denominator_has_pole = any(item is None for item in denominator_data)

    if numerator_has_pole and denominator_has_pole:
        # A special analytic limit would be needed for a pole/pole cancellation.
        return math.nan
    if numerator_has_pole:
        return math.nan
    if denominator_has_pole:
        return 0.0

    log_abs = 0.0
    sign = 1
    for item in numerator_data:
        assert item is not None
        log_abs += item[0]
        sign *= item[1]
    for item in denominator_data:
        assert item is not None
        log_abs -= item[0]
        sign *= item[1]

    if log_abs > 709.0:
        return math.copysign(math.inf, sign)
    if log_abs < -744.0:
        return math.copysign(0.0, sign)
    return sign * math.exp(log_abs)


def quantization_lhs(omega: float, p: Parameters) -> float:
    """Left side common to the general condition."""

    return p.kappa * (omega * omega - p.boundary_offset)


def massless_rhs(omega: float) -> float:
    """-omega*cot(pi*omega/2), with the removable value at omega=0."""

    if abs(omega) < 1.0e-12:
        return -2.0 / math.pi
    angle = math.pi * omega / 2.0
    sine = math.sin(angle)
    if abs(sine) < 2.0e-12:
        return math.nan
    return -omega * math.cos(angle) / sine


def massless_residual(omega: float, p: Parameters) -> float:
    """LHS - RHS; roots solve the massless quantization condition."""

    rhs = massless_rhs(omega)
    if not is_finite(rhs):
        return math.nan
    return quantization_lhs(omega, p) - rhs


def massive_rhs(omega: float, p: Parameters) -> float:
    """Gamma quotient on the right side of the general massive condition."""

    nu = p.nu
    lam = p.lam
    return signed_gamma_ratio(
        (
            lam + nu / 2.0 - omega / 2.0,
            lam + nu / 2.0 + omega / 2.0,
            -nu,
        ),
        (
            lam - nu / 2.0 - omega / 2.0,
            lam - nu / 2.0 + omega / 2.0,
            nu,
        ),
    )


def massive_residual(omega: float, p: Parameters) -> float:
    """LHS - Gamma quotient; roots solve the massive condition."""

    rhs = massive_rhs(omega, p)
    if not is_finite(rhs):
        return math.nan
    return quantization_lhs(omega, p) - rhs


def validate_massive_parameters(p: Parameters) -> None:
    """Catch the singular parameter choices of the written Gamma formula."""

    nu = p.nu
    if is_nonpositive_integer(-nu):
        raise ValueError(
            f"nu={nu:g} makes Gamma(-nu) singular.  The displayed massive "
            "formula needs a separate limiting/logarithmic treatment at integer "
            "nu; choose a nearby noninteger nu for this script."
        )


def massless_poles(omega_min: float, omega_max: float) -> list[float]:
    """Nonzero even-integer poles of -omega*cot(pi*omega/2)."""

    largest_n = int(math.ceil(max(abs(omega_min), abs(omega_max)) / 2.0)) + 1
    candidates: list[float] = []
    for n in range(1, largest_n + 1):
        candidates.extend((-2.0 * n, 2.0 * n))
    return sorted(x for x in candidates if omega_min < x < omega_max)


def massive_poles(
    p: Parameters, omega_min: float, omega_max: float
) -> list[float]:
    """Real frequency poles from the numerator Gamma pair.

    For positive frequency, lambda+nu/2-omega/2 = -n gives
        omega = 2*lambda + nu + 2*n.
    Evenness supplies the corresponding negative poles.
    """

    validate_massive_parameters(p)
    first_positive = 2.0 * p.lam + p.nu
    if first_positive <= 0.0:
        # This is not expected for d>=2 and ell>=0, but a generic scan is safer.
        first_positive = abs(first_positive)
    largest = max(abs(omega_min), abs(omega_max))
    count = max(0, int(math.ceil((largest - first_positive) / 2.0)) + 2)
    candidates: list[float] = []
    for n in range(count):
        pole = first_positive + 2.0 * n
        candidates.extend((-pole, pole))
    return sorted(x for x in candidates if omega_min < x < omega_max)


def massive_zeros(
    p: Parameters, omega_min: float, omega_max: float
) -> list[float]:
    """Usual real zeros caused by poles of a denominator Gamma factor."""

    first_positive = 2.0 * p.lam - p.nu
    candidates: list[float] = []
    n = 0
    largest = max(abs(omega_min), abs(omega_max))
    while n < 200:
        zero = first_positive + 2.0 * n
        if zero > largest + 2.0:
            break
        if zero > 0.0:
            candidates.extend((-zero, zero))
        n += 1
    return sorted(x for x in candidates if omega_min < x < omega_max)


def continuous_intervals(
    omega_min: float, omega_max: float, poles: Sequence[float]
) -> list[tuple[float, float]]:
    """Split a plotting/root-search domain at poles."""

    interior_poles = sorted(p for p in poles if omega_min < p < omega_max)
    boundaries = [omega_min, *interior_poles, omega_max]
    intervals: list[tuple[float, float]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        width = right - left
        if width <= 0.0:
            continue
        margin = max(2.0e-7, width * 2.0e-7)
        safe_left = left + (margin if left in interior_poles else 0.0)
        safe_right = right - (margin if right in interior_poles else 0.0)
        if safe_right > safe_left:
            intervals.append((safe_left, safe_right))
    return intervals


def sampled_curve(
    function: Callable[[float], float],
    omega_min: float,
    omega_max: float,
    poles: Sequence[float],
    points_per_unit: int = POINTS_PER_UNIT,
    discard_above: float | None = None,
) -> tuple[list[float], list[float]]:
    """Sample a curve in separate pole-free pieces, inserting NaN breaks."""

    x_values: list[float] = []
    y_values: list[float] = []
    for left, right in continuous_intervals(omega_min, omega_max, poles):
        count = max(45, int((right - left) * points_per_unit) + 1)
        for omega in linspace(left, right, count):
            try:
                value = function(omega)
            except (OverflowError, ValueError, ZeroDivisionError):
                value = math.nan
            if not is_finite(value):
                value = math.nan
            elif discard_above is not None and abs(value) > discard_above:
                value = math.nan
            x_values.append(omega)
            y_values.append(value)
        x_values.append(math.nan)
        y_values.append(math.nan)
    return x_values, y_values


def bisect_root(
    function: Callable[[float], float],
    left: float,
    right: float,
    f_left: float,
    f_right: float,
    x_tolerance: float = 2.0e-11,
    max_iterations: int = 100,
) -> float | None:
    """Bisection on a known finite sign-changing bracket."""

    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right > 0.0:
        return None

    a, b = left, right
    fa, fb = f_left, f_right
    for _ in range(max_iterations):
        middle = 0.5 * (a + b)
        try:
            fm = function(middle)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        if not is_finite(fm):
            return None
        if abs(fm) < 2.0e-11 or abs(b - a) < x_tolerance:
            return middle
        if fa * fm <= 0.0:
            b, fb = middle, fm
        else:
            a, fa = middle, fm
    return 0.5 * (a + b)


def deduplicate(values: Iterable[float], tolerance: float = 2.0e-7) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def find_roots(
    function: Callable[[float], float],
    omega_min: float,
    omega_max: float,
    poles: Sequence[float],
    samples_per_unit: int = ROOT_SAMPLES_PER_UNIT,
) -> list[float]:
    """Find roots separately on every continuous interval.

    Dense bracketing plus bisection is deliberately simple.  It is sufficient for
    these one-dimensional transcendental conditions and avoids a SciPy dependency.
    """

    candidates: list[float] = []
    for left, right in continuous_intervals(omega_min, omega_max, poles):
        count = max(80, int((right - left) * samples_per_unit) + 1)
        previous_x: float | None = None
        previous_y: float | None = None

        for omega in linspace(left, right, count):
            try:
                value = function(omega)
            except (OverflowError, ValueError, ZeroDivisionError):
                value = math.nan

            if not is_finite(value):
                previous_x = None
                previous_y = None
                continue

            if abs(value) < 1.0e-9:
                candidates.append(omega)

            if (
                previous_x is not None
                and previous_y is not None
                and previous_y * value < 0.0
            ):
                root = bisect_root(
                    function, previous_x, omega, previous_y, value
                )
                if root is not None:
                    candidates.append(root)

            previous_x = omega
            previous_y = value

    return deduplicate(candidates)


def log_gamma_complex(z: complex) -> complex:
    """log Gamma(z) for Re z >= 0.5 (Lanczos)."""
    z -= 1.0
    series = LANCZOS_COEFFICIENTS[0]
    for index in range(1, len(LANCZOS_COEFFICIENTS)):
        series += LANCZOS_COEFFICIENTS[index] / (z + index)
    t = z + LANCZOS_G + 0.5
    return (
        0.5 * math.log(2.0 * math.pi) + (z + 0.5) * cmath.log(t) - t + cmath.log(series)
    )


def rgamma_complex(z: complex) -> complex:
    """1/Gamma(z) for complex z; exactly zero at the poles of Gamma."""
    product = 1.0 + 0.0j
    while z.real < 0.5:  # 1/Gamma(z) = z(z+1)...(z+k-1)/Gamma(z+k)
        product *= z
        z += 1.0
    return product * cmath.exp(-log_gamma_complex(z))


def residual_terms(omega: complex, p: Parameters) -> tuple[complex, complex]:
    """kappa*(omega^2 - Lambda_b)*B and A, both divided by Gamma(c)."""
    s = p.lam + p.nu / 2.0
    t = p.lam - p.nu / 2.0
    b_part = (
        math.gamma(p.nu)
        * rgamma_complex(s - omega / 2.0)
        * rgamma_complex(s + omega / 2.0)
    )
    a_part = (
        math.gamma(-p.nu)
        * rgamma_complex(t - omega / 2.0)
        * rgamma_complex(t + omega / 2.0)
    )
    return p.kappa * (omega * omega - p.boundary_offset) * b_part, a_part


def entire_residual(omega: complex, p: Parameters) -> complex:
    """kappa*(omega^2 - Lambda_b)*B - A, divided by Gamma(c); entire in omega."""
    lhs, a_part = residual_terms(omega, p)
    return lhs - a_part


def secant_complex(
    function: Callable[[complex], complex],
    z0: complex,
    z1: complex,
    tolerance: float = 1.0e-12,
    max_iterations: int = 60,
) -> complex | None:
    """Secant iteration in the complex plane; None if it fails to converge."""
    try:
        f0, f1 = function(z0), function(z1)
        for _ in range(max_iterations):
            if f1 == f0:
                return None
            z0, z1 = z1, z1 - f1 * (z1 - z0) / (f1 - f0)
            if abs(z1 - z0) < tolerance * max(1.0, abs(z1)):
                return z1
            if abs(z1) > 1.0e3:
                return None
            f0, f1 = f1, function(z1)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return None


def growing_modes(p: Parameters) -> list[complex]:
    """Roots with Im(omega) > 0 in the first quadrant, most unstable first."""
    validate_massive_parameters(p)
    roots: list[complex] = []
    for x in linspace(0.0, GROWTH_RE_MAX, 6):
        for y in (0.2, 0.7, 1.5, 3.0, 6.0, 12.0, GROWTH_IM_MAX):
            seed = complex(x, y)
            root = secant_complex(lambda w: entire_residual(w, p), seed, seed + 0.05j)
            if root is None or not 1.0e-7 < root.imag <= GROWTH_IM_MAX:
                continue
            lhs, a_part = residual_terms(root, p)
            if abs(lhs - a_part) > 1.0e-8 * (abs(lhs) + abs(a_part)):
                continue  # secant stall, not a root
            real = abs(root.real) if abs(root.real) > 1.0e-9 else 0.0
            root = complex(real, root.imag)
            if all(abs(root - known) > 1.0e-6 for known in roots):
                roots.append(root)
    return sorted(roots, key=lambda z: -z.imag)


def critical_kappa(p: Parameters) -> float:
    """kappa_c = |A/B|(omega=0) / Lambda_b; inf if Lambda_b <= 0."""
    s = p.lam + p.nu / 2.0
    t = p.lam - p.nu / 2.0
    ratio = math.gamma(s) / math.gamma(t)
    r0 = abs(math.gamma(-p.nu) / math.gamma(p.nu)) * ratio**2
    if p.boundary_offset <= 0.0:
        return math.inf
    return r0 / p.boundary_offset


def imaginary_mode_guaranteed(p: Parameters) -> bool:
    """0 < kappa < kappa_c at omega=0 forces an imaginary root."""
    return p.kappa > 0.0 and p.kappa < critical_kappa(p)


def growth_rows(
    p: Parameters, kappas: Sequence[float]
) -> list[tuple[float, complex | None, bool]]:
    """(kappa, most unstable mode or None, growing mode present) per kappa."""
    rows: list[tuple[float, complex | None, bool]] = []
    for kappa in kappas:
        varied = replace(p, kappa=kappa)
        modes = growing_modes(varied)
        mode = modes[0] if modes else None
        unstable = mode is not None or imaginary_mode_guaranteed(varied)
        rows.append((kappa, mode, unstable))
    return rows


# =============================================================================
# Plotting helpers
# =============================================================================


def parameter_text(p: Parameters, include_mass: bool) -> str:
    base = (
        rf"$d={p.d}$, $\ell={p.ell}$, $\kappa={p.kappa:g}$, "
        rf"$m_b^2={p.mb2:g}$, $\xi_b={p.xi_b:g}$"
    )
    if include_mass:
        base += rf", $M^2={p.bulk_m2:g}$, $\nu={p.nu:.3g}$"
    return base


def style_axes(
    ax: Axes,
    x_label: str = r"frequency $\omega$",
    y_label: str | None = None,
    zero_line: bool = True,
) -> None:
    if zero_line:
        ax.axhline(0.0, color=GRAY, linewidth=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.75)
    ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)


def draw_poles(ax: Axes, poles: Sequence[float], label_first: bool = False) -> None:
    for index, pole in enumerate(poles):
        label = "poles" if label_first and index == 0 else None
        ax.axvline(
            pole,
            color=LIGHT_GRAY,
            linewidth=0.75,
            linestyle=(0, (2, 3)),
            zorder=0,
            label=label,
        )


def draw_root_markers(ax: Axes, roots: Sequence[float], p: Parameters) -> None:
    if not roots:
        return
    y_values = [quantization_lhs(root, p) for root in roots]
    ax.scatter(
        roots,
        y_values,
        s=31,
        facecolor="white",
        edgecolor=BRICK,
        linewidth=1.25,
        zorder=5,
        label="quantized frequencies",
    )


def draw_zero_root_markers(ax: Axes, roots: Sequence[float]) -> None:
    if not roots:
        return
    ax.scatter(
        roots,
        [0.0] * len(roots),
        s=28,
        facecolor="white",
        edgecolor=BRICK,
        linewidth=1.2,
        zorder=5,
        label="roots",
    )


def legend_above(ax: Axes, title: str | None = None, ncol: int = 3) -> None:
    """Legend outside the top of the axes, with parameters as its title."""
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=ncol,
        frameon=False,
        title=title,
        alignment="left",
    )


def save_figure(
    fig: Figure,
    stem: str,
    settings: RunSettings,
    reporter: Reporter,
) -> None:
    fig.tight_layout()
    for file_format in settings.formats:
        path = settings.output_dir / f"{stem}.{file_format}"
        temporary_path = settings.output_dir / f".{stem}.{file_format}.tmp"
        save_kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "format": file_format,
        }
        if file_format.lower() == "png":
            save_kwargs["dpi"] = settings.dpi
        fig.savefig(temporary_path, **save_kwargs)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Matplotlib did not finish writing {path.name}.")
        temporary_path.replace(path)
        reporter.record(path, "figure")
    plt.close(fig)


def plot_massless_intersections(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> list[float]:
    poles = massless_poles(settings.omega_min, settings.omega_max)
    roots = find_roots(
        lambda w: massless_residual(w, p),
        settings.omega_min,
        settings.omega_max,
        poles,
    )
    x_rhs, y_rhs = sampled_curve(
        massless_rhs,
        settings.omega_min,
        settings.omega_max,
        poles,
        discard_above=12.0 * settings.y_limit,
    )
    x_lhs = linspace(settings.omega_min, settings.omega_max, 900)
    y_lhs = [quantization_lhs(w, p) for w in x_lhs]

    fig, ax = plt.subplots()
    ax.plot(x_lhs, y_lhs, color=BLUE, label=r"$\kappa(\omega^2-\Lambda_b)$")
    ax.plot(x_rhs, y_rhs, color=BLACK, label=r"$-\omega\cot(\pi\omega/2)$")
    draw_poles(ax, poles)
    draw_root_markers(ax, roots, p)
    ax.set_xlim(settings.omega_min, settings.omega_max)
    ax.set_ylim(-settings.y_limit, settings.y_limit)
    style_axes(ax, y_label="both sides of the condition")
    legend_above(ax, parameter_text(p, include_mass=False))
    save_figure(fig, "01_massless_intersections", settings, reporter)
    return roots


def plot_massless_residual(
    p: Parameters,
    roots: Sequence[float],
    settings: RunSettings,
    reporter: Reporter,
) -> None:
    poles = massless_poles(settings.omega_min, settings.omega_max)
    x_values, y_values = sampled_curve(
        lambda w: massless_residual(w, p),
        settings.omega_min,
        settings.omega_max,
        poles,
        discard_above=12.0 * settings.y_limit,
    )
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, color=BLACK, label=r"$Q_0(\omega)$")
    draw_poles(ax, poles)
    draw_zero_root_markers(ax, roots)
    ax.set_xlim(settings.omega_min, settings.omega_max)
    ax.set_ylim(-settings.y_limit, settings.y_limit)
    style_axes(ax, y_label=r"$Q_0(\omega)=\mathrm{LHS}-\mathrm{RHS}$")
    legend_above(ax, parameter_text(p, include_mass=False))
    save_figure(fig, "02_massless_residual", settings, reporter)


def plot_massless_symmetric(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> None:
    extent = min(8.0, max(abs(settings.omega_min), abs(settings.omega_max)))
    omega_min, omega_max = -extent, extent
    poles = massless_poles(omega_min, omega_max)
    x_rhs, y_rhs = sampled_curve(
        massless_rhs,
        omega_min,
        omega_max,
        poles,
        discard_above=80.0,
    )
    x_lhs = linspace(omega_min, omega_max, 900)
    y_lhs = [quantization_lhs(w, p) for w in x_lhs]

    fig, ax = plt.subplots()
    ax.plot(x_rhs, y_rhs, color=BLACK, label=r"$-\omega\cot(\pi\omega/2)$")
    ax.plot(x_lhs, y_lhs, color=BLUE, linestyle="--", label="boundary side")
    draw_poles(ax, poles)
    ax.set_xlim(omega_min, omega_max)
    ax.set_ylim(-8.0, 8.0)
    style_axes(ax, y_label="quantization functions")
    legend_above(ax, ncol=2)
    save_figure(fig, "03_massless_symmetric_overview", settings, reporter)


def plot_massless_kappa_curves(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> None:
    poles = massless_poles(settings.omega_min, settings.omega_max)
    fig, ax = plt.subplots()
    for index, kappa in enumerate(KAPPA_CURVES):
        varied = replace(p, kappa=kappa)
        x_values, y_values = sampled_curve(
            lambda w, q=varied: massless_residual(w, q),
            settings.omega_min,
            settings.omega_max,
            poles,
            points_per_unit=120,
            discard_above=10.0 * settings.y_limit,
        )
        ax.plot(
            x_values,
            y_values,
            color=PALETTE[index % len(PALETTE)],
            linestyle=LINE_STYLES[index % len(LINE_STYLES)],
            linewidth=1.2,
            label=rf"$\kappa={kappa:g}$",
        )
    draw_poles(ax, poles)
    ax.set_xlim(settings.omega_min, settings.omega_max)
    ax.set_ylim(-settings.y_limit, settings.y_limit)
    style_axes(ax, y_label=r"$Q_0(\omega)$")
    legend_above(ax, rf"$d=3$, $\ell=0$, $m_b^2={p.mb2:g}$, $\xi_b={p.xi_b:g}$", ncol=3)
    save_figure(fig, "04_massless_kappa_curves", settings, reporter)


def kappa_grid() -> list[float]:
    return linspace(KAPPA_SPECTRUM_MIN, KAPPA_SPECTRUM_MAX, KAPPA_SPECTRUM_COUNT)


def root_spectrum_by_parameter(
    base: Parameters,
    parameter_values: Sequence[float],
    setter: Callable[[Parameters, float], Parameters],
    residual_builder: Callable[[Parameters], Callable[[float], float]],
    pole_builder: Callable[[Parameters, float, float], Sequence[float]],
    omega_min: float,
    omega_max: float,
) -> list[tuple[float, int, float]]:
    rows: list[tuple[float, int, float]] = []
    for parameter_value in parameter_values:
        varied = setter(base, parameter_value)
        poles = pole_builder(varied, omega_min, omega_max)
        roots = find_roots(
            residual_builder(varied),
            omega_min,
            omega_max,
            poles,
            samples_per_unit=170,
        )
        for mode_index, root in enumerate(roots):
            rows.append((parameter_value, mode_index, root))
    return rows


def plot_spectrum_rows(
    ax: Axes,
    rows: Sequence[tuple[float, int, float]],
    x_label: str,
    growth: Sequence[tuple[float, complex | None, bool]],
    max_modes: int = 7,
) -> None:
    ax.fill_between(
        [row[0] for row in growth],
        0.0,
        1.0,
        where=[row[2] for row in growth],
        step="mid",
        transform=ax.get_xaxis_transform(),
        color=BRICK,
        alpha=0.13,
        linewidth=0.0,
        label="growing mode present",
    )
    selected = [row for row in rows if row[1] < max_modes]
    ax.plot(
        [row[0] for row in selected],
        [row[2] for row in selected],
        color=BLACK,
        linestyle="none",
        marker="o",
        markersize=2.4,
        label=r"real roots $\omega_n$",
    )
    style_axes(ax, x_label=x_label, y_label=r"positive root $\omega_n$")


def draw_critical_kappa(ax: Axes, p: Parameters, label_y: float = 2.0) -> None:
    """Dashed line at kappa_c with a small label just inside the shaded zone."""
    kappa_c = critical_kappa(p)
    if not math.isfinite(kappa_c) or not KAPPA_SPECTRUM_MIN < kappa_c < KAPPA_SPECTRUM_MAX:
        return
    ax.axvline(
        kappa_c,
        color=BRICK,
        linestyle="--",
        linewidth=1.1,
        zorder=4,
        label=rf"$\kappa_c={kappa_c:.3f}$",
    )
    ax.text(
        kappa_c - 0.03,
        label_y,
        r"$\kappa_c$",
        color=BRICK,
        ha="right",
        va="center",
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def plot_massless_kappa_spectrum(
    p: Parameters,
    growth: Sequence[tuple[float, complex | None, bool]],
    settings: RunSettings,
    reporter: Reporter,
) -> list[tuple[float, int, float]]:
    kappas = kappa_grid()
    rows = root_spectrum_by_parameter(
        p,
        kappas,
        lambda q, value: replace(q, kappa=value),
        lambda q: lambda w: massless_residual(w, q),
        lambda _q, low, high: massless_poles(low, high),
        settings.omega_min,
        settings.omega_max,
    )
    fig, ax = plt.subplots()
    plot_spectrum_rows(ax, rows, r"Wentzell coupling $\kappa$", growth)
    draw_critical_kappa(ax, p)
    legend_above(
        ax, rf"$m_b^2={p.mb2:g}$, $\xi_b={p.xi_b:g}$; markers are sorted roots", ncol=3
    )
    save_figure(fig, "05_massless_spectrum_vs_kappa", settings, reporter)
    return rows


def plot_massless_boundary_sweeps(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> None:
    poles = massless_poles(settings.omega_min, settings.omega_max)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

    for index, mb2 in enumerate(BOUNDARY_MASS_CURVES):
        varied = replace(p, mb2=mb2)
        roots = find_roots(
            lambda w, q=varied: massless_residual(w, q),
            settings.omega_min,
            settings.omega_max,
            poles,
            samples_per_unit=200,
        )
        axes[0].plot(
            range(1, len(roots) + 1),
            roots,
            marker="o",
            markersize=3.5,
            color=PALETTE[index],
            linestyle=LINE_STYLES[index],
            label=rf"$m_b^2={mb2:g}$",
        )

    for index, xi_b in enumerate(XI_B_CURVES):
        varied = replace(p, xi_b=xi_b)
        roots = find_roots(
            lambda w, q=varied: massless_residual(w, q),
            settings.omega_min,
            settings.omega_max,
            poles,
            samples_per_unit=200,
        )
        axes[1].plot(
            range(1, len(roots) + 1),
            roots,
            marker="o",
            markersize=3.5,
            color=PALETTE[index],
            linestyle=LINE_STYLES[index],
            label=rf"$\xi_b={xi_b:g}$",
        )

    for ax in axes:
        style_axes(ax, x_label="root index", y_label=r"positive root $\omega_n$")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    save_figure(fig, "06_massless_boundary_parameter_sweeps", settings, reporter)


def plot_massive_intersections(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> list[float]:
    validate_massive_parameters(p)
    poles = massive_poles(p, settings.omega_min, settings.omega_max)
    roots = find_roots(
        lambda w: massive_residual(w, p),
        settings.omega_min,
        settings.omega_max,
        poles,
    )
    x_rhs, y_rhs = sampled_curve(
        lambda w: massive_rhs(w, p),
        settings.omega_min,
        settings.omega_max,
        poles,
        discard_above=12.0 * settings.y_limit,
    )
    x_lhs = linspace(settings.omega_min, settings.omega_max, 900)
    y_lhs = [quantization_lhs(w, p) for w in x_lhs]

    fig, ax = plt.subplots()
    ax.plot(x_lhs, y_lhs, color=BLUE, label=r"$\kappa(\omega^2-\Lambda_b)$")
    ax.plot(x_rhs, y_rhs, color=BLACK, label=r"$A(\omega)/B(\omega)$")
    draw_poles(ax, poles)
    draw_root_markers(ax, roots, p)
    ax.set_xlim(settings.omega_min, settings.omega_max)
    ax.set_ylim(-settings.y_limit, settings.y_limit)
    style_axes(ax, y_label="both sides of the condition")
    legend_above(ax, parameter_text(p, include_mass=True))
    save_figure(fig, "07_massive_intersections", settings, reporter)
    return roots


def plot_massive_residual(
    p: Parameters,
    roots: Sequence[float],
    settings: RunSettings,
    reporter: Reporter,
) -> None:
    poles = massive_poles(p, settings.omega_min, settings.omega_max)
    x_values, y_values = sampled_curve(
        lambda w: massive_residual(w, p),
        settings.omega_min,
        settings.omega_max,
        poles,
        discard_above=12.0 * settings.y_limit,
    )
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, color=BLACK, label=r"$Q_M(\omega)$")
    draw_poles(ax, poles)
    draw_zero_root_markers(ax, roots)
    ax.set_xlim(settings.omega_min, settings.omega_max)
    ax.set_ylim(-settings.y_limit, settings.y_limit)
    style_axes(ax, y_label=r"$Q_M(\omega)=\mathrm{LHS}-\mathrm{RHS}$")
    legend_above(ax, parameter_text(p, include_mass=True))
    save_figure(fig, "08_massive_residual", settings, reporter)


def plot_massive_literature_style(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> None:
    extent = min(8.0, max(abs(settings.omega_min), abs(settings.omega_max)))
    omega_min, omega_max = -extent, extent
    fig, ax = plt.subplots()
    for index, ell in enumerate((0, 1)):
        varied = replace(p, ell=ell)
        poles = massive_poles(varied, omega_min, omega_max)
        x_values, y_values = sampled_curve(
            lambda w, q=varied: massive_rhs(w, q),
            omega_min,
            omega_max,
            poles,
            discard_above=100.0,
        )
        ax.plot(
            x_values,
            y_values,
            color=(BLACK, GRAY)[index],
            linestyle=("-", "--")[index],
            label=rf"$\ell={ell}$",
        )
    ax.set_xlim(omega_min, omega_max)
    ax.set_ylim(-10.0, 10.0)
    style_axes(ax, y_label="Gamma quotient")
    legend_above(ax, rf"$d={p.d}$, $\nu={p.nu:.3g}$", ncol=2)
    save_figure(fig, "09_massive_gamma_quotient_literature_style", settings, reporter)


def plot_massive_mass_sweep(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> list[tuple[float, int, float]]:
    fig, ax = plt.subplots()
    csv_rows: list[tuple[float, int, float]] = []
    for index, bulk_m2 in enumerate(BULK_MASS_SQUARED_CURVES):
        varied = replace(p, bulk_m2=bulk_m2)
        validate_massive_parameters(varied)
        poles = massive_poles(varied, settings.omega_min, settings.omega_max)
        roots = find_roots(
            lambda w, q=varied: massive_residual(w, q),
            settings.omega_min,
            settings.omega_max,
            poles,
            samples_per_unit=210,
        )
        for mode_index, root in enumerate(roots):
            csv_rows.append((bulk_m2, mode_index, root))
        ax.plot(
            range(1, len(roots) + 1),
            roots,
            marker="o",
            markersize=3.5,
            color=PALETTE[index],
            linestyle=LINE_STYLES[index],
            label=rf"$M^2={bulk_m2:g}$, $\nu={varied.nu:.3g}$",
        )
    style_axes(ax, x_label="root index", y_label=r"positive root $\omega_n$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left")
    save_figure(fig, "10_massive_bulk_mass_sweep", settings, reporter)
    return csv_rows


def plot_massive_ell_sweep(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> None:
    fig, ax = plt.subplots()
    for index, ell in enumerate(ELL_CURVES):
        varied = replace(p, ell=ell)
        poles = massive_poles(varied, settings.omega_min, settings.omega_max)
        roots = find_roots(
            lambda w, q=varied: massive_residual(w, q),
            settings.omega_min,
            settings.omega_max,
            poles,
            samples_per_unit=210,
        )
        ax.plot(
            range(1, len(roots) + 1),
            roots,
            marker="o",
            markersize=3.5,
            color=PALETTE[index],
            linestyle=LINE_STYLES[index],
            label=rf"$\ell={ell}$",
        )
    style_axes(ax, x_label="root index", y_label=r"positive root $\omega_n$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    legend_above(ax, rf"$d={p.d}$, $M^2={p.bulk_m2:g}$, $\nu={p.nu:.3g}$", ncol=4)
    save_figure(fig, "11_massive_ell_sweep", settings, reporter)


def plot_massive_kappa_spectrum(
    p: Parameters,
    growth: Sequence[tuple[float, complex | None, bool]],
    settings: RunSettings,
    reporter: Reporter,
) -> list[tuple[float, int, float]]:
    kappas = kappa_grid()
    rows = root_spectrum_by_parameter(
        p,
        kappas,
        lambda q, value: replace(q, kappa=value),
        lambda q: lambda w: massive_residual(w, q),
        lambda q, low, high: massive_poles(q, low, high),
        settings.omega_min,
        settings.omega_max,
    )
    fig, ax = plt.subplots()
    plot_spectrum_rows(ax, rows, r"Wentzell coupling $\kappa$", growth)
    legend_above(
        ax, rf"$M^2={p.bulk_m2:g}$, $\nu={p.nu:.3g}$; markers are sorted roots", ncol=2
    )
    save_figure(fig, "12_massive_spectrum_vs_kappa", settings, reporter)
    return rows


def plot_massless_limit_check(
    p: Parameters, settings: RunSettings, reporter: Reporter
) -> None:
    """Show that the massive Gamma formula at nu=1/2 equals the cotangent form."""

    check_parameters = replace(p, d=3, ell=0, bulk_m2=0.0)
    poles = massless_poles(settings.omega_min, settings.omega_max)
    x_values, cot_values = sampled_curve(
        massless_rhs,
        settings.omega_min,
        settings.omega_max,
        poles,
        points_per_unit=130,
        discard_above=200.0,
    )
    gamma_values: list[float] = []
    errors: list[float] = []
    for omega, cot_value in zip(x_values, cot_values):
        if not is_finite(omega) or not is_finite(cot_value):
            gamma_values.append(math.nan)
            errors.append(math.nan)
            continue
        gamma_value = massive_rhs(omega, check_parameters)
        gamma_values.append(gamma_value)
        if is_finite(gamma_value):
            errors.append(max(abs(gamma_value - cot_value), 1.0e-16))
        else:
            errors.append(math.nan)

    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.2), sharex=True)
    axes[0].plot(x_values, cot_values, color=BLACK, label="cotangent form")
    axes[0].plot(
        x_values,
        gamma_values,
        color=BRICK,
        linestyle="--",
        linewidth=1.0,
        label=r"Gamma form at $\nu=1/2$",
    )
    axes[0].set_ylim(-settings.y_limit, settings.y_limit)
    style_axes(axes[0], x_label="", y_label="right-hand side")
    axes[0].legend(loc="lower right")

    axes[1].semilogy(x_values, errors, color=BLUE)
    axes[1].set_ylim(1.0e-16, 1.0e-10)
    style_axes(
        axes[1],
        y_label="absolute numerical difference",
        zero_line=False,
    )
    for ax in axes:
        ax.set_xlim(settings.omega_min, settings.omega_max)
        draw_poles(ax, poles)
    save_figure(fig, "13_massless_limit_validation", settings, reporter)


def plot_low_mode_comparison(
    massless_roots: Sequence[float],
    massive_roots: Sequence[float],
    p_massive: Parameters,
    settings: RunSettings,
    reporter: Reporter,
) -> None:
    count = min(len(massless_roots), len(massive_roots), 8)
    indices = list(range(1, count + 1))
    fig, ax = plt.subplots()
    ax.plot(
        indices,
        list(massless_roots[:count]),
        color=BLACK,
        marker="o",
        markersize=4.0,
        label=r"massless bulk ($M^2=0$)",
    )
    ax.plot(
        indices,
        list(massive_roots[:count]),
        color=BRICK,
        linestyle="--",
        marker="s",
        markersize=3.8,
        label=rf"massive bulk ($M^2={p_massive.bulk_m2:g}$)",
    )
    style_axes(ax, x_label="root index", y_label=r"positive root $\omega_n$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left")
    save_figure(fig, "14_massless_massive_low_mode_comparison", settings, reporter)


def plot_growth_rate_vs_kappa(
    massless_growth: Sequence[tuple[float, complex | None, bool]],
    massive_growth: Sequence[tuple[float, complex | None, bool]],
    p_massive: Parameters,
    settings: RunSettings,
    reporter: Reporter,
) -> None:
    fig, ax = plt.subplots()
    massive_label = rf"$M^2={p_massive.bulk_m2:g}$, $\nu={p_massive.nu:.3g}$"
    cases = (
        (massless_growth, BLACK, "-", r"$M^2=0$"),
        (massive_growth, BRICK, "--", massive_label),
    )
    for growth, color, linestyle, label in cases:
        rates = [
            mode.imag if mode is not None else (math.nan if unstable else 0.0)
            for _, mode, unstable in growth
        ]
        kappas = [row[0] for row in growth]
        ax.plot(kappas, rates, color=color, linestyle=linestyle, label=label)
    ax.set_xlim(KAPPA_SPECTRUM_MIN, KAPPA_SPECTRUM_MAX)
    ax.set_ylim(0.0, 6.0)
    style_axes(
        ax,
        x_label=r"Wentzell coupling $\kappa$",
        y_label=r"growth rate $\mathrm{Im}\;\omega$",
    )
    legend_above(
        ax,
        rf"$d={p_massive.d}$, $\ell={p_massive.ell}$, "
        rf"$m_b^2={p_massive.mb2:g}$, $\xi_b={p_massive.xi_b:g}$",
        ncol=2,
    )
    save_figure(fig, "15_growth_rate_vs_kappa", settings, reporter)


# =============================================================================
# Tables, manifest, and validation
# =============================================================================


def write_root_table(
    path: Path,
    case_name: str,
    roots: Sequence[float],
    p: Parameters,
    residual: Callable[[float], float],
    reporter: Reporter,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case",
                "mode_index",
                "omega",
                "residual",
                "d",
                "ell",
                "kappa",
                "mb2",
                "xi_b",
                "R_b",
                "bulk_M2",
                "nu",
                "lambda",
            )
        )
        for index, root in enumerate(roots, start=1):
            writer.writerow(
                (
                    case_name,
                    index,
                    f"{root:.14g}",
                    f"{residual(root):.6e}",
                    p.d,
                    p.ell,
                    f"{p.kappa:.14g}",
                    f"{p.mb2:.14g}",
                    f"{p.xi_b:.14g}",
                    f"{p.boundary_curvature:.14g}",
                    f"{p.bulk_m2:.14g}",
                    f"{p.nu:.14g}",
                    f"{p.lam:.14g}",
                )
            )
    reporter.record(path, "data")


def write_sweep_table(
    path: Path,
    rows_by_case: Sequence[tuple[str, Sequence[tuple[float, int, float]]]],
    reporter: Reporter,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "parameter_value", "mode_index", "omega"))
        for case_name, rows in rows_by_case:
            for parameter_value, mode_index, root in rows:
                writer.writerow(
                    (
                        case_name,
                        f"{parameter_value:.14g}",
                        mode_index + 1,
                        f"{root:.14g}",
                    )
                )
    reporter.record(path, "data")


def write_growth_table(
    path: Path,
    rows_by_case: Sequence[tuple[str, Sequence[tuple[float, complex | None, bool]]]],
    reporter: Reporter,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "kappa", "re_omega", "im_omega", "growing_mode"))
        for case_name, rows in rows_by_case:
            for kappa, mode, unstable in rows:
                writer.writerow(
                    (
                        case_name,
                        f"{kappa:.14g}",
                        "" if mode is None else f"{mode.real:.14g}",
                        "" if mode is None else f"{mode.imag:.14g}",
                        int(unstable),
                    )
                )
    reporter.record(path, "data")


def format_modes(modes: Sequence[complex]) -> str:
    return ", ".join(f"{z.real:.9g}{z.imag:+.9g}i" for z in modes) or "none"


def run_numerical_checks(
    massless_roots: Sequence[float],
    massive_roots: Sequence[float],
    massless_parameters: Parameters,
    massive_parameters: Parameters,
) -> None:
    """Fast checks for the identity, evenness, and reported roots."""

    check_parameters = replace(massive_parameters, d=3, ell=0, bulk_m2=0.0)
    safe_points = (0.0, 0.17, 0.73, 1.0, 1.37, 2.31, 5.13, 9.41)
    for omega in safe_points:
        cot_value = massless_rhs(omega)
        gamma_value = massive_rhs(omega, check_parameters)
        scale = max(1.0, abs(cot_value), abs(gamma_value))
        if abs(cot_value - gamma_value) > 2.0e-11 * scale:
            raise RuntimeError(
                "The nu=1/2 Gamma quotient failed to reproduce the cotangent form "
                f"at omega={omega:g}."
            )

    for omega in (0.23, 1.11, 3.37, 7.19):
        if abs(massless_rhs(omega) - massless_rhs(-omega)) > 1.0e-10:
            raise RuntimeError("Massless RHS evenness check failed.")
        positive = massive_rhs(omega, massive_parameters)
        negative = massive_rhs(-omega, massive_parameters)
        if abs(positive - negative) > 2.0e-10 * max(1.0, abs(positive)):
            raise RuntimeError("Massive Gamma quotient evenness check failed.")

    s = massive_parameters.lam + massive_parameters.nu / 2.0
    for omega in (0.37, 1.73, 3.1, 6.9):  # complex Lanczos path vs real log-Gamma path
        b_part = math.gamma(massive_parameters.nu) * rgamma_complex(s - omega / 2.0)
        b_part *= rgamma_complex(s + omega / 2.0)
        complex_value = entire_residual(omega, massive_parameters) / b_part
        real_value = massive_residual(omega, massive_parameters)
        if abs(complex_value - real_value) > 1.0e-9 * max(1.0, abs(real_value)):
            raise RuntimeError(
                f"Complex and real Gamma evaluations disagree at omega={omega:g}."
            )

    for root in massless_roots:
        residual = abs(massless_residual(root, massless_parameters))
        if residual > 5.0e-7:
            raise RuntimeError(f"Massless root residual is too large at {root:g}.")
    for root in massive_roots:
        residual = abs(massive_residual(root, massive_parameters))
        if residual > 5.0e-7:
            raise RuntimeError(f"Massive root residual is too large at {root:g}.")


def write_manifest(
    path: Path,
    reporter: Reporter,
    massless_parameters: Parameters,
    massive_parameters: Parameters,
    massless_roots: Sequence[float],
    massive_roots: Sequence[float],
    massless_growing: Sequence[complex],
    massive_growing: Sequence[complex],
) -> None:
    lines = [
        "Wentzell quantization graph batch",
        "===================================",
        "",
        "Massless special case:",
        f"  {parameter_text(massless_parameters, include_mass=False)}",
        "  positive roots: " + ", ".join(f"{root:.9g}" for root in massless_roots),
        "  growing modes (Im omega > 0): " + format_modes(massless_growing),
        "",
        "Massive case:",
        f"  {parameter_text(massive_parameters, include_mass=True)}",
        "  positive roots: " + ", ".join(f"{root:.9g}" for root in massive_roots),
        "  growing modes (Im omega > 0): " + format_modes(massive_growing),
        "",
        "Generated files:",
    ]
    # The manifest itself is recorded after it is written, so list prior outputs.
    lines.extend(f"  {item.name}" for item in reporter.generated)
    lines.extend(
        [
            f"  {path.name}",
            "",
            "Notes:",
            "  - Curves are broken at analytic poles.",
            "  - Root searches are performed independently on pole-free intervals.",
            "  - The Gamma quotient uses signed log-gamma arithmetic.",
            "  - Positive roots are tabulated; the equations are even in omega.",
            "  - Growing modes are complex roots from secant iteration on a seed grid.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reporter.record(path, "manifest")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate massless and massive Wentzell spectrum graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_OUTPUT_FOLDER,
        help="folder for figures and CSV files",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=list(DEFAULT_FORMATS),
        help="one or more figure formats",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG resolution")
    parser.add_argument(
        "--omega-min", type=float, default=OMEGA_MIN, help="minimum plotted frequency"
    )
    parser.add_argument(
        "--omega-max", type=float, default=OMEGA_MAX, help="maximum plotted frequency"
    )
    parser.add_argument(
        "--y-limit",
        type=float,
        default=PLOT_Y_LIMIT,
        help="symmetric y limit for intersection/residual plots",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.omega_max <= args.omega_min:
        raise ValueError("--omega-max must be greater than --omega-min.")
    if args.dpi <= 0 or args.y_limit <= 0.0:
        raise ValueError("--dpi and --y-limit must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = RunSettings(
        output_dir=output_dir,
        formats=tuple(dict.fromkeys(args.formats)),
        dpi=args.dpi,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        y_limit=args.y_limit,
    )
    reporter = Reporter(output_dir)
    configure_matplotlib()

    # The cotangent formula is the d=3, ell=0, M^2=0 specialization.
    base = Parameters()
    massless_parameters = replace(base, d=3, ell=0, bulk_m2=0.0)
    massive_parameters = base
    validate_massive_parameters(massive_parameters)

    print("Generating Wentzell quantization figures (no GUI windows will open).")
    print(f"Output folder: {output_dir}")
    print(f"Figure formats: {', '.join(settings.formats)}")
    print(
        "Massless parameters: "
        + parameter_text(massless_parameters, include_mass=False)
    )
    print(
        "Massive parameters:  "
        + parameter_text(massive_parameters, include_mass=True)
    )
    print("")

    growth_kappas = linspace(KAPPA_SPECTRUM_MIN, KAPPA_SPECTRUM_MAX, GROWTH_KAPPA_COUNT)
    massless_growth = growth_rows(massless_parameters, growth_kappas)
    massive_growth = growth_rows(massive_parameters, growth_kappas)
    massless_growing = growing_modes(massless_parameters)
    massive_growing = growing_modes(massive_parameters)

    massless_roots = plot_massless_intersections(
        massless_parameters, settings, reporter
    )
    plot_massless_residual(
        massless_parameters, massless_roots, settings, reporter
    )
    plot_massless_symmetric(massless_parameters, settings, reporter)
    plot_massless_kappa_curves(massless_parameters, settings, reporter)
    massless_kappa_rows = plot_massless_kappa_spectrum(
        massless_parameters, massless_growth, settings, reporter
    )
    plot_massless_boundary_sweeps(massless_parameters, settings, reporter)

    massive_roots = plot_massive_intersections(
        massive_parameters, settings, reporter
    )
    plot_massive_residual(massive_parameters, massive_roots, settings, reporter)
    plot_massive_literature_style(massive_parameters, settings, reporter)
    massive_mass_rows = plot_massive_mass_sweep(
        massive_parameters, settings, reporter
    )
    plot_massive_ell_sweep(massive_parameters, settings, reporter)
    massive_kappa_rows = plot_massive_kappa_spectrum(
        massive_parameters, massive_growth, settings, reporter
    )
    plot_massless_limit_check(massive_parameters, settings, reporter)
    plot_low_mode_comparison(
        massless_roots,
        massive_roots,
        massive_parameters,
        settings,
        reporter,
    )
    plot_growth_rate_vs_kappa(
        massless_growth, massive_growth, massive_parameters, settings, reporter
    )

    run_numerical_checks(
        massless_roots,
        massive_roots,
        massless_parameters,
        massive_parameters,
    )
    print("  [check   ] Numerical identity, symmetry, and root checks passed.")

    write_root_table(
        output_dir / "massless_roots.csv",
        "massless",
        massless_roots,
        massless_parameters,
        lambda w: massless_residual(w, massless_parameters),
        reporter,
    )
    write_root_table(
        output_dir / "massive_roots.csv",
        "massive",
        massive_roots,
        massive_parameters,
        lambda w: massive_residual(w, massive_parameters),
        reporter,
    )
    write_sweep_table(
        output_dir / "parameter_sweep_roots.csv",
        (
            ("massless_kappa", massless_kappa_rows),
            ("massive_kappa", massive_kappa_rows),
            ("massive_bulk_M2", massive_mass_rows),
        ),
        reporter,
    )
    write_growth_table(
        output_dir / "growing_modes_vs_kappa.csv",
        (("massless", massless_growth), ("massive", massive_growth)),
        reporter,
    )
    write_manifest(
        output_dir / "run_manifest.txt",
        reporter,
        massless_parameters,
        massive_parameters,
        massless_roots,
        massive_roots,
        massless_growing,
        massive_growing,
    )

    print("")
    print(
        f"Done. Generated {len(reporter.generated)} files in {settings.output_dir}"
    )
    print(
        "Massless positive roots: "
        + ", ".join(f"{root:.8g}" for root in massless_roots)
    )
    print(
        "Massive positive roots:  "
        + ", ".join(f"{root:.8g}" for root in massive_roots)
    )
    print("Massless growing modes:  " + format_modes(massless_growing))
    print("Massive growing modes:   " + format_modes(massive_growing))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
