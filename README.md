# Wentzell spectra in global AdS

Companion code for the preprint *Klein-Gordon Fields with Dynamical Wentzell Boundary Conditions in Global Anti-de Sitter Spacetime*.

`wentzell_spectrum_plots.py` solves the quantization condition for a Klein-Gordon field in global AdS with dynamical Wentzell boundary conditions. It finds the real mode frequencies and any growing modes, then saves the figures used in the preprint.

## Requirements

Python 3 and Matplotlib 3.6 or newer.

```
pip install matplotlib
```

## Usage

```
python wentzell_spectrum_plots.py
```

The script asks for each parameter. Press Enter at every prompt to use the preprint values. Roots and growing modes are printed to the terminal, and figures are saved to `generated_graphs/`.

## Figures

| File | Contents |
|---|---|
| `01_massless_intersections` | Both sides of the M² = 0 condition, roots marked |
| `02_massless_spectrum_vs_kappa` | M² = 0 roots vs κ, growing-mode region shaded |
| `03_massless_boundary_sweeps` | M² = 0 roots vs m_b² and ξ_b |
| `04_massive_intersections` | Both sides of the massive condition, roots marked |
| `05_massive_spectrum_vs_kappa` | Massive roots vs κ, growing-mode region shaded |
| `06_massive_bulk_mass_sweep` | Roots vs M² |
| `07_massive_ell_sweep` | Roots vs ℓ |
| `08_growth_rate_vs_kappa` | Im ω of the most unstable mode vs κ |
| `09_growing_modes_complex_plane` | Growing modes in the complex ω plane, colored by κ |
