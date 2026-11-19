# Wentzell spectra in global AdS

Companion code for the preprint *Klein-Gordon Fields with Dynamical Wentzell Boundary Conditions in Global Anti-de Sitter Spacetime*.

`spectrum_analysis.py` computes the mode spectrum and stability of a Klein-Gordon field in global AdS with dynamical Wentzell boundary conditions, and generates the figures in the preprint.

## Requirements

Python 3, NumPy, SciPy, and Matplotlib

```
pip install numpy scipy matplotlib
```

## Usage

Parameters can be set in `PAR` at the top of the file.

```
python spectrum_analysis_v2.py
```
 
Results are printed to the terminal, figures are saved to `generated_graphs/`, and tables are saved to `generated_data/`.

## Figures

| File | Contents |
|---|---|
| `01_massless_intersections` | Both sides of the M^2 = 0 condition, roots marked |
| `02_spectrum_vs_kappa` | Roots vs kappa for M^2 = 0 and 0.25, unstable region shaded |
| `03_massive_intersections` | Both sides of the massive condition, roots marked |
| `04_massless_boundary_sweeps` | M^2 = 0 roots vs m_b^2 and xi_b |
| `06_massive_bulk_mass_sweep` | Roots vs M^2 |
| `07_massive_ell_sweep` | Roots vs ell |
| `08_growth_rate_vs_kappa` | Im omega of the growing modes vs kappa |
| `09_growing_modes_complex_plane` | Growing modes in the complex omega plane, colored by kappa |
| `10_stability_diagram` | Stable and unstable regions in the (M^2, kappa) plane |
| `11_bifurcation_diagram` | Re omega and Im omega vs kappa near the Krein collision |
| `12_mode_profiles` | Radial profiles of the first real modes |
