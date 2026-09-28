# Wentzell spectra in global AdS

Companion code for the preprint *Klein-Gordon Fields with Dynamical Wentzell Boundary Conditions in Global Anti-de Sitter Spacetime*

`wentzell_spectrum_plots.py` solves the mode quantization condition for a Klein-Gordon field in global AdS whose boundary value obeys its own Klein-Gordon equation on the Einstein static universe (dynamical Wentzell boundary conditions). It finds the real normal-mode frequencies and any exponentially growing modes, then writes figures and CSV tables. It needs only the Python standard library and Matplotlib.

## Requirements
 
Python 3 and Matplotlib 3.6 or newer (tested with Python 3.12 and Matplotlib 3.10).
 
```
pip install matplotlib
```
 
## Usage
 
```
python wentzell_spectrum_plots.py
```
 
Output goes to `generated_graphs/` next to the script. A full run takes well under a minute.

For more detailed run instructions (Not needed)

```
python wentzell_spectrum_plots.py --help
```
  
## Outputs
 
Each figure is written as PNG and PDF by default.
 
| File | Contents |
|---|---|
| `01_massless_intersections` | Both sides of the M² = 0 condition, roots marked |
| `02_massless_residual` | M² = 0 residual and its zeros |
| `03_massless_symmetric_overview` | M² = 0 condition for positive and negative ω |
| `04_massless_kappa_curves` | M² = 0 residual for several κ |
| `05_massless_spectrum_vs_kappa` | M² = 0 real roots vs κ, shaded where a growing mode exists |
| `06_massless_boundary_parameter_sweeps` | M² = 0 root ladders vs m_b² and ξ_b |
| `07_massive_intersections` | Both sides of the Gamma-function condition, roots marked |
| `08_massive_residual` | Massive residual and its zeros |
| `09_massive_gamma_quotient_literature_style` | A/B for ℓ = 0 and 1, positive and negative ω |
| `10_massive_bulk_mass_sweep` | Root ladders vs M² |
| `11_massive_ell_sweep` | Root ladders by ℓ |
| `12_massive_spectrum_vs_kappa` | Massive real roots vs κ, shaded where a growing mode exists |
| `13_massless_limit_validation` | Gamma form at ν = 1/2 against the cotangent form, and their difference |
| `14_massless_massive_low_mode_comparison` | Lowest roots, M² = 0 vs massive |
| `15_growth_rate_vs_kappa` | Im ω of the most unstable mode vs κ, M² = 0 and massive |
 
Data files:
 
- `massless_roots.csv`, `massive_roots.csv`: roots with residuals and parameters
- `parameter_sweep_roots.csv`: real roots from the κ and M² sweeps
- `growing_modes_vs_kappa.csv`: most unstable mode found at each κ, with a flag for whether a growing mode exists
- `run_manifest.txt`: parameters, roots, growing modes, and the list of generated files


