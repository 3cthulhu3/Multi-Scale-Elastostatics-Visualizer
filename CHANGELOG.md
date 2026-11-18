# Changelog

## 0.2.0

### Added
- `msev` package: sparse 1D FE solver, numerical unit-cell (RVE) homogenization,
  localization of macro strains to phase strains/stresses, fully resolved
  reference model and scale-separation study, closed-form verification.
- Microstructure inputs: phase moduli, volume fraction, series/parallel arrangement,
  plus cross-section area and distributed axial load for the bar.
- Streamlit app rewritten: live (cached) solving, presets, *Unit cell*,
  *Verification*, *Export* and *Theory* tabs, small-strain warning.
- `msev` command-line interface with CSV/JSON/HTML output.
- 91-test pytest suite, GitHub Actions CI (Python 3.9/3.11/3.13), ruff config,
  `pyproject.toml`, `.gitignore`, `docs/THEORY.md`.
- `scripts/build_demo.py` regenerates the static `index.html`.

### Fixed
- The app crashed after clicking *Solve*: `plot_multiscale` wrote to a non-existent
  `docs/` folder. The legacy function now creates the folder.
- `plot_multiscale` no longer opens a browser on the server (`show=False` by default).
- The stress at the clamped end was reported as `0`; it is now the element stress.
- Solver and plotter communicated via `output/*.txt` in the working directory, which
  is unsafe with several concurrent app users. The app now works fully in memory.
- Dual y-axis plot replaced by two aligned panels (displacement and stress).
- Pinned `numpy==1.26.4` could not be installed on Python ≥ 3.13; requirements now
  use lower bounds. Unused `matplotlib` and `meshio` were removed.
- README no longer advertises a "live demo" that pointed to `localhost`.

### Changed
- `index.html` is regenerated with Plotly.js from a CDN (3.6 MB → about 50 kB).
- `src/solver.py` and `src/visualizer.py` are deprecated shims around `msev.legacy`.

### Removed
- `beammsh.msh` (placeholder text file, not a mesh; never read by the code).
