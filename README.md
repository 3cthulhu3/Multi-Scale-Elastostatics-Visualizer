# Multi-Scale Elastostatics Visualizer

[![CI](https://github.com/3cthulhu3/Multi-Scale-Elastostatics-Visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/3cthulhu3/Multi-Scale-Elastostatics-Visualizer/actions/workflows/ci.yml)

An interactive, **verified** two-scale finite element model of a composite bar.
It shows how the micro-structure of a two-phase composite (matrix + reinforcement)
determines the macroscopic stiffness of a part, and how macroscopic loads are
carried by each phase.

```
 micro: unit cell (RVE)  ──upscaling──▶  E_eff
 macro: clamped bar      ──FE solve──▶   u(x), σ(x)
 macro strain            ──localization──▶ strain & stress in each phase
```

**Who it is for:** students and instructors of composite mechanics, FEM or
homogenization, and engineers who want a quick, transparent sanity check of
rule-of-mixtures estimates.

**What you can do with it**

- Change phase moduli, volume fraction and arrangement (*series* = layers stacked
  along the load, *parallel* = continuous phases along the load) and see the
  effective modulus move between the Reuss and Voigt bounds.
- Load the bar with a tip force and/or a distributed axial load and inspect
  displacement, macro stress and the *localized* stress in matrix and reinforcement.
- Compare the homogenized model with a **fully resolved** model in which every
  layer is meshed, and watch the error vanish as the cells get smaller (scale
  separation, error ∝ 1/N).
- Export nodal/element results (CSV), inputs + summary (JSON) and an interactive
  HTML report.

A static snapshot of the app lives in [`index.html`](index.html) (≈50 kB, Plotly.js
from a CDN). Enable GitHub Pages for the repository root to publish it.

## Quick start

```bash
git clone https://github.com/3cthulhu3/Multi-Scale-Elastostatics-Visualizer.git
cd Multi-Scale-Elastostatics-Visualizer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Results are computed in memory; nothing
is written to disk.

### Command line

```bash
pip install -e .                       # installs the `msev` command
msev --e-matrix 3.5 --e-reinforcement 72 --vf 0.5 --arrangement series \
     --length 1 --area 1000 --tip-load 10 --q 5 --cells 5 --out results --html
```

Units on the command line: GPa, mm², kN, kN/m. `msev --help` lists all options.
Invalid input exits with code 2 and a message naming the bad parameter.

### Python API

```python
from msev import BarProblem, Microstructure, solve_multiscale, solve_resolved

micro = Microstructure(E_matrix=3.5e9, E_reinforcement=72e9,
                       volume_fraction=0.5, arrangement="series")      # SI units
problem = BarProblem(length=1.0, area=1e-3, tip_load=1e4, distributed_load=5e3)

result = solve_multiscale(problem, micro)
print(result.E_eff / 1e9, "GPa")                  # homogenized modulus from the RVE
print(result.macro.u[-1] * 1e3, "mm")             # tip displacement
print(result.phase_stress["matrix"].max() / 1e6)  # localized matrix stress, MPa

reference = solve_resolved(problem, micro, n_cells=10)  # every layer meshed
```

Figures are plain Plotly dicts (`msev.plotting.*_figure`) and can be shown with
`st.plotly_chart`, `plotly.graph_objects.Figure(spec)` or written to a static page
with `msev.plotting.to_html`.

## How it works

| Step | Module | What happens |
|---|---|---|
| Inputs | `msev.materials`, `msev.multiscale.BarProblem` | validated dataclasses (SI units) |
| Upscaling | `msev.homogenization.solve_rve` | FE solve of a unit cell under an average strain → `E_eff`, strain-concentration factors |
| Macro solve | `msev.fem.solve_bar` | sparse 1D linear FE, consistent loads, Dirichlet by elimination |
| Downscaling | `RVEResult.phase_strain / phase_stress` | macro element strain → phase strain/stress |
| Reference | `msev.multiscale.solve_resolved` | fully resolved layered bar (series only) |
| Output | `msev.plotting`, `msev.export`, `app.py`, `msev.cli` | figures, CSV/JSON/HTML, UI |

The equations, assumptions and their limits are in [`docs/THEORY.md`](docs/THEORY.md).

## Verification

Every result shown in the app is backed by an automated check (`pytest`), and the
app's **Verification** tab repeats the key checks live for the current inputs.

| Check | Expected | Test |
|---|---|---|
| Macro FE vs analytical bar solution | nodally exact (≤ 1e-10) | `tests/test_fem.py` |
| Element stress vs exact stress at midpoints | exact | `tests/test_fem.py` |
| Global equilibrium `R + P + qL = 0` | ≤ 1e-10 | `tests/test_fem.py`, `test_multiscale.py` |
| RVE `E_eff` vs Reuss (series) / Voigt (parallel) | ≤ 1e-10 for V_f ∈ [0, 1] | `tests/test_homogenization.py` |
| Reuss ≤ E_eff ≤ Voigt | always | `tests/test_homogenization.py` |
| Resolved vs homogenized tip displacement | equal for whole cells | `tests/test_multiscale.py` |
| Homogenization error vs cell count | slope −1 on log-log | `tests/test_multiscale.py` |
| Original v0.1 numbers through the legacy API | identical | `tests/test_interfaces.py` |

```bash
pip install -r requirements-dev.txt
pytest -q          # 91 tests
ruff check .
```

## Project layout

```
app.py                 Streamlit UI (inputs, tabs, downloads) – no physics here
msev/
  materials.py         Microstructure + Voigt/Reuss
  fem.py               1D bar FE solver (sparse)
  homogenization.py    unit-cell (RVE) solve, concentration factors
  multiscale.py        BarProblem, two-scale chain, resolved model, scale study
  analytical.py        closed-form bar solution for verification
  plotting.py          Plotly figure dicts + static HTML
  export.py            CSV / JSON strings
  cli.py               `msev` command
  legacy.py            v0.1 API (solve_elastostatics, plot_multiscale)
src/                   deprecated shims re-exporting msev.legacy
scripts/build_demo.py  regenerates index.html
tests/                 pytest suite
docs/THEORY.md         equations and assumptions
```

## Migrating from v0.1

`from src.solver import solve_elastostatics` and `from src.visualizer import
plot_multiscale` still work (with a `DeprecationWarning`). Prefer
`msev.legacy` for the same behaviour, or `msev.solve_multiscale` for the new
model. Behaviour changes, all bug fixes, are listed in [`CHANGELOG.md`](CHANGELOG.md).

## Scope and limitations

- 1D, linear elastic, small strains, perfectly bonded phases. The app warns when
  a phase strain exceeds 1 %.
- Two phases, periodic microstructure, one unit cell type per bar.
- The *parallel* arrangement is exact with the Voigt modulus in 1D, so the
  resolved-model comparison is offered for *series* cells only.

Possible next steps: 2D unit cells with periodic boundary conditions
(Hashin–Shtrikman bounds as a check), spatially graded volume fraction along
the bar, and damage or plasticity in the matrix (true FE² coupling).
