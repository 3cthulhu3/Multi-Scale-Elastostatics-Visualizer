"""Regenerate the static demo page (``index.html``) from the solver.

    python scripts/build_demo.py            # writes ./index.html
    python scripts/build_demo.py out.html   # custom path

The page loads Plotly.js from a CDN, so it stays small (tens of kB instead of
several MB) and can be served by GitHub Pages without the Streamlit server.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msev import (  # noqa: E402
    BarProblem,
    Microstructure,
    __version__,
    plotting,  # noqa: E402
    scale_separation_study,
    solve_multiscale,
    solve_resolved,
)

INTRO = """
<p>Static snapshot of the <a href="https://github.com/3cthulhu3/Multi-Scale-Elastostatics-Visualizer">
Multi-Scale Elastostatics Visualizer</a> (msev {version}). Scenario: a 1&nbsp;m glass/epoxy bar
(A&nbsp;=&nbsp;1000&nbsp;mm², layers stacked along the axis, V<sub>f</sub>&nbsp;=&nbsp;0.5),
clamped at x&nbsp;=&nbsp;0, tip load 10&nbsp;kN plus a distributed load of 5&nbsp;kN/m.</p>
<p>The unit cell gives E<sub>eff</sub>&nbsp;=&nbsp;{E_eff:.3g}&nbsp;GPa (Reuss bound
{reuss:.3g}&nbsp;GPa, Voigt bound {voigt:.3g}&nbsp;GPa). The homogenized bar is compared with a fully
resolved model of 5 cells. For interactive parameters run <code>streamlit run app.py</code>.</p>
"""


def build(path: Path) -> Path:
    micro = Microstructure(E_matrix=3.5e9, E_reinforcement=72e9, volume_fraction=0.5,
                           arrangement="series")
    problem = BarProblem(length=1.0, area=1000e-6, tip_load=10e3, distributed_load=5e3,
                         n_elements=20)
    result = solve_multiscale(problem, micro)
    resolved = solve_resolved(problem, micro, n_cells=5)
    study = scale_separation_study(problem, micro)
    intro = INTRO.format(version=__version__, E_eff=result.E_eff / 1e9,
                         reuss=micro.reuss / 1e9, voigt=micro.voigt / 1e9)
    figures = [
        ("Macro response: homogenized vs fully resolved",
         plotting.macro_response_figure(result, resolved)),
        ("Unit cell: strain concentration and fluctuation", plotting.rve_figure(result.rve)),
        ("Effective modulus bounds", plotting.bounds_figure(micro, result.E_eff)),
        ("Scale separation", plotting.convergence_figure(study)),
    ]
    html = plotting.to_html(figures, "Multi-Scale Elastostatics Visualizer", intro)
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "index.html"
    out = build(target)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} kB)")
