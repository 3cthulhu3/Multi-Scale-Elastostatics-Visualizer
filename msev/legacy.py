"""Backward-compatible versions of the original ``src.solver`` / ``src.visualizer`` API.

Kept so that existing scripts keep working. New code should use
:func:`msev.solve_multiscale`, which models the microstructure explicitly.

Differences from the original implementation (bug fixes):

* The nodal stress array no longer starts with a spurious ``0`` at the clamped
  end; it repeats the first element stress (the stress is constant there).
* ``plot_multiscale`` creates the HTML output folder if needed (the original
  crashed because ``docs/`` did not exist) and does not open a browser unless
  ``show=True``.
"""

from __future__ import annotations

import os

import numpy as np

from .fem import solve_bar, uniform_mesh
from .plotting import as_plotly_figure, nodal_fields_figure

LEGACY_INCLUSION_FRACTION = 0.3  # the original hard-coded "alpha"


def solve_elastostatics(
    E_macro: float = 50e9,
    E_micro: float = 10e9,
    length: float = 1.0,
    load: float = 1e6,
    nx: int = 50,
    output_dir: str | None = "output",
) -> dict:
    """Original API: Voigt mix with 30 % micro phase, unit cross-section (A = 1 m^2).

    Returns ``{"x", "u", "stress", "E_eff"}`` with nodal ``stress`` (length ``nx + 1``).
    Writes ``x.txt``, ``displacement.txt`` and ``stress.txt`` to ``output_dir``
    unless it is ``None``.
    """
    alpha = LEGACY_INCLUSION_FRACTION
    E_eff = alpha * E_micro + (1 - alpha) * E_macro
    sol = solve_bar(uniform_mesh(length, nx), E_eff, area=1.0, tip_load=load)
    stress = np.insert(sol.stress, 0, sol.stress[0])

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(os.path.join(output_dir, "x.txt"), sol.x)
        np.savetxt(os.path.join(output_dir, "displacement.txt"), sol.u)
        np.savetxt(os.path.join(output_dir, "stress.txt"), stress)

    return {"x": sol.x, "u": sol.u, "stress": stress, "E_eff": E_eff}


def plot_multiscale(output_dir: str = "output", html_path: str | None = None, show: bool = False):
    """Original API: plot the files written by :func:`solve_elastostatics`.

    Returns a ``plotly.graph_objects.Figure`` (requires plotly).
    """
    x = np.loadtxt(os.path.join(output_dir, "x.txt"))
    u = np.loadtxt(os.path.join(output_dir, "displacement.txt"))
    nodal_stress = np.loadtxt(os.path.join(output_dir, "stress.txt"))
    fig = as_plotly_figure(nodal_fields_figure(x, u, nodal_stress[1:]))
    if html_path:
        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        fig.write_html(html_path, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig

