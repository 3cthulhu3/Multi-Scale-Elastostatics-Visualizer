"""Figure builders.

Every builder returns a **plain, JSON-serialisable Plotly figure dict** (``{"data":
[...], "layout": {...}}``). That keeps the numerical core free of a hard Plotly
dependency, makes figures testable without a browser, and lets the same spec be
rendered by ``st.plotly_chart``, by ``plotly.graph_objects.Figure`` (see
:func:`as_plotly_figure`) or by Plotly.js in a static page (see :func:`to_html`).

Design rules: one y-scale per panel (no dual axes), piecewise-constant element
fields are drawn as steps, colour follows the entity (model or phase).
"""

from __future__ import annotations

import json

import numpy as np

from .fem import BarSolution
from .homogenization import REINFORCEMENT, RVEResult
from .materials import Microstructure, reuss_modulus, voigt_modulus
from .multiscale import MultiscaleResult, ScaleSeparationStudy

# Categorical colours, fixed order (validated for colour-vision deficiency).
COLORS = {
    "homogenized": "#2a78d6",
    "resolved": "#eb6834",
    "matrix": "#1baf7a",
    "reinforcement": "#eda100",
    "reference": "#5f5e5a",
}
PLOTLY_JS_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_BASE_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "size": 13},
    "margin": {"l": 70, "r": 20, "t": 70, "b": 55},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    "hovermode": "x unified",
}


def _layout(**overrides) -> dict:
    layout = json.loads(json.dumps(_BASE_LAYOUT))  # deep copy
    layout.update(overrides)
    return layout


def _list(a) -> list:
    return np.asarray(a, dtype=float).tolist()


def _step(x_nodes, element_values) -> tuple[list, list]:
    """Coordinates that draw a piecewise-constant element field as a step line."""
    x = np.asarray(x_nodes, dtype=float)
    v = np.asarray(element_values, dtype=float)
    return _list(np.repeat(x, 2)[1:-1]), _list(np.repeat(v, 2))


def _line(x, y, name, color, *, dash=None, width=2, mode="lines", xaxis="x", yaxis="y", **kw):
    trace = {
        "type": "scatter",
        "x": x,
        "y": y,
        "name": name,
        "mode": mode,
        "line": {"color": color, "width": width},
        "xaxis": xaxis,
        "yaxis": yaxis,
    }
    if dash:
        trace["line"]["dash"] = dash
    trace.update(kw)
    return trace


def _two_row_axes(top_title: str, bottom_title: str, x_title: str) -> dict:
    return {
        "xaxis": {"anchor": "y", "matches": "x2", "showticklabels": False},
        "yaxis": {"domain": [0.56, 1.0], "title": {"text": top_title}},
        "xaxis2": {"anchor": "y2", "title": {"text": x_title}},
        "yaxis2": {"domain": [0.0, 0.44], "title": {"text": bottom_title}},
    }


# --------------------------------------------------------------------------- macro
def macro_response_figure(
    result: MultiscaleResult,
    resolved: BarSolution | None = None,
    show_analytical: bool = True,
    show_phases: bool = True,
) -> dict:
    """Displacement (top) and axial stress (bottom) along the bar."""
    m = result.macro
    data = []

    if show_analytical:
        xf = np.linspace(0.0, result.problem.length, 300)
        data.append(
            _line(_list(xf), _list(result.analytical_displacement(xf) * 1e3),
                  "Analytical (E_eff)", COLORS["reference"], dash="dash", width=2)
        )
    if resolved is not None:
        data.append(
            _line(_list(resolved.x), _list(resolved.u * 1e3), "Fully resolved",
                  COLORS["resolved"], width=2)
        )
    data.append(
        _line(_list(m.x), _list(m.u * 1e3), "Homogenized FE", COLORS["homogenized"],
              mode="lines+markers", marker={"size": 8 if m.n_elements <= 40 else 5})
    )

    # bottom panel: stresses (element-wise constant -> steps)
    if resolved is not None:
        xs, ys = _step(resolved.x, resolved.stress / 1e6)
        data.append(_line(xs, ys, "Fully resolved", COLORS["resolved"],
                          xaxis="x2", yaxis="y2", showlegend=False))
    xs, ys = _step(m.x, m.stress / 1e6)
    data.append(_line(xs, ys, "Homogenized σ", COLORS["homogenized"], width=3,
                      xaxis="x2", yaxis="y2", showlegend=False))
    if show_phases:
        for phase in ("matrix", "reinforcement"):
            xs, ys = _step(m.x, result.phase_stress[phase] / 1e6)
            data.append(_line(xs, ys, f"{phase.capitalize()} σ (localized)", COLORS[phase],
                              dash="dot", xaxis="x2", yaxis="y2"))

    layout = _layout(
        title={"text": "Macro scale: displacement and axial stress"},
        height=620,
        **_two_row_axes("Displacement u (mm)", "Stress σ (MPa)", "Position x (m)"),
    )
    return {"data": data, "layout": layout}


# --------------------------------------------------------------------------- micro
def _phase_segments(x, element_phase):
    """Contiguous (start, end, phase_code) runs of the cell."""
    runs, start = [], 0
    for i in range(1, element_phase.size + 1):
        if i == element_phase.size or element_phase[i] != element_phase[start]:
            runs.append((float(x[start]), float(x[i]), int(element_phase[start])))
            start = i
    return runs


def rve_figure(rve: RVEResult) -> dict:
    """Unit-cell response to a unit average strain."""
    micro = rve.microstructure
    if rve.x is None:  # parallel: show per-phase ratios as bars
        phases = ["matrix", "reinforcement"]
        sigma_bar = rve.E_eff * rve.applied_strain
        stress = rve.phase_stress(rve.applied_strain)
        data = [
            {"type": "bar", "x": phases, "y": [rve.concentration[p] for p in phases],
             "name": "Strain ratio ε_i / ε̄", "marker": {"color": COLORS["homogenized"]}},
            {"type": "bar", "x": phases, "y": [float(stress[p]) / sigma_bar for p in phases],
             "name": "Stress ratio σ_i / σ̄", "marker": {"color": COLORS["resolved"]}},
        ]
        layout = _layout(
            title={"text": "Micro scale (parallel cell): iso-strain, stress concentrates in the stiffer phase"},
            barmode="group", bargap=0.35, bargroupgap=0.08, height=420, hovermode="closest",
            yaxis={"title": {"text": "Ratio to cell average"}},
        )
        return {"data": data, "layout": layout}

    xs, ys = _step(rve.x, rve.strain / rve.applied_strain)
    data = [
        _line(xs, ys, "ε / ε̄ (strain concentration)", COLORS["homogenized"], width=3),
        _line(_list(rve.x), _list(rve.fluctuation / rve.applied_strain),
              "Fluctuation (u − ε̄x) / ε̄", COLORS["resolved"], xaxis="x2", yaxis="y2"),
    ]
    shapes, annotations = [], []
    for x0, x1, code in _phase_segments(rve.x, rve.element_phase):
        name = "reinforcement" if code == REINFORCEMENT else "matrix"
        shapes.append({
            "type": "rect", "xref": "x2", "yref": "paper", "x0": x0, "x1": x1, "y0": 0, "y1": 1,
            "fillcolor": COLORS[name], "opacity": 0.14, "line": {"width": 0}, "layer": "below",
        })
        annotations.append({
            "x": 0.5 * (x0 + x1), "y": 1.0, "xref": "x2", "yref": "paper", "yanchor": "bottom",
            "text": name, "showarrow": False, "font": {"size": 12},
        })
    layout = _layout(
        title={"text": f"Micro scale (series cell, V_f = {micro.volume_fraction:.2f}): response to unit ε̄",
               "y": 0.98},
        height=560, shapes=shapes, annotations=annotations,
        **_two_row_axes("ε / ε̄", "(u − ε̄x) / ε̄", "Position in cell ξ = x / ℓ"),
    )
    layout["legend"]["y"] = 1.08
    layout["margin"] = {"l": 70, "r": 20, "t": 110, "b": 55}
    return {"data": data, "layout": layout}


def bounds_figure(micro: Microstructure, E_eff_numeric: float | None = None) -> dict:
    """Voigt/Reuss bounds versus volume fraction, with the current design point."""
    f = np.linspace(0.0, 1.0, 201)
    voigt = voigt_modulus(micro.E_matrix, micro.E_reinforcement, f) / 1e9
    reuss = reuss_modulus(micro.E_matrix, micro.E_reinforcement, f) / 1e9
    E_point = micro.analytical_modulus if E_eff_numeric is None else E_eff_numeric
    data = [
        _line(_list(f), _list(reuss), "Reuss (series, lower bound)", COLORS["resolved"]),
        _line(_list(f), _list(voigt), "Voigt (parallel, upper bound)", COLORS["homogenized"],
              fill="tonexty", fillcolor="rgba(95,94,90,0.08)"),
        {"type": "scatter", "x": [micro.volume_fraction], "y": [E_point / 1e9],
         "mode": "markers", "name": f"Current design ({micro.arrangement})",
         "marker": {"size": 12, "color": COLORS["reference"],
                    "line": {"color": "#fcfcfb", "width": 2}}},
    ]
    layout = _layout(
        title={"text": "Effective modulus bounds"}, height=420,
        xaxis={"title": {"text": "Reinforcement volume fraction V_f"}, "range": [0, 1]},
        yaxis={"title": {"text": "E_eff (GPa)"}},
    )
    return {"data": data, "layout": layout}


def convergence_figure(study: ScaleSeparationStudy) -> dict:
    """Homogenization error versus number of microstructural cells (log-log)."""
    n = study.cell_counts.astype(float)
    floor = 1e-16
    max_err = np.maximum(study.max_rel_error, floor)
    tip_err = np.maximum(study.tip_rel_error, floor)
    ref = max_err[0] * n[0] / n  # slope -1 guide anchored at the first point
    data = [
        _line(_list(n), _list(max_err), "max |u_res − u_hom| / max |u_hom|",
              COLORS["homogenized"], mode="lines+markers", marker={"size": 9}),
        _line(_list(n), _list(tip_err), "tip displacement error",
              COLORS["resolved"], mode="lines+markers", marker={"size": 9}),
        _line(_list(n), _list(ref), "O(1/N) reference", COLORS["reference"], dash="dash"),
    ]
    layout = _layout(
        title={"text": "Scale separation: homogenized vs fully resolved model"}, height=440,
        hovermode="closest",
        xaxis={"type": "log", "title": {"text": "Number of cells along the bar N"}},
        yaxis={"type": "log", "title": {"text": "Relative error"}, "exponentformat": "power"},
    )
    return {"data": data, "layout": layout}


def nodal_fields_figure(x, u, element_stress) -> dict:
    """Generic displacement/stress figure (used by the legacy API)."""
    xs, ys = _step(x, np.asarray(element_stress) / 1e6)
    data = [
        _line(_list(x), _list(np.asarray(u) * 1e6), "Displacement", COLORS["homogenized"], width=3),
        _line(xs, ys, "Stress", COLORS["resolved"], width=3, xaxis="x2", yaxis="y2"),
    ]
    layout = _layout(
        title={"text": "Beam analysis: displacement and stress"}, height=560,
        **_two_row_axes("Displacement (μm)", "Stress (MPa)", "Position (m)"),
    )
    return {"data": data, "layout": layout}


# --------------------------------------------------------------------------- output
def as_plotly_figure(spec: dict):
    """Convert a figure dict to ``plotly.graph_objects.Figure`` (validates the spec)."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError("plotly is required: pip install plotly") from exc
    return go.Figure(spec)


def to_html(figures: list[tuple[str, dict]], title: str, intro_html: str = "") -> str:
    """Self-contained HTML page that renders the figures with Plotly.js from a CDN."""
    blocks, scripts = [], []
    for i, (heading, spec) in enumerate(figures):
        div = f"fig{i}"
        blocks.append(f'<section><h2>{heading}</h2><div id="{div}" class="fig"></div></section>')
        payload = json.dumps(spec, separators=(",", ":"))
        scripts.append(
            f'(function () {{ const s = {payload}; Plotly.newPlot("{div}", s.data, s.layout, '
            '{"responsive": true, "displaylogo": false}); })();'
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="{PLOTLY_JS_CDN}" charset="utf-8"></script>
<style>
  body {{ font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; margin: 0 auto;
         max-width: 1100px; padding: 24px; color: #2b2b29; background: #fcfcfb; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
  .fig {{ width: 100%; }}
  p, li {{ line-height: 1.5; }}
  code {{ background: #efeeea; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{intro_html}
{''.join(blocks)}
<script>
{chr(10).join(scripts)}
</script>
</body>
</html>
"""
