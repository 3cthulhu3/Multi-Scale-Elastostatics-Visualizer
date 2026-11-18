"""Streamlit front end: ``streamlit run app.py``.

All physics lives in the ``msev`` package; this file only collects inputs,
calls the solver (cached, no files written) and renders the results.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from msev import (
    SMALL_STRAIN_LIMIT,
    BarProblem,
    Microstructure,
    __version__,
    analytical,
    export,
    plotting,
    scale_separation_study,
    solve_multiscale,
    solve_resolved,
)

st.set_page_config(page_title="Multi-Scale Elastostatics Visualizer", page_icon="🧮", layout="wide")

PRESETS = {
    "Glass/epoxy, layers across the load (series)": dict(e_m=3.5, e_r=72.0, vf=0.5, arr="series"),
    "Carbon/epoxy, fibres along the load (parallel)": dict(e_m=3.5, e_r=230.0, vf=0.6, arr="parallel"),
    "Aluminium/steel stack (series)": dict(e_m=70.0, e_r=200.0, vf=0.3, arr="series"),
}
DEFAULT_PRESET = next(iter(PRESETS))


def _apply_preset() -> None:
    p = PRESETS[st.session_state["preset"]]
    st.session_state.update(e_m=p["e_m"], e_r=p["e_r"], vf=p["vf"], arr=p["arr"])


if "e_m" not in st.session_state:  # first run: initialise widget state from the preset
    st.session_state["preset"] = DEFAULT_PRESET
    _apply_preset()


# ----------------------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Microstructure (micro scale)")
    st.selectbox("Preset", list(PRESETS), key="preset", on_change=_apply_preset)
    e_m = st.number_input("Matrix modulus E_m (GPa)", min_value=0.1, max_value=2000.0, step=0.5, key="e_m")
    e_r = st.number_input("Reinforcement modulus E_r (GPa)", min_value=0.1, max_value=2000.0, step=1.0, key="e_r")
    vf = st.slider("Reinforcement volume fraction V_f", 0.0, 1.0, step=0.01, key="vf")
    arrangement = st.radio(
        "Arrangement relative to the load",
        ["series", "parallel"],
        key="arr",
        horizontal=True,
        help="series: layers stacked along the bar (iso-stress, Reuss). "
        "parallel: continuous phases along the bar (iso-strain, Voigt).",
    )

    st.header("Bar (macro scale)")
    length = st.slider("Length L (m)", 0.1, 5.0, 1.0, 0.1)
    area_mm2 = st.number_input("Cross-section A (mm²)", min_value=1.0, max_value=1e6, value=1000.0, step=100.0)
    tip_kN = st.number_input("Tip load P (kN)", min_value=-1e4, max_value=1e4, value=10.0, step=1.0,
                             help="Positive = tension.")
    q_kNm = st.number_input("Distributed axial load q (kN/m)", min_value=-1e4, max_value=1e4,
                            value=0.0, step=1.0, help="Uniform load along the bar, e.g. self-weight.")
    n_el = st.slider("Macro finite elements", 1, 200, 40)

    st.header("Reference model")
    show_resolved = st.checkbox(
        "Overlay fully resolved model", value=arrangement == "series",
        disabled=arrangement != "series",
        help="Meshes every layer explicitly (series arrangement only).",
    )
    n_cells = st.slider("Cells along the bar N", 1, 50, 5, disabled=not show_resolved)


# ----------------------------------------------------------------------------- solve
@st.cache_data(show_spinner=False)
def run_models(e_m, e_r, vf, arrangement, length, area_mm2, tip_kN, q_kNm, n_el, n_cells, show_resolved):
    micro = Microstructure(e_m * 1e9, e_r * 1e9, vf, arrangement)
    problem = BarProblem(length, area_mm2 * 1e-6, tip_kN * 1e3, q_kNm * 1e3, n_el)
    result = solve_multiscale(problem, micro)
    resolved = solve_resolved(problem, micro, n_cells) if show_resolved and arrangement == "series" else None
    study = scale_separation_study(problem, micro) if arrangement == "series" else None
    return result, resolved, study


try:
    result, resolved, study = run_models(
        float(e_m), float(e_r), float(vf), arrangement, float(length), float(area_mm2),
        float(tip_kN), float(q_kNm), int(n_el), int(n_cells), bool(show_resolved),
    )
except (TypeError, ValueError, FloatingPointError) as exc:
    st.error(f"Invalid input: {exc}")
    st.stop()

micro, problem, macro = result.microstructure, result.problem, result.macro
summary = result.summary()

# ----------------------------------------------------------------------------- header
st.title("🧮 Multi-Scale Elastostatics Visualizer")
st.caption(
    "Two-scale finite element model of a composite bar: the unit cell (micro) gives the "
    "effective stiffness, the bar (macro) is solved with it, and macro strains are mapped back "
    "to stresses in each phase."
)

c = st.columns(5)
c[0].metric("Effective modulus E_eff", f"{result.E_eff/1e9:.3g} GPa",
            help=f"Reuss {summary['E_reuss_Pa']/1e9:.3g} GPa ≤ E_eff ≤ Voigt {summary['E_voigt_Pa']/1e9:.3g} GPa")
c[1].metric("Tip displacement", f"{macro.tip_displacement*1e3:.4g} mm")
c[2].metric("Max |σ| macro", f"{macro.max_abs_stress/1e6:.4g} MPa")
c[3].metric("Max |σ| matrix", f"{summary['max_abs_matrix_stress_Pa']/1e6:.4g} MPa")
c[4].metric("Max |σ| reinforcement", f"{summary['max_abs_reinforcement_stress_Pa']/1e6:.4g} MPa")

if summary["max_abs_phase_strain"] > SMALL_STRAIN_LIMIT:
    st.warning(
        f"Maximum phase strain is {summary['max_abs_phase_strain']:.1%} (> {SMALL_STRAIN_LIMIT:.0%}). "
        "The linear small-strain model is questionable here and the matrix may already have failed; "
        "reduce the load or increase the cross-section."
    )

tab_macro, tab_micro, tab_verify, tab_export, tab_theory = st.tabs(
    ["Macro response", "Unit cell", "Verification", "Export", "Theory"]
)

with tab_macro:
    st.plotly_chart(plotting.macro_response_figure(result, resolved))
    if micro.arrangement == "series":
        st.info(
            "Series cell: every layer carries the same stress, so the localized matrix and "
            "reinforcement stresses coincide with the macro stress, while their strains differ "
            "(see *Unit cell*). The resolved model shows the zig-zag displacement the homogenized "
            "model averages out."
        )
    else:
        st.info(
            "Parallel cell: both phases share the macro strain; the stiffer phase carries "
            "proportionally more stress."
        )

with tab_micro:
    st.plotly_chart(plotting.rve_figure(result.rve))
    rows = [
        {"phase": p,
         "E (GPa)": micro.phase_moduli[p] / 1e9,
         "volume fraction": micro.phase_fractions[p],
         "strain concentration ε_i/ε̄": result.rve.concentration[p],
         "max |σ| in bar (MPa)": float(np.max(np.abs(result.phase_stress[p]))) / 1e6}
        for p in ("matrix", "reinforcement")
    ]
    st.dataframe(rows, hide_index=True)

with tab_verify:
    x_fine = macro.x
    u_exact = analytical.bar_displacement(x_fine, result.E_eff, problem.area, problem.length,
                                          problem.tip_load, problem.distributed_load)
    fe_err = float(np.max(np.abs(macro.u - u_exact)) / (np.max(np.abs(u_exact)) or 1.0))
    rve_err = abs(result.E_eff / micro.analytical_modulus - 1.0)
    checks = [
        ("Macro FE vs analytical displacement (nodal, relative)", fe_err, 1e-9),
        (f"RVE E_eff vs closed-form {'Reuss' if micro.arrangement == 'series' else 'Voigt'}", rve_err, 1e-9),
        ("Global equilibrium |R + P + qL| / |P + qL|", summary["equilibrium_residual"], 1e-9),
    ]
    st.dataframe(
        [{"check": n, "value": f"{v:.2e}", "tolerance": f"{t:.0e}", "status": "✅ pass" if v <= t else "❌ fail"}
         for n, v, t in checks],
        hide_index=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plotting.bounds_figure(micro, result.E_eff))
    with col_b:
        if study is not None:
            st.plotly_chart(plotting.convergence_figure(study))
            st.caption(
                "The homogenized model converges to the resolved one as the cells shrink "
                "(error ∝ 1/N). Tip displacement is exact whenever the bar holds whole, "
                "symmetric cells."
            )
        else:
            st.write("For the parallel arrangement the homogenized 1D model is exact; "
                     "there is no scale-separation error to study.")

with tab_export:
    st.download_button("Nodes (CSV)", export.nodes_csv(macro), "nodes.csv", "text/csv")
    st.download_button("Elements incl. phase stresses (CSV)", export.elements_csv(result),
                       "elements.csv", "text/csv")
    st.download_button("Inputs and summary (JSON)", export.summary_json(result),
                       "summary.json", "application/json")
    figs = [("Macro response", plotting.macro_response_figure(result, resolved)),
            ("Unit cell", plotting.rve_figure(result.rve)),
            ("Bounds", plotting.bounds_figure(micro, result.E_eff))]
    if study is not None:
        figs.append(("Scale separation", plotting.convergence_figure(study)))
    st.download_button("Interactive report (HTML)", plotting.to_html(figs, "msev report"),
                       "msev_report.html", "text/html")

with tab_theory:
    st.markdown("**Macro problem** (clamped at x = 0, tip load P, distributed load q):")
    st.latex(r"\frac{d}{dx}\left(E_\mathrm{eff} A \frac{du}{dx}\right) + q = 0,\qquad "
             r"u(0)=0,\qquad E_\mathrm{eff} A\,u'(L) = P")
    st.markdown("**Micro problem** (unit cell under average strain ε̄, periodic BCs):")
    st.latex(r"E_\mathrm{eff} = \frac{\langle \sigma \rangle}{\bar\varepsilon},\qquad "
             r"\varepsilon_i = A_i\,\bar\varepsilon,\qquad \sigma_i = E_i\,A_i\,\bar\varepsilon")
    st.markdown("**Closed-form 1D limits** (also the bounds for any arrangement):")
    st.latex(r"E_\mathrm{Reuss} = \left(\frac{V_f}{E_r} + \frac{1-V_f}{E_m}\right)^{-1} "
             r"\le E_\mathrm{eff} \le V_f E_r + (1-V_f) E_m = E_\mathrm{Voigt}")
    st.markdown(
        "Assumptions: small strains, linear elasticity, perfect bonding between phases, "
        "uniaxial stress, scale separation (cell size ≪ bar length). See `docs/THEORY.md`."
    )

st.caption(f"msev {__version__} · results are computed in memory; nothing is written to disk.")
