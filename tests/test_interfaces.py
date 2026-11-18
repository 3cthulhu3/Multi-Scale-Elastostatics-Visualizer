"""Exports, figures, CLI and the backward-compatible legacy API."""

import csv
import io
import json
import warnings

import numpy as np
import pytest

from msev import (
    BarProblem,
    Microstructure,
    export,
    plotting,
    scale_separation_study,
    solve_multiscale,
    solve_resolved,
)
from msev.cli import run
from msev.legacy import solve_elastostatics


@pytest.fixture(scope="module", params=["series", "parallel"])
def result(request):
    micro = Microstructure(3.5e9, 72e9, 0.5, request.param)
    return solve_multiscale(BarProblem(n_elements=12, distributed_load=1e3), micro)


# ---------------------------------------------------------------- exports
def test_csv_exports_roundtrip(result):
    rows = list(csv.DictReader(io.StringIO(export.nodes_csv(result.macro))))
    assert len(rows) == result.macro.x.size
    assert float(rows[-1]["u_m"]) == pytest.approx(result.macro.tip_displacement, rel=1e-9)
    rows = list(csv.DictReader(io.StringIO(export.elements_csv(result))))
    assert len(rows) == result.macro.n_elements
    assert "reinforcement_stress_Pa" in rows[0]


def test_summary_json(result):
    payload = json.loads(export.summary_json(result))
    assert payload["microstructure"]["arrangement"] == result.microstructure.arrangement
    assert payload["results"]["E_eff_Pa"] == pytest.approx(result.E_eff)


# ---------------------------------------------------------------- figures
def _figures(result):
    micro, problem = result.microstructure, result.problem
    figs = [
        plotting.macro_response_figure(result),
        plotting.rve_figure(result.rve),
        plotting.bounds_figure(micro, result.E_eff),
    ]
    if micro.arrangement == "series":
        figs.append(plotting.macro_response_figure(result, solve_resolved(problem, micro, 3)))
        figs.append(plotting.convergence_figure(scale_separation_study(problem, micro, (1, 2, 4))))
    return figs


def test_figures_are_json_serialisable_without_nans(result):
    for fig in _figures(result):
        text = json.dumps(fig, allow_nan=False)
        assert '"data"' in text and fig["data"]
        for trace in fig["data"]:
            if "x" in trace and "y" in trace:
                assert len(trace["x"]) == len(trace["y"])


def test_figures_have_no_dual_y_axes(result):
    for fig in _figures(result):
        for key, axis in fig["layout"].items():
            if key.startswith("yaxis"):
                assert "overlaying" not in axis


def test_figures_validate_with_plotly(result):
    pytest.importorskip("plotly")
    for fig in _figures(result):
        plotting.as_plotly_figure(fig)


def test_html_report(result):
    html = plotting.to_html([("A", plotting.bounds_figure(result.microstructure))], "t", "<p>x</p>")
    assert html.startswith("<!DOCTYPE html>")
    assert plotting.PLOTLY_JS_CDN in html and "Plotly.newPlot" in html
    assert len(html) < 200_000  # no embedded plotly.js bundle


# ---------------------------------------------------------------- CLI
def test_cli_writes_outputs(tmp_path, capsys):
    assert run(["--cells", "3", "--q", "1", "--out", str(tmp_path), "--html"]) == 0
    assert {p.name for p in tmp_path.iterdir()} == {
        "nodes.csv", "elements.csv", "summary.json", "report.html"}
    assert "E_eff" in capsys.readouterr().out


def test_cli_reports_invalid_input(capsys):
    assert run(["--vf", "2"]) == 2
    assert "volume_fraction" in capsys.readouterr().err


# ---------------------------------------------------------------- legacy API
def test_legacy_solver_reproduces_original_results(tmp_path):
    r = solve_elastostatics(output_dir=str(tmp_path))
    # reference values produced by the original implementation (v0.1)
    assert r["E_eff"] == pytest.approx(38e9)
    assert r["u"][-1] == pytest.approx(2.6315789473681806e-05, rel=1e-10)
    assert len(r["stress"]) == 51
    assert np.allclose(r["stress"], 1e6, rtol=1e-10)  # fixed: no spurious 0 at x = 0
    assert {p.name for p in tmp_path.iterdir()} == {"x.txt", "displacement.txt", "stress.txt"}
    r = solve_elastostatics(80e9, 5e9, 2.0, 3e6, 20, output_dir=None)
    assert r["u"][-1] == pytest.approx(0.0001043478260869571, rel=1e-10)


def test_legacy_src_shim_still_importable():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import importlib

        import src.solver as legacy_solver

        importlib.reload(legacy_solver)
    assert legacy_solver.solve_elastostatics is solve_elastostatics
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_legacy_plot_creates_missing_folder(tmp_path):
    pytest.importorskip("plotly")
    from msev.legacy import plot_multiscale

    solve_elastostatics(output_dir=str(tmp_path / "out"))
    html = tmp_path / "docs" / "index.html"
    fig = plot_multiscale(str(tmp_path / "out"), html_path=str(html))
    assert html.exists() and fig is not None
