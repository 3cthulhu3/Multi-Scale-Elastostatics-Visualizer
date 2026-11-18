import numpy as np
import pytest

from msev import (
    BarProblem,
    Microstructure,
    scale_separation_study,
    solve_multiscale,
    solve_resolved,
)

MICRO = Microstructure(3.5e9, 72e9, 0.4, "series")


@pytest.fixture
def problem():
    return BarProblem(length=1.5, area=1e-3, tip_load=1e4, distributed_load=4e3, n_elements=30)


def test_macro_solution_matches_analytical(problem):
    res = solve_multiscale(problem, MICRO)
    assert np.allclose(res.macro.u, res.analytical_displacement(), rtol=1e-10)
    assert res.equilibrium_residual() < 1e-10


def test_series_phase_stresses_equal_macro_stress(problem):
    res = solve_multiscale(problem, MICRO)
    for phase in ("matrix", "reinforcement"):
        assert np.allclose(res.phase_stress[phase], res.macro.stress, rtol=1e-10)
    # ...while the softer matrix strains more
    assert np.all(np.abs(res.phase_strain["matrix"]) > np.abs(res.phase_strain["reinforcement"]))


def test_parallel_phase_stresses_scale_with_modulus(problem):
    micro = Microstructure(3.5e9, 230e9, 0.6, "parallel")
    res = solve_multiscale(problem, micro)
    ratio = res.phase_stress["reinforcement"] / res.phase_stress["matrix"]
    assert np.allclose(ratio, 230 / 3.5)


@pytest.mark.parametrize("n_cells", [1, 3, 10])
def test_resolved_tip_displacement_equals_homogenized(problem, n_cells):
    """Whole symmetric cells: homogenization is exact at cell boundaries in 1D."""
    hom = solve_multiscale(problem, MICRO)
    ref = solve_resolved(problem, MICRO, n_cells)
    assert ref.tip_displacement == pytest.approx(hom.macro.tip_displacement, rel=1e-10)
    assert ref.reaction_left == pytest.approx(hom.macro.reaction_left, rel=1e-10)


def test_scale_separation_error_decays_like_one_over_n(problem):
    study = scale_separation_study(problem, MICRO, cell_counts=(2, 4, 8, 16, 32))
    assert np.all(np.diff(study.max_rel_error) < 0)
    slope = np.polyfit(np.log(study.cell_counts), np.log(study.max_rel_error), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)


def test_resolved_rejects_parallel(problem):
    with pytest.raises(ValueError, match="series"):
        solve_resolved(problem, Microstructure(1e9, 2e9, 0.5, "parallel"), 3)


def test_summary_is_consistent(problem):
    s = solve_multiscale(problem, MICRO).summary()
    assert s["E_reuss_Pa"] <= s["E_eff_Pa"] * (1 + 1e-12) <= s["E_voigt_Pa"] * (1 + 1e-12)
    assert s["reaction_N"] == pytest.approx(-problem.total_load)
    assert all(np.isfinite(v) for v in s.values())


@pytest.mark.parametrize(
    "kwargs",
    [dict(length=0), dict(area=-1), dict(n_elements=0), dict(tip_load=float("inf"))],
)
def test_problem_validation(kwargs):
    with pytest.raises(ValueError):
        BarProblem(**kwargs)


def test_no_files_written(tmp_path, monkeypatch, problem):
    monkeypatch.chdir(tmp_path)
    solve_multiscale(problem, MICRO)
    solve_resolved(problem, MICRO, 4)
    assert list(tmp_path.iterdir()) == []
