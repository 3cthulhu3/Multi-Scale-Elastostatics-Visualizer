import numpy as np
import pytest

from msev import analytical
from msev.fem import solve_bar, uniform_mesh

E, A, L = 70e9, 1e-3, 2.0


@pytest.mark.parametrize("n", [1, 3, 17, 100])
@pytest.mark.parametrize("P, q", [(1e4, 0.0), (0.0, 5e3), (-2e4, 3e3)])
def test_linear_fe_is_nodally_exact(n, P, q):
    """Linear elements + consistent loads reproduce the 1D solution exactly at the nodes."""
    sol = solve_bar(uniform_mesh(L, n), E, A, P, q)
    exact = analytical.bar_displacement(sol.x, E, A, L, P, q)
    assert np.allclose(sol.u, exact, rtol=1e-10, atol=1e-18)


def test_element_stress_equals_exact_stress_at_midpoints():
    sol = solve_bar(uniform_mesh(L, 25), E, A, 1e4, 2e3)
    exact = analytical.bar_stress(sol.midpoints, A, L, 1e4, 2e3)
    assert np.allclose(sol.stress, exact, rtol=1e-10)


def test_global_equilibrium():
    P, q = 1e4, 2e3
    sol = solve_bar(uniform_mesh(L, 10), E, A, P, q)
    assert sol.reaction_left == pytest.approx(-(P + q * L), rel=1e-12)


def test_stress_is_constant_and_nonzero_at_clamped_end():
    """Regression: the original code reported zero stress at the clamped end."""
    sol = solve_bar(uniform_mesh(1.0, 50), 38e9, 1.0, 1e6)
    assert np.allclose(sol.stress, 1e6, rtol=1e-12)
    assert sol.stress[0] != 0.0


def test_prescribed_tip_displacement_and_reactions_balance():
    x = np.array([0.0, 0.3, 1.0])
    sol = solve_bar(x, [10e9, 30e9], 1.0, tip_displacement=1e-3)
    assert sol.u[-1] == pytest.approx(1e-3)
    assert sol.reaction_left == pytest.approx(-sol.reaction_right)
    assert np.allclose(sol.axial_force, sol.axial_force[0])  # series bar: same force


def test_heterogeneous_series_bar_matches_compliance_sum():
    x = np.array([0.0, 0.4, 1.0])
    Es = np.array([10e9, 40e9])
    sol = solve_bar(x, Es, 2e-3, tip_load=5e3)
    expected = 5e3 / 2e-3 * (0.4 / 10e9 + 0.6 / 40e9)
    assert sol.tip_displacement == pytest.approx(expected, rel=1e-12)


def test_non_uniform_mesh():
    x = np.sort(np.concatenate([[0.0, L], np.random.default_rng(0).uniform(0, L, 30)]))
    sol = solve_bar(x, E, A, 1e4, 1e3)
    assert np.allclose(sol.u, analytical.bar_displacement(x, E, A, L, 1e4, 1e3), rtol=1e-9)


@pytest.mark.parametrize(
    "kwargs, err",
    [
        (dict(x=[0.0, 1.0, 1.0]), ValueError),
        (dict(x=[0.0]), ValueError),
        (dict(modulus=0.0), ValueError),
        (dict(modulus=[1e9, -1e9]), ValueError),
        (dict(area=0.0), ValueError),
        (dict(tip_load=float("nan")), ValueError),
        (dict(area="1"), TypeError),
    ],
)
def test_invalid_inputs_raise(kwargs, err):
    args = dict(x=[0.0, 0.5, 1.0], modulus=1e9, area=1.0, tip_load=1.0)
    args.update(kwargs)
    with pytest.raises(err):
        solve_bar(**args)


def test_uniform_mesh_validation():
    with pytest.raises(ValueError):
        uniform_mesh(1.0, 0)
    with pytest.raises(TypeError):
        uniform_mesh(1.0, 2.5)
