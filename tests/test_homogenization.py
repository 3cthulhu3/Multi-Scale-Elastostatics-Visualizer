import numpy as np
import pytest

from msev import Microstructure, solve_rve
from msev.homogenization import REINFORCEMENT, series_cell_mesh

Em, Er = 3.5e9, 72e9


@pytest.mark.parametrize("vf", [0.0, 0.05, 0.37, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("arrangement", ["series", "parallel"])
def test_rve_reproduces_closed_form(vf, arrangement):
    micro = Microstructure(Em, Er, vf, arrangement)
    rve = solve_rve(micro, n_elements=41)
    assert rve.E_eff == pytest.approx(micro.analytical_modulus, rel=1e-10)


@pytest.mark.parametrize("vf", np.linspace(0, 1, 11))
def test_bounds_order(vf):
    m = Microstructure(Em, Er, float(vf))
    assert m.reuss <= m.voigt * (1 + 1e-12)
    assert min(Em, Er) <= m.reuss * (1 + 1e-12) and m.voigt <= max(Em, Er) * (1 + 1e-12)


def test_series_mesh_places_nodes_on_interfaces():
    x, phase = series_cell_mesh(0.3, 20)
    h = np.diff(x)
    assert np.sum(h[phase == REINFORCEMENT]) == pytest.approx(0.3, abs=1e-14)
    assert x[0] == 0.0 and x[-1] == pytest.approx(1.0)
    assert phase[0] != REINFORCEMENT and phase[-1] != REINFORCEMENT


def test_concentration_factors_series():
    micro = Microstructure(Em, Er, 0.4, "series")
    rve = solve_rve(micro)
    for name, E in micro.phase_moduli.items():
        assert rve.concentration[name] == pytest.approx(rve.E_eff / E, rel=1e-10)
    # volume average of phase strains recovers the applied strain
    f = micro.phase_fractions
    avg = sum(f[p] * rve.concentration[p] for p in f)
    assert avg == pytest.approx(1.0, rel=1e-12)


def test_concentration_factors_parallel_and_stress_average():
    micro = Microstructure(Em, Er, 0.6, "parallel")
    rve = solve_rve(micro)
    assert rve.concentration == pytest.approx({"matrix": 1.0, "reinforcement": 1.0})
    stress = rve.phase_stress(2e-3)
    f = micro.phase_fractions
    assert sum(f[p] * stress[p] for p in f) == pytest.approx(rve.E_eff * 2e-3, rel=1e-12)


def test_fluctuation_is_periodic_and_zero_mean_strain():
    rve = solve_rve(Microstructure(Em, Er, 0.3, "series"), applied_strain=1e-3)
    w = rve.fluctuation
    assert w[0] == pytest.approx(0.0, abs=1e-18) and w[-1] == pytest.approx(0.0, abs=1e-15)


def test_localization_is_linear():
    rve = solve_rve(Microstructure(Em, Er, 0.3, "series"))
    eps = np.array([1e-4, -2e-4])
    out = rve.phase_strain(eps)
    assert np.allclose(out["matrix"], rve.concentration["matrix"] * eps)


@pytest.mark.parametrize(
    "kwargs",
    [dict(E_matrix=0), dict(E_reinforcement=-1), dict(volume_fraction=1.2),
     dict(volume_fraction=float("nan")), dict(arrangement="random")],
)
def test_microstructure_validation(kwargs):
    args = dict(E_matrix=Em, E_reinforcement=Er, volume_fraction=0.5, arrangement="series")
    args.update(kwargs)
    with pytest.raises(ValueError):
        Microstructure(**args)


def test_rve_rejects_too_coarse_mesh_and_zero_strain():
    micro = Microstructure(Em, Er, 0.5)
    with pytest.raises(ValueError):
        solve_rve(micro, n_elements=2)
    with pytest.raises(ValueError):
        solve_rve(micro, applied_strain=0.0)
