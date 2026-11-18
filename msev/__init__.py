"""msev - Multi-Scale Elastostatics Visualizer.

A small, verified two-scale finite element model of a composite bar:

    micro (unit cell / RVE)  --upscaling-->  E_eff
    macro (clamped bar)      --FE solve-->   u(x), sigma(x)
    macro strain             --localization--> phase strains / stresses

Quick start::

    from msev import BarProblem, Microstructure, solve_multiscale

    micro = Microstructure(E_matrix=3.5e9, E_reinforcement=70e9,
                           volume_fraction=0.4, arrangement="series")
    result = solve_multiscale(BarProblem(length=1.0, area=1e-2, tip_load=1e5), micro)
    print(result.summary())
"""

from .fem import BarSolution, solve_bar, uniform_mesh
from .homogenization import RVEResult, solve_rve
from .materials import ARRANGEMENTS, Microstructure, reuss_modulus, voigt_modulus
from .multiscale import (
    SMALL_STRAIN_LIMIT,
    BarProblem,
    MultiscaleResult,
    ScaleSeparationStudy,
    scale_separation_study,
    solve_multiscale,
    solve_resolved,
)

__version__ = "0.2.0"

__all__ = [
    "ARRANGEMENTS",
    "SMALL_STRAIN_LIMIT",
    "BarProblem",
    "BarSolution",
    "Microstructure",
    "MultiscaleResult",
    "RVEResult",
    "ScaleSeparationStudy",
    "reuss_modulus",
    "scale_separation_study",
    "solve_bar",
    "solve_multiscale",
    "solve_resolved",
    "solve_rve",
    "uniform_mesh",
    "voigt_modulus",
    "__version__",
]
