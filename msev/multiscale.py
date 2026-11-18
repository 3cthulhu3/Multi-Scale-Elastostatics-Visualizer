"""Two-scale workflow: micro (RVE) -> macro (bar) -> back to micro.

1. **Upscaling**   ``solve_rve`` gives the effective modulus ``E_eff``.
2. **Macro solve** the bar is solved with the homogenized modulus.
3. **Downscaling** each macro element strain is mapped to phase strains and
   stresses with the RVE concentration factors.

``solve_resolved`` builds the *fully resolved* reference model (every layer of
the microstructure meshed explicitly, a.k.a. direct numerical simulation).
Comparing it with the homogenized model as the number of cells grows
(``scale_separation_study``) shows when the two-scale approximation is valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import analytical
from .fem import BarSolution, solve_bar, uniform_mesh
from .homogenization import REINFORCEMENT, RVEResult, series_cell_mesh, solve_rve
from .materials import Microstructure
from .validation import require_finite, require_int_at_least, require_positive

#: Above this strain the linear, small-strain assumption should be questioned.
SMALL_STRAIN_LIMIT = 0.01


@dataclass(frozen=True)
class BarProblem:
    """Macro-scale boundary value problem: a bar clamped at x = 0."""

    length: float = 1.0  # m
    area: float = 1.0e-2  # m^2
    tip_load: float = 1.0e5  # N
    distributed_load: float = 0.0  # N/m
    n_elements: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", require_positive("length", self.length))
        object.__setattr__(self, "area", require_positive("area", self.area))
        object.__setattr__(self, "tip_load", require_finite("tip_load", self.tip_load))
        object.__setattr__(
            self, "distributed_load", require_finite("distributed_load", self.distributed_load)
        )
        object.__setattr__(
            self, "n_elements", require_int_at_least("n_elements", self.n_elements, 1)
        )

    @property
    def total_load(self) -> float:
        return self.tip_load + self.distributed_load * self.length


@dataclass(frozen=True)
class MultiscaleResult:
    problem: BarProblem
    microstructure: Microstructure
    rve: RVEResult
    macro: BarSolution
    phase_strain: dict[str, np.ndarray] = field(repr=False)
    phase_stress: dict[str, np.ndarray] = field(repr=False)

    @property
    def E_eff(self) -> float:
        return self.rve.E_eff

    @property
    def bounds(self) -> tuple[float, float]:
        """(Reuss, Voigt) bounds for the current volume fraction."""
        return self.microstructure.reuss, self.microstructure.voigt

    def analytical_displacement(self, x=None) -> np.ndarray:
        x = self.macro.x if x is None else x
        p = self.problem
        return analytical.bar_displacement(
            x, self.E_eff, p.area, p.length, p.tip_load, p.distributed_load
        )

    def equilibrium_residual(self) -> float:
        """|reaction + applied load| / |applied load| (0 for a converged solve)."""
        total = self.problem.total_load
        scale = abs(total) if total else 1.0
        return abs(self.macro.reaction_left + total) / scale

    def summary(self) -> dict[str, float]:
        reuss, voigt = self.bounds
        return {
            "E_eff_Pa": self.E_eff,
            "E_reuss_Pa": reuss,
            "E_voigt_Pa": voigt,
            "tip_displacement_m": self.macro.tip_displacement,
            "max_abs_displacement_m": self.macro.max_abs_displacement,
            "max_abs_macro_stress_Pa": self.macro.max_abs_stress,
            "max_abs_matrix_stress_Pa": float(np.max(np.abs(self.phase_stress["matrix"]))),
            "max_abs_reinforcement_stress_Pa": float(
                np.max(np.abs(self.phase_stress["reinforcement"]))
            ),
            "max_abs_phase_strain": float(
                max(np.max(np.abs(e)) for e in self.phase_strain.values())
            ),
            "reaction_N": self.macro.reaction_left,
            "equilibrium_residual": self.equilibrium_residual(),
        }


def solve_multiscale(
    problem: BarProblem, micro: Microstructure, n_micro: int = 200
) -> MultiscaleResult:
    """Run the full upscaling -> macro solve -> localization chain."""
    rve = solve_rve(micro, n_elements=n_micro)
    x = uniform_mesh(problem.length, problem.n_elements)
    macro = solve_bar(x, rve.E_eff, problem.area, problem.tip_load, problem.distributed_load)
    return MultiscaleResult(
        problem=problem,
        microstructure=micro,
        rve=rve,
        macro=macro,
        phase_strain=rve.phase_strain(macro.strain),
        phase_stress=rve.phase_stress(macro.strain),
    )


def solve_resolved(
    problem: BarProblem, micro: Microstructure, n_cells: int, elements_per_cell: int = 12
) -> BarSolution:
    """Fully resolved reference model with ``n_cells`` layered cells along the bar.

    Only defined for the ``series`` arrangement: in 1D a parallel arrangement is
    already represented exactly by its Voigt modulus, so there is nothing to
    resolve along the axis.
    """
    if micro.arrangement != "series":
        raise ValueError("solve_resolved is only defined for the 'series' arrangement")
    n_cells = require_int_at_least("n_cells", n_cells, 1)
    cell_x, cell_phase = series_cell_mesh(micro.volume_fraction, elements_per_cell)
    cell_len = problem.length / n_cells
    starts = np.arange(n_cells) * cell_len
    x = np.concatenate(
        [s + cell_len * cell_x[:-1] for s in starts] + [np.array([problem.length])]
    )
    phase = np.tile(cell_phase, n_cells)
    E = np.where(phase == REINFORCEMENT, micro.E_reinforcement, micro.E_matrix)
    return solve_bar(x, E, problem.area, problem.tip_load, problem.distributed_load)


@dataclass(frozen=True)
class ScaleSeparationStudy:
    cell_counts: np.ndarray
    max_rel_error: np.ndarray  # max |u_resolved - u_homogenized| / max |u_homogenized|
    tip_rel_error: np.ndarray  # |u_tip,resolved - u_tip,homogenized| / |u_tip,homogenized|


def scale_separation_study(
    problem: BarProblem,
    micro: Microstructure,
    cell_counts=(1, 2, 4, 8, 16, 32, 64),
    elements_per_cell: int = 12,
) -> ScaleSeparationStudy:
    """Error of the homogenized model w.r.t. resolved models with more and more cells."""
    E_eff = micro.analytical_modulus
    counts = np.asarray(cell_counts, dtype=int)
    max_err, tip_err = [], []
    for n in counts:
        ref = solve_resolved(problem, micro, int(n), elements_per_cell)
        u_hom = analytical.bar_displacement(
            ref.x, E_eff, problem.area, problem.length, problem.tip_load, problem.distributed_load
        )
        scale = np.max(np.abs(u_hom)) or 1.0
        max_err.append(np.max(np.abs(ref.u - u_hom)) / scale)
        tip_scale = abs(u_hom[-1]) or 1.0
        tip_err.append(abs(ref.u[-1] - u_hom[-1]) / tip_scale)
    return ScaleSeparationStudy(counts, np.asarray(max_err), np.asarray(tip_err))
