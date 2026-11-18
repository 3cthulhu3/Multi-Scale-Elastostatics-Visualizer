"""Micro scale: numerical homogenization of a periodic unit cell (RVE).

The unit cell (normalised length 1, unit area) is loaded by an average strain
``eps_bar``. In 1D the periodic boundary conditions reduce to prescribing the
end displacements ``u(0) = 0`` and ``u(1) = eps_bar``. The FE solution gives

* the effective (homogenized) modulus ``E_eff = <sigma> / eps_bar``  (upscaling)
* the strain concentration factor of each phase ``A_i = <eps>_i / eps_bar``,
  which maps any macro strain back to phase strains/stresses (downscaling /
  localization).

For a series cell the numerical result reproduces the Reuss average; for a
parallel cell it reproduces the Voigt average. Both identities are checked in
the test suite and shown in the app's *Verification* tab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fem import solve_bar
from .materials import PHASES, Microstructure
from .validation import require_finite, require_int_at_least

MATRIX, REINFORCEMENT = 0, 1


def series_cell_mesh(volume_fraction: float, n_elements: int) -> tuple[np.ndarray, np.ndarray]:
    """Mesh of a symmetric layered cell ``matrix | reinforcement | matrix`` on [0, 1].

    Nodes are placed exactly on the phase interfaces, so the meshed volume
    fraction equals the requested one. Returns ``(nodes, element_phase)``.
    """
    n_elements = require_int_at_least("n_elements", n_elements, 3)
    f = float(volume_fraction)
    if f <= 0.0:
        return np.linspace(0.0, 1.0, n_elements + 1), np.full(n_elements, MATRIX)
    if f >= 1.0:
        return np.linspace(0.0, 1.0, n_elements + 1), np.full(n_elements, REINFORCEMENT)

    n_r = int(np.clip(round(f * n_elements), 1, n_elements - 2))
    n_left = (n_elements - n_r) // 2
    n_right = n_elements - n_r - n_left
    a, b = 0.5 * (1.0 - f), 0.5 * (1.0 + f)
    nodes = np.concatenate(
        [
            np.linspace(0.0, a, n_left + 1)[:-1],
            np.linspace(a, b, n_r + 1)[:-1],
            np.linspace(b, 1.0, n_right + 1),
        ]
    )
    phase = np.concatenate(
        [np.full(n_left, MATRIX), np.full(n_r, REINFORCEMENT), np.full(n_right, MATRIX)]
    )
    return nodes, phase


@dataclass(frozen=True)
class RVEResult:
    """Outcome of a unit-cell solve.

    ``x``, ``u``, ``element_phase``, ``strain`` and ``stress`` describe the
    resolved cell field and are only available for the ``series`` arrangement
    (in a parallel cell the field is trivially affine and identical in both
    phases); they are ``None`` otherwise.
    """

    microstructure: Microstructure
    E_eff: float
    applied_strain: float
    concentration: dict[str, float]
    x: np.ndarray | None = None
    u: np.ndarray | None = None
    element_phase: np.ndarray | None = None
    strain: np.ndarray | None = None
    stress: np.ndarray | None = None

    @property
    def fluctuation(self) -> np.ndarray | None:
        """Periodic displacement fluctuation ``u - eps_bar * x`` (series only)."""
        if self.x is None or self.u is None:
            return None
        return self.u - self.applied_strain * self.x

    def phase_strain(self, macro_strain) -> dict[str, np.ndarray]:
        """Localize a macro strain (scalar or array) to the average strain in each phase."""
        eps = np.asarray(macro_strain, dtype=float)
        return {p: self.concentration[p] * eps for p in PHASES}

    def phase_stress(self, macro_strain) -> dict[str, np.ndarray]:
        moduli = self.microstructure.phase_moduli
        return {p: moduli[p] * e for p, e in self.phase_strain(macro_strain).items()}


def solve_rve(
    micro: Microstructure, n_elements: int = 200, applied_strain: float = 1.0
) -> RVEResult:
    """Solve the unit-cell problem and return homogenized and localization data."""
    n_elements = require_int_at_least("n_elements", n_elements, 3)
    eps = require_finite("applied_strain", applied_strain)
    if eps == 0.0:
        raise ValueError("applied_strain must be non-zero")
    moduli = micro.phase_moduli

    if micro.arrangement == "series":
        x, phase = series_cell_mesh(micro.volume_fraction, n_elements)
        E = np.where(phase == REINFORCEMENT, micro.E_reinforcement, micro.E_matrix)
        sol = solve_bar(x, E, area=1.0, tip_displacement=eps)
        E_eff = sol.mean_stress() / eps
        h = sol.element_lengths
        concentration = {}
        for code, name in ((MATRIX, "matrix"), (REINFORCEMENT, "reinforcement")):
            mask = phase == code
            if mask.any():
                concentration[name] = float(np.sum(sol.strain[mask] * h[mask]) / np.sum(h[mask])) / eps
            else:  # dilute limit: an infinitesimal layer carries the same stress
                concentration[name] = E_eff / moduli[name]
        return RVEResult(
            microstructure=micro,
            E_eff=float(E_eff),
            applied_strain=eps,
            concentration=concentration,
            x=sol.x,
            u=sol.u,
            element_phase=phase,
            strain=sol.strain,
            stress=sol.stress,
        )

    # parallel: each phase spans the whole cell with area = its volume fraction;
    # the phases share the end nodes, so they see the same end displacements.
    mean_stress = 0.0
    concentration = {}
    x = np.linspace(0.0, 1.0, n_elements + 1)
    for name in PHASES:
        frac = micro.phase_fractions[name]
        sol = solve_bar(x, moduli[name], area=1.0, tip_displacement=eps)
        mean_stress += frac * sol.mean_stress()
        concentration[name] = float(np.mean(sol.strain)) / eps
    return RVEResult(
        microstructure=micro,
        E_eff=float(mean_stress / eps),
        applied_strain=eps,
        concentration=concentration,
    )
