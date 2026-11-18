"""1D linear finite elements for an axially loaded elastic bar.

Governing equation (small strain, linear elasticity)::

    d/dx ( E(x) A du/dx ) + q = 0     on (0, L)
    u(0) = 0                          (clamped left end)
    E A du/dx (L) = P                 (tip load)   -- or --   u(L) = u_L

Two-node linear elements, consistent load vector, sparse assembly. Dirichlet
conditions are imposed by eliminating the constrained DOFs, which keeps the
reduced system symmetric positive definite.

This module performs **no file I/O**; results are returned as arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .validation import require_finite, require_int_at_least, require_positive


@dataclass(frozen=True)
class BarSolution:
    """Nodal and element results of a 1D bar solve.

    ``x`` and ``u`` are nodal arrays (length ``n + 1``). ``modulus``, ``strain``,
    ``stress`` and ``axial_force`` are element arrays (length ``n``); linear
    elements give a constant strain per element.
    """

    x: np.ndarray
    u: np.ndarray
    modulus: np.ndarray
    area: float
    strain: np.ndarray
    stress: np.ndarray
    axial_force: np.ndarray
    reaction_left: float
    reaction_right: float

    @property
    def n_elements(self) -> int:
        return int(self.strain.size)

    @property
    def length(self) -> float:
        return float(self.x[-1] - self.x[0])

    @property
    def midpoints(self) -> np.ndarray:
        return 0.5 * (self.x[:-1] + self.x[1:])

    @property
    def element_lengths(self) -> np.ndarray:
        return np.diff(self.x)

    @property
    def tip_displacement(self) -> float:
        return float(self.u[-1])

    @property
    def max_abs_displacement(self) -> float:
        return float(np.max(np.abs(self.u)))

    @property
    def max_abs_stress(self) -> float:
        return float(np.max(np.abs(self.stress)))

    def mean_stress(self) -> float:
        """Volume average of the stress over the bar."""
        h = self.element_lengths
        return float(np.sum(self.stress * h) / np.sum(h))


def uniform_mesh(length: float, n_elements: int) -> np.ndarray:
    length = require_positive("length", length)
    n_elements = require_int_at_least("n_elements", n_elements, 1)
    return np.linspace(0.0, length, n_elements + 1)


def solve_bar(
    x: np.ndarray,
    modulus,
    area: float,
    tip_load: float = 0.0,
    distributed_load: float = 0.0,
    *,
    tip_displacement: float | None = None,
) -> BarSolution:
    """Solve the clamped bar problem on the mesh ``x``.

    Parameters
    ----------
    x:
        Strictly increasing node coordinates (m), ``x[0]`` is clamped.
    modulus:
        Young's modulus (Pa); a scalar or one value per element.
    area:
        Cross-section area (m^2).
    tip_load:
        Axial point load at the right end (N). Ignored if ``tip_displacement``
        is given (the right end is then constrained).
    distributed_load:
        Uniform axial load per unit length q (N/m).
    tip_displacement:
        If given, the right end displacement is prescribed (m).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D array with at least two nodes")
    h = np.diff(x)
    if not np.all(np.isfinite(x)) or np.any(h <= 0.0):
        raise ValueError("x must be finite and strictly increasing")
    n_el = h.size
    n_nodes = n_el + 1

    E = np.broadcast_to(np.asarray(modulus, dtype=float), (n_el,)).copy()
    if not np.all(np.isfinite(E)) or np.any(E <= 0.0):
        raise ValueError("modulus must be finite and > 0 in every element")
    area = require_positive("area", area)
    tip_load = require_finite("tip_load", tip_load)
    q = require_finite("distributed_load", distributed_load)

    # --- assembly (COO -> CSR); duplicates are summed on conversion --------
    k = E * area / h
    i = np.arange(n_el)
    rows = np.concatenate([i, i, i + 1, i + 1])
    cols = np.concatenate([i, i + 1, i, i + 1])
    vals = np.concatenate([k, -k, -k, k])
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    F = np.zeros(n_nodes)
    F[:-1] += 0.5 * q * h  # consistent load vector of a uniform q
    F[1:] += 0.5 * q * h

    # --- boundary conditions by elimination --------------------------------
    u = np.zeros(n_nodes)
    fixed = [0]
    if tip_displacement is None:
        F[-1] += tip_load
    else:
        u[-1] = require_finite("tip_displacement", tip_displacement)
        fixed.append(n_nodes - 1)
    free = np.setdiff1d(np.arange(n_nodes), fixed)

    if free.size:
        K_ff = K[free][:, free].tocsc()
        rhs = F[free] - K[free] @ u
        u[free] = np.atleast_1d(spla.spsolve(K_ff, rhs))
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("linear solve produced non-finite displacements")

    reactions = K @ u - F
    strain = np.diff(u) / h
    stress = E * strain
    return BarSolution(
        x=x,
        u=u,
        modulus=E,
        area=area,
        strain=strain,
        stress=stress,
        axial_force=stress * area,
        reaction_left=float(reactions[0]),
        reaction_right=float(reactions[-1]) if tip_displacement is not None else 0.0,
    )
