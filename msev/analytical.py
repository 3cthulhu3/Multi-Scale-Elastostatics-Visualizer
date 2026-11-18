"""Closed-form solution of a homogeneous clamped bar (used for verification).

    u(x)  = [ (P + qL) x - q x^2 / 2 ] / (E A)
    N(x)  = P + q (L - x)
    sigma = N / A
"""

from __future__ import annotations

import numpy as np


def bar_axial_force(x, length: float, tip_load: float, distributed_load: float = 0.0):
    x = np.asarray(x, dtype=float)
    return tip_load + distributed_load * (length - x)


def bar_displacement(
    x, modulus: float, area: float, length: float, tip_load: float, distributed_load: float = 0.0
):
    x = np.asarray(x, dtype=float)
    q = distributed_load
    return ((tip_load + q * length) * x - 0.5 * q * x**2) / (modulus * area)


def bar_stress(x, area: float, length: float, tip_load: float, distributed_load: float = 0.0):
    return bar_axial_force(x, length, tip_load, distributed_load) / area
