"""Small input-validation helpers.

Every public entry point validates its inputs here so that a bad value fails
fast with a message that names the offending parameter, instead of surfacing
later as a singular matrix, a NaN or a silently wrong plot.
"""

from __future__ import annotations

import math
from numbers import Integral, Real


def require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def require_positive(name: str, value: float) -> float:
    value = require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def require_fraction(name: str, value: float) -> float:
    value = require_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return value


def require_int_at_least(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def require_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value
