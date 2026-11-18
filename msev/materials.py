"""Two-phase microstructure description and closed-form effective moduli.

Conventions
-----------
* SI units throughout (Pa, m, N).
* ``volume_fraction`` is the volume fraction of the *reinforcement* phase.
* ``arrangement`` describes how the phases sit relative to the load axis:

  - ``"parallel"``: continuous phases aligned with the bar axis (e.g. long
    fibres). Both phases see the same strain (iso-strain) -> Voigt average.
  - ``"series"``: layers stacked along the bar axis (laminate loaded through
    its thickness). Both phases carry the same stress (iso-stress) -> Reuss
    average.

In 1D these two averages are exact for their arrangement and they bound the
effective modulus of *any* two-phase microstructure with the same volume
fraction (Reuss <= E_eff <= Voigt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import require_choice, require_fraction, require_positive

ARRANGEMENTS: tuple[str, ...] = ("series", "parallel")
PHASES: tuple[str, ...] = ("matrix", "reinforcement")


def voigt_modulus(E_matrix, E_reinforcement, volume_fraction):
    """Iso-strain (rule of mixtures) modulus. Vectorised over ``volume_fraction``."""
    f = np.asarray(volume_fraction, dtype=float)
    return f * E_reinforcement + (1.0 - f) * E_matrix


def reuss_modulus(E_matrix, E_reinforcement, volume_fraction):
    """Iso-stress (inverse rule of mixtures) modulus. Vectorised over ``volume_fraction``."""
    f = np.asarray(volume_fraction, dtype=float)
    return 1.0 / (f / E_reinforcement + (1.0 - f) / E_matrix)


@dataclass(frozen=True)
class Microstructure:
    """A periodic two-phase microstructure (the micro scale)."""

    E_matrix: float
    E_reinforcement: float
    volume_fraction: float
    arrangement: str = "series"

    def __post_init__(self) -> None:
        # frozen dataclass: use object.__setattr__ to store normalised floats
        object.__setattr__(self, "E_matrix", require_positive("E_matrix", self.E_matrix))
        object.__setattr__(
            self, "E_reinforcement", require_positive("E_reinforcement", self.E_reinforcement)
        )
        object.__setattr__(
            self, "volume_fraction", require_fraction("volume_fraction", self.volume_fraction)
        )
        require_choice("arrangement", self.arrangement, ARRANGEMENTS)

    @property
    def phase_moduli(self) -> dict[str, float]:
        return {"matrix": self.E_matrix, "reinforcement": self.E_reinforcement}

    @property
    def phase_fractions(self) -> dict[str, float]:
        return {"matrix": 1.0 - self.volume_fraction, "reinforcement": self.volume_fraction}

    @property
    def voigt(self) -> float:
        return float(voigt_modulus(self.E_matrix, self.E_reinforcement, self.volume_fraction))

    @property
    def reuss(self) -> float:
        return float(reuss_modulus(self.E_matrix, self.E_reinforcement, self.volume_fraction))

    @property
    def analytical_modulus(self) -> float:
        """Exact 1D effective modulus for this arrangement."""
        return self.voigt if self.arrangement == "parallel" else self.reuss
