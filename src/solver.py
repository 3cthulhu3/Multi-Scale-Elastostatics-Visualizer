"""Deprecated compatibility shim. Use ``msev.legacy`` (or ``msev.solve_multiscale``)."""

import os
import sys
import warnings

# allow `python src/solver.py` from a checkout without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msev.legacy import solve_elastostatics  # noqa: E402

warnings.warn(
    "src.solver is deprecated; use msev.legacy.solve_elastostatics or msev.solve_multiscale",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["solve_elastostatics"]

if __name__ == "__main__":
    import numpy as np

    results = solve_elastostatics()
    print("Solver done! Files in output/")
    print(f"E_eff = {results['E_eff']/1e9:.1f} GPa")
    print(f"Max displacement: {np.max(np.abs(results['u']))*1e6:.2f} μm")
