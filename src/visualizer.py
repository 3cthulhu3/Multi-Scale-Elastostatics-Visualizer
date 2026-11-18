"""Deprecated compatibility shim. Use ``msev.legacy`` / ``msev.plotting``."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msev.legacy import plot_multiscale  # noqa: E402

warnings.warn(
    "src.visualizer is deprecated; use msev.legacy.plot_multiscale or msev.plotting",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["plot_multiscale"]

if __name__ == "__main__":
    plot_multiscale(html_path="docs/index.html", show=True)
    print("Interactive plot in browser + docs/index.html saved!")
