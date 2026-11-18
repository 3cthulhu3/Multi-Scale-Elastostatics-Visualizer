"""In-memory exports (CSV / JSON strings). Nothing is written to disk here."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict

from .fem import BarSolution
from .multiscale import MultiscaleResult


def _csv(header: list[str], columns: list) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in zip(*columns, strict=True):
        writer.writerow([f"{float(v):.10g}" for v in row])
    return buf.getvalue()


def nodes_csv(solution: BarSolution) -> str:
    """Nodal table: x [m], u [m]."""
    return _csv(["x_m", "u_m"], [solution.x, solution.u])


def elements_csv(result: MultiscaleResult) -> str:
    """Element table of the homogenized macro model incl. localized phase fields."""
    m = result.macro
    header = [
        "x_mid_m", "strain", "stress_Pa", "axial_force_N",
        "matrix_strain", "matrix_stress_Pa", "reinforcement_strain", "reinforcement_stress_Pa",
    ]
    cols = [
        m.midpoints, m.strain, m.stress, m.axial_force,
        result.phase_strain["matrix"], result.phase_stress["matrix"],
        result.phase_strain["reinforcement"], result.phase_stress["reinforcement"],
    ]
    return _csv(header, cols)


def summary_json(result: MultiscaleResult) -> str:
    """Inputs and key outputs, for reproducibility."""
    payload = {
        "problem": asdict(result.problem),
        "microstructure": asdict(result.microstructure),
        "results": result.summary(),
        "strain_concentration": result.rve.concentration,
    }
    return json.dumps(payload, indent=2)
