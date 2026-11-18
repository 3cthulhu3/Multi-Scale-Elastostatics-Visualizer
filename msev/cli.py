"""Command-line interface: ``python -m msev --help``.

Inputs use engineering units (GPa, mm^2, kN, kN/m); everything is converted
to SI before solving.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, export, plotting
from .materials import ARRANGEMENTS, Microstructure
from .multiscale import (
    SMALL_STRAIN_LIMIT,
    BarProblem,
    scale_separation_study,
    solve_multiscale,
    solve_resolved,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="msev",
        description="Two-scale (RVE -> bar) elastostatics of a composite bar.",
    )
    p.add_argument("--version", action="version", version=f"msev {__version__}")
    g = p.add_argument_group("microstructure")
    g.add_argument("--e-matrix", type=float, default=3.5, help="matrix modulus [GPa] (3.5)")
    g.add_argument("--e-reinforcement", type=float, default=70.0,
                   help="reinforcement modulus [GPa] (70)")
    g.add_argument("--vf", type=float, default=0.4, help="reinforcement volume fraction (0.4)")
    g.add_argument("--arrangement", choices=ARRANGEMENTS, default="series")
    g = p.add_argument_group("macro problem")
    g.add_argument("--length", type=float, default=1.0, help="bar length [m] (1.0)")
    g.add_argument("--area", type=float, default=1000.0, help="cross-section [mm^2] (1000)")
    g.add_argument("--tip-load", type=float, default=10.0, help="tip load [kN] (10)")
    g.add_argument("--q", type=float, default=0.0, help="distributed axial load [kN/m] (0)")
    g.add_argument("--elements", type=int, default=50, help="macro elements (50)")
    g.add_argument("--cells", type=int, default=0,
                   help="also solve a fully resolved model with N cells (series only)")
    g = p.add_argument_group("output")
    g.add_argument("--out", type=Path, help="directory for CSV/JSON/HTML results")
    g.add_argument("--html", action="store_true", help="also write report.html (needs --out)")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        micro = Microstructure(
            E_matrix=args.e_matrix * 1e9,
            E_reinforcement=args.e_reinforcement * 1e9,
            volume_fraction=args.vf,
            arrangement=args.arrangement,
        )
        problem = BarProblem(
            length=args.length,
            area=args.area * 1e-6,
            tip_load=args.tip_load * 1e3,
            distributed_load=args.q * 1e3,
            n_elements=args.elements,
        )
        result = solve_multiscale(problem, micro)
        resolved = solve_resolved(problem, micro, args.cells) if args.cells > 0 else None
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    s = result.summary()
    print(f"E_eff      = {s['E_eff_Pa']/1e9:.4g} GPa  "
          f"(Reuss {s['E_reuss_Pa']/1e9:.4g}, Voigt {s['E_voigt_Pa']/1e9:.4g})")
    print(f"u_tip      = {s['tip_displacement_m']*1e3:.6g} mm")
    print(f"max |σ|    = {s['max_abs_macro_stress_Pa']/1e6:.6g} MPa (macro), "
          f"{s['max_abs_matrix_stress_Pa']/1e6:.6g} MPa (matrix), "
          f"{s['max_abs_reinforcement_stress_Pa']/1e6:.6g} MPa (reinforcement)")
    print(f"equilibrium residual = {s['equilibrium_residual']:.2e}")
    if s["max_abs_phase_strain"] > SMALL_STRAIN_LIMIT:
        print(f"warning: max phase strain {s['max_abs_phase_strain']:.2%} exceeds "
              f"{SMALL_STRAIN_LIMIT:.0%}; the linear small-strain model is questionable",
              file=sys.stderr)
    if resolved is not None:
        print(f"resolved u_tip = {resolved.tip_displacement*1e3:.6g} mm with {args.cells} cells")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "nodes.csv").write_text(export.nodes_csv(result.macro))
        (args.out / "elements.csv").write_text(export.elements_csv(result))
        (args.out / "summary.json").write_text(export.summary_json(result))
        if args.html:
            figs = [
                ("Macro response", plotting.macro_response_figure(result, resolved)),
                ("Unit cell", plotting.rve_figure(result.rve)),
                ("Bounds", plotting.bounds_figure(micro, result.E_eff)),
            ]
            if micro.arrangement == "series":
                study = scale_separation_study(problem, micro)
                figs.append(("Scale separation", plotting.convergence_figure(study)))
            (args.out / "report.html").write_text(plotting.to_html(figs, "msev report"))
        print(f"results written to {args.out}/")
    elif args.html:
        print("note: --html requires --out", file=sys.stderr)
    return 0


def main() -> None:  # console-script entry point
    raise SystemExit(run())


if __name__ == "__main__":
    main()
