"""OpenFOAM v2306 nondimensional case writer for M1 (Newtonian & BirdCarreau) in star units.

Equations & scaling follow knowledge/11-math-model.md §2.2 & §3 and plan/CORPUS_PLAN.md §1:
- Length scale: h1* = 1
- Velocity scale: U_mean* = 1
- Density: rho* = 1
- Newtonian kinematic viscosity: nu* = 1 / Re
- BirdCarreau kinematic viscosity: nu0* = beta / Re, nuInf* = 1 / Re, k* = Cu, n, a
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cfd.gen.geometry import MeshSpec, build_geometry, format_block_mesh_dict


def _fmt(val: float) -> str:
    """Format a float cleanly for OpenFOAM ASCII dictionaries."""
    if abs(val) < 1e-15:
        return "0"
    return f"{val:.10g}"


def _format_u_field(mesh: MeshSpec, rheology: str) -> str:
    """Generate 0/U in star units (parabolic inlet for Newtonian, uniform (1 0 0) otherwise)."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volVectorField;",
        "    object      U;",
        "}",
        "",
        "dimensions      [0 1 -1 0 0 0 0];",
        "",
        "internalField   uniform (1 0 0);",
        "",
        "boundaryField",
        "{",
    ]

    for patch in mesh.patches:
        if patch.name == "inlet":
            if rheology == "Newtonian":
                if mesh.family == "KG":
                    # Fully developed Hagen-Poiseuille profile on r in [0, 0.5] (d1* = 1):
                    # u*(r*) = 2 * (1 - 4 r*^2), area-averaged mean = 1.
                    expr = (
                        "vector(2.0 * max(0.0, 1.0 - 4.0 * "
                        "(pos().y()*pos().y() + pos().z()*pos().z())), 0, 0)"
                    )
                else:
                    # Fully developed plane Poiseuille profile on y in [0, 1]: u = 6 y (1 - y), mean = 1.
                    expr = "vector(6.0 * pos().y() * (1.0 - pos().y()), 0, 0)"
                lines.extend(
                    [
                        "    inlet",
                        "    {",
                        "        type            exprFixedValue;",
                        "        value           uniform (1 0 0);",
                        f'        valueExpr       "{expr}";',
                        "    }",
                    ]
                )
            else:
                lines.extend(
                    [
                        "    inlet",
                        "    {",
                        "        type            fixedValue;",
                        "        value           uniform (1 0 0);",
                        "    }",
                    ]
                )
        elif patch.name.startswith("outlet"):
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            zeroGradient;",
                    "    }",
                ]
            )
        elif patch.patch_type == "wall":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            noSlip;",
                    "    }",
                ]
            )
        elif patch.patch_type == "empty":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            empty;",
                    "    }",
                ]
            )
        elif patch.patch_type == "wedge":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            wedge;",
                    "    }",
                ]
            )
        else:
            raise ValueError(f"Unhandled patch '{patch.name}' (type='{patch.patch_type}')")

    lines.extend(["}", ""])
    return "\n".join(lines)


def _format_p_field(mesh: MeshSpec) -> str:
    """Generate 0/p in star units (fixedValue 0 at outlet(s), zeroGradient elsewhere)."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volScalarField;",
        "    object      p;",
        "}",
        "",
        "dimensions      [0 2 -2 0 0 0 0];",
        "",
        "internalField   uniform 0;",
        "",
        "boundaryField",
        "{",
    ]

    for patch in mesh.patches:
        if patch.name.startswith("outlet"):
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            fixedValue;",
                    "        value           uniform 0;",
                    "    }",
                ]
            )
        elif patch.patch_type == "empty":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            empty;",
                    "    }",
                ]
            )
        elif patch.patch_type == "wedge":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            wedge;",
                    "    }",
                ]
            )
        else:
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            zeroGradient;",
                    "    }",
                ]
            )

    lines.extend(["}", ""])
    return "\n".join(lines)


def _format_transport_properties(params: Mapping[str, Any]) -> str:
    """Generate constant/transportProperties in star units (11-math-model.md §2.2)."""
    rheology = str(params.get("rheology", "Newtonian"))
    Re = float(params["Re"])
    if Re <= 0.0:
        raise ValueError(f"Re must be > 0, got {Re}")

    nu_inf = 1.0 / Re
    # Paired-design V1 only: Newtonian viscosity of the whole suspension relative to the
    # carrier, mu_eff(phi)/mu_carrier. With Re built on the carrier viscosity, nu* = ratio/Re
    # keeps one star scale across the four variants of a design (plan/CORPUS_PLAN.md §6.3).
    viscosity_ratio = float(params.get("viscosity_ratio", 1.0))
    if viscosity_ratio <= 0.0:
        raise ValueError(f"viscosity_ratio must be > 0, got {viscosity_ratio}")

    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      transportProperties;",
        "}",
        "",
    ]

    if rheology == "Newtonian":
        lines.extend(
            [
                "transportModel  Newtonian;",
                f"nu              nu [0 2 -1 0 0 0 0] {_fmt(viscosity_ratio * nu_inf)};",
                "",
            ]
        )
    elif rheology == "BirdCarreau":
        beta = float(params["beta"])
        Cu = float(params["Cu"])
        n = float(params["n"])
        a = float(params.get("a", 2.0))
        if beta < 1.0:
            raise ValueError(f"BirdCarreau beta = mu0/muInf must be >= 1.0, got {beta}")
        if Cu < 0.0:
            raise ValueError(f"BirdCarreau Cu must be >= 0, got {Cu}")
        if not (0.0 < n <= 1.0):
            raise ValueError(f"BirdCarreau n must be in (0, 1], got {n}")
        if a <= 0.0:
            raise ValueError(f"BirdCarreau a must be > 0, got {a}")

        nu0 = viscosity_ratio * beta / Re
        nu_inf_bc = viscosity_ratio * nu_inf
        lines.extend(
            [
                "transportModel  BirdCarreau;",
                "BirdCarreauCoeffs",
                "{",
                f"    nu0     nu0 [0 2 -1 0 0 0 0] {_fmt(nu0)};",
                f"    nuInf   nuInf [0 2 -1 0 0 0 0] {_fmt(nu_inf_bc)};",
                f"    k       k [0 0 1 0 0 0 0] {_fmt(Cu)};",
                f"    n       n [0 0 0 0 0 0 0] {_fmt(n)};",
                f"    a       a [0 0 0 0 0 0 0] {_fmt(a)};",
                "}",
                "",
            ]
        )
    else:
        raise ValueError(f"Unsupported rheology '{rheology}'; expected 'Newtonian' or 'BirdCarreau'")

    return "\n".join(lines)


def _format_turbulence_properties() -> str:
    """Generate constant/turbulenceProperties for laminar flow."""
    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      turbulenceProperties;",
            "}",
            "",
            "simulationType  laminar;",
            "",
        ]
    )


def _format_fv_schemes() -> str:
    """Generate system/fvSchemes aligned with validated cfd/openfoam/carreau2d (plain Gauss linear)."""
    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      fvSchemes;",
            "}",
            "",
            "ddtSchemes",
            "{",
            "    default         steadyState;",
            "}",
            "",
            "gradSchemes",
            "{",
            "    default         Gauss linear;",
            "}",
            "",
            "divSchemes",
            "{",
            "    default                         none;",
            "    div(phi,U)                      bounded Gauss linearUpwind grad(U);",
            "    div((nuEff*dev2(T(grad(U)))))   Gauss linear;",
            "}",
            "",
            "laplacianSchemes",
            "{",
            "    default         Gauss linear corrected;",
            "}",
            "",
            "interpolationSchemes",
            "{",
            "    default         linear;",
            "}",
            "",
            "snGradSchemes",
            "{",
            "    default         corrected;",
            "}",
            "",
        ]
    )


def _format_fv_solution(
    residual_tol: float = 1e-5,
    n_non_orthogonal_correctors: int = 0,
    relax_p: float = 0.3,
    relax_u: float = 0.7,
) -> str:
    """Generate system/fvSolution aligned with carreau2d (residualControl 1e-5, nNonOrtho 0).

    ``relax_p`` / ``relax_u`` default to the corpus values; a design record may override them
    (T7 v2.3 M1 retry uses p 0.2 / U 0.5 for the runs that did not converge).
    """
    tol_str = f"{residual_tol:.1e}"
    if not (0.0 < relax_p <= 1.0):
        raise ValueError(f"relax_p must be in (0, 1], got {relax_p}")
    if not (0.0 < relax_u <= 1.0):
        raise ValueError(f"relax_u must be in (0, 1], got {relax_u}")
    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      fvSolution;",
            "}",
            "",
            "solvers",
            "{",
            "    p",
            "    {",
            "        solver          GAMG;",
            "        smoother        GaussSeidel;",
            "        tolerance       1e-7;",
            "        relTol          0.01;",
            "    }",
            "    pFinal",
            "    {",
            "        $p;",
            "        relTol          0;",
            "    }",
            "    U",
            "    {",
            "        solver          smoothSolver;",
            "        smoother        GaussSeidel;",
            "        tolerance       1e-8;",
            "        relTol          0.1;",
            "    }",
            "    UFinal",
            "    {",
            "        $U;",
            "        tolerance       1e-9;",
            "        relTol          0;",
            "    }",
            "}",
            "",
            "SIMPLE",
            "{",
            f"    nNonOrthogonalCorrectors {n_non_orthogonal_correctors};",
            "    consistent      yes;",
            "    residualControl",
            "    {",
            f"        p               {tol_str};",
            f"        U               {tol_str};",
            "    }",
            "}",
            "",
            "relaxationFactors",
            "{",
            "    fields",
            "    {",
            f"        p               {_fmt(relax_p)};",
            "    }",
            "    equations",
            "    {",
            f"        U               {_fmt(relax_u)};",
            "    }",
            "}",
            "",
        ]
    )


def _outlet_centreline_probes(mesh: MeshSpec) -> list[tuple[float, float, float]]:
    """Return outlet-centreline probe coordinate(s) strictly inside the fluid for QoI monitoring."""
    sl_by_name = {sl.name: sl for sl in mesh.sample_lines}
    if mesh.family == "KG":
        # Axisymmetric wedge: outlet centreline is along the axis-adjacent start of outletProfile.
        return [sl_by_name["outletProfile"].start]
    if mesh.family in ("G4", "G5", "G7"):
        pts: list[tuple[float, float, float]] = []
        for name in ("daughter1Mid", "daughter2Mid"):
            sl = sl_by_name[name]
            pts.append(
                (
                    0.5 * (sl.start[0] + sl.end[0]),
                    0.5 * (sl.start[1] + sl.end[1]),
                    0.5 * (sl.start[2] + sl.end[2]),
                )
            )
        return pts
    sl = sl_by_name["outletProfile"]
    return [
        (
            0.5 * (sl.start[0] + sl.end[0]),
            0.5 * (sl.start[1] + sl.end[1]),
            0.5 * (sl.start[2] + sl.end[2]),
        )
    ]


def _format_control_dict(mesh: MeshSpec, end_time: int = 5000, write_interval: int = 1000) -> str:
    """Generate system/controlDict with QoI monitors (every iteration) and onEnd field/set outputs."""
    probe_pts = _outlet_centreline_probes(mesh)
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      controlDict;",
        "}",
        "",
        "application     simpleFoam;",
        "startFrom       startTime;",
        "startTime       0;",
        "stopAt          endTime;",
        f"endTime         {end_time};",
        "deltaT          1;",
        "writeControl    timeStep;",
        f"writeInterval   {write_interval};",
        "purgeWrite      2;",
        "writeFormat     ascii;",
        "writePrecision  8;",
        "writeCompression off;",
        "timeFormat      general;",
        "timePrecision   6;",
        "runTimeModifiable true;",
        "",
        "functions",
        "{",
        "    inletPressure",
        "    {",
        "        type            surfaceFieldValue;",
        "        libs            (fieldFunctionObjects);",
        "        executeControl  timeStep;",
        "        executeInterval 1;",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "        writeFields     false;",
        "        log             false;",
        "        regionType      patch;",
        "        name            inlet;",
        "        operation       areaAverage;",
        "        fields          (p);",
        "    }",
        "",
        "    qoiProbes",
        "    {",
        "        type            probes;",
        '        libs            ("libsampling.so");',
        "        executeControl  timeStep;",
        "        executeInterval 1;",
        "        writeControl    timeStep;",
        "        writeInterval   1;",
        "        fields          (U);",
        "        probeLocations",
        "        (",
    ]

    for px, py, pz in probe_pts:
        lines.append(f"            ({_fmt(px)} {_fmt(py)} {_fmt(pz)})")

    lines.extend(
        [
            "        );",
            "    }",
            "",
            "    writeCellCentres",
            "    {",
            "        type            writeCellCentres;",
            "        libs            (fieldFunctionObjects);",
            "        writeControl    onEnd;",
            "    }",
            "",
            "    writeCellVolumes",
            "    {",
            "        type            writeCellVolumes;",
            "        libs            (fieldFunctionObjects);",
            "        writeControl    onEnd;",
            "    }",
            "",
            "    gradU",
            "    {",
            "        type            grad;",
            "        libs            (fieldFunctionObjects);",
            "        field           U;",
            "        writeControl    onEnd;",
            "    }",
            "",
            "    sample",
            "    {",
            "        type                sets;",
            '        libs                ("libsampling.so");',
            "        writeControl        onEnd;",
            "        interpolationScheme cellPoint;",
            "        setFormat           raw;",
            "        fields              (U p);",
            "        sets",
            "        {",
        ]
    )

    for sl in mesh.sample_lines:
        sx, sy, sz = sl.start
        ex, ey, ez = sl.end
        lines.extend(
            [
                f"            {sl.name}",
                "            {",
                "                type    uniform;",
                f"                axis    {sl.axis};",
                f"                start   ({_fmt(sx)} {_fmt(sy)} {_fmt(sz)});",
                f"                end     ({_fmt(ex)} {_fmt(ey)} {_fmt(ez)});",
                f"                nPoints {sl.n_points};",
                "            }",
            ]
        )

    lines.extend(["        }", "    }", "}", ""])
    return "\n".join(lines)


def _read_run_case_template() -> str:
    """Load the companion cfd/gen/run_case.sh runner script."""
    script_path = Path(__file__).resolve().parent / "run_case.sh"
    return script_path.read_text(encoding="utf-8")


def write_case(
    case_dir: Path | str,
    params: Mapping[str, Any],
    level: int = 1,
    end_time: int = 5000,
    residual_tol: float = 1e-5,
) -> MeshSpec:
    """Write a complete nondimensional OpenFOAM v2306 case directory.

    Parameters
    ----------
    case_dir : Path | str
        Destination directory (created if it does not exist).
    params : Mapping[str, Any]
        Must contain:
        - ``family``: one of 'G2'..'G7' or 'KG'
        - ``rheology``: 'Newtonian' or 'BirdCarreau'
        - ``Re``: Reynolds number (> 0)
        - For BirdCarreau: ``beta``, ``Cu``, ``n`` (and optional ``a``, default 2.0)
        - Geometry ratios either nested under ``params["geometry"]`` or at top level.
    level : int in {0, 1, 2}
        Mesh refinement level (r = 1.5 per level).
    end_time : int
        Maximum SIMPLE iterations before stopping (early exit via residualControl).
    residual_tol : float
        Convergence residual threshold for p and U (default 1e-5).

    Returns
    -------
    MeshSpec
        The generated mesh specification.
    """
    family = str(params["family"]).upper()
    rheology = str(params.get("rheology", "Newtonian"))
    viscosity_ratio = float(params.get("viscosity_ratio", 1.0))

    geom_params: dict[str, Any] = dict(params.get("geometry", params))
    # Enforce entry length >= 5h and outlet length >= 10 h_out (or L_out >= 10 for KG; L_ratio for G1).
    if family != "G1" and float(geom_params.get("L_in", 5.0)) < 5.0:
        raise ValueError("Entry length L_in must be >= 5.0 in star units")
    if family not in ("KG", "G1") and float(geom_params.get("L_out_ratio", 10.0)) < 10.0:
        raise ValueError("Outlet length ratio L_out_ratio must be >= 10.0")

    if family == "G1":
        from cfd.gen.m2b import build_g1

        mesh = build_g1(geom_params, level=level)
    else:
        mesh = build_geometry(family, geom_params, level=level)

    root = Path(case_dir)
    for sub in ("0", "constant", "system"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / "system" / "blockMeshDict").write_text(format_block_mesh_dict(mesh), encoding="utf-8")
    (root / "system" / "controlDict").write_text(
        _format_control_dict(mesh, end_time=end_time), encoding="utf-8"
    )
    (root / "system" / "fvSchemes").write_text(_format_fv_schemes(), encoding="utf-8")
    (root / "system" / "fvSolution").write_text(
        _format_fv_solution(
            residual_tol=residual_tol,
            relax_p=float(params.get("relax_p", 0.3)),
            relax_u=float(params.get("relax_u", 0.7)),
        ),
        encoding="utf-8",
    )
    (root / "0" / "U").write_text(_format_u_field(mesh, rheology=rheology), encoding="utf-8")
    (root / "0" / "p").write_text(_format_p_field(mesh), encoding="utf-8")
    (root / "constant" / "transportProperties").write_text(
        _format_transport_properties(params), encoding="utf-8"
    )
    (root / "constant" / "turbulenceProperties").write_text(
        _format_turbulence_properties(), encoding="utf-8"
    )

    run_script_dst = root / "run_case.sh"
    run_script_dst.write_text(_read_run_case_template(), encoding="utf-8")
    os.chmod(
        run_script_dst,
        run_script_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    # Write machine-readable case metadata for downstream post-processing / manifests.
    total_cells = sum(nx * ny * nz for _, (nx, ny, nz), _ in mesh.blocks)
    meta = {
        "run_id": params.get("run_id"),
        "family": family,
        "level": level,
        "rheology": rheology,
        "Re": float(params["Re"]),
        "beta": float(params["beta"]) if rheology == "BirdCarreau" else None,
        "Cu": float(params["Cu"]) if rheology == "BirdCarreau" else None,
        "n": float(params["n"]) if rheology == "BirdCarreau" else None,
        "a": float(params.get("a", 2.0)) if rheology == "BirdCarreau" else None,
        "geometry": mesh.params,
        "n_blocks": len(mesh.blocks),
        "n_cells": total_cells,
        "inlet_height": mesh.inlet_height,
        "outlet_heights": list(mesh.outlet_heights),
        "entry_length": mesh.entry_length,
        "outlet_lengths": list(mesh.outlet_lengths),
        "sample_lines": [sl.name for sl in mesh.sample_lines],
        "viscosity_ratio": viscosity_ratio,
        "relax_p": float(params.get("relax_p", 0.3)),
        "relax_u": float(params.get("relax_u", 0.7)),
    }
    (root / "params.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return mesh
