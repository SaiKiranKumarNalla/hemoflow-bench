"""OpenFOAM ESI v2306 ``multiphaseEulerFoam`` case writer (M2b) in star units.

Generates two-fluid Euler–Euler cases (phases ``RBC`` + ``plasma``) for families:
- ``G1``: 2D planar straight microchannel (built locally; ``L_ratio = L / h1``)
- ``G2``: 2D planar symmetric sudden expansion (reuses ``cfd.gen.geometry.build_g2``)
- ``KG``: axisymmetric 2D wedge sudden expansion (reuses ``cfd.gen.geometry.build_kg``)

Nondimensionalization (star units, plasma-referenced)
-----------------------------------------------------
Scales: length ``h1`` (or inlet diameter ``d1`` for ``KG``), velocity ``U = U_mean``,
density ``rho_p`` (pure plasma), dynamic viscosity ``mu_p`` (pure plasma), time ``h1 / U``.
With these scales the governing Euler–Euler parameters are:
- ``Re``        = ``rho_p * U * h1 / mu_p``  (plasma-based Reynolds number)
- ``rho_ratio`` = ``rho_RBC / rho_p``        (density ratio; Wu 2015: 1093/1027 = 1.0643;
                                              2023 COMSOL: 1098/1038 = 1.0578)
- ``mu_ratio``  = ``mu_RBC_eff / mu_p``      (effective RBC-to-plasma dynamic viscosity ratio)
- ``d_p_star``  = ``d_p / h1``               (dimensionless RBC diameter; d_p = 8 um)
- ``phi``       = inlet RBC volume fraction  (``alpha.RBC = phi``, ``alpha.plasma = 1 - phi``)

Inside OpenFOAM ESI v2306 ``multiphaseEulerFoam`` (incompressible two-fluid solver,
verified against ESI OpenFOAM v2212–v2312 ``multiphaseEulerFoam.C``, ``phaseModel.C``,
``multiphaseSystem.C``, and ``dragModels/*.C``):
- ``plasma``: ``rho = 1``, ``nu = 1 / Re``
- ``RBC``:    ``rho = rho_ratio``, ``nu = mu_ratio / (rho_ratio * Re)``, ``d = d_p_star``
- ``p_rgh`` has dynamic pressure units ``p_OF = p / (rho_p * U^2)``, so viscous-scaled
  star pressure is ``p* = p * h1 / (mu_p * U) = Re * p_OF``. With ``g = (0 0 0)``,
  ``p == p_rgh`` identically.

Stock OpenFOAM ESI v2306 closure mapping vs. Wu et al. (2015) (``knowledge/14-wu2015-closures.md``)
---------------------------------------------------------------------------------------------------
1. **Drag law (Wu 2015 Eq. 10: Stokes drag x Rusche–Issa hindrance ``exp(2.68 phi) + phi^0.43``):**
   No stock Rusche–Issa hindrance exists in ESI v2306 ``multiphaseEulerFoam`` (stock drag models are
   ``SchillerNaumann``, ``WenYu``, ``Gibilaro``, ``GidaspowErgunWenYu``, ``GidaspowSchillerNaumann``,
   ``Ergun``, ``SyamlalOBrien``, ``blended``).
   - **Chosen stock model:** ``WenYu`` on pair ``(RBC plasma)`` (dispersed ``phase1 = RBC``,
     continuous ``phase2 = plasma``). At microfluidic particle Reynolds numbers ``Re_p << 1``,
     ``WenYu`` recovers exact Stokes drag ``(18 mu_p / d_p^2)`` multiplied by the voidage hindrance
     ``(1 - phi)^(-2.65)`` (times ``phi * (1 - phi)`` in ``multiphaseSystem::dragCoeffs``), giving
     smooth hindered drag across both dilute (``phi = 0.01``) and dense (``phi = 0.20..0.45``)
     hematocrits without the ``phi = 0.20`` discontinuity of ``GidaspowErgunWenYu``.
   - Optional alternatives supported via ``drag_model``: ``"SchillerNaumann"``, ``"Gibilaro"``,
     ``"GidaspowErgunWenYu"``.
2. **RBC-phase viscosity / stress (Wu 2015 Eqs. 6–9c: Yeleswarapu shear-thinning ``mu_rbc(phi, gdot)``):**
   No stock shear-rate- or ``phi``-dependent viscosity exists in ESI v2306 ``multiphaseEulerFoam``
   (``phaseModel`` reads a constant scalar ``nu`` per phase from ``constant/transportProperties``).
   - **Documented fallback:** Constant effective RBC kinematic viscosity
     ``nu_RBC* = mu_ratio / (rho_ratio * Re)`` per case, where ``mu_ratio`` is computed from Wu 2015
     Eqs. (7, 9a–9c) at the case's inlet ``phi`` and representative channel shear rate (or passed
     explicitly). Because Wu 2015 Eq. (6) omits the factor of 2 before ``mu_rbc phi D_rbc`` whereas
     OpenFOAM's ``UEqns.H`` uses ``2 alpha nu D``, ``wu2015_mu_ratio`` documents this factor clearly.
3. **Lift force (Wu 2015 Eq. 10: Saffman shear-lift):**
   No stock lift model is active in ESI v2306 ``multiphaseEulerFoam`` (``interfacialModels`` only
   contains ``dragModels`` and ``heatTransferModels``; ``fluid.lift(phase)`` is commented out in
   ``UEqns.H``).
   - **Documented fallback:** Lift omitted in stock ``multiphaseEulerFoam`` (flagged in
     ``plan/REVIEW_QUEUE.md``; requires ``reactingTwoPhaseEulerFoam`` or a custom solver to add
     Saffman/Tomiyama lift).
4. **Diffusion / virtual mass / surface tension / packing limit:**
   Wu 2015 includes no diffusion/dispersion, no virtual mass, and no surface tension; accordingly
   ``sigmas ()``, ``interfaceCompression ()``, and ``virtualMass ()`` are empty. Wu 2015 discusses
   a packing limit ``phi_m`` without stating a numeric value (``not found``); MULES bounds
   ``0 <= alpha <= 1`` with exact phase-sum conservation ``alpha.RBC + alpha.plasma = 1``.
5. **Boundary conditions (Wu 2015 Table 1 & 2023 COMSOL slide 6):**
   - Inlet: ``U.plasma`` & ``U.RBC`` fixedValue (uniform ``(1 0 0)`` default, or parabolic),
     ``alpha.RBC = phi``, ``alpha.plasma = 1 - phi`` (sum = 1), ``p_rgh = fixedFluxPressure``.
   - Walls: ``U.plasma = U.RBC = (0 0 0)`` (no-slip), ``alpha.* = zeroGradient``,
     ``p_rgh = fixedFluxPressure``.
   - Outlet: ``p_rgh = fixedValue 0``, ``U.* = pressureInletOutletVelocity`` (zeroGradient on
     outflow), ``alpha.* = inletOutlet`` (zeroGradient on outflow, inletValue = inlet ``phi``).
"""

from __future__ import annotations

import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cfd.gen.geometry import (
    Block2D,
    MeshSpec,
    SampleLine,
    _base_cells_for_length,
    _extrude_2d_mesh,
    _line_pts,
    _scale_cells,
    build_g2,
    build_kg,
    format_block_mesh_dict,
)

# Verified primary-source physical constants
# Wu et al. 2015 §3 (PMC4521229, verified-primary):
WU2015_RHO_PLASMA = 1027.0       # kg/m^3
WU2015_RHO_RBC = 1093.0          # kg/m^3
WU2015_MU_PLASMA = 0.96e-3       # Pa.s (0.96 cP)
WU2015_DP_RBC = 8.0e-6           # m (8 um)
WU2015_RHO_RATIO = WU2015_RHO_RBC / WU2015_RHO_PLASMA  # 1.0642648...

# 2023 COMSOL slides (knowledge/13-2023-comsol-model.md, slide 6):
COMSOL2023_RHO_PLASMA = 1038.0   # kg/m^3 (most likely continuous phase, TODO: confirm)
COMSOL2023_RHO_RBC = 1098.0      # kg/m^3
COMSOL2023_MU_PLASMA = 4.0e-3    # Pa.s
COMSOL2023_DP_RBC = 8.0e-6       # m (8 um)
COMSOL2023_RHO_RATIO = COMSOL2023_RHO_RBC / COMSOL2023_RHO_PLASMA  # 1.0578034...

SUPPORTED_DRAG_MODELS = (
    "WenYu",
    "SchillerNaumann",
    "Gibilaro",
    "GidaspowErgunWenYu",
)


def _fmt(val: float) -> str:
    """Format a float cleanly for OpenFOAM ASCII dictionaries."""
    if abs(val) < 1e-15:
        return "0"
    return f"{val:.10g}"


# ---------------------------------------------------------------------------
# Wu 2015 viscosity helper & dimensional <-> star-unit conversion
# ---------------------------------------------------------------------------

def wu2015_rbc_viscosity_cP(phi: float, gdot_dim: float) -> float:
    """Evaluate Wu 2015 Eqs. (7, 9a–9c) RBC viscosity factor ``mu_rbc * phi`` [cP].

    Note: Wu 2015 Eqs. (9a)–(9c) are polynomial fits to Brooks' data in cP (= mPa.s):
      mu_0(phi)   = 537.002 phi^3 + 55.006 phi^2 - 0.129 phi
      mu_inf(phi) =  27.873 phi^3 - 21.218 phi^2 + 14.439 phi
      k           = 11 [s]
    """
    if not (0.0 < phi < 1.0):
        raise ValueError(f"phi must be in (0, 1), got {phi}")
    if gdot_dim < 0.0:
        raise ValueError(f"gdot_dim must be >= 0, got {gdot_dim}")
    mu0 = 537.002 * phi**3 + 55.006 * phi**2 - 0.129 * phi
    mu_inf = 27.873 * phi**3 - 21.218 * phi**2 + 14.439 * phi
    k_gdot = 11.0 * gdot_dim
    shear_factor = (1.0 + math.log(1.0 + k_gdot)) / (1.0 + k_gdot)
    return float(mu_inf + (mu0 - mu_inf) * shear_factor)


def wu2015_mu_ratio(
    phi: float,
    gdot_dim: float = 1.0e4,
    mu_plasma_cP: float = 0.96,
    stress_factor_half: bool = True,
) -> float:
    """Compute effective star-unit viscosity ratio ``mu_RBC_eff / mu_plasma`` from Wu 2015.

    In Wu 2015 Eq. (6)/(13), the viscous stress is written ``mu_rbc * phi * D_rbc`` (without
    the factor of 2 that appears in OpenFOAM's ``div(2 * alpha * mu * D)``). When
    ``stress_factor_half=True`` (default), ``mu_rbc_OF = 0.5 * (mu_rbc_Wu * phi) / phi``
    so the OpenFOAM stress matches Wu 2015 Eq. (13) at ``(phi, gdot_dim)``.
    """
    mu_rbc_times_phi = wu2015_rbc_viscosity_cP(phi, gdot_dim)
    mu_rbc_cP = mu_rbc_times_phi / phi
    if stress_factor_half:
        mu_rbc_cP *= 0.5
    return float(mu_rbc_cP / mu_plasma_cP)


def dimensional_to_star(
    *,
    h1: float,
    U_mean: float,
    rho_plasma: float,
    mu_plasma: float,
    rho_rbc: float,
    mu_rbc: float,
    d_p: float,
    phi: float,
) -> dict[str, float]:
    """Convert dimensional two-fluid parameters to M2b star-unit dimensionless groups."""
    if h1 <= 0.0 or U_mean <= 0.0 or rho_plasma <= 0.0 or mu_plasma <= 0.0:
        raise ValueError("h1, U_mean, rho_plasma, mu_plasma must all be > 0")
    if rho_rbc <= 0.0 or mu_rbc <= 0.0 or d_p <= 0.0:
        raise ValueError("rho_rbc, mu_rbc, d_p must all be > 0")
    if not (0.0 < phi < 1.0):
        raise ValueError(f"phi must be in (0, 1), got {phi}")

    Re = rho_plasma * U_mean * h1 / mu_plasma
    rho_ratio = rho_rbc / rho_plasma
    mu_ratio = mu_rbc / mu_plasma
    d_p_star = d_p / h1
    nu_plasma_star = 1.0 / Re
    nu_rbc_star = mu_ratio / (rho_ratio * Re)
    return {
        "Re": float(Re),
        "rho_ratio": float(rho_ratio),
        "mu_ratio": float(mu_ratio),
        "d_p_star": float(d_p_star),
        "phi": float(phi),
        "rho_plasma_star": 1.0,
        "rho_rbc_star": float(rho_ratio),
        "nu_plasma_star": float(nu_plasma_star),
        "nu_rbc_star": float(nu_rbc_star),
    }


def star_to_dimensional(
    *,
    star: Mapping[str, float],
    h1: float,
    U_mean: float,
    rho_plasma: float,
) -> dict[str, float]:
    """Scale M2b star-unit parameters back to dimensional SI quantities.

    Given reference scales ``(h1, U_mean, rho_plasma)`` and star groups
    ``(Re, rho_ratio, mu_ratio, d_p_star, phi)``, recovers exact dimensional phase
    densities, viscosities, particle diameter, and reference pressure/time/stress scales.
    """
    Re = float(star["Re"])
    rho_ratio = float(star["rho_ratio"])
    mu_ratio = float(star["mu_ratio"])
    d_p_star = float(star["d_p_star"])
    phi = float(star["phi"])

    nu_plasma_star = float(star.get("nu_plasma_star", 1.0 / Re))
    nu_rbc_star = float(star.get("nu_rbc_star", mu_ratio / (rho_ratio * Re)))

    mu_plasma = rho_plasma * U_mean * h1 / Re
    nu_plasma = nu_plasma_star * U_mean * h1
    rho_rbc = rho_ratio * rho_plasma
    mu_rbc = mu_ratio * mu_plasma
    nu_rbc = nu_rbc_star * U_mean * h1
    d_p = d_p_star * h1

    return {
        "h1": float(h1),
        "U_mean": float(U_mean),
        "rho_plasma": float(rho_plasma),
        "mu_plasma": float(mu_plasma),
        "nu_plasma": float(nu_plasma),
        "rho_rbc": float(rho_rbc),
        "mu_rbc": float(mu_rbc),
        "nu_rbc": float(nu_rbc),
        "d_p": float(d_p),
        "phi": float(phi),
        "t_ref": float(h1 / U_mean),
        "p_dyn_ref": float(rho_plasma * U_mean**2),
        "tau_ref": float(mu_plasma * U_mean / h1),
        "gdot_ref": float(U_mean / h1),
    }


# ---------------------------------------------------------------------------
# G1 straight-channel builder + family dispatcher (G1, G2, KG)
# ---------------------------------------------------------------------------

def build_g1(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G1 (2D planar straight microchannel) in star units (h1 = 1).

    Inlet sits on ``x = 0, y in [0, 1]`` and outlet sits on ``x = L_ratio, y in [0, 1]``.

    Parameters
    ----------
    params : Mapping[str, Any]
        Optional keys:
        - ``L_ratio`` : float in [10.0, 30.0], default 20.0 (``L / h1``)
    level : int in {0, 1, 2}
        Mesh refinement level (refinement ratio ``r = 1.5``).
    """
    L_ratio = float(params.get("L_ratio", 20.0))
    if not (10.0 <= L_ratio <= 30.0):
        raise ValueError(f"G1 L_ratio must be in [10.0, 30.0], got {L_ratio}")

    verts_2d = [
        (0.0, 0.0),      # 0
        (L_ratio, 0.0),  # 1
        (L_ratio, 1.0),  # 2
        (0.0, 1.0),      # 3
    ]

    ny = _scale_cells(16, level)
    nx = _scale_cells(_base_cells_for_length(L_ratio, 6.0, minimum=40), level)

    blocks_2d = [Block2D((0, 1, 2, 3), nx, ny)]
    patch_edges = [
        ("inlet", "patch", [(3, 0)]),
        ("outlet", "patch", [(1, 2)]),
        ("walls", "wall", [(0, 1), (2, 3)]),
    ]
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(0.5, 0.0, 0.5, 1.0), n_points=80),
        SampleLine(
            "midProfile",
            "y",
            *_line_pts(0.5 * L_ratio, 0.0, 0.5 * L_ratio, 1.0),
            n_points=80,
        ),
        SampleLine(
            "centerline",
            "x",
            (0.02, 0.5, 0.5),
            (L_ratio - 0.02, 0.5, 0.5),
            n_points=200,
        ),
        SampleLine(
            "outletProfile",
            "y",
            *_line_pts(L_ratio - 1.0, 0.0, L_ratio - 1.0, 1.0),
            n_points=100,
        ),
    ]

    return _extrude_2d_mesh(
        family="G1",
        level=level,
        params={"L_ratio": L_ratio},
        verts_2d=verts_2d,
        blocks_2d=blocks_2d,
        patch_edges=patch_edges,
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_ratio,
        outlet_lengths=(L_ratio,),
    )


def build_m2b_geometry(family: str, params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build MeshSpec for M2b-supported families ``G1``, ``G2``, or ``KG``."""
    key = family.upper()
    if key == "G1":
        return build_g1(params, level=level)
    if key == "G2":
        return build_g2(params, level=level)
    if key == "KG":
        return build_kg(params, level=level)
    raise ValueError(f"M2b supports families ('G1', 'G2', 'KG'), got '{family}'")


# ---------------------------------------------------------------------------
# OpenFOAM ESI v2306 multiphaseEulerFoam dictionary formatters
# ---------------------------------------------------------------------------

def _format_alpha_field(mesh: MeshSpec, phase_name: str, alpha_inlet: float) -> str:
    """Generate ``0/alpha.<phase>`` with exact inlet phase fraction."""
    val_str = _fmt(alpha_inlet)
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volScalarField;",
        f"    object      alpha.{phase_name};",
        "}",
        "",
        "dimensions      [0 0 0 0 0 0 0];",
        "",
        f"internalField   uniform {val_str};",
        "",
        "boundaryField",
        "{",
    ]
    for patch in mesh.patches:
        if patch.name == "inlet":
            lines.extend(
                [
                    "    inlet",
                    "    {",
                    "        type            fixedValue;",
                    f"        value           uniform {val_str};",
                    "    }",
                ]
            )
        elif patch.name.startswith("outlet"):
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            inletOutlet;",
                    f"        phi             phi.{phase_name};",
                    f"        inletValue      uniform {val_str};",
                    f"        value           uniform {val_str};",
                    "    }",
                ]
            )
        elif patch.patch_type == "wall":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            zeroGradient;",
                    "    }",
                ]
            )
        elif patch.patch_type == "empty":
            lines.extend([f"    {patch.name}", "    {", "        type            empty;", "    }"])
        elif patch.patch_type == "wedge":
            lines.extend([f"    {patch.name}", "    {", "        type            wedge;", "    }"])
        else:
            raise ValueError(f"Unhandled patch '{patch.name}' (type='{patch.patch_type}')")
    lines.extend(["}", ""])
    return "\n".join(lines)


def _format_phase_u_field(mesh: MeshSpec, phase_name: str, inlet_profile: str) -> str:
    """Generate ``0/U.<phase>`` in star units (uniform or parabolic inlet, no-slip walls)."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volVectorField;",
        f"    object      U.{phase_name};",
        "}",
        "",
        "dimensions      [0 1 -1 0 0 0 0];",
        "",
        "internalField   uniform (0 0 0);",
        "",
        "boundaryField",
        "{",
    ]
    for patch in mesh.patches:
        if patch.name == "inlet":
            if inlet_profile == "parabolic":
                if mesh.family == "KG":
                    expr = (
                        "vector(2.0 * max(0.0, 1.0 - 4.0 * "
                        "(pos().y()*pos().y() + pos().z()*pos().z())), 0, 0)"
                    )
                else:
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
            elif inlet_profile == "uniform":
                lines.extend(
                    [
                        "    inlet",
                        "    {",
                        "        type            fixedValue;",
                        "        value           uniform (1 0 0);",
                        "    }",
                    ]
                )
            else:
                raise ValueError(
                    f"inlet_profile must be 'uniform' or 'parabolic', got '{inlet_profile}'"
                )
        elif patch.name.startswith("outlet"):
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            pressureInletOutletVelocity;",
                    f"        phi             phi.{phase_name};",
                    "        value           uniform (0 0 0);",
                    "    }",
                ]
            )
        elif patch.patch_type == "wall":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            fixedValue;",
                    "        value           uniform (0 0 0);",
                    "    }",
                ]
            )
        elif patch.patch_type == "empty":
            lines.extend([f"    {patch.name}", "    {", "        type            empty;", "    }"])
        elif patch.patch_type == "wedge":
            lines.extend([f"    {patch.name}", "    {", "        type            wedge;", "    }"])
        else:
            raise ValueError(f"Unhandled patch '{patch.name}' (type='{patch.patch_type}')")
    lines.extend(["}", ""])
    return "\n".join(lines)


def _format_p_rgh_field(mesh: MeshSpec) -> str:
    """Generate ``0/p_rgh`` (dynamic star pressure, Wu 2015 Table 1 BCs)."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volScalarField;",
        "    object      p_rgh;",
        "}",
        "",
        "dimensions      [1 -1 -2 0 0 0 0];",
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
        elif patch.name == "inlet" or patch.patch_type == "wall":
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            fixedFluxPressure;",
                    "        value           uniform 0;",
                    "    }",
                ]
            )
        elif patch.patch_type == "empty":
            lines.extend([f"    {patch.name}", "    {", "        type            empty;", "    }"])
        elif patch.patch_type == "wedge":
            lines.extend([f"    {patch.name}", "    {", "        type            wedge;", "    }"])
        else:
            raise ValueError(f"Unhandled patch '{patch.name}' (type='{patch.patch_type}')")
    lines.extend(["}", ""])
    return "\n".join(lines)


def _format_transport_properties(
    *,
    Re: float,
    rho_ratio: float,
    mu_ratio: float,
    d_p_star: float,
    drag_model: str,
) -> str:
    """Generate ``constant/transportProperties`` for ESI v2306 ``multiphaseEulerFoam``."""
    if Re <= 0.0:
        raise ValueError(f"Re must be > 0, got {Re}")
    if rho_ratio <= 0.0:
        raise ValueError(f"rho_ratio must be > 0, got {rho_ratio}")
    if mu_ratio <= 0.0:
        raise ValueError(f"mu_ratio must be > 0, got {mu_ratio}")
    if d_p_star <= 0.0:
        raise ValueError(f"d_p_star must be > 0, got {d_p_star}")
    if drag_model not in SUPPORTED_DRAG_MODELS:
        raise ValueError(
            f"Unsupported drag_model '{drag_model}'; expected one of {SUPPORTED_DRAG_MODELS}"
        )

    nu_plasma = 1.0 / Re
    nu_rbc = mu_ratio / (rho_ratio * Re)

    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      transportProperties;",
            "}",
            "",
            "// Star units: rho_plasma* = 1, h1* = 1, U_mean* = 1",
            f"// Re = {_fmt(Re)}, rho_RBC/rho_p = {_fmt(rho_ratio)}, "
            f"mu_RBC/mu_p = {_fmt(mu_ratio)}, d_p/h1 = {_fmt(d_p_star)}",
            "phases",
            "(",
            "    RBC",
            "    {",
            f"        rho             {_fmt(rho_ratio)};",
            f"        nu              {_fmt(nu_rbc)};",
            "        kappa           0;",
            "        Cp              0;",
            "",
            "        diameterModel   constant;",
            "        constantCoeffs",
            "        {",
            f"            d               {_fmt(d_p_star)};",
            "        }",
            "    }",
            "",
            "    plasma",
            "    {",
            "        rho             1;",
            f"        nu              {_fmt(nu_plasma)};",
            "        kappa           0;",
            "        Cp              0;",
            "",
            "        diameterModel   constant;",
            "        constantCoeffs",
            "        {",
            f"            d               {_fmt(d_p_star)};",
            "        }",
            "    }",
            ");",
            "",
            "sigmas",
            "(",
            ");",
            "",
            "interfaceCompression",
            "(",
            ");",
            "",
            "virtualMass",
            "(",
            ");",
            "",
            "// Dispersed phase (RBC) listed first, continuous phase (plasma) second",
            "// so stock dragModel uses d_RBC, rho_plasma, nu_plasma, and alpha_plasma = 1 - phi.",
            "drag",
            "(",
            "    (RBC plasma)",
            "    {",
            f"        type                    {drag_model};",
            "        residualPhaseFraction   1e-6;",
            "        residualSlip            1e-6;",
            "    }",
            ");",
            "",
            "// Required dummy entries for singlePhaseTransportModel / turbulenceModel",
            "transportModel  Newtonian;",
            "nu              0;",
            "",
        ]
    )


def _format_turbulence_properties() -> str:
    """Generate ``constant/turbulenceProperties`` (laminar flow)."""
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


def _format_g() -> str:
    """Generate ``constant/g`` with zero gravity (microchannel, Wu 2015 neglects body forces)."""
    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       uniformDimensionedVectorField;",
            "    object      g;",
            "}",
            "",
            "dimensions      [0 1 -2 0 0 0 0];",
            "value           (0 0 0);",
            "",
        ]
    )


def _format_fv_schemes() -> str:
    """Generate ``system/fvSchemes`` for ESI v2306 ``multiphaseEulerFoam``."""
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
            "    default         Euler;",
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
            '    "div\\(phi,alpha.*\\)"            Gauss vanLeer;',
            '    "div\\(phir,alpha.*,alpha.*\\)"   Gauss vanLeer;',
            '    "div\\(alphaPhi.*,U.*\\)"         Gauss limitedLinearV 1;',
            "    div(Rc)                         Gauss linear;",
            '    "div\\(phi.*,U.*\\)"              Gauss limitedLinearV 1;',
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


def _format_fv_solution() -> str:
    """Generate ``system/fvSolution`` for ESI v2306 ``multiphaseEulerFoam`` (PIMPLE)."""
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
            '    "alpha.*"',
            "    {",
            "        nAlphaSubCycles 2;",
            "    }",
            "",
            "    p_rgh",
            "    {",
            "        solver          GAMG;",
            "        smoother        DIC;",
            "        tolerance       1e-8;",
            "        relTol          0.01;",
            "    }",
            "",
            "    p_rghFinal",
            "    {",
            "        $p_rgh;",
            "        tolerance       1e-9;",
            "        relTol          0;",
            "    }",
            "",
            '    "pcorr.*"',
            "    {",
            "        $p_rgh;",
            "        tolerance       1e-6;",
            "        relTol          0;",
            "    }",
            "",
            '    "U.*"',
            "    {",
            "        solver          smoothSolver;",
            "        smoother        symGaussSeidel;",
            "        tolerance       1e-7;",
            "        relTol          0;",
            "    }",
            "}",
            "",
            "PIMPLE",
            "{",
            "    nOuterCorrectors            2;",
            "    nCorrectors                 2;",
            "    nNonOrthogonalCorrectors    1;",
            "}",
            "",
            "relaxationFactors",
            "{",
            "    equations",
            "    {",
            '        "U.*"           0.7;',
            '        "U.*Final"      1.0;',
            "    }",
            "}",
            "",
        ]
    )


def _format_control_dict(
    mesh: MeshSpec,
    *,
    end_time: float,
    delta_t: float,
    max_co: float,
    max_delta_t: float,
) -> str:
    """Generate ``system/controlDict`` for transient-to-steady ``multiphaseEulerFoam``."""
    write_interval = max(1.0, round(end_time / 4.0, 4))
    monitor_interval = max(delta_t, round(end_time / 100.0, 6))
    steady_start = round(0.88 * end_time, 6)
    steady_interval = max(delta_t, round(0.02 * end_time, 6))
    near_wall_names = {
        "G2": "nearWallLower",
        "KG": "nearWallOuter",
    }
    steady_line_names = ["outletProfile"]
    if mesh.family in near_wall_names:
        steady_line_names.append(near_wall_names[mesh.family])
    steady_lines = [sl for sl in mesh.sample_lines if sl.name in steady_line_names]

    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      controlDict;",
        "}",
        "",
        "application     multiphaseEulerFoam;",
        "startFrom       startTime;",
        "startTime       0;",
        "stopAt          endTime;",
        f"endTime         {_fmt(end_time)};",
        f"deltaT          {_fmt(delta_t)};",
        "writeControl    adjustableRunTime;",
        f"writeInterval   {_fmt(write_interval)};",
        "purgeWrite      2;",
        "writeFormat     ascii;",
        "writePrecision  8;",
        "writeCompression off;",
        "timeFormat      general;",
        "timePrecision   6;",
        "runTimeModifiable true;",
        "adjustTimeStep  yes;",
        f"maxCo           {_fmt(max_co)};",
        f"maxDeltaT       {_fmt(max_delta_t)};",
        "",
        "functions",
        "{",
        "    inletPressure",
        "    {",
        "        type            surfaceFieldValue;",
        "        libs            (fieldFunctionObjects);",
        "        writeControl    runTime;",
        f"        writeInterval   {_fmt(monitor_interval)};",
        "        writeFields     false;",
        "        log             false;",
        "        regionType      patch;",
        "        name            inlet;",
        "        operation       areaAverage;",
        "        fields          (p);",
        "    }",
        "",
        "    outletPhi",
        "    {",
        "        type            surfaceFieldValue;",
        "        libs            (fieldFunctionObjects);",
        "        writeControl    runTime;",
        f"        writeInterval   {_fmt(monitor_interval)};",
        "        writeFields     false;",
        "        log             false;",
        "        regionType      patch;",
        "        name            outlet;",
        "        operation       areaAverage;",
        "        fields          (alpha.RBC);",
        "    }",
        "",
        "    steadySample",
        "    {",
        "        type                sets;",
        '        libs                ("libsampling.so");',
        f"        timeStart           {_fmt(steady_start)};",
        "        writeControl        runTime;",
        f"        writeInterval       {_fmt(steady_interval)};",
        "        interpolationScheme cellPoint;",
        "        setFormat           raw;",
        "        fields              (alpha.RBC U.plasma U.RBC U p);",
        "        sets",
        "        {",
    ]
    for sl in steady_lines:
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
    lines.extend(
        [
            "        }",
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
            "    gradUPlasma",
            "    {",
            "        type            grad;",
            "        libs            (fieldFunctionObjects);",
            "        field           U.plasma;",
            "        writeControl    onEnd;",
            "    }",
            "",
            "    gradURBC",
            "    {",
            "        type            grad;",
            "        libs            (fieldFunctionObjects);",
            "        field           U.RBC;",
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
            "        fields              (alpha.RBC alpha.plasma U.plasma U.RBC U p);",
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


def _format_run_case_m2b_sh() -> str:
    """Return bash runner script for an M2b ``multiphaseEulerFoam`` case."""
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Run a generated M2b nondimensional OpenFOAM v2306 multiphaseEulerFoam case:",
            "#   blockMesh -> checkMesh -> multiphaseEulerFoam -> postProcess",
            "# Exits non-zero on mesh errors, solver failure, NaN/Inf, or incomplete run.",
            "",
            "set -euo pipefail",
            "",
            "check_m2b_log() {",
            '    local log_file="$1"',
            "    local filtered",
            '    filtered="$(grep -v -E \'trapFpe:|FOAM_SIGFPE\' "$log_file" || true)"',
            '    if printf \'%s\\n\' "$filtered" | grep -Eqi "FOAM FATAL|solution singularity|Floating point exception \\(core dumped\\)"; then',
            '        echo "ERROR: multiphaseEulerFoam log shows fatal error or crash" >&2',
            '        printf \'%s\\n\' "$filtered" | grep -Ei "FOAM FATAL|solution singularity|Floating point exception \\(core dumped\\)" | head -n 20 >&2 || true',
            "        return 4",
            "    fi",
            '    if printf \'%s\\n\' "$filtered" | grep -Eq "\\bsigFpe\\b"; then',
            '        echo "ERROR: multiphaseEulerFoam log shows sigFpe crash" >&2',
            '        printf \'%s\\n\' "$filtered" | grep -E "\\bsigFpe\\b" | head -n 20 >&2 || true',
            "        return 4",
            "    fi",
            '    if printf \'%s\\n\' "$filtered" | grep -E "(Solving for|continuity errors|volume fraction)" | grep -Eqi "(^|[^A-Za-z_])[-+]?(nan|inf)([^A-Za-z_]|$)"; then',
            '        echo "ERROR: multiphaseEulerFoam log shows NaN/Inf divergence" >&2',
            '        printf \'%s\\n\' "$filtered" | grep -E "(Solving for|continuity errors|volume fraction)" | grep -Ei "(^|[^A-Za-z_])[-+]?(nan|inf)([^A-Za-z_]|$)" | head -n 20 >&2 || true',
            "        return 4",
            "    fi",
            "    return 0",
            "}",
            "",
            'if [ "${1:-}" = "--check-log" ]; then',
            '    check_m2b_log "$2"',
            "    exit $?",
            "fi",
            "",
            'CASE_DIR="${1:-.}"',
            'cd "$CASE_DIR"',
            "",
            "set +eu",
            "if [ -f /usr/lib/openfoam/openfoam2306/etc/bashrc ]; then",
            "    # shellcheck disable=SC1091",
            "    . /usr/lib/openfoam/openfoam2306/etc/bashrc || true",
            "fi",
            "set -eu",
            "",
            'echo "=== [1/4] blockMesh ($CASE_DIR) ==="',
            "blockMesh > log.blockMesh 2>&1",
            "",
            'echo "=== [2/4] checkMesh ==="',
            "checkMesh -allGeometry -allTopology > log.checkMesh 2>&1",
            'if grep -Eq "Failed [0-9]+ mesh checks|FOAM FATAL" log.checkMesh; then',
            '    echo "ERROR: checkMesh reported mesh check failures:" >&2',
            '    grep -E "Failed|ERROR|FOAM FATAL|non-orthogonality|skewness" log.checkMesh >&2 || true',
            "    exit 2",
            "fi",
            'if ! grep -q "Mesh OK\\." log.checkMesh; then',
            '    echo "ERROR: checkMesh did not report Mesh OK." >&2',
            "    tail -n 30 log.checkMesh >&2 || true",
            "    exit 2",
            "fi",
            "",
            'echo "=== [3/4] multiphaseEulerFoam ==="',
            "if ! multiphaseEulerFoam > log.multiphaseEulerFoam 2>&1; then",
            '    echo "ERROR: multiphaseEulerFoam exited non-zero" >&2',
            "    tail -n 40 log.multiphaseEulerFoam >&2 || true",
            "    exit 3",
            "fi",
            "",
            'check_m2b_log log.multiphaseEulerFoam || exit 4',
            "",
            'if ! grep -q "^End$" log.multiphaseEulerFoam; then',
            '    echo "ERROR: multiphaseEulerFoam did not reach End" >&2',
            "    tail -n 25 log.multiphaseEulerFoam >&2 || true",
            "    exit 5",
            "fi",
            "",
            'echo "=== [4/4] postProcess (C, V, grad(U.plasma), grad(U.RBC), sample) ==="',
            "postProcess -func writeCellCentres -latestTime > log.postProcess_C 2>&1",
            "postProcess -func writeCellVolumes -latestTime > log.postProcess_V 2>&1",
            "postProcess -func 'grad(U.plasma)' -latestTime > log.postProcess_gradUPlasma 2>&1",
            "postProcess -func 'grad(U.RBC)' -latestTime > log.postProcess_gradURBC 2>&1",
            "postProcess -func sample -latestTime > log.postProcess_sample 2>&1",
            "",
            'echo "=== DONE: multiphaseEulerFoam reached End ==="',
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Public case writer
# ---------------------------------------------------------------------------

def write_m2b_case(
    case_dir: Path | str,
    params: Mapping[str, Any],
    level: int = 1,
    end_time: float | None = None,
    delta_t: float = 1e-4,
    max_co: float = 0.5,
    max_delta_t: float = 0.05,
) -> MeshSpec:
    """Write a complete nondimensional OpenFOAM ESI v2306 ``multiphaseEulerFoam`` case.

    Parameters
    ----------
    case_dir : Path | str
        Destination case directory (created if missing).
    params : Mapping[str, Any]
        Must contain:
        - ``family``: ``'G1'``, ``'G2'``, or ``'KG'``
        - ``Re``: plasma-based Reynolds number (> 0)
        - ``phi``: inlet RBC volume fraction in ``(0, 1)``
        - ``d_p_star``: dimensionless RBC diameter ``d_p / h1`` (> 0)
        Optional keys:
        - ``rho_ratio``: ``rho_RBC / rho_p`` (default Wu 2015 ``1093 / 1027``)
        - ``mu_ratio``: ``mu_RBC_eff / mu_p`` (default evaluated via ``wu2015_mu_ratio(phi)``)
        - ``drag_model``: one of ``SUPPORTED_DRAG_MODELS`` (default ``'WenYu'``)
        - ``inlet_profile``: ``'uniform'`` (default, Wu 2015 Table 1) or ``'parabolic'``
        - ``end_time``: star-unit simulation end time (default ``max(60.0, 2.5 * domain_length)``)
        - ``geometry``: family geometry dictionary (or top-level geometry keys)
    level : int in {0, 1, 2}
        Mesh refinement level (``r = 1.5`` per level).
    """
    family = str(params["family"]).upper()
    Re = float(params["Re"])
    phi = float(params["phi"])
    if not (0.0 < phi < 1.0):
        raise ValueError(f"Inlet RBC volume fraction phi must be in (0, 1), got {phi}")

    rho_ratio = float(params.get("rho_ratio", WU2015_RHO_RATIO))
    if "mu_ratio" in params and params["mu_ratio"] is not None:
        mu_ratio = float(params["mu_ratio"])
    else:
        mu_ratio = wu2015_mu_ratio(phi)
    d_p_star = float(params["d_p_star"])
    drag_model = str(params.get("drag_model", "WenYu"))
    inlet_profile = str(params.get("inlet_profile", "uniform"))

    geom_params: dict[str, Any] = dict(params.get("geometry", params))
    mesh = build_m2b_geometry(family, geom_params, level=level)

    # Choose default steady-state horizon = at least 2.5 domain flow-through times in star units.
    xs_v = [pt[0] for pt in mesh.vertices_2d]
    domain_len = max(xs_v) - min(xs_v)
    if end_time is None:
        end_time_val = float(params.get("end_time", max(60.0, round(2.5 * domain_len, 1))))
    else:
        end_time_val = float(end_time)
    if end_time_val <= 0.0:
        raise ValueError(f"end_time must be > 0, got {end_time_val}")

    alpha_rbc = phi
    alpha_plasma = 1.0 - phi

    root = Path(case_dir)
    for sub in ("0", "constant", "system"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / "system" / "blockMeshDict").write_text(format_block_mesh_dict(mesh), encoding="utf-8")
    (root / "system" / "controlDict").write_text(
        _format_control_dict(
            mesh,
            end_time=end_time_val,
            delta_t=delta_t,
            max_co=max_co,
            max_delta_t=max_delta_t,
        ),
        encoding="utf-8",
    )
    (root / "system" / "fvSchemes").write_text(_format_fv_schemes(), encoding="utf-8")
    (root / "system" / "fvSolution").write_text(_format_fv_solution(), encoding="utf-8")

    (root / "0" / "alpha.RBC").write_text(
        _format_alpha_field(mesh, "RBC", alpha_rbc), encoding="utf-8"
    )
    (root / "0" / "alpha.plasma").write_text(
        _format_alpha_field(mesh, "plasma", alpha_plasma), encoding="utf-8"
    )
    (root / "0" / "U.RBC").write_text(
        _format_phase_u_field(mesh, "RBC", inlet_profile), encoding="utf-8"
    )
    (root / "0" / "U.plasma").write_text(
        _format_phase_u_field(mesh, "plasma", inlet_profile), encoding="utf-8"
    )
    (root / "0" / "p_rgh").write_text(_format_p_rgh_field(mesh), encoding="utf-8")

    (root / "constant" / "transportProperties").write_text(
        _format_transport_properties(
            Re=Re,
            rho_ratio=rho_ratio,
            mu_ratio=mu_ratio,
            d_p_star=d_p_star,
            drag_model=drag_model,
        ),
        encoding="utf-8",
    )
    (root / "constant" / "turbulenceProperties").write_text(
        _format_turbulence_properties(), encoding="utf-8"
    )
    (root / "constant" / "g").write_text(_format_g(), encoding="utf-8")

    run_script = root / "run_case.sh"
    run_script.write_text(_format_run_case_m2b_sh(), encoding="utf-8")
    os.chmod(
        run_script,
        run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    total_cells = sum(nx * ny * nz for _, (nx, ny, nz), _ in mesh.blocks)
    meta = {
        "run_id": params.get("run_id"),
        "tier": "m2b",
        "solver": "multiphaseEulerFoam",
        "family": family,
        "level": level,
        "Re": Re,
        "phi": phi,
        "alpha_RBC_inlet": alpha_rbc,
        "alpha_plasma_inlet": alpha_plasma,
        "rho_ratio": rho_ratio,
        "mu_ratio": mu_ratio,
        "d_p_star": d_p_star,
        "nu_plasma_star": 1.0 / Re,
        "nu_rbc_star": mu_ratio / (rho_ratio * Re),
        "drag_model": drag_model,
        "lift_model": "none (not supported in stock ESI v2306 multiphaseEulerFoam)",
        "rbc_viscosity_model": "constant effective nu (Wu 2015 Yeleswarapu fallback)",
        "inlet_profile": inlet_profile,
        "end_time": end_time_val,
        "geometry": mesh.params,
        "n_blocks": len(mesh.blocks),
        "n_cells": total_cells,
        "sample_lines": [sl.name for sl in mesh.sample_lines],
    }
    (root / "params.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return mesh
