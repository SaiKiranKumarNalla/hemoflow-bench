"""OpenFOAM ESI v2306 ``reactingTwoPhaseEulerFoam`` lift-capable case writer (M2b-B) in star units.

Provides the lift-enabled companion to M2b-A (``cfd/gen/m2b.py``, stock ``multiphaseEulerFoam``
without lift) using ESI v2306 ``reactingTwoPhaseEulerFoam`` in isothermal, constant-density
(``rhoConst``, ``nEnergyCorrectors 0``) mode with ``noBlending`` (``RBC`` always dispersed in
continuous ``plasma``), ``WenYu`` drag, and stock ESI v2306 lift.

T7 (2026-11-24) extends the writer for the EE-full variant and for pseudo-steady runs:

- ``wall_lubrication_model``: ``none`` | ``Antal`` | ``Frank`` | ``Tomiyama`` (the only stock
  wall-lubrication models in v2306). Coefficients are never defaulted — ``Antal`` needs ``Cw1``,
  ``Cw2``, ``Frank`` needs ``Cwd``, ``Cwc``, ``p``, ``Tomiyama`` needs ``Cwd`` (a length). v2306 has
  no ``Lubchenko``/``Hosokawa`` and no wall-damped virtual mass.
- ``virtual_mass_model``: ``none`` | ``constantCoefficient`` (``cvm``) | ``Lamb``.
- any wall model activates ``fvSchemes.wallDist { method meshWave; }``, which ``wallDist::New`` reads.
- ``local_euler``: ``localEuler`` LTS, wired in v2306 through ``createRDeltaT.H``/``setRDeltaT.H``;
  ``maxCo``/``maxDeltaT``/``rDeltaTSmoothingCoeff`` then live in the ``PIMPLE`` dict and ``deltaT``
  is set to ``maxDeltaT`` because the solver ignores ``adjustTimeStep`` in that branch.
  ``multiphaseEulerFoam`` has no LTS in v2306, so M2b-A stays transient with ``adjustTimeStep``.

Verified ESI v2306 lift models (``knowledge/14-wu2015-closures.md`` §6, verified verbatim in
``api.openfoam.com/2306/`` & OpenFOAM-v2306 source):
1. **Default: ``LegendreMagnaudet``** (Legendre & Magnaudet 1998, JFM 368:81–126; Otromke 2013).
   In the microfluidic low-``Re_p`` regime (``Sr = Re_G / Re_p >> 0.2 Re_p``), ``LegendreMagnaudet::Cl()``
   reduces analytically to exact Saffman shear-lift scaling:

       Cl_low -> (6 * 2.255 / pi^2) / sqrt(Re_p * Sr)
               = 1.3709 / sqrt(Re_G)
               = 1.3709 / (d_p* * sqrt(Re * gdot*))

   so ``Re_p`` cancels identically and the volumetric lift force on dispersed RBCs is

       |F_lift| = phi * (6 * 2.255 / pi^2) * (rho_p * mu_p)^(1/2) / d_p * gdot^(1/2) * |U_r|

   with prefactor ``1.3709`` sitting between Wu 2015 Eq. (10)'s printed prefactor
   (``3 * 6.46 / (8 pi) = 0.7711``) and classic rigid-sphere Saffman (1965)
   (``3 * 6.46 / (2 pi) = 3.0844``), and the exact sign verified from ``liftModel.C``,
   ``MomentumTransferPhaseSystem.C``, and ``pU/pEqn.H`` (positive ``Cl`` pushes lagging RBCs
   away from walls toward the channel centreline).
2. **Optional: ``constantCoefficient``** (``constantLiftCoefficient``, dict key ``Cl``).
   Matched to Saffman lift at a nominal star shear rate ``gdot_star_nom`` via:

       Cl_saff* = K_saff / (d_p* * sqrt(Re * gdot_star_nom))

   where ``K_saff = K_WU_EQ10 = 3 * 6.46 / (8 pi) = 0.7711`` (Wu 2015 Eq. 10) or
   ``K_SAFFMAN_SPHERE = 3 * 6.46 / (2 pi) = 3.0844`` (classic rigid-sphere Saffman 1965).
3. **Optional: ``none``** (``noLift``, control comparison with lift switched off).

Star-unit thermophysical properties (isothermal constant-density ``heRhoThermo``):
- ``plasma``: ``rho = 1``, ``mu = 1 / Re`` (so ``nu = mu / rho = 1 / Re``)
- ``RBC``:    ``rho = rho_ratio``, ``mu = mu_ratio / Re`` (so ``nu = mu_ratio / (rho_ratio * Re)``)
- ``p_rgh`` outlet fixed at ``0`` with ``pMin = -1e12`` (``rhoConst`` is independent of ``p`` and ``T``),
  so ``p* = Re * p_OF`` matches M2b-A and M1 identically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from campaign.design_m2b import generate_m2b_design
from cfd.gen.geometry import MeshSpec, format_block_mesh_dict
from cfd.gen.m2b import (
    SUPPORTED_DRAG_MODELS,
    WU2015_RHO_RATIO,
    _fmt,
    _format_alpha_field,
    _format_g,
    _format_p_rgh_field,
    _format_phase_u_field,
    build_m2b_geometry,
    wu2015_mu_ratio,
)

# Verified Saffman / Legendre-Magnaudet prefactors (knowledge/14-wu2015-closures.md §6.3)
K_WU_EQ10 = 3.0 * 6.46 / (8.0 * math.pi)             # 0.77110569... (Wu 2015 Eq. 10 printed)
K_LEGENDRE_MAGNAUDET = 6.0 * 2.255 / (math.pi ** 2)  # 1.37086153... (ESI v2306 low-Re limit)
K_SAFFMAN_SPHERE = 3.0 * 6.46 / (2.0 * math.pi)      # 3.08442276... (Saffman 1965 rigid sphere)

SUPPORTED_LIFT_MODELS = (
    "LegendreMagnaudet",
    "constantCoefficient",
    "none",
)

# Stock ESI v2306 interfacial models (verified in the v2306 source, knowledge/14 §6,
# plan/tasks/T7 §4b follow-up): wall lubrication = Antal | Frank | Tomiyama | none,
# virtual mass = constantCoefficient | Lamb | none. There is no Lubchenko/Hosokawa and no
# wall-damped virtual mass in v2306; coefficient values have no built-in defaults.
SUPPORTED_WALL_LUBRICATION_MODELS = ("none", "Antal", "Frank", "Tomiyama")
SUPPORTED_VIRTUAL_MASS_MODELS = ("none", "constantCoefficient", "Lamb")

DEFAULT_LIFT_MODEL = "LegendreMagnaudet"
DEFAULT_RESIDUAL_RE = 1.0e-3
DEFAULT_GDOT_STAR_NOM = 4.0  # representative channel shear rate (between mean 3 and wall 6)


# ---------------------------------------------------------------------------
# Star-unit Saffman lift coefficient helpers
# ---------------------------------------------------------------------------

def legendre_magnaudet_cl_star(
    *,
    Re: float,
    d_p_star: float,
    gdot_star: float,
    Re_p: float = 1.0e-3,
    residual_re: float = DEFAULT_RESIDUAL_RE,
) -> float:
    """Evaluate ESI v2306 ``LegendreMagnaudet::Cl()`` exactly in star units.

    Mirrors ``LegendreMagnaudet.C`` lines 70–98 verbatim:
      Re_eff = max(Re_p, residual_re)
      Re_G   = Re * (d_p_star^2) * gdot_star
      Sr     = Re_G / Re_eff
      Cl_low^2  = (6 * 2.255)^2 * Sr^2 / (pi^4 * Re_eff * (Sr + 0.2 * Re_eff)^3)
      Cl_high^2 = (0.5 * (Re_eff + 16) / (Re_eff + 29))^2
      Cl        = sqrt(Cl_low^2 + Cl_high^2)
    """
    if Re <= 0.0 or d_p_star <= 0.0 or gdot_star < 0.0 or residual_re <= 0.0:
        raise ValueError("Re, d_p_star, residual_re must be > 0 and gdot_star >= 0")
    re_eff = max(float(Re_p), float(residual_re))
    re_g = float(Re) * (float(d_p_star) ** 2) * float(gdot_star)
    sr = re_g / re_eff
    cl_low_sqr = (
        ((6.0 * 2.255) ** 2) * (sr ** 2)
        / ((math.pi ** 4) * re_eff * ((sr + 0.2 * re_eff) ** 3))
    )
    cl_high_sqr = (0.5 * (re_eff + 16.0) / (re_eff + 29.0)) ** 2
    return float(math.sqrt(cl_low_sqr + cl_high_sqr))


def saffman_constant_cl(
    *,
    Re: float,
    d_p_star: float,
    gdot_star_nom: float = DEFAULT_GDOT_STAR_NOM,
    prefactor: float = K_WU_EQ10,
) -> float:
    """Derive constant lift coefficient ``Cl`` matched to Saffman lift in star units.

    Equating OpenFOAM's ``phi * Cl * rho_p* * |U_r*| * gdot*`` to Saffman lift
    ``phi * K * (Re)^(-1/2) * (d_p*)^(-1) * (gdot*)^(1/2) * |U_r*|`` at nominal
    star shear rate ``gdot_star_nom`` gives:

        Cl* = K / (d_p* * sqrt(Re * gdot_star_nom)) = K / sqrt(Re_G,nom)

    Defaults to ``prefactor = K_WU_EQ10 = 3 * 6.46 / (8 pi) = 0.7711`` (Wu 2015 Eq. 10).
    """
    if Re <= 0.0 or d_p_star <= 0.0 or gdot_star_nom <= 0.0 or prefactor <= 0.0:
        raise ValueError("Re, d_p_star, gdot_star_nom, prefactor must all be > 0")
    re_g_nom = float(Re) * (float(d_p_star) ** 2) * float(gdot_star_nom)
    return float(prefactor / math.sqrt(re_g_nom))


# ---------------------------------------------------------------------------
# OpenFOAM ESI v2306 reactingTwoPhaseEulerFoam dictionary formatters
# ---------------------------------------------------------------------------

def _format_p_field(mesh: MeshSpec) -> str:
    """Generate ``0/p`` (calculated from ``p_rgh`` with ``g = 0``, so ``p == p_rgh``)."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volScalarField;",
        "    object      p;",
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
        if patch.patch_type == "empty":
            lines.extend([f"    {patch.name}", "    {", "        type            empty;", "    }"])
        elif patch.patch_type == "wedge":
            lines.extend([f"    {patch.name}", "    {", "        type            wedge;", "    }"])
        else:
            lines.extend(
                [
                    f"    {patch.name}",
                    "    {",
                    "        type            calculated;",
                    "        value           $internalField;",
                    "    }",
                ]
            )
    lines.extend(["}", ""])
    return "\n".join(lines)


def _format_t_field(mesh: MeshSpec, phase_name: str, temperature: float = 300.0) -> str:
    """Generate isothermal ``0/T.<phase>`` required by ``heRhoThermo`` (energy eqns skipped)."""
    t_str = _fmt(temperature)
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volScalarField;",
        f"    object      T.{phase_name};",
        "}",
        "",
        "dimensions      [0 0 0 1 0 0 0];",
        "",
        f"internalField   uniform {t_str};",
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
                    "        value           $internalField;",
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
                    "        inletValue      $internalField;",
                    "        value           $internalField;",
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


def _format_thermophysical_properties(
    *,
    phase_name: str,
    rho_star: float,
    mu_star: float,
) -> str:
    """Generate constant-density isothermal ``constant/thermophysicalProperties.<phase>``."""
    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            f"    object      thermophysicalProperties.{phase_name};",
            "}",
            "",
            "// Star units: constant-density (rhoConst) isothermal heRhoThermo",
            f"// rho* = {_fmt(rho_star)}, mu* = {_fmt(mu_star)}, nu* = {_fmt(mu_star / rho_star)}",
            "thermoType",
            "{",
            "    type            heRhoThermo;",
            "    mixture         pureMixture;",
            "    transport       const;",
            "    thermo          eConst;",
            "    equationOfState rhoConst;",
            "    specie          specie;",
            "    energy          sensibleInternalEnergy;",
            "}",
            "",
            "mixture",
            "{",
            "    specie",
            "    {",
            "        molWeight       18;",
            "    }",
            "    equationOfState",
            "    {",
            f"        rho             {_fmt(rho_star)};",
            "    }",
            "    thermodynamics",
            "    {",
            "        Cv              1;",
            "        Hf              0;",
            "    }",
            "    transport",
            "    {",
            f"        mu              {_fmt(mu_star)};",
            "        Pr              1;",
            "    }",
            "}",
            "",
        ]
    )


def _format_phase_turbulence_properties(phase_name: str) -> str:
    """Generate ``constant/turbulenceProperties.<phase>`` (laminar)."""
    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            f"    object      turbulenceProperties.{phase_name};",
            "}",
            "",
            "simulationType  laminar;",
            "",
        ]
    )


def _virtual_mass_block(model: str, cvm: float | None) -> list[str]:
    """Return the ``virtualMass`` pair entry for a stock ESI v2306 virtual-mass model."""
    if model == "none":
        return []
    if model == "Lamb":
        return [
            "    (RBC in plasma)",
            "    {",
            "        type            Lamb;",
            "    }",
        ]
    return [
        "    (RBC in plasma)",
        "    {",
        "        type            constantCoefficient;",
        f"        Cvm             {_fmt(float(cvm))};",
        "    }",
    ]


def _wall_lubrication_block(model: str, coeffs: dict[str, float]) -> list[str]:
    """Return the ``wallLubrication`` pair entry for a stock ESI v2306 wall-lubrication model."""
    if model == "none":
        return []
    if model == "Antal":
        return [
            "    (RBC in plasma)",
            "    {",
            "        type            Antal;",
            f"        Cw1             {_fmt(coeffs['Cw1'])};",
            f"        Cw2             {_fmt(coeffs['Cw2'])};",
            "    }",
        ]
    if model == "Frank":
        return [
            "    (RBC in plasma)",
            "    {",
            "        type            Frank;",
            f"        Cwd             {_fmt(coeffs['Cwd'])};",
            f"        Cwc             {_fmt(coeffs['Cwc'])};",
            f"        p               {_fmt(coeffs['p'])};",
            "    }",
        ]
    return [
        "    (RBC in plasma)",
        "    {",
        "        type            Tomiyama;",
        f"        Cwd             {_fmt(coeffs['Cwd'])};",
        "    }",
    ]


def _format_phase_properties(
    *,
    Re: float,
    rho_ratio: float,
    mu_ratio: float,
    d_p_star: float,
    drag_model: str,
    lift_model: str,
    residual_re: float,
    lift_cl: float | None,
    wall_lubrication_model: str = "none",
    wall_lubrication_coeffs: dict[str, float] | None = None,
    virtual_mass_model: str = "none",
    cvm: float | None = None,
) -> str:
    """Generate ``constant/phaseProperties`` for ESI v2306 ``reactingTwoPhaseEulerFoam``.

    ``wall_lubrication_model`` / ``virtual_mass_model`` select stock ESI v2306 closures
    (wall lubrication: ``none``/``Antal``/``Frank``/``Tomiyama``; virtual mass:
    ``none``/``constantCoefficient``/``Lamb``). Coefficient values are never defaulted here:
    a model that needs coefficients raises unless they are given, so a case cannot silently run
    with invented numbers.
    """
    if drag_model not in SUPPORTED_DRAG_MODELS:
        raise ValueError(
            f"Unsupported drag_model '{drag_model}'; expected one of {SUPPORTED_DRAG_MODELS}"
        )
    if lift_model not in SUPPORTED_LIFT_MODELS:
        raise ValueError(
            f"Unsupported lift_model '{lift_model}'; expected one of {SUPPORTED_LIFT_MODELS}"
        )
    if wall_lubrication_model not in SUPPORTED_WALL_LUBRICATION_MODELS:
        raise ValueError(
            f"Unsupported wall_lubrication_model '{wall_lubrication_model}'; "
            f"expected one of {SUPPORTED_WALL_LUBRICATION_MODELS}"
        )
    if virtual_mass_model not in SUPPORTED_VIRTUAL_MASS_MODELS:
        raise ValueError(
            f"Unsupported virtual_mass_model '{virtual_mass_model}'; "
            f"expected one of {SUPPORTED_VIRTUAL_MASS_MODELS}"
        )
    wl = dict(wall_lubrication_coeffs or {})
    if wall_lubrication_model == "Antal":
        missing = [k for k in ("Cw1", "Cw2") if k not in wl]
        if missing:
            raise ValueError(f"Antal wall lubrication requires {missing} (no v2306 defaults exist)")
    elif wall_lubrication_model == "Frank":
        missing = [k for k in ("Cwd", "Cwc", "p") if k not in wl]
        if missing:
            raise ValueError(f"Frank wall lubrication requires {missing}")
    elif wall_lubrication_model == "Tomiyama":
        if "Cwd" not in wl:
            raise ValueError("Tomiyama wall lubrication requires Cwd (a length in v2306)")
    if virtual_mass_model == "constantCoefficient" and cvm is None:
        raise ValueError("constantCoefficient virtual mass requires Cvm (no v2306 default exists)")

    if lift_model == "LegendreMagnaudet":
        lift_block = [
            "    (RBC in plasma)",
            "    {",
            "        type            LegendreMagnaudet;",
            f"        residualRe      {_fmt(residual_re)};",
            "    }",
        ]
    elif lift_model == "constantCoefficient":
        if lift_cl is None or not math.isfinite(lift_cl):
            raise ValueError("lift_cl must be a finite float for constantCoefficient lift")
        lift_block = [
            "    (RBC in plasma)",
            "    {",
            "        type            constantCoefficient;",
            f"        Cl              {_fmt(lift_cl)};",
            "    }",
        ]
    else:
        lift_block = [
            "    (RBC in plasma)",
            "    {",
            "        type            none;",
            "    }",
        ]

    return "\n".join(
        [
            "FoamFile",
            "{",
            "    version     2.0;",
            "    format      ascii;",
            "    class       dictionary;",
            "    object      phaseProperties;",
            "}",
            "",
            "// Star units: rho_plasma* = 1, h1* = 1, U_mean* = 1",
            f"// Re = {_fmt(Re)}, rho_RBC/rho_p = {_fmt(rho_ratio)}, "
            f"mu_RBC/mu_p = {_fmt(mu_ratio)}, d_p/h1 = {_fmt(d_p_star)}",
            "type            basicTwoPhaseSystem;",
            "",
            "phases          (RBC plasma);",
            "",
            "RBC",
            "{",
            "    type            purePhaseModel;",
            "    diameterModel   constant;",
            "    constantCoeffs",
            "    {",
            f"        d               {_fmt(d_p_star)};",
            "    }",
            "",
            "    residualAlpha   1e-6;",
            "}",
            "",
            "plasma",
            "{",
            "    type            purePhaseModel;",
            "    diameterModel   constant;",
            "    constantCoeffs",
            "    {",
            f"        d               {_fmt(d_p_star)};",
            "    }",
            "",
            "    residualAlpha   1e-6;",
            "}",
            "",
            "// noBlending (TypeName 'none') keeps plasma strictly continuous at all phi",
            "blending",
            "{",
            "    default",
            "    {",
            "        type            none;",
            "        continuousPhase plasma;",
            "    }",
            "}",
            "",
            "surfaceTension",
            "(",
            ");",
            "",
            "aspectRatio",
            "(",
            ");",
            "",
            "drag",
            "(",
            "    (RBC in plasma)",
            "    {",
            f"        type            {drag_model};",
            f"        residualRe      {_fmt(residual_re)};",
            "        swarmCorrection",
            "        {",
            "            type        none;",
            "        }",
            "    }",
            ");",
            "",
            "virtualMass",
            "(",
            *_virtual_mass_block(virtual_mass_model, cvm),
            ");",
            "",
            "heatTransfer",
            "(",
            ");",
            "",
            "lift",
            "(",
            *lift_block,
            ");",
            "",
            "wallLubrication",
            "(",
            *_wall_lubrication_block(wall_lubrication_model, wl),
            ");",
            "",
            "turbulentDispersion",
            "(",
            ");",
            "",
            "phaseTransfer",
            "(",
            ");",
            "",
            "// Allow gauge pressure p = 0 at outlet without clipping negative step pressures",
            "pMin            -1e12;",
            "",
        ]
    )


def _format_fv_schemes_lift(local_euler: bool = False, wall_dist: bool = False) -> str:
    """Generate ``system/fvSchemes`` for ESI v2306 ``reactingTwoPhaseEulerFoam``.

    ``local_euler`` selects ``localEuler`` (local time stepping), which v2306 wires into
    ``reactingTwoPhaseEulerFoam`` via ``createRDeltaT.H``/``setRDeltaT.H`` and is the intended
    pseudo-steady mode for a transient code (plan/CORPUS_PLAN.md §6.1).
    """
    ddt = "localEuler" if local_euler else "Euler"
    # Wall-lubrication models call wallDist::New(mesh), which reads fvSchemes.wallDist.method.
    wall_dist_block = ["", "wallDist", "{", "    method          meshWave;", "}"] if wall_dist else []
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
            f"    default         {ddt};",
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
            "    div(phi,alpha.RBC)              Gauss vanLeer;",
            "    div(phir,alpha.RBC)             Gauss vanLeer;",
            '    "div\\(alphaRhoPhi.*,U.*\\)"      Gauss limitedLinearV 1;',
            '    "div\\(phi.*,U.*\\)"              Gauss limitedLinearV 1;',
            '    "div\\(alphaRhoPhi.*,(h|e).*\\)"  Gauss limitedLinear 1;',
            '    "div\\(alphaRhoPhi.*,K.*\\)"      Gauss limitedLinear 1;',
            '    "div\\(alphaPhi.*,p\\)"           Gauss limitedLinear 1;',
            '    "div\\(\\(\\(\\(alpha.*\\*thermo:rho.*\\)\\*nuEff.*\\)\\*dev2\\(T\\(grad\\(U.*\\)\\)\\)\\)\\)" Gauss linear;',
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
            *wall_dist_block,
            "",
        ]
    )


def _format_fv_solution_lift(
    local_euler: bool = False,
    max_co: float = 0.5,
    max_delta_t: float = 0.05,
    rdelta_t_smoothing: float = 0.02,
) -> str:
    """Generate ``system/fvSolution`` for isothermal ``reactingTwoPhaseEulerFoam`` (PIMPLE).

    With ``local_euler`` the PIMPLE dictionary must carry the ``setRDeltaT.H`` controls
    (``maxCo``, ``maxDeltaT``, ``rDeltaTSmoothingCoeff``), because the local time step is built
    from them inside the time loop.
    """
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
            "        nAlphaCorr      1;",
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
            '    "U.*"',
            "    {",
            "        solver          smoothSolver;",
            "        smoother        symGaussSeidel;",
            "        tolerance       1e-7;",
            "        relTol          0;",
            "        minIter         1;",
            "    }",
            "",
            '    "e.*"',
            "    {",
            "        solver          smoothSolver;",
            "        smoother        symGaussSeidel;",
            "        tolerance       1e-8;",
            "        relTol          0;",
            "        minIter         1;",
            "    }",
            "}",
            "",
            "PIMPLE",
            "{",
            "    nOuterCorrectors            2;",
            "    nCorrectors                 2;",
            "    nNonOrthogonalCorrectors    1;",
            "    nEnergyCorrectors           0;",
            *(
                [
                    f"    maxCo                       {_fmt(max_co)};",
                    f"    maxDeltaT                   {_fmt(max_delta_t)};",
                    f"    rDeltaTSmoothingCoeff       {_fmt(rdelta_t_smoothing)};",
                ]
                if local_euler
                else []
            ),
            "}",
            "",
            "relaxationFactors",
            "{",
            "    equations",
            "    {",
            '        "U.*"           0.7;',
            '        "U.*Final"      1.0;',
            '        ".*"            1;',
            "    }",
            "}",
            "",
        ]
    )


def _format_control_dict_lift(
    mesh: MeshSpec,
    *,
    end_time: float,
    delta_t: float,
    max_co: float,
    max_delta_t: float,
) -> str:
    """Generate ``system/controlDict`` for transient-to-steady ``reactingTwoPhaseEulerFoam``."""
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
        "application     reactingTwoPhaseEulerFoam;",
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
        "        fields              (alpha.RBC U.plasma U.RBC p);",
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
            "        fields              (alpha.RBC alpha.plasma U.plasma U.RBC p);",
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


def _format_run_case_m2b_lift_sh() -> str:
    """Return bash runner script for an M2b-B ``reactingTwoPhaseEulerFoam`` case."""
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Run a generated M2b-B nondimensional OpenFOAM v2306 reactingTwoPhaseEulerFoam case:",
            "#   blockMesh -> checkMesh -> reactingTwoPhaseEulerFoam -> postProcess",
            "# Exits non-zero on mesh errors, solver failure, NaN/Inf, or incomplete run.",
            "",
            "set -euo pipefail",
            "",
            "check_m2b_lift_log() {",
            '    local log_file="$1"',
            "    local filtered",
            '    filtered="$(grep -v -E \'trapFpe:|FOAM_SIGFPE\' "$log_file" || true)"',
            '    if printf \'%s\\n\' "$filtered" | grep -Eqi "FOAM FATAL|solution singularity|Floating point exception \\(core dumped\\)"; then',
            '        echo "ERROR: reactingTwoPhaseEulerFoam log shows fatal error or crash" >&2',
            '        printf \'%s\\n\' "$filtered" | grep -Ei "FOAM FATAL|solution singularity|Floating point exception \\(core dumped\\)" | head -n 20 >&2 || true',
            "        return 4",
            "    fi",
            '    if printf \'%s\\n\' "$filtered" | grep -Eq "\\bsigFpe\\b"; then',
            '        echo "ERROR: reactingTwoPhaseEulerFoam log shows sigFpe crash" >&2',
            '        printf \'%s\\n\' "$filtered" | grep -E "\\bsigFpe\\b" | head -n 20 >&2 || true',
            "        return 4",
            "    fi",
            '    if printf \'%s\\n\' "$filtered" | grep -E "(Solving for|continuity errors|volume fraction|Phase-sum)" | grep -Eqi "(^|[^A-Za-z_])[-+]?(nan|inf)([^A-Za-z_]|$)"; then',
            '        echo "ERROR: reactingTwoPhaseEulerFoam log shows NaN/Inf divergence" >&2',
            '        printf \'%s\\n\' "$filtered" | grep -E "(Solving for|continuity errors|volume fraction|Phase-sum)" | grep -Ei "(^|[^A-Za-z_])[-+]?(nan|inf)([^A-Za-z_]|$)" | head -n 20 >&2 || true',
            "        return 4",
            "    fi",
            "    return 0",
            "}",
            "",
            'if [ "${1:-}" = "--check-log" ]; then',
            '    check_m2b_lift_log "$2"',
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
            'echo "=== [3/4] reactingTwoPhaseEulerFoam ==="',
            "if ! reactingTwoPhaseEulerFoam > log.reactingTwoPhaseEulerFoam 2>&1; then",
            '    echo "ERROR: reactingTwoPhaseEulerFoam exited non-zero" >&2',
            "    tail -n 40 log.reactingTwoPhaseEulerFoam >&2 || true",
            "    exit 3",
            "fi",
            "",
            'check_m2b_lift_log log.reactingTwoPhaseEulerFoam || exit 4',
            "",
            'if ! grep -q "^End$" log.reactingTwoPhaseEulerFoam; then',
            '    echo "ERROR: reactingTwoPhaseEulerFoam did not reach End" >&2',
            "    tail -n 25 log.reactingTwoPhaseEulerFoam >&2 || true",
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
            'echo "=== DONE: reactingTwoPhaseEulerFoam reached End ==="',
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Public case writer & 20-anchor M2b-B design generator
# ---------------------------------------------------------------------------

def write_m2b_lift_case(
    case_dir: Path | str,
    params: Mapping[str, Any],
    level: int = 1,
    end_time: float | None = None,
    delta_t: float = 1e-4,
    max_co: float = 0.5,
    max_delta_t: float = 0.05,
) -> MeshSpec:
    """Write a complete nondimensional OpenFOAM ESI v2306 ``reactingTwoPhaseEulerFoam`` case.

    Parameters match ``cfd.gen.m2b.write_m2b_case`` plus lift options:
    - ``lift_model``: ``'LegendreMagnaudet'`` (default), ``'constantCoefficient'``, or ``'none'``
    - ``residual_re``: residual particle Reynolds number (default ``1e-3``)
    - ``lift_cl``: optional explicit ``Cl`` when ``lift_model == 'constantCoefficient'``
      (defaults to ``saffman_constant_cl(Re=Re, d_p_star=d_p_star)``)
    """
    family = str(params["family"]).upper()
    Re = float(params["Re"])
    phi = float(params["phi"])
    if Re <= 0.0:
        raise ValueError(f"Re must be > 0, got {Re}")
    if not (0.0 < phi < 1.0):
        raise ValueError(f"Inlet RBC volume fraction phi must be in (0, 1), got {phi}")

    rho_ratio = float(params.get("rho_ratio", WU2015_RHO_RATIO))
    if rho_ratio <= 0.0:
        raise ValueError(f"rho_ratio must be > 0, got {rho_ratio}")
    if "mu_ratio" in params and params["mu_ratio"] is not None:
        mu_ratio = float(params["mu_ratio"])
    else:
        mu_ratio = wu2015_mu_ratio(phi)
    if mu_ratio <= 0.0:
        raise ValueError(f"mu_ratio must be > 0, got {mu_ratio}")

    d_p_star = float(params["d_p_star"])
    if d_p_star <= 0.0:
        raise ValueError(f"d_p_star must be > 0, got {d_p_star}")

    drag_model = str(params.get("drag_model", "WenYu"))
    lift_model = str(params.get("lift_model", DEFAULT_LIFT_MODEL))
    residual_re = float(params.get("residual_re", DEFAULT_RESIDUAL_RE))
    inlet_profile = str(params.get("inlet_profile", "uniform"))
    wall_lubrication_model = str(params.get("wall_lubrication_model", "none"))
    virtual_mass_model = str(params.get("virtual_mass_model", "none"))
    cvm = params.get("cvm")
    wall_lubrication_coeffs = {
        key: float(params[key])
        for key in ("Cw1", "Cw2", "Cwd", "Cwc")
        if params.get(key) is not None
    }
    if params.get("wl_p") is not None:
        wall_lubrication_coeffs["p"] = float(params["wl_p"])
    local_euler = bool(params.get("local_euler", False))

    lift_cl: float | None = None
    if lift_model == "constantCoefficient":
        if "lift_cl" in params and params["lift_cl"] is not None:
            lift_cl = float(params["lift_cl"])
        else:
            lift_cl = saffman_constant_cl(Re=Re, d_p_star=d_p_star)

    geom_params: dict[str, Any] = dict(params.get("geometry", params))
    mesh = build_m2b_geometry(family, geom_params, level=level)

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
    mu_plasma_star = 1.0 / Re
    mu_rbc_star = mu_ratio / Re

    root = Path(case_dir)
    for sub in ("0", "constant", "system"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / "system" / "blockMeshDict").write_text(format_block_mesh_dict(mesh), encoding="utf-8")
    # With localEuler the solver ignores adjustTimeStep/maxCo in controlDict and advances the
    # reported time by deltaT, while rDeltaT (from maxCo in the PIMPLE dict) drives the equations.
    # deltaT therefore has to be ~maxDeltaT, otherwise endTime needs deltaT=1e-4-sized steps.
    delta_t_eff = max(max_delta_t, delta_t) if local_euler else delta_t
    (root / "system" / "controlDict").write_text(
        _format_control_dict_lift(
            mesh,
            end_time=end_time_val,
            delta_t=delta_t_eff,
            max_co=max_co,
            max_delta_t=max_delta_t,
        ),
        encoding="utf-8",
    )
    (root / "system" / "fvSchemes").write_text(
        _format_fv_schemes_lift(
            local_euler=local_euler, wall_dist=wall_lubrication_model != "none"
        ),
        encoding="utf-8",
    )
    (root / "system" / "fvSolution").write_text(
        _format_fv_solution_lift(
            local_euler=local_euler, max_co=max_co, max_delta_t=max_delta_t
        ),
        encoding="utf-8",
    )

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
    (root / "0" / "p").write_text(_format_p_field(mesh), encoding="utf-8")
    (root / "0" / "T.RBC").write_text(_format_t_field(mesh, "RBC"), encoding="utf-8")
    (root / "0" / "T.plasma").write_text(_format_t_field(mesh, "plasma"), encoding="utf-8")

    (root / "constant" / "phaseProperties").write_text(
        _format_phase_properties(
            Re=Re,
            rho_ratio=rho_ratio,
            mu_ratio=mu_ratio,
            d_p_star=d_p_star,
            drag_model=drag_model,
            lift_model=lift_model,
            residual_re=residual_re,
            lift_cl=lift_cl,
            wall_lubrication_model=wall_lubrication_model,
            wall_lubrication_coeffs=wall_lubrication_coeffs,
            virtual_mass_model=virtual_mass_model,
            cvm=None if cvm is None else float(cvm),
        ),
        encoding="utf-8",
    )
    (root / "constant" / "thermophysicalProperties.plasma").write_text(
        _format_thermophysical_properties(
            phase_name="plasma", rho_star=1.0, mu_star=mu_plasma_star
        ),
        encoding="utf-8",
    )
    (root / "constant" / "thermophysicalProperties.RBC").write_text(
        _format_thermophysical_properties(
            phase_name="RBC", rho_star=rho_ratio, mu_star=mu_rbc_star
        ),
        encoding="utf-8",
    )
    (root / "constant" / "turbulenceProperties.plasma").write_text(
        _format_phase_turbulence_properties("plasma"), encoding="utf-8"
    )
    (root / "constant" / "turbulenceProperties.RBC").write_text(
        _format_phase_turbulence_properties("RBC"), encoding="utf-8"
    )
    (root / "constant" / "g").write_text(_format_g(), encoding="utf-8")

    run_script = root / "run_case.sh"
    run_script.write_text(_format_run_case_m2b_lift_sh(), encoding="utf-8")
    os.chmod(
        run_script,
        run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )

    total_cells = sum(nx * ny * nz for _, (nx, ny, nz), _ in mesh.blocks)
    meta = {
        "run_id": params.get("run_id"),
        "tier": "m2b-b",
        "solver": "reactingTwoPhaseEulerFoam",
        "family": family,
        "level": level,
        "Re": Re,
        "phi": phi,
        "alpha_RBC_inlet": alpha_rbc,
        "alpha_plasma_inlet": alpha_plasma,
        "rho_ratio": rho_ratio,
        "mu_ratio": mu_ratio,
        "d_p_star": d_p_star,
        "mu_plasma_star": mu_plasma_star,
        "mu_rbc_star": mu_rbc_star,
        "nu_plasma_star": mu_plasma_star,
        "nu_rbc_star": mu_rbc_star / rho_ratio,
        "drag_model": drag_model,
        "lift_model": lift_model,
        "residual_re": residual_re,
        "lift_cl": lift_cl,
        "cl_saffman_wu_nom": round(saffman_constant_cl(Re=Re, d_p_star=d_p_star), 6),
        "cl_lm_nom": round(
            legendre_magnaudet_cl_star(
                Re=Re,
                d_p_star=d_p_star,
                gdot_star=DEFAULT_GDOT_STAR_NOM,
                residual_re=residual_re,
            ),
            6,
        ),
        "rbc_viscosity_model": "constant effective mu (Wu 2015 Yeleswarapu fallback, stress_factor_half=True)",
        "inlet_profile": inlet_profile,
        "end_time": end_time_val,
        "geometry": mesh.params,
        "n_blocks": len(mesh.blocks),
        "n_cells": total_cells,
        "sample_lines": [sl.name for sl in mesh.sample_lines],
    }
    (root / "params.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return mesh


def generate_m2b_lift_design(
    level: int = 1,
    lift_model: str = DEFAULT_LIFT_MODEL,
    residual_re: float = DEFAULT_RESIDUAL_RE,
) -> list[dict[str, Any]]:
    """Return the exact same 20 anchors as M2b-A configured for M2b-B (``reactingTwoPhaseEulerFoam``)."""
    base_records = generate_m2b_design(level=level)
    lift_records: list[dict[str, Any]] = []
    for rec in base_records:
        item = dict(rec)
        run_id = str(rec["run_id"])
        case_id = str(rec["case_id"])
        item["run_id"] = run_id.replace("M2B_", "M2BL_", 1) if run_id.startswith("M2B_") else f"M2BL_{run_id}"
        item["case_id"] = case_id.replace("M2B_", "M2BL_", 1) if case_id.startswith("M2B_") else f"M2BL_{case_id}"
        item["tier"] = "m2b-b"
        item["solver"] = "reactingTwoPhaseEulerFoam"
        item["lift_model"] = lift_model
        item["residual_re"] = residual_re
        if lift_model == "constantCoefficient":
            item["lift_cl"] = round(
                saffman_constant_cl(Re=float(rec["Re"]), d_p_star=float(rec["d_p_star"])),
                6,
            )
        item["geometry"] = dict(rec["geometry"])
        lift_records.append(item)
    assert len(lift_records) == 20, f"Expected 20 M2b-B anchors, got {len(lift_records)}"
    return lift_records


def write_m2b_lift_design_jsonl(output_path: Path | str, records: list[dict[str, Any]]) -> Path:
    """Write M2b-B design records to a JSONL file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return out


def main() -> None:
    """CLI entrypoint to generate ``campaign/m2b_lift_design.jsonl``."""
    parser = argparse.ArgumentParser(
        description="Generate 20-anchor M2b-B (reactingTwoPhaseEulerFoam + lift) design JSONL."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "campaign" / "m2b_lift_design.jsonl",
        help="Output JSONL path (default: campaign/m2b_lift_design.jsonl)",
    )
    parser.add_argument("--level", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument(
        "--lift-model",
        type=str,
        default=DEFAULT_LIFT_MODEL,
        choices=SUPPORTED_LIFT_MODELS,
    )
    parser.add_argument("--residual-re", type=float, default=DEFAULT_RESIDUAL_RE)
    args = parser.parse_args()

    records = generate_m2b_lift_design(
        level=args.level,
        lift_model=args.lift_model,
        residual_re=args.residual_re,
    )
    out = write_m2b_lift_design_jsonl(args.out, records)
    print(
        f"Wrote {len(records)} M2b-B anchors (level={args.level}, lift={args.lift_model}) to {out}"
    )


if __name__ == "__main__":
    main()
