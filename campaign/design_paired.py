"""Paired 4-variant design (V1 ``N``, V2 ``GN``, V3 ``EE-DL``, V4 ``EE-full``) for the 2D corpus.

Implements ``plan/CORPUS_PLAN.md`` §6 + amendments v2.1/v2.2/v2.3,
``plan/tasks/T7_mistral_archive_m2b.md``, and the team 2026-11-24 nominal wall-shear redesign (T10):

* Joint scrambled-Sobol design per family over ``(geometry ratios, h, tau_w, phi, mu_carrier)`` in
  **device space**: ``h in [80, 150] um`` (continuum limit ``d_RBC/h <= 0.1``, v2.1), nominal wall
  shear ``tau_w in [2, 150] Pa`` (log-uniform; covers W3 1-7 Pa, W2 15-45 Pa, W1 30-60 Pa, and the
  hemolysis range), ``phi`` split so that 12 designs per family fall in the device regime
  ``[0.01, 0.1]`` and 12 in the whole-blood range ``[0.1, 0.6]`` (v2.3), plus 3 designs pinned at
  clinical haematocrit (anaemic 0.30 / normal 0.445 / polycythaemic 0.55, ``knowledge/16`` §1).
* Mean inlet velocity ``U`` is derived self-consistently from ``tau_w = 6 mu_eff(phi, 3U/h) U / h``
  using the existing Quemada ``mu_eff``; any Sobol draw with ``Re = rho U h / mu_carrier > 60`` is
  rejected and the next Sobol point is taken so every design remains in the steady-laminar regime.
* Carrier viscosity is a **run parameter** (v2.3): plasma 0.96 mPa s for ``phi > 0.1``, a buffer sampled
  log-uniformly in 1.1-1.7 mPa s for ``phi <= 0.1`` (Bernardelli 2025 Table 3 via ``knowledge/16`` §4).
  ``mu_ref`` of a run is its own carrier, so the four variants of a design share ``Re``.
* Every record carries ``design_id`` and ``pair_id`` (= ``design_id``: the four variants of one design),
  ``variant``, the device scale (``h_um``, ``U_mps``, ``tau_w_nominal_pa``, ``mu_carrier``, ``rho``),
  ``d_rbc_over_h``, ``t_ref = h/U`` and ``tau_ref = mu_carrier U / h``. **No shear window is baked in**
  (v2.3): the window is a post-processing input, recomputable from the stored outlet profiles.
* V3/V4 records are emitted with ``solver: fenicsx`` for the team steady two-fluid solver (v2.2); they
  are never executed by ``campaign.run_batch`` (which skips them).

Two records of one design differ only in the constitutive physics, and share the mesh: the mesh is a
function of ``(family, geometry ratios, level)`` only, so identical geometry => identical ``blockMesh``.

``--per-family`` is designs **per haematocrit branch**: ``1`` gives the v2.1 pilot (one design per family,
24 runs, 12 of them V1/V2 in OpenFOAM), ``12`` gives the v2.3 main corpus (12 device-regime + 12 whole-blood
+ 3 pinned clinical designs per family + 2 G1 anchors = 164 designs, 656 runs).

CLI: ``python -m campaign.design_paired --per-family 1 --out campaign/paired_design_pilot.jsonl``
(pilot) or ``--per-family 12`` for the main batch (``campaign/paired_design.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc

from campaign import blood_rheology as rheo
from campaign.design_m1 import _family_specs  # single definition of the family geometry ranges

VARIANTS: tuple[tuple[str, str], ...] = (
    ("N", "simpleFoam"),
    ("GN", "simpleFoam"),
    ("EE-DL", "fenicsx"),
    ("EE-full", "fenicsx"),
)
FAMILIES = ("G2", "G3", "G4", "G5", "G6", "G7")
ANCHOR_FAMILIES = ("G1",)

H_UM_RANGE = (80.0, 150.0)  # d_RBC/h <= 0.1 with d_RBC = 8 um (v2.1 continuum limit)
TAU_W_PA_RANGE = (2.0, 150.0)  # log-uniform nominal wall shear [Pa]
RE_MAX = 60.0  # steady-laminar Reynolds-number rejection cap
PHI_DEVICE_RANGE = (0.01, 0.1)  # 12 designs/family (v2.3)
PHI_BLOOD_RANGE = (0.1, 0.6)  # 12 designs/family (v2.3)
PINNED_PHI = (0.30, 0.445, 0.55)  # anaemic / normal / polycythaemic (knowledge/16 §1)
D_RBC_UM = 8.0
RHO_PLASMA = 1027.0  # kg/m3, Wu 2015 / Sezer 2026
LEVEL = 2  # GCI-chosen production mesh for all six families (T3 resume, cfd/openfoam/gci_summary.md)
PINNED_PILOT_PHI = 0.445  # the pilot uses the normal-haematocrit design of every family


def _log_uniform(u: float, lo: float, hi: float) -> float:
    """Map ``u`` in [0, 1) to log-uniform on [lo, hi]."""
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


def velocity_from_nominal_tau_w(
    tau_w_pa: float,
    h_um: float,
    phi: float,
    mu_carrier: float,
) -> float:
    """Derive mean velocity ``U`` [m/s] from nominal wall shear ``tau_w = 6 mu_eff(phi, 3U/h) U / h``.

    With ``gdot_ref = 3 U / h``, ``tau_w = 2 * gdot_ref * quemada_mu(phi, gdot_ref, mu_carrier)``.
    For ``phi <= 0.6`` and ``tau_w >= 2 Pa`` the right-hand side is strictly increasing on the
    high-shear branch (``gdot_ref >= 10 s^-1``, well above the Quemada zero-shear singularity),
    giving a unique self-consistent ``U = tau_w * h / (6 * mu_eff)``.
    """
    if tau_w_pa <= 0.0:
        raise ValueError(f"tau_w_pa must be > 0, got {tau_w_pa}")
    if h_um <= 0.0:
        raise ValueError(f"h_um must be > 0, got {h_um}")
    h_m = h_um * 1e-6

    def residual(gdot: float) -> float:
        return 2.0 * gdot * rheo.quemada_mu(phi, gdot, mu_carrier) - tau_w_pa

    # Start safely on the increasing branch above the Quemada singularity (gdot_min < 8 s^-1 at phi=0.6).
    g_lo = 10.0
    while True:
        try:
            if residual(g_lo) <= 0.0:
                break
        except ValueError as exc:
            raise ValueError(
                f"Cannot bracket tau_w={tau_w_pa} Pa above Quemada singularity at phi={phi}"
            ) from exc
        g_lo *= 0.5
        if g_lo < 1e-6:
            raise ValueError(f"Failed to bracket lower shear rate for tau_w={tau_w_pa} Pa, phi={phi}")

    mu_inf = rheo.quemada_high_shear_mu(phi, mu_carrier)
    g_hi = max(100.0, tau_w_pa / (2.0 * mu_inf))
    while residual(g_hi) < 0.0:
        g_hi *= 2.0

    gdot_ref = float(brentq(residual, g_lo, g_hi, xtol=1e-12, rtol=1e-12))
    return gdot_ref * h_m / 3.0


class _SobolStream:
    """Deterministic chunked stream of scrambled-Sobol points (powers-of-2 draws, no warnings)."""

    def __init__(self, d: int, seed: int, chunk_pow: int = 4) -> None:
        self._sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        self._chunk_pow = chunk_pow
        self._buf: np.ndarray = np.empty((0, d), dtype=np.float64)
        self._pos = 0

    def next_row(self) -> np.ndarray:
        if self._pos >= len(self._buf):
            self._buf = self._sampler.random_base2(m=self._chunk_pow)
            self._pos = 0
        row = self._buf[self._pos]
        self._pos += 1
        return row


def _geometry(family: str, row: np.ndarray) -> dict[str, Any]:
    """Build the star-unit geometry dict for ``family`` from the Sobol row tail."""
    spec = next(s for s in _family_specs() if s.name == family)
    geom: dict[str, Any] = {"L_in": 5.0, "L_out_ratio": 10.0}
    for idx, (name, lo, hi, is_int) in enumerate(spec.geom_dims):
        u = float(row[idx])
        if is_int:
            n_choices = int(hi - lo) + 1
            geom[name] = min(int(hi), int(lo) + int(math.floor(u * n_choices)))
        else:
            geom[name] = round(lo + u * (hi - lo), 6)
    return geom


def _variants_for(
    family: str,
    design_id: str,
    pair_id: str,
    geom: dict[str, Any],
    h_um: float,
    u_mps: float,
    phi: float,
    mu_carrier: float,
    seed: int,
    tau_w_nominal_pa: float | None = None,
) -> list[dict[str, Any]]:
    """Return the four variant records of one design (same mesh, same Re, different physics)."""
    h_m = h_um * 1e-6
    gdot_ref = rheo.plane_poiseuille_mean_shear_rate(u_mps, h_m)
    mu_eff = rheo.quemada_mu(phi, gdot_ref, mu_carrier)
    carreau = rheo.carreau_params_from_quemada(phi, mu_carrier)
    Re = RHO_PLASMA * u_mps * h_m / mu_carrier
    tau_w = float(6.0 * mu_eff * u_mps / h_m if tau_w_nominal_pa is None else tau_w_nominal_pa)
    common: dict[str, Any] = {
        "design_id": design_id,
        "pair_id": pair_id,
        "family": family,
        "level": LEVEL,
        "tier": "paired",
        "phi": round(phi, 6),
        "h_um": round(h_um, 6),
        "U_mps": round(u_mps, 8),
        "tau_w_nominal_pa": round(tau_w, 6),
        "mu_carrier": round(mu_carrier, 10),
        "rho": RHO_PLASMA,
        "Re": round(Re, 8),
        "d_rbc_um": D_RBC_UM,
        "d_rbc_over_h": round(D_RBC_UM / h_um, 8),
        "t_ref": round(h_m / u_mps, 10),
        "tau_ref": round(mu_carrier * u_mps / h_m, 10),
        "gdot_ref": round(gdot_ref, 6),
        "mu_eff": round(mu_eff, 12),
        "geometry": geom,
        "seed": seed,
        "code_version_note": "filled by campaign.run_batch at run time",
    }
    out: list[dict[str, Any]] = []
    for variant, solver in VARIANTS:
        rec: dict[str, Any] = {
            **common,
            "run_id": f"P_{family}_{design_id}_{variant}",
            "variant": variant,
            "solver": solver,
        }
        if variant == "N":
            # V1: one Newtonian viscosity per design, matched at the mean shear rate of the flow.
            rec.update(
                {
                    "rheology": "Newtonian",
                    "viscosity_ratio": round(mu_eff / mu_carrier, 10),
                    "beta": None,
                    "Cu": None,
                    "n": None,
                    "a": None,
                }
            )
        elif variant == "GN":
            # V2: BirdCarreau derived from the verified Quemada law at this phi (knowledge/16 §3).
            rec.update(
                {
                    "rheology": "BirdCarreau",
                    "beta": round(carreau.beta, 8),
                    "Cu": round(carreau.lam * u_mps / h_m, 10),
                    "n": round(carreau.n, 8),
                    "a": carreau.a,
                    "viscosity_ratio": round(carreau.mu_inf / mu_carrier, 10),
                    "mu_inf": round(carreau.mu_inf, 12),
                    "mu_0": round(carreau.mu_0, 12),
                    "lambda_s": carreau.lam,
                    "carreau_fit_max_rel_err": round(carreau.max_rel_err, 8),
                    "carreau_fit_gdot_range": list(carreau.gdot_range),
                }
            )
        else:
            # V3/V4: run by the FEniCSx two-fluid solver; run_batch skips them (v2.2).
            rec.update(
                {
                    "rheology": "two-fluid",
                    "viscosity_ratio": None,
                    "beta": None,
                    "Cu": None,
                    "n": None,
                    "a": None,
                    "mu_ratio_rbc": round(1.0, 8),  # constant RBC-phase viscosity (v2.1)
                }
            )
        out.append(rec)
    return out


def _design_ids(n_family: int) -> list[str]:
    """Deterministic design ids: device-regime designs first, then whole-blood ones."""
    ids: list[str] = []
    for idx in range(n_family):
        ids.append(f"D{idx:03d}")
    for idx in range(n_family):
        ids.append(f"B{idx:03d}")
    return ids


def _geometry_mid(family: str) -> dict[str, Any]:
    """Mid-range geometry of a family (used by the 1-design pilot)."""
    spec = next(s for s in _family_specs() if s.name == family)
    geom: dict[str, Any] = {"L_in": 5.0, "L_out_ratio": 10.0}
    for name, lo, hi, is_int in spec.geom_dims:
        geom[name] = int(round(0.5 * (lo + hi))) if is_int else round(0.5 * (lo + hi), 6)
    return geom


def pilot_design(
    families: tuple[str, ...] = FAMILIES, seed: int = 20261124
) -> list[dict[str, Any]]:
    """One representative design per family: mid-range geometry, h/tau_w mid-range, normal Hct.

    The pilot must measure seconds per run for V1/V2 on a mesh and a haematocrit that are typical,
    so it uses the geometric mean nominal wall shear of the sampled range, the midpoint height,
    the normal clinical haematocrit (plasma carrier) and the mid-range geometry of each family.
    """
    h_um = 0.5 * (H_UM_RANGE[0] + H_UM_RANGE[1])
    tau_w = math.sqrt(TAU_W_PA_RANGE[0] * TAU_W_PA_RANGE[1])
    u_mps = velocity_from_nominal_tau_w(tau_w, h_um, PINNED_PILOT_PHI, rheo.MU_PLASMA)
    records: list[dict[str, Any]] = []
    for fam_idx, family in enumerate(families):
        records.extend(
            _variants_for(
                family,
                "PILOT",
                f"{family}_PILOT",
                _geometry_mid(family),
                h_um,
                u_mps,
                PINNED_PILOT_PHI,
                rheo.MU_PLASMA,
                seed + 7919 * (fam_idx + 1),
                tau_w_nominal_pa=tau_w,
            )
        )
    return records


def generate_design(
    per_family: int = 1,
    *,
    seed: int = 20261124,
    families: tuple[str, ...] = FAMILIES,
    anchor_families: tuple[str, ...] = ANCHOR_FAMILIES,
) -> list[dict[str, Any]]:
    """Generate the paired design JSONL records.

    ``per_family`` is the number of Sobol designs **per haematocrit branch** (v2.1: pilot 1,
    v2.3 main corpus: 12). With ``per_family == 1`` the design is the single pilot design per
    family (v2.1: 24 runs, 12 of them the OpenFOAM V1/V2 runs).
    """
    if per_family <= 0:
        raise ValueError(f"per_family must be positive, got {per_family}")

    if per_family == 1:
        # v2.1 pilot: exactly one design per family G2-G7 (24 runs, 12 of them V1/V2 in OpenFOAM).
        return pilot_design(families, seed)

    records: list[dict[str, Any]] = []

    for fam_idx, family in enumerate(families):
        spec = next(s for s in _family_specs() if s.name == family)
        n_geom = len(spec.geom_dims)
        fam_seed = seed + 7919 * (fam_idx + 1)
        stream = _SobolStream(d=n_geom + 4, seed=fam_seed)

        design_ids = _design_ids(per_family)
        for branch, (phi_lo, phi_hi) in enumerate((PHI_DEVICE_RANGE, PHI_BLOOD_RANGE)):
            accepted = 0
            while accepted < per_family:
                row = stream.next_row()
                geom_u = row[:n_geom]
                u_h, u_tau, u_phi, u_mu = row[n_geom : n_geom + 4]
                h_um = H_UM_RANGE[0] + float(u_h) * (H_UM_RANGE[1] - H_UM_RANGE[0])
                tau_w = _log_uniform(float(u_tau), *TAU_W_PA_RANGE)
                phi = phi_lo + float(u_phi) * (phi_hi - phi_lo)
                mu_carrier = rheo.carrier_viscosity(phi, float(u_mu))
                u_mps = velocity_from_nominal_tau_w(tau_w, h_um, phi, mu_carrier)
                Re = RHO_PLASMA * u_mps * (h_um * 1e-6) / mu_carrier
                if Re > RE_MAX:
                    continue
                geom = _geometry(family, geom_u)
                design_id = design_ids[branch * per_family + accepted]
                records.extend(
                    _variants_for(
                        family,
                        design_id,
                        f"{family}_{design_id}",
                        geom,
                        h_um,
                        u_mps,
                        phi,
                        mu_carrier,
                        fam_seed,
                        tau_w_nominal_pa=tau_w,
                    )
                )
                accepted += 1

        # Pinned clinical haematocrit designs, own Sobol draw, carrier = plasma (whole blood).
        pin_stream = _SobolStream(d=n_geom + 2, seed=fam_seed + 31)
        for pin_idx, phi in enumerate(PINNED_PHI):
            while True:
                row = pin_stream.next_row()
                u_h, u_tau = row[n_geom], row[n_geom + 1]
                h_um = H_UM_RANGE[0] + float(u_h) * (H_UM_RANGE[1] - H_UM_RANGE[0])
                tau_w = _log_uniform(float(u_tau), *TAU_W_PA_RANGE)
                u_mps = velocity_from_nominal_tau_w(tau_w, h_um, float(phi), rheo.MU_PLASMA)
                Re = RHO_PLASMA * u_mps * (h_um * 1e-6) / rheo.MU_PLASMA
                if Re <= RE_MAX:
                    break
            geom = _geometry(family, row[:n_geom])
            design_id = f"C{pin_idx:03d}"
            records.extend(
                _variants_for(
                    family,
                    design_id,
                    f"{family}_{design_id}",
                    geom,
                    h_um,
                    u_mps,
                    float(phi),
                    rheo.MU_PLASMA,
                    fam_seed + 13,
                    tau_w_nominal_pa=tau_w,
                )
            )

    # G1 straight-channel anchors: one device-regime and one whole-blood design.
    for family in anchor_families:
        fam_seed = seed + 7919 * 97
        anchor_stream = _SobolStream(d=3, seed=fam_seed)
        for idx, phi_range in enumerate((PHI_DEVICE_RANGE, PHI_BLOOD_RANGE)):
            while True:
                u_h, u_tau, u_phi = anchor_stream.next_row()
                h_um = H_UM_RANGE[0] + float(u_h) * (H_UM_RANGE[1] - H_UM_RANGE[0])
                tau_w = _log_uniform(float(u_tau), *TAU_W_PA_RANGE)
                phi = phi_range[0] + float(u_phi) * (phi_range[1] - phi_range[0])
                mu_carrier = rheo.carrier_viscosity(phi, 0.5)
                u_mps = velocity_from_nominal_tau_w(tau_w, h_um, phi, mu_carrier)
                Re = RHO_PLASMA * u_mps * (h_um * 1e-6) / mu_carrier
                if Re <= RE_MAX:
                    break
            geom = {"L_ratio": 20.0}
            design_id = f"A{idx:03d}"
            records.extend(
                _variants_for(
                    family,
                    design_id,
                    f"{family}_{design_id}",
                    geom,
                    h_um,
                    u_mps,
                    phi,
                    mu_carrier,
                    fam_seed,
                    tau_w_nominal_pa=tau_w,
                )
            )
    return records


def _mesh_signature(rec: dict[str, Any]) -> str:
    """Geometry + level: identical for the four variants of a design (same mesh)."""
    return json.dumps({"family": rec["family"], "level": rec["level"], "geometry": rec["geometry"]}, sort_keys=True)


def write_design_jsonl(output_path: Path | str, records: list[dict[str, Any]]) -> Path:
    """Write design records to a JSONL file (one JSON object per line)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return out


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count designs and runs per family, variant, and report Re / tau_w distributions."""
    per_variant: dict[str, int] = {}
    per_family: dict[str, int] = {}
    designs: dict[str, set[str]] = {}
    unique_designs: dict[str, dict[str, Any]] = {}
    for rec in records:
        per_variant[rec["variant"]] = per_variant.get(rec["variant"], 0) + 1
        per_family[rec["family"]] = per_family.get(rec["family"], 0) + 1
        designs.setdefault(rec["family"], set()).add(str(rec["design_id"]))
        unique_designs.setdefault(str(rec["pair_id"]), rec)
    re_vals = np.array([float(r["Re"]) for r in unique_designs.values()], dtype=np.float64)
    tw_vals = np.array([float(r["tau_w_nominal_pa"]) for r in unique_designs.values()], dtype=np.float64)
    return {
        "runs": len(records),
        "n_designs": len(unique_designs),
        "designs": {fam: len(ids) for fam, ids in designs.items()},
        "per_variant": per_variant,
        "per_family": per_family,
        "openfoam_runs": sum(1 for r in records if r["solver"] == "simpleFoam"),
        "fenicsx_runs": sum(1 for r in records if r["solver"] == "fenicsx"),
        "Re": {
            "min": round(float(np.min(re_vals)), 4) if re_vals.size else None,
            "median": round(float(np.median(re_vals)), 4) if re_vals.size else None,
            "max": round(float(np.max(re_vals)), 4) if re_vals.size else None,
        },
        "tau_w_nominal_pa": {
            "min": round(float(np.min(tw_vals)), 4) if tw_vals.size else None,
            "median": round(float(np.median(tw_vals)), 4) if tw_vals.size else None,
            "max": round(float(np.max(tw_vals)), 4) if tw_vals.size else None,
        },
    }


def main() -> None:
    """CLI entrypoint for the paired design."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate the paired 4-variant design JSONL.")
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "paired_design.jsonl",
        help="Output JSONL path (default: campaign/paired_design.jsonl)",
    )
    parser.add_argument(
        "--per-family",
        type=int,
        default=1,
        help="Sobol designs per haematocrit branch per family (pilot 1, v2.3 main corpus 12)",
    )
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    records = generate_design(per_family=args.per_family, seed=args.seed)
    out = write_design_jsonl(args.out, records)
    print(json.dumps({**summarise(records), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
