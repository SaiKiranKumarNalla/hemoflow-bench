"""Resumable parallel OpenFOAM batch runner, star-unit field extractor, and GCS uploader.

Runs on a multi-core VM (e.g. n2-highcpu-8 SPOT):
1. Reads a design ``.jsonl`` (with optional ``--level`` and ``--ids`` filters).
2. Syncs the existing ``manifest.jsonl`` from GCS (if uploading) and skips any ``run_id``
   already recorded with ``status == "ok"``.
3. Runs up to ``N = nproc`` OpenFOAM cases in parallel (one core per case) via
   ``cfd.gen.case.write_case`` and ``run_case.sh``.
4. Extracts nondimensional star-unit fields (cell centres ``x*``, ``y*``, cell volumes,
   ``U*``, viscous-scaled ``p* = Re * p_OF``, ``gdot*`` from ``grad(U)``, ``tau*``),
   resamples onto a fixed per-family grid with an exact fluid mask, stores the
   outlet-section profile (for recomputing any shear window later), computes star-unit
   scalars (``dp_star``, ``xr_over_h1``, ``max_wall_tau_star``, ``f_window``), and
   writes compressed ``<run_id>.npz`` + one JSON line in ``manifest.jsonl``.
5. Uploads ``<run_id>.npz`` to ``gs://<corpus-bucket>/corpus/v1/runs/<tier>/<family>/``
   and ``manifest.jsonl`` to ``gs://<corpus-bucket>/corpus/v1/``.
6. Deletes the local OpenFOAM case directory only after successful upload; failed cases
   are kept locally for inspection.

Nondimensional conventions follow ``knowledge/11-math-model.md`` §3 and ``plan/CORPUS_PLAN.md`` §1:
- Length scale: ``h1* = 1`` (or inlet diameter ``d1* = 1`` for axisymmetric ``KG``).
- Velocity scale: ``U_mean* = 1``.
- Kinematic viscosity in OpenFOAM: ``nu* = mu_r(gdot*) / Re``.
- Viscous-scaled star pressure: ``p* = p h1 / (mu_inf U) = Re * p_OF``.
- Star shear stress: ``tau* = mu_r(gdot*) * gdot*``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from cfd.gen.case import write_case
from cfd.gen.geometry import MeshSpec, SampleLine, build_geometry
from cfd.gen.m2b import build_m2b_geometry, write_m2b_case, wu2015_mu_ratio
from cfd.gen.m2b_lift import write_m2b_lift_case
from rbc.postprocess import reattachment_length, trapezoid_weights, window_fraction

DEFAULT_BUCKET = "gs://<corpus-bucket>"
DEFAULT_PREFIX = "corpus/v1"
DEFAULT_TIER = "m1"
DEFAULT_MANIFEST_NAME = "manifest.jsonl"

# Code files whose contents define the corpus provenance hash (T7 Part 1).
CODE_VERSION_PATTERNS: tuple[str, ...] = ("cfd/gen/*.py", "cfd/gen/run_case.sh", "campaign/*.py")
# Relative change above which a re-run scalar is reported as non-reproducible (T7 Part 1).
REPRODUCIBILITY_REL_TOL = 1e-6

# Convergence criteria (plan/DECISIONS.md, 2026-11-24):
# (a) all initial residuals < 1e-5 ("residual"), OR
# (b) all initial residuals < 1e-3, continuity error (sum local) < 1e-8, AND both
#     dp* and outlet-centreline velocity change by < 1e-5 (relative) over last 500 iters ("qoi").
DEFAULT_RESIDUAL_TOL = 1e-5
QOI_MAX_RESIDUAL = 1e-3
QOI_MAX_REL_CHANGE = 1e-5
QOI_MAX_CONTINUITY_ERROR = 1e-8
QOI_WINDOW_ITERS = 500

# M2b transient-to-steady convergence criteria (plan/DECISIONS.md, 2026-11-24):
# relative change of dp*, outlet phi-profile mean, and reattachment length (G2/KG)
# < 1e-4 over the last 10 % of simulated time; otherwise status: failed, reason: not_steady.
M2B_STEADY_MAX_REL_CHANGE = 1e-4
M2B_STEADY_WINDOW_FRAC = 0.10
DEFAULT_CFL_THRESHOLD_FRAC = 0.5

# Default star-unit shear window corresponding to 30-60 Pa at the 2023 baseline
# (h1 = 50 um, U = 0.25 m/s, mu_inf = 3.45 mPa.s => tau_ref = 17.25 Pa).
DEFAULT_TAU_STAR_LO = 30.0 / 17.25
DEFAULT_TAU_STAR_HI = 60.0 / 17.25

# Fixed resampling grid dimensions (Nx, Ny) per geometry family for ML field surrogates.
FAMILY_GRID_SHAPES: dict[str, tuple[int, int]] = {
    "G1": (256, 64),
    "G2": (256, 96),
    "G3": (256, 64),
    "G4": (192, 192),
    "G5": (192, 192),
    "G6": (192, 192),
    "G7": (256, 192),
    "KG": (256, 96),
}

# Expansion-type families for reattachment length extraction via rbc.postprocess.reattachment_length.
# Maps family -> (sample_line_name, x_step_key_or_zero).
EXPANSION_REATTACHMENT_SPECS: dict[str, tuple[str, str | None]] = {
    "G2": ("nearWallLower", None),
    "G3": ("nearWallDownstream", "L_t"),
    "G7": ("nearWallExpansion", None),
    "KG": ("nearWallOuter", None),
}


# ---------------------------------------------------------------------------
# OpenFOAM ASCII field & log parsers
# ---------------------------------------------------------------------------

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def _strip_foam_comments(text: str) -> str:
    """Remove C/C++ style comments from OpenFOAM dictionary/field ASCII text."""
    text = _BLOCK_COMMENT_RE.sub("", text)
    return _LINE_COMMENT_RE.sub("", text)


def _find_matching_delim(text: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    """Return index of the matching closing delimiter starting from open_pos."""
    depth = 0
    for idx in range(open_pos, len(text)):
        ch = text[idx]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"Unmatched '{open_ch}' at offset {open_pos}")


def _parse_value_spec(text: str, width: int) -> tuple[str, np.ndarray] | None:
    """Parse an OpenFOAM uniform or nonuniform List<...> value specification.

    Returns ``("uniform", array)`` or ``("nonuniform", array)``, or ``None`` if absent.
    """
    m_non = re.search(
        r"\bnonuniform\s+List<\s*(scalar|vector|tensor|symmTensor)\s*>\s*(\d+)\s*\(",
        text,
    )
    if m_non:
        n_items = int(m_non.group(2))
        open_paren = m_non.end() - 1
        close_paren = _find_matching_delim(text, open_paren, "(", ")")
        body = text[open_paren + 1 : close_paren]
        flat = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ", dtype=np.float64)
        expected = n_items * width
        if flat.size != expected:
            raise ValueError(
                f"Expected {expected} numeric entries in nonuniform List ({n_items} x {width}), "
                f"got {flat.size}"
            )
        if width == 1:
            return "nonuniform", flat
        if width == 9:
            return "nonuniform", flat.reshape((n_items, 3, 3))
        return "nonuniform", flat.reshape((n_items, width))

    m_uni_vec = re.search(r"\buniform\s*\(([^)]+)\)", text)
    if m_uni_vec:
        vals = np.fromstring(m_uni_vec.group(1), sep=" ", dtype=np.float64)
        if vals.size != width:
            raise ValueError(f"Expected {width} components in uniform value, got {vals.size}")
        if width == 9:
            return "uniform", vals.reshape((3, 3))
        return "uniform", vals

    m_uni_sca = re.search(r"\buniform\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if m_uni_sca:
        return "uniform", np.array(float(m_uni_sca.group(1)), dtype=np.float64)

    return None


def _width_from_foam_class(foam_class: str) -> int:
    """Infer component count per element from OpenFOAM FoamFile class string."""
    lower = foam_class.lower()
    if "tensor" in lower and "symm" not in lower:
        return 9
    if "symmtensor" in lower:
        return 6
    if "vector" in lower:
        return 3
    return 1


def parse_openfoam_field(path: Path) -> dict[str, Any]:
    """Parse an ASCII OpenFOAM field file (volScalar/Vector/TensorField or Internal).

    Returns a dictionary with keys:
    - ``class``: OpenFOAM class string
    - ``internal``: ``(kind, np.ndarray)`` from ``internalField`` (or top-level ``value``
      for ``volScalarField::Internal`` such as ``V``)
    - ``boundary``: ``dict[patch_name, {"type": str | None, "value": (kind, np.ndarray) | None}]``
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    clean = _strip_foam_comments(raw)

    m_cls = re.search(r"\bclass\s+([^\s;]+)\s*;", clean)
    foam_class = m_cls.group(1) if m_cls else "volScalarField"
    width = _width_from_foam_class(foam_class)

    # Locate boundaryField block (if present) so internal parsing does not cross into it.
    bf_match = re.search(r"\bboundaryField\s*\{", clean)
    header_and_internal = clean[: bf_match.start()] if bf_match else clean

    # Parse internalField (or top-level value for DimensionedField / volScalarField::Internal).
    m_int = re.search(r"\binternalField\b", header_and_internal)
    if m_int:
        internal_spec = _parse_value_spec(header_and_internal[m_int.end() :], width)
    else:
        # Remove FoamFile { ... } before searching for top-level 'value'
        ff = re.search(r"\bFoamFile\s*\{", header_and_internal)
        search_start = _find_matching_delim(header_and_internal, ff.end() - 1, "{", "}") + 1 if ff else 0
        m_val = re.search(r"\bvalue\b", header_and_internal[search_start:])
        internal_spec = (
            _parse_value_spec(header_and_internal[search_start + m_val.end() :], width)
            if m_val
            else None
        )

    if internal_spec is None:
        raise ValueError(f"Could not locate internalField/value in OpenFOAM file {path}")

    boundary: dict[str, dict[str, Any]] = {}
    if bf_match:
        bf_open = bf_match.end() - 1
        bf_close = _find_matching_delim(clean, bf_open, "{", "}")
        bf_body = clean[bf_open + 1 : bf_close]
        pos = 0
        while pos < len(bf_body):
            m_patch = re.search(r"([A-Za-z0-9_.\-]+)\s*\{", bf_body[pos:])
            if not m_patch:
                break
            patch_name = m_patch.group(1)
            p_open = pos + m_patch.end() - 1
            p_close = _find_matching_delim(bf_body, p_open, "{", "}")
            p_body = bf_body[p_open + 1 : p_close]
            m_type = re.search(r"\btype\s+([^\s;]+)\s*;", p_body)
            m_pval = re.search(r"\bvalue\b", p_body)
            pval = _parse_value_spec(p_body[m_pval.end() :], width) if m_pval else None
            boundary[patch_name] = {
                "type": m_type.group(1) if m_type else None,
                "value": pval,
            }
            pos = p_close + 1

    return {"class": foam_class, "internal": internal_spec, "boundary": boundary}


def _expand_internal(spec: tuple[str, np.ndarray], n_cells: int) -> np.ndarray:
    """Broadcast a uniform internal field to n_cells or validate nonuniform length."""
    kind, arr = spec
    if kind == "uniform":
        if arr.ndim == 0:
            return np.full((n_cells,), float(arr), dtype=np.float64)
        return np.broadcast_to(arr, (n_cells, *arr.shape)).copy()
    if arr.shape[0] != n_cells:
        raise ValueError(f"Field length {arr.shape[0]} != n_cells {n_cells}")
    return arr


def latest_time_dir(case_dir: Path) -> Path:
    """Return the numerically largest positive time directory in an OpenFOAM case."""
    candidates: list[tuple[float, Path]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            t_val = float(child.name)
        except ValueError:
            continue
        if t_val > 0.0:
            candidates.append((t_val, child))
    if not candidates:
        raise FileNotFoundError(f"No positive time directory found in {case_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def parse_simplefoam_log(log_path: Path) -> dict[str, Any]:
    """Parse OpenFOAM version, iteration count, convergence flag, residuals, and continuity error."""
    if not log_path.is_file():
        return {
            "openfoam_version": "unknown",
            "iterations": 0,
            "converged": False,
            "final_residuals": {},
            "continuity_error": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")

    version = "unknown"
    m_ver = re.search(r"Build\s*:\s*([^\n]+)", text)
    if m_ver:
        version = m_ver.group(1).strip()
    else:
        m_v2 = re.search(r"Version\s*:\s*([^\n]+)", text)
        if m_v2:
            version = m_v2.group(1).strip()

    times = re.findall(r"^Time\s*=\s*([0-9.eE+-]+)", text, flags=re.MULTILINE)
    iterations = int(round(float(times[-1]))) if times else 0
    converged = "SIMPLE solution converged" in text

    # Track per-timestep initial residuals (taking first solve per field within each step,
    # matching OpenFOAM's SIMPLE residualControl check when nNonOrthogonalCorrectors >= 1).
    last_step_residuals: dict[str, float] = {}
    current_step_residuals: dict[str, float] = {}
    continuity_error: float | None = None

    time_re = re.compile(r"^Time\s*=\s*([0-9.eE+-]+)")
    res_re = re.compile(
        r"Solving for\s+([A-Za-z0-9_]+),\s+Initial residual\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?nan|[+-]?inf)",
        flags=re.IGNORECASE,
    )
    cont_re = re.compile(
        r"time step continuity errors\s*:\s*sum local\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?nan|[+-]?inf)",
        flags=re.IGNORECASE,
    )

    for line in text.splitlines():
        if time_re.match(line):
            if current_step_residuals:
                last_step_residuals = current_step_residuals
                current_step_residuals = {}
            continue
        m_res = res_re.search(line)
        if m_res:
            field = m_res.group(1)
            val = float(m_res.group(2))
            if field not in current_step_residuals:
                current_step_residuals[field] = val
            continue
        m_cont = cont_re.search(line)
        if m_cont:
            continuity_error = abs(float(m_cont.group(1)))

    if current_step_residuals:
        last_step_residuals = current_step_residuals

    return {
        "openfoam_version": version,
        "iterations": iterations,
        "converged": converged,
        "final_residuals": last_step_residuals,
        "continuity_error": continuity_error,
    }


def _sorted_numeric_subdirs(parent: Path) -> list[Path]:
    """Return numerically sorted subdirectories under ``parent`` (ignoring non-numeric names)."""
    if not parent.is_dir():
        return []
    items: list[tuple[float, Path]] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        try:
            t_val = float(child.name)
        except ValueError:
            continue
        items.append((t_val, child))
    items.sort(key=lambda item: item[0])
    return [p for _, p in items]


def read_inlet_pressure_history(case_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read per-iteration inlet patch areaAverage(p) from ``postProcessing/inletPressure``."""
    parent = case_dir / "postProcessing" / "inletPressure"
    subdirs = _sorted_numeric_subdirs(parent)
    if not subdirs:
        return None

    by_iter: dict[float, float] = {}
    for sub in subdirs:
        dat = sub / "surfaceFieldValue.dat"
        if not dat.is_file():
            continue
        for line in dat.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                t_val = float(parts[0])
                p_val = float(parts[1])
            except ValueError:
                continue
            by_iter[t_val] = p_val

    if not by_iter:
        return None
    iters_sorted = sorted(by_iter)
    return (
        np.array(iters_sorted, dtype=np.float64),
        np.array([by_iter[t] for t in iters_sorted], dtype=np.float64),
    )


_VEC_TUPLE_RE = re.compile(r"\(\s*([^\s)]+)\s+([^\s)]+)\s+([^\s)]+)\s*\)")


def read_outlet_probe_velocity_history(case_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read per-iteration outlet-centreline speed ``|U|`` from ``postProcessing/qoiProbes/*/U``."""
    parent = case_dir / "postProcessing" / "qoiProbes"
    subdirs = _sorted_numeric_subdirs(parent)
    if not subdirs:
        return None

    by_iter: dict[float, list[float]] = {}
    n_probes: int | None = None
    for sub in subdirs:
        u_path = sub / "U"
        if not u_path.is_file():
            continue
        for line in u_path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m_first = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+", s)
            if not m_first:
                continue
            t_val = float(m_first.group(1))
            vecs = _VEC_TUPLE_RE.findall(s[m_first.end() :])
            if not vecs:
                continue
            try:
                mags = [math.sqrt(float(vx) ** 2 + float(vy) ** 2 + float(vz) ** 2) for vx, vy, vz in vecs]
            except ValueError:
                continue
            if n_probes is None:
                n_probes = len(mags)
            if len(mags) != n_probes:
                continue
            by_iter[t_val] = mags

    if not by_iter or n_probes is None:
        return None
    iters_sorted = sorted(by_iter)
    arr = np.array([by_iter[t] for t in iters_sorted], dtype=np.float64)
    return np.array(iters_sorted, dtype=np.float64), arr


def _window_relative_change(
    iters: np.ndarray,
    values: np.ndarray,
    window_iters: int,
) -> float | None:
    """Compute peak-to-peak relative change ``(max - min) / |final|`` over the last ``window_iters``.

    Returns ``None`` if the time series spans fewer than ``window_iters`` iterations, has
    non-finite values, or has near-zero final value (``<= 1e-12``).
    """
    if iters.size < 2:
        return None
    t_end = float(iters[-1])
    t_start = float(iters[0])
    if t_end - t_start < float(window_iters):
        return None

    sel = iters >= (t_end - float(window_iters) - 1e-9)
    win = values[sel]
    if win.shape[0] < 2 or not np.all(np.isfinite(win)):
        return None

    if win.ndim == 1:
        denom = abs(float(win[-1]))
        if denom <= 1e-12:
            return None
        return float((np.max(win) - np.min(win)) / denom)

    # Multi-probe array (n_steps, n_probes): return worst (maximum) relative change across probes.
    worst = 0.0
    for col in range(win.shape[1]):
        series = win[:, col]
        denom = abs(float(series[-1]))
        if denom <= 1e-12:
            return None
        rel = float((np.max(series) - np.min(series)) / denom)
        if rel > worst:
            worst = rel
    return worst


def evaluate_convergence(
    case_dir: Path,
    *,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
    qoi_max_residual: float = QOI_MAX_RESIDUAL,
    qoi_max_rel_change: float = QOI_MAX_REL_CHANGE,
    qoi_max_continuity_error: float = QOI_MAX_CONTINUITY_ERROR,
    qoi_window_iters: int = QOI_WINDOW_ITERS,
) -> dict[str, Any]:
    """Evaluate dual convergence criterion (plan/DECISIONS.md):

    - Criterion (a) ``converged_by = "residual"``: all initial residuals < ``residual_tol`` (1e-5).
    - Criterion (b) ``converged_by = "qoi"``: all initial residuals < 1e-3, continuity error
      (``sum local``) < 1e-8, and both ``dp*`` and outlet-centreline velocity change by
      < 1e-5 (relative) over the last 500 iterations.
    - Otherwise ``converged = False, converged_by = None`` (run must fail with ``not_converged``).
    """
    sf_log = parse_simplefoam_log(case_dir / "log.simpleFoam")
    residuals: dict[str, float] = sf_log["final_residuals"]
    cont_err: float | None = sf_log["continuity_error"]

    all_res_finite = bool(residuals) and all(math.isfinite(v) for v in residuals.values())
    max_res = max(residuals.values()) if all_res_finite else None

    # Criterion (a): residual convergence (< 1e-5).
    if residuals:
        res_converged = all_res_finite and (max_res is not None) and (max_res < residual_tol)
    else:
        res_converged = bool(sf_log["converged"])

    if res_converged:
        return {
            "converged": True,
            "converged_by": "residual",
            "iterations": sf_log["iterations"],
            "final_residuals": residuals,
            "max_initial_residual": max_res,
            "continuity_error": cont_err,
            "dp_rel_change_500": None,
            "u_out_rel_change_500": None,
            "window_iters": qoi_window_iters,
            "openfoam_version": sf_log["openfoam_version"],
        }

    # Criterion (b): QoI plateau convergence.
    dp_hist = read_inlet_pressure_history(case_dir)
    u_hist = read_outlet_probe_velocity_history(case_dir)
    dp_rel = _window_relative_change(dp_hist[0], dp_hist[1], qoi_window_iters) if dp_hist is not None else None
    u_rel = _window_relative_change(u_hist[0], u_hist[1], qoi_window_iters) if u_hist is not None else None

    qoi_converged = (
        all_res_finite
        and max_res is not None
        and max_res < qoi_max_residual
        and cont_err is not None
        and math.isfinite(cont_err)
        and cont_err < qoi_max_continuity_error
        and dp_rel is not None
        and dp_rel < qoi_max_rel_change
        and u_rel is not None
        and u_rel < qoi_max_rel_change
    )

    return {
        "converged": qoi_converged,
        "converged_by": "qoi" if qoi_converged else None,
        "iterations": sf_log["iterations"],
        "final_residuals": residuals,
        "max_initial_residual": max_res,
        "continuity_error": cont_err,
        "dp_rel_change_500": dp_rel,
        "u_out_rel_change_500": u_rel,
        "window_iters": qoi_window_iters,
        "openfoam_version": sf_log["openfoam_version"],
    }


def parse_checkmesh_log(log_path: Path) -> dict[str, Any]:
    """Parse cell count and mesh status from log.checkMesh."""
    if not log_path.is_file():
        return {"n_cells": None, "mesh_ok": None}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m_cells = re.search(r"^\s*cells:\s*(\d+)", text, flags=re.MULTILINE)
    return {
        "n_cells": int(m_cells.group(1)) if m_cells else None,
        "mesh_ok": "Mesh OK." in text,
    }


def read_sample_xy(case_dir: Path, time_name: str, line_name: str) -> dict[str, np.ndarray]:
    """Read an OpenFOAM ``sets`` raw ``.xy`` file for ``line_name`` at ``time_name``."""
    sample_dir = case_dir / "postProcessing" / "sample" / time_name
    if not sample_dir.is_dir():
        # Fall back to latest numerically sorted directory under postProcessing/sample
        parent = case_dir / "postProcessing" / "sample"
        if not parent.is_dir():
            raise FileNotFoundError(f"Missing sample directory {parent}")
        dirs = sorted(
            (d for d in parent.iterdir() if d.is_dir()),
            key=lambda p: float(p.name),
        )
        if not dirs:
            raise FileNotFoundError(f"No time subdirectories in {parent}")
        sample_dir = dirs[-1]

    matches = sorted(sample_dir.glob(f"{line_name}_*.xy"))
    if not matches:
        raise FileNotFoundError(f"No sample file matching {line_name}_*.xy in {sample_dir}")

    # Prefer combined _p_U.xy / _U_p.xy; fall back to merging separate _p.xy and _U.xy files.
    combined = [m for m in matches if m.stem.endswith("_p_U") or m.stem.endswith("_U_p")]
    if combined:
        path = combined[0]
        data = np.loadtxt(path, comments="#", dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 5:
            raise ValueError(
                f"Expected >= 5 columns (coord, p, Ux, Uy, Uz) in {path}, got {data.shape}"
            )
        if path.stem.endswith("_U_p"):
            return {
                "coord": data[:, 0],
                "ux": data[:, 1],
                "uy": data[:, 2],
                "uz": data[:, 3],
                "p": data[:, 4],
            }
        return {
            "coord": data[:, 0],
            "p": data[:, 1],
            "ux": data[:, 2],
            "uy": data[:, 3],
            "uz": data[:, 4],
        }

    u_file = sample_dir / f"{line_name}_U.xy"
    p_file = sample_dir / f"{line_name}_p.xy"
    if u_file.is_file():
        u_data = np.loadtxt(u_file, comments="#", dtype=np.float64)
        if u_data.ndim == 1:
            u_data = u_data[None, :]
        p_col = (
            np.loadtxt(p_file, comments="#", dtype=np.float64).reshape(-1, 2)[:, 1]
            if p_file.is_file()
            else np.zeros(u_data.shape[0], dtype=np.float64)
        )
        return {
            "coord": u_data[:, 0],
            "p": p_col,
            "ux": u_data[:, 1],
            "uy": u_data[:, 2],
            "uz": u_data[:, 3],
        }
    raise FileNotFoundError(f"Unrecognized sample file layout in {sample_dir}: {matches}")


# ---------------------------------------------------------------------------
# Nondimensional physics helpers
# ---------------------------------------------------------------------------

def strain_rate_magnitude(grad_u: np.ndarray) -> np.ndarray:
    """Compute scalar shear rate gdot* = sqrt(2 D*:D*) from 3x3 velocity gradient tensor(s).

    Because D* = 0.5 * (grad(U) + grad(U)^T) is symmetric, gdot* is invariant to whether
    grad(U)_{ij} stores dU_i/dx_j or dU_j/dx_i.
    """
    g = np.asarray(grad_u, dtype=np.float64)
    d_sym = 0.5 * (g + np.swapaxes(g, -1, -2))
    return np.sqrt(np.maximum(0.0, 2.0 * np.sum(d_sym * d_sym, axis=(-2, -1))))


def viscosity_ratio_star(gdot_star: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Dimensionless viscosity ratio mu_r = mu / mu_inf (knowledge/11-math-model.md §3.2)."""
    gdot = np.asarray(gdot_star, dtype=np.float64)
    rheology = str(params.get("rheology", "Newtonian"))
    if rheology == "Newtonian":
        # V1 of a paired design runs the suspension as a Newtonian fluid of mu_eff(phi); the
        # ratio to the carrier is carried in the design record, so tau* must include it.
        return np.full_like(gdot, float(params.get("viscosity_ratio", 1.0)), dtype=np.float64)
    if rheology == "BirdCarreau":
        beta = float(params["beta"])
        Cu = float(params["Cu"])
        n = float(params["n"])
        a = float(params.get("a", 2.0))
        ratio = float(params.get("viscosity_ratio", 1.0))
        return ratio * (1.0 + (beta - 1.0) * (1.0 + (Cu * gdot) ** a) ** ((n - 1.0) / a))
    raise ValueError(f"Unsupported rheology '{rheology}'")


def shear_stress_star(gdot_star: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Dimensionless scalar shear stress tau* = mu_r(gdot*) * gdot*."""
    return viscosity_ratio_star(gdot_star, params) * np.asarray(gdot_star, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixed per-family resampling grid + exact fluid mask
# ---------------------------------------------------------------------------

def _points_in_convex_quad(
    px: np.ndarray,
    py: np.ndarray,
    quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
    tol: float = 1e-9,
) -> np.ndarray:
    """Vectorized point-in-CCW-convex-quadrilateral test."""
    inside = np.ones(px.shape, dtype=bool)
    for k in range(4):
        x0, y0 = quad[k]
        x1, y1 = quad[(k + 1) % 4]
        cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        inside &= cross >= -tol
    return inside


def build_fluid_mask(mesh: MeshSpec, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Build exact 2D fluid boolean mask (Ny, Nx) for any family G2-G7 or KG."""
    xx, yy = np.meshgrid(grid_x, grid_y)
    mask = np.zeros(xx.shape, dtype=bool)

    if mesh.family == "G6":
        # G6 consists of straight rectangular blocks plus circular-annulus 180-deg bends.
        Rc = float(mesh.params["Rc_ratio"])
        n_bends = int(mesh.params["n_bends"])
        L_straight = float(mesh.params.get("L_straight", 2.0))
        r_in = Rc - 0.5
        r_out = Rc + 0.5
        tol = 1e-9

        # Identify which blocks are straight vs curved (blocks 0 and 1+3k+2 are straight).
        curved_indices: set[int] = set()
        for k in range(n_bends):
            curved_indices.add(1 + 3 * k)
            curved_indices.add(1 + 3 * k + 1)

        for idx, blk in enumerate(mesh.blocks_2d):
            if idx in curved_indices:
                continue
            quad = tuple(mesh.vertices_2d[i] for i in blk.verts)
            mask |= _points_in_convex_quad(xx, yy, quad, tol=tol)  # type: ignore[arg-type]

        for k in range(n_bends):
            yc = 2.0 * k * Rc + 0.5 + Rc
            if k % 2 == 0:
                cx = 0.0
                r = np.hypot(xx - cx, yy - yc)
                mask |= (xx >= cx - tol) & (r >= r_in - tol) & (r <= r_out + tol)
            else:
                cx = -L_straight
                r = np.hypot(xx - cx, yy - yc)
                mask |= (xx <= cx + tol) & (r >= r_in - tol) & (r <= r_out + tol)
        return mask

    # G2, G3, G4, G5, G7, and KG (in meridian x*-r* plane): exact union of convex 2D quads.
    for blk in mesh.blocks_2d:
        quad = tuple(mesh.vertices_2d[i] for i in blk.verts)
        mask |= _points_in_convex_quad(xx, yy, quad)  # type: ignore[arg-type]
    return mask


def resample_to_fixed_grid(
    mesh: MeshSpec,
    x_star: np.ndarray,
    y_star: np.ndarray,
    fields: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Resample unstructured cell-center fields onto the fixed per-family grid + fluid mask."""
    nx, ny = FAMILY_GRID_SHAPES[mesh.family]
    xs_v = [pt[0] for pt in mesh.vertices_2d]
    ys_v = [pt[1] for pt in mesh.vertices_2d]
    x_min, x_max = min(xs_v), max(xs_v)
    y_min, y_max = min(ys_v), max(ys_v)

    if mesh.family == "G6":
        # Arc midpoints extend horizontally to +r_out (even bends) and -L_straight - r_out (odd bends).
        Rc = float(mesh.params["Rc_ratio"])
        r_out = Rc + 0.5
        L_straight = float(mesh.params.get("L_straight", 2.0))
        x_max = max(x_max, r_out)
        if int(mesh.params["n_bends"]) >= 2:
            x_min = min(x_min, -L_straight - r_out)

    grid_x = np.linspace(x_min, x_max, nx, dtype=np.float64)
    grid_y = np.linspace(y_min, y_max, ny, dtype=np.float64)
    grid_mask = build_fluid_mask(mesh, grid_x, grid_y)

    xx, yy = np.meshgrid(grid_x, grid_y)
    pts_src = np.column_stack([x_star, y_star])
    pts_dst = np.column_stack([xx[grid_mask], yy[grid_mask]])

    out: dict[str, np.ndarray] = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_mask": grid_mask,
    }
    if pts_dst.size == 0 or pts_src.shape[0] < 3:
        for name in fields:
            out[f"grid_{name}"] = np.zeros((ny, nx), dtype=np.float32)
        return out

    # Reuse Delaunay triangulation across all fields by building one nearest + linear stack.
    names = list(fields.keys())
    val_matrix = np.column_stack([np.asarray(fields[k], dtype=np.float64) for k in names])
    lin = LinearNDInterpolator(pts_src, val_matrix)
    interp = lin(pts_dst)
    nan_rows = np.any(np.isnan(interp), axis=1)
    if np.any(nan_rows):
        near = NearestNDInterpolator(pts_src, val_matrix)
        interp[nan_rows] = near(pts_dst[nan_rows])

    for col_idx, name in enumerate(names):
        arr = np.zeros((ny, nx), dtype=np.float32)
        arr[grid_mask] = interp[:, col_idx].astype(np.float32)
        out[f"grid_{name}"] = arr

    return out


# ---------------------------------------------------------------------------
# Scalar & outlet-section extraction
# ---------------------------------------------------------------------------

def _sample_line_xy(sl: SampleLine, coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map 1D sample coordinate back to 2D (x*, y*) coordinates along ``sl``."""
    sx, sy, _ = sl.start
    ex, ey, _ = sl.end
    if sl.axis == "x":
        t = (coord - sx) / (ex - sx) if abs(ex - sx) > 1e-12 else np.zeros_like(coord)
    elif sl.axis == "y":
        t = (coord - sy) / (ey - sy) if abs(ey - sy) > 1e-12 else np.zeros_like(coord)
    else:
        length = math.hypot(ex - sx, ey - sy)
        t = coord / length if length > 1e-12 else np.zeros_like(coord)
    t = np.clip(t, 0.0, 1.0)
    return sx + t * (ex - sx), sy + t * (ey - sy)


def _outlet_section_specs(mesh: MeshSpec) -> list[tuple[str, tuple[float, float]]]:
    """Return list of ``(sample_line_name, (nx, ny))`` outward unit normal(s) for outlet sections."""
    fam = mesh.family
    if fam in ("G1", "G2", "G3", "KG"):
        return [("outletProfile", (1.0, 0.0))]
    if fam == "G6":
        n_bends = int(mesh.params["n_bends"])
        normal = (-1.0, 0.0) if (n_bends % 2 == 1) else (1.0, 0.0)
        return [("outletProfile", normal)]
    if fam in ("G4", "G5", "G7"):
        theta = math.radians(float(mesh.params["theta_deg"]))
        ct, st = math.cos(theta), math.sin(theta)
        return [
            ("daughter1Mid", (ct, st)),
            ("daughter2Mid", (ct, -st)),
        ]
    raise ValueError(f"No outlet section specification for family '{fam}'")


def _compute_dp_star(
    mesh: MeshSpec,
    x_star: np.ndarray,
    y_star: np.ndarray,
    u_star: np.ndarray,
    p_star: np.ndarray,
    cell_volumes: np.ndarray,
) -> float:
    """Compute inlet-to-outlet pressure drop dp* in viscous star units.

    Outlet patch has fixedValue p* = 0. Inlet pressure is evaluated strictly from
    cells inside the inlet channel (bounded by the ``inlet`` patch y/r extent and
    ``[x_in, x_in + 0.5 * entry_length]`` — essential for G6 serpentines whose
    outlet leg extends to smaller x at higher y), with half-cell linear
    extrapolation back to ``x_in`` when two inlet columns are present.
    """
    inlet_patch = next((p for p in mesh.patches if p.name == "inlet"), None)
    if inlet_patch is not None:
        inlet_verts = [mesh.vertices[idx] for face in inlet_patch.faces for idx in face]
        x_in = min(v[0] for v in inlet_verts)
        if mesh.family == "KG":
            y_lo = 0.0
            y_hi = max(math.hypot(v[1], v[2]) for v in inlet_verts)
        else:
            y_lo = min(v[1] for v in inlet_verts)
            y_hi = max(v[1] for v in inlet_verts)
    else:
        x_in = min(pt[0] for pt in mesh.vertices_2d)
        y_lo = 0.0
        y_hi = 0.5 if mesh.family == "KG" else 1.0

    inlet_region = (
        (y_star >= y_lo - 1e-6)
        & (y_star <= y_hi + 1e-6)
        & (x_star >= x_in - 1e-6)
        & (x_star <= x_in + 0.5 * mesh.entry_length)
    )
    if not np.any(inlet_region):
        inlet_region = (y_star >= y_lo - 1e-6) & (y_star <= y_hi + 1e-6)

    x_sub = x_star[inlet_region]
    u_sub = u_star[inlet_region]
    p_sub = p_star[inlet_region]
    v_sub = cell_volumes[inlet_region]

    unique_x = np.unique(np.round(x_sub, 6))

    def _col_mean_p(x_col: float, tol: float) -> tuple[float, float]:
        sel = np.abs(x_sub - x_col) <= tol
        if not np.any(sel):
            return float("nan"), float("nan")
        ux_c = u_sub[sel, 0]
        p_c = p_sub[sel]
        vol_c = v_sub[sel]
        w = np.where(ux_c > 0.0, ux_c * vol_c, 0.0)
        if np.sum(w) > 0.0:
            return float(np.sum(w * p_c) / np.sum(w)), float(np.mean(x_sub[sel]))
        return float(np.sum(vol_c * p_c) / np.sum(vol_c)), float(np.mean(x_sub[sel]))

    if unique_x.size >= 2:
        dx_est = float(unique_x[1] - unique_x[0])
        tol = max(1e-5, 0.25 * dx_est)
        p0, x0 = _col_mean_p(float(unique_x[0]), tol)
        p1, x1 = _col_mean_p(float(unique_x[1]), tol)
        if abs(x1 - x0) > 1e-12 and math.isfinite(p0) and math.isfinite(p1):
            p_inlet = p0 + (x0 - x_in) * (p0 - p1) / (x1 - x0)
            return float(p_inlet)
        return float(p0)

    p0, _ = _col_mean_p(float(unique_x[0]), 1e-3)
    return float(p0)


def recompute_window_fraction(
    npz_or_path: Path | str | Mapping[str, np.ndarray],
    tau_lo: float,
    tau_hi: float,
) -> float:
    """Recompute flux-weighted outlet window fraction for any star-unit [tau_lo, tau_hi].

    Uses ``outlet_phi`` weighting automatically when present (M2b multiphase runs);
    remains identical to single-phase M1 when absent.
    """
    if isinstance(npz_or_path, (Path, str)):
        with np.load(npz_or_path) as data:
            tau = data["outlet_tau"]
            u_n = data["outlet_u_n"]
            weights = data["outlet_weights"]
            phi = data["outlet_phi"] if "outlet_phi" in data.files else None
    else:
        tau = npz_or_path["outlet_tau"]
        u_n = npz_or_path["outlet_u_n"]
        weights = npz_or_path["outlet_weights"]
        phi = npz_or_path.get("outlet_phi") if hasattr(npz_or_path, "get") else (
            npz_or_path["outlet_phi"] if "outlet_phi" in npz_or_path else None
        )
    return float(window_fraction(tau, u_n, weights, phi=phi, lo=tau_lo, hi=tau_hi))


# ---------------------------------------------------------------------------
# M2b (multiphaseEulerFoam / reactingTwoPhaseEulerFoam) helpers & extractor
# ---------------------------------------------------------------------------

_M2B_SOLVERS = ("multiphaseEulerFoam", "reactingTwoPhaseEulerFoam")

_M2B_KNOWN_SAMPLE_FIELDS: tuple[tuple[str, int], ...] = (
    ("alpha.plasma", 1),
    ("alpha.RBC", 1),
    ("U.plasma", 3),
    ("U.RBC", 3),
    ("p_rgh", 1),
    ("U", 3),
    ("p", 1),
)


def _is_m2b_record(record: Mapping[str, Any]) -> bool:
    """Return True when ``record`` belongs to an M2b Euler–Euler tier/solver."""
    tier_val = str(record.get("tier") or "").lower()
    solver_val = str(record.get("solver") or "")
    return tier_val.startswith("m2b") or solver_val in _M2B_SOLVERS


def _m2b_solver_name(record: Mapping[str, Any]) -> str:
    """Resolve M2b OpenFOAM solver name (``multiphaseEulerFoam`` or ``reactingTwoPhaseEulerFoam``)."""
    solver_val = str(record.get("solver") or "")
    if solver_val in _M2B_SOLVERS:
        return solver_val
    tier_val = str(record.get("tier") or "").lower()
    if tier_val in ("m2b-b", "m2b_b", "m2b_lift"):
        return "reactingTwoPhaseEulerFoam"
    return "multiphaseEulerFoam"


def parse_m2b_log(log_path: Path) -> dict[str, Any]:
    """Parse OpenFOAM version, final time, step count, ``End`` flag, and continuity error."""
    if not log_path.is_file():
        return {
            "openfoam_version": "unknown",
            "t_end": 0.0,
            "n_steps": 0,
            "reached_end": False,
            "continuity_error": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")

    version = "unknown"
    m_ver = re.search(r"Build\s*:\s*([^\n]+)", text)
    if m_ver:
        version = m_ver.group(1).strip()
    else:
        m_v2 = re.search(r"Version\s*:\s*([^\n]+)", text)
        if m_v2:
            version = m_v2.group(1).strip()

    times = [float(t) for t in re.findall(r"^Time\s*=\s*([0-9.eE+-]+)", text, flags=re.MULTILINE)]
    t_end = times[-1] if times else 0.0
    reached_end = re.search(r"^End\s*$", text, flags=re.MULTILINE) is not None

    cont_re = re.compile(
        r"time step continuity errors\s*:\s*sum local\s*=\s*"
        r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?nan|[+-]?inf)",
        flags=re.IGNORECASE,
    )
    continuity_error: float | None = None
    for m_cont in cont_re.finditer(text):
        continuity_error = abs(float(m_cont.group(1)))

    return {
        "openfoam_version": version,
        "t_end": t_end,
        "n_steps": len(times),
        "reached_end": reached_end,
        "continuity_error": continuity_error,
    }


def read_surface_field_history(case_dir: Path, func_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read per-time scalar series from ``postProcessing/<func_name>/*/surfaceFieldValue.dat``."""
    parent = case_dir / "postProcessing" / func_name
    subdirs = _sorted_numeric_subdirs(parent)
    if not subdirs:
        return None

    by_time: dict[float, float] = {}
    for sub in subdirs:
        dat = sub / "surfaceFieldValue.dat"
        if not dat.is_file():
            continue
        for line in dat.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                t_val = float(parts[0])
                v_val = float(parts[1])
            except ValueError:
                continue
            by_time[t_val] = v_val

    if not by_time:
        return None
    ts_sorted = sorted(by_time)
    return (
        np.array(ts_sorted, dtype=np.float64),
        np.array([by_time[t] for t in ts_sorted], dtype=np.float64),
    )


def _tokenize_m2b_sample_suffix(suffix: str) -> list[tuple[str, int]]:
    """Split an OpenFOAM raw ``.xy`` field suffix (e.g. ``alpha.RBC_p``) into ``(field, width)``."""
    fields: list[tuple[str, int]] = []
    rem = suffix
    while rem:
        matched = False
        for fname, width in _M2B_KNOWN_SAMPLE_FIELDS:
            if rem == fname:
                fields.append((fname, width))
                rem = ""
                matched = True
                break
            if rem.startswith(fname + "_"):
                fields.append((fname, width))
                rem = rem[len(fname) + 1 :]
                matched = True
                break
        if not matched:
            return []
    return fields


def read_m2b_sample_xy(
    case_dir: Path,
    time_name: str,
    line_name: str,
    *,
    sample_root: str = "sample",
) -> dict[str, np.ndarray]:
    """Read OpenFOAM ``sets`` raw ``.xy`` file(s) for an M2b multiphase sample line.

    Supports any ESI v2306 ``raw`` grouping (combined scalar file + separate vector files,
    single combined file, or individual per-field files).
    """
    parent = case_dir / "postProcessing" / sample_root
    sample_dir = parent / time_name
    if not sample_dir.is_dir():
        subdirs = _sorted_numeric_subdirs(parent)
        if not subdirs:
            raise FileNotFoundError(f"No sample time subdirectories in {parent}")
        sample_dir = subdirs[-1]

    matches = sorted(sample_dir.glob(f"{line_name}_*.xy"))
    if not matches:
        raise FileNotFoundError(f"No sample file matching {line_name}_*.xy in {sample_dir}")

    coord: np.ndarray | None = None
    parsed: dict[str, np.ndarray] = {}

    prefix = f"{line_name}_"
    for path in matches:
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        field_specs = _tokenize_m2b_sample_suffix(suffix)
        if not field_specs:
            continue

        data = np.loadtxt(path, comments="#", dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        expected_cols = 1 + sum(w for _, w in field_specs)
        if data.shape[1] != expected_cols:
            continue

        if coord is None:
            coord = data[:, 0]
        elif data.shape[0] != coord.shape[0]:
            continue

        col = 1
        for fname, width in field_specs:
            if width == 1:
                parsed[fname] = data[:, col]
            else:
                parsed[fname] = data[:, col : col + width]
            col += width

    if coord is None or not parsed:
        raise FileNotFoundError(
            f"Could not parse any recognized M2b sample files for '{line_name}' in {sample_dir}"
        )

    out: dict[str, np.ndarray] = {"coord": coord}
    if "alpha.RBC" in parsed:
        out["alpha_rbc"] = parsed["alpha.RBC"]
    if "alpha.plasma" in parsed:
        out["alpha_plasma"] = parsed["alpha.plasma"]
    elif "alpha_rbc" in out:
        out["alpha_plasma"] = 1.0 - out["alpha_rbc"]

    if "U.plasma" in parsed:
        out["u_plasma"] = parsed["U.plasma"]
    if "U.RBC" in parsed:
        out["u_rbc"] = parsed["U.RBC"]

    if "U" in parsed:
        u_mix = parsed["U"]
    elif "u_plasma" in out and "u_rbc" in out and "alpha_rbc" in out:
        phi = out["alpha_rbc"][:, None]
        u_mix = phi * out["u_rbc"] + (1.0 - phi) * out["u_plasma"]
    elif "u_plasma" in out:
        u_mix = out["u_plasma"]
    elif "u_rbc" in out:
        u_mix = out["u_rbc"]
    else:
        u_mix = np.zeros((coord.shape[0], 3), dtype=np.float64)

    out["u_mix"] = u_mix
    out["ux"] = u_mix[:, 0]
    out["uy"] = u_mix[:, 1]
    out["uz"] = u_mix[:, 2]
    out["p"] = parsed.get("p", parsed.get("p_rgh", np.zeros_like(coord)))
    return out


def _last_fraction_rel_change(
    times: np.ndarray,
    values: np.ndarray,
    *,
    t_end: float | None = None,
    frac: float = M2B_STEADY_WINDOW_FRAC,
    allow_zero_steady: bool = False,
) -> float | None:
    """Compute peak-to-peak relative change over the last ``frac`` (default 10 %) of simulated time."""
    if times.size < 2 or values.size != times.size:
        return None
    order = np.argsort(times)
    t_s = np.asarray(times, dtype=np.float64)[order]
    v_s = np.asarray(values, dtype=np.float64)[order]

    t_hi = float(t_end) if (t_end is not None and t_end > 0.0) else float(t_s[-1])
    if t_hi <= 0.0:
        return None
    t_lo = (1.0 - frac) * t_hi

    sel = (t_s >= t_lo - 1e-9) & (t_s <= t_hi + 1e-9)
    win = v_s[sel]
    if win.size < 2 or not np.all(np.isfinite(win)):
        return None

    if allow_zero_steady and float(np.max(np.abs(win))) <= 1e-12:
        return 0.0

    denom = abs(float(win[-1]))
    if denom <= 1e-12:
        return None
    return float((np.max(win) - np.min(win)) / denom)


def _sample_history_steady_series(
    case_dir: Path,
    *,
    line_name: str,
    t_end: float,
    frac: float,
    extractor: Callable[[dict[str, np.ndarray]], float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate a scalar quantity over time from ``steadySample`` (or ``sample``) snapshots."""
    t_lo = (1.0 - frac) * t_end
    for root_name in ("steadySample", "sample"):
        parent = case_dir / "postProcessing" / root_name
        subdirs = _sorted_numeric_subdirs(parent)
        if not subdirs:
            continue
        ts: list[float] = []
        vals: list[float] = []
        for sub in subdirs:
            t_val = float(sub.name)
            if t_val < t_lo - 1e-9 or t_val > t_end + 1e-9:
                continue
            try:
                prof = read_m2b_sample_xy(case_dir, sub.name, line_name, sample_root=root_name)
                val = float(extractor(prof))
            except (FileNotFoundError, KeyError, ValueError):
                continue
            if math.isfinite(val):
                ts.append(t_val)
                vals.append(val)
        if len(ts) >= 2:
            return np.array(ts, dtype=np.float64), np.array(vals, dtype=np.float64)
    return None


def evaluate_m2b_convergence(
    case_dir: Path,
    *,
    family: str,
    solver: str = "multiphaseEulerFoam",
    max_rel_change: float = M2B_STEADY_MAX_REL_CHANGE,
    window_frac: float = M2B_STEADY_WINDOW_FRAC,
) -> dict[str, Any]:
    """Evaluate transient-to-steady convergence for an M2b case (plan/DECISIONS.md).

    Requires:
    1. ``log.<solver>`` reached ``End`` with ``t_end > 0``.
    2. Peak-to-peak relative change over the last 10 % of simulated time ``[0.9*t_end, t_end]``
       is ``< 1e-4`` for:
       - ``dp*`` (``postProcessing/inletPressure``)
       - outlet ``phi``-profile mean (``postProcessing/outletPhi`` or ``steadySample``/``sample``)
       - reattachment length ``xr`` (for expansion families ``G2`` and ``KG``; N/A for ``G1``)
    Otherwise ``converged = False`` (runner marks ``status: failed, reason: not_steady``).
    """
    fam = family.upper()
    log_path = case_dir / f"log.{solver}"
    if not log_path.is_file():
        for alt in _M2B_SOLVERS:
            cand = case_dir / f"log.{alt}"
            if cand.is_file():
                log_path = cand
                solver = alt
                break

    log_info = parse_m2b_log(log_path)
    t_end = float(log_info["t_end"])
    reached_end = bool(log_info["reached_end"])

    # 1. Inlet pressure drop dp* relative change over last 10 % of simulated time.
    dp_hist = read_surface_field_history(case_dir, "inletPressure")
    dp_rel = (
        _last_fraction_rel_change(dp_hist[0], dp_hist[1], t_end=t_end, frac=window_frac)
        if dp_hist is not None and t_end > 0.0
        else None
    )

    # 2. Outlet phi-profile mean relative change over last 10 % of simulated time.
    phi_hist = read_surface_field_history(case_dir, "outletPhi")
    if phi_hist is not None and t_end > 0.0:
        phi_rel = _last_fraction_rel_change(phi_hist[0], phi_hist[1], t_end=t_end, frac=window_frac)
    elif t_end > 0.0:
        phi_samp = _sample_history_steady_series(
            case_dir,
            line_name="outletProfile",
            t_end=t_end,
            frac=window_frac,
            extractor=lambda prof: float(np.mean(prof["alpha_rbc"])),
        )
        phi_rel = (
            _last_fraction_rel_change(phi_samp[0], phi_samp[1], t_end=t_end, frac=window_frac)
            if phi_samp is not None
            else None
        )
    else:
        phi_rel = None

    # 3. Reattachment length relative change over last 10 % of simulated time (G2, KG).
    xr_required = fam in EXPANSION_REATTACHMENT_SPECS
    xr_rel: float | None = None
    if xr_required and t_end > 0.0:
        nw_line, _ = EXPANSION_REATTACHMENT_SPECS[fam]
        xr_samp = _sample_history_steady_series(
            case_dir,
            line_name=nw_line,
            t_end=t_end,
            frac=window_frac,
            extractor=lambda prof: float(reattachment_length(prof["coord"], prof["ux"], x_step=0.0)),
        )
        if xr_samp is not None:
            xr_rel = _last_fraction_rel_change(
                xr_samp[0],
                xr_samp[1],
                t_end=t_end,
                frac=window_frac,
                allow_zero_steady=True,
            )

    steady = (
        reached_end
        and t_end > 0.0
        and dp_rel is not None
        and dp_rel < max_rel_change
        and phi_rel is not None
        and phi_rel < max_rel_change
        and (not xr_required or (xr_rel is not None and xr_rel < max_rel_change))
    )

    return {
        "converged": steady,
        "converged_by": "steady_10pct" if steady else None,
        "solver": solver,
        "t_end": t_end,
        "n_steps": log_info["n_steps"],
        "reached_end": reached_end,
        "continuity_error": log_info["continuity_error"],
        "dp_rel_change_10pct": dp_rel,
        "phi_out_rel_change_10pct": phi_rel,
        "xr_rel_change_10pct": xr_rel,
        "steady_window_frac": window_frac,
        "steady_tol": max_rel_change,
        "openfoam_version": log_info["openfoam_version"],
    }


def compute_cfl_thickness_star(
    s: np.ndarray,
    phi: np.ndarray,
    *,
    phi_inlet: float,
    family: str,
    threshold_frac: float = DEFAULT_CFL_THRESHOLD_FRAC,
    wall_lo: float | None = None,
    wall_hi: float | None = None,
) -> dict[str, float | None]:
    """Compute dimensionless cell-free layer (CFL) thickness ``delta_CFL*`` from outlet ``phi(s)``.

    Defined as the distance from the solid wall to the first location where ``phi`` reaches
    ``threshold_frac * phi_inlet`` (default 50 % of inlet hematocrit), linearly interpolated
    between adjacent sample points:
    - Planar channels (``G1``, ``G2``): average of lower-wall (``wall_lo``) and upper-wall
      (``wall_hi``) CFL thicknesses.
    - Axisymmetric tube (``KG``): outer-wall (``r = wall_hi``) CFL thickness inward toward ``r = 0``.
    """
    if phi_inlet <= 0.0:
        raise ValueError(f"phi_inlet must be > 0, got {phi_inlet}")
    s_arr = np.asarray(s, dtype=np.float64)
    p_arr = np.asarray(phi, dtype=np.float64)
    if s_arr.size < 2 or s_arr.size != p_arr.size:
        raise ValueError("Outlet profile must have >= 2 matching (s, phi) samples")

    order = np.argsort(s_arr)
    s_arr = s_arr[order]
    p_arr = p_arr[order]
    phi_thresh = float(threshold_frac * phi_inlet)
    fam = family.upper()

    y_lo = float(wall_lo) if wall_lo is not None else float(s_arr[0])
    y_hi = float(wall_hi) if wall_hi is not None else float(s_arr[-1])

    def _walk_from_low() -> float:
        if p_arr[0] >= phi_thresh:
            return 0.0
        for idx in range(1, s_arr.size):
            if p_arr[idx] >= phi_thresh:
                dp = p_arr[idx] - p_arr[idx - 1]
                frac = (phi_thresh - p_arr[idx - 1]) / dp if abs(dp) > 1e-15 else 0.0
                s_cross = s_arr[idx - 1] + frac * (s_arr[idx] - s_arr[idx - 1])
                return float(max(0.0, s_cross - y_lo))
        return float(0.5 * (y_hi - y_lo))

    def _walk_from_high() -> float:
        if p_arr[-1] >= phi_thresh:
            return 0.0
        for idx in range(s_arr.size - 2, -1, -1):
            if p_arr[idx] >= phi_thresh:
                dp = p_arr[idx] - p_arr[idx + 1]
                frac = (phi_thresh - p_arr[idx + 1]) / dp if abs(dp) > 1e-15 else 0.0
                s_cross = s_arr[idx + 1] + frac * (s_arr[idx] - s_arr[idx + 1])
                return float(max(0.0, y_hi - s_cross))
        return float(y_hi - y_lo if fam == "KG" else 0.5 * (y_hi - y_lo))

    if fam == "KG":
        cfl_outer = _walk_from_high()
        return {
            "cfl_thickness_star": cfl_outer,
            "cfl_bot_star": None,
            "cfl_top_star": cfl_outer,
            "cfl_threshold_phi": phi_thresh,
        }

    cfl_bot = _walk_from_low()
    cfl_top = _walk_from_high()
    return {
        "cfl_thickness_star": 0.5 * (cfl_bot + cfl_top),
        "cfl_bot_star": cfl_bot,
        "cfl_top_star": cfl_top,
        "cfl_threshold_phi": phi_thresh,
    }


def extract_m2b_case(
    case_dir: Path,
    design_record: Mapping[str, Any],
    level: int,
    wall_seconds: float,
    out_npz_path: Path,
    tau_lo: float = DEFAULT_TAU_STAR_LO,
    tau_hi: float = DEFAULT_TAU_STAR_HI,
    convergence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract star-unit multiphase fields, fixed grids, outlet profile, CFL metric, and scalars."""
    family = str(design_record["family"]).upper()
    Re = float(design_record["Re"])
    phi_inlet = float(design_record["phi"])
    mu_ratio = (
        float(design_record["mu_ratio"])
        if design_record.get("mu_ratio") is not None
        else wu2015_mu_ratio(phi_inlet)
    )
    solver = _m2b_solver_name(design_record)
    geom_params = dict(design_record.get("geometry", design_record))
    mesh = build_m2b_geometry(family, geom_params, level=level)

    t_dir = latest_time_dir(case_dir)
    time_name = t_dir.name

    c_parsed = parse_openfoam_field(t_dir / "C")
    v_parsed = parse_openfoam_field(t_dir / "V")
    a_rbc_parsed = parse_openfoam_field(t_dir / "alpha.RBC")
    u_pla_parsed = parse_openfoam_field(t_dir / "U.plasma")
    u_rbc_parsed = parse_openfoam_field(t_dir / "U.RBC")
    p_path = t_dir / "p" if (t_dir / "p").is_file() else t_dir / "p_rgh"
    p_parsed = parse_openfoam_field(p_path)
    g_pla_parsed = parse_openfoam_field(t_dir / "grad(U.plasma)")
    g_rbc_path = t_dir / "grad(U.RBC)"
    g_rbc_parsed = parse_openfoam_field(g_rbc_path) if g_rbc_path.is_file() else g_pla_parsed

    c_kind, c_arr = c_parsed["internal"]
    if c_kind != "nonuniform":
        raise ValueError(f"Expected nonuniform cell centres C in {t_dir / 'C'}")
    n_cells = int(c_arr.shape[0])

    x_star = c_arr[:, 0].astype(np.float64)
    if family == "KG":
        y_star = np.hypot(c_arr[:, 1], c_arr[:, 2]).astype(np.float64)
    else:
        y_star = c_arr[:, 1].astype(np.float64)

    cell_volumes = _expand_internal(v_parsed["internal"], n_cells)
    phi_star = _expand_internal(a_rbc_parsed["internal"], n_cells)
    u_plasma_star = _expand_internal(u_pla_parsed["internal"], n_cells)
    u_rbc_star = _expand_internal(u_rbc_parsed["internal"], n_cells)

    u_mix_path = t_dir / "U"
    if u_mix_path.is_file():
        u_star = _expand_internal(parse_openfoam_field(u_mix_path)["internal"], n_cells)
    else:
        u_star = phi_star[:, None] * u_rbc_star + (1.0 - phi_star[:, None]) * u_plasma_star

    p_of = _expand_internal(p_parsed["internal"], n_cells)
    p_outlet_ref = 0.0
    out_p_bc = p_parsed["boundary"].get("outlet")
    if out_p_bc is not None and out_p_bc.get("value") is not None:
        p_kind, p_val = out_p_bc["value"]
        p_outlet_ref = float(p_val) if p_kind == "uniform" else float(np.mean(p_val))
    p_star = Re * (p_of - p_outlet_ref)

    grad_u_plasma = _expand_internal(g_pla_parsed["internal"], n_cells)
    grad_u_rbc = _expand_internal(g_rbc_parsed["internal"], n_cells)
    gdot_plasma_star = strain_rate_magnitude(grad_u_plasma)
    gdot_rbc_star = strain_rate_magnitude(grad_u_rbc)
    gdot_star = gdot_plasma_star
    # Plasma-referenced star shear stress: tau_plasma* = 1 * gdot_plasma*,
    # mixture star shear stress (Wu 2015 Eq. 8): (1 - phi)*gdot_plasma* + phi*mu_ratio*gdot_rbc*.
    tau_plasma_star = gdot_plasma_star.copy()
    tau_star = (1.0 - phi_star) * gdot_plasma_star + phi_star * mu_ratio * gdot_rbc_star

    # Resample onto fixed per-family grid + fluid mask.
    grid_data = resample_to_fixed_grid(
        mesh,
        x_star,
        y_star,
        {
            "ux": u_star[:, 0],
            "uy": u_star[:, 1],
            "u_plasma_x": u_plasma_star[:, 0],
            "u_plasma_y": u_plasma_star[:, 1],
            "u_rbc_x": u_rbc_star[:, 0],
            "u_rbc_y": u_rbc_star[:, 1],
            "phi": phi_star,
            "p": p_star,
            "gdot": gdot_star,
            "tau": tau_star,
        },
    )

    # 1. Pressure drop dp* (plasma-referenced viscous star units).
    dp_star = _compute_dp_star(mesh, x_star, y_star, u_star, p_star, cell_volumes)

    # 2. Reattachment length x_r / h1 (or x_r / d1 for KG); None for G1.
    xr_over_h1: float | None = None
    reattach_note: str | None = None
    if family in EXPANSION_REATTACHMENT_SPECS:
        line_name, step_key = EXPANSION_REATTACHMENT_SPECS[family]
        x_step = float(mesh.params[step_key]) if step_key is not None else 0.0
        nw = read_m2b_sample_xy(case_dir, time_name, line_name)
        try:
            xr_over_h1 = float(reattachment_length(nw["coord"], nw["ux"], x_step=x_step))
        except ValueError as exc:
            xr_over_h1 = None
            reattach_note = str(exc)

    # 3. Max wall shear stress tau* from walls boundary grad(U.plasma) / grad(U.RBC).
    walls_pla = g_pla_parsed["boundary"].get("walls")
    max_wall_tau_source = "walls_patch"
    if walls_pla is not None and walls_pla.get("value") is not None:
        wp_kind, wp_arr = walls_pla["value"]
        wp_grad = wp_arr[None, :, :] if wp_kind == "uniform" else wp_arr
        w_gdot_pla = strain_rate_magnitude(wp_grad)

        walls_rbc = g_rbc_parsed["boundary"].get("walls")
        if walls_rbc is not None and walls_rbc.get("value") is not None:
            wr_kind, wr_arr = walls_rbc["value"]
            wr_grad = wr_arr[None, :, :] if wr_kind == "uniform" else wr_arr
            w_gdot_rbc = strain_rate_magnitude(wr_grad)
            if w_gdot_rbc.size == 1 and w_gdot_pla.size > 1:
                w_gdot_rbc = np.full_like(w_gdot_pla, float(w_gdot_rbc[0]))
        else:
            w_gdot_rbc = w_gdot_pla

        walls_alpha = a_rbc_parsed["boundary"].get("walls")
        if walls_alpha is not None and walls_alpha.get("value") is not None:
            wa_kind, wa_arr = walls_alpha["value"]
            w_phi = (
                np.full_like(w_gdot_pla, float(wa_arr))
                if wa_kind == "uniform"
                else np.asarray(wa_arr, dtype=np.float64)
            )
            if w_phi.size != w_gdot_pla.size:
                w_phi = np.full_like(w_gdot_pla, float(np.mean(w_phi)))
        else:
            w_phi = np.zeros_like(w_gdot_pla)

        w_tau = (1.0 - w_phi) * w_gdot_pla + w_phi * mu_ratio * w_gdot_rbc
        max_wall_tau_star = float(np.max(w_tau)) if w_tau.size > 0 else float(np.max(tau_star))
        if max_wall_tau_star <= 0.0:
            max_wall_tau_star = float(np.max(tau_star))
            max_wall_tau_source = "internal_fallback"
    else:
        max_wall_tau_star = float(np.max(tau_star))
        max_wall_tau_source = "internal_fallback"

    # 4. Outlet-section profile, phi-weighted window fraction, and CFL thickness metric.
    sl_by_name = {sl.name: sl for sl in mesh.sample_lines}
    pts_src = np.column_stack([x_star, y_star])
    gdot_p_lin = LinearNDInterpolator(pts_src, gdot_plasma_star)
    gdot_p_near = NearestNDInterpolator(pts_src, gdot_plasma_star)
    gdot_r_lin = LinearNDInterpolator(pts_src, gdot_rbc_star)
    gdot_r_near = NearestNDInterpolator(pts_src, gdot_rbc_star)
    phi_lin = LinearNDInterpolator(pts_src, phi_star)
    phi_near = NearestNDInterpolator(pts_src, phi_star)

    out_s_list: list[np.ndarray] = []
    out_un_list: list[np.ndarray] = []
    out_up_n_list: list[np.ndarray] = []
    out_phi_list: list[np.ndarray] = []
    out_gdot_list: list[np.ndarray] = []
    out_tau_list: list[np.ndarray] = []
    out_w_list: list[np.ndarray] = []
    out_branch_list: list[np.ndarray] = []

    for branch_idx, (line_name, (nx_dir, ny_dir)) in enumerate(_outlet_section_specs(mesh)):
        sl = sl_by_name[line_name]
        prof = read_m2b_sample_xy(case_dir, time_name, line_name)
        coord = prof["coord"]
        order = np.argsort(coord)
        coord = coord[order]

        px, py = _sample_line_xy(sl, coord)
        pts_line = np.column_stack([px, py])

        if "alpha_rbc" in prof:
            phi_s = np.clip(prof["alpha_rbc"][order], 0.0, 1.0)
        else:
            phi_s = phi_lin(pts_line)
            nans = np.isnan(phi_s)
            if np.any(nans):
                phi_s[nans] = phi_near(pts_line[nans])
            phi_s = np.clip(phi_s, 0.0, 1.0)

        u_rbc_s = prof["u_rbc"][order] if "u_rbc" in prof else prof["u_mix"][order]
        u_pla_s = prof["u_plasma"][order] if "u_plasma" in prof else prof["u_mix"][order]
        u_rbc_n = u_rbc_s[:, 0] * nx_dir + u_rbc_s[:, 1] * ny_dir
        u_pla_n = u_pla_s[:, 0] * nx_dir + u_pla_s[:, 1] * ny_dir

        gdot_p_s = gdot_p_lin(pts_line)
        nans_p = np.isnan(gdot_p_s)
        if np.any(nans_p):
            gdot_p_s[nans_p] = gdot_p_near(pts_line[nans_p])
        gdot_p_s = np.maximum(0.0, gdot_p_s)

        gdot_r_s = gdot_r_lin(pts_line)
        nans_r = np.isnan(gdot_r_s)
        if np.any(nans_r):
            gdot_r_s[nans_r] = gdot_r_near(pts_line[nans_r])
        gdot_r_s = np.maximum(0.0, gdot_r_s)

        tau_s = (1.0 - phi_s) * gdot_p_s + phi_s * mu_ratio * gdot_r_s

        w = trapezoid_weights(coord)
        if family == "KG":
            w = 2.0 * math.pi * np.maximum(0.0, py) * w

        out_s_list.append(coord)
        out_un_list.append(u_rbc_n)
        out_up_n_list.append(u_pla_n)
        out_phi_list.append(phi_s)
        out_gdot_list.append(gdot_p_s)
        out_tau_list.append(tau_s)
        out_w_list.append(w)
        out_branch_list.append(np.full(coord.shape, branch_idx, dtype=np.int32))

    outlet_s = np.concatenate(out_s_list)
    outlet_u_n = np.concatenate(out_un_list)
    outlet_u_plasma_n = np.concatenate(out_up_n_list)
    outlet_phi = np.concatenate(out_phi_list)
    outlet_gdot = np.concatenate(out_gdot_list)
    outlet_tau = np.concatenate(out_tau_list)
    outlet_weights = np.concatenate(out_w_list)
    outlet_branch_id = np.concatenate(out_branch_list)

    # RBC flux-weighted window fraction: weighted by alpha.RBC * u_RBC_n * dA.
    f_window = float(
        window_fraction(outlet_tau, outlet_u_n, outlet_weights, phi=outlet_phi, lo=tau_lo, hi=tau_hi)
    )

    # Exact wall coordinates for the outlet section so CFL thickness measures from the true wall.
    if family == "G1":
        wall_lo, wall_hi = 0.0, 1.0
    elif family == "G2":
        E_val = float(mesh.params["E"])
        step = 0.5 * (E_val - 1.0)
        wall_lo, wall_hi = -step, 1.0 + step
    elif family == "KG":
        wall_lo, wall_hi = 0.0, 0.5 * float(mesh.params["E"])
    else:
        wall_lo, wall_hi = float(np.min(outlet_s)), float(np.max(outlet_s))

    cfl_info = compute_cfl_thickness_star(
        outlet_s,
        outlet_phi,
        phi_inlet=phi_inlet,
        family=family,
        wall_lo=wall_lo,
        wall_hi=wall_hi,
    )
    w_sum = float(np.sum(outlet_weights))
    outlet_phi_mean = (
        float(np.sum(outlet_weights * outlet_phi) / w_sum)
        if w_sum > 0.0
        else float(np.mean(outlet_phi))
    )

    conv = (
        dict(convergence)
        if convergence is not None
        else evaluate_m2b_convergence(case_dir, family=family, solver=solver)
    )
    cm_log = parse_checkmesh_log(case_dir / "log.checkMesh")

    scalars: dict[str, float | None] = {
        "dp_star": dp_star,
        "xr_over_h1": xr_over_h1,
        "max_wall_tau_star": max_wall_tau_star,
        "f_window": f_window,
        "cfl_thickness_star": cfl_info["cfl_thickness_star"],
        "outlet_phi_mean": outlet_phi_mean,
        "max_phi": float(np.max(phi_star)),
        "min_phi": float(np.min(phi_star)),
    }

    params_payload: dict[str, Any] = {
        "run_id": design_record["run_id"],
        "case_id": design_record.get("case_id"),
        "tier": design_record.get("tier", "m2b"),
        "family": family,
        "solver": conv["solver"],
        "Re": Re,
        "phi": phi_inlet,
        "rho_ratio": design_record.get("rho_ratio"),
        "mu_ratio": mu_ratio,
        "d_p_star": design_record.get("d_p_star"),
        "drag_model": design_record.get("drag_model", "WenYu"),
        "lift_model": design_record.get(
            "lift_model",
            "none" if conv["solver"] == "multiphaseEulerFoam" else "LegendreMagnaudet",
        ),
        "geometry": mesh.params,
        "level": level,
        "n_cells": cm_log["n_cells"] if cm_log["n_cells"] is not None else n_cells,
        "t_end": conv["t_end"],
        "n_steps": conv["n_steps"],
        "continuity_error": conv["continuity_error"],
        "wall_seconds": round(float(wall_seconds), 4),
        "converged": bool(conv["converged"]),
        "converged_by": conv["converged_by"],
        "dp_rel_change_10pct": conv["dp_rel_change_10pct"],
        "phi_out_rel_change_10pct": conv["phi_out_rel_change_10pct"],
        "xr_rel_change_10pct": conv["xr_rel_change_10pct"],
        "openfoam_version": conv["openfoam_version"],
        "pressure_scaling": "viscous_plasma (p* = Re * (p_OF - p_outlet))",
        "stress_definition": "mixture_star: (1 - phi)*gdot_plasma* + phi*mu_ratio*gdot_rbc*",
        "window_weighting": "rbc_flux (alpha.RBC * u_RBC_n * dA)",
        "cfl_definition": {
            "threshold_frac": DEFAULT_CFL_THRESHOLD_FRAC,
            "threshold_phi": cfl_info["cfl_threshold_phi"],
            "cfl_bot_star": cfl_info["cfl_bot_star"],
            "cfl_top_star": cfl_info["cfl_top_star"],
        },
        "max_wall_tau_source": max_wall_tau_source,
        "tau_window_star": [float(tau_lo), float(tau_hi)],
        "code_version": design_record.get("code_version", code_version()),
        "scalars": scalars,
    }
    if reattach_note is not None:
        params_payload["reattachment_note"] = reattach_note

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        x_star=x_star,
        y_star=y_star,
        cell_volumes=cell_volumes,
        U_star=u_star,
        U_plasma_star=u_plasma_star,
        U_rbc_star=u_rbc_star,
        phi_star=phi_star,
        p_star=p_star,
        gdot_star=gdot_star,
        gdot_plasma_star=gdot_plasma_star,
        gdot_rbc_star=gdot_rbc_star,
        tau_star=tau_star,
        tau_plasma_star=tau_plasma_star,
        outlet_s=outlet_s,
        outlet_u_n=outlet_u_n,
        outlet_u_plasma_n=outlet_u_plasma_n,
        outlet_phi=outlet_phi,
        outlet_gdot=outlet_gdot,
        outlet_tau=outlet_tau,
        outlet_weights=outlet_weights,
        outlet_branch_id=outlet_branch_id,
        params=json.dumps(params_payload),
        **grid_data,
    )

    sha256_hex = compute_sha256(out_npz_path)
    return {
        "scalars": scalars,
        "params": params_payload,
        "sha256": sha256_hex,
        "npz_path": out_npz_path,
    }


def extract_case(
    case_dir: Path,
    design_record: Mapping[str, Any],
    level: int,
    wall_seconds: float,
    out_npz_path: Path,
    tau_lo: float = DEFAULT_TAU_STAR_LO,
    tau_hi: float = DEFAULT_TAU_STAR_HI,
    convergence: Mapping[str, Any] | None = None,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
) -> dict[str, Any]:
    """Extract star-unit fields, fixed-grid arrays, outlet profile, and scalars to ``.npz``.

    Returns a dictionary with ``scalars``, ``params``, ``sha256``, and ``npz_path``.
    """
    if _is_m2b_record(design_record):
        return extract_m2b_case(
            case_dir=case_dir,
            design_record=design_record,
            level=level,
            wall_seconds=wall_seconds,
            out_npz_path=out_npz_path,
            tau_lo=tau_lo,
            tau_hi=tau_hi,
            convergence=convergence,
        )

    family = str(design_record["family"]).upper()
    Re = float(design_record["Re"])
    geom_params = dict(design_record.get("geometry", design_record))
    mesh = (
        build_m2b_geometry(family, geom_params, level=level)
        if family == "G1"
        else build_geometry(family, geom_params, level=level)
    )

    t_dir = latest_time_dir(case_dir)
    time_name = t_dir.name

    c_parsed = parse_openfoam_field(t_dir / "C")
    v_parsed = parse_openfoam_field(t_dir / "V")
    u_parsed = parse_openfoam_field(t_dir / "U")
    p_parsed = parse_openfoam_field(t_dir / "p")
    gu_parsed = parse_openfoam_field(t_dir / "grad(U)")

    c_kind, c_arr = c_parsed["internal"]
    if c_kind != "nonuniform":
        raise ValueError(f"Expected nonuniform cell centres C in {t_dir / 'C'}")
    n_cells = int(c_arr.shape[0])

    x_star = c_arr[:, 0].astype(np.float64)
    if family == "KG":
        y_star = np.hypot(c_arr[:, 1], c_arr[:, 2]).astype(np.float64)
    else:
        y_star = c_arr[:, 1].astype(np.float64)

    cell_volumes = _expand_internal(v_parsed["internal"], n_cells)
    u_star = _expand_internal(u_parsed["internal"], n_cells)
    p_of = _expand_internal(p_parsed["internal"], n_cells)
    grad_u = _expand_internal(gu_parsed["internal"], n_cells)

    # Convert OpenFOAM kinematic pressure p_OF = p/(rho U^2) to viscous star pressure p* = Re * p_OF.
    p_star = Re * p_of
    gdot_star = strain_rate_magnitude(grad_u)
    tau_star = shear_stress_star(gdot_star, design_record)

    # Resample onto fixed per-family grid + fluid mask.
    grid_data = resample_to_fixed_grid(
        mesh,
        x_star,
        y_star,
        {
            "ux": u_star[:, 0],
            "uy": u_star[:, 1],
            "p": p_star,
            "gdot": gdot_star,
            "tau": tau_star,
        },
    )

    # 1. Pressure drop dp* (viscous star units).
    dp_star = _compute_dp_star(mesh, x_star, y_star, u_star, p_star, cell_volumes)

    # 2. Reattachment length x_r / h1 (or x_r / d1 for KG) for expansion-type families.
    xr_over_h1: float | None = None
    reattach_note: str | None = None
    if family in EXPANSION_REATTACHMENT_SPECS:
        line_name, step_key = EXPANSION_REATTACHMENT_SPECS[family]
        x_step = float(mesh.params[step_key]) if step_key is not None else 0.0
        nw = read_sample_xy(case_dir, time_name, line_name)
        try:
            xr_over_h1 = float(reattachment_length(nw["coord"], nw["ux"], x_step=x_step))
        except ValueError as exc:
            xr_over_h1 = None
            reattach_note = str(exc)

    # 3. Max wall shear stress tau* from grad(U) on walls patch (fallback to internal max).
    walls_entry = gu_parsed["boundary"].get("walls")
    max_wall_tau_source = "walls_patch"
    if walls_entry is not None and walls_entry.get("value") is not None:
        w_kind, w_arr = walls_entry["value"]
        w_grad = w_arr[None, :, :] if w_kind == "uniform" else w_arr
        w_gdot = strain_rate_magnitude(w_grad)
        w_tau = shear_stress_star(w_gdot, design_record)
        max_wall_tau_star = float(np.max(w_tau)) if w_tau.size > 0 else float(np.max(tau_star))
        if max_wall_tau_star <= 0.0:
            max_wall_tau_star = float(np.max(tau_star))
            max_wall_tau_source = "internal_fallback"
    else:
        max_wall_tau_star = float(np.max(tau_star))
        max_wall_tau_source = "internal_fallback"

    # 4. Outlet-section profile + flux-weighted window fraction for [tau_lo, tau_hi].
    sl_by_name = {sl.name: sl for sl in mesh.sample_lines}
    pts_src = np.column_stack([x_star, y_star])
    gdot_lin = LinearNDInterpolator(pts_src, gdot_star)
    gdot_near = NearestNDInterpolator(pts_src, gdot_star)

    out_s_list: list[np.ndarray] = []
    out_un_list: list[np.ndarray] = []
    out_gdot_list: list[np.ndarray] = []
    out_tau_list: list[np.ndarray] = []
    out_w_list: list[np.ndarray] = []
    out_branch_list: list[np.ndarray] = []

    for branch_idx, (line_name, (nx_dir, ny_dir)) in enumerate(_outlet_section_specs(mesh)):
        sl = sl_by_name[line_name]
        prof = read_sample_xy(case_dir, time_name, line_name)
        coord = prof["coord"]
        order = np.argsort(coord)
        coord = coord[order]
        ux_s = prof["ux"][order]
        uy_s = prof["uy"][order]
        u_n = ux_s * nx_dir + uy_s * ny_dir

        px, py = _sample_line_xy(sl, coord)
        pts_line = np.column_stack([px, py])
        gdot_s = gdot_lin(pts_line)
        nans = np.isnan(gdot_s)
        if np.any(nans):
            gdot_s[nans] = gdot_near(pts_line[nans])
        gdot_s = np.maximum(0.0, gdot_s)
        tau_s = shear_stress_star(gdot_s, design_record)

        w = trapezoid_weights(coord)
        if family == "KG":
            # Axisymmetric area element dA = 2 * pi * r * dr
            w = 2.0 * math.pi * np.maximum(0.0, py) * w

        out_s_list.append(coord)
        out_un_list.append(u_n)
        out_gdot_list.append(gdot_s)
        out_tau_list.append(tau_s)
        out_w_list.append(w)
        out_branch_list.append(np.full(coord.shape, branch_idx, dtype=np.int32))

    outlet_s = np.concatenate(out_s_list)
    outlet_u_n = np.concatenate(out_un_list)
    outlet_gdot = np.concatenate(out_gdot_list)
    outlet_tau = np.concatenate(out_tau_list)
    outlet_weights = np.concatenate(out_w_list)
    outlet_branch_id = np.concatenate(out_branch_list)

    f_window = float(
        window_fraction(outlet_tau, outlet_u_n, outlet_weights, lo=tau_lo, hi=tau_hi)
    )

    conv = (
        dict(convergence)
        if convergence is not None
        else evaluate_convergence(case_dir, residual_tol=residual_tol)
    )
    cm_log = parse_checkmesh_log(case_dir / "log.checkMesh")

    scalars: dict[str, float | None] = {
        "dp_star": dp_star,
        "xr_over_h1": xr_over_h1,
        "max_wall_tau_star": max_wall_tau_star,
        "f_window": f_window,
    }

    params_payload: dict[str, Any] = {
        "run_id": design_record["run_id"],
        "case_id": design_record.get("case_id"),
        "family": family,
        "rheology": design_record.get("rheology", "Newtonian"),
        "Re": Re,
        "beta": design_record.get("beta"),
        "Cu": design_record.get("Cu"),
        "n": design_record.get("n"),
        "a": design_record.get("a"),
        "geometry": mesh.params,
        "seed": design_record.get("seed"),
        "level": level,
        "n_cells": cm_log["n_cells"] if cm_log["n_cells"] is not None else n_cells,
        "iterations": conv["iterations"],
        "final_residuals": conv["final_residuals"],
        "continuity_error": conv["continuity_error"],
        "wall_seconds": round(float(wall_seconds), 4),
        "converged": bool(conv["converged"]),
        "converged_by": conv["converged_by"],
        "dp_rel_change_500": conv["dp_rel_change_500"],
        "u_out_rel_change_500": conv["u_out_rel_change_500"],
        "solver": "simpleFoam",
        "openfoam_version": conv["openfoam_version"],
        "pressure_scaling": "viscous (p* = Re * p_OF)",
        "max_wall_tau_source": max_wall_tau_source,
        "tau_window_star": [float(tau_lo), float(tau_hi)],
        "code_version": design_record.get("code_version", code_version()),
        "scalars": scalars,
    }
    # Paired-design provenance: device-space scale and variant labels travel with the fields so
    # post-processing never has to re-open the design file.
    for key in (
        "design_id",
        "pair_id",
        "variant",
        "phi",
        "h_um",
        "U_mps",
        "mu_carrier",
        "mu_eff",
        "viscosity_ratio",
        "tau_ref",
        "t_ref",
        "d_rbc_over_h",
        "relax_p",
        "relax_u",
        "retry_of",
    ):
        if key in design_record:
            params_payload[key] = design_record[key]
    if reattach_note is not None:
        params_payload["reattachment_note"] = reattach_note

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        x_star=x_star,
        y_star=y_star,
        cell_volumes=cell_volumes,
        U_star=u_star,
        p_star=p_star,
        gdot_star=gdot_star,
        tau_star=tau_star,
        outlet_s=outlet_s,
        outlet_u_n=outlet_u_n,
        outlet_gdot=outlet_gdot,
        outlet_tau=outlet_tau,
        outlet_weights=outlet_weights,
        outlet_branch_id=outlet_branch_id,
        params=json.dumps(params_payload),
        **grid_data,
    )

    sha256_hex = compute_sha256(out_npz_path)
    return {
        "scalars": scalars,
        "params": params_payload,
        "sha256": sha256_hex,
        "npz_path": out_npz_path,
    }


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Provenance (code_version) and complete-case archiving
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Return the repository root inferred from this module's location."""
    return Path(__file__).resolve().parent.parent


def _hash_code_files(root_str: str) -> str:
    root = Path(root_str)
    files: list[Path] = []
    for pattern in CODE_VERSION_PATTERNS:
        files.extend(f for f in root.glob(pattern) if f.is_file())
    if not files:
        raise FileNotFoundError(f"no code files matched {CODE_VERSION_PATTERNS} under {root}")
    digest = hashlib.sha256()
    for path in sorted({f.resolve() for f in files}):
        digest.update(path.relative_to(root.resolve()).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_version(root: Path | str | None = None) -> str:
    """Return the SHA-256 provenance hash of ``cfd/gen`` and ``campaign`` code (T7 Part 1).

    The hash is stable across processes and machines and only changes when one of the
    generator/runner files changes, so every manifest entry and ``.npz`` ``params`` can
    record which code produced it.
    """
    root_path = Path(root) if root is not None else repo_root()
    return _hash_code_files(str(root_path.resolve()))


_NUMERIC_DIR_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _archive_entries(case_dir: Path) -> list[Path]:
    """Return the case-directory entries to archive: mesh, dicts, ``0``, final time, logs, fields.

    Older time directories are skipped (plan/CORPUS_PLAN.md §6.1: final time only). Returns
    an empty list when the case directory does not exist, so a crashed generation still
    produces a (possibly empty) archive instead of raising.
    """
    if not case_dir.is_dir():
        return []
    try:
        final_time = latest_time_dir(case_dir).name
    except (FileNotFoundError, ValueError):
        final_time = None
    entries: list[Path] = []
    for child in sorted(case_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            if child.name.startswith("processor"):
                continue
            if _NUMERIC_DIR_RE.match(child.name):
                # Keep the initial condition (0, 0.orig, ...) and the final time only.
                if child.name.startswith("0") or (
                    final_time is not None and child.name == final_time
                ):
                    entries.append(child)
                continue
            entries.append(child)
        else:
            entries.append(child)
    return entries


def archive_case(case_dir: Path, out_tar: Path) -> Path:
    """Write ``<run_id>.tar.gz`` of the whole case directory (mesh, dicts, fields, logs)."""
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, mode="w:gz") as tar:
        for entry in _archive_entries(case_dir):
            tar.add(entry, arcname=entry.name, recursive=True)
    return out_tar


# ---------------------------------------------------------------------------
# GCS sync/upload + manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load JSONL manifest entries from disk (returns empty list if missing)."""
    if not manifest_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def append_manifest_row(manifest_path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object line to ``manifest.jsonl``."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False) + "\n")


def dedupe_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse append-only manifest rows to one row per ``run_id``, keeping the latest.

    The manifest is append-only (retries add a row), so a ``failed`` row can be followed by an
    ``ok`` row for the same ``run_id``. The last occurrence in file order wins (T7 v2.3).
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row.get("run_id"))] = row
    return list(latest.values())


def write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite ``manifest_path`` with exactly ``rows`` (used by ``--dedupe-manifest``)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=False) + "\n")
    tmp.replace(manifest_path)


def scalar_reproducibility_diffs(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    rel_tol: float = REPRODUCIBILITY_REL_TOL,
) -> dict[str, dict[str, float]]:
    """Report scalars whose re-run value changed by more than ``rel_tol`` relative.

    Returns ``{scalar_name: {"previous": .., "current": .., "rel_change": ..}}`` for scalars that
    moved (runs must be reproducible), ignoring scalars that are ``None`` on either side.
    """
    diffs: dict[str, dict[str, float]] = {}
    if not previous or not current:
        return diffs
    for key, cur in current.items():
        prev = previous.get(key)
        if cur is None or prev is None:
            continue
        cur_f, prev_f = float(cur), float(prev)
        scale = max(abs(prev_f), abs(cur_f), 1e-30)
        rel = abs(cur_f - prev_f) / scale
        if rel > rel_tol:
            diffs[key] = {"previous": prev_f, "current": cur_f, "rel_change": rel}
    return diffs



def gcs_download_if_exists(gcs_uri: str, local_path: Path) -> bool:
    """Download ``gcs_uri`` to ``local_path`` if it exists in the bucket."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["gcloud", "storage", "cp", gcs_uri, str(local_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def gcs_upload_file(local_path: Path, gcs_uri: str) -> None:
    """Upload a local file to GCS via ``gcloud storage cp``."""
    proc = subprocess.run(
        ["gcloud", "storage", "cp", str(local_path), gcs_uri],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud storage cp failed for {gcs_uri}: {proc.stderr.strip()}")


# ---------------------------------------------------------------------------
# Single-case worker and parallel batch coordinator
# ---------------------------------------------------------------------------

def _default_case_runner(case_dir: Path, timeout: float) -> tuple[bool, str | None]:
    """Execute ``bash run_case.sh`` inside ``case_dir``."""
    try:
        proc = subprocess.run(
            ["bash", "run_case.sh"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout:.0f}s"
    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        reason = f"run_case.sh exit {proc.returncode}: " + " | ".join(err_tail)
        return False, reason
    return True, None


def _execute_one_case(
    record: dict[str, Any],
    level: int,
    cases_root: Path,
    npz_root: Path,
    end_time: int,
    residual_tol: float,
    timeout: float,
    tau_lo: float,
    tau_hi: float,
    runner_fn: Callable[[Path, float], tuple[bool, str | None]] | None = None,
) -> dict[str, Any]:
    """Generate, run, and extract a single OpenFOAM case (1 core)."""
    run_id = str(record["run_id"])
    family = str(record["family"]).upper()
    # Provenance travels into every params payload (npz and manifest), including failures.
    record = {**record, "code_version": code_version()}
    case_dir = cases_root / family / run_id
    out_npz = npz_root / family / f"{run_id}.npz"
    t0 = time.monotonic()

    try:
        if case_dir.exists():
            shutil.rmtree(case_dir)

        if _is_m2b_record(record):
            solver = _m2b_solver_name(record)
            if solver == "reactingTwoPhaseEulerFoam":
                write_m2b_lift_case(case_dir, record, level=level)
            else:
                write_m2b_case(case_dir, record, level=level)

            run_fn = runner_fn if runner_fn is not None else _default_case_runner
            ok, fail_reason = run_fn(case_dir, timeout)
            wall_s = time.monotonic() - t0
            if not ok:
                return {
                    "run_id": run_id,
                    "case_id": record.get("case_id"),
                    "family": family,
                    "level": level,
                    "status": "failed",
                    "reason": fail_reason or "solver failed",
                    "converged_by": None,
                    "params": {**record, "level": level, "wall_seconds": round(wall_s, 4)},
                    "scalars": None,
                    "sha256": None,
                    "npz_path": None,
                    "case_dir": str(case_dir),
                }

            conv = evaluate_m2b_convergence(case_dir, family=family, solver=solver)
            if not conv["converged"]:
                return {
                    "run_id": run_id,
                    "case_id": record.get("case_id"),
                    "family": family,
                    "level": level,
                    "status": "failed",
                    "reason": "not_steady",
                    "converged_by": None,
                    "params": {
                        **record,
                        "level": level,
                        "wall_seconds": round(wall_s, 4),
                        "solver": conv["solver"],
                        "t_end": conv["t_end"],
                        "n_steps": conv["n_steps"],
                        "reached_end": conv["reached_end"],
                        "continuity_error": conv["continuity_error"],
                        "converged": False,
                        "converged_by": None,
                        "dp_rel_change_10pct": conv["dp_rel_change_10pct"],
                        "phi_out_rel_change_10pct": conv["phi_out_rel_change_10pct"],
                        "xr_rel_change_10pct": conv["xr_rel_change_10pct"],
                    },
                    "scalars": None,
                    "sha256": None,
                    "npz_path": None,
                    "case_dir": str(case_dir),
                }

            extracted = extract_m2b_case(
                case_dir=case_dir,
                design_record=record,
                level=level,
                wall_seconds=wall_s,
                out_npz_path=out_npz,
                tau_lo=tau_lo,
                tau_hi=tau_hi,
                convergence=conv,
            )
            return {
                "run_id": run_id,
                "case_id": record.get("case_id"),
                "family": family,
                "level": level,
                "status": "ok",
                "reason": None,
                "converged_by": conv["converged_by"],
                "params": extracted["params"],
                "scalars": extracted["scalars"],
                "sha256": extracted["sha256"],
                "npz_path": str(extracted["npz_path"]),
                "case_dir": str(case_dir),
            }

        write_case(case_dir, record, level=level, end_time=end_time, residual_tol=residual_tol)

        run_fn = runner_fn if runner_fn is not None else _default_case_runner
        ok, fail_reason = run_fn(case_dir, timeout)
        wall_s = time.monotonic() - t0
        if not ok:
            return {
                "run_id": run_id,
                "case_id": record.get("case_id"),
                "family": family,
                "level": level,
                "status": "failed",
                "reason": fail_reason or "solver failed",
                "converged_by": None,
                "params": {**record, "level": level, "wall_seconds": round(wall_s, 4)},
                "scalars": None,
                "sha256": None,
                "npz_path": None,
                "case_dir": str(case_dir),
            }

        conv = evaluate_convergence(case_dir, residual_tol=residual_tol)
        if not conv["converged"]:
            return {
                "run_id": run_id,
                "case_id": record.get("case_id"),
                "family": family,
                "level": level,
                "status": "failed",
                "reason": "not_converged",
                "converged_by": None,
                "params": {
                    **record,
                    "level": level,
                    "wall_seconds": round(wall_s, 4),
                    "iterations": conv["iterations"],
                    "final_residuals": conv["final_residuals"],
                    "continuity_error": conv["continuity_error"],
                    "converged": False,
                    "converged_by": None,
                    "dp_rel_change_500": conv["dp_rel_change_500"],
                    "u_out_rel_change_500": conv["u_out_rel_change_500"],
                },
                "scalars": None,
                "sha256": None,
                "npz_path": None,
                "case_dir": str(case_dir),
            }

        extracted = extract_case(
            case_dir=case_dir,
            design_record=record,
            level=level,
            wall_seconds=wall_s,
            out_npz_path=out_npz,
            tau_lo=tau_lo,
            tau_hi=tau_hi,
            convergence=conv,
            residual_tol=residual_tol,
        )
        return {
            "run_id": run_id,
            "case_id": record.get("case_id"),
            "family": family,
            "level": level,
            "status": "ok",
            "reason": None,
            "converged_by": conv["converged_by"],
            "params": extracted["params"],
            "scalars": extracted["scalars"],
            "sha256": extracted["sha256"],
            "npz_path": str(extracted["npz_path"]),
            "case_dir": str(case_dir),
        }
    except Exception as exc:  # noqa: BLE001
        wall_s = time.monotonic() - t0
        return {
            "run_id": run_id,
            "case_id": record.get("case_id"),
            "family": family,
            "level": level,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "converged_by": None,
            "params": {**record, "level": level, "wall_seconds": round(wall_s, 4)},
            "scalars": None,
            "sha256": None,
            "npz_path": None,
            "case_dir": str(case_dir),
        }


def load_design_jsonl(design_path: Path, ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Read design records from a JSONL file, optionally filtering by ``run_id``."""
    records: list[dict[str, Any]] = []
    with design_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            if ids is None or str(rec["run_id"]) in ids:
                records.append(rec)
    return records


def run_batch(
    design_path: Path,
    *,
    work_dir: Path,
    level_override: int | None = None,
    ids: set[str] | None = None,
    workers: int | None = None,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    tier: str = DEFAULT_TIER,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    upload: bool = True,
    keep_cases: bool = False,
    archive: bool = True,
    rerun_archive: bool = False,
    end_time: int = 5000,
    residual_tol: float = DEFAULT_RESIDUAL_TOL,
    timeout: float = 1800.0,
    tau_lo: float = DEFAULT_TAU_STAR_LO,
    tau_hi: float = DEFAULT_TAU_STAR_HI,
    runner_fn: Callable[[Path, float], tuple[bool, str | None]] | None = None,
    uploader_fn: Callable[[Path, str], None] | None = None,
    downloader_fn: Callable[[str, Path], bool] | None = None,
) -> dict[str, Any]:
    """Run a resumable parallel batch of OpenFOAM cases with archiving, extraction and GCS upload.

    Every finished run (``ok`` or ``failed``) is archived as ``<run_id>.tar.gz`` of the whole case
    directory (mesh, dicts, ``0``, final time, logs, ``postProcessing``) and uploaded next to the
    ``.npz``; the manifest entry carries ``case_archive`` and ``case_archive_sha256``. A successful
    case directory is deleted only after its archive reached the bucket. ``rerun_archive``
    re-runs runs that are ``ok`` in the manifest but have no archive and reports any scalar that
    changed by more than ``REPRODUCIBILITY_REL_TOL`` relative.
    """
    work_dir = Path(work_dir)
    cases_root = work_dir / "cases"
    npz_root = work_dir / "npz"
    archive_root = work_dir / "archives"
    manifest_path = work_dir / manifest_name
    work_dir.mkdir(parents=True, exist_ok=True)

    bucket_clean = bucket.rstrip("/")
    prefix_clean = prefix.strip("/")
    manifest_uri = f"{bucket_clean}/{prefix_clean}/{manifest_name}"

    dl = downloader_fn if downloader_fn is not None else gcs_download_if_exists
    ul = uploader_fn if uploader_fn is not None else gcs_upload_file

    # Sync existing manifest from GCS for resumability (merge with local if present).
    if upload:
        remote_tmp = work_dir / f".remote_{manifest_name}"
        if dl(manifest_uri, remote_tmp):
            if not manifest_path.is_file():
                shutil.move(str(remote_tmp), str(manifest_path))
            else:
                # Merge any remote rows whose (run_id, status, sha256) are missing locally.
                local_rows = load_manifest(manifest_path)
                local_keys = {(r.get("run_id"), r.get("status"), r.get("sha256")) for r in local_rows}
                for r in load_manifest(remote_tmp):
                    key = (r.get("run_id"), r.get("status"), r.get("sha256"))
                    if key not in local_keys:
                        append_manifest_row(manifest_path, r)
                        local_keys.add(key)
                remote_tmp.unlink(missing_ok=True)

    # Latest row per run_id (the manifest is append-only, retries add a row).
    existing_rows = load_manifest(manifest_path)
    latest_by_id = {str(r["run_id"]): r for r in dedupe_manifest_rows(existing_rows)}
    completed_ids = {rid for rid, r in latest_by_id.items() if r.get("status") == "ok"}
    needs_archive_ids: set[str] = (
        {rid for rid in completed_ids if not latest_by_id[rid].get("case_archive")}
        if rerun_archive
        else set()
    )
    completed_ids -= needs_archive_ids

    all_records = load_design_jsonl(design_path, ids=ids)
    todo: list[tuple[dict[str, Any], int]] = []
    skipped = 0
    skipped_external = 0
    for rec in all_records:
        # V3/V4 records are produced by the FEniCSx solver (plan/CORPUS_PLAN.md §6.2), never here.
        if str(rec.get("solver") or "") == "fenicsx":
            skipped_external += 1
            continue
        if str(rec["run_id"]) in completed_ids:
            skipped += 1
            continue
        lvl = level_override if level_override is not None else int(rec.get("level", 1))
        todo.append((rec, lvl))

    n_workers = max(1, workers if workers is not None else (os.cpu_count() or 1))
    summary: dict[str, Any] = {
        "total_selected": len(all_records),
        "skipped": skipped,
        "skipped_external_solver": skipped_external,
        "attempted": len(todo),
        "succeeded": 0,
        "failed": 0,
        "archived": 0,
        "archive_bytes": 0,
        "rerun_archive_selected": sorted(needs_archive_ids) if rerun_archive else [],
        "reproducibility_diffs": {},
        "manifest_path": str(manifest_path),
        "code_version": code_version(),
    }

    if not todo:
        return summary

    def _finalize_result(res: dict[str, Any]) -> None:
        family = res["family"]
        run_id = res["run_id"]
        case_dir = Path(res["case_dir"])
        npz_local = Path(res["npz_path"]) if res["npz_path"] else None
        npz_uri: str | None = None

        if res["status"] == "ok" and npz_local is not None:
            if upload:
                npz_uri = f"{bucket_clean}/{prefix_clean}/runs/{tier}/{family}/{run_id}.npz"
                try:
                    ul(npz_local, npz_uri)
                except Exception as exc:  # noqa: BLE001
                    res["status"] = "failed"
                    res["reason"] = f"upload failed: {exc}"
                    npz_uri = None
            else:
                npz_uri = str(npz_local)

        # Complete-case archive (mesh, dicts, 0/, final time, logs, postProcessing). Built for
        # ok and failed runs so nothing about a simulation is lost (T7 Part 1).
        case_archive_uri: str | None = None
        case_archive_sha: str | None = None
        if archive and case_dir.is_dir():
            tar_local = archive_root / family / f"{run_id}.tar.gz"
            try:
                archive_case(case_dir, tar_local)
                case_archive_sha = compute_sha256(tar_local)
                size = tar_local.stat().st_size
                if upload:
                    case_archive_uri = (
                        f"{bucket_clean}/{prefix_clean}/cases/{tier}/{family}/{run_id}.tar.gz"
                    )
                    ul(tar_local, case_archive_uri)
                else:
                    case_archive_uri = str(tar_local)
                summary["archived"] += 1
                summary["archive_bytes"] += size
                tar_local.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                case_archive_uri = None
                case_archive_sha = None
                res["archive_error"] = f"{type(exc).__name__}: {exc}"

        # Reproducibility check for archive re-runs: same run_id, must give the same scalars.
        previous = latest_by_id.get(run_id)
        if rerun_archive and previous is not None and res["status"] == "ok":
            diffs = scalar_reproducibility_diffs(previous.get("scalars"), res["scalars"])
            if diffs:
                summary["reproducibility_diffs"][run_id] = diffs

        manifest_row = {
            "run_id": run_id,
            "case_id": res.get("case_id"),
            "family": family,
            "level": res["level"],
            "status": res["status"],
            "reason": res["reason"],
            "converged_by": res.get("converged_by") if res["status"] == "ok" else None,
            "params": res["params"],
            "scalars": res["scalars"],
            "sha256": res["sha256"] if res["status"] == "ok" else None,
            "npz_uri": npz_uri,
            "case_archive": case_archive_uri,
            "case_archive_sha256": case_archive_sha,
            "code_version": code_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_manifest_row(manifest_path, manifest_row)
        latest_by_id[run_id] = manifest_row

        if upload:
            ul(manifest_path, manifest_uri)

        if res["status"] == "ok":
            summary["succeeded"] += 1
            if not keep_cases and case_dir.exists():
                shutil.rmtree(case_dir)
        else:
            summary["failed"] += 1

    # Use ThreadPoolExecutor when a custom in-process runner_fn is injected (e.g. unit tests)
    # or ProcessPoolExecutor in production to parallelize both subprocesses and scipy extraction.
    pool_cls = (
        concurrent.futures.ThreadPoolExecutor
        if runner_fn is not None or n_workers == 1
        else concurrent.futures.ProcessPoolExecutor
    )
    with pool_cls(max_workers=n_workers) as pool:
        future_map = {
            pool.submit(
                _execute_one_case,
                rec,
                lvl,
                cases_root,
                npz_root,
                end_time,
                residual_tol,
                timeout,
                tau_lo,
                tau_hi,
                runner_fn,
            ): rec["run_id"]
            for rec, lvl in todo
        }
        for fut in concurrent.futures.as_completed(future_map):
            res = fut.result()
            _finalize_result(res)
            status_tag = res["status"].upper()
            print(
                f"[{status_tag}] {res['run_id']} (fam={res['family']}, L={res['level']})"
                + (f" — {res['reason']}" if res["reason"] else ""),
                flush=True,
            )

    return summary


def main() -> None:
    """CLI entrypoint for batch OpenFOAM execution and GCS ingestion."""
    parser = argparse.ArgumentParser(
        description="Resumable parallel OpenFOAM batch runner, field extractor, and GCS uploader."
    )
    parser.add_argument("--design", type=Path, required=True, help="Path to design .jsonl file")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        choices=(0, 1, 2, 3),
        help="Override mesh level for all runs (default: use record['level'] or 1)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Optional comma-separated list of run_ids to execute",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count (default: nproc, 1 core per case)",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("work_batch"))
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--tier", type=str, default=DEFAULT_TIER)
    parser.add_argument("--manifest-name", type=str, default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--end-time", type=int, default=5000)
    parser.add_argument("--residual-tol", type=float, default=DEFAULT_RESIDUAL_TOL)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--tau-lo", type=float, default=DEFAULT_TAU_STAR_LO)
    parser.add_argument("--tau-hi", type=float, default=DEFAULT_TAU_STAR_HI)
    parser.add_argument("--no-upload", action="store_true", help="Skip GCS sync/upload")
    parser.add_argument("--keep-cases", action="store_true", help="Keep local case dirs on success")
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not write <run_id>.tar.gz of the case directory (not recommended)",
    )
    parser.add_argument(
        "--rerun-archive",
        action="store_true",
        help="Re-run run_ids that are ok in the manifest but have no case_archive",
    )
    parser.add_argument(
        "--dedupe-manifest",
        action="store_true",
        help="Rewrite the work-dir manifest with one (latest) row per run_id, then exit",
    )
    args = parser.parse_args()

    if args.dedupe_manifest:
        manifest_path = args.work_dir / args.manifest_name
        rows = load_manifest(manifest_path)
        deduped = dedupe_manifest_rows(rows)
        write_manifest(manifest_path, deduped)
        print(
            json.dumps(
                {"before": len(rows), "after": len(deduped), "manifest_path": str(manifest_path)},
                indent=2,
            )
        )
        return

    ids = {tok.strip() for tok in args.ids.split(",") if tok.strip()} if args.ids else None
    summary = run_batch(
        args.design,
        work_dir=args.work_dir,
        level_override=args.level,
        ids=ids,
        workers=args.workers,
        bucket=args.bucket,
        prefix=args.prefix,
        tier=args.tier,
        manifest_name=args.manifest_name,
        upload=not args.no_upload,
        keep_cases=args.keep_cases,
        archive=not args.no_archive,
        rerun_archive=args.rerun_archive,
        end_time=args.end_time,
        residual_tol=args.residual_tol,
        timeout=args.timeout,
        tau_lo=args.tau_lo,
        tau_hi=args.tau_hi,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
