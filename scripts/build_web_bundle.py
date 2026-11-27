#!/usr/bin/env python3
"""Build the compact web corpus bundle under ``site/data/`` from a ``corpus_flowfields`` dir.

Reads ``<src>/G<k>/{V1,V2,V4}/*.npz`` + ``*.txt`` (G1-G13, 1,139 runs) and writes:

- ``<out>/index.json``  -- family descriptions, slider metadata (param names/bounds),
  exact run records & BC/QoI summaries (no heavy arrays).
- ``<out>/G<k>.bin.gz`` -- per-family gzipped JSON: each run's downsampled grid
  (max dim <= 128, block-mean), packed-bit fluid mask, and uint8-quantized
  ``ux/uy/p/phi`` fields with float32 [min,max] scales.

Total ``site/data/`` stays well under 18 MB.  Run directly:
    python scripts/build_web_bundle.py --src <corpus_flowfields> --out site/data
or via the CLI:  hemoflow build-web-bundle --src <corpus_flowfields> --out site/data
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

# make `import hemoflow` work when run from the repo root (for family metadata)
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from hemoflow.geometries.families import FAMILIES, FLOW_PARAMS, VARIANTS
except Exception:  # pragma: no cover - fallback if hemoflow not installed
    FAMILIES, FLOW_PARAMS, VARIANTS = {}, [], {}

FIELD_KEYS = ("grid_ux", "grid_uy", "grid_p", "grid_phi")
MAX_DIM = 128
QOI_KEYS = ("dp_star", "xr_over_h1", "cfl_over_h_local", "cfl_thickness_star",
            "f_window", "max_phi", "min_phi", "max_wall_tau_star", "max_slip", "outlet_phi_mean")


def _factor(dim: int, cap: int = MAX_DIM) -> int:
    return max(1, -(-dim // cap))  # ceil(dim / cap)


def _block_mean(arr: np.ndarray, fx: int, fy: int) -> np.ndarray:
    ny, nx = arr.shape
    ny2, nx2 = ny // fy, nx // fx
    a = arr[: ny2 * fy, : nx2 * fx].astype(np.float32)
    return a.reshape(ny2, fy, nx2, fx).mean(axis=(1, 3))


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def parse_run_txt(txt_path: Path) -> dict:
    """Parse a per-run BC/QoI .txt into a flat dict."""
    text = txt_path.read_text(errors="replace")

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, text)
        if not m:
            return default
        raw = m.group(1).strip()
        if raw.lower() == "none":
            return None
        try:
            return cast(raw)
        except Exception:
            return default

    out: dict = {}
    out["run_id"] = grab(r"Run ID\s*:\s*(\S+)", str, txt_path.stem)
    out["pair_id"] = grab(r"Pair ID\s*:\s*(\S+)", str, None)
    out["status"] = grab(r"Status\s*:\s*(\S+)", str, "ok")
    m = re.search(r"Family / Variant\s*:\s*(\S+)\s*/\s*(\S+)", text)
    if m:
        out["family"] = m.group(1)
        out["variant"] = m.group(2).split()[0]
    out["h_um"] = grab(r"h1 \(inlet height\)\s*:\s*([0-9.eE+-]+)\s*um")
    out["d_hat"] = grab(r"d_RBC / \(d_RBC/h1\)\s*:\s*[0-9.]+\s*um\s*/\s*([0-9.eE+-]+)")
    out["Re"] = grab(r"Re\s*=\s*([0-9.eE+-]+)")
    out["tau_w_Pa"] = grab(r"tau_w_nominal \(in\)\s*:\s*([0-9.eE+-]+)\s*Pa")
    out["phi_in"] = grab(r"phi_in \(Hct\)\s*:\s*([0-9.eE+-]+)")
    gm = re.search(r"Geometry Params\s*:\s*(\{.*\})", text)
    if gm:
        try:
            out["geometry_params"] = json.loads(gm.group(1))
        except Exception:
            out["geometry_params"] = {}
    else:
        out["geometry_params"] = {}
    qoi = {}
    for k in QOI_KEYS:
        v = grab(rf"{re.escape(k)}\s*=\s*([0-9.eE+-]+|None)", float, None)
        qoi[k] = v
    out["qoi"] = qoi
    return out


def encode_run(npz_path: Path) -> dict:
    """Load a trimmed .npz and produce the compact encoded frame."""
    z = np.load(npz_path)
    gx, gy, gm = z["grid_x"], z["grid_y"], z["grid_mask"]
    nx, ny = gx.shape[0], gy.shape[0]
    fx, fy = _factor(nx), _factor(ny)
    nx2, ny2 = nx // fx, ny // fy
    # downsampled coordinate arrays (block-mean of the uniform grid)
    xd = _block_mean(gx.astype(np.float32)[None, :], fx, 1)[0]
    yd = _block_mean(gy.astype(np.float32)[:, None], 1, fy)[:, 0]
    mask = _block_mean(gm.astype(np.float32), fx, fy) >= 0.5
    mask_bits = np.packbits(mask.astype(np.uint8))  # row-major
    frame = {
        "nx": int(nx2), "ny": int(ny2),
        "x": _b64(xd.astype(np.float32)), "y": _b64(yd.astype(np.float32)),
        "mask": _b64(mask_bits),
        "fields": {},
    }
    for key in FIELD_KEYS:
        if key not in z.files:
            continue
        a = _block_mean(z[key].astype(np.float32), fx, fy)
        fmin = float(a[mask].min()) if mask.any() else 0.0
        fmax = float(a[mask].max()) if mask.any() else 1.0
        if fmax <= fmin:
            fmax = fmin + 1.0
        q = np.zeros_like(a, dtype=np.uint8)
        q[mask] = np.clip(np.round((a[mask] - fmin) / (fmax - fmin) * 255.0), 0, 255).astype(np.uint8)
        name = key.replace("grid_", "")
        frame["fields"][name] = {"min": fmin, "max": fmax, "data": _b64(q)}
    return frame


def build_family(src: Path, fam: str, out_dir: Path) -> dict:
    """Build one family's G<k>.bin.gz and return its index.json summary."""
    fam_dir = src / fam
    runs = []
    for variant in ("V1", "V2", "V4"):
        vdir = fam_dir / variant
        if not vdir.is_dir():
            continue
        for npz in sorted(vdir.glob("*.npz")):
            txt = npz.with_suffix(".txt")
            meta = parse_run_txt(txt) if txt.exists() else {"run_id": npz.stem, "status": "ok"}
            meta.setdefault("variant", variant)
            meta["family"] = fam
            frame = encode_run(npz)
            frame["id"] = meta["run_id"]
            frame["variant"] = meta.get("variant", variant)
            frame["status"] = meta.get("status", "ok")
            frame["Re"] = meta.get("Re")
            frame["phi_in"] = meta.get("phi_in")
            frame["h_um"] = meta.get("h_um")
            frame["tau_w_Pa"] = meta.get("tau_w_Pa")
            frame["d_hat"] = meta.get("d_hat")
            frame["geometry_params"] = meta.get("geometry_params", {})
            frame["qoi"] = meta.get("qoi", {})
            runs.append(frame)
    payload = json.dumps({"family": fam, "runs": runs}, separators=(",", ":")).encode()
    gz = gzip.compress(payload, compresslevel=9)
    (out_dir / f"{fam}.bin.gz").write_bytes(gz)
    return {
        "family": fam,
        "n_runs": len(runs),
        "variants": sorted({r["variant"] for r in runs}),
        "bytes_gz": len(gz),
        "runs": [
            {k: r[k] for k in ("id", "variant", "status", "Re", "phi_in", "h_um", "tau_w_Pa",
                               "d_hat", "geometry_params", "qoi")}
            for r in runs
        ],
    }


def build(src: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    families = sorted(d.name for d in src.iterdir() if d.is_dir() and re.fullmatch(r"G\d+", d.name))
    index = {
        "version": "0.2.0",
        "generator": "hemoflow build-web-bundle",
        "max_grid_dim": MAX_DIM,
        "flow_params": FLOW_PARAMS,
        "variants": VARIANTS,
        "families": {},
        "runs": {},
    }
    total = 0
    n_runs = 0
    for fam in families:
        summary = build_family(src, fam, out)
        meta = FAMILIES.get(fam, {})
        index["families"][fam] = {
            "name": meta.get("name", fam),
            "topology": meta.get("topology", ""),
            "description": meta.get("description", ""),
            "params": meta.get("params", []),
            "n_runs": summary["n_runs"],
            "variants": summary["variants"],
            "bundle": f"{fam}.bin.gz",
            "bytes_gz": summary["bytes_gz"],
        }
        index["runs"][fam] = summary["runs"]
        total += summary["bytes_gz"]
        n_runs += summary["n_runs"]
        print(f"{fam}: {summary['n_runs']} runs, {summary['bytes_gz']/1e6:.2f} MB gz", flush=True)
    index["n_runs_total"] = n_runs
    index["total_bundle_bytes"] = total
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"TOTAL: {n_runs} runs, bundle {total/1e6:.2f} MB, index.json "
          f"{(out/'index.json').stat().st_size/1e6:.2f} MB", flush=True)
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build-web-bundle", description=__doc__)
    ap.add_argument("--src", required=True, help="corpus_flowfields dir")
    ap.add_argument("--out", required=True, help="output site/data dir")
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        print(f"src not found: {src}", file=sys.stderr)
        return 2
    build(src, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
