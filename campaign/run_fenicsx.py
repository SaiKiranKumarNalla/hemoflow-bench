"""Resumable parallel FEniCSx Euler–Euler batch runner and GCS uploader.

Executes a JSONL design file across ``N`` parallel single-core processes (each isolated
in its own Python/PETSc subprocess or Docker container with single-threaded BLAS/OpenMP),
resumes cleanly by skipping ``run_id``s already recorded in ``manifest_fenicsx.jsonl``,
and uploads ``<run_id>.npz`` + ``manifest_fenicsx.jsonl`` to:
    ``gs://<corpus-bucket>/corpus/v1/runs/fenicsx/<tier>/<family>/``
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BUCKET = "gs://<corpus-bucket>"
DEFAULT_PREFIX = "corpus/v1"
DEFAULT_TIER = "v3"
DEFAULT_MANIFEST_NAME = "manifest_fenicsx.jsonl"
DEFAULT_IMAGE = "ghcr.io/fenics/dolfinx/dolfinx:v0.9.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def fenicsx_code_version(root: Path | str | None = None) -> str:
    """Return SHA-256 hash over ``cfd/fenicsx/*.py`` (independent of dolfinx import)."""
    base = Path(root).resolve() if root is not None else _REPO_ROOT.resolve()
    fenicsx_dir = base / "cfd" / "fenicsx"
    files = sorted(p for p in fenicsx_dir.glob("*.py") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .py files found in {fenicsx_dir}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_design_jsonl(design_path: Path, ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Read design records from a JSONL file, optionally filtering by ``run_id``."""
    records: list[dict[str, Any]] = []
    with Path(design_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            if ids is None or str(rec["run_id"]) in ids:
                records.append(rec)
    return records


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load JSONL manifest rows from disk (empty list if missing)."""
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
    """Append one JSON object line to ``manifest_fenicsx.jsonl``."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False) + "\n")


def dedupe_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest row per ``run_id`` (append-only manifest)."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row.get("run_id"))] = row
    return list(latest.values())


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


def _has_local_dolfinx() -> bool:
    return importlib.util.find_spec("dolfinx") is not None


def _single_thread_env() -> dict[str, str]:
    env = dict(os.environ)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[var] = "1"
    return env


def _run_one_subprocess(
    record: dict[str, Any],
    *,
    level: int,
    work_dir: Path,
    npz_path: Path,
    mode: str,
    image: str,
    timeout: float,
    tau_lo: float,
    tau_hi: float,
) -> dict[str, Any]:
    """Run one single-core FEniCSx case via isolated subprocess (local python or docker)."""
    run_id = str(record["run_id"])
    family = str(record["family"]).upper()
    job_dir = work_dir / "jobs" / run_id
    job_dir.mkdir(parents=True, exist_ok=True)
    rec_json = job_dir / "record.json"
    res_json = job_dir / "result.json"
    rec_json.write_text(json.dumps(record), encoding="utf-8")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    if mode == "local":
        cmd = [
            sys.executable,
            "-m",
            "cfd.fenicsx.run_case",
            "--record-json",
            str(rec_json.resolve()),
            "--level",
            str(level),
            "--out-npz",
            str(npz_path.resolve()),
            "--result-json",
            str(res_json.resolve()),
            "--tau-lo",
            str(tau_lo),
            "--tau-hi",
            str(tau_hi),
        ]
        env = _single_thread_env()
        cwd: Path | None = _REPO_ROOT
    elif mode == "docker":
        repo_abs = _REPO_ROOT.resolve()
        work_abs = work_dir.resolve()
        rec_rel = rec_json.resolve().relative_to(work_abs).as_posix()
        res_rel = res_json.resolve().relative_to(work_abs).as_posix()
        npz_rel = npz_path.resolve().relative_to(work_abs).as_posix()
        cmd = [
            "docker",
            "run",
            "--rm",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-e",
            "MKL_NUM_THREADS=1",
            "-e",
            "NUMEXPR_NUM_THREADS=1",
            "-v",
            f"{repo_abs}:/work",
            "-v",
            f"{work_abs}:/run_work",
            "-w",
            "/work",
            image,
            "python3",
            "-m",
            "cfd.fenicsx.run_case",
            "--record-json",
            f"/run_work/{rec_rel}",
            "--level",
            str(level),
            "--out-npz",
            f"/run_work/{npz_rel}",
            "--result-json",
            f"/run_work/{res_rel}",
            "--tau-lo",
            str(tau_lo),
            "--tau-hi",
            str(tau_hi),
        ]
        env = dict(os.environ)
        cwd = None
    else:
        raise ValueError(f"Unsupported runner mode '{mode}'; expected 'local' or 'docker'")

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        wall_s = time.monotonic() - t0
        return {
            "run_id": run_id,
            "case_id": record.get("case_id"),
            "family": family,
            "level": level,
            "status": "failed",
            "reason": f"timeout after {timeout:.0f}s",
            "newton_iterations": 0,
            "n_continuation_steps": 0,
            "n_halvings": 0,
            "wall_seconds": round(wall_s, 4),
            "peak_ram_mb": None,
            "params": {**record, "level": level, "solver": "fenicsx", "wall_seconds": round(wall_s, 4)},
            "scalars": None,
            "sha256": None,
            "npz_path": None,
            "code_version": fenicsx_code_version(),
        }

    wall_s = time.monotonic() - t0
    if res_json.is_file():
        res = json.loads(res_json.read_text(encoding="utf-8"))
        # Normalize container npz_path back to host path.
        if res.get("status") == "ok" and npz_path.is_file():
            res["npz_path"] = str(npz_path)
        return res

    err_tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
    reason = f"worker exit {proc.returncode}: " + " | ".join(err_tail)
    return {
        "run_id": run_id,
        "case_id": record.get("case_id"),
        "family": family,
        "level": level,
        "status": "failed",
        "reason": reason,
        "newton_iterations": 0,
        "n_continuation_steps": 0,
        "n_halvings": 0,
        "wall_seconds": round(wall_s, 4),
        "peak_ram_mb": None,
        "params": {**record, "level": level, "solver": "fenicsx", "wall_seconds": round(wall_s, 4)},
        "scalars": None,
        "sha256": None,
        "npz_path": None,
        "code_version": fenicsx_code_version(),
    }


def _is_fenicsx_record(rec: Mapping[str, Any]) -> bool:
    solver = rec.get("solver")
    return solver is None or str(solver).lower() == "fenicsx"


def run_fenicsx_batch(
    design_path: Path,
    *,
    work_dir: Path,
    level_override: int | None = None,
    ids: set[str] | None = None,
    variants: set[str] | None = None,
    workers: int | None = None,
    runner: str = "auto",
    image: str = DEFAULT_IMAGE,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    tier: str = DEFAULT_TIER,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    upload: bool = True,
    skip_failed: bool = False,
    timeout: float = 1800.0,
    tau_lo: float = 30.0 / 17.25,
    tau_hi: float = 60.0 / 17.25,
    runner_fn: Callable[[dict[str, Any], int, Path], dict[str, Any]] | None = None,
    uploader_fn: Callable[[Path, str], None] | None = None,
    downloader_fn: Callable[[str, Path], bool] | None = None,
) -> dict[str, Any]:
    """Run a resumable parallel FEniCSx batch with GCS upload.

    Uploads:
    - ``<run_id>.npz`` -> ``<bucket>/<prefix>/runs/fenicsx/<tier>/<family>/<run_id>.npz``
    - ``manifest_fenicsx.jsonl`` -> ``<bucket>/<prefix>/runs/fenicsx/<tier>/<manifest_name>``
      and ``<bucket>/<prefix>/<manifest_name>``
    """
    work_dir = Path(work_dir)
    npz_root = work_dir / "npz"
    manifest_path = work_dir / manifest_name
    work_dir.mkdir(parents=True, exist_ok=True)

    mode = ("local" if _has_local_dolfinx() else "docker") if runner == "auto" else runner
    bucket_clean = bucket.rstrip("/")
    prefix_clean = prefix.strip("/")
    tier_manifest_uri = f"{bucket_clean}/{prefix_clean}/runs/fenicsx/{tier}/{manifest_name}"
    top_manifest_uri = f"{bucket_clean}/{prefix_clean}/{manifest_name}"

    dl = downloader_fn if downloader_fn is not None else gcs_download_if_exists
    ul = uploader_fn if uploader_fn is not None else gcs_upload_file

    # Sync existing manifest from GCS for resumability.
    if upload:
        remote_tmp = work_dir / f".remote_{manifest_name}"
        found = dl(tier_manifest_uri, remote_tmp) or dl(top_manifest_uri, remote_tmp)
        if found:
            if not manifest_path.is_file():
                shutil.move(str(remote_tmp), str(manifest_path))
            else:
                local_rows = load_manifest(manifest_path)
                local_keys = {(r.get("run_id"), r.get("status"), r.get("sha256")) for r in local_rows}
                for r in load_manifest(remote_tmp):
                    key = (r.get("run_id"), r.get("status"), r.get("sha256"))
                    if key not in local_keys:
                        append_manifest_row(manifest_path, r)
                        local_keys.add(key)
                remote_tmp.unlink(missing_ok=True)

    existing_rows = load_manifest(manifest_path)
    latest_by_id = {str(r["run_id"]): r for r in dedupe_manifest_rows(existing_rows)}
    if skip_failed:
        skip_ids = set(latest_by_id.keys())
    else:
        skip_ids = {rid for rid, r in latest_by_id.items() if r.get("status") == "ok"}

    all_records = load_design_jsonl(design_path, ids=ids)
    todo: list[tuple[dict[str, Any], int]] = []
    skipped = 0
    skipped_non_fenicsx = 0
    for rec in all_records:
        if not _is_fenicsx_record(rec):
            skipped_non_fenicsx += 1
            continue
        if variants is not None and str(rec.get("variant", "EE-DL")) not in variants:
            skipped_non_fenicsx += 1
            continue
        if str(rec["run_id"]) in skip_ids:
            skipped += 1
            continue
        lvl = level_override if level_override is not None else int(rec.get("level", 2))
        todo.append((rec, lvl))

    n_workers = max(1, workers if workers is not None else (os.cpu_count() or 1))
    code_ver = fenicsx_code_version()
    summary: dict[str, Any] = {
        "total_selected": len(all_records),
        "skipped": skipped,
        "skipped_non_fenicsx": skipped_non_fenicsx,
        "attempted": len(todo),
        "succeeded": 0,
        "failed": 0,
        "runner": mode,
        "workers": n_workers,
        "manifest_path": str(manifest_path),
        "code_version": code_ver,
        "results": [],
    }

    if not todo:
        return summary

    def _run_one(rec: dict[str, Any], lvl: int) -> dict[str, Any]:
        fam = str(rec["family"]).upper()
        rid = str(rec["run_id"])
        out_npz = npz_root / fam / f"{rid}.npz"
        if runner_fn is not None:
            return runner_fn(rec, lvl, out_npz)
        return _run_one_subprocess(
            rec,
            level=lvl,
            work_dir=work_dir,
            npz_path=out_npz,
            mode=mode,
            image=image,
            timeout=timeout,
            tau_lo=tau_lo,
            tau_hi=tau_hi,
        )

    def _finalize(res: dict[str, Any]) -> None:
        fam = str(res["family"]).upper()
        rid = str(res["run_id"])
        npz_local = Path(res["npz_path"]) if res.get("npz_path") else None
        npz_uri: str | None = None

        if res["status"] == "ok" and npz_local is not None and npz_local.is_file():
            if upload:
                npz_uri = f"{bucket_clean}/{prefix_clean}/runs/fenicsx/{tier}/{fam}/{rid}.npz"
                try:
                    ul(npz_local, npz_uri)
                except Exception as exc:  # noqa: BLE001
                    res["status"] = "failed"
                    res["reason"] = f"upload failed: {exc}"
                    npz_uri = None
            else:
                npz_uri = str(npz_local)

        row = {
            "run_id": rid,
            "case_id": res.get("case_id"),
            "family": fam,
            "level": res["level"],
            "solver": "fenicsx",
            "variant": res.get("params", {}).get("variant", "EE-DL"),
            "status": res["status"],
            "reason": res.get("reason"),
            "newton_iterations": res.get("newton_iterations"),
            "n_continuation_steps": res.get("n_continuation_steps"),
            "n_halvings": res.get("n_halvings"),
            "wall_seconds": res.get("wall_seconds"),
            "peak_ram_mb": res.get("peak_ram_mb"),
            "params": res.get("params"),
            "scalars": res.get("scalars"),
            "sha256": res.get("sha256") if res["status"] == "ok" else None,
            "npz_uri": npz_uri,
            "code_version": res.get("code_version", code_ver),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        append_manifest_row(manifest_path, row)
        if upload:
            ul(manifest_path, tier_manifest_uri)
            ul(manifest_path, top_manifest_uri)

        if res["status"] == "ok":
            summary["succeeded"] += 1
        else:
            summary["failed"] += 1
        summary["results"].append(
            {
                "run_id": rid,
                "family": fam,
                "status": res["status"],
                "reason": res.get("reason"),
                "newton_iterations": res.get("newton_iterations"),
                "n_continuation_steps": res.get("n_continuation_steps"),
                "wall_seconds": res.get("wall_seconds"),
                "peak_ram_mb": res.get("peak_ram_mb"),
                "scalars": res.get("scalars"),
            }
        )

    # ThreadPoolExecutor dispatches N isolated single-core subprocesses / containers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_one, rec, lvl): rec["run_id"] for rec, lvl in todo}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            _finalize(res)
            tag = res["status"].upper()
            extra = ""
            if res["status"] == "ok" and res.get("scalars"):
                sc = res["scalars"]
                extra = (
                    f" it={res.get('newton_iterations')} steps={res.get('n_continuation_steps')} "
                    f"t={res.get('wall_seconds')}s ram={res.get('peak_ram_mb')}MB "
                    f"dp*={sc.get('dp_star'):.3f} phi=[{sc.get('min_phi'):.4f},{sc.get('max_phi'):.4f}]"
                )
            elif res.get("reason"):
                extra = f" — {res['reason']}"
            print(f"[{tag}] {res['run_id']} (fam={res['family']}, L={res['level']}){extra}", flush=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable parallel FEniCSx Euler–Euler batch runner and GCS uploader."
    )
    parser.add_argument("--design", type=Path, required=True, help="Path to design .jsonl file")
    parser.add_argument("--work-dir", type=Path, default=Path("work_fenicsx"))
    parser.add_argument("--level", type=int, default=None, choices=(0, 1, 2))
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated run_ids to run")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated variants filter (e.g. EE-DL)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel single-core worker count")
    parser.add_argument("--runner", type=str, default="auto", choices=("auto", "local", "docker"))
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--tier", type=str, default=DEFAULT_TIER)
    parser.add_argument("--manifest-name", type=str, default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--no-upload", action="store_true", help="Skip GCS sync and upload")
    parser.add_argument("--skip-failed", action="store_true", help="Skip failed ids in manifest too")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--tau-lo", type=float, default=30.0 / 17.25)
    parser.add_argument("--tau-hi", type=float, default=60.0 / 17.25)
    args = parser.parse_args()

    ids = {tok.strip() for tok in args.ids.split(",") if tok.strip()} if args.ids else None
    variants = {tok.strip() for tok in args.variants.split(",") if tok.strip()} if args.variants else None

    summary = run_fenicsx_batch(
        args.design,
        work_dir=args.work_dir,
        level_override=args.level,
        ids=ids,
        variants=variants,
        workers=args.workers,
        runner=args.runner,
        image=args.image,
        bucket=args.bucket,
        prefix=args.prefix,
        tier=args.tier,
        manifest_name=args.manifest_name,
        upload=not args.no_upload,
        skip_failed=args.skip_failed,
        timeout=args.timeout,
        tau_lo=args.tau_lo,
        tau_hi=args.tau_hi,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
