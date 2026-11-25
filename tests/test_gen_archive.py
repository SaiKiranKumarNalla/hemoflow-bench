"""T7 Part 1 tests: complete-case archiving, code_version provenance, rerun-archive, dedup.

Run with ``.venv/bin/python -m pytest tests/test_gen_archive.py``.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from campaign.run_batch import (
    archive_case,
    code_version,
    dedupe_manifest_rows,
    load_manifest,
    run_batch,
    scalar_reproducibility_diffs,
    viscosity_ratio_star,
)
from cfd.gen.case import write_case

_BUCKET = "gs://<corpus-bucket>"


def _make_case(case_dir: Path, family: str = "G2", Re: float = 8.0, **extra: Any) -> None:
    """Write a real generated case plus fake mesh/log/time artefacts."""
    params = {
        "run_id": "ARCH_G2_000",
        "family": family,
        "rheology": "Newtonian",
        "Re": Re,
        "geometry": {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
        **extra,
    }
    write_case(case_dir, params, level=0)
    (case_dir / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant" / "polyMesh" / "points").write_text("points\n", encoding="utf-8")
    for t in ("1", "10"):
        (case_dir / t).mkdir(exist_ok=True)
        (case_dir / t / "U").write_text(f"U at {t}\n", encoding="utf-8")
    (case_dir / "log.simpleFoam").write_text("log\n", encoding="utf-8")
    (case_dir / "postProcessing" / "qoiProbes" / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "postProcessing" / "qoiProbes" / "0" / "U.dat").write_text("0 1\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. code_version
# ---------------------------------------------------------------------------

def test_code_version_is_stable_and_content_addressed(tmp_path: Path) -> None:
    """The provenance hash is stable for unchanged files and changes with file contents."""
    v1 = code_version()
    assert v1 == code_version()
    assert len(v1) == 64 and all(c in "0123456789abcdef" for c in v1)

    # A private fake repo tree: hash must follow the sorted (name, contents) pairs.
    root = tmp_path / "repo"
    (root / "cfd" / "gen").mkdir(parents=True)
    (root / "campaign").mkdir()
    (root / "cfd" / "gen" / "a.py").write_text("A = 1\n", encoding="utf-8")
    (root / "campaign" / "b.py").write_text("B = 2\n", encoding="utf-8")
    first = code_version(root)
    assert len(first) == 64

    # Touching mtime is irrelevant; contents are what count.
    (root / "cfd" / "gen" / "a.py").write_text("A = 1\n", encoding="utf-8")
    assert code_version(root) == first

    (root / "cfd" / "gen" / "a.py").write_text("A = 2\n", encoding="utf-8")
    assert code_version(root) != first

    with pytest.raises(FileNotFoundError):
        code_version(tmp_path / "empty_repo")


# ---------------------------------------------------------------------------
# 2. Archive naming, contents and checksum
# ---------------------------------------------------------------------------

def test_archive_case_contains_mesh_dicts_final_time_and_logs(tmp_path: Path) -> None:
    """<run_id>.tar.gz holds polyMesh, dicts, 0/, the final time only, logs and postProcessing."""
    case_dir = tmp_path / "cases" / "G2" / "ARCH_G2_000"
    _make_case(case_dir)
    out = archive_case(case_dir, tmp_path / "archives" / "G2" / "ARCH_G2_000.tar.gz")

    assert out.name == "ARCH_G2_000.tar.gz"
    assert out.is_file() and out.stat().st_size > 0
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    tops = {n.split("/")[0] for n in names}
    assert {"0", "constant", "system", "log.simpleFoam", "postProcessing", "10"} <= tops
    assert "constant/polyMesh/points" in names
    assert "system/blockMeshDict" in names and "system/fvSolution" in names
    assert "0/U" in names
    # Final time only (v2.1): intermediate time 1 is dropped.
    assert "10/U" in names
    assert not any(n == "1" or n.startswith("1/") for n in names)


def test_archive_case_missing_dir_is_empty_archive(tmp_path: Path) -> None:
    """A case directory that never got created yields an empty (but valid) archive."""
    out = archive_case(tmp_path / "nope", tmp_path / "a.tar.gz")
    assert out.is_file()
    with tarfile.open(out, "r:gz") as tar:
        assert tar.getnames() == []


# ---------------------------------------------------------------------------
# 3. Manifest dedup, archive re-run selection and reproducibility report
# ---------------------------------------------------------------------------

def _fake_bucket_fns(tmp_path: Path):
    root = tmp_path / "fake_gcs"
    root.mkdir(exist_ok=True)

    def uri_to_path(uri: str) -> Path:
        return root / uri.replace(f"{_BUCKET}/", "")

    def upload(local_path: Path, uri: str) -> None:
        dst = uri_to_path(uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    def download(uri: str, local_path: Path) -> bool:
        src = uri_to_path(uri)
        if not src.is_file():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return True

    return root, upload, download


def test_dedupe_manifest_rows_keeps_latest_per_run_id() -> None:
    """Append-only manifests collapse to the last row per run_id (T7 v2.3)."""
    rows = [
        {"run_id": "A", "status": "failed"},
        {"run_id": "B", "status": "ok"},
        {"run_id": "A", "status": "ok"},
        {"run_id": "B", "status": "failed"},
        {"run_id": "C", "status": "ok"},
    ]
    deduped = dedupe_manifest_rows(rows)
    by_id = {r["run_id"]: r["status"] for r in deduped}
    assert by_id == {"A": "ok", "B": "failed", "C": "ok"}
    assert {r["run_id"] for r in deduped} == {"A", "B", "C"}
    assert len(deduped) == 3


def test_viscosity_ratio_star_newtonian_and_carreau() -> None:
    """V1's mu_eff/mu_carrier ratio enters tau*, and stays 1 for the plain Newtonian corpus."""
    gdot = np.array([0.5, 2.0])
    assert np.allclose(viscosity_ratio_star(gdot, {"rheology": "Newtonian"}), 1.0)
    assert np.allclose(
        viscosity_ratio_star(gdot, {"rheology": "Newtonian", "viscosity_ratio": 3.2}), 3.2
    )
    carreau = {"rheology": "BirdCarreau", "beta": 5.0, "Cu": 1.0, "n": 0.5, "a": 2.0}
    assert np.all(viscosity_ratio_star(np.array([0.0]), carreau) == pytest.approx(5.0))


def test_run_batch_archives_and_rerun_archive_reports_drift(tmp_path: Path) -> None:
    """Every run gets a case archive; --rerun-archive re-runs archived-incomplete ok runs."""
    design_path = tmp_path / "design.jsonl"
    design_path.write_text(
        json.dumps(
            {
                "run_id": "ARCH_G2_000",
                "family": "G2",
                "rheology": "Newtonian",
                "Re": 8.0,
                "level": 0,
                "geometry": {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bucket_root, upload, download = _fake_bucket_fns(tmp_path)

    calls: list[str] = []
    scale = {"dp": 1.0}

    def fake_runner(case_dir: Path, _timeout: float) -> tuple[bool, str | None]:
        calls.append(str(case_dir))
        (case_dir / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
        (case_dir / "constant" / "polyMesh" / "points").write_text("p\n", encoding="utf-8")
        (case_dir / "log.simpleFoam").write_text("log\n", encoding="utf-8")
        (case_dir / "postProcessing").mkdir(exist_ok=True)
        # Fake extracted scalars via a real (tiny) npz written by a patched extractor below.
        return True, None

    work = tmp_path / "work"

    from campaign import run_batch as rb

    def fake_extract_case(**kwargs: Any) -> dict[str, Any]:
        out_npz = Path(kwargs["out_npz_path"])
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, U_star=np.zeros((1, 3)))
        return {
            "scalars": {"dp_star": 12.0 * scale["dp"], "xr_over_h1": 1.0, "f_window": 0.5},
            "params": {"run_id": "ARCH_G2_000", "code_version": code_version()},
            "sha256": rb.compute_sha256(out_npz),
            "npz_path": out_npz,
        }

    orig_extract = rb.extract_case
    orig_conv = rb.evaluate_convergence
    rb.extract_case = fake_extract_case  # type: ignore[assignment]
    rb.evaluate_convergence = lambda *a, **k: {  # type: ignore[assignment]
        "converged": True,
        "converged_by": "residual",
        "iterations": 10,
        "final_residuals": {"Ux": 1e-6},
        "continuity_error": 1e-10,
        "dp_rel_change_500": 0.0,
        "u_out_rel_change_500": 0.0,
        "openfoam_version": "v2306",
    }
    try:
        # Legacy state: an ok manifest row with no archive (pre-T7 corpus).
        manifest = work / "manifest.jsonl"
        work.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "run_id": "ARCH_G2_000",
                    "family": "G2",
                    "level": 0,
                    "status": "ok",
                    "scalars": {"dp_star": 12.0, "xr_over_h1": 1.0, "f_window": 0.5},
                    "case_archive": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # Plain resume would skip it; rerun_archive selects it and measures drift.
        s_skip = run_batch(
            design_path,
            work_dir=work,
            workers=1,
            upload=False,
            runner_fn=fake_runner,
            uploader_fn=upload,
            downloader_fn=download,
        )
        assert s_skip["attempted"] == 0 and s_skip["skipped"] == 1

        scale["dp"] = 1.0 + 1e-4  # 1e-4 relative drift, above the 1e-6 tolerance
        s = run_batch(
            design_path,
            work_dir=work,
            workers=1,
            upload=True,
            rerun_archive=True,
            runner_fn=fake_runner,
            uploader_fn=upload,
            downloader_fn=download,
        )
        assert s["rerun_archive_selected"] == ["ARCH_G2_000"]
        assert s["attempted"] == 1 and s["succeeded"] == 1
        assert s["archived"] == 1 and s["archive_bytes"] > 0
        assert "ARCH_G2_000" in s["reproducibility_diffs"]
        assert "dp_star" in s["reproducibility_diffs"]["ARCH_G2_000"]
    finally:
        rb.extract_case = orig_extract  # type: ignore[assignment]
        rb.evaluate_convergence = orig_conv  # type: ignore[assignment]

    # Archive uploaded under cases/<tier>/<family>/<run_id>.tar.gz with a manifest checksum.
    archive_path = bucket_root / "corpus/v1/cases/m1/G2/ARCH_G2_000.tar.gz"
    assert archive_path.is_file()
    rows = load_manifest(manifest)
    latest = [r for r in rows if r["run_id"] == "ARCH_G2_000"][-1]
    assert latest["case_archive"] == f"{_BUCKET}/corpus/v1/cases/m1/G2/ARCH_G2_000.tar.gz"
    assert len(latest["case_archive_sha256"]) == 64
    assert latest["code_version"] == code_version()
    # Successful case dir deleted only after the archive was uploaded.
    assert not (work / "cases" / "G2" / "ARCH_G2_000").exists()
    with tarfile.open(archive_path, "r:gz") as tar:
        assert "constant/polyMesh/points" in tar.getnames()


def test_scalar_reproducibility_diffs_threshold() -> None:
    """Only scalars moving by more than the relative tolerance are reported."""
    prev = {"dp_star": 10.0, "xr_over_h1": 2.0, "f_window": None}
    cur = {"dp_star": 10.0 * (1 + 5e-7), "xr_over_h1": 2.0 * (1 + 1e-3), "f_window": 0.3}
    diffs = scalar_reproducibility_diffs(prev, cur)
    assert set(diffs) == {"xr_over_h1"}
    assert diffs["xr_over_h1"]["rel_change"] == pytest.approx(1e-3, rel=1e-2)


def test_run_batch_skips_fenicsx_records(tmp_path: Path) -> None:
    """V3/V4 records (solver: fenicsx) are selected but never executed by run_batch (v2.2)."""
    design_path = tmp_path / "paired.jsonl"
    base = {
        "family": "G2",
        "rheology": "Newtonian",
        "Re": 8.0,
        "level": 0,
        "geometry": {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
    }
    design_path.write_text(
        "\n".join(
            json.dumps(
                {
                    **base,
                    "run_id": f"P_G2_000_{v}",
                    "variant": v,
                    "solver": "fenicsx" if v.startswith("EE") else "simpleFoam",
                }
            )
            for v in ("N", "GN", "EE-DL", "EE-full")
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_runner(case_dir: Path, _timeout: float) -> tuple[bool, str | None]:
        (case_dir / "log.simpleFoam").write_text("log\n", encoding="utf-8")
        return False, "not executed in this test"

    summary = run_batch(
        design_path,
        work_dir=tmp_path / "work",
        workers=1,
        upload=False,
        runner_fn=fake_runner,
    )
    assert summary["skipped_external_solver"] == 2
    assert summary["attempted"] == 2
