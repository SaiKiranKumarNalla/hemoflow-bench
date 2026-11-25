"""Build the M1 retry design for the runs that did not converge (T7 v2.3).

Reads the corpus manifest (``manifest.jsonl`` downloaded from GCS) and the original
``campaign/m1_design.jsonl``; emits ``campaign/m1_retry_design.jsonl`` with one record per
``run_id`` whose latest manifest row is ``status: failed`` with ``reason: not_converged``.

Only the relaxation factors change (``p 0.2``, ``U 0.5`` instead of ``p 0.3``, ``U 0.7``); every
physical parameter is copied from the original design record so the retry is a pure solver-setting
experiment. The record also carries ``retry_of`` and the relaxation values, which flow into the
manifest ``params`` and the ``.npz`` ``params`` — so a run kept after the retry is honestly
labelled with the settings that produced it (``converged_by`` stays what the run actually met).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campaign.run_batch import dedupe_manifest_rows, load_manifest

DEFAULT_RETRY_RELAX_P = 0.2
DEFAULT_RETRY_RELAX_U = 0.5
RETRYABLE_REASONS = ("not_converged",)


def select_retry_ids(manifest_rows: list[dict[str, Any]]) -> list[str]:
    """Return run_ids whose latest manifest row failed for a retryable reason."""
    latest = dedupe_manifest_rows(manifest_rows)
    return sorted(
        str(row["run_id"])
        for row in latest
        if row.get("status") != "ok" and str(row.get("reason")) in RETRYABLE_REASONS
    )


def build_retry_design(
    manifest_path: Path,
    design_path: Path,
    *,
    relax_p: float = DEFAULT_RETRY_RELAX_P,
    relax_u: float = DEFAULT_RETRY_RELAX_U,
) -> list[dict[str, Any]]:
    """Return retry design records for every not-converged run in ``manifest_path``."""
    retry_ids = set(select_retry_ids(load_manifest(manifest_path)))
    records: list[dict[str, Any]] = []
    with design_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            if str(rec["run_id"]) not in retry_ids:
                continue
            records.append(
                {
                    **rec,
                    "relax_p": relax_p,
                    "relax_u": relax_u,
                    "retry_of": "not_converged",
                    "retry_note": (
                        f"stronger under-relaxation (p {relax_p}, U {relax_u}) after not_converged"
                    ),
                }
            )
    if not records:
        raise ValueError(f"no retryable run_ids found ({len(retry_ids)} ids, none in {design_path})")
    return sorted(records, key=lambda r: str(r["run_id"]))


def write_design_jsonl(output_path: Path | str, records: list[dict[str, Any]]) -> Path:
    """Write design records to a JSONL file (one JSON object per line)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return out


def main() -> None:
    """CLI entrypoint: build campaign/m1_retry_design.jsonl from a local manifest."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the M1 not-converged retry design.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Local manifest.jsonl (download with: gcloud storage cp gs://.../manifest.jsonl .)",
    )
    parser.add_argument("--design", type=Path, default=here / "m1_design.jsonl")
    parser.add_argument("--out", type=Path, default=here / "m1_retry_design.jsonl")
    parser.add_argument("--relax-p", type=float, default=DEFAULT_RETRY_RELAX_P)
    parser.add_argument("--relax-u", type=float, default=DEFAULT_RETRY_RELAX_U)
    args = parser.parse_args()

    records = build_retry_design(
        args.manifest, args.design, relax_p=args.relax_p, relax_u=args.relax_u
    )
    out = write_design_jsonl(args.out, records)
    print(f"Wrote {len(records)} retry records to {out}")


if __name__ == "__main__":
    main()
