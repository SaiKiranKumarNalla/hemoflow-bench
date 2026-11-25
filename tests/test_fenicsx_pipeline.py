"""Tests for FEniCSx Euler–Euler pipeline: mesh_from_spec, run_case, and campaign/run_fenicsx.

Run inside the FEniCSx container:
    DOCKER_HOST=unix://$HOME/.orbstack/run/docker.sock DOCKER_CONFIG=/tmp/empty_docker_cfg \
        cfd/fenicsx/run_docker.sh -m unittest discover -s tests -p 'test_fenicsx_*.py' -v

Also safe under host pytest (dolfinx-dependent tests are cleanly skipped when dolfinx is absent).
"""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from campaign import run_fenicsx

HAS_DOLFINX = importlib.util.find_spec("dolfinx") is not None

if HAS_DOLFINX:
    import ufl
    from dolfinx import fem

    from cfd.fenicsx import mesh_from_spec as mfs
    from cfd.fenicsx import run_case as rc
    from cfd.fenicsx import twofluid as tf


# Default designs matching campaign/paired_design_pilot.jsonl geometries for G2–G7.
DEFAULT_GEOMS: dict[str, dict[str, float | int]] = {
    "G2": {"E": 2.75, "L_in": 5.0, "L_out_ratio": 10.0},
    "G3": {"throat_ratio": 0.55, "L_t": 3.0, "L_in": 5.0, "L_out_ratio": 10.0},
    "G4": {"theta_deg": 55.0, "daughter_ratio": 0.75, "L_in": 5.0, "L_out_ratio": 10.0},
    "G5": {"width_ratio": 0.75, "theta_deg": 60.0, "L_in": 5.0, "L_out_ratio": 10.0},
    "G6": {"Rc_ratio": 3.75, "n_bends": 2, "L_in": 5.0, "L_out_ratio": 10.0},
    "G7": {"E": 2.75, "theta_deg": 55.0, "L_in": 5.0, "L_out_ratio": 10.0},
}

# Exact OpenFOAM M1 .npz schema keys from campaign/run_batch.py extract_case.
OPENFOAM_BASE_KEYS: set[str] = {
    "x_star",
    "y_star",
    "cell_volumes",
    "U_star",
    "p_star",
    "gdot_star",
    "tau_star",
    "outlet_s",
    "outlet_u_n",
    "outlet_gdot",
    "outlet_tau",
    "outlet_weights",
    "outlet_branch_id",
    "params",
    "grid_x",
    "grid_y",
    "grid_mask",
    "grid_ux",
    "grid_uy",
    "grid_p",
    "grid_gdot",
    "grid_tau",
}

FENICSX_MULTIPHASE_KEYS: set[str] = {
    "phi",
    "phi_star",
    "U_rbc_star",
    "U_plasma_star",
    "gdot_plasma_star",
    "gdot_rbc_star",
    "outlet_phi",
    "outlet_u_plasma_n",
    "grid_phi",
    "grid_urbc_x",
    "grid_urbc_y",
    "grid_u_plasma_x",
    "grid_u_plasma_y",
}


@unittest.skipUnless(HAS_DOLFINX, "requires dolfinx container")
class TestMeshFromSpec(unittest.TestCase):
    """Verify dolfinx quad mesh counts, areas, grading, and facet tags against MeshSpec."""

    def test_g2_to_g7_level2_cell_counts_areas_and_tags(self) -> None:
        for fam, geom in DEFAULT_GEOMS.items():
            with self.subTest(family=fam):
                msh, tags, spec = mfs.build_dolfinx_mesh(fam, geom, level=2)
                n_cells = int(msh.topology.index_map(2).size_local)
                expected_cells = mfs.expected_cell_count(spec)
                self.assertEqual(n_cells, expected_cells)

                area = float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh))))
                expected_area = mfs.expected_domain_area(spec)
                rel_tol = 1e-3 if fam == "G6" else 1e-10
                self.assertLess(
                    abs(area - expected_area) / expected_area,
                    rel_tol,
                    f"{fam}: mesh area {area:.6f} vs expected {expected_area:.6f}",
                )

                # Verify INLET=1, OUTLET=2, WALL=3 lengths match spec.
                ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
                l_inlet = float(fem.assemble_scalar(fem.form(1.0 * ds(mfs.INLET))))
                l_outlet = float(fem.assemble_scalar(fem.form(1.0 * ds(mfs.OUTLET))))
                l_wall = float(fem.assemble_scalar(fem.form(1.0 * ds(mfs.WALL))))

                self.assertAlmostEqual(l_inlet, spec.inlet_height, places=10)
                self.assertAlmostEqual(l_outlet, sum(spec.outlet_heights), places=10)
                self.assertGreater(l_wall, 0.0)
                self.assertEqual(len(tags.find(mfs.AXIS)), 0)

    def test_g4_direct_6block_and_kg_axis_and_g1(self) -> None:
        # G4 acute wide daughter triggers the 6-block direct topology.
        msh4, tags4, spec4 = mfs.build_dolfinx_mesh(
            "G4", {"theta_deg": 25.0, "daughter_ratio": 0.9, "L_in": 5.0, "L_out_ratio": 10.0}, level=1
        )
        self.assertEqual(len(spec4.blocks_2d), 6)
        self.assertEqual(int(msh4.topology.index_map(2).size_local), mfs.expected_cell_count(spec4))
        area4 = float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh4))))
        self.assertAlmostEqual(area4, mfs.expected_domain_area(spec4), places=9)

        # KG axisymmetric meridian mesh has AXIS=4 tag of length L_in + L_out.
        msh_kg, tags_kg, spec_kg = mfs.build_dolfinx_mesh("KG", {}, level=1)
        self.assertEqual(int(msh_kg.topology.index_map(2).size_local), mfs.expected_cell_count(spec_kg))
        ds_kg = ufl.Measure("ds", domain=msh_kg, subdomain_data=tags_kg)
        l_axis = float(fem.assemble_scalar(fem.form(1.0 * ds_kg(mfs.AXIS))))
        self.assertAlmostEqual(l_axis, spec_kg.entry_length + spec_kg.outlet_lengths[0], places=9)
        l_in = float(fem.assemble_scalar(fem.form(1.0 * ds_kg(mfs.INLET))))
        l_out = float(fem.assemble_scalar(fem.form(1.0 * ds_kg(mfs.OUTLET))))
        self.assertAlmostEqual(l_in, 0.5, places=9)
        self.assertAlmostEqual(l_out, 0.5 * float(spec_kg.params["E"]), places=9)

        # G1 straight channel.
        msh1, tags1, spec1 = mfs.build_dolfinx_mesh("G1", {"L_ratio": 10.0}, level=0)
        self.assertEqual(int(msh1.topology.index_map(2).size_local), mfs.expected_cell_count(spec1))
        area1 = float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=msh1))))
        self.assertAlmostEqual(area1, 10.0, places=10)

    def test_simple_grading_1d_matches_openfoam(self) -> None:
        t = mfs._simple_grading_1d(10, 4.0)
        self.assertEqual(len(t), 11)
        self.assertAlmostEqual(t[0], 0.0)
        self.assertAlmostEqual(t[-1], 1.0)
        dx = np.diff(t)
        self.assertAlmostEqual(dx[-1] / dx[0], 4.0, places=10)


@unittest.skipUnless(HAS_DOLFINX, "requires dolfinx container")
class TestRunCase(unittest.TestCase):
    """Verify straight-channel gate, expansion run, .npz schema, and V4 TODO guard."""

    def test_straight_channel_gate_through_run_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for phi_in in (0.01, 0.45):
                with self.subTest(phi_in=phi_in):
                    out_npz = Path(tmp) / f"G1_phi_{phi_in:.2f}.npz"
                    res = rc.run_case(
                        run_id=f"GATE_G1_{phi_in:.2f}",
                        family="G1",
                        geometry_params={"L_ratio": 10.0},
                        level=0,
                        re_p=10.0,
                        d_hat=0.08,
                        phi_in=phi_in,
                        out_npz_path=out_npz,
                        variant="EE-DL",
                        viscosity="wu-high-shear",
                    )
                    self.assertEqual(res["status"], "ok", f"G1 failed: {res.get('reason')}")
                    self.assertTrue(out_npz.is_file())

                    mu_hat = float(tf.mixture_viscosity_hat(phi_in))
                    dp_exact = 12.0 * mu_hat * 10.0
                    dp_got = res["scalars"]["dp_star"]
                    rel_err = abs(dp_got - dp_exact) / dp_exact
                    self.assertLess(
                        rel_err,
                        0.015,
                        f"phi={phi_in}: dp*={dp_got:.4f} vs exact {dp_exact:.4f} (err={rel_err:.2e})",
                    )
                    self.assertLess(res["scalars"]["max_slip"], 1e-3)
                    self.assertLess(res["scalars"]["max_phi"] - res["scalars"]["min_phi"], 1e-3)
                    self.assertEqual(res["scalars"]["cfl_thickness_star"], 0.0)

                    # Outlet shear stress profile matches exact Poiseuille tau*(y) = 6 * mu_hat * |1 - 2y|.
                    with np.load(out_npz, allow_pickle=False) as d_g1:
                        y_out = d_g1["outlet_s"]
                        tau_out = d_g1["outlet_tau"]
                    tau_exact = 6.0 * mu_hat * np.abs(1.0 - 2.0 * y_out)
                    max_prof_err = float(np.max(np.abs(tau_out - tau_exact))) / (6.0 * mu_hat)
                    self.assertLess(max_prof_err, 0.01)
                    self.assertGreater(res["scalars"]["max_wall_tau_star"], 0.9 * 6.0 * mu_hat)

    def test_g2_run_case_schema_and_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = Path(tmp) / "G2_test.npz"
            res = rc.run_case(
                run_id="TEST_G2_L0",
                family="G2",
                geometry_params={"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
                level=0,
                re_p=5.0,
                d_hat=0.08,
                phi_in=0.10,
                out_npz_path=out_npz,
                variant="EE-DL",
                viscosity="wu-high-shear",
            )
            self.assertEqual(res["status"], "ok", f"G2 failed: {res.get('reason')}")
            self.assertTrue(out_npz.is_file())
            self.assertGreater(res["newton_iterations"], 0)
            self.assertGreater(res["wall_seconds"], 0.0)
            self.assertGreater(res["peak_ram_mb"], 0.0)

            with np.load(out_npz, allow_pickle=False) as data:
                keys = set(data.files)
                expected_keys = OPENFOAM_BASE_KEYS | FENICSX_MULTIPHASE_KEYS
                self.assertEqual(keys, expected_keys)

                params = json.loads(str(data["params"]))
                self.assertEqual(params["solver"], "fenicsx")
                self.assertEqual(params["variant"], "EE-DL")
                self.assertEqual(params["viscosity"], "wu-high-shear")
                self.assertEqual(len(params["code_version"]), 64)
                self.assertEqual(params["status"], "ok")

                n_cells = params["n_cells"]
                self.assertEqual(data["x_star"].shape, (n_cells,))
                self.assertEqual(data["y_star"].shape, (n_cells,))
                self.assertEqual(data["cell_volumes"].shape, (n_cells,))
                self.assertEqual(data["U_star"].shape, (n_cells, 3))
                self.assertEqual(data["U_rbc_star"].shape, (n_cells, 3))
                self.assertEqual(data["phi"].shape, (n_cells,))
                self.assertEqual(data["p_star"].shape, (n_cells,))

                nx, ny = rc.FAMILY_GRID_SHAPES["G2"]
                mask = data["grid_mask"]
                self.assertEqual(mask.shape, (ny, nx))
                for k in ("grid_ux", "grid_uy", "grid_p", "grid_gdot", "grid_tau", "grid_phi", "grid_urbc_x", "grid_urbc_y"):
                    arr = data[k]
                    self.assertEqual(arr.shape, (ny, nx))
                    self.assertTrue(np.all(np.isfinite(arr[mask])), f"NaN/Inf in {k} inside mask")
                    self.assertTrue(np.all(arr[~mask] == 0.0), f"Non-zero outside mask in {k}")

            sc = res["scalars"]
            self.assertGreater(sc["dp_star"], 0.0)
            self.assertIsNotNone(sc["xr_over_h1"])
            self.assertGreaterEqual(sc["xr_over_h1"], 0.0)
            self.assertTrue(0.0 <= sc["f_window"] <= 1.0)
            self.assertTrue(math.isfinite(sc["outlet_phi_mean"]))

    def test_v4_ee_full_raises_todo_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = Path(tmp) / "v4.npz"
            res = rc.run_case(
                run_id="TEST_V4_TODO",
                family="G1",
                geometry_params={"L_ratio": 10.0},
                level=0,
                re_p=5.0,
                d_hat=0.08,
                phi_in=0.20,
                out_npz_path=out_npz,
                variant="EE-full",
            )
            self.assertEqual(res["status"], "failed")
            self.assertIn("NotImplementedError", res["reason"])
            self.assertFalse(out_npz.exists())


class TestBatchRunner(unittest.TestCase):
    """Verify campaign/run_fenicsx.py parallel dispatch, filtering, resume, and GCS paths."""

    def test_batch_mock_resume_and_gcs_uris(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            design = tmp_path / "design.jsonl"
            records = [
                {
                    "run_id": "R_OF_1",
                    "family": "G2",
                    "level": 0,
                    "solver": "simpleFoam",
                    "variant": "N",
                    "Re": 5.0,
                    "phi": 0.2,
                    "geometry": {"E": 2.0},
                },
                {
                    "run_id": "R_FX_1",
                    "family": "G2",
                    "level": 0,
                    "solver": "fenicsx",
                    "variant": "EE-DL",
                    "Re": 5.0,
                    "phi": 0.05,
                    "d_rbc_over_h": 0.08,
                    "geometry": {"E": 2.0},
                },
                {
                    "run_id": "R_FX_2",
                    "family": "G4",
                    "level": 0,
                    "solver": "fenicsx",
                    "variant": "EE-DL",
                    "Re": 5.0,
                    "phi": 0.45,
                    "d_rbc_over_h": 0.08,
                    "geometry": {"theta_deg": 55.0, "daughter_ratio": 0.75},
                },
            ]
            design.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

            uploaded: list[tuple[Path, str]] = []
            calls: list[str] = []

            def fake_runner(rec: dict, lvl: int, out_npz: Path) -> dict:
                calls.append(rec["run_id"])
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_npz, dummy=np.array([1.0]))
                return {
                    "run_id": rec["run_id"],
                    "case_id": None,
                    "family": rec["family"],
                    "level": lvl,
                    "status": "ok",
                    "reason": None,
                    "newton_iterations": 8,
                    "n_continuation_steps": 3,
                    "n_halvings": 0,
                    "wall_seconds": 1.25,
                    "peak_ram_mb": 200.0,
                    "params": {"variant": rec["variant"]},
                    "scalars": {"dp_star": 100.0, "min_phi": 0.04, "max_phi": 0.06},
                    "sha256": "a" * 64,
                    "npz_path": str(out_npz),
                    "code_version": run_fenicsx.fenicsx_code_version(),
                }

            def fake_upload(local: Path, uri: str) -> None:
                uploaded.append((local, uri))

            def fake_download(uri: str, local: Path) -> bool:
                return False

            work_dir = tmp_path / "work"
            s1 = run_fenicsx.run_fenicsx_batch(
                design,
                work_dir=work_dir,
                workers=2,
                tier="pilot",
                upload=True,
                runner_fn=fake_runner,
                uploader_fn=fake_upload,
                downloader_fn=fake_download,
            )
            self.assertEqual(s1["skipped_non_fenicsx"], 1)
            self.assertEqual(s1["attempted"], 2)
            self.assertEqual(s1["succeeded"], 2)
            self.assertEqual(s1["failed"], 0)
            self.assertEqual(set(calls), {"R_FX_1", "R_FX_2"})

            uris = {u for _, u in uploaded}
            self.assertIn(
                "gs://<corpus-bucket>/corpus/v1/runs/fenicsx/pilot/G2/R_FX_1.npz", uris
            )
            self.assertIn(
                "gs://<corpus-bucket>/corpus/v1/runs/fenicsx/pilot/G4/R_FX_2.npz", uris
            )
            self.assertIn(
                "gs://<corpus-bucket>/corpus/v1/runs/fenicsx/pilot/manifest_fenicsx.jsonl", uris
            )

            # Second invocation resumes and skips all completed ids.
            calls.clear()
            s2 = run_fenicsx.run_fenicsx_batch(
                design,
                work_dir=work_dir,
                workers=2,
                tier="pilot",
                upload=False,
                runner_fn=fake_runner,
            )
            self.assertEqual(s2["skipped"], 2)
            self.assertEqual(s2["attempted"], 0)
            self.assertEqual(calls, [])

    @unittest.skipUnless(HAS_DOLFINX, "requires dolfinx container")
    def test_batch_real_subprocess_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            design = tmp_path / "mini.jsonl"
            records = [
                {
                    "run_id": "MINI_G1",
                    "family": "G1",
                    "level": 0,
                    "solver": "fenicsx",
                    "variant": "EE-DL",
                    "Re": 5.0,
                    "phi": 0.05,
                    "d_rbc_over_h": 0.08,
                    "geometry": {"L_ratio": 10.0},
                },
                {
                    "run_id": "MINI_G2",
                    "family": "G2",
                    "level": 0,
                    "solver": "fenicsx",
                    "variant": "EE-DL",
                    "Re": 5.0,
                    "phi": 0.05,
                    "d_rbc_over_h": 0.08,
                    "geometry": {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
                },
            ]
            design.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            work_dir = tmp_path / "work"

            s1 = run_fenicsx.run_fenicsx_batch(
                design, work_dir=work_dir, workers=2, runner="local", upload=False
            )
            self.assertEqual(s1["succeeded"], 2)
            self.assertEqual(s1["failed"], 0)
            self.assertTrue((work_dir / "npz" / "G1" / "MINI_G1.npz").is_file())
            self.assertTrue((work_dir / "npz" / "G2" / "MINI_G2.npz").is_file())
            self.assertTrue((work_dir / "manifest_fenicsx.jsonl").is_file())

            # Resume skips both.
            s2 = run_fenicsx.run_fenicsx_batch(
                design, work_dir=work_dir, workers=2, runner="local", upload=False
            )
            self.assertEqual(s2["skipped"], 2)
            self.assertEqual(s2["attempted"], 0)


if __name__ == "__main__":
    unittest.main()
