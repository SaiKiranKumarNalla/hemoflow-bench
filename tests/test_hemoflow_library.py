"""Host-runnable tests for the hemoflow library (NO dolfinx/ufl required).

Run with the rbc-shear-surrogate venv:
  .venv/bin/pytest tests/test_hemoflow_library.py -v
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys

import numpy as np
import pytest


def test_import_without_dolfinx():
    """import hemoflow must succeed with dolfinx/ufl/basix/petsc4py/mpi4py absent."""
    for mod in ("dolfinx", "ufl", "basix", "petsc4py", "mpi4py"):
        assert importlib.util.find_spec(mod) is None, f"{mod} unexpectedly installed; test needs a host without it"
    import hemoflow
    assert hemoflow.__version__ == "0.2.0"
    # non-solve subpackages import cleanly
    import hemoflow.registry, hemoflow.closures, hemoflow.geometries
    import hemoflow.solver.config, hemoflow.solver.analytic, hemoflow.solver.result, hemoflow.cli


def test_list_plugins_builtin():
    import hemoflow
    p = hemoflow.list_plugins()
    assert set(p["viscosity"]) >= {"newtonian", "wu_high_shear", "wu_shear_thinning", "krieger", "quemada", "bird_carreau"}
    assert set(p["drag"]) >= {"wu_rusche_issa", "schiller_naumann"}
    assert set(p["lift"]) >= {"none", "saffman", "abkarian_wall", "abkarian_twowall", "saffman_plus_twowall"}
    assert set(p["migration"]) >= {"none", "phillips_local", "phillips_nonlocal"}
    assert set(p["geometry"]) >= {f"G{i}" for i in range(1, 14)} | {"KG"}


def test_custom_plugin_registration_and_swap():
    import hemoflow

    @hemoflow.register_viscosity("my_test_mu")
    def my_mu(phi):
        return 1.0 + 3.0 * phi

    assert "my_test_mu" in hemoflow.list_plugins("viscosity")["viscosity"]
    assert hemoflow.get_plugin("viscosity", "my_test_mu")(0.2) == pytest.approx(1.6)

    # pass a direct callable into the config (plug-out)
    cfg = hemoflow.EulerEulerConfig(re_p=5.0, d_hat=0.06, phi_in=0.3, viscosity=my_mu)
    assert cfg.resolve()["viscosity"] is my_mu
    # and a registered name
    cfg2 = hemoflow.EulerEulerConfig(re_p=5.0, d_hat=0.06, phi_in=0.3, viscosity="krieger")
    assert callable(cfg2.resolve()["viscosity"])
    # unknown name raises
    with pytest.raises(Exception):
        hemoflow.EulerEulerConfig(re_p=5.0, d_hat=0.06, phi_in=0.3, viscosity="does_not_exist")


# ----------------------------- closure exact values -----------------------------

MU_PLASMA_CP = 0.96


def mu_inf_cp(phi):
    return 27.873 * phi**3 - 21.218 * phi**2 + 14.439 * phi


def mu0_cp(phi):
    return 537.002 * phi**3 + 55.006 * phi**2 - 0.129 * phi


def test_viscosity_closures_exact_numpy():
    from hemoflow.closures import viscosity as V
    phi = np.array([0.0, 0.1, 0.3, 0.45, 0.6])
    # wu_high_shear = (1-phi) + 0.5 mu_inf_cp/mu_plasma_cp
    expect = (1.0 - phi) + 0.5 * mu_inf_cp(phi) / MU_PLASMA_CP
    np.testing.assert_allclose(V.wu_high_shear(phi), expect, rtol=1e-12)
    # newtonian
    np.testing.assert_allclose(V.newtonian(phi, mu_hat=2.5), np.full_like(phi, 2.5))
    # krieger
    phi_max = 2.0 / 2.07
    expect_k = (1.0 - np.minimum(phi, 0.999 * phi_max) / phi_max) ** (-2.5 * phi_max)
    np.testing.assert_allclose(V.krieger(phi, phi_max), expect_k, rtol=1e-12)
    # wu_shear_thinning
    gdot = np.array([1.0, 10.0, 100.0, 1000.0, 1e4])
    k_star = 2092.0
    kg = k_star * gdot
    num = mu_inf_cp(phi) + (mu0_cp(phi) - mu_inf_cp(phi)) * (1.0 + np.log(1.0 + kg)) / (1.0 + kg)
    expect_w = np.maximum(num / MU_PLASMA_CP, 1e-4)
    np.testing.assert_allclose(V.wu_shear_thinning(phi, gdot, k_star), expect_w, rtol=1e-10)


def test_quemada_and_bird_carreau_exact():
    from hemoflow.closures import viscosity as V
    phi, gdot = 0.45, 100.0
    root = math.sqrt(gdot / 1.88)
    k_eff = (4.33 + 2.07 * root) / (1.0 + root)
    bracket = 1.0 - 0.5 * k_eff * phi
    np.testing.assert_allclose(V.quemada(phi, gdot), bracket**-2.0, rtol=1e-12)
    np.testing.assert_allclose(V.quemada_mu(phi, gdot), 0.96e-3 * bracket**-2.0, rtol=1e-12)
    # bird_carreau
    mu0, mu_inf, lam, n, a = 0.05, 0.004, 0.110, 0.6, 2.0
    expect = mu_inf + (mu0 - mu_inf) * (1.0 + (lam * gdot) ** a) ** ((n - 1.0) / a)
    np.testing.assert_allclose(V.bird_carreau(gdot, mu0, mu_inf, lam, n, a), expect, rtol=1e-12)


def test_drag_closures_exact_numpy():
    from hemoflow.closures import drag as DR
    phi = np.array([0.0, 0.2, 0.45])
    d_hat = 0.063
    # hindrance
    np.testing.assert_allclose(DR.hindrance(phi), np.exp(2.68 * phi) + phi**0.43, rtol=1e-12)
    # wu_rusche_issa (phi_pos = max(phi, 1e-4))
    phi_reg = np.clip(phi, 0.0, 1.0)
    phi_pos = np.maximum(phi, 1e-4)
    expect = 18.0 * phi_reg * (np.exp(2.68 * phi_pos) + phi_pos**0.43) / d_hat**2
    np.testing.assert_allclose(DR.wu_rusche_issa(phi, d_hat), expect, rtol=1e-12)
    # schiller_naumann
    re_p, slip = 10.0, 0.05
    re_d = re_p * d_hat * slip
    expect_sn = 18.0 * phi_reg * (1.0 + 0.15 * re_d**0.687) / d_hat**2
    np.testing.assert_allclose(DR.schiller_naumann(phi, d_hat, re_p, slip), expect_sn, rtol=1e-12)


def test_lift_closures_exact_numpy():
    from hemoflow.closures import lift as L
    from hemoflow.closures.constants import SAFFMAN_C, WALL_LIFT_C
    phi, re, d_hat = 0.3, 10.0, 0.063
    gdot = 5.0
    D_u = np.array([[0.0, 2.0], [2.0, 0.0]])  # 2x2 symmetric
    slip = np.array([0.1, 0.0])
    expect_saff = SAFFMAN_C * phi * math.sqrt(re) / d_hat * gdot**-0.5 * (D_u @ slip)
    np.testing.assert_allclose(L.saffman(phi, re, d_hat, gdot, D_u, slip), expect_saff, rtol=1e-12)
    # abkarian_twowall: grad_psi, psi
    grad_psi = np.array([0.0, 1.0])  # vertical wall normal
    psi = 0.25
    gnorm = math.sqrt(1.0 + 1e-12)
    s_psi = math.sqrt(1.0 + 2.0 * max(psi, 0.0) + 1e-16)
    d1, d2 = s_psi - gnorm, s_psi + gnorm
    r_hat = 0.5 * d_hat
    def wl(d):
        yg = max(d - r_hat, 0.1 * r_hat)
        return 1.0 / yg
    mag = WALL_LIFT_C * phi * gdot * (wl(d1) - wl(d2))
    expect_tw = -mag * grad_psi / gnorm
    np.testing.assert_allclose(L.abkarian_twowall(phi, gdot, grad_psi, psi, WALL_LIFT_C, d_hat, "abk"),
                               expect_tw, rtol=1e-10)
    # combined (note: saffman_plus_twowall reuses d_hat from position 3; pass law/wall_lift_C by keyword)
    np.testing.assert_allclose(
        L.saffman_plus_twowall(phi, re, d_hat, gdot, D_u, slip, grad_psi, psi,
                               wall_lift_C=WALL_LIFT_C, law="abk"),
        expect_saff + expect_tw, rtol=1e-10)
    assert L.none() == 0.0


def test_migration_closures_exact_numpy():
    from hemoflow.closures import migration as M
    from hemoflow.closures.constants import K_C, K_MU, PHI_JAM
    phi = 0.3
    gdot = 2.0
    grad_gdot = np.array([0.1, 0.0])
    grad_phi = np.array([0.0, 0.5])
    dlnmu = 1.3
    a = 0.03
    phi_c = min(max(phi, 0.0), PHI_JAM)
    gate = 0.5 * (1.0 - math.tanh((phi_c - 0.63) / 0.015))
    expect = -a**2 * (K_C * gate * phi_c**2 * grad_gdot
                      + (K_C * phi_c * gdot + K_MU * phi_c**2 * gdot * dlnmu) * grad_phi)
    np.testing.assert_allclose(M.phillips_flux(phi, gdot, grad_gdot, grad_phi, dlnmu, a), expect, rtol=1e-12)
    np.testing.assert_allclose(M.phillips_local(phi, gdot, grad_gdot, grad_phi, dlnmu, a), expect, rtol=1e-12)
    # nonlocal gdot helper
    j = np.array([1.0, 0.2])
    C = 4.0
    expect_g = math.sqrt(gdot**2 + (C * a) ** 2 * float(j @ j) + 1e-6)
    np.testing.assert_allclose(M.nonlocal_gdot(gdot, j, a, C), expect_g, rtol=1e-12)
    # phillips_nonlocal folds the floor when j given
    out = M.phillips_nonlocal(phi, gdot, grad_gdot, grad_phi, dlnmu, a, j=j, C=C)
    gate2 = 0.5 * (1.0 - math.tanh((phi_c - 0.63) / 0.015))
    expect_nl = -a**2 * (K_C * gate2 * phi_c**2 * grad_gdot
                         + (K_C * phi_c * expect_g + K_MU * phi_c**2 * expect_g * dlnmu) * grad_phi)
    np.testing.assert_allclose(out, expect_nl, rtol=1e-12)
    assert M.none() == 0.0


# ----------------------------- geometries -----------------------------

GEOM_SAMPLES = {
    "G1": {"L_ratio": 20.0},
    "G2": {"E": 3.0, "L_in": 5.0, "L_out_ratio": 10.0},
    "G3": {"throat_ratio": 0.5, "L_t": 2.0, "L_in": 5.0, "L_out_ratio": 10.0, "fillet_r": 0.08},
    "G4": {"theta_deg": 45.0, "daughter_ratio": 0.8, "L_in": 5.0, "L_out_ratio": 10.0, "fillet_r": 0.08},
    "G5": {"width_ratio": 0.75, "theta_deg": 60.0, "L_in": 5.0, "L_out_ratio": 10.0, "fillet_r": 0.08},
    "G6": {"Rc_ratio": 3.5, "n_bends": 2, "L_in": 5.0, "L_straight": 2.0, "L_out_ratio": 10.0},
    "G7": {"E": 2.0, "theta_deg": 55.0, "L_in": 5.0, "L_mid": 4.0, "L_out_ratio": 10.0},
    "G8": {"throat_ratio": 0.7, "L_conv": 2.5, "L_div": 5.0, "L_in": 5.0, "L_out_ratio": 10.0},
    "G9": {"throat_ratio": 0.7, "cavity_ratio": 1.45, "n_stages": 3, "L_in": 5.0, "L_out_ratio": 10.0},
    "G10": {"pocket_ratio": 2.1, "L_pocket": 4.0, "L_ramp": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
    "G11": {"branch_bias": 0.42, "L_loop": 6.0, "theta_deg": 35.0, "L_in": 5.0, "L_out_ratio": 10.0},
    "G12": {"amp_ratio": 0.4, "wavelength": 5.0, "n_waves": 3, "L_in": 5.0, "L_out_ratio": 10.0},
    "G13": {"center_ratio": 0.7, "outer_ratio": 0.65, "theta_deg": 42.0, "L_in": 5.0, "L_out_ratio": 10.0},
}


@pytest.mark.parametrize("fam", sorted(GEOM_SAMPLES))
def test_geometry_families_build_valid(fam):
    from hemoflow.geometries import builders, mesher
    params = GEOM_SAMPLES[fam]
    spec = builders.build_geometry(fam, params, level=1)
    assert spec.family.upper() == fam
    assert len(spec.blocks_2d) >= 1
    nodes, cells, tagged, min_wall_dy = mesher.graded_mesh_arrays(fam, params, refine=0.5)
    areas = mesher.cell_signed_areas(nodes, cells)
    assert np.all(areas > 0), f"{fam}: non-positive quad area"
    tags = {t[2] for t in tagged}
    assert mesher.INLET in tags and mesher.OUTLET in tags and mesher.WALL in tags, f"{fam}: missing bc tags {tags}"
    assert np.isfinite(min_wall_dy) and min_wall_dy > 0


def test_geometry_registry_metadata():
    import hemoflow
    for fam in [f"G{i}" for i in range(1, 14)] + ["KG"]:
        meta = hemoflow.geometry_metadata(fam)
        assert meta.get("topology"), fam
        assert meta.get("params"), fam
        builder = hemoflow.get_plugin("geometry", fam)
        assert callable(builder)


# ----------------------------- analytic + solver helpers -----------------------------

def test_analytic_profiles():
    from hemoflow.solver.analytic import poiseuille, brinkman_inlet, developed_phillips_profile
    y = np.linspace(0, 1, 11)
    np.testing.assert_allclose(poiseuille(y), 6 * y * (1 - y))
    np.testing.assert_allclose(poiseuille(0.5), 1.5)
    # brinkman reduces to parabolic when depth_hat<=0
    np.testing.assert_allclose(brinkman_inlet(y - 0.5, 0.5, 0.0), 1.5 * (1 - ((y - 0.5) / 0.5) ** 2))
    yy, u, phi = developed_phillips_profile(0.3, n=201)
    assert yy.shape == u.shape == phi.shape == (201,)
    assert abs(np.trapezoid(u, yy) - 1.0) < 1e-6
    assert abs(np.trapezoid(phi * u, yy) - 0.3) < 1e-3
    assert np.all(phi <= 0.95 + 1e-9)


def test_cfl_comsol_and_scaled_error():
    from hemoflow.solver.stationary import cfl_comsol, comsol_scaled_error
    assert cfl_comsol(1) == pytest.approx(1.3)
    assert cfl_comsol(9) == pytest.approx(1.3**9)
    assert cfl_comsol(21) == pytest.approx(1.3**9 + 9.0 * 1.3)
    assert cfl_comsol(41) == pytest.approx(1.3**9 + 9.0 * 1.3**9 + 90.0 * 1.3)
    # scaled error: zero update -> zero error
    z = comsol_scaled_error([np.zeros(4)] * 4, [np.ones(4)] * 4, 1.0)
    assert z["E"] == 0.0
    # floors: phi floor 0.05
    e = comsol_scaled_error([np.array([0.05])] * 4, [np.zeros(4)] * 4, 1.0)
    assert e["phi"] == pytest.approx(1.0)  # 0.05 / max(0, 0.05)
    assert e["u"] == pytest.approx(0.05)    # 0.05 / max(0, 1.0)


# ----------------------------- FlowResult -----------------------------

def _synthetic_poiseuille_result():
    from hemoflow.solver.result import FlowResult
    nx, ny = 120, 60
    gx = np.linspace(0.0, 10.0, nx)
    gy = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(gx, gy)
    ux = 6.0 * Y * (1.0 - Y)
    uy = np.zeros_like(ux)
    p = 10.0 - X  # linear pressure drop
    mask = np.ones((ny, nx), dtype=bool)
    phi = np.full((ny, nx), 0.3)
    return FlowResult(grid_x=gx, grid_y=gy, grid_mask=mask, grid_ux=ux, grid_uy=uy,
                      grid_p=p, grid_phi=phi, params={"Re": 10.0}, metadata={"family": "G1", "variant": "V1"})


def test_flowresult_npz_roundtrip_and_streamlines(tmp_path):
    res = _synthetic_poiseuille_result()
    path = tmp_path / "flow.npz"
    res.to_npz(path)
    from hemoflow.solver.result import FlowResult
    res2 = FlowResult.from_npz(path)
    np.testing.assert_allclose(res2.grid_ux, res.grid_ux)
    np.testing.assert_allclose(res2.grid_p, res.grid_p)
    np.testing.assert_array_equal(res2.grid_mask, res.grid_mask)
    assert res2.grid_phi is not None
    np.testing.assert_allclose(res2.grid_phi, 0.3)
    assert res2.params["Re"] == 10.0
    assert res2.metadata["family"] == "G1"
    s = res2.summary()
    assert s["dp_star"] == pytest.approx(10.0)
    assert 1.49 < s["umax"] <= 1.5 + 1e-9  # discrete grid; max of 6y(1-y) <= 1.5
    assert s["phi_mean"] == pytest.approx(0.3)
    # streamlines: horizontal in +x
    lines = res2.compute_streamlines(n_seeds=8, ds=0.1, max_steps=200)
    assert len(lines) >= 4
    for ln in lines:
        assert ln[:, 0].max() > ln[:, 0].min()  # advected downstream
        assert np.all(np.abs(np.diff(ln[:, 1])) < 1e-6)  # y ~ constant for plane Poiseuille


def test_flowresult_derived_fields():
    res = _synthetic_poiseuille_result()
    g = res.shear_rate()
    # plane Poiseuille shear rate = |du/dy| = 6(1-2y); at centerline y=0.5 -> 0, at walls -> 6
    assert g[:, 0].max() > 5.0 or g.max() > 5.0
    w = res.vorticity()
    assert w.shape == res.grid_mask.shape


# ----------------------------- CLI -----------------------------

def test_cli_list_plugins_and_inspect_geometry(capsys):
    from hemoflow.cli import main
    assert main(["list-plugins"]) == 0
    out = capsys.readouterr().out
    assert "wu_high_shear" in out and "G2" in out and "phillips_nonlocal" in out
    assert main(["inspect-geometry", "G2"]) == 0
    out = capsys.readouterr().out
    assert "Sudden symmetric expansion" in out and "1 -> 1" in out
    assert main(["inspect-geometry", "G13", "--refine", "0.25"]) == 0
    out = capsys.readouterr().out
    assert "Trifurcation" in out and "1 -> 3" in out


def test_cli_module_entrypoint():
    # `python -m hemoflow.cli list-plugins` exits 0
    r = subprocess.run([sys.executable, "-m", "hemoflow.cli", "list-plugins"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "saffman_plus_twowall" in r.stdout
