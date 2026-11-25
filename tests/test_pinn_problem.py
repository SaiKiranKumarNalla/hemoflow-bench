"""Tests for the physics-only problem built from the legacy M1 corpus (T9)."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from pinn.problem import (
    BC_INLET,
    BC_OUTLET,
    BC_WALL,
    CORPUS_ROOT,
    Problem,
    build_problem,
    read_manifest,
    sample_problem,
)

pytestmark = pytest.mark.skipif(
    not (CORPUS_ROOT / "manifest.jsonl").exists(),
    reason="legacy corpus not downloaded (data/ is gitignored)",
)


@functools.lru_cache(maxsize=1)
def _problem() -> Problem:
    return build_problem()


def _g2_domain(problem: Problem) -> tuple[float, float, float]:
    """``(L_in, L_out, s)`` of the G2 case under test."""
    x_in, _, _ = problem.inlet_section
    x_out = float(problem.segments[BC_OUTLET][:, 0, 0].max())
    return -x_in, x_out, problem.step_height


def test_selected_case_is_lowest_re_newtonian_g2() -> None:
    """The default case is the lowest-Re ``ok`` Newtonian G2 run in the manifest."""
    problem = _problem()
    assert problem.family == "G2"
    assert problem.rheology == "Newtonian"
    assert problem.run_id == "M1_G2_052"
    candidates = [
        row["params"]["Re"]
        for row in read_manifest()
        if row["params"].get("family") == "G2"
        and row["params"].get("rheology") == "Newtonian"
        and row["status"] == "ok"
    ]
    assert problem.Re == pytest.approx(min(candidates), rel=1e-12)


def test_rebuilt_boundary_matches_g2_geometry() -> None:
    """Boundary segments reproduce the analytic G2 outline (independent of the sampler)."""
    problem = _problem()
    l_in, l_out, s = _g2_domain(problem)
    assert problem.inlet_section[0] == pytest.approx(-l_in, abs=1e-9)
    inlet_len = float(np.hypot(*np.diff(problem.segments[BC_INLET], axis=1).reshape(-1)))
    outlet_len = float(np.sum(np.hypot(*np.diff(problem.segments[BC_OUTLET], axis=1).reshape(-1, 2).T)))
    wall_len = float(np.sum(np.hypot(*np.diff(problem.segments[BC_WALL], axis=1).reshape(-1, 2).T)))
    assert inlet_len == pytest.approx(1.0, rel=1e-9)
    assert outlet_len == pytest.approx(1.0 + 2.0 * s, rel=1e-9)
    assert wall_len == pytest.approx(2.0 * (l_in + l_out + s), rel=1e-9)


@pytest.mark.parametrize("kind", [BC_INLET, BC_WALL, BC_OUTLET])
def test_boundary_points_lie_on_the_geometry(kind: str) -> None:
    """Every sampled boundary point lies on the analytic G2 outline (within 1e-8)."""
    problem = _problem()
    l_in, l_out, s = _g2_domain(problem)
    pts = problem.sample_boundary(kind, 500, np.random.default_rng(1))
    tol = 1e-8

    if kind == BC_INLET:
        assert np.allclose(pts[:, 0], -l_in, atol=tol)
        assert np.all(pts[:, 1] >= -tol) and np.all(pts[:, 1] <= 1.0 + tol)
        return

    if kind == BC_OUTLET:
        assert np.allclose(pts[:, 0], l_out, atol=tol)
        assert np.all(pts[:, 1] >= -s - tol) and np.all(pts[:, 1] <= 1.0 + s + tol)
        return

    x, y = pts[:, 0], pts[:, 1]
    on_inlet_walls = (x >= -l_in - tol) & (x <= tol) & (
        np.isclose(y, 0.0, atol=tol) | np.isclose(y, 1.0, atol=tol)
    )
    on_outlet_walls = (x >= -tol) & (x <= l_out + tol) & (
        np.isclose(y, -s, atol=tol) | np.isclose(y, 1.0 + s, atol=tol)
    )
    on_step = np.isclose(x, 0.0, atol=tol) & (
        ((y >= -s - tol) & (y <= tol)) | ((y >= 1.0 - tol) & (y <= 1.0 + s + tol))
    )
    assert np.all(on_inlet_walls | on_outlet_walls | on_step)


def test_interior_points_are_inside_the_fluid() -> None:
    """Collocation points land in the fluid (union of the two G2 channels)."""
    problem = _problem()
    l_in, l_out, s = _g2_domain(problem)
    pts = problem.sample_interior(4000, np.random.default_rng(2))
    x, y = pts[:, 0], pts[:, 1]
    inlet_channel = (x >= -l_in) & (x <= 0.0) & (y >= 0.0) & (y <= 1.0)
    outlet_channel = (x >= 0.0) & (x <= l_out) & (y >= -s) & (y <= 1.0 + s)
    assert np.all(inlet_channel | outlet_channel)


def test_inlet_profile_has_mean_one_and_is_parabolic() -> None:
    """Parabolic inlet BC: zero at both walls, peak 1.5 at mid-height, mean exactly 1."""
    problem = _problem()
    assert problem.parabolic_inlet
    _, y_lo, y_hi = problem.inlet_section
    y = np.linspace(y_lo, y_hi, 2001)
    pts = np.column_stack([np.full_like(y, problem.inlet_section[0]), y])
    u, v = problem.inlet_velocity(pts)
    assert np.allclose(v, 0.0)
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert u[-1] == pytest.approx(0.0, abs=1e-12)
    assert u.max() == pytest.approx(1.5, rel=1e-6)
    # trapezoid on 2001 nodes is exact to O(dy^2) for the parabola
    assert float(np.trapezoid(u, y) / (y_hi - y_lo)) == pytest.approx(1.0, rel=1e-6)


def test_sampled_inlet_points_reproduce_unit_mean() -> None:
    """A Monte-Carlo average over sampled inlet points gives the unit mean flow rate."""
    problem = _problem()
    pts = problem.sample_boundary(BC_INLET, 20000, np.random.default_rng(3))
    u, _ = problem.inlet_velocity(pts)
    assert float(np.mean(u)) == pytest.approx(1.0, abs=0.01)


def test_reference_cloud_is_the_cfd_point_cloud() -> None:
    """The reference fields are the CFD cell centres, used for evaluation only."""
    points = sample_problem(_problem(), n_interior=64, n_inlet=8, n_wall=8, n_outlet=8)
    ref = _problem().reference
    assert points["reference_x"].shape == (ref["x"].size, 2)
    assert np.allclose(points["reference_x"][:, 0], ref["x"])
    assert ref["u"].shape == ref["x"].shape


def test_momentum_balance_scaling_of_the_stored_pressure() -> None:
    """The stored ``p_star`` is viscous-scaled: ``dp*/dx = d2u*/dy*2`` in developed flow.

    This pins the residual convention used by ``pinn/pinn.py`` (1/Re on both the pressure
    gradient and the Laplacian, none on advection).
    """
    problem = _problem()
    npz = np.load(CORPUS_ROOT / "m1" / problem.family / f"{problem.run_id}.npz", allow_pickle=True)
    gx, gy, mask = npz["grid_x"], npz["grid_y"], npz["grid_mask"] > 0.5
    u = np.where(mask, npz["grid_ux"], np.nan)
    p = np.where(mask, npz["grid_p"], np.nan)

    # Fully developed inlet channel, away from the walls and the expansion.
    band = mask & (gx[None, :] < -1.0) & (gx[None, :] > -4.0)
    band &= (gy[:, None] > 0.25) & (gy[:, None] < 0.75)

    dp_dx = np.gradient(p, gx, axis=1)[band]
    d2u_dy2 = np.gradient(np.gradient(u, gy, axis=0), gy, axis=0)[band]
    assert float(np.nanmean(dp_dx)) == pytest.approx(float(np.nanmean(d2u_dy2)), rel=0.02)


def test_relative_l2_and_gauge_offset() -> None:
    """Error metrics behave: exact match, pure offset removed by the gauge, sign convention."""
    from pinn.evaluate import gauge_offset, relative_l2

    ref = np.array([1.0, -2.0, 3.0, 0.5])
    weights = np.array([1.0, 1.0, 2.0, 1.0])
    assert relative_l2(ref, ref, weights) == pytest.approx(0.0, abs=1e-15)

    offset = gauge_offset(ref + 2.5, ref, weights)
    assert offset == pytest.approx(-2.5, rel=1e-12)
    assert relative_l2(ref + 2.5 + offset, ref, weights) == pytest.approx(0.0, abs=1e-12)


def test_analytic_poiseuille_has_zero_residual() -> None:
    """Guard the residual convention: the exact channel solution has zero momentum residual.

    ``u = 6y(1-y)``, ``v = 0``, ``p = -12 x`` (star units) is plane Poiseuille with mean 1, so
    ``p_x = u_yy = -12`` must cancel in the form used by ``pinn/pinn.py``.
    """
    import torch

    from pinn.pinn import GeometryScale, pde_residuals

    scale = GeometryScale.from_problem(_problem())

    class Exact(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, xy: torch.Tensor):
            x, y = xy[:, 0:1], xy[:, 1:2]
            u = 6.0 * y * (1.0 - y)
            return u, 0.0 * u, -12.0 * x + 0.0 * u

    xy = torch.tensor([[-3.0, 0.3], [-3.0, 0.7], [2.0, 0.2]])
    _, (r_x, r_y, r_c) = pde_residuals(Exact(), xy)
    assert float(torch.max(torch.abs(torch.cat([r_x, r_y, r_c]))).detach()) == pytest.approx(0.0, abs=1e-10)


def test_section_flux_of_the_exact_inlet_parabola_is_one() -> None:
    """The flux diagnostic returns 1 for the exact inlet parabola."""
    import torch

    from pinn.evaluate import section_flux
    from pinn.pinn import GeometryScale

    scale = GeometryScale.from_problem(_problem())

    class Parabolic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, xy: torch.Tensor):
            u = 6.0 * xy[:, 1:2] * (1.0 - xy[:, 1:2])
            return u, 0.0 * u, 0.0 * u

    # the parabola carries unit flux at every section of the inlet channel
    for x in (-4.5, -2.0, -0.2):
        assert section_flux(Parabolic(), x, 0.0, 1.0, n_points=400) == pytest.approx(1.0, rel=1e-4)
