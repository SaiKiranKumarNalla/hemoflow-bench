import numpy as np
import pytest

from rbc import postprocess as pp
from rbc.physics import Newtonian, solve_channel


def test_shear_stress_newtonian():
    gdot = np.array([0.0, 10.0, 3e4])
    assert np.allclose(pp.shear_stress(np.full(3, 3.5e-3), gdot), 3.5e-3 * gdot)


def test_trapezoid_weights_match_numpy():
    y = np.sort(np.random.default_rng(0).random(30))
    f = np.sin(y)
    assert np.sum(pp.trapezoid_weights(y) * f) == pytest.approx(np.trapezoid(f, y))


def test_window_fraction_poiseuille_is_17_over_49():
    # Newtonian, τ_w = 105 Pa: window |η| ∈ [2/7, 4/7]; ∫(1-η²) over both bands / (4/3) = 17/49.
    flow = solve_channel(Newtonian(3.5e-3), H=50e-6, U_mean=0.25, n_pts=40001)
    f = pp.window_fraction(flow.tau, flow.u, pp.trapezoid_weights(flow.y))
    assert f == pytest.approx(17 / 49, abs=1e-4)


def test_window_fraction_phi_uniform_cancels():
    flow = solve_channel(Newtonian(), H=50e-6, U_mean=0.25)
    w = pp.trapezoid_weights(flow.y)
    assert pp.window_fraction(flow.tau, flow.u, w, phi=np.full_like(flow.u, 0.45)) == pytest.approx(
        pp.window_fraction(flow.tau, flow.u, w))


def test_hemolysis_reference_value():
    assert pp.hemolysis_index(50.0, 0.01) == pytest.approx(0.0124, rel=1e-3)


def test_residence_time_is_inf_at_wall():
    t = pp.residence_time(np.array([0.0, 0.5]), length=1.0)
    assert np.isinf(t[0]) and t[1] == 2.0


def test_flux_weighted_mean_ignores_zero_flux_infinities():
    u = np.array([0.0, 1.0, 1.0, 0.0])
    v = np.array([np.inf, 2.0, 4.0, np.inf])
    assert pp.flux_weighted_mean(v, u, np.ones(4)) == pytest.approx(3.0)


def test_reattachment_length_linear_crossing():
    x = np.linspace(0.0, 10.0, 11)
    ux = x - 4.5                      # negative up to x = 4, crosses zero at 4.5
    assert pp.reattachment_length(x, ux, x_step=0.0) == pytest.approx(4.5)
    assert pp.reattachment_length(x, np.abs(ux) + 1.0) == 0.0


def test_pressure_drop_unsorted_input():
    assert pp.pressure_drop(np.array([2.0, 0.0, 1.0]), np.array([0.0, 8.0, 4.0])) == 8.0
