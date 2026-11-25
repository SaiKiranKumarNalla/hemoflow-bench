import numpy as np
import pytest

from rbc.physics import Carreau, Newtonian, solve_channel


def test_newtonian_channel_matches_poiseuille():
    mu, H, U = 3.5e-3, 50e-6, 0.25
    flow = solve_channel(Newtonian(mu), H=H, U_mean=U, n_pts=2001)
    assert flow.G == pytest.approx(12 * mu * U / H**2, rel=1e-5)          # G = 12 μ U / H²
    assert flow.tau_wall == pytest.approx(6 * mu * U / H, rel=1e-5)       # τ_w = 6 μ U / H = 105 Pa
    assert flow.u.max() == pytest.approx(1.5 * U, rel=1e-5)               # u_max = 3/2 U_mean


def test_carreau_limits():
    c = Carreau()
    assert c.viscosity(np.array([0.0]))[0] == pytest.approx(c.mu0)
    assert c.viscosity(np.array([1e9]))[0] == pytest.approx(c.mu_inf, rel=1e-3)


def test_carreau_inversion_roundtrip():
    c = Carreau()
    gdot = np.logspace(-3, 6, 50)
    tau = c.viscosity(gdot) * gdot
    assert np.allclose(c.shear_rate_from_stress(tau), gdot, rtol=1e-9)


def test_carreau_channel_is_blunter_than_newtonian():
    flow = solve_channel(Carreau(), H=50e-6, U_mean=0.25)
    assert flow.u.max() / 0.25 < 1.5                                       # shear-thinning flattens the core
    assert np.trapezoid(flow.u, flow.y) / 50e-6 == pytest.approx(0.25, rel=1e-9)
