"""Host-safe analytic profiles (NumPy only), ported verbatim from ``twofluid.py``.

No dolfinx/ufl import here, so this module imports on any host.
"""
from __future__ import annotations

import numpy as np

from hemoflow.closures.constants import K_C, K_MU, PHI_MAX

__all__ = ["poiseuille", "brinkman_inlet", "developed_phillips_profile", "mixture_viscosity_hat"]


def poiseuille(y):
    """Developed plane-Poiseuille profile, channel y in [0, 1], mean 1."""
    return 6.0 * y * (1.0 - y)


def brinkman_inlet(y, y_half: float = 0.5, depth_hat: float = 0.0):
    """Developed 2.5D Brinkman profile on [-y_half, y_half] with unit mean (Poiseuille when depth_hat <= 0)."""
    if depth_hat <= 0.0:
        return 1.5 * (1.0 - (y / y_half) ** 2)
    kb = np.sqrt(12.0) / depth_hat
    kh = min(float(kb * y_half), 200.0)
    ay = np.abs(y)
    ratio = (np.exp(kb * (ay - y_half)) + np.exp(-kb * (ay + y_half))) / (1.0 + np.exp(-2.0 * kh))
    return (1.0 - ratio) / (1.0 - np.tanh(kh) / kh)


def mixture_viscosity_hat(phi):
    """Newtonian high-shear mixture viscosity / mu_p (Wu Eq. 8 with Eq. 9b)."""
    from hemoflow.closures.viscosity import wu_high_shear
    return wu_high_shear(phi)


def developed_phillips_profile(phi_bulk: float, n: int = 401, mu_fun=None,
                               phi_cap: float = PHI_MAX, tau_floor: float = 1e-9):
    """Fully developed planar channel y in [0,1] with Phillips migration (comsol form, u = w).

    N_y = 0 and the linear shear stress tau ~ |y - 1/2| give
    ``phi * tau * mu(phi)^(K_mu/K_c - 1) = C`` (algebraic), capped at ``phi_cap``; ``C`` is
    set so that the flux-weighted mean phi equals ``phi_bulk``.  Returns ``(y, u, phi)`` with
    mean u = 1.  Verbatim from ``twofluid.developed_phillips_profile``.
    """
    def brentq(g, lo, hi, tol=1e-12):  # plain bisection (no scipy needed here)
        glo = g(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if (gm > 0) == (glo > 0):
                lo, glo = mid, gm
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    mu_of = mu_fun if mu_fun is not None else mixture_viscosity_hat
    y = np.linspace(0.0, 1.0, n)
    tau = np.abs(y - 0.5) + tau_floor
    expo = K_MU / K_C - 1.0

    def phi_of(c):
        out = np.empty_like(y)
        for i, t in enumerate(tau):
            g = lambda f: f * t * mu_of(f) ** expo - c  # noqa: E731
            out[i] = phi_cap if g(phi_cap) <= 0 else brentq(g, 0.0, phi_cap)
        return out

    def profile(c):
        phi = phi_of(c)
        mu = mu_of(phi)
        dudy = -(y - 0.5) / mu
        u = np.concatenate([[0.0], np.cumsum(0.5 * (dudy[1:] + dudy[:-1]) * np.diff(y))])
        u /= np.trapezoid(u, y)
        return phi, u

    def mismatch(c):
        phi, u = profile(c)
        return np.trapezoid(phi * u, y) - phi_bulk

    c = brentq(mismatch, 1e-8, 10.0)
    phi, u = profile(c)
    return y, u, phi
