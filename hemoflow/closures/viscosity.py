"""Mixture / phase viscosity closures (dual NumPy+UFL).

All closures return the **relative** viscosity ``mu_hat = mu / mu_plasma`` so they plug
straight into the nondimensional Euler-Euler residual (star units, mu_ref = mu_p), except
:func:`quemada_mu` / :func:`carreau_mu` which return absolute Pa.s.  Equations and
constants are ported verbatim from ``twofluid.py`` (Wu 2015 Eq. 7/8/9a/9b) and
``campaign/blood_rheology.py`` (Quemada / Bird-Carreau).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hemoflow.registry import register_viscosity
from . import _dual as D
from .constants import (
    CARREAU_A, LAMBDA_BLOOD, MU_PLASMA, MU_PLASMA_CP,
    QUEMADA_GDOT_C, QUEMADA_K0, QUEMADA_K_INF,
)

__all__ = [
    "mu0_cp", "mu_inf_cp", "newtonian", "wu_high_shear", "wu_shear_thinning",
    "krieger", "quemada", "quemada_mu", "bird_carreau", "CarreauFit",
    "carreau_params_from_quemada", "plane_poiseuille_mean_shear_rate",
]


def mu0_cp(phi):
    """Wu 2015 Eq. 9a, zero-shear RBC contribution (cP)."""
    return 537.002 * phi**3 + 55.006 * phi**2 - 0.129 * phi


def mu_inf_cp(phi):
    """Wu 2015 Eq. 9b, high-shear RBC contribution (cP)."""
    return 27.873 * phi**3 - 21.218 * phi**2 + 14.439 * phi


@register_viscosity("newtonian")
def newtonian(phi=None, mu_hat: float = 1.0):
    """Newtonian constant viscosity ``mu_hat`` (default 1.0 = plasma; or ``mu_eff/mu_p``)."""
    if phi is None:
        return float(mu_hat)
    return mu_hat + 0.0 * phi  # broadcast to array shape, works for ufl too


@register_viscosity("wu_high_shear")
def wu_high_shear(phi):
    """Wu 2015 Eq. 8 with Eq. 9b: ``mu_hat = (1 - phi) + 0.5 * mu_inf_cp(phi) / MU_PLASMA_CP``."""
    return (1.0 - phi) + 0.5 * mu_inf_cp(phi) / MU_PLASMA_CP


@register_viscosity("wu_shear_thinning")
def wu_shear_thinning(phi, gdot, k_star: float = 0.0):
    """Wu 2015 Eq. 7 with Eq. 9a-c (RBC-phase relative viscosity, floored at 1e-4).

    ``mu_r = [mu_inf + (mu0 - mu_inf) (1 + ln(1 + k g)) / (1 + k g)] / MU_PLASMA_CP``
    with ``k g = k_star * gdot`` (star units, k_star = k U_ref / L_ref).
    """
    kg = k_star * gdot
    num = mu_inf_cp(phi) + (mu0_cp(phi) - mu_inf_cp(phi)) * (1.0 + D.ln(1.0 + kg)) / (1.0 + kg)
    return D.maximum(num / MU_PLASMA_CP, 1.0e-4)


@register_viscosity("krieger")
def krieger(phi, phi_max: float = 2.0 / 2.07):
    """Krieger-Dougherty (COMSOL solids): ``(1 - min(phi, 0.999 phi_max)/phi_max)^(-2.5 phi_max)``."""
    phi_k = D.minimum(phi, 0.999 * phi_max)
    return (1.0 - phi_k / phi_max) ** (-2.5 * phi_max)


def _quemada_bracket(phi, gdot):
    root = D.sqrt(gdot / QUEMADA_GDOT_C)
    k_eff = (QUEMADA_K0 + QUEMADA_K_INF * root) / (1.0 + root)
    return 1.0 - 0.5 * k_eff * phi


@register_viscosity("quemada")
def quemada(phi, gdot):
    """Quemada relative viscosity ``mu/mu_p = bracket^-2`` (dimensionless hat).

    Coefficients k0=4.33, k_inf=2.07, gdot_c=1.88 1/s (Mimouni 2016 §3.1.3, verified).
    """
    return _quemada_bracket(phi, gdot) ** -2.0


def quemada_mu(phi, gdot, mu_p: float = MU_PLASMA):
    """Quemada dynamic viscosity [Pa.s] (verbatim :func:`blood_rheology.quemada_mu`)."""
    phi = float(phi)
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"phi must be in [0, 1), got {phi}")
    if gdot <= 0.0:
        raise ValueError(f"gdot must be > 0 to evaluate the Quemada law, got {gdot}")
    b = float(_quemada_bracket(phi, gdot))
    if b <= 0.0:
        raise ValueError(f"Quemada bracket closed at phi={phi:.3f}, gdot={gdot:g} 1/s")
    return mu_p * b**-2.0


@register_viscosity("bird_carreau")
def bird_carreau(gdot, mu_0, mu_inf, lam: float = LAMBDA_BLOOD, n: float = 0.6, a: float = CARREAU_A):
    """Bird-Carreau-Yasuda: ``mu_inf + (mu_0 - mu_inf)(1 + (lam gdot)^a)^((n-1)/a)``.

    Returns whatever units ``mu_0``/``mu_inf`` are in (Pa.s or dimensionless hat).
    """
    return mu_inf + (mu_0 - mu_inf) * (1.0 + (lam * gdot) ** a) ** ((n - 1.0) / a)


def plane_poiseuille_mean_shear_rate(U: float, h: float) -> float:
    """Mean |du/dy| of a plane Poiseuille profile, ``3 U / h`` (verbatim blood_rheology)."""
    if U <= 0.0 or h <= 0.0:
        raise ValueError(f"U and h must be > 0, got U={U}, h={h}")
    return 3.0 * U / h


@dataclass(frozen=True)
class CarreauFit:
    """Bird-Carreau parameters of whole blood at one haematocrit (verbatim blood_rheology)."""

    phi: float
    mu_0: float
    mu_inf: float
    lam: float
    n: float
    a: float
    max_rel_err: float
    gdot_range: tuple[float, float]

    @property
    def beta(self) -> float:
        return self.mu_0 / self.mu_inf

    def mu(self, gdot):
        g = np.asarray(gdot, dtype=np.float64)
        ratio = (1.0 + (self.lam * g) ** self.a) ** ((self.n - 1.0) / self.a)
        out = self.mu_inf + (self.mu_0 - self.mu_inf) * ratio
        return float(out) if np.isscalar(gdot) else out


def carreau_params_from_quemada(
    phi: float,
    mu_p: float = MU_PLASMA,
    *,
    gdot_range: tuple[float, float] = (10.0, 1.0e4),
    lam: float = LAMBDA_BLOOD,
    a: float = CARREAU_A,
    n_points: int = 200,
) -> CarreauFit:
    """Fit ``(mu_0, n)`` of a Bird-Carreau law to the Quemada curve at ``phi`` (verbatim)."""
    from scipy.optimize import least_squares

    mu_inf = mu_p * (1.0 - 0.5 * QUEMADA_K_INF * phi) ** -2.0
    gmin, gmax = gdot_range
    gd = np.logspace(math.log10(gmin), math.log10(gmax), n_points)
    target = np.array([quemada_mu(phi, float(g), mu_p) for g in gd])
    log_target = np.log(target)

    def residual(x):
        mu_0 = math.exp(x[0])
        n_try = x[1]
        ratio = (1.0 + (lam * gd) ** a) ** ((n_try - 1.0) / a)
        return np.log(mu_inf + (mu_0 - mu_inf) * ratio) - log_target

    sol = least_squares(
        residual,
        x0=np.array([math.log(5.0 * mu_inf), 0.6]),
        bounds=(np.array([math.log(mu_inf * 1.0000001), 0.05]), np.array([math.log(1000.0 * mu_p), 1.0])),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    fit = CarreauFit(
        phi=phi, mu_0=float(math.exp(sol.x[0])), mu_inf=mu_inf, lam=lam,
        n=float(sol.x[1]), a=a, max_rel_err=0.0, gdot_range=gdot_range,
    )
    rel = np.abs(fit.mu(gd) - target) / target
    return CarreauFit(phi=fit.phi, mu_0=fit.mu_0, mu_inf=fit.mu_inf, lam=fit.lam,
                      n=fit.n, a=fit.a, max_rel_err=float(np.max(rel)), gdot_range=gdot_range)
