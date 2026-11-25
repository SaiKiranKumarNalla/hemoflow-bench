"""Whole-blood rheology for the paired 4-variant corpus (V1 ``N`` and V2 ``GN``).

Sources and labels are in ``knowledge/16-haematocrit-rheology.md``. Short version:

- Quemada law with the coefficients printed by Mimouni 2016 §3.1.3 (``verified``):
  ``mu(gdot, phi) = mu_p * (1 - 0.5 * (k0 + k_inf*sqrt(gdot/gdot_c)) / (1 + sqrt(gdot/gdot_c)) * phi)**-2``
  with ``k0 = 4.33``, ``k_inf = 2.07``, ``gdot_c = 1.88 1/s``. The fit is at H = 0.45 and the source states no
  temperature; using the same law at other haematocrits is the standard Quemada extrapolation and is FLAGged.
- ``LAMBDA_BLOOD = 0.110 s``: Jung et al.'s Carreau time constant as used by Sezer 2026 Eqs. 5-11 (``verified``).
- No published phi-dependent Carreau fit exists in readable form (knowledge/16 §3), so V2's BirdCarreau parameters
  are **derived here** by fitting the Quemada curve at each phi: levels ``mu_inf`` from the high-shear limit,
  ``lambda = 0.110 s``, ``a = 2``, and ``mu_0, n`` from a least-squares fit. The fit residual is returned and stored
  per record.

Nothing here invents a literature constant: the only numbers are the verified Quemada coefficients, the verified
Carreau time constant, and the fit that this module performs reproducibly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# Verified Quemada coefficients (Mimouni 2016 §3.1.3; knowledge/16 §2).
QUEMADA_K0 = 4.33
QUEMADA_K_INF = 2.07
QUEMADA_GDOT_C = 1.88  # 1/s

# Verified Carreau time constant (Jung et al. via Sezer 2026 Eqs. 5-11; knowledge/16 §3).
LAMBDA_BLOOD = 0.110  # s
CARREAU_A = 2.0  # Carreau-Yasuda exponent used by the corpus (TODO: verify citation)

# Carrier viscosities in Pa.s (knowledge/16 §4).
MU_PLASMA = 0.96e-3  # Wu 2015 §3 / Sezer 2026 §2.2 (verified)
MU_BUFFER_RANGE = (1.1e-3, 1.7e-3)  # Bernardelli 2025 Table 3 suspension range (verified)
BUFFER_MAX_PHI = 0.1  # buffer carrier only for the device regime (CORPUS_PLAN §6.3)

DEFAULT_SHEAR_RANGE = (10.0, 1.0e4)  # 1/s, fit window for the derived BirdCarreau parameters


def quemada_mu(phi: float, gdot: float, mu_p: float = MU_PLASMA) -> float:
    """Whole-blood dynamic viscosity [Pa.s] from the Quemada law at haematocrit ``phi``.

    Raises ``ValueError`` when the bracket closes (the law is singular at zero shear above
    phi ~ 0.46 with k0 = 4.33), instead of returning a negative or complex viscosity.
    """
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"phi must be in [0, 1), got {phi}")
    if gdot <= 0.0:
        raise ValueError(f"gdot must be > 0 to evaluate the Quemada law, got {gdot}")
    root = math.sqrt(gdot / QUEMADA_GDOT_C)
    k_eff = (QUEMADA_K0 + QUEMADA_K_INF * root) / (1.0 + root)
    bracket = 1.0 - 0.5 * k_eff * phi
    if bracket <= 0.0:
        raise ValueError(
            f"Quemada bracket closed at phi={phi:.3f}, gdot={gdot:g} 1/s "
            f"(bracket={bracket:.4f}); evaluate at a higher shear rate"
        )
    return mu_p * bracket**-2.0


def quemada_high_shear_mu(phi: float, mu_p: float = MU_PLASMA) -> float:
    """High-shear limit mu_inf(phi) [Pa.s] of the Quemada law (gamma_dot >> gamma_c)."""
    if not (0.0 <= phi < 1.0):
        raise ValueError(f"phi must be in [0, 1), got {phi}")
    bracket = 1.0 - 0.5 * QUEMADA_K_INF * phi
    if bracket <= 0.0:
        raise ValueError(f"Quemada high-shear bracket closed at phi={phi}")
    return mu_p * bracket**-2.0


def plane_poiseuille_mean_shear_rate(U: float, h: float) -> float:
    """Mean |du/dy| of a plane Poiseuille profile, ``3 U / h``, used as gamma_dot_ref.

    This is a modelling choice for the Newtonian variant V1: a Newtonian surrogate of a
    shear-thinning fluid should match the viscosity at the shear rate that dominates the flow,
    and the cross-section average of |du/dy| = 6U(1 - 2y/h)/h is 3U/h.
    """
    if U <= 0.0 or h <= 0.0:
        raise ValueError(f"U and h must be > 0, got U={U}, h={h}")
    return 3.0 * U / h


@dataclass(frozen=True)
class CarreauFit:
    """BirdCarreau parameters of whole blood at one haematocrit, in absolute units."""

    phi: float
    mu_0: float  # Pa.s
    mu_inf: float  # Pa.s
    lam: float  # s
    n: float
    a: float
    max_rel_err: float  # largest |fit - Quemada| / Quemada over the fit window
    gdot_range: tuple[float, float]

    @property
    def beta(self) -> float:
        """Zero-shear to infinite-shear viscosity ratio mu_0 / mu_inf (the corpus ``beta``)."""
        return self.mu_0 / self.mu_inf

    def mu(self, gdot: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the fitted BirdCarreau viscosity [Pa.s]."""
        g = np.asarray(gdot, dtype=np.float64)
        ratio = (1.0 + (self.lam * g) ** self.a) ** ((self.n - 1.0) / self.a)
        out = self.mu_inf + (self.mu_0 - self.mu_inf) * ratio
        return float(out) if np.isscalar(gdot) else out


def carreau_params_from_quemada(
    phi: float,
    mu_p: float = MU_PLASMA,
    *,
    gdot_range: tuple[float, float] = DEFAULT_SHEAR_RANGE,
    lam: float = LAMBDA_BLOOD,
    a: float = CARREAU_A,
    n_points: int = 200,
) -> CarreauFit:
    """Fit ``(mu_0, n)`` of a BirdCarreau law to the Quemada curve at ``phi``.

    ``mu_inf`` is the analytic high-shear limit and ``lam``/``a`` are the verified/assumed
    constants, so only the two shape parameters are fitted. The fit is a log-space least squares
    over ``gdot_range``; the returned ``max_rel_err`` is the largest relative deviation between
    the fitted Carreau curve and the Quemada law inside that window.
    """
    mu_inf = quemada_high_shear_mu(phi, mu_p)
    gmin, gmax = gdot_range
    gd = np.logspace(math.log10(gmin), math.log10(gmax), n_points)
    target = np.array([quemada_mu(phi, float(g), mu_p) for g in gd])
    log_target = np.log(target)

    def residual(x: np.ndarray) -> np.ndarray:
        mu_0 = math.exp(x[0])
        n_try = x[1]
        ratio = (1.0 + (lam * gd) ** a) ** ((n_try - 1.0) / a)
        return np.log(mu_inf + (mu_0 - mu_inf) * ratio) - log_target

    sol = least_squares(
        residual,
        x0=np.array([math.log(5.0 * mu_inf), 0.6]),
        bounds=(np.array([math.log(mu_inf * 1.0000001), 0.05]), np.array([math.log(1000.0 * mu_p), 1.0])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    fit = CarreauFit(
        phi=phi,
        mu_0=float(math.exp(sol.x[0])),
        mu_inf=mu_inf,
        lam=lam,
        n=float(sol.x[1]),
        a=a,
        max_rel_err=0.0,
        gdot_range=gdot_range,
    )
    rel = np.abs(fit.mu(gd) - target) / target
    return CarreauFit(
        phi=fit.phi,
        mu_0=fit.mu_0,
        mu_inf=fit.mu_inf,
        lam=fit.lam,
        n=fit.n,
        a=fit.a,
        max_rel_err=float(np.max(rel)),
        gdot_range=gdot_range,
    )


def carrier_viscosity(phi: float, u01: float, mu_p: float = MU_PLASMA) -> float:
    """Carrier viscosity [Pa.s] for a design: plasma for whole blood, buffer for phi <= 0.1.

    ``u01`` is a Sobol coordinate in [0, 1) that samples the verified buffer range
    (1.1-1.7 mPa s, Bernardelli 2025 Table 3) in log space.
    """
    if phi > BUFFER_MAX_PHI:
        return mu_p
    lo, hi = MU_BUFFER_RANGE
    return float(math.exp(math.log(lo) + u01 * (math.log(hi) - math.log(lo))))
