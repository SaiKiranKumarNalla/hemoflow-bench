"""Blood rheology and fully developed plane-channel flow (model M1, geometry G1).

Equations: knowledge/11-math-model.md §2.2 (viscosity models) and §2.4 (plane Poiseuille).

Fully developed flow, u = u(y): the x-momentum balance reduces to dτ/dy = -G with G = -dp/dx, so the
shear stress is linear across the channel for ANY generalized-Newtonian viscosity:

    τ(y) = G · |y - H/2|

The rheology only enters when converting τ back to a shear rate, γ̇ = τ⁻¹(τ), and integrating
du/dy = ±γ̇ from the wall. G is then found so that the mean velocity equals the imposed U_mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq


class Viscosity(Protocol):
    """Anything that maps shear rate -> viscosity and can invert stress -> shear rate."""

    @property
    def mu_min(self) -> float: ...

    @property
    def mu_max(self) -> float: ...

    def viscosity(self, gdot: np.ndarray) -> np.ndarray: ...

    def shear_rate_from_stress(self, tau: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Newtonian:
    """Constant viscosity. Default 3.5 mPa·s (whole blood, 2023 project)."""

    mu: float = 3.5e-3

    @property
    def mu_min(self) -> float:
        return self.mu

    @property
    def mu_max(self) -> float:
        return self.mu

    def viscosity(self, gdot: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(gdot, dtype=float), self.mu)

    def shear_rate_from_stress(self, tau: np.ndarray) -> np.ndarray:
        return np.asarray(tau, dtype=float) / self.mu


@dataclass(frozen=True)
class Carreau:
    """Carreau–Yasuda viscosity, Cho & Kensey 1991 constants (`TODO: verify` in 11-math-model §2.2).

    μ(γ̇) = μ∞ + (μ₀ − μ∞) [1 + (λγ̇)^a]^((n−1)/a)
    """

    mu0: float = 0.056
    mu_inf: float = 0.00345
    lam: float = 3.313
    n: float = 0.3568
    a: float = 2.0

    @property
    def mu_min(self) -> float:
        return self.mu_inf

    @property
    def mu_max(self) -> float:
        return self.mu0

    def viscosity(self, gdot: np.ndarray) -> np.ndarray:
        gdot = np.asarray(gdot, dtype=float)
        return self.mu_inf + (self.mu0 - self.mu_inf) * (1.0 + (self.lam * gdot) ** self.a) ** (
            (self.n - 1.0) / self.a
        )

    def shear_rate_from_stress(self, tau: np.ndarray, iters: int = 100) -> np.ndarray:
        """Invert τ = μ(γ̇)·γ̇ by vectorized bisection.

        τ(γ̇) is strictly increasing for n > 0, and μ ≥ μ∞ gives γ̇ ≤ τ/μ∞, so the root lies in
        [0, τ/μ∞]. Each iteration halves the bracket for every point at once (no Python loop over points).
        """
        tau = np.asarray(tau, dtype=float)
        lo = np.zeros_like(tau)
        hi = tau / self.mu_inf
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            too_big = self.viscosity(mid) * mid > tau
            hi = np.where(too_big, mid, hi)
            lo = np.where(too_big, lo, mid)
        return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ChannelFlow:
    """Fully developed plane-channel solution sampled on y ∈ [0, H]."""

    y: np.ndarray      # wall-normal coordinate [m]
    u: np.ndarray      # velocity [m/s]
    tau: np.ndarray    # shear stress magnitude [Pa]
    gdot: np.ndarray   # shear rate [1/s]
    G: float           # pressure gradient -dp/dx [Pa/m]
    H: float           # channel height [m]
    U_mean: float      # imposed mean velocity [m/s]

    @property
    def tau_wall(self) -> float:
        return self.G * self.H / 2.0


def _profile(model: Viscosity, y: np.ndarray, H: float, G: float) -> tuple[np.ndarray, ...]:
    eta = y - H / 2.0
    tau = G * np.abs(eta)
    gdot = model.shear_rate_from_stress(tau)
    u = cumulative_trapezoid(-np.sign(eta) * gdot, y, initial=0.0)  # du/dy = -sign(η)·γ̇, u(0) = 0
    return u, tau, gdot


def solve_channel(model: Viscosity, H: float, U_mean: float, n_pts: int = 401) -> ChannelFlow:
    """Fully developed flow at fixed mean velocity (flow rate per unit depth Q = U_mean·H).

    Bracket for G: μ_min ≤ μ ≤ μ_max everywhere, so the Newtonian results with μ_min and μ_max bound G:
    12·μ_min·U/H² ≤ G ≤ 12·μ_max·U/H². A 10 % margin absorbs the trapezoid-rule error.
    """
    if n_pts % 2 == 0:
        raise ValueError("n_pts must be odd so that the centreline y = H/2 is a grid point")
    y = np.linspace(0.0, H, n_pts)

    def mean_error(G: float) -> float:
        u, _, _ = _profile(model, y, H, G)
        return float(np.trapezoid(u, y) / H - U_mean)

    g_newton = 12.0 * U_mean / H**2
    G = brentq(mean_error, 0.9 * g_newton * model.mu_min, 1.1 * g_newton * model.mu_max, rtol=1e-12)
    u, tau, gdot = _profile(model, y, H, G)
    return ChannelFlow(y=y, u=u, tau=tau, gdot=gdot, G=G, H=H, U_mean=U_mean)
