"""Derived quantities from a flow solution (knowledge/11-math-model.md §4 and §6.3).

Works on any set of sample points carrying a weight (cell area dA in 2D CFD, quadrature weight
on a 1D cross-section). Velocities passed in are the components NORMAL to the cross-section.
"""
from __future__ import annotations

import numpy as np

WINDOW_PA: tuple[float, float] = (30.0, 60.0)  # shear window [Pa], 2023 project (knowledge/01-background)
GIERSIEPEN = {"C": 3.62e-5, "alpha": 2.416, "beta": 0.785}  # HI in %, τ in Pa, t in s (`TODO: verify`)


def shear_stress(mu: np.ndarray, gdot: np.ndarray) -> np.ndarray:
    """Scalar shear stress τ = μ(γ̇)·γ̇ [Pa]."""
    return np.asarray(mu, dtype=float) * np.asarray(gdot, dtype=float)


def trapezoid_weights(y: np.ndarray) -> np.ndarray:
    """Weights w such that sum(w * f) equals np.trapezoid(f, y)."""
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    w = np.zeros_like(y)
    w[:-1] += dy / 2.0
    w[1:] += dy / 2.0
    return w


def in_window(tau: np.ndarray, lo: float = WINDOW_PA[0], hi: float = WINDOW_PA[1]) -> np.ndarray:
    """Boolean mask of points whose shear stress lies in [lo, hi]."""
    tau = np.asarray(tau, dtype=float)
    return (tau >= lo) & (tau <= hi)


def flux_weighted_mean(
    values: np.ndarray,
    u: np.ndarray,
    weights: np.ndarray,
    phi: np.ndarray | None = None,
) -> float:
    """∫ v φ u dA / ∫ φ u dA — the average a passing RBC experiences.

    Points with u ≤ 0 carry no forward flux and are excluded, so values that are infinite there
    (e.g. residence time at the wall) do not poison the sum.
    """
    u = np.asarray(u, dtype=float)
    values = np.asarray(values, dtype=float)
    phi_arr = np.ones_like(u) if phi is None else np.asarray(phi, dtype=float)
    flux = np.where(u > 0.0, phi_arr * u * np.asarray(weights, dtype=float), 0.0)
    total = flux.sum()
    if total <= 0.0:
        raise ValueError("no forward flux through the section")
    safe_values = np.where(flux > 0.0, values, 0.0)
    return float((safe_values * flux).sum() / total)


def window_fraction(
    tau: np.ndarray,
    u: np.ndarray,
    weights: np.ndarray,
    phi: np.ndarray | None = None,
    lo: float = WINDOW_PA[0],
    hi: float = WINDOW_PA[1],
) -> float:
    """Flux-weighted fraction of RBCs crossing the section inside the shear window (§6.3 item 3)."""
    return flux_weighted_mean(in_window(tau, lo, hi).astype(float), u, weights, phi)


def residence_time(u: np.ndarray, length: float) -> np.ndarray:
    """Time to travel `length` at speed u along a straight streamline [s]; inf where u ≤ 0."""
    u = np.asarray(u, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(u > 0.0, length / np.where(u > 0.0, u, 1.0), np.inf)


def exposure_time(
    tau: np.ndarray,
    u: np.ndarray,
    length: float,
    lo: float = WINDOW_PA[0],
    hi: float = WINDOW_PA[1],
) -> np.ndarray:
    """Time spent inside the window along each straight streamline (G1: τ is constant along x)."""
    return np.where(in_window(tau, lo, hi), residence_time(u, length), 0.0)


def hemolysis_index(
    tau: np.ndarray,
    t: np.ndarray,
    C: float = GIERSIEPEN["C"],
    alpha: float = GIERSIEPEN["alpha"],
    beta: float = GIERSIEPEN["beta"],
) -> np.ndarray:
    """Power-law hemolysis HI(%) = C·τ^α·t^β (Giersiepen et al. 1990, constant τ along the path)."""
    return C * np.asarray(tau, dtype=float) ** alpha * np.asarray(t, dtype=float) ** beta


def pressure_drop(x: np.ndarray, p: np.ndarray) -> float:
    """Inlet minus outlet pressure from samples along the channel."""
    order = np.argsort(np.asarray(x, dtype=float))
    p_sorted = np.asarray(p, dtype=float)[order]
    return float(p_sorted[0] - p_sorted[-1])


def reattachment_length(x: np.ndarray, ux_near_wall: np.ndarray, x_step: float = 0.0) -> float:
    """Distance from the step to where near-wall ux turns from negative back to positive (G2).

    Returns 0.0 when there is no recirculation. Linear interpolation between the last negative
    sample and the next one locates the zero crossing.
    """
    order = np.argsort(np.asarray(x, dtype=float))
    x = np.asarray(x, dtype=float)[order]
    ux = np.asarray(ux_near_wall, dtype=float)[order]
    neg = np.flatnonzero((ux < 0.0) & (x >= x_step))
    if neg.size == 0:
        return 0.0
    k = int(neg[-1])
    if k == x.size - 1:
        raise ValueError("recirculation reaches the end of the sampled line; extend the domain")
    x_zero = x[k] - ux[k] * (x[k + 1] - x[k]) / (ux[k + 1] - ux[k])
    return float(x_zero - x_step)


def channel_scalars(
    y: np.ndarray,
    u: np.ndarray,
    tau: np.ndarray,
    G: float,
    H: float,
    length: float,
) -> dict[str, float]:
    """All scalar targets for one fully developed G1 run."""
    w = trapezoid_weights(y)
    t_res = residence_time(u, length)
    return {
        "G": float(G),
        "dp": float(G * length),
        "tau_wall": float(G * H / 2.0),
        "f_window": window_fraction(tau, u, w),
        "t_exp_mean": flux_weighted_mean(exposure_time(tau, u, length), u, w),
        "hi_mean": flux_weighted_mean(hemolysis_index(tau, t_res), u, w),
    }
