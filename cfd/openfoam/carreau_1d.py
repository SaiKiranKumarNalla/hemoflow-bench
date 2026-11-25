"""1D shooting reference for fully developed plane Carreau blood flow (M1 validation).

Solves:
    tau(eta) = mu(gamma_dot) * gamma_dot = G * |eta|,   eta = y - H/2 in [-H/2, H/2]
    du/deta  = -sign(eta) * gamma_dot,                  u(-H/2) = 0
and shoots for G = -dp/dx so that mean(u) = U_mean.
Constants: Cho & Kensey 1991 (knowledge/11-math-model.md §2.2).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

H = 50e-6          # channel height [m]
U_MEAN = 0.25      # mean inlet velocity [m/s] (decision 2026-11-24)
RHO = 1060.0       # blood density [kg/m^3]
MU0 = 0.056        # zero-shear viscosity [Pa.s]
MU_INF = 0.00345   # infinite-shear viscosity [Pa.s]
LAM = 3.313        # relaxation time [s]
N = 0.3568         # power-law index [-]
A = 2.0            # Carreau exponent [-]


def mu_carreau(gdot: float | np.ndarray) -> float | np.ndarray:
    """Cho & Kensey (1991) Carreau viscosity [Pa.s]."""
    return MU_INF + (MU0 - MU_INF) * (1.0 + (LAM * gdot) ** A) ** ((N - 1.0) / A)


def solve_carreau_1d(height: float = H, u_mean: float = U_MEAN, n_pts: int = 401):
    eta = np.linspace(-height / 2.0, height / 2.0, n_pts)

    def profile(G: float):
        tau = G * np.abs(eta)
        gdot = np.array([
            0.0 if t == 0.0 else brentq(lambda g: mu_carreau(g) * g - t, 0.0, max(t / MU_INF, 1.0) * 2.0)
            for t in tau
        ])
        u = cumulative_trapezoid(-np.sign(eta) * gdot, eta, initial=0.0)
        return u, tau, gdot

    def mean_err(G: float) -> float:
        u, _, _ = profile(G)
        return float(np.trapezoid(u, eta) / height - u_mean)

    G_hi = 12.0 * MU_INF * u_mean / (height ** 2)
    while mean_err(G_hi) < 0.0:
        G_hi *= 2.0
    G = brentq(mean_err, 0.0, G_hi, rtol=1e-10)
    u, tau, gdot = profile(G)
    y = eta + height / 2.0
    return y, u, tau, gdot, G


def main() -> None:
    y, u, tau, gdot, G = solve_carreau_1d()
    tau_wall = G * H / 2.0
    u_max = float(u.max())
    in_win = (tau >= 30.0) & (tau <= 60.0)
    flux_win = float(np.trapezoid(u * in_win, y) / np.trapezoid(u, y))
    out = Path(__file__).resolve().parent / "carreau_1d_ref.npz"
    np.savez(out, y=y, u=u, tau=tau, gdot=gdot, G=G)
    print(f"G = -dp/dx   = {G:.4e} Pa/m")
    print(f"tau_wall     = {tau_wall:.2f} Pa  (Newtonian mu_inf: {6*MU_INF*U_MEAN/H:.2f} Pa)")
    print(f"u_max        = {u_max:.5f} m/s  (u_max/U_mean = {u_max/U_MEAN:.4f}; Newtonian = 1.5000)")
    print(f"flux in [30, 60] Pa = {flux_win:.2%}")
    print(f"saved reference -> {out}")


if __name__ == "__main__":
    main()
