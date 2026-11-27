"""Lift closures (dual NumPy+UFL), verbatim from ``twofluid.py``.

Each lift closure returns an **interphase force vector** (added to the RBC momentum,
reaction on the plasma).  They take primitive fields (already differentiated where
needed) so they are pure algebra and evaluate identically on NumPy arrays and UFL
expressions:

- ``saffman``: needs ``gdot`` and ``D_u`` (symmetric gradient of plasma velocity) and ``slip``.
- ``abkarian_wall`` / ``abkarian_twowall``: need ``gdot``, ``grad_psi`` and ``psi``
  (the Spalding wall-distance potential, ``-lap(psi)=1``, ``psi=0`` on walls).
"""
from __future__ import annotations

from hemoflow.registry import register_lift
from . import _dual as D
from .constants import SAFFMAN_C, WALL_LIFT_C

__all__ = ["none", "saffman", "abkarian_wall", "abkarian_twowall", "saffman_plus_twowall", "wall_law"]

_WALL_LAWS = ("abk", "abk_capR", "abk_yR", "olla")


def wall_law(d_wall, r_hat: float, law: str = "abk", farfield: bool = False):
    """Wall-lift radial law ``f(d)`` (Abkarian & Viallat 2005 Eq. 4 family), verbatim twofluid.

    ``law`` in ``{"abk", "abk_capR", "abk_yR", "olla"}``.
    """
    if law == "olla":
        yw = D.maximum(d_wall, r_hat)
        return r_hat / (yw * yw)
    if law == "abk_yR":
        return 1.0 / D.maximum(d_wall, r_hat)
    if law == "abk_capR":
        yg = D.maximum(d_wall - r_hat, r_hat)
        return 1.0 / yg
    # "abk": 0.1 R floor
    yg = D.maximum(d_wall - r_hat, 0.1 * r_hat)
    m = 1.0 / yg
    return m * D.minimum(1.0, r_hat / yg) if farfield else m


def _wall_geom(grad_psi, psi):
    """Spalding nearest-wall distance pieces: ``gnorm``, ``s_psi``, ``d_1``, ``d_2``."""
    gnorm = D.sqrt(D.dot(grad_psi, grad_psi) + 1.0e-12)
    s_psi = D.sqrt(D.dot(grad_psi, grad_psi) + 2.0 * D.maximum(psi, 0.0) + 1.0e-16)
    d1 = s_psi - gnorm
    d2 = s_psi + gnorm
    return gnorm, d1, d2


@register_lift("none")
def none(*args, **kwargs):
    """No lift."""
    return 0.0


@register_lift("saffman")
def saffman(phi, re: float, d_hat: float, gdot, D_u, slip):
    """Saffman lift: ``SAFFMAN_C phi sqrt(Re)/d_hat gdot^-0.5 D(u) . slip``."""
    phi_reg = D.minimum(D.maximum(phi, 0.0), 1.0)
    return SAFFMAN_C * phi_reg * D.sqrt(re) / d_hat * gdot**-0.5 * D.matvec(D_u, slip)


@register_lift("abkarian_wall")
def abkarian_wall(phi, gdot, grad_psi, psi, wall_lift_C: float = WALL_LIFT_C,
                  d_hat: float = 1.0, law: str = "abk", farfield: bool = False):
    """Single-wall Abkarian lift: ``-C phi gdot f(d_1) grad(psi)/|grad(psi)|``."""
    phi_reg = D.minimum(D.maximum(phi, 0.0), 1.0)
    gnorm, d1, _ = _wall_geom(grad_psi, psi)
    r_hat = 0.5 * d_hat
    mag = wall_lift_C * phi_reg * gdot * wall_law(d1, r_hat, law, farfield)
    return -mag * grad_psi / gnorm


@register_lift("abkarian_twowall")
def abkarian_twowall(phi, gdot, grad_psi, psi, wall_lift_C: float = WALL_LIFT_C,
                     d_hat: float = 1.0, law: str = "abk", farfield: bool = False):
    """Two-wall opposing lift: ``-C phi gdot (f(d_1) - f(d_2)) grad(psi)/|grad(psi)|`` (vanishes at centreline)."""
    phi_reg = D.minimum(D.maximum(phi, 0.0), 1.0)
    gnorm, d1, d2 = _wall_geom(grad_psi, psi)
    r_hat = 0.5 * d_hat
    mag = wall_lift_C * phi_reg * gdot * (wall_law(d1, r_hat, law, farfield) - wall_law(d2, r_hat, law, farfield))
    return -mag * grad_psi / gnorm


@register_lift("saffman_plus_twowall")
def saffman_plus_twowall(phi, re: float, d_hat: float, gdot, D_u, slip, grad_psi, psi,
                         wall_lift_C: float = WALL_LIFT_C, law: str = "abk", farfield: bool = False):
    """Default V4 full lift: Saffman + two-wall Abkarian."""
    return (saffman(phi, re, d_hat, gdot, D_u, slip)
            + abkarian_twowall(phi, gdot, grad_psi, psi, wall_lift_C, d_hat, law, farfield))
