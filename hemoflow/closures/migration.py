"""Shear-induced migration flux closures (dual NumPy+UFL), verbatim from ``twofluid.py``.

The Phillips flux ``N`` moves RBC volume relative to the RBC-phase velocity:

``N = -a^2 [ K_c * jam_gate * phi_c^2 * grad(gdot) + (K_c phi_c gdot + K_mu phi_c^2 gdot dln(mu)/dphi) grad(phi) ]``

with ``a = d_hat/2``, ``phi_c = clip(phi, 0, PHI_JAM)``, and the smooth jam taper
``jam_gate = 0.5 (1 - tanh((phi_c - 0.63)/0.015))``.
"""
from __future__ import annotations

from hemoflow.registry import register_migration
from . import _dual as D
from .constants import GDOT_EPS, K_C, K_MU, PHI_JAM, PHILLIPS_NL_C

__all__ = ["none", "phillips_flux", "phillips_local", "phillips_nonlocal", "nonlocal_gdot"]


def _jam_gate(phi_c):
    return 0.5 * (1.0 - D.tanh((phi_c - 0.63) / 0.015))


def phillips_flux(phi, gdot, grad_gdot, grad_phi, dlnmu, a: float,
                  k_c: float = K_C, k_mu: float = K_MU, phi_jam: float = PHI_JAM):
    """Core Phillips migration flux ``N`` (dual algebra; gradients are passed in)."""
    phi_c = D.minimum(D.maximum(phi, 0.0), phi_jam)
    gate = _jam_gate(phi_c)
    return -a**2 * (
        k_c * gate * phi_c**2 * grad_gdot
        + (k_c * phi_c * gdot + k_mu * phi_c**2 * gdot * dlnmu) * grad_phi
    )


@register_migration("none")
def none(*args, **kwargs):
    """No migration."""
    return 0.0


@register_migration("phillips_local")
def phillips_local(phi, gdot, grad_gdot, grad_phi, dlnmu, a: float, **kw):
    """Phillips 1992 local shear-induced migration (K_c=0.41, K_mu=0.62, phi capped at 0.68)."""
    return phillips_flux(phi, gdot, grad_gdot, grad_phi, dlnmu, a)


def nonlocal_gdot(gdot_local, j, a: float, C: float = PHILLIPS_NL_C, eps: float = GDOT_EPS):
    """Non-local effective shear rate: ``sqrt(gdot_local^2 + (C a |j|)^2)`` (removes the
    centreline zero-shear singularity)."""
    return D.sqrt(gdot_local**2 + (C * a) ** 2 * D.dot(j, j) + eps**2)


@register_migration("phillips_nonlocal")
def phillips_nonlocal(phi, gdot, grad_gdot, grad_phi, dlnmu, a: float,
                      j=None, C: float = PHILLIPS_NL_C, **kw):
    """Default V4 Phillips migration with the non-local centreline floor ``(C a)^2 |j|^2``
    folded into ``gdot^2`` (C = 4).  Pass ``j`` (mixture velocity) to fold the floor here;
    otherwise ``gdot``/``grad_gdot`` are taken as already non-local."""
    if j is not None:
        gdot = nonlocal_gdot(gdot, j, a, C)
    return phillips_flux(phi, gdot, grad_gdot, grad_phi, dlnmu, a)
