"""Interphase drag closures ``k_drag`` (dual NumPy+UFL), verbatim from ``twofluid.py``.

``k_drag`` multiplies the slip ``w - u`` to give the interphase force density (star units).
"""
from __future__ import annotations

from hemoflow.registry import register_drag
from . import _dual as D
from .constants import PHI_RES

__all__ = ["hindrance", "wu_rusche_issa", "schiller_naumann"]


def hindrance(phi):
    """Rusche & Issa hindrance used by Wu 2015: ``f = exp(2.68 phi) + phi^0.43``."""
    return D.exp(2.68 * phi) + phi**0.43


@register_drag("wu_rusche_issa")
def wu_rusche_issa(phi, d_hat: float):
    """Wu/Rusche-Issa drag: ``k = 18 phi_reg (exp(2.68 phi_pos) + phi_pos^0.43) / d_hat^2``.

    ``phi_reg = clip(phi, 0, 1)`` for the leading factor and ``phi_pos = max(phi, PHI_RES)``
    inside the hindrance (avoids the ``d(phi^0.43)/dphi -> inf`` NaN at phi = 0).
    """
    phi_reg = D.minimum(D.maximum(phi, 0.0), 1.0)
    phi_pos = D.maximum(phi, PHI_RES)
    return 18.0 * phi_reg * hindrance(phi_pos) / d_hat**2


@register_drag("schiller_naumann")
def schiller_naumann(phi, d_hat: float, re_p: float, slip_mag):
    """Schiller-Naumann (COMSOL solids): ``k = 18 phi_reg (1 + 0.15 Re_d^0.687) / d_hat^2``.

    ``Re_d = re_p * d_hat * |slip|`` (star units).
    """
    phi_reg = D.minimum(D.maximum(phi, 0.0), 1.0)
    re_d = re_p * d_hat * D.sqrt(slip_mag**2 + 1.0e-20)
    return 18.0 * phi_reg * (1.0 + 0.15 * re_d**0.687) / d_hat**2
