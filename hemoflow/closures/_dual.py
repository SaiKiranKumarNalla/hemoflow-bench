"""Dual-mode math helpers: every closure in :mod:`hemoflow.closures` is written once with
these operators and therefore evaluates identically on NumPy arrays (host inspection /
unit tests) and on UFL expressions (FEniCSx assembly).

A UFL expression is detected via ``isinstance(x, ufl.Expr)``; when ``ufl`` is not installed
(or no argument is a UFL expression) the NumPy branch is used.  ``dolfinx``/``ufl`` are only
ever imported lazily inside these helpers, so importing this module is host-safe.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ufl_mode", "is_ufl_expr", "exp", "sqrt", "ln", "tanh", "abs_", "pow_",
    "minimum", "maximum", "dot", "inner", "matvec", "outer",
]


def ufl_mode(*args):
    """Return the ``ufl`` module if any argument is a UFL expression, else ``None``."""
    try:
        import ufl
    except Exception:
        return None
    for a in args:
        if isinstance(a, ufl.Expr):
            return ufl
    return None


def is_ufl_expr(x) -> bool:
    return ufl_mode(x) is not None


def exp(x):
    u = ufl_mode(x)
    return u.exp(x) if u else np.exp(x)


def sqrt(x):
    u = ufl_mode(x)
    return u.sqrt(x) if u else np.sqrt(x)


def ln(x):
    u = ufl_mode(x)
    return u.ln(x) if u else np.log(x)


def tanh(x):
    u = ufl_mode(x)
    return u.tanh(x) if u else np.tanh(x)


def abs_(x):
    u = ufl_mode(x)
    return u.abs(x) if u else np.abs(x)


def pow_(x, p):
    # ``**`` works natively on both numpy arrays and ufl expressions
    return x ** p


def minimum(a, b):
    u = ufl_mode(a, b)
    return u.min_value(a, b) if u else np.minimum(a, b)


def maximum(a, b):
    u = ufl_mode(a, b)
    return u.max_value(a, b) if u else np.maximum(a, b)


def dot(a, b):
    """Vector dot product (last axis)."""
    u = ufl_mode(a, b)
    return u.dot(a, b) if u else np.sum(a * b, axis=-1)


def inner(a, b):
    """Tensor inner product (last two axes)."""
    u = ufl_mode(a, b)
    return u.inner(a, b) if u else np.sum(a * b, axis=(-2, -1))


def matvec(D, v):
    """Matrix-vector product ``D @ v`` (last axis of v)."""
    u = ufl_mode(D, v)
    return u.dot(D, v) if u else np.einsum("...ij,...j->...i", D, v)


def outer(a, b):
    u = ufl_mode(a, b)
    return u.outer(a, b) if u else np.einsum("...i,...j->...ij", a, b)
