"""Plug-in registry for :mod:`hemoflow`.

Closures and geometries can be referenced two ways, interchangeably:

1. by **registered name** (``str``), e.g. ``EulerEulerConfig(viscosity="wu_high_shear")``;
2. by **direct callable / class instance**, e.g. ``EulerEulerConfig(viscosity=my_mu_fn)``.

Use the ``@register_<kind>("name")`` decorators (or ``register_plugin(kind, name, obj)``)
to add your own closure to the global registry, and ``get_plugin(kind, name_or_obj)`` to
resolve either form to the callable.  ``list_plugins(kind=None)`` lists the registry.

Registry kinds: ``"viscosity"``, ``"drag"``, ``"lift"``, ``"migration"``, ``"geometry"``,
``"bc"``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

KINDS = ("viscosity", "drag", "lift", "migration", "geometry", "bc")

_REGISTRY: dict[str, dict[str, Any]] = {k: {} for k in KINDS}


class PluginNotFoundError(KeyError):
    """Raised when a plugin name is not registered in its kind."""


def register_plugin(kind: str, name: str, obj: Any) -> Any:
    """Register ``obj`` under ``name`` in ``kind``. Returns ``obj`` (decorator-friendly)."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")
    if not isinstance(name, str) or not name:
        raise ValueError("plugin name must be a non-empty string")
    _REGISTRY[kind][name] = obj
    return obj


def register_viscosity(name: str) -> Callable:
    return lambda obj: register_plugin("viscosity", name, obj)


def register_drag(name: str) -> Callable:
    return lambda obj: register_plugin("drag", name, obj)


def register_lift(name: str) -> Callable:
    return lambda obj: register_plugin("lift", name, obj)


def register_migration(name: str) -> Callable:
    return lambda obj: register_plugin("migration", name, obj)


def register_geometry(key: str, **metadata: Any) -> Callable:
    """Register a geometry family builder under ``key``; ``metadata`` (name/topology/params) is stored."""
    def deco(obj: Any) -> Any:
        register_plugin("geometry", key, obj)
        if metadata:
            _GEOMETRY_META[key] = dict(metadata)
        return obj
    return deco


def register_bc(name: str) -> Callable:
    return lambda obj: register_plugin("bc", name, obj)


_GEOMETRY_META: dict[str, dict[str, Any]] = {}


def get_plugin(kind: str, name_or_obj: Any) -> Any:
    """Resolve a registered name (``str``) or a direct callable/instance to the object."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")
    if isinstance(name_or_obj, str):
        try:
            return _REGISTRY[kind][name_or_obj]
        except KeyError as exc:
            raise PluginNotFoundError(
                f"no {kind} plugin named {name_or_obj!r}; registered: {sorted(_REGISTRY[kind])}"
            ) from exc
    if callable(name_or_obj) or hasattr(name_or_obj, "__call__"):
        return name_or_obj
    raise TypeError(
        f"{kind} plugin must be a registered name (str) or a callable/instance, "
        f"got {type(name_or_obj).__name__}"
    )


def list_plugins(kind: str | None = None) -> dict[str, list[str]]:
    """List registered plugin names, optionally filtered by ``kind``."""
    if kind is not None:
        if kind not in _REGISTRY:
            raise ValueError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")
        return {kind: sorted(_REGISTRY[kind])}
    return {k: sorted(v) for k, v in _REGISTRY.items() if v}


def geometry_metadata(name: str) -> dict[str, Any]:
    """Return the stored metadata (topology, params, ranges) for a geometry family."""
    return dict(_GEOMETRY_META.get(name.upper(), {}))
