"""Solver configuration dataclasses for :mod:`hemoflow.solver`.

Closure fields (``viscosity``, ``drag``, ``lift``, ``migration``) accept **either** a
registered plugin name (``str``) **or** a direct custom callable / class instance, so
users can plug in/out closures effortlessly.  ``__post_init__`` validates that string
names exist in :mod:`hemoflow.registry` (callables are used as-is).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union

from hemoflow.registry import get_plugin, KINDS
from hemoflow.closures.constants import (
    PHILLIPS_NL_C, PHI_JAM, SUPG_W_FLOOR, WALL_LIFT_C, CROSSWIND_C, GDOT_EPS,
)

Closure = Union[str, Callable, Any]

_FORMS = ("comsol", "wu-phase")


def _check_closure(kind: str, value: Closure) -> Closure:
    if isinstance(value, str):
        get_plugin(kind, value)  # raises PluginNotFoundError if unknown
    elif not (callable(value) or hasattr(value, "__call__")):
        raise TypeError(f"{kind} closure must be a registered name (str) or callable, got {type(value).__name__}")
    return value


@dataclass
class EulerEulerConfig:
    """Two-phase Euler-Euler model configuration (star units: h1=1, U_mean=1, mu_ref=mu_p).

    Closure fields are string-or-callable.  Scalar knobs mirror ``TwoFluidParams``.
    """
    re_p: float
    d_hat: float
    phi_in: float
    viscosity: Closure = "wu_high_shear"
    drag: Closure = "wu_rusche_issa"
    lift: Closure = "saffman_plus_twowall"
    migration: Closure = "phillips_nonlocal"
    form: str = "wu-phase"            # "wu-phase" (Wu 2015 Eqs. 12-13) or "comsol" (shared mu)
    closure: str = "wu"               # legacy combined selector: "wu" | "comsol-solid"
    k_star: float = 0.0               # Wu k in star units (needed for wu_shear_thinning)
    phi_max: float = 2.0 / 2.07       # Krieger packing limit 2/k_inf (Quemada high-shear)
    phi_jam: float = PHI_JAM
    phi_diff: float = 1.0e-4
    phi_neg_diff: float = 3.0e-3
    supg: bool = True
    crosswind_C: float = CROSSWIND_C
    supg_w_floor: float = SUPG_W_FLOOR
    gdot_eps: float = GDOT_EPS
    phillips_nl_c: float = PHILLIPS_NL_C
    wall_lift_C: float = WALL_LIFT_C
    wall_lift_law: str = "abk"        # "abk" | "abk_capR" | "abk_yR" | "olla"
    wall_lift_farfield: bool = False
    depth_hat: float = 0.0            # 2.5D Hele-Shaw out-of-plane depth (0 = 2D planar)
    gravity: float = 0.0
    qdeg: int = 4
    axisym: bool = False
    lag_phillips_stab: bool = True
    lift_ramp_stages: int = 0

    def __post_init__(self):
        if self.re_p <= 0:
            raise ValueError(f"re_p must be > 0, got {self.re_p}")
        if self.d_hat <= 0:
            raise ValueError(f"d_hat must be > 0, got {self.d_hat}")
        if not (0.0 < self.phi_in < 1.0):
            raise ValueError(f"phi_in must be in (0, 1), got {self.phi_in}")
        if self.form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, got {self.form!r}")
        if self.closure not in ("wu", "comsol-solid"):
            raise ValueError(f"closure must be 'wu' or 'comsol-solid', got {self.closure!r}")
        if self.wall_lift_law not in ("abk", "abk_capR", "abk_yR", "olla"):
            raise ValueError(f"wall_lift_law invalid: {self.wall_lift_law!r}")
        self.viscosity = _check_closure("viscosity", self.viscosity)
        self.drag = _check_closure("drag", self.drag)
        self.lift = _check_closure("lift", self.lift)
        self.migration = _check_closure("migration", self.migration)

    def resolve(self) -> dict[str, Callable]:
        """Return the four closures as callables (registered name -> plugin, or the callable itself)."""
        return {
            "viscosity": get_plugin("viscosity", self.viscosity),
            "drag": get_plugin("drag", self.drag),
            "lift": get_plugin("lift", self.lift),
            "migration": get_plugin("migration", self.migration),
        }

    # ---- presets ----
    @classmethod
    def v4_wu(cls, re_p: float, d_hat: float, phi_in: float, k_star: float = 0.0) -> "EulerEulerConfig":
        """V4 corpus default: Wu two-phase (wu-phase form) + Saffman + two-wall Abkarian lift +
        non-local Phillips, shear-thinning RBC phase, SUPG+crosswind."""
        return cls(re_p=re_p, d_hat=d_hat, phi_in=phi_in, k_star=k_star,
                   viscosity="wu_shear_thinning", drag="wu_rusche_issa",
                   lift="saffman_plus_twowall", migration="phillips_nonlocal",
                   form="wu-phase", closure="wu", supg=True)

    @classmethod
    def v2_carreau(cls, re_p: float, d_hat: float, phi_in: float, lam: float = 0.110,
                   n: float = 0.6, a: float = 2.0) -> "EulerEulerConfig":
        """V2-style single-phase Bird-Carreau shear-thinning limit (no drag/lift/migration)."""
        def _carreau(phi, gdot, _lam=lam, _n=n, _a=a):
            from hemoflow.closures.viscosity import bird_carreau
            return bird_carreau(gdot, mu_0=5.0, mu_inf=1.0, lam=_lam, n=_n, a=_a)
        return cls(re_p=re_p, d_hat=d_hat, phi_in=phi_in, viscosity=_carreau,
                   drag="none", lift="none", migration="none", form="comsol", closure="wu", supg=False)

    @classmethod
    def v1_newtonian(cls, re_p: float, d_hat: float, phi_in: float) -> "EulerEulerConfig":
        """V1-style Newtonian single-phase limit (mu_hat = 1)."""
        return cls(re_p=re_p, d_hat=d_hat, phi_in=phi_in, viscosity="newtonian",
                   drag="none", lift="none", migration="none", form="comsol", closure="wu", supg=False)

    @classmethod
    def comsol_solid(cls, re_p: float, d_hat: float, phi_in: float, phi_max: float = 0.62) -> "EulerEulerConfig":
        """COMSOL 2023 solid-particles setup: Krieger mixture viscosity + Schiller-Naumann drag."""
        return cls(re_p=re_p, d_hat=d_hat, phi_in=phi_in, viscosity="krieger",
                   drag="schiller_naumann", lift="none", migration="none",
                   form="comsol", closure="comsol-solid", phi_max=phi_max, supg=False)


@dataclass
class BoundaryConfig:
    """Boundary-condition spec.  ``inlet_profile`` is a name ("poiseuille"/"brinkman") or callable(y)."""
    inlet_profile: Union[str, Callable] = "poiseuille"
    inlet_height: float = 1.0
    phi_in_dirichlet: bool = True
    wall_no_slip: bool = True
    outlet_zero_gradient: bool = True
    pin_outlet_p: bool = False
    brinkman_depth_hat: float = 0.0

    def __post_init__(self):
        if isinstance(self.inlet_profile, str) and self.inlet_profile not in ("poiseuille", "brinkman"):
            raise ValueError("inlet_profile must be 'poiseuille', 'brinkman', or a callable")

    def inlet_function(self):
        if callable(self.inlet_profile):
            return self.inlet_profile
        if self.inlet_profile == "brinkman":
            from hemoflow.solver.analytic import brinkman_inlet
            return lambda y: brinkman_inlet(y, 0.5, self.brinkman_depth_hat)
        from hemoflow.solver.analytic import poiseuille
        return poiseuille


@dataclass
class SolverConfig:
    """COMSOL-style stationary solver settings (see ``comsol_solver.py``)."""
    cfl_const: float = 1.0          # initial pseudo-CFL constant (COMSOL schedule multiplies in)
    cfl_scale: float = 1.0
    lam_min: float = 1.0e-4         # minimum line-search damping
    phi_cap: float = PHI_JAM        # trust-region clip of the phi iterate
    trust_region: float = 0.25      # cap on relative Newton update |dU|/|U|
    max_outer: int = 200            # max nonlinear iterations
    rtol: float = 1.0e-9
    atol: float = 1.0e-11
    tol_E: float = 1.0e-6           # COMSOL scaled-error convergence target
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    mat_solver: str = "mumps"

    def __post_init__(self):
        if self.cfl_scale <= 0:
            raise ValueError("cfl_scale must be > 0")
        if not (0.0 < self.trust_region <= 1.0):
            raise ValueError("trust_region must be in (0, 1]")
