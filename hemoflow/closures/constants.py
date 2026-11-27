"""Verified physical constants and model constants, ported verbatim (release-guard: 2D/2.5D)
from ``cfd/fenicsx/twofluid.py``, ``cfd/fenicsx/comsol_solver.py``, ``cfd/gen/m2b.py`` and
``campaign/blood_rheology.py``.  No constant here is invented.
"""
from __future__ import annotations

import numpy as np

# --- Wu et al. 2015 (PMC4521229, verified-primary) ---
MU_PLASMA_CP = 0.96                       # plasma viscosity, cP
MU_PLASMA = 0.96e-3                       # plasma viscosity, Pa.s (knowledge/16 §4, verified)
RHO_RATIO = 1093.0 / 1027.0               # rho_RBC / rho_plasma (Wu 2015 §3)
WU2015_RHO_PLASMA = 1027.0
WU2015_RHO_RBC = 1093.0
WU2015_DP_RBC = 8.0e-6                    # 8 um

# --- Lift / migration / stabilisation constants (twofluid.py) ---
SAFFMAN_C = 3.0 * 6.46 / (4.0 * np.pi)    # Saffman lift coefficient
WALL_LIFT_C = 3.0 * 2.0 / (4.0 * np.pi)   # Abkarian 3 f(nu)/(4 pi), f=2 for RBC nu=0.7
K_C, K_MU = 0.41, 0.62                    # Phillips et al. 1992
PHI_RES = 1.0e-4
PHI_MAX = 0.95                            # cap for the Phillips core
PHI_JAM = 0.68                            # jam margin for the Phillips flux cap
GDOT_EPS = 1.0e-3                         # shear-rate floor
CROSSWIND_C = 0.7                         # Codina-type shock-capturing constant
SUPG_W_FLOOR = 0.1                        # star units (mean velocity = 1)
PHILLIPS_NL_C = 4.0                       # non-local shear-rate floor constant C

# --- COMSOL 2023 (knowledge/13-2023-comsol-model.md, slide 6) ---
COMSOL2023_RHO_PLASMA = 1038.0
COMSOL2023_RHO_RBC = 1098.0
COMSOL2023_MU_PLASMA = 4.0e-3
COMSOL2023_DP_RBC = 8.0e-6
COMSOL2023_RHO_RATIO = COMSOL2023_RHO_RBC / COMSOL2023_RHO_PLASMA

# --- Quemada coefficients (Mimouni 2016 §3.1.3, verified; knowledge/16 §2) ---
QUEMADA_K0 = 4.33
QUEMADA_K_INF = 2.07
QUEMADA_GDOT_C = 1.88                     # 1/s

# --- Carreau time constant (Jung et al. via Sezer 2026 Eqs. 5-11, verified) ---
LAMBDA_BLOOD = 0.110                      # s
CARREAU_A = 2.0

# --- Carrier buffer range (Bernardelli 2025 Table 3, verified) ---
MU_BUFFER_RANGE = (1.1e-3, 1.7e-3)
BUFFER_MAX_PHI = 0.1

# --- Boundary facet tags (mesh_from_spec / twofluid) ---
INLET, OUTLET, WALL, AXIS = 1, 2, 3, 4
