"""Canonical metadata for the geometry families G1-G13 + KG.

``FAMILIES`` maps family id -> {name, topology, description, params}.  Each param is a
dict ``{name, label, default, min, max, int?, step?}`` used both by ``@register_geometry``
metadata and by the web bundle slider definitions (``scripts/build_web_bundle.py``).
Ranges are the validation bounds enforced by the builders in :mod:`.builders`.
"""
from __future__ import annotations

# Shared inlet/outlet length params (present for G2-G13; G1 uses L_ratio, KG uses L_out).
_LIN = {"name": "L_in", "label": "L_in (h1)", "default": 5.0, "min": 5.0, "max": 12.0, "step": 0.5}
_LOUT = {"name": "L_out_ratio", "label": "L_out (h1)", "default": 10.0, "min": 10.0, "max": 20.0, "step": 1.0}
_FILLET = {"name": "fillet_r", "label": "fillet r (h1)", "default": 0.08, "min": 0.0, "max": 0.15, "step": 0.01}


def _p(name, label, default, lo, hi, step=None, integer=False):
    d = {"name": name, "label": label, "default": default, "min": lo, "max": hi}
    if step is not None:
        d["step"] = step
    if integer:
        d["int"] = True
    return d


FAMILIES: dict[str, dict] = {
    "G1": {
        "name": "Straight microchannel",
        "topology": "1 -> 1",
        "description": "Planar straight channel, inlet y* in [0,1], walls y*=0/1, outlet +x.",
        "params": [_p("L_ratio", "L / h1", 20.0, 10.0, 30.0, 1.0)],
    },
    "G2": {
        "name": "Sudden symmetric expansion",
        "topology": "1 -> 1",
        "description": "Step expansion at x*=0 to height E; inlet y* in [0,1], outlet +x.",
        "params": [_p("E", "expansion E (h1)", 3.0, 1.5, 4.0, 0.1), _LIN, _LOUT],
    },
    "G3": {
        "name": "Stenosis (symmetric constriction)",
        "topology": "1 -> 1",
        "description": "Filleted stenosis throat_ratio over length L_t; inlet/outlet as G2.",
        "params": [_p("throat_ratio", "throat ratio", 0.5, 0.3, 0.8, 0.02), _p("L_t", "L_t (h1)", 2.0, 1.0, 5.0, 0.25),
                   _FILLET, _LIN, _LOUT],
    },
    "G4": {
        "name": "Symmetric Y-bifurcation",
        "topology": "1 -> 2",
        "description": "Symmetric Y-split by theta_deg, daughter width daughter_ratio, filleted carina.",
        "params": [_p("theta_deg", "branch angle (deg)", 45.0, 20.0, 90.0, 1.0),
                   _p("daughter_ratio", "daughter width ratio", 0.75, 0.5, 1.0, 0.05), _FILLET, _LIN, _LOUT],
    },
    "G5": {
        "name": "Asymmetric Y-bifurcation",
        "topology": "1 -> 2",
        "description": "Asymmetric Y-split: width_ratio splits the flow, theta_deg branch angle.",
        "params": [_p("width_ratio", "width ratio", 0.75, 0.5, 1.0, 0.05),
                   _p("theta_deg", "branch angle (deg)", 60.0, 30.0, 90.0, 1.0), _FILLET, _LIN, _LOUT],
    },
    "G6": {
        "name": "Serpentine U-bends",
        "topology": "1 -> 1",
        "description": "n_bends 180-deg arc U-bends of radius Rc_ratio with straight connectors.",
        "params": [_p("Rc_ratio", "bend radius Rc (h1)", 3.5, 1.5, 6.0, 0.25),
                   _p("n_bends", "n bends", 2, 1, 4, 1, integer=True),
                   _p("L_straight", "L_straight (h1)", 2.0, 0.5, 6.0, 0.5), _LIN, _LOUT],
    },
    "G7": {
        "name": "Expansion chamber + Y-split",
        "topology": "1 -> 2",
        "description": "Sudden expansion E then Y-bifurcation theta_deg after chamber length L_mid.",
        "params": [_p("E", "expansion E (h1)", 2.0, 1.5, 4.0, 0.1),
                   _p("theta_deg", "branch angle (deg)", 55.0, 20.0, 90.0, 1.0),
                   _p("L_mid", "L_mid (h1)", 4.0, 2.0, 8.0, 0.5), _LIN, _LOUT],
    },
    "G8": {
        "name": "Cosine Venturi nozzle",
        "topology": "1 -> 1",
        "description": "C1 cosine Venturi: converging L_conv to throat_ratio, diverging L_div.",
        "params": [_p("throat_ratio", "throat ratio", 0.7, 0.50, 0.85, 0.02),
                   _p("L_conv", "L_conv (h1)", 2.5, 1.5, 4.0, 0.25),
                   _p("L_div", "L_div (h1)", 5.0, 2.5, 7.0, 0.25), _LIN, _LOUT],
    },
    "G9": {
        "name": "Periodic pinch-relaxation stages",
        "topology": "1 -> 1",
        "description": "n_stages cosine pinch (throat_ratio) + relax (cavity_ratio) stages.",
        "params": [_p("throat_ratio", "throat ratio", 0.7, 0.55, 0.85, 0.02),
                   _p("cavity_ratio", "cavity ratio", 1.45, 1.15, 1.75, 0.05),
                   _p("n_stages", "n stages", 3, 2, 4, 1, integer=True),
                   _p("L_pinch", "L_pinch (h1)", 1.5, 1.0, 3.0, 0.25),
                   _p("L_cavity", "L_cavity (h1)", 2.0, 1.0, 4.0, 0.25), _LIN, _LOUT],
    },
    "G10": {
        "name": "Vortex pocket (filleted top)",
        "topology": "1 -> 1",
        "description": "Flat bottom y*=0 + filleted top recess pocket_ratio deep over L_pocket.",
        "params": [_p("pocket_ratio", "pocket ratio", 2.1, 1.4, 2.8, 0.1),
                   _p("L_pocket", "L_pocket (h1)", 4.0, 2.5, 7.0, 0.25),
                   _p("L_ramp", "L_ramp (h1)", 2.0, 1.0, 5.0, 0.25), _LIN, _LOUT],
    },
    "G11": {
        "name": "Bifurcation loop with island (genus 1)",
        "topology": "1 -> 2 -> 1",
        "description": "Two branches rejoining around a capsule island (wallsIsland, psi_wall=0); branch_bias sets the split.",
        "params": [_p("branch_bias", "branch bias", 0.42, 0.35, 0.50, 0.01),
                   _p("L_loop", "L_loop (h1)", 6.0, 4.0, 9.0, 0.25),
                   _p("theta_deg", "branch angle (deg)", 35.0, 20.0, 50.0, 1.0), _LIN, _LOUT],
    },
    "G12": {
        "name": "Sinusoidal meander",
        "topology": "1 -> 1",
        "description": "C1-tapered sinusoidal meander of constant width w*=1, amplitude amp_ratio, n_waves.",
        "params": [_p("amp_ratio", "amplitude ratio", 0.4, 0.15, 0.60, 0.02),
                   _p("wavelength", "wavelength (h1)", 5.0, 3.0, 7.0, 0.25),
                   _p("n_waves", "n waves", 3, 2, 4, 1, integer=True), _LIN, _LOUT],
    },
    "G13": {
        "name": "Trifurcation",
        "topology": "1 -> 3",
        "description": "Filleted trifurcation into centre (center_ratio) + two outer branches (outer_ratio) at +/- theta_deg.",
        "params": [_p("center_ratio", "centre width ratio", 0.7, 0.50, 0.90, 0.02),
                   _p("outer_ratio", "outer width ratio", 0.65, 0.50, 0.80, 0.02),
                   _p("theta_deg", "branch angle (deg)", 42.0, 25.0, 60.0, 1.0), _LIN, _LOUT],
    },
    "KG": {
        "name": "Krogh wedge (tapered tissue wedge)",
        "topology": "1 -> 1",
        "description": "Tapered wedge of half-angle wedge_angle_deg expanding to height E (Krogh cylinder model).",
        "params": [_p("E", "expansion E (h1)", 3.0, 1.5, 5.0, 0.1),
                   _p("L_in", "L_in (h1)", 6.0, 5.0, 12.0, 0.5),
                   _p("L_out", "L_out (h1)", 20.0, 10.0, 40.0, 1.0),
                   _p("wedge_angle_deg", "wedge angle (deg)", 2.0, 0.5, 5.0, 0.25)],
    },
}

# Flow / blood sliders shared by every family (ranges from the 1,139-run corpus BC tables).
FLOW_PARAMS = [
    {"name": "Re", "label": "Re (inlet)", "default": 10.0, "min": 0.73, "max": 59.1, "step": 0.1},
    {"name": "phi_in", "label": "phi_in (Hct)", "default": 0.30, "min": 0.0112, "max": 0.5967, "step": 0.005},
    {"name": "h_um", "label": "h1 (um)", "default": 110.0, "min": 80.0, "max": 149.8, "step": 1.0},
]

VARIANTS = {
    "V1": "V1: Newtonian (OpenFOAM simpleFoam)",
    "V2": "V2: Bird-Carreau shear-thinning (OpenFOAM)",
    "V4": "V4: Wu two-phase Euler-Euler + Saffman + two-wall lift + non-local Phillips (FEniCSx)",
}
