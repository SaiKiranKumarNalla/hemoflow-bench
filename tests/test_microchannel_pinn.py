"""Tests for the parameterized microchannel PINN."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from pinn.microchannel import L, MU, UMAX, p_scale, pde_residuals, ParamPINN


def test_analytic_poiseuille_satisfies_stokes() -> None:
    """Harness check: the ansatz with (Nu, Nv, Np) = (0, 0, 1) is exact Poiseuille -> zero residual."""
    xi = torch.linspace(0, 1, 20).unsqueeze(1)
    eta = torch.linspace(0, 1, 20).unsqueeze(1)
    XI, ETA = torch.meshgrid(xi.squeeze(1), eta.squeeze(1), indexing="ij")
    xi = XI.reshape(-1, 1)
    eta = ETA.reshape(-1, 1)
    hh = torch.full((xi.shape[0], 1), 0.1)

    class Exact:
        def __call__(self, xi, eta, h):
            # (Nu, Nv, Np) = (0, 0, 1); keep graph connections to all inputs
            return xi * 0.0, eta * 0.0, 1.0 + xi * 0.0 + eta * 0.0 + h * 0.0

    _, (r1, r2, r3) = pde_residuals(Exact(), xi, eta, hh)
    assert torch.allclose(r1, torch.zeros_like(r1), atol=1e-8)
    assert torch.allclose(r2, torch.zeros_like(r2), atol=1e-8)
    assert torch.allclose(r3, torch.zeros_like(r3), atol=1e-8)


def test_trained_pinn_matches_poiseuille() -> None:
    """End-to-end: trained PINN results (if present) are within 5% of analytic."""
    res = Path(__file__).resolve().parents[1] / "results" / "pinn" / "results.json"
    assert res.exists(), "run pinn/run_microchannel.py first"
    data = json.loads(res.read_text())
    for key, r in data["per_h"].items():
        assert r["rel_L2_u"] < 0.05, f"{key}: rel_L2_u={r['rel_L2_u']}"
