"""Parameterized PINN for steady laminar microchannel flow (Stokes regime).

Channel x in [0, L], y in [0, H]; normalized coords xi = x/L in [0,1], eta = y/H in [0,1].
The network maps (xi, eta, h) -> (Nu, Nv, Np) with h = H/L the aspect-ratio parameter, so a
single trained network covers a whole family of channel geometries (cf. Gammaidoni et al.
2026, Computers & Fluids: parameterized microchannel PINN).

Hard BC ansatz (boundary conditions satisfied by construction):
    u = 4 Umax eta(1-eta) + eta(1-eta) xi(1-xi) Nu     -> parabolic inlet, no-slip walls
    v = eta(1-eta) xi(1-xi) Nv                           -> v = 0 on all boundaries
    p = p_scale(h) (1-xi) Np                             -> p = 0 at outlet; inlet pressure
                                                           p_in = p_scale(h) Np is LEARNED
Only the PDE residual loss remains, which is what makes training stable.

Governing equations (steady Stokes, constant viscosity mu, Re << 1 — internal laminar flow),
in normalized coords multiplied through by h^2 / h so every aspect ratio contributes O(1):
    r1 = -mu (h^2 u_xixi + u_etaeta) + p_xi L h^2 = 0
    r2 = -mu (h^2 v_xixi + v_etaeta) + p_eta L h   = 0
    r3 = h u_xi + v_eta                            = 0

Validation target (exact here because the inlet BC is the fully developed profile):
Poiseuille u = 4 Umax eta(1-eta), v = 0, p = G (L - x) with G = 8 mu Umax / H^2,
i.e. p_scale(h) = G L = 8 mu Umax / (h^2 L), p_hat = 1 - xi.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

L = 10.0          # channel length
MU = 1.0          # dynamic viscosity
UMAX = 1.0        # inlet peak velocity
H_MIN, H_MAX = 0.05, 0.2   # aspect-ratio parameter range h = H/L


def p_scale(h: torch.Tensor) -> torch.Tensor:
    """Pressure scale G*L = 8 mu Umax / (h^2 L)."""
    return 8.0 * MU * UMAX / (h * h * L)


class ParamPINN(nn.Module):
    """Outputs the correction/pressure-scaled fields (Nu, Nv, Np)."""

    def __init__(self, hidden: int = 96, n_hidden: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, xi, eta, h):
        out = self.net(torch.cat([xi, eta, h], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def solution(model, xi, eta, h):
    """Apply the hard-BC ansatz: returns physical (u, v, p)."""
    Nu, Nv, Np = model(xi, eta, h)
    u = 4.0 * UMAX * eta * (1.0 - eta) + eta * (1.0 - eta) * xi * (1.0 - xi) * Nu
    v = eta * (1.0 - eta) * xi * (1.0 - xi) * Nv
    p = p_scale(h) * (1.0 - xi) * Np
    return u, v, p


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def pde_residuals(model, xi, eta, h):
    xi = xi.requires_grad_(True)
    eta = eta.requires_grad_(True)
    u, v, p = solution(model, xi, eta, h)
    u_xi, u_eta = _grad(u, xi), _grad(u, eta)
    v_xi, v_eta = _grad(v, xi), _grad(v, eta)
    p_xi, p_eta = _grad(p, xi), _grad(p, eta)
    u_xixi, u_etaeta = _grad(u_xi, xi), _grad(u_eta, eta)
    v_xixi, v_etaeta = _grad(v_xi, xi), _grad(v_eta, eta)
    r1 = -MU * (h**2 * u_xixi + u_etaeta) + p_xi * L * h**2
    r2 = -MU * (h**2 * v_xixi + v_etaeta) + p_eta * L * h
    r3 = h * u_xi + v_eta
    return (u, v, p), (r1, r2, r3)


def sample_batch(n, device, gen):
    xi = torch.rand(n, 1, generator=gen, device=device)
    eta = torch.rand(n, 1, generator=gen, device=device)
    # log-uniform aspect ratio: balances thin vs wide channels in the loss
    log_h = torch.log(torch.tensor(H_MIN)) + (torch.log(torch.tensor(H_MAX)) - torch.log(torch.tensor(H_MIN))) * torch.rand(n, 1, generator=gen, device=device)
    h = torch.exp(log_h)
    return xi, eta, h


def compute_loss(model, gen, device, n_int=2048):
    xi, eta, h = sample_batch(n_int, device, gen)
    _, (r1, r2, r3) = pde_residuals(model, xi, eta, h)
    loss_pde = (r1**2 + r2**2 + r3**2).mean()
    return loss_pde, loss_pde.detach()


def train(epochs=6000, lr=1e-3, seed=0, lbfgs_steps=1500, log_every=500, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    model = ParamPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        loss, loss_pde = compute_loss(model, gen, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if epoch == 1 or epoch % log_every == 0:
            history.append((epoch, float(loss_pde)))
            print(f"epoch {epoch:5d}  pde {float(loss_pde):.3e}", flush=True)
    if lbfgs_steps > 0:
        opt2 = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=lbfgs_steps,
                                 history_size=100, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss, _ = compute_loss(model, gen, device)
            loss.backward()
            return loss

        opt2.step(closure)
        _, loss_pde = compute_loss(model, gen, device)
        print(f"lbfgs done  pde {float(loss_pde):.3e}", flush=True)
        history.append(("lbfgs", float(loss_pde)))
    return model, history


@torch.no_grad()
def validate(model, hs, n_xi=200, n_eta=50, device="cpu"):
    """Compare PINN vs analytic Poiseuille for each aspect ratio h."""
    xig = torch.linspace(0, 1, n_xi, device=device)
    etag = torch.linspace(0, 1, n_eta, device=device)
    XI, ETA = torch.meshgrid(xig, etag, indexing="ij")
    xi = XI.reshape(-1, 1)
    eta = ETA.reshape(-1, 1)
    results = {}
    for h in hs:
        hv = torch.full((xi.shape[0], 1), h, device=device)
        u, v, p = solution(model, xi, eta, hv)
        u_a = 4.0 * UMAX * eta * (1.0 - eta)
        p_a = p_scale(hv) * (1.0 - xi)          # physical pressure
        rel_u = torch.sqrt(((u - u_a) ** 2).mean() / (u_a**2).mean()).item()
        rel_v = torch.sqrt((v**2).mean()).item()
        p_in = float(p[xi.flatten() == 0].mean())
        results[f"h={h:.3f}"] = {
            "H": h * L,
            "rel_L2_u": rel_u,
            "rel_L2_v": rel_v,
            "p_in_pred": p_in,
            "p_in_exact": float(p_scale(torch.tensor([[h]], device=device))),
        }
    return results


def plot(model, h, out_png, n_xi=100, n_eta=40, device="cpu"):
    xig = torch.linspace(0, 1, n_xi, device=device)
    etag = torch.linspace(0, 1, n_eta, device=device)
    XI, ETA = torch.meshgrid(xig, etag, indexing="ij")
    xi = XI.reshape(-1, 1)
    eta = ETA.reshape(-1, 1)
    hv = torch.full((xi.shape[0], 1), h, device=device)
    with torch.no_grad():
        u, v, p = solution(model, xi, eta, hv)
    u_a = (4.0 * UMAX * eta * (1.0 - eta)).reshape(n_xi, n_eta).cpu().numpy()
    u_p = u.reshape(n_xi, n_eta).cpu().numpy()
    err = np.abs(u_p - u_a)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(axes, [u_p, u_a, err], ["PINN u", "analytic u", "|error|"]):
        im = ax.imshow(field.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        ax.set_title(f"{title} (h={h:.2f})")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "pinn"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, history = train()
    hs = [0.05, 0.10, 0.15, 0.20]
    results = validate(model, hs)
    plot(model, 0.10, out_dir / "microchannel_pinn.png")
    torch.save(model.state_dict(), out_dir / "model.pt")
    payload = {
        "problem": "steady Stokes microchannel, parameterized by aspect ratio h=H/L",
        "method": "parameterized PINN, hard-BC ansatz, PDE-residual loss only, h-balanced residuals",
        "validation": "analytic Poiseuille (exact for parabolic inlet BC)",
        "config": {"L": L, "mu": MU, "Umax": UMAX, "h_range": [H_MIN, H_MAX],
                   "epochs": 6000, "lbfgs_steps": 1500, "net": "96x5 tanh", "h_sampling": "log-uniform"},
        "per_h": results,
        "loss_history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
