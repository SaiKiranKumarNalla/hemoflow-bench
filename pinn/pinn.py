"""Physics-only PINN for the steady incompressible problem of task T9 (step 2).

The network maps ``(x, y) -> (u, v, p)`` and is trained on the boundary-value problem only:
the steady Navier-Stokes residual in the interior, continuity, the inlet profile, no-slip on
the walls and ``p = 0`` at the outlet. No CFD field enters the loss — the reference solution
is read only by ``pinn/evaluate.py``.

Residual convention (star units, derived and verified in :mod:`pinn.problem`):

.. code-block:: none

    r_x = u u_x + v u_y + (p_x - (u_xx + u_yy)) / Re
    r_y = u v_x + v v_y + (p_y - (v_xx + v_yy)) / Re
    r_c = u_x + v_y

The ``1/Re`` multiplies the pressure gradient and the Laplacian but not the advection term,
because the corpus stores the viscous-scaled pressure ``p* = Re * p_OF``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pinn.problem import (
    BC_INLET,
    BC_OUTLET,
    BC_WALL,
    Problem,
    build_problem,
    sample_problem,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "pinn_results"

# --- architecture ---------------------------------------------------------
HIDDEN_WIDTH = 64
HIDDEN_LAYERS = 4
ACTIVATION = nn.Tanh

# --- optimiser ------------------------------------------------------------
ADAM_STEPS = 4000
ADAM_LEARNING_RATE = 1.0e-3
ADAM_FINAL_LR_FRACTION = 0.02
LBFGS_MAX_ITER = 1500
LBFGS_LEARNING_RATE = 1.0
SEED = 0
LOG_EVERY = 200

# --- collocation / boundary sampling (fixed sets, sampled once per run) ---
N_INTERIOR = 6144
# Chebyshev clustering of the collocation points in the wall-normal direction: a uniform
# sample cannot see the O(1e-3) wall layer in which an early attempt hid a mass-flux leak.
WALL_CLUSTERING = True
N_INLET = 256
N_WALL = 1024
N_OUTLET = 256

# --- loss weights ---------------------------------------------------------
# Scale argument: in these star units the momentum residual has natural size 12/Re (the
# plane-Poiseuille second derivative is -12) while the continuity residual has size 1, so unit
# weights already over-trust continuity. A first run with WEIGHT_CONTINUITY = 1 produced a
# field that satisfied momentum to 0.3 % of its natural scale but lost 86 % of the inlet mass
# flux along the inlet channel, so continuity is weighted 100x to force global mass balance.
# The boundary weights set the amplitude of the solution: the momentum loss grows with the
# square of the flow rate, so a modest BC weight lets the optimiser settle on a self-similar
# slower flow (runs 2-3 lost 80 % of the inlet mass flux and kept 0.09 wall slip). The BC
# weights must dominate the interior losses for the inlet profile to fix the flow rate.
WEIGHT_MOMENTUM = 1.0
WEIGHT_CONTINUITY = 100.0
WEIGHT_INLET = 1000.0
WEIGHT_WALL = 1000.0
WEIGHT_OUTLET_PRESSURE = 10.0


def pressure_scale(problem: Problem) -> float:
    """Plane-Poiseuille estimate of the inlet-to-outlet pressure drop in star units.

    ``dp*/dx* = 12 u_mean / H^2`` for a plane channel (mean 1 in the inlet channel, ``1/E`` in
    the expanded one), so the drop is ``12 L_in + 12 L_out / E^3``. Used only to bring the
    network's pressure output into ``O(1)`` before scaling back; it is a geometry estimate,
    not a value read from the reference solution.
    """
    mesh = problem.mesh
    l_in = float(mesh.entry_length)
    l_out = float(problem.segments[BC_OUTLET][:, 0, 0].max())
    expansion = float(mesh.inlet_height + 2.0 * problem.step_height)
    return 12.0 * (l_in + l_out / expansion**3)


@dataclass
class GeometryScale:
    """Affine input normalisation and the pressure output scale."""

    x_centre: float
    x_half: float
    y_centre: float
    y_half: float
    pressure: float
    reynolds: float

    @classmethod
    def from_problem(cls, problem: Problem) -> GeometryScale:
        segs = np.concatenate(list(problem.segments.values()), axis=0).reshape(-1, 2)
        x_lo, x_hi = float(segs[:, 0].min()), float(segs[:, 0].max())
        y_lo, y_hi = float(segs[:, 1].min()), float(segs[:, 1].max())
        return cls(
            x_centre=0.5 * (x_lo + x_hi),
            x_half=0.5 * (x_hi - x_lo),
            y_centre=0.5 * (y_lo + y_hi),
            y_half=0.5 * (y_hi - y_lo),
            pressure=pressure_scale(problem),
            reynolds=problem.Re,
        )


class PINN(nn.Module):
    """MLP ``(x, y) -> (u, v, p)`` with tanh activations."""

    def __init__(self, scale: GeometryScale, width: int = HIDDEN_WIDTH, depth: int = HIDDEN_LAYERS):
        super().__init__()
        self.scale = scale
        layers: list[nn.Module] = [nn.Linear(2, width), ACTIVATION()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), ACTIVATION()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.scale
        x_hat = (xy[:, 0:1] - s.x_centre) / s.x_half
        y_hat = (xy[:, 1:2] - s.y_centre) / s.y_half
        out = self.net(torch.cat([x_hat, y_hat], dim=1))
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = s.pressure * out[:, 2:3]
        return u, v, p


def _grad(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """First derivative of ``output`` w.r.t. ``inputs`` (graph kept for higher derivatives)."""
    return torch.autograd.grad(
        output, inputs, grad_outputs=torch.ones_like(output), create_graph=True
    )[0]


def pde_residuals(
    model: PINN, xy: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Momentum and continuity residuals at ``xy`` (physical star coordinates)."""
    xy = xy.clone().requires_grad_(True)
    u, v, p = model(xy)
    grad_u, grad_v, grad_p = _grad(u, xy), _grad(v, xy), _grad(p, xy)
    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]
    u_xx = _grad(u_x, xy)[:, 0:1]
    u_yy = _grad(u_y, xy)[:, 1:2]
    v_xx = _grad(v_x, xy)[:, 0:1]
    v_yy = _grad(v_y, xy)[:, 1:2]

    re = model.scale.reynolds
    r_x = u * u_x + v * u_y + (p_x - (u_xx + u_yy)) / re
    r_y = u * v_x + v * v_y + (p_y - (v_xx + v_yy)) / re
    r_c = u_x + v_y
    return (u, v, p), (r_x, r_y, r_c)


def _to_tensor(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(points, dtype=torch.float32, device=device)


@dataclass
class Batches:
    """Fixed point sets for one training run (tensors on one device)."""

    interior: torch.Tensor
    inlet: torch.Tensor
    inlet_u: torch.Tensor
    wall: torch.Tensor
    outlet: torch.Tensor

    @classmethod
    def from_problem(cls, problem: Problem, device: torch.device, seed: int = SEED) -> Batches:
        points = sample_problem(
            problem,
            n_interior=N_INTERIOR,
            n_inlet=N_INLET,
            n_wall=N_WALL,
            n_outlet=N_OUTLET,
            seed=seed,
            wall_clustering=WALL_CLUSTERING,
        )
        return cls(
            interior=_to_tensor(points["interior"], device),
            inlet=_to_tensor(points[BC_INLET], device),
            inlet_u=_to_tensor(points["inlet_u"], device),
            wall=_to_tensor(points[BC_WALL], device),
            outlet=_to_tensor(points[BC_OUTLET], device),
        )


@dataclass
class LossTerms:
    """One evaluation of every loss contribution."""

    momentum: torch.Tensor
    continuity: torch.Tensor
    inlet: torch.Tensor
    wall: torch.Tensor
    outlet_pressure: torch.Tensor
    total: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "momentum": float(self.momentum.detach()),
            "continuity": float(self.continuity.detach()),
            "inlet": float(self.inlet.detach()),
            "wall": float(self.wall.detach()),
            "outlet_pressure": float(self.outlet_pressure.detach()),
            "total": float(self.total.detach()),
        }


def compute_loss(model: PINN, batches: Batches) -> LossTerms:
    """Weighted sum of the PDE residual, continuity and the three boundary conditions."""
    _, (r_x, r_y, r_c) = pde_residuals(model, batches.interior)
    momentum = torch.mean(r_x**2) + torch.mean(r_y**2)
    continuity = torch.mean(r_c**2)

    u_in, v_in, _ = model(batches.inlet)
    inlet = torch.mean((u_in - batches.inlet_u[:, 0:1]) ** 2) + torch.mean(v_in**2)

    u_w, v_w, _ = model(batches.wall)
    wall = torch.mean(u_w**2) + torch.mean(v_w**2)

    _, _, p_out = model(batches.outlet)
    outlet_pressure = torch.mean(p_out**2)

    total = (
        WEIGHT_MOMENTUM * momentum
        + WEIGHT_CONTINUITY * continuity
        + WEIGHT_INLET * inlet
        + WEIGHT_WALL * wall
        + WEIGHT_OUTLET_PRESSURE * outlet_pressure
    )
    return LossTerms(momentum, continuity, inlet, wall, outlet_pressure, total)


@dataclass
class TrainingLog:
    """Loss history and timings."""

    adam: list[dict] = field(default_factory=list)
    lbfgs: list[dict] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)
    adam_steps: int = 0
    lbfgs_steps: int = 0

    def as_dict(self) -> dict:
        return {
            "adam": self.adam,
            "lbfgs": self.lbfgs,
            "seconds": self.seconds,
            "adam_steps": self.adam_steps,
            "lbfgs_steps": self.lbfgs_steps,
        }


def _device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    problem: Problem | None = None,
    *,
    adam_steps: int = ADAM_STEPS,
    lbfgs_max_iter: int = LBFGS_MAX_ITER,
    device: str = "auto",
    seed: int = SEED,
    log: bool = True,
) -> tuple[PINN, TrainingLog]:
    """Train the PINN: Adam with cosine decay, then full-batch L-BFGS."""
    problem = problem or build_problem()
    torch.manual_seed(seed)
    dev = _device(device)

    model = PINN(GeometryScale.from_problem(problem)).to(dev)
    batches = Batches.from_problem(problem, dev, seed=seed)
    history = TrainingLog()

    def report(stage: str, step: int, terms: LossTerms) -> None:
        row = {"step": step, **terms.as_dict()}
        (history.adam if stage == "adam" else history.lbfgs).append(row)
        if log:
            print(
                f"{stage:5s} {step:5d}  total {row['total']:.3e}  mom {row['momentum']:.3e}"
                f"  cont {row['continuity']:.3e}  in {row['inlet']:.3e}"
                f"  wall {row['wall']:.3e}  p_out {row['outlet_pressure']:.3e}",
                flush=True,
            )

    t0 = time.perf_counter()
    adam = torch.optim.Adam(model.parameters(), lr=ADAM_LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        adam, T_max=max(1, adam_steps), eta_min=ADAM_LEARNING_RATE * ADAM_FINAL_LR_FRACTION
    )
    for step in range(1, adam_steps + 1):
        adam.zero_grad(set_to_none=True)
        terms = compute_loss(model, batches)
        terms.total.backward()
        adam.step()
        schedule.step()
        if step % LOG_EVERY == 0 or step == 1:
            report("adam", step, terms)
    history.adam_steps = adam_steps
    history.seconds["adam"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LEARNING_RATE,
        max_iter=lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    state = {"step": 0}

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        terms = compute_loss(model, batches)
        terms.total.backward()
        state["step"] += 1
        if state["step"] % 50 == 0:
            report("lbfgs", state["step"], terms)
        return terms.total

    lbfgs.step(closure)
    history.lbfgs_steps = state["step"]
    history.seconds["lbfgs"] = time.perf_counter() - t1
    history.seconds["total"] = time.perf_counter() - t0
    model.eval()
    return model, history


def evaluate_on(model: PINN, xy: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate ``u``, ``v``, ``p`` at ``xy`` (no autograd); runs on the model's device."""
    params = list(model.parameters())
    dev = params[0].device if params else torch.device("cpu")
    with torch.no_grad():
        u, v, p = model(_to_tensor(xy, dev))
    return {"u": u.cpu().numpy()[:, 0], "v": v.cpu().numpy()[:, 0], "p": p.cpu().numpy()[:, 0]}


def plot_losses(history: TrainingLog, out_png: Path) -> Path:
    """Loss history on a log scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for key in ("total", "momentum", "continuity", "inlet", "wall", "outlet_pressure"):
        steps = [row["step"] for row in history.adam]
        values = [row[key] for row in history.adam]
        ax.plot(steps, values, lw=1.0, label=f"{key} (Adam)")
    n_adam = history.adam[-1]["step"] if history.adam else 0
    for key, style in (("total", "-"), ("momentum", "--")):
        steps = [n_adam + row["step"] for row in history.lbfgs]
        values = [row[key] for row in history.lbfgs]
        ax.plot(steps, values, style, lw=1.0, color="black" if key == "total" else "tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("step (Adam, then L-BFGS)")
    ax.set_ylabel("loss contribution")
    ax.set_title("T9 PINN loss history (black solid: L-BFGS total, dashed: momentum)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def run_dir_for(problem: Problem) -> Path:
    """Per-case output directory under ``data/pinn_results/``."""
    return RESULTS_DIR / f"{problem.run_id}_mlp"


def save_run(model: PINN, history: TrainingLog, problem: Problem, out_dir: Path) -> Path:
    """Write the checkpoint, the loss history and the loss figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history.as_dict(), indent=1))
    plot_losses(history, out_dir / "losses.png")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": problem.run_id,
                "family": problem.family,
                "rheology": problem.rheology,
                "Re": problem.Re,
                "hidden_width": HIDDEN_WIDTH,
                "hidden_layers": HIDDEN_LAYERS,
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "adam_steps": history.adam_steps,
                "lbfgs_steps": history.lbfgs_steps,
                "seconds": history.seconds,
                "weights": {
                    "momentum": WEIGHT_MOMENTUM,
                    "continuity": WEIGHT_CONTINUITY,
                    "inlet": WEIGHT_INLET,
                    "wall": WEIGHT_WALL,
                    "outlet_pressure": WEIGHT_OUTLET_PRESSURE,
                },
                "sampling": {
                    "interior": N_INTERIOR,
                    "inlet": N_INLET,
                    "wall": N_WALL,
                    "outlet": N_OUTLET,
                    "wall_clustering": WALL_CLUSTERING,
                },
                "seed": SEED,
            },
            indent=1,
        )
    )
    return out_dir


def main() -> None:
    """Step 2: train on ``M1_G2_052`` (lowest-Re Newtonian G2) and log the losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adam-steps", type=int, default=ADAM_STEPS)
    parser.add_argument("--lbfgs-iter", type=int, default=LBFGS_MAX_ITER)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    problem = build_problem()
    if args.run_id is not None:
        from pinn.problem import read_manifest

        record = next(r for r in read_manifest() if r["run_id"] == args.run_id)
        problem = build_problem(record)

    print(f"case           {problem.run_id} (Re = {problem.Re:.4f}, E = {problem.mesh.params['E']:.4f})")
    print(f"pressure scale {pressure_scale(problem):.2f}")
    print(f"device         {_device(args.device)}")
    model, history = train(
        problem, adam_steps=args.adam_steps, lbfgs_max_iter=args.lbfgs_iter, device=args.device
    )
    out_dir = save_run(model, history, problem, run_dir_for(problem))
    print(f"adam  {history.seconds['adam']:.1f} s   lbfgs {history.seconds['lbfgs']:.1f} s"
          f"   total {history.seconds['total']:.1f} s")
    print(f"written        {out_dir}")


if __name__ == "__main__":
    main()
