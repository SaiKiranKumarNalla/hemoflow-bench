"""Evaluation of a trained physics-only PINN against the CFD reference (task T9, step 3).

Nothing here feeds back into training: the CFD point cloud is read once, the network is
evaluated on the same coordinates, and the two are compared with

* volume-weighted relative L2 errors of ``u``, ``v`` and ``p`` (``p`` up to the outlet gauge,
  i.e. after removing the best constant offset — the network gets ``p = 0`` at the outlet only
  as a soft boundary condition, so a small constant offset is expected and physically
  irrelevant);
* the inlet-to-outlet pressure drop ``dp*``, computed with the same definition the corpus
  uses (``campaign/run_batch.py::_compute_dp_star``: flux-weighted inlet column, linearly
  extrapolated back to the inlet plane);
* the reattachment length ``x_r/h1`` from the near-wall velocity zero crossing
  (``rbc/postprocess.py::reattachment_length`` on the ``nearWallLower`` sample line).

Plots (fields, errors) and ``metrics.json`` are written next to the model checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from campaign.run_batch import _compute_dp_star
from pinn.pinn import (
    PINN,
    GeometryScale,
    evaluate_on,
    run_dir_for,
)
from pinn.problem import BC_OUTLET, Problem, build_problem, read_manifest
from rbc.postprocess import reattachment_length

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "pinn_results"

# The corpus samples the near-wall line 0.02 h1 off the wall (cfd/gen/geometry.py
# SampleLine "nearWallLower"); the PINN is probed at the same offset and resolution.
NEAR_WALL_OFFSET = 0.02
NEAR_WALL_SAMPLES = 200
END_MARGIN = 0.02


@dataclass
class Metrics:
    """Everything the T9 report table needs for one case."""

    run_id: str
    family: str
    rheology: str
    Re: float
    architecture: str
    n_parameters: int
    train_seconds: float
    rel_l2_u: float
    rel_l2_v: float
    rel_l2_p: float
    flux_error: float
    pressure_offset: float
    dp_star_pinn: float
    dp_star_cfd: float
    dp_star_error: float
    xr_pinn: float
    xr_cfd: float
    xr_error: float
    xr_note: str

    def as_dict(self) -> dict[str, float | str]:
        return dict(self.__dict__)

    def table_row(self) -> str:
        """One markdown table row."""
        return (
            f"| {self.run_id} | {self.architecture} | {self.n_parameters} |"
            f" {self.train_seconds:.0f} s | {self.rel_l2_u:.2%} | {self.rel_l2_v:.2%} |"
            f" {self.rel_l2_p:.2%} ({self.flux_error:.1%}) | {self.dp_star_error:.2%} |"
            f" {_fmt(self.xr_error)} |"
        )


def _fmt(value: float) -> str:
    """Format a possibly-undefined metric for the report table."""
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.3f}"


def relative_l2(pred: np.ndarray, ref: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Volume-weighted relative L2 error (plain L2 when ``weights`` is ``None``)."""
    diff = pred - ref
    if weights is None:
        return float(np.sqrt(np.sum(diff**2) / np.sum(ref**2)))
    return float(np.sqrt(np.sum(weights * diff**2) / np.sum(weights * ref**2)))


def gauge_offset(pred: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> float:
    """Best constant offset ``c`` minimising ``sum w (pred + c - ref)^2``."""
    return float(np.sum(weights * (ref - pred)) / np.sum(weights))


def near_wall_line(problem: Problem) -> np.ndarray:
    """``nearWallLower`` sample line of the expansion geometry (x from 0.02 to L_out - 0.02)."""
    x_out = float(problem.segments[BC_OUTLET][:, 0, 0].max())
    x = np.linspace(END_MARGIN, x_out - END_MARGIN, NEAR_WALL_SAMPLES)
    y = np.full(x.shape, -problem.step_height + NEAR_WALL_OFFSET)
    return np.column_stack([x, y])


def section_flux(model: PINN, x: float, y_lo: float, y_hi: float, n_points: int = 120) -> float:
    """``Q(x) = int u dy`` across the channel section at ``x``."""
    y = np.linspace(y_lo, y_hi, n_points)
    u = evaluate_on(model, np.column_stack([np.full_like(y, x), y]))["u"]
    return float(np.trapezoid(u, y))


def flux_error(problem: Problem, model: PINN, n_sections: int = 24, n_points: int = 120) -> float:
    """Largest relative deviation of the streamwise flow rate from the unit inlet flux.

    With ``div u = 0`` and no-slip walls the flux must equal 1 in every channel section, so this
    single number exposes a network that satisfies the PDE pointwise but loses mass (the failure
    mode of an under-weighted continuity term).
    """
    l_in = float(problem.mesh.entry_length)
    x_out = float(problem.segments[BC_OUTLET][:, 0, 0].max())
    s = problem.step_height
    worst = 0.0
    for x in np.linspace(-l_in + 0.1, x_out - 0.1, n_sections):
        if abs(x) < 0.1:  # straddles the step: skip the section itself
            continue
        bounds = (0.0, 1.0) if x < 0 else (-s, 1.0 + s)
        worst = max(worst, abs(section_flux(model, x, *bounds, n_points=n_points) - 1.0))
    return worst


def reattachment(problem: Problem, model: PINN) -> tuple[float, str]:
    """``x_r/h1`` from the near-wall zero crossing; ``nan`` when the PINN never reattaches."""
    line = near_wall_line(problem)
    u_near_wall = evaluate_on(model, line)["u"]
    try:
        return float(reattachment_length(line[:, 0], u_near_wall, x_step=0.0)), "ok"
    except ValueError as exc:  # recirculation still present at the end of the sampled line
        return float("nan"), str(exc)


def pressure_drop(problem: Problem, fields: dict[str, np.ndarray]) -> float:
    """Inlet-to-outlet ``dp*`` with the corpus definition on the reference point cloud."""
    ref = problem.reference
    velocity = np.column_stack([fields["u"], fields["v"], np.zeros_like(fields["u"])])
    return float(
        _compute_dp_star(
            problem.mesh,
            ref["x"],
            ref["y"],
            velocity,
            fields["p"],
            ref["cell_volumes"],
        )
    )


def reference_grid_points(problem: Problem) -> np.ndarray:
    """``(x, y)`` of every node of the corpus resampling grid (evaluation only)."""
    ref = problem.reference
    x, y = np.meshgrid(ref["grid_x"], ref["grid_y"])
    return np.column_stack([x.ravel(), y.ravel()])


def evaluate_model(
    model: PINN,
    problem: Problem,
    *,
    train_seconds: float = float("nan"),
    n_parameters: int | None = None,
    architecture: str = "MLP tanh",
) -> tuple[Metrics, dict[str, np.ndarray]]:
    """Compare a trained network with the CFD reference of ``problem``."""
    ref = problem.reference
    xy = np.column_stack([ref["x"], ref["y"]])
    fields = evaluate_on(model, xy)
    weights = ref["cell_volumes"]

    offset = gauge_offset(fields["p"], ref["p"], weights)
    p_shifted = fields["p"] + offset

    dp_pinn = pressure_drop(problem, {"u": fields["u"], "v": fields["v"], "p": p_shifted})
    dp_cfd = float(problem.scalars["dp_star"])

    xr_pinn, xr_note = reattachment(problem, model)
    xr_cfd = float(problem.scalars["xr_over_h1"] or 0.0)

    metrics = Metrics(
        run_id=problem.run_id,
        family=problem.family,
        rheology=problem.rheology,
        Re=problem.Re,
        architecture=architecture,
        n_parameters=n_parameters
        if n_parameters is not None
        else sum(p.numel() for p in model.parameters()),
        train_seconds=train_seconds,
        rel_l2_u=relative_l2(fields["u"], ref["u"], weights),
        rel_l2_v=relative_l2(fields["v"], ref["v"], weights),
        rel_l2_p=relative_l2(p_shifted, ref["p"], weights),
        flux_error=flux_error(problem, model),
        pressure_offset=offset,
        dp_star_pinn=dp_pinn,
        dp_star_cfd=dp_cfd,
        dp_star_error=abs(dp_pinn - dp_cfd) / abs(dp_cfd),
        xr_pinn=xr_pinn,
        xr_cfd=xr_cfd,
        xr_error=abs(xr_pinn - xr_cfd) if np.isfinite(xr_pinn) else float("nan"),
        xr_note=xr_note,
    )
    shape = (ref["grid_y"].size, ref["grid_x"].size)
    grid = evaluate_on(model, reference_grid_points(problem))
    fields.update(
        {
            "grid_u": grid["u"].reshape(shape),
            "grid_v": grid["v"].reshape(shape),
            "grid_p": grid["p"].reshape(shape) + offset,
        }
    )
    return metrics, fields


def plot_fields(problem: Problem, fields: dict[str, np.ndarray], out_png: Path) -> Path:
    """CFD vs PINN fields on the corpus grid (rows: u, v, p; columns: CFD, PINN, error)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = problem.reference
    mask = ref["grid_mask"]
    gx, gy = ref["grid_x"], ref["grid_y"]
    pairs = (
        ("u*", ref["grid_u"], fields["grid_u"]),
        ("v*", ref["grid_v"], fields["grid_v"]),
        ("p*", ref["grid_p"], fields["grid_p"]),
    )
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 6.4), sharex=True, sharey=True)
    for row, (label, cfd, pinn) in enumerate(pairs):
        err = pinn - cfd
        vmin = float(np.nanmin(cfd[mask]))
        vmax = float(np.nanmax(cfd[mask]))
        emax = float(np.nanmax(np.abs(err[mask]))) or 1.0
        for col, (data, title, lo, hi, cmap) in enumerate(
            (
                (cfd, "CFD (reference)", vmin, vmax, "viridis"),
                (pinn, "PINN (physics only)", vmin, vmax, "viridis"),
                (err, "PINN - CFD", -emax, emax, "RdBu_r"),
            )
        ):
            ax = axes[row, col]
            mesh = ax.pcolormesh(gx, gy, np.where(mask, data, np.nan), shading="auto",
                                 cmap=cmap, vmin=lo, vmax=hi)
            ax.set_aspect("equal")
            fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.01, format="%.3g")
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{label}\ny*")
            if row == 2:
                ax.set_xlabel("x*")
    fig.suptitle(
        f"{problem.run_id}: {problem.family} {problem.rheology}, Re = {problem.Re:.2f}"
        " (star units; p* up to the outlet gauge)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_wall_profile(problem: Problem, model: PINN, metrics: Metrics, out_png: Path) -> Path:
    """Near-wall velocity along the lower wall: the reattachment metric, visualised."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    line = near_wall_line(problem)
    pinn = evaluate_on(model, line)["u"]
    ref = problem.reference
    band = (ref["y"] < -problem.step_height + 1.5 * NEAR_WALL_OFFSET) & (ref["y"] > -problem.step_height)
    cfd_x, cfd_u = ref["x"][band], ref["u"][band]
    order = np.argsort(cfd_x)

    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.plot(line[:, 0], pinn, lw=1.4, label="PINN, y = -s + 0.02")
    ax.plot(cfd_x[order], cfd_u[order], ".", ms=2.0, color="0.4", label="CFD cells, near-wall band")
    ax.axvline(metrics.xr_pinn, color="tab:blue", ls="--", lw=1.0,
               label=f"PINN x_r = {metrics.xr_pinn:.3f}")
    ax.axvline(metrics.xr_cfd, color="0.4", ls=":", lw=1.2,
               label=f"CFD x_r = {metrics.xr_cfd:.3f}")
    ax.set_xlabel("x*")
    ax.set_ylabel("u*")
    ax.set_title("Recirculation behind the expansion step (lower wall)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def load_model(problem: Problem, out_dir: Path | None = None) -> PINN:
    """Rebuild the network and load the checkpoint written by ``pinn.pinn``."""
    out_dir = out_dir or run_dir_for(problem)
    model = PINN(GeometryScale.from_problem(problem))
    state = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    """Evaluate one trained case and print the report-table row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    problem = build_problem()
    if args.run_id is not None:
        record = next(r for r in read_manifest() if r["run_id"] == args.run_id)
        problem = build_problem(record)

    out_dir = Path(args.out_dir) if args.out_dir else run_dir_for(problem)
    meta = json.loads((out_dir / "meta.json").read_text())
    model = load_model(problem, out_dir)
    metrics, fields = evaluate_model(
        model,
        problem,
        train_seconds=float(meta["seconds"]["total"]),
        n_parameters=int(meta["n_parameters"]),
        architecture=f"MLP tanh {meta['hidden_width']}x{meta['hidden_layers']}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics.as_dict(), indent=1))
    plot_fields(problem, fields, out_dir / "fields.png")
    plot_wall_profile(problem, model, metrics, out_dir / "wall_profile.png")

    print(f"case           {metrics.run_id} (Re = {metrics.Re:.4f})")
    print(f"rel L2         u {metrics.rel_l2_u:.4%}   v {metrics.rel_l2_v:.4%}"
          f"   p {metrics.rel_l2_p:.4%}  (p gauge offset {metrics.pressure_offset:+.3f})")
    print(f"mass flux      max |Q(x) - 1| = {metrics.flux_error:.2%}")
    print(f"dp*            CFD {metrics.dp_star_cfd:.3f}   PINN {metrics.dp_star_pinn:.3f}"
          f"   err {metrics.dp_star_error:.2%}")
    print(f"x_r/h1         CFD {metrics.xr_cfd:.4f}   PINN {metrics.xr_pinn:.4f}"
          f"   err {metrics.xr_error:.4f}")
    print(f"train time     {metrics.train_seconds:.1f} s")
    print(f"artifacts      {out_dir}")


if __name__ == "__main__":
    main()
