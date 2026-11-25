"""Physics-only problem definition for the legacy M1 corpus (task T9, step 1).

The CFD runs in ``data/corpus_v1_legacy/`` are used **only as reference solutions** for
evaluation; nothing here reads a CFD velocity or pressure into a training target. What the
network needs is the geometry, the boundary conditions and the PDE, all rebuilt from
``params["geometry"]`` via :func:`cfd.gen.geometry.build_geometry`.

Star units (``data/corpus_v1_legacy/README.md``): length ``h1 = 1``, velocity ``U_mean = 1``,
viscosity ``mu_inf = 1``, pressure ``p* = p h1 / (mu_inf U) = Re * p_OF``.

Steady momentum balance written in star units for the **stored** ``p_star``:

.. code-block:: none

    (u*.grad)u* + (1/Re) ( grad p* - lap u* ) = 0      (Newtonian, mu_r = 1)
    div u* = 0

The ``1/Re`` sits on the pressure gradient and on the viscous term alike; advection carries
none. Verified against the data on ``M1_G2_000`` (Re = 2.33): in the fully developed inlet
channel the resampled grid gives ``dp*/dx = -11.97`` against ``d2u*/dy*2 = -12.00``, which
cancels only in the form above. ``data/corpus_v1_legacy/README.md`` writes the balance as
``(u.grad)u = -grad p + (1/Re) div[2 mu D]``; read against the stored ``p_star`` that form is
off by a factor ``Re`` on the pressure term (FLAGged for review in ``plan/REVIEW_QUEUE.md``).
The two agree if the README's ``p`` means ``p_OF = p*/Re``.

The inlet BC depends on the rheology (``cfd/gen/case.py``): Newtonian runs get a plane
parabolic profile with mean 1, Carreau runs a uniform ``u = (1, 0)``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfd.gen.geometry import MeshSpec, build_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "data" / "corpus_v1_legacy"
MANIFEST_PATH = CORPUS_ROOT / "manifest.jsonl"
RESULTS_DIR = REPO_ROOT / "data" / "pinn_results"

# Plane parabolic inlet profile u(y) = PARABOLIC_PEAK * y_hat (1 - y_hat) * U_mean, which has
# mean 1 over the inlet section (knowledge/11-math-model.md section 2.4, plane Poiseuille).
PARABOLIC_PEAK = 6.0
INLET_MEAN = 1.0

# Boundary-condition kinds we build from the mesh patches.
BC_INLET = "inlet"
BC_WALL = "wall"
BC_OUTLET = "outlet"
BC_KINDS = (BC_INLET, BC_WALL, BC_OUTLET)


# ---------------------------------------------------------------------------
# Manifest and run data (reference fields, evaluation only)
# ---------------------------------------------------------------------------


def read_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    """Read ``manifest.jsonl`` (one JSON record per run)."""
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select_run(
    family: str = "G2",
    rheology: str = "Newtonian",
    status: str = "ok",
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    """Lowest-Re run matching ``family``/``rheology``/``status`` (README: skip non-``ok`` runs)."""
    rows = [
        row
        for row in read_manifest(manifest_path)
        if row["params"].get("family") == family
        and row["params"].get("rheology") == rheology
        and row["status"] == status
    ]
    if not rows:
        raise LookupError(
            f"no {family} {rheology} run with status={status!r} in {manifest_path}"
        )
    return min(rows, key=lambda row: float(row["params"]["Re"]))


def run_npz_path(run_id: str, family: str) -> Path:
    """Local ``.npz`` path for a run (``m1/<family>/<run_id>.npz``)."""
    return CORPUS_ROOT / "m1" / family / f"{run_id}.npz"


def load_reference(path: Path) -> dict:
    """Load the reference point cloud of one run (evaluation targets only).

    Includes the resampled regular grid (``grid_*``) that the corpus ships for plotting;
    nothing here is used for training.
    """
    with np.load(path, allow_pickle=True) as npz:
        params = json.loads(str(npz["params"]))
        return {
            "x": np.asarray(npz["x_star"], dtype=float),
            "y": np.asarray(npz["y_star"], dtype=float),
            "u": np.asarray(npz["U_star"][:, 0], dtype=float),
            "v": np.asarray(npz["U_star"][:, 1], dtype=float),
            "p": np.asarray(npz["p_star"], dtype=float),
            "cell_volumes": np.asarray(npz["cell_volumes"], dtype=float),
            "grid_x": np.asarray(npz["grid_x"], dtype=float),
            "grid_y": np.asarray(npz["grid_y"], dtype=float),
            "grid_mask": np.asarray(npz["grid_mask"], dtype=bool),
            "grid_u": np.asarray(npz["grid_ux"], dtype=float),
            "grid_v": np.asarray(npz["grid_uy"], dtype=float),
            "grid_p": np.asarray(npz["grid_p"], dtype=float),
            "params": params,
        }


# ---------------------------------------------------------------------------
# Boundary of the domain, rebuilt from the geometry spec
# ---------------------------------------------------------------------------


def _patch_bc_kind(patch) -> str | None:
    """Map a mesh patch onto a BC kind; ``None`` for the 2D front/back (``empty``) patch."""
    if patch.patch_type == "empty":
        return None
    name = patch.name.lower()
    if "inlet" in name:
        return BC_INLET
    if "outlet" in name:
        return BC_OUTLET
    if patch.patch_type == "wall" or "wall" in name:
        return BC_WALL
    raise ValueError(f"unrecognised boundary patch {patch.name!r} ({patch.patch_type})")


def boundary_segments(mesh: MeshSpec) -> dict[str, np.ndarray]:
    """Boundary segments as ``{kind: (n, 2, 2)}`` arrays of ``(x, y)`` endpoints.

    Each patch face of the extruded mesh is one block edge, so its 2D footprint is a single
    straight segment of the physical boundary. Curved families (G6 arcs) are therefore
    approximated by their chord — exact for all straight-edged families, G2 included.
    """
    segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    for patch in mesh.patches:
        kind = _patch_bc_kind(patch)
        if kind is None:
            continue
        for face in patch.faces:
            pts = sorted(
                {
                    (round(mesh.vertices[i][0], 9), round(mesh.vertices[i][1], 9))
                    for i in face
                }
            )
            if len(pts) != 2:
                raise ValueError(
                    f"patch {patch.name!r} face {face} is not a straight boundary segment"
                )
            segments.setdefault(kind, []).append((pts[0], pts[1]))

    missing = [kind for kind in (BC_INLET, BC_OUTLET) if kind not in segments]
    if missing:
        raise ValueError(f"geometry {mesh.family} has no {missing} patch")
    return {kind: np.asarray(segs, dtype=float) for kind, segs in segments.items()}


def segment_lengths(segments: np.ndarray) -> np.ndarray:
    """Euclidean length of each ``(x, y)`` segment."""
    delta = segments[:, 1, :] - segments[:, 0, :]
    return np.hypot(delta[:, 0], delta[:, 1])


def sample_segments(segments: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` points uniformly **by arc length** on a set of segments."""
    lengths = segment_lengths(segments)
    prob = lengths / lengths.sum()
    idx = rng.choice(segments.shape[0], size=n, p=prob)
    t = rng.random(n)
    start = segments[idx, 0, :]
    return start + t[:, None] * (segments[idx, 1, :] - start)


def block_quads(mesh: MeshSpec) -> np.ndarray:
    """2D corners of every mesh block, shape ``(n_blocks, 4, 2)`` (counter-clockwise)."""
    return np.asarray(
        [[mesh.vertices_2d[i] for i in block.verts] for block in mesh.blocks_2d], dtype=float
    )


def quad_areas(quads: np.ndarray) -> np.ndarray:
    """Signed polygon area of each quad (positive for counter-clockwise corners)."""
    x = quads[:, :, 0]
    y = quads[:, :, 1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    )


def sample_interior(
    quads: np.ndarray, n: int, rng: np.random.Generator, wall_clustering: bool = False
) -> np.ndarray:
    """Sample ``n`` collocation points over the fluid domain.

    The domain is the union of the mesh blocks; each block is filled through its bilinear
    map from the unit square, with blocks chosen in proportion to their area. Exact for
    parallelogram blocks (all of G2) and conservative for slightly sheared ones.

    With ``wall_clustering`` the second (wall-normal) block parameter is mapped through
    ``t -> (1 - cos(pi t)) / 2``, a Chebyshev distribution that resolves the thin boundary
    layers where a plain PINN loses its mass flux (see ``plan/REVIEW_QUEUE.md``, T9).
    """
    areas = quad_areas(quads)
    idx = rng.choice(quads.shape[0], size=n, p=areas / areas.sum())
    s = rng.random(n)
    t = rng.random(n)
    if wall_clustering:
        t = 0.5 * (1.0 - np.cos(np.pi * t))
    corners = quads[idx]
    w00 = (1.0 - s) * (1.0 - t)
    w10 = s * (1.0 - t)
    w11 = s * t
    w01 = (1.0 - s) * t
    return (
        w00[:, None] * corners[:, 0, :]
        + w10[:, None] * corners[:, 1, :]
        + w11[:, None] * corners[:, 2, :]
        + w01[:, None] * corners[:, 3, :]
    )


# ---------------------------------------------------------------------------
# The problem itself
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """One corpus run turned into a physics-only boundary-value problem."""

    run_id: str
    family: str
    rheology: str
    Re: float
    mesh: MeshSpec
    segments: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    scalars: dict[str, float | None]

    # -- scales ------------------------------------------------------------
    @property
    def mu_ratio(self) -> float:
        """``mu_r = mu/mu_inf`` (1 for Newtonian; the Carreau law later reads this)."""
        return 1.0

    @property
    def viscosity(self) -> float:
        """Star-unit viscosity ``nu* = 1/Re`` for the Newtonian runs."""
        return 1.0 / self.Re

    @property
    def parabolic_inlet(self) -> bool:
        """Newtonian runs get the parabolic inlet; Carreau runs a uniform one (README)."""
        return self.rheology.lower() == "newtonian"

    # -- geometry ----------------------------------------------------------
    @property
    def quads(self) -> np.ndarray:
        return block_quads(self.mesh)

    @property
    def inlet_section(self) -> tuple[float, float, float]:
        """``(x_in, y_lo, y_hi)`` of the inlet patch."""
        pts = self.segments[BC_INLET].reshape(-1, 2)
        return float(pts[:, 0].mean()), float(pts[:, 1].min()), float(pts[:, 1].max())

    @property
    def step_height(self) -> float:
        """Downstream wall offset of a G2 expansion, ``s = (E - 1)/2`` in star units."""
        y_lo = float(self.segments[BC_OUTLET].reshape(-1, 2)[:, 1].min())
        return abs(y_lo)

    def inlet_velocity(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inlet BC ``(u, v)`` at ``points``: parabolic (mean 1) or uniform, ``v = 0``."""
        _, y_lo, y_hi = self.inlet_section
        height = y_hi - y_lo
        v = np.zeros(points.shape[0])
        if not self.parabolic_inlet:
            return np.full(points.shape[0], INLET_MEAN), v
        eta = (points[:, 1] - y_lo) / height
        return PARABOLIC_PEAK * INLET_MEAN * eta * (1.0 - eta), v

    # -- sampling ----------------------------------------------------------
    def sample_interior(
        self, n: int, rng: np.random.Generator, wall_clustering: bool = False
    ) -> np.ndarray:
        """Interior collocation points, shape ``(n, 2)``."""
        return sample_interior(self.quads, n, rng, wall_clustering=wall_clustering)

    def sample_boundary(
        self, kind: str, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``n`` points on the ``inlet`` / ``wall`` / ``outlet`` boundary, shape ``(n, 2)``."""
        if kind not in self.segments:
            raise KeyError(f"no {kind!r} boundary in {self.run_id}")
        return sample_segments(self.segments[kind], n, rng)


def build_problem(
    record: Mapping | None = None,
    *,
    family: str = "G2",
    rheology: str = "Newtonian",
    manifest_path: Path = MANIFEST_PATH,
) -> Problem:
    """Rebuild one run's boundary-value problem (default: lowest-Re Newtonian ``ok`` G2)."""
    if record is None:
        record = select_run(
            family=family, rheology=rheology, manifest_path=manifest_path
        )
    family = str(record["family"])
    params = record["params"]
    level = int(params.get("level", 1))
    mesh = build_geometry(family, params["geometry"], level=level)
    reference = load_reference(run_npz_path(str(record["run_id"]), family))
    return Problem(
        run_id=str(record["run_id"]),
        family=family,
        rheology=str(params["rheology"]),
        Re=float(params["Re"]),
        mesh=mesh,
        segments=boundary_segments(mesh),
        reference=reference,
        scalars=dict(params.get("scalars", {})),
    )


def sample_problem(
    problem: Problem,
    *,
    n_interior: int = 4096,
    n_inlet: int = 256,
    n_wall: int = 1024,
    n_outlet: int = 256,
    seed: int = 0,
    wall_clustering: bool = False,
) -> dict[str, np.ndarray]:
    """Sample every point set the training loss needs, plus the reference cloud."""
    rng = np.random.default_rng(seed)
    points = {
        "interior": problem.sample_interior(n_interior, rng, wall_clustering=wall_clustering),
        BC_INLET: problem.sample_boundary(BC_INLET, n_inlet, rng),
        BC_WALL: problem.sample_boundary(BC_WALL, n_wall, rng),
        BC_OUTLET: problem.sample_boundary(BC_OUTLET, n_outlet, rng),
    }
    u_in, v_in = problem.inlet_velocity(points[BC_INLET])
    points["inlet_u"] = np.stack([u_in, v_in], axis=1)
    points["reference_x"] = np.stack(
        [problem.reference["x"], problem.reference["y"]], axis=1
    )
    return points


# ---------------------------------------------------------------------------
# Figure: where the points sit
# ---------------------------------------------------------------------------


def plot_points(problem: Problem, points: Mapping[str, np.ndarray], out_png: Path) -> Path:
    """Draw the geometry, the sampled point sets and the CFD reference cloud."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.scatter(
        points["reference_x"][::4, 0],
        points["reference_x"][::4, 1],
        s=1.0,
        c="0.80",
        label="CFD cell centres (reference, evaluation only)",
    )
    for kind, colour in (
        (BC_WALL, "black"),
        (BC_INLET, "tab:blue"),
        (BC_OUTLET, "tab:red"),
    ):
        for seg in problem.segments[kind]:
            ax.plot(seg[:, 0], seg[:, 1], color=colour, lw=1.0)
        ax.scatter(
            points[kind][:, 0],
            points[kind][:, 1],
            s=3.0,
            c=colour,
            label=f"{kind} points",
        )
    ax.scatter(
        points["interior"][:, 0],
        points["interior"][:, 1],
        s=0.6,
        c="tab:green",
        alpha=0.5,
        label=f"interior collocation points ({points['interior'].shape[0]})",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x*")
    ax.set_ylabel("y*")
    ax.set_title(
        rf"{problem.run_id}: {problem.family} {problem.rheology}, Re = {problem.Re:.2f},"
        rf" E = {problem.mesh.params['E']:.3f} (star units, h1 = 1)"
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=8,
        framealpha=0.9,
        markerscale=3.0,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main() -> None:
    """Step 1 check: build the problem, sample the points, plot them, print a summary."""
    problem = build_problem()
    points = sample_problem(problem)
    out_png = RESULTS_DIR / f"points_{problem.run_id}.png"
    plot_points(problem, points, out_png)

    x_in, y_lo, y_hi = problem.inlet_section
    seg_len = {
        kind: float(segment_lengths(problem.segments[kind]).sum())
        for kind in problem.segments
    }
    print(f"run            {problem.run_id} ({problem.family}, {problem.rheology})")
    print(f"Re             {problem.Re:.4f}   nu* = 1/Re = {problem.viscosity:.4f}")
    print(f"geometry       {json.dumps(problem.mesh.params)}")
    print(f"inlet section  x = {x_in:.4f}, y in [{y_lo:.4f}, {y_hi:.4f}]")
    print(f"outlet/step    x = {problem.segments[BC_OUTLET][:, 0, 0].max():.4f}, "
          f"s = {problem.step_height:.4f}")
    print("boundary length " + ", ".join(f"{k} {v:.4f}" for k, v in seg_len.items()))
    print(f"points         interior {points['interior'].shape[0]}, "
          f"inlet {points[BC_INLET].shape[0]}, wall {points[BC_WALL].shape[0]}, "
          f"outlet {points[BC_OUTLET].shape[0]}")
    print(f"reference      {points['reference_x'].shape[0]} cell centres")
    print(f"scalars        {json.dumps(problem.scalars)}")
    print(f"figure         {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
