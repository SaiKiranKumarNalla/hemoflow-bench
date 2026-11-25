"""Runs on disk -> (X, Y) table -> reproducible train/val/test split.

Run file format (one `.npz` per run, the same for analytic and CFD runs):
    y, u, tau, gdot : 1D cross-section arrays (G1); CFD runs add their own field arrays
    G               : pressure gradient -dp/dx [Pa/m]
    params          : JSON string with at least U_mean, H, L, rho, Re, Cu, source
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rbc.physics import Carreau, Viscosity, solve_channel
from rbc.postprocess import channel_scalars

FEATURES: tuple[str, ...] = ("U_mean", "H")
TARGETS: tuple[str, ...] = ("G", "tau_wall", "f_window", "t_exp_mean", "hi_mean")

# Design ranges: a design choice spanning the 2023 cases (D = 23–60 µm, U = 0.25–0.33 m/s), not literature values.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {"U_mean": (0.05, 0.5), "H": (25e-6, 100e-6)}


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """n points in [0, 1)^dim, exactly one point in each of the n equal-width bins of every axis."""
    bins = np.stack([rng.permutation(n) for _ in range(dim)], axis=1)  # (n, dim) bin index per axis
    return (bins + rng.random((n, dim))) / n


def sample_design(n: int, seed: int, bounds: dict[str, tuple[float, float]] = DEFAULT_BOUNDS) -> np.ndarray:
    """Latin-hypercube sample, log-uniform in each parameter (they span a decade)."""
    rng = np.random.default_rng(seed)
    unit = latin_hypercube(n, len(bounds), rng)
    lo = np.log10([b[0] for b in bounds.values()])
    hi = np.log10([b[1] for b in bounds.values()])
    return 10.0 ** (lo + unit * (hi - lo))


def save_run(path: Path, arrays: dict[str, np.ndarray], params: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, params=json.dumps(params), **arrays)


def load_run(path: Path) -> dict:
    with np.load(path) as data:
        run = {k: data[k] for k in data.files if k != "params"}
        run["params"] = json.loads(str(data["params"]))
    return run


def generate_channel_runs(
    out_dir: Path,
    n: int,
    seed: int,
    model: Viscosity | None = None,
    L_over_H: float = 20.0,
    rho: float = 1060.0,
) -> list[Path]:
    """Analytic fully developed G1 runs (no CFD) — lets the pipeline run before the CFD campaign."""
    model = Carreau() if model is None else model
    paths = []
    for i, (U, H) in enumerate(sample_design(n, seed)):
        flow = solve_channel(model, H=H, U_mean=U)
        mu_ref = model.mu_min
        params = {
            "U_mean": float(U), "H": float(H), "L": float(L_over_H * H), "rho": rho,
            "Re": float(rho * U * H / mu_ref),
            "Cu": float(getattr(model, "lam", 0.0) * U / H),
            "source": f"analytic-1D {type(model).__name__}", "seed": seed,
        }
        arrays = {"y": flow.y, "u": flow.u, "tau": flow.tau, "gdot": flow.gdot, "G": np.array(flow.G)}
        path = out_dir / f"run_{i:04d}.npz"
        save_run(path, arrays, params)
        paths.append(path)
    return paths


def build_table(
    run_dir: Path,
    features: tuple[str, ...] = FEATURES,
    targets: tuple[str, ...] = TARGETS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load every run, compute its scalars, and stack them into X (n, n_feat) and Y (n, n_targ)."""
    paths = sorted(run_dir.glob("run_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no run_*.npz files in {run_dir}")
    X, Y, ids = [], [], []
    for path in paths:
        run = load_run(path)
        p = run["params"]
        s = channel_scalars(run["y"], run["u"], run["tau"], float(run["G"]), p["H"], p["L"])
        X.append([p[f] for f in features])
        Y.append([s[t] for t in targets])
        ids.append(path.stem)
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float), ids


def split_indices(
    n: int, seed: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> dict[str, np.ndarray]:
    """Disjoint random train/val/test indices; the same seed always gives the same split."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train:n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:]),
    }


def save_dataset(path: Path, X: np.ndarray, Y: np.ndarray, ids: list[str],
                 splits: dict[str, np.ndarray], features=FEATURES, targets=TARGETS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, Y=Y, ids=np.asarray(ids), features=np.asarray(features),
             targets=np.asarray(targets), **{f"idx_{k}": v for k, v in splits.items()})


def load_dataset(path: Path) -> dict:
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    out["splits"] = {k[4:]: out.pop(k) for k in list(out) if k.startswith("idx_")}
    return out
