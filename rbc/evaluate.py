"""Metrics and plots for surrogates (held-out test set only)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


class Predictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def rel_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-column ‖pred − true‖₂ / ‖true‖₂."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    return np.linalg.norm(pred - true, axis=0) / np.linalg.norm(true, axis=0)


def r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-column coefficient of determination 1 − SS_res / SS_tot."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def parity_plot(pred: np.ndarray, true: np.ndarray, names: list[str], out_png: Path | None = None):
    """One panel per target: prediction vs truth, with the y = x line."""
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    for j, ax in enumerate(np.atleast_1d(axes)):
        lo = min(true[:, j].min(), pred[:, j].min())
        hi = max(true[:, j].max(), pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=1)
        ax.scatter(true[:, j], pred[:, j], s=12)
        ax.set_title(names[j])
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
    fig.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
    return fig


def report(models: dict[str, Predictor], X_test: np.ndarray, Y_test: np.ndarray,
           names: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """{model: {target: {"rel_l2": ..., "r2": ...}}} on the test set."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model in models.items():
        pred = model.predict(X_test)
        e, s = rel_l2(pred, Y_test), r2(pred, Y_test)
        out[model_name] = {t: {"rel_l2": float(e[j]), "r2": float(s[j])} for j, t in enumerate(names)}
    return out
