"""Surrogates: (design parameters) -> (scalar targets).

Two models with the same `.predict(X)` interface:
- MLPSurrogate: small neural network trained with Adam, best-validation checkpoint.
- GPSurrogate: one Gaussian process per target (scikit-learn) — the baseline to beat.
Both work in a transformed space: optional log10, then standardization to zero mean / unit std.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from torch import nn


@dataclass
class Transform:
    """Column-wise: log10 where `log` is True, then (z - mean) / std."""

    log: np.ndarray
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))

    def _log(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float).copy()
        if np.any(A[:, self.log] <= 0.0):
            raise ValueError("log transform needs strictly positive values")
        A[:, self.log] = np.log10(A[:, self.log])
        return A

    def fit(self, A: np.ndarray) -> "Transform":
        Z = self._log(A)
        self.mean = Z.mean(axis=0)
        self.std = Z.std(axis=0)
        self.std[self.std == 0.0] = 1.0
        return self

    def forward(self, A: np.ndarray) -> np.ndarray:
        return (self._log(A) - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        A = np.asarray(Z, dtype=float) * self.std + self.mean
        A[:, self.log] = 10.0 ** A[:, self.log]
        return A


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = n_in
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.SiLU()]
            size = width
        layers.append(nn.Linear(size, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPSurrogate:
    net: MLP
    tx: Transform
    ty: Transform
    history: dict[str, list[float]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.as_tensor(self.tx.forward(X), dtype=torch.float32)).numpy()
        return self.ty.inverse(z.astype(float))


def train_mlp(
    X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray,
    log_x: np.ndarray, log_y: np.ndarray, *,
    width: int = 64, depth: int = 2, epochs: int = 3000, lr: float = 3e-3,
    weight_decay: float = 0.0, seed: int = 0,
) -> MLPSurrogate:
    """Full-batch Adam on the MSE in transformed space; keeps the weights with the lowest val loss."""
    torch.manual_seed(seed)
    tx = Transform(np.asarray(log_x, dtype=bool)).fit(X_train)
    ty = Transform(np.asarray(log_y, dtype=bool)).fit(Y_train)
    xt, yt = (torch.as_tensor(a, dtype=torch.float32) for a in (tx.forward(X_train), ty.forward(Y_train)))
    xv, yv = (torch.as_tensor(a, dtype=torch.float32) for a in (tx.forward(X_val), ty.forward(Y_val)))

    net = MLP(X_train.shape[1], Y_train.shape[1], width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val, best_state = float("inf"), copy.deepcopy(net.state_dict())

    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            val = loss_fn(net(xv), yv).item()
        history["train"].append(loss.item())
        history["val"].append(val)
        if val < best_val:
            best_val, best_state = val, copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    return MLPSurrogate(net=net, tx=tx, ty=ty, history=history)


@dataclass
class GPSurrogate:
    gps: list[GaussianProcessRegressor]
    tx: Transform
    ty: Transform

    def predict(self, X: np.ndarray, return_std: bool = False):
        Z = self.tx.forward(X)
        if not return_std:
            return self.ty.inverse(np.column_stack([gp.predict(Z) for gp in self.gps]))
        mean_std = [gp.predict(Z, return_std=True) for gp in self.gps]
        mean = self.ty.inverse(np.column_stack([m for m, _ in mean_std]))
        std_z = np.column_stack([s for _, s in mean_std])  # std in the standardized (and maybe log) space
        return mean, std_z


def fit_gp(X_train: np.ndarray, Y_train: np.ndarray, log_x: np.ndarray, log_y: np.ndarray,
           seed: int = 0) -> GPSurrogate:
    """One GP per target: constant × anisotropic RBF + white noise, hyperparameters by max likelihood."""
    tx = Transform(np.asarray(log_x, dtype=bool)).fit(X_train)
    ty = Transform(np.asarray(log_y, dtype=bool)).fit(Y_train)
    Zx, Zy = tx.forward(X_train), ty.forward(Y_train)
    kernel = (ConstantKernel(1.0, (1e-3, 1e7)) * RBF(np.ones(Zx.shape[1]), (1e-2, 1e3))
              + WhiteKernel(1e-6, (1e-12, 1e-1)))
    gps = []
    for j in range(Zy.shape[1]):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=seed)
        # Near-noise-free power-law targets drive the noise to its lower bound and L-BFGS reports
        # "ABNORMAL" line-search exits; the fitted model is still fine (checked on the test set).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gps.append(gp.fit(Zx, Zy[:, j]))
    return GPSurrogate(gps=gps, tx=tx, ty=ty)
