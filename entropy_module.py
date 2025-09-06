from __future__ import annotations
import numpy as np
from typing import Callable

_EPS = 1e-12

def entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("weights must have positive, finite sum")
    w = w / s
    return -np.sum(w * np.log(w + _EPS))

def entropy_gradient(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, _EPS)
    return -(1.0 + np.log(w))

def projected_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if s <= 0:
        raise ValueError("s must be > 0")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    w = w / w.sum()
    return w

def portfolio_variance_gradient(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    Σ = np.asarray(cov_matrix, dtype=float)
    return 2.0 * Σ.dot(w)

def entropy_regularized_update(weights: np.ndarray, grad_L, eta: float = 0.05, gamma: float = 0.05, project: bool = True) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("weights must have positive, finite sum")
    w = np.maximum(w, _EPS)
    w = w / w.sum()

    if callable(grad_L):
        gL = grad_L(w)
    else:
        gL = np.asarray(grad_L, dtype=float)
    gH = entropy_gradient(w)
    update = -eta * (gL - gamma * gH)
    new_w = w * np.exp(update - update.max())
    new_w = new_w / new_w.sum()
    if project:
        new_w = projected_simplex(new_w, s=1.0)
    return new_w
