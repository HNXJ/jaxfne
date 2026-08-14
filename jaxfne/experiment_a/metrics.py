"""Quantitative metrics for Experiment A (shared with runner)."""

from __future__ import annotations

import numpy as np


def r90_1d(weights: np.ndarray, z: np.ndarray, center: float) -> float:
    w = np.abs(np.asarray(weights, dtype=np.float64))
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    dist = np.abs(np.asarray(z, dtype=np.float64) - float(center))
    order = np.argsort(dist)
    csum = np.cumsum(w[order])
    hit = np.searchsorted(csum, 0.9 * total, side="left")
    hit = min(int(hit), len(order) - 1)
    return float(dist[order[hit]])


def mean_r90(kernel: np.ndarray, z: np.ndarray, contacts: np.ndarray) -> float:
    vals = [r90_1d(kernel[p], z, float(contacts[p])) for p in range(kernel.shape[0])]
    return float(np.nanmean(vals))


def max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b) / denom)
