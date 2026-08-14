"""Synthetic field generators for Protocol C1 validation."""

from __future__ import annotations

import numpy as np


def make_time_axis(duration_ms: float, dt_ms: float) -> np.ndarray:
    n_steps = int(round(duration_ms / dt_ms))
    return np.arange(n_steps, dtype=np.float64) * float(dt_ms) / 1000.0


def make_positions_1d(n_sites: int, length: float = 1.0) -> np.ndarray:
    z = np.linspace(0.0, length, n_sites, dtype=np.float64)
    return z[:, None]


def make_positions_2d(n_x: int, n_y: int, lx: float = 1.0, ly: float = 1.0) -> np.ndarray:
    xs = np.linspace(0.0, lx, n_x, dtype=np.float64)
    ys = np.linspace(0.0, ly, n_y, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def planar_traveling_wave(
    positions: np.ndarray,
    time_s: np.ndarray,
    *,
    k_vector: np.ndarray,
    frequency_hz: float,
    amplitude: float = 1.0,
    phi0: float = 0.0,
) -> np.ndarray:
    """Phi(r,t) = A cos(k·r - omega t + phi0), shape (T, N)."""
    k = np.asarray(k_vector, dtype=np.float64).reshape(-1)
    r = np.asarray(positions, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    omega = 2.0 * np.pi * float(frequency_hz)
    phase = r @ k - omega * t[:, None] + float(phi0)
    return (float(amplitude) * np.cos(phase)).astype(np.float64)


def synchronous_oscillation(
    positions: np.ndarray,
    time_s: np.ndarray,
    *,
    frequency_hz: float,
    amplitude: float = 1.0,
    phi0: float = 0.0,
) -> np.ndarray:
    """Phi_i(t) = A cos(omega t + phi0) — zero spatial gradient."""
    t = np.asarray(time_s, dtype=np.float64)
    omega = 2.0 * np.pi * float(frequency_hz)
    val = float(amplitude) * np.cos(omega * t + float(phi0))
    n_sites = int(np.asarray(positions).shape[0])
    return np.broadcast_to(val[:, None], (val.shape[0], n_sites)).astype(np.float64)


def random_spatial_phases(
    positions: np.ndarray,
    time_s: np.ndarray,
    *,
    frequency_hz: float,
    amplitude: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Band-limited carrier with random spatial phase per site."""
    rng = np.random.default_rng(seed)
    n_sites = int(np.asarray(positions).shape[0])
    spatial_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_sites)
    t = np.asarray(time_s, dtype=np.float64)
    omega = 2.0 * np.pi * float(frequency_hz)
    return (
        float(amplitude)
        * np.cos(omega * t[:, None] + spatial_phase[None, :])
    ).astype(np.float64)


def noise_only_field(
    positions: np.ndarray,
    time_s: np.ndarray,
    *,
    amplitude: float = 0.05,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.asarray(time_s)
    n_sites = int(np.asarray(positions).shape[0])
    return (rng.normal(0.0, float(amplitude), size=(t.shape[0], n_sites))).astype(np.float64)


def standing_wave_field(
    positions: np.ndarray,
    time_s: np.ndarray,
    *,
    k_scalar: float,
    frequency_hz: float,
    amplitude: float = 1.0,
    axis: int = 0,
) -> np.ndarray:
    """Phi(x,t) = A cos(k x) cos(omega t) along one axis."""
    r = np.asarray(positions, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    x = r[:, axis]
    omega = 2.0 * np.pi * float(frequency_hz)
    spatial = np.cos(float(k_scalar) * x)
    temporal = np.cos(omega * t)
    return (float(amplitude) * temporal[:, None] * spatial[None, :]).astype(np.float64)
