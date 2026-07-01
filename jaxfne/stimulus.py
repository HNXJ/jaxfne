"""Stimulus generation for jaxfne."""

from __future__ import annotations
import jax.numpy as jnp
import numpy as np

def triangular_drive(
    duration_ms: float,
    dt_ms: float,
    freq_hz: float = 6.0,
    amplitude: float = 5.0,
) -> jnp.ndarray:
    """Generates a triangular drive trace.

    Fully deterministic given ``duration_ms``/``dt_ms``/``freq_hz``/``amplitude``
    -- there is no stochastic element, so this takes no PRNG key or seed.

    Args:
        duration_ms: Total stimulus duration in milliseconds.
        dt_ms: Simulation timestep in milliseconds.
        freq_hz: Wave frequency in Hz (default 6.0 Hz).
        amplitude: Peak amplitude.

    Returns:
        jnp.ndarray: Triangular drive signal.
    """
    n_steps = int(duration_ms / dt_ms)
    t = np.arange(n_steps) * dt_ms / 1000.0  # seconds
    period = 1.0 / freq_hz
    # Triangular wave formula centered around 0 in [-amplitude, amplitude]
    phase = t / period - np.floor(t / period + 0.5)
    wave = 2.0 * np.abs(2.0 * phase)
    drive = amplitude * (wave - 1.0)
    return jnp.array(drive)
