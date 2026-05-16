"""Emitter kernels for jaxfne.

Minimal EIG Izhikevich scaffold. Detailed biophysical emitters can enter through
Jaxley bridges later.
"""

from __future__ import annotations

from typing import Dict, Mapping

import jax
import jax.numpy as jnp


def izhikevich_eig_params(n: int, cell_type_fractions: Mapping[str, float]) -> Dict[str, jnp.ndarray]:
    """Create E/PV/SST/VIP-like Izhikevich parameters.

    E: regular spiking-like
    PV: fast-spiking-like
    SST: low-threshold-like
    VIP: currently treated as fast/regular hybrid placeholder
    """
    labels = []
    remaining = n
    items = list(cell_type_fractions.items())
    for i, (name, frac) in enumerate(items):
        count = int(round(n * float(frac))) if i < len(items) - 1 else remaining
        count = max(0, min(count, remaining))
        labels.extend([name] * count)
        remaining -= count
    labels = labels[:n] + ["E"] * max(0, n - len(labels))

    a = []
    b = []
    c = []
    d = []
    drive = []
    sign = []
    for name in labels:
        if name == "E":
            a.append(0.02); b.append(0.20); c.append(-65.0); d.append(8.0); drive.append(5.0); sign.append(1.0)
        elif name in {"PV", "Inl"}:
            a.append(0.10); b.append(0.20); c.append(-65.0); d.append(2.0); drive.append(3.0); sign.append(-1.0)
        elif name in {"SST", "Ing"}:
            a.append(0.02); b.append(0.25); c.append(-65.0); d.append(2.0); drive.append(3.5); sign.append(-1.0)
        else:
            a.append(0.05); b.append(0.20); c.append(-65.0); d.append(4.0); drive.append(3.0); sign.append(-1.0)

    W = _default_eig_connectivity(jnp.asarray(sign))
    return {
        "a": jnp.asarray(a),
        "b": jnp.asarray(b),
        "c": jnp.asarray(c),
        "d": jnp.asarray(d),
        "drive": jnp.asarray(drive),
        "sign": jnp.asarray(sign),
        "W": W,
        "v0": jnp.full((n,), -65.0),
        "u0": jnp.asarray(b) * -65.0,
        "source_scale": jnp.asarray(1.0),
    }


def _default_eig_connectivity(sign: jnp.ndarray) -> jnp.ndarray:
    n = sign.shape[0]
    pre = sign[None, :]
    W = jnp.ones((n, n), dtype=jnp.float32) * pre
    W = W * (1.0 - jnp.eye(n)) / jnp.sqrt(jnp.maximum(1, n))
    return 0.5 * W


def simulate_izhikevich_eig(params: Dict[str, jnp.ndarray], n_steps: int, dt_ms: float, key: jax.Array):
    """Simulate an EIG reduced point-neuron scaffold using lax.scan."""
    a = params["a"]; b = params["b"]; c = params["c"]; d = params["d"]
    drive = params["drive"]; W = params["W"]
    source_scale = params.get("source_scale", jnp.asarray(1.0))

    def step(carry, x):
        v, u, prev_spikes, key = carry
        key, noise_key = jax.random.split(key)
        noise = 0.5 * jax.random.normal(noise_key, shape=v.shape)
        syn = W @ prev_spikes
        I = drive + syn + noise
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = a * (b * v - u)
        v_next = v + dt_ms * dv
        u_next = u + dt_ms * du
        spikes = v_next >= 30.0
        v_reset = jnp.where(spikes, c, v_next)
        u_reset = jnp.where(spikes, u_next + d, u_next)
        source = source_scale * (I + 20.0 * spikes.astype(jnp.float32))
        return (v_reset, u_reset, spikes.astype(jnp.float32), key), (v_reset, spikes.astype(jnp.float32), source)

    init = (params["v0"], params["u0"], jnp.zeros_like(params["v0"]), key)
    _, (V, spikes, sources) = jax.lax.scan(step, init, jnp.arange(n_steps))
    return V, spikes, sources
