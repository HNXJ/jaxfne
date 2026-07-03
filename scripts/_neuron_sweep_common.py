"""Shared single-neuron-in-isolation simulate step for the neuron-sweep scripts.

Extracted 2026-07-02 from near-identical duplication between
find_5hz_vmap.py and run_neuron_sweeps.py. Left undecorated (no @jax.jit)
so each caller can apply its own jit/vmap composition strategy — the two
scripts jit at different points (run_neuron_sweeps.py jits this function
directly; find_5hz_vmap.py jits the outer vmapped composition instead),
which is a real, deliberate difference, not something to collapse away.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxfne as jtfne


def simulate_single_step(a_val, b_val, c_val, d_val, drive_val, noise_amp, key,
                          n_steps: int, dt_ms: float):
    params = jtfne.IzhikevichParams(
        a=jnp.array([a_val], dtype=jnp.float32),
        b=jnp.array([b_val], dtype=jnp.float32),
        c=jnp.array([c_val], dtype=jnp.float32),
        d=jnp.array([d_val], dtype=jnp.float32),
        drive=jnp.array([drive_val], dtype=jnp.float32),
        sign=jnp.array([1.0], dtype=jnp.float32),
        W=jnp.zeros((1, 1), dtype=jnp.float32),
        v0=jnp.array([-65.0], dtype=jnp.float32),
        u0=jnp.array([b_val * -65.0], dtype=jnp.float32),
        source_scale=jnp.array([1.0], dtype=jnp.float32),
        labels=("cell",),
    )

    noise_seq = jax.random.normal(key, shape=(n_steps, 1)) * noise_amp

    _, spikes, _ = jtfne.simulate_eig_izhikevich(
        params, n_steps, dt_ms, key, drive_schedule=noise_seq
    )

    return jnp.sum(spikes)
