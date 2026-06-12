"""Unit tests for the new v0.3.34 STDP plasticity and streaming geometry APIs."""

from __future__ import annotations
import jax.numpy as jnp
import pytest
import jaxfne as jtfne

def test_stdp_geometry_generation():
    """Verify that build_stdp_network_geometry returns correct shapes and cell type splits."""
    pos, exc_mask, inh_mask = jtfne.build_stdp_network_geometry(100, seed=42)
    assert pos.shape == (100, 3)
    assert exc_mask.shape == (100,)
    assert inh_mask.shape == (100,)
    assert float(jnp.sum(exc_mask)) == 70.0
    assert float(jnp.sum(inh_mask)) == 30.0
    
    # Test sphere bounds
    dists = jnp.sqrt(jnp.sum(pos ** 2, axis=1))
    assert jnp.all(dists <= 1.0)

def test_stdp_weight_generation():
    """Verify build_initial_stdp_weights signs and diagonals."""
    pos, exc_mask, inh_mask = jtfne.build_stdp_network_geometry(100, seed=42)
    W = jtfne.build_initial_stdp_weights(100, exc_mask, seed=42)
    
    # Excitatory presynaptic columns (exc_mask is True) should be positive or zero
    exc_cols = W[:, exc_mask]
    assert jnp.all(exc_cols >= 0.0)
    
    # Inhibitory presynaptic columns should be negative or zero
    inh_cols = W[:, inh_mask]
    assert jnp.all(inh_cols <= 0.0)
    
    # Diagonal should be all zeros
    assert jnp.all(jnp.diagonal(W) == 0.0)

def test_triangular_drive_wave():
    """Verify shape, amplitude, and frequency of the triangular drive stimulus."""
    wave = jtfne.generate_triangular_drive(duration_ms=1000.0, dt_ms=0.5, freq_hz=6.0, amplitude=5.0)
    assert wave.shape == (2000,)
    assert float(jnp.max(wave)) <= 5.0
    assert float(jnp.min(wave)) >= -5.0

def test_stdp_simulation_run_smoke():
    """Smoke test for running a short stdp simulation chunk."""
    n_neurons = 100
    pos, exc_mask, inh_mask = jtfne.build_stdp_network_geometry(n_neurons, seed=42)
    W = jtfne.build_initial_stdp_weights(n_neurons, exc_mask, seed=42)
    
    v = jnp.zeros(n_neurons) - 65.0
    u = jnp.zeros(n_neurons)
    s = jnp.zeros(n_neurons)
    trace_pre = jnp.zeros(n_neurons)
    
    stim = jnp.zeros((10, n_neurons))
    noise = jnp.zeros((10, n_neurons))
    
    a = jnp.full(n_neurons, 0.02)
    b = jnp.full(n_neurons, 0.2)
    c = jnp.full(n_neurons, -65.0)
    d = jnp.full(n_neurons, 8.0)
    
    state_final, (vm_traj, spk_traj) = jtfne.run_stdp_simulation_chunk(
        v, u, s, trace_pre, W, stim, noise, a, b, c, d, exc_mask, inh_mask,
        dt_ms=0.5, plasticity_scale=0.1
    )
    
    assert vm_traj.shape == (10, n_neurons)
    assert spk_traj.shape == (10, n_neurons)
    assert state_final[4].shape == (n_neurons, n_neurons)
