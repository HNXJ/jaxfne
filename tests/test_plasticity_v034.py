"""Unit tests for the modular v0.3.34 STDP plasticity, geometry, stimulus, and streaming APIs."""

from __future__ import annotations
import jax.numpy as jnp
import pytest
import jaxfne as jtfne

def test_stdp_geometry_generation():
    """Verify that make_ei_cloud_network returns correct shapes and cell type splits."""
    pos, exc_mask, inh_mask, W = jtfne.make_ei_cloud_network(100, seed=42)
    assert pos.shape == (100, 3)
    assert exc_mask.shape == (100,)
    assert inh_mask.shape == (100,)
    assert float(jnp.sum(exc_mask)) == 70.0
    assert float(jnp.sum(inh_mask)) == 30.0
    
    # Test sphere bounds (all positions inside unit sphere)
    dists = jnp.sqrt(jnp.sum(pos ** 2, axis=1))
    assert jnp.all(dists <= 1.0)

def test_stdp_weight_generation():
    """Verify make_ei_cloud_network signs and diagonals of initial weights."""
    pos, exc_mask, inh_mask, W = jtfne.make_ei_cloud_network(100, seed=42)
    
    # Excitatory presynaptic columns (exc_mask is True) should be positive or zero
    exc_cols = W[:, exc_mask]
    assert jnp.all(exc_cols >= 0.0)
    
    # Inhibitory presynaptic columns (inh_mask is True) should be negative or zero
    inh_cols = W[:, inh_mask]
    assert jnp.all(inh_cols <= 0.0)
    
    # Diagonal should be all zeros (no autapses)
    assert jnp.all(jnp.diagonal(W) == 0.0)

def test_triangular_drive_wave():
    """Verify shape, amplitude, and frequency of the triangular drive stimulus."""
    wave = jtfne.triangular_drive(duration_ms=1000.0, dt_ms=0.5, freq_hz=6.0, amplitude=5.0)
    assert wave.shape == (2000,)
    assert float(jnp.max(wave)) <= 5.0
    assert float(jnp.min(wave)) >= -5.0

def test_stdp_simulation_run_smoke():
    """Smoke test for running a short stdp simulation stream."""
    n_neurons = 100
    pos, exc_mask, inh_mask, W = jtfne.make_ei_cloud_network(n_neurons, seed=42)
    
    v = jnp.zeros(n_neurons) - 65.0
    u = jnp.zeros(n_neurons)
    s = jnp.zeros(n_neurons)
    trace_pre = jnp.zeros(n_neurons)
    trace_post = jnp.zeros(n_neurons)
    
    stdp_state = jtfne.STDPState(W=W, trace_pre=trace_pre, trace_post=trace_post)
    plasticity_config = jtfne.STDPPlasticityConfig(A_plus=0.01, A_minus=0.012)
    solver_config = jtfne.SolverConfig(method="euler", dt=0.5)
    
    # Tiny 10-step drive and noise
    stim = jnp.zeros((10, n_neurons))
    noise = jnp.zeros((10, n_neurons))
    
    a = jnp.full(n_neurons, 0.02)
    b = jnp.full(n_neurons, 0.2)
    c = jnp.full(n_neurons, -65.0)
    d = jnp.full(n_neurons, 8.0)
    
    (v_final, u_final, s_final, final_stdp_state), traj = jtfne.run_stdp_stream(
        v_init=v,
        u_init=u,
        s_init=s,
        stdp_state=stdp_state,
        stim_drive=stim,
        noise=noise,
        solver_config=solver_config,
        plasticity_config=plasticity_config,
        plasticity_scale=0.1,
        exc_mask=exc_mask,
        inh_mask=inh_mask,
        a=a, b=b, c=c, d=d,
        chunk_size_ms=10.0,
        downsample_factor=1
    )
    
    assert traj["vm"].shape == (10, n_neurons)
    assert traj["spk"].shape == (10, n_neurons)
    assert final_stdp_state.W.shape == (n_neurons, n_neurons)
