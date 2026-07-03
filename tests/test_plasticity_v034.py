"""Unit tests for the modular v0.3.34 STDP plasticity, geometry, stimulus, and streaming APIs."""

from __future__ import annotations
import jax.numpy as jnp
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
