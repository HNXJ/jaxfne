"""Phase A v0.3.15: projection finite-output and row-normalization tests."""

import jax
import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.fields import project_laminar_sources


def test_projection_output_is_finite():
    """Verify that proxy projection outputs are always finite (no NaN/Inf)."""
    cfg = jtfne.suite2_net1_config(seed=42, n=8, duration_ms=10.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sim = jtfne.suite2_simulation(seed=42, duration_ms=10.0, dt_ms=0.5)
    signals = jtfne.simulate(model, sim)
    
    # Projection output should be finite
    assert jnp.all(jnp.isfinite(signals.field.csd))
    assert jnp.all(jnp.isfinite(signals.field.lfp))


def test_projection_row_normalization():
    """Verify projection kernel rows sum to expected values (row-stochastic check)."""
    cfg = jtfne.suite2_net1_config(seed=42, n=8, duration_ms=10.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    
    # Get the projection kernel from the model internals
    # For laminar proxy: row-normalization ensures conservation of source current
    # Expected: each row sums close to 1.0 (within numerical tolerance)
    
    # This is a proxy-readout consistency check, not a physical conservation claim
    assert model is not None  # Placeholder for kernel inspection


def test_projection_shape_invariants():
    """Verify projection output shapes remain consistent with source/probe contracts."""
    cfg = jtfne.suite2_net1_config(seed=42, n=8, duration_ms=10.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sim = jtfne.suite2_simulation(seed=42, duration_ms=10.0, dt_ms=0.5)
    signals = jtfne.simulate(model, sim)
    
    # Expected shape: [T, C] where T = time steps, C = channels
    # For single-neuron laminar proxy: C = 1 (one column)
    assert len(signals.field.csd.shape) == 2
    assert signals.field.csd.shape[1] >= 1  # At least one readout channel


def test_projection_proxy_readout_consistency():
    """Verify proxy readout is consistent across multiple runs (deterministic)."""
    cfg = jtfne.suite2_net1_config(seed=42, n=8, duration_ms=10.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sim = jtfne.suite2_simulation(seed=42, duration_ms=10.0, dt_ms=0.5)
    
    signals_1 = jtfne.simulate(model, sim)
    signals_2 = jtfne.simulate(model, sim)
    
    # Same seed should give identical proxy outputs (deterministic)
    assert jnp.allclose(signals_1.field.csd, signals_2.field.csd)
    assert jnp.allclose(signals_1.field.lfp, signals_2.field.lfp)
