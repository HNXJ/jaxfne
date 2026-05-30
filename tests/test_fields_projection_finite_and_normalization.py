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
    
    # 1. Verify kernel on simulated signals
    sim = jtfne.suite2_simulation(seed=42, duration_ms=10.0, dt_ms=0.5)
    signals = jtfne.simulate(model, sim)
    assert signals.field is not None
    assert hasattr(signals.field, "kernel")
    kernel = signals.field.kernel
    row_sums = jnp.sum(kernel, axis=1)

    # Each row must sum exactly to 1.0 within tolerance <= 1e-6
    assert jnp.all(jnp.isfinite(row_sums))
    assert jnp.allclose(row_sums, 1.0, atol=1e-6)

    # 2. Verify via direct function call
    sources = jnp.ones((10, 8), dtype=jnp.float32)
    positions = jnp.zeros((8, 3), dtype=jnp.float32)
    # Give some random relative depths
    positions = positions.at[:, 2].set(jnp.linspace(0.0, 1.0, 8))

    field_out = project_laminar_sources(sources, positions, n_contacts=16)
    kernel_direct = field_out.kernel
    row_sums_direct = jnp.sum(kernel_direct, axis=1)

    assert jnp.all(jnp.isfinite(row_sums_direct))
    assert jnp.allclose(row_sums_direct, 1.0, atol=1e-6)



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
