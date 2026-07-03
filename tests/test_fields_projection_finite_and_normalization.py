"""Phase A v0.3.15: projection finite-output and row-normalization tests."""

import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.fields import project_laminar_sources


def test_projection_density_preserving_default_on_simulated_signals():
    """Default mode (density_preserving) does NOT make kernel rows sum to 1 --
    that property is specific to mode="row_normalize", which is no longer the
    default (it erases source density/attenuation; see fields/proxy.py)."""
    cfg = jtfne.suite2_net1_config(seed=42, n=8, duration_ms=10.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sim = jtfne.suite2_simulation(seed=42, duration_ms=10.0, dt_ms=0.5)
    signals = jtfne.simulate(model, sim)
    assert signals.field is not None
    assert hasattr(signals.field, "kernel")
    kernel = signals.field.kernel
    row_sums = jnp.sum(kernel, axis=1)

    assert jnp.all(jnp.isfinite(row_sums))
    assert signals.field.diagnostics["field_admissibility"]["kernel_normalization_definition"] == "contact_rows_density_preserving"


def test_projection_row_normalization():
    """Verify projection kernel rows sum to 1.0 under mode="row_normalize" (explicit opt-in)."""
    sources = jnp.ones((10, 8), dtype=jnp.float32)
    positions = jnp.zeros((8, 3), dtype=jnp.float32)
    # Give some random relative depths
    positions = positions.at[:, 2].set(jnp.linspace(0.0, 1.0, 8))

    field_out = project_laminar_sources(sources, positions, n_contacts=16, mode="row_normalize")
    kernel_direct = field_out.kernel
    row_sums_direct = jnp.sum(kernel_direct, axis=1)

    assert jnp.all(jnp.isfinite(row_sums_direct))
    assert jnp.allclose(row_sums_direct, 1.0, atol=1e-6)



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
