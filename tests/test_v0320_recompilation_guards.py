import jax
import jax.numpy as jnp
import pytest
import warnings
import jaxfne as jtfne
from jaxfne import Model, compilation_registry
from jaxfne.core import Simulation, RuntimeConfig


def _base_model():
    cfg = (
        jtfne.configuration()
        .network(name="test", kind="isolated_neuron", n=4, cell_types={"E": 1.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="p", modes=["spikes", "V_m", "source"], n_contacts=16)
    )
    return jtfne.construct(cfg)


def test_static_cache_verification():
    """Verify that multiple simulation calls with uniform dimensions trigger exactly one compilation."""
    compilation_registry.reset()
    compilation_registry.set_mode("exception")

    model = _base_model()
    
    # Use JIT and enable exception recompilation guard in runtime
    runtime_cfg = RuntimeConfig(jit=True, recompilation_guard="exception")
    
    sim1 = jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=42, runtime=runtime_cfg)
    sim2 = jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=43, runtime=runtime_cfg)
    
    # Run once - should compile (trace count = 1)
    res1 = model.simulate(sim1)
    
    # Run second time with same dimensions - should NOT compile (use cache, trace count stays 1)
    res2 = model.simulate(sim2)
    
    # Verify exactly 1 trace was recorded in registry for simulate JIT path
    assert ("simulate", (1, 16, 4, 10)) in compilation_registry.traced_signatures
    assert compilation_registry.traced_signatures[("simulate", (1, 16, 4, 10))] == 1


def test_dynamic_shape_regression_warning():
    """Verify that shape mutations trigger user warnings under warning mode."""
    compilation_registry.reset()
    compilation_registry.set_mode("warning")

    model = _base_model()
    runtime_cfg = RuntimeConfig(jit=True, recompilation_guard="warning")

    # 1. Compile baseline signature (1, 16, 4, 10)
    sim1 = jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=42, runtime=runtime_cfg)
    model.simulate(sim1)

    # 2. Mutate temporal dimension: duration_ms=20.0 -> n_steps=20. Should trigger UserWarning
    sim2 = jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=42, runtime=runtime_cfg)
    
    with pytest.warns(UserWarning, match="shape mutation compile loop"):
        model.simulate(sim2)

    # Verify both signatures are logged in registry
    assert ("simulate", (1, 16, 4, 10)) in compilation_registry.traced_signatures
    assert ("simulate", (1, 16, 4, 20)) in compilation_registry.traced_signatures


def test_dynamic_shape_regression_exception():
    """Verify that shape mutations raise ValueError under exception mode."""
    compilation_registry.reset()
    compilation_registry.set_mode("exception")

    model = _base_model()
    runtime_cfg = RuntimeConfig(jit=True, recompilation_guard="exception")

    sim1 = jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=42, runtime=runtime_cfg)
    model.simulate(sim1)

    # Mutate Z (contacts count) to 32
    sim2_cfg = (
        jtfne.configuration()
        .network(name="test", kind="isolated_neuron", n=4, cell_types={"E": 1.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="p", modes=["spikes", "V_m", "source"], n_contacts=32)
    )
    model2 = jtfne.construct(sim2_cfg)
    sim2 = jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=42, runtime=runtime_cfg)

    # Cross-examine signature check inside same model/sweeper context
    with pytest.raises(ValueError, match="underwent shape mutation compile loop"):
        model2.simulate(sim2)
