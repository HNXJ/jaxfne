"""
Tests for v0.3.3 API extensions:
1. simulate_dynamic_ei_coupling() — dynamic E/I coupling with correct carry state
2. with_emitter_parameters() — per-neuron parameter overrides with explicit None checks
3. Model.with_recurrent_coupling() — frozen dataclass immutability with replace()

16 tests total.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import (
    izhikevich_eig_params,
    simulate_dynamic_ei_coupling,
)


# ============================================================================
# Fixture: two-neuron E/I model
# ============================================================================

@pytest.fixture
def two_neuron_model():
    cfg = (
        jtfne.configuration()
        .network(name="test_ei", kind="coupled_neurons", n=2,
                 cell_types={"E": 0.5, "I": 0.5})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="test_probe", modes=["spikes", "V_m"], n_contacts=4)
    )
    return jtfne.construct(cfg)


@pytest.fixture
def two_neuron_params():
    return izhikevich_eig_params(2, {"E": 0.5, "I": 0.5})


# ============================================================================
# Group 1: simulate_dynamic_ei_coupling — carry state correctness (5 tests)
# ============================================================================

def test_dynamic_coupling_returns_four_arrays(two_neuron_params):
    """simulate_dynamic_ei_coupling returns (voltages, spikes, syn_currents, sources)."""
    key = jax.random.PRNGKey(0)
    result = simulate_dynamic_ei_coupling(two_neuron_params, n_steps=100, dt_ms=0.1, key=key)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"


def test_dynamic_coupling_output_shapes(two_neuron_params):
    """Output shapes are (n_steps, n_neurons) for all four arrays."""
    key = jax.random.PRNGKey(1)
    n_steps = 200
    voltages, spikes, syn_currents, sources = simulate_dynamic_ei_coupling(
        two_neuron_params, n_steps=n_steps, dt_ms=0.1, key=key
    )
    assert voltages.shape == (n_steps, 2)
    assert spikes.shape == (n_steps, 2)
    assert syn_currents.shape == (n_steps, 2)
    assert sources.shape == (n_steps, 2)


def test_dynamic_coupling_syn_traces_evolve(two_neuron_params):
    """Synaptic currents change over time (not all zero after first spike)."""
    key = jax.random.PRNGKey(2)
    voltages, spikes, syn_currents, sources = simulate_dynamic_ei_coupling(
        two_neuron_params, n_steps=1000, dt_ms=0.1, key=key,
        g_ei=10.0, g_ie=5.0
    )
    # After sufficient time, at least one neuron should spike and produce nonzero syn current
    assert float(jnp.max(jnp.abs(syn_currents))) > 0.0, \
        "syn_currents are all zero — likely carry state bug (syn_traces not in carry)"


def test_dynamic_coupling_voltages_finite(two_neuron_params):
    """All voltage values must be finite."""
    key = jax.random.PRNGKey(3)
    voltages, _, _, _ = simulate_dynamic_ei_coupling(
        two_neuron_params, n_steps=500, dt_ms=0.1, key=key
    )
    assert bool(jnp.all(jnp.isfinite(voltages))), "Voltages contain NaN or Inf"


def test_dynamic_coupling_deterministic(two_neuron_params):
    """Same key → same outputs (deterministic PRNG)."""
    key = jax.random.PRNGKey(42)
    v1, s1, c1, src1 = simulate_dynamic_ei_coupling(
        two_neuron_params, n_steps=100, dt_ms=0.1, key=key
    )
    v2, s2, c2, src2 = simulate_dynamic_ei_coupling(
        two_neuron_params, n_steps=100, dt_ms=0.1, key=key
    )
    assert jnp.allclose(v1, v2), "Outputs not deterministic for same key"
    assert jnp.allclose(s1, s2), "Spikes not deterministic for same key"


# ============================================================================
# Group 2: with_emitter_parameters per-neuron overrides (6 tests)
# ============================================================================

def test_per_neuron_a_override(two_neuron_model):
    """a_per_neuron overrides each neuron independently."""
    a_vals = jnp.array([0.01, 0.05])
    new_model = jtfne.with_emitter_parameters(two_neuron_model, a_per_neuron=a_vals)
    emitter = new_model.params["emitter"]
    assert float(emitter.a[0]) == pytest.approx(0.01, rel=1e-5)
    assert float(emitter.a[1]) == pytest.approx(0.05, rel=1e-5)


def test_per_neuron_drive_override(two_neuron_model):
    """drive_per_neuron sets absolute per-neuron drive (not scaled)."""
    drive_vals = jnp.array([0.0, 7.0])
    new_model = jtfne.with_emitter_parameters(two_neuron_model, drive_per_neuron=drive_vals)
    emitter = new_model.params["emitter"]
    assert float(emitter.drive[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(emitter.drive[1]) == pytest.approx(7.0, rel=1e-5)


def test_zero_array_per_neuron_a_not_falsy(two_neuron_model):
    """a_per_neuron=zeros array is correctly applied (not treated as falsy None)."""
    a_vals = jnp.zeros(2)
    new_model = jtfne.with_emitter_parameters(two_neuron_model, a_per_neuron=a_vals)
    emitter = new_model.params["emitter"]
    assert float(emitter.a[0]) == pytest.approx(0.0, abs=1e-7), \
        "Zero array should set a=0, not fall through to original value"
    assert float(emitter.a[1]) == pytest.approx(0.0, abs=1e-7)


def test_scalar_and_per_neuron_priority(two_neuron_model):
    """Per-neuron array takes priority over scalar when both provided."""
    a_vals = jnp.array([0.01, 0.02])
    new_model = jtfne.with_emitter_parameters(
        two_neuron_model, a=0.99, a_per_neuron=a_vals
    )
    emitter = new_model.params["emitter"]
    # per_neuron wins over scalar
    assert float(emitter.a[0]) == pytest.approx(0.01, rel=1e-5)
    assert float(emitter.a[1]) == pytest.approx(0.02, rel=1e-5)


def test_original_model_not_mutated_per_neuron(two_neuron_model):
    """Original model is not mutated when per-neuron overrides are applied."""
    original_a = float(two_neuron_model.params["emitter"].a[0])
    a_vals = jnp.array([0.001, 0.002])
    _ = jtfne.with_emitter_parameters(two_neuron_model, a_per_neuron=a_vals)
    assert float(two_neuron_model.params["emitter"].a[0]) == pytest.approx(
        original_a, rel=1e-5
    ), "Original model was mutated"


def test_drive_per_neuron_overrides_drive_scale(two_neuron_model):
    """drive_per_neuron takes priority over drive_scale when both specified."""
    drive_vals = jnp.array([3.0, 6.0])
    new_model = jtfne.with_emitter_parameters(
        two_neuron_model, drive_scale=10.0, drive_per_neuron=drive_vals
    )
    emitter = new_model.params["emitter"]
    assert float(emitter.drive[0]) == pytest.approx(3.0, rel=1e-5)
    assert float(emitter.drive[1]) == pytest.approx(6.0, rel=1e-5)


# ============================================================================
# Group 3: Model.with_recurrent_coupling — frozen dataclass safety (5 tests)
# ============================================================================

def test_with_recurrent_coupling_returns_new_model(two_neuron_model):
    """with_recurrent_coupling() returns a new Model instance."""
    new_model = two_neuron_model.with_recurrent_coupling(g_ei=5.0, g_ie=3.0)
    assert new_model is not two_neuron_model


def test_with_recurrent_coupling_stores_params(two_neuron_model):
    """Coupling parameters are stored in static['recurrent_coupling']."""
    new_model = two_neuron_model.with_recurrent_coupling(
        g_ei=7.0, g_ie=4.0, tau_syn_e_ms=3.0, tau_syn_i_ms=8.0
    )
    coupling = new_model.static["recurrent_coupling"]
    assert coupling["g_ei"] == pytest.approx(7.0)
    assert coupling["g_ie"] == pytest.approx(4.0)
    assert coupling["tau_syn_e_ms"] == pytest.approx(3.0)
    assert coupling["tau_syn_i_ms"] == pytest.approx(8.0)


def test_with_recurrent_coupling_does_not_mutate_original(two_neuron_model):
    """Original model.static is unchanged after with_recurrent_coupling()."""
    assert "recurrent_coupling" not in two_neuron_model.static
    _ = two_neuron_model.with_recurrent_coupling(g_ei=5.0, g_ie=3.0)
    assert "recurrent_coupling" not in two_neuron_model.static, \
        "Original model was mutated"


def test_with_recurrent_coupling_claim_gates(two_neuron_model):
    """Coupling params include correct claim gates."""
    new_model = two_neuron_model.with_recurrent_coupling()
    coupling = new_model.static["recurrent_coupling"]
    assert coupling["physical_amplitude_claim_allowed"] is False
    assert coupling["claim_level"] == "computational_scaffold"
    assert coupling["source_calibration_status"] == "uncalibrated_izhikevich_native_current"


def test_with_recurrent_coupling_chaining(two_neuron_model):
    """Method chaining: with_emitter_parameters().with_recurrent_coupling() works."""
    new_model = (
        jtfne.with_emitter_parameters(two_neuron_model, a=0.02)
        .with_recurrent_coupling(g_ei=5.0, g_ie=3.0)
    )
    assert float(new_model.params["emitter"].a[0]) == pytest.approx(0.02)
    assert "recurrent_coupling" in new_model.static