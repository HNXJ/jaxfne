"""Verifies jaxfne/_construct.py's _construct_homeostatic_ei_model generalizes
beyond the original hardcoded 2-neuron circuit (added for scripts/
snt_pipeline.py's 8-neuron mode) without changing the N=2 canonical circuit's
behavior, and that simulate_homeostatic_ei's spikes output is edge-triggered
(a genuine pulse count, not a level indicator that inflates to ~2000Hz)."""
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei
import jax


def _build(n):
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=100.0, dt_ms=0.5)
        .network(name=f"ei{n}", n=n)
        .set_emitter("homeostatic_ei")
        .field(domain="none")
        .probe(modes=["vm"])
    )
    return jtfne.construct(cfg)


def test_n2_stays_bit_identical_to_original_hardcoded_defaults():
    model = _build(2)
    emitter = model.params["emitter"]
    assert emitter.labels == ("E", "I")
    assert np.allclose(np.asarray(emitter.x0), [0.1, 0.1])
    assert np.allclose(np.asarray(emitter.G0), [[0.5, -0.5], [0.5, -0.5]])
    assert np.allclose(np.asarray(emitter.H0), [0.3, 0.3])
    assert np.allclose(np.asarray(emitter.drive), [0.5, 0.3])
    assert np.allclose(np.asarray(model.params["positions"]), [[0, 0, 0], [0, 0, 1]])


def test_n8_gives_6e_2i_split_with_normalized_g0():
    model = _build(8)
    emitter = model.params["emitter"]
    assert emitter.labels == ("E", "E", "E", "E", "E", "E", "I", "I")
    assert emitter.x0.shape == (8,)
    assert emitter.G0.shape == (8, 8)
    # Every row identical (uniform incoming connectivity), E columns positive
    # and normalized by E count, I columns negative and normalized by I count.
    row0 = np.asarray(emitter.G0[0])
    assert np.allclose(row0[:6], 0.5 / 6)
    assert np.allclose(row0[6:], -0.5 / 2)
    for r in range(1, 8):
        assert np.allclose(np.asarray(emitter.G0[r]), row0)


def test_n8_model_stays_finite_over_a_real_run():
    model = _build(8)
    sim = jtfne.simulation(duration_ms=2000.0, dt_ms=0.5, seed=0)
    signals = model.simulate(sim)
    assert bool(jnp.all(jnp.isfinite(signals.V_m)))
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(signals.metadata["hdp"]["H_trace"]))))


def test_n1_below_minimum_raises():
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=100.0, dt_ms=0.5)
        .network(name="ei1", n=1)
        .set_emitter("homeostatic_ei")
        .field(domain="none")
        .probe(modes=["vm"])
    )
    import pytest
    with pytest.raises(ValueError, match="at least 2 neurons"):
        jtfne.construct(cfg)


def test_spikes_are_edge_triggered_not_a_level_indicator():
    """A neuron whose x stays positive for many consecutive steps must
    produce ONE spike (on the rising crossing), not one spike per step --
    the pre-fix behavior inflated to a ~2000Hz nonsense rate at N=8."""
    params = HomeostaticEIParams(
        x0=jnp.array([-0.5]),
        G0=jnp.array([[0.0]]),
        H0=jnp.array([1.0]),
        drive=jnp.array([2.0]),  # strong constant positive drive -> x crosses 0 once and stays positive
        tau_x_ms=jnp.array(1.0), tau_G_ms=jnp.array(200.0), tau_H_ms=jnp.array(1000.0),
        G_min=jnp.array(-5.0), G_max=jnp.array(5.0), H_min=jnp.array(0.1), H_max=jnp.array(10.0),
        source_scale=jnp.array([1.0]),
    )
    voltages, spikes, sources, G_hist, H_hist, diag = simulate_homeostatic_ei(
        params, n_steps=200, dt_ms=0.5, key=jax.random.PRNGKey(0),
        activation_rule="linear", noise_scale=0.0, freeze_G=True, freeze_H=True,
    )
    total_spikes = float(jnp.asarray(spikes).sum())
    assert total_spikes <= 2.0, f"expected ~1 edge-triggered spike for a monotonic crossing, got {total_spikes}"
    assert bool(jnp.all(jnp.asarray(voltages)[-50:] > 0.0)), "x should stay positive after crossing (test setup check)"
