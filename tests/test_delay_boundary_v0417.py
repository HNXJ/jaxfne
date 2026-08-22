"""Delay-domain boundary regression tests (v0.4.17 maintenance).

Covers the adversarial delay cases that previously had no coverage:

* negative ``delay_steps`` must be rejected loudly by every entry point
  (they would otherwise index future history slots);
* the ``receptor_exponential`` synaptic kernel has no finite-delay path,
  so nonzero delays must be rejected instead of silently ignored;
* the dense recurrent backend has no finite-delay path, so nonzero edge
  delays must be rejected there too;
* delays larger than the simulated horizon are valid and must produce
  trajectories bit-identical to the disconnected null under a matched
  seed (no presynaptic spike can arrive within the run).
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import simulate_receptor_exponential_izhikevich


def _sim(*, seed=1, duration_ms=10.0, dt_ms=1.0, backend="dense"):
    runtime = None
    if backend == "edge_list":
        runtime = jtfne.RuntimeConfig(
            dtype="float32",
            recurrent_backend="edge_list",
            enable_hdp=False,
            hdp_params={"noise_scale": 0.0},
        )
    return jtfne.Simulation(
        duration_ms=duration_ms, dt_ms=dt_ms, seed=seed, runtime=runtime
    )


def _model_with_delays(*, n: int = 4, delay_steps=None, zero_weights=False):
    cfg = jtfne.suite2_net1_config(seed=7, n=n, duration_ms=20.0, dt_ms=1.0)
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    if delay_steps is not None:
        edges = replace(
            edges,
            delay_steps=jnp.full((edges.n_edges,), int(delay_steps), dtype=jnp.int32),
        )
    if zero_weights:
        edges = replace(edges, weight=jnp.zeros_like(edges.weight))
    object.__setattr__(model, "params", {**model.params, "edge_list": edges})
    return model


def test_dense_backend_rejects_nonzero_delay_steps():
    """The dense kernel has no delay path; nonzero delays must fail loudly."""
    model = _model_with_delays(delay_steps=2)
    with pytest.raises(ValueError, match="no finite-delay path"):
        model.simulate(_sim(backend="dense"))


def test_model_simulate_rejects_negative_delay_steps():
    model = _model_with_delays(delay_steps=-1)
    with pytest.raises(ValueError, match="delay_steps must be >= 0"):
        model.simulate(_sim(backend="edge_list"))


def test_base_edge_kernel_rejects_negative_delay_steps():
    from jaxfne.emitters import simulate_edge_recurrent_izhikevich

    model = _model_with_delays(delay_steps=-1)
    edges = model.params["edge_list"]
    emitter = model.params["emitter"]
    with pytest.raises(ValueError, match="delay_steps must be >= 0"):
        simulate_edge_recurrent_izhikevich(
            emitter, edges, 10, 1.0, jax.random.PRNGKey(0)
        )


def test_receptor_kernel_rejects_negative_delay_steps():
    model = _model_with_delays(delay_steps=-1)
    edges = model.params["edge_list"]
    emitter = model.params["emitter"]
    with pytest.raises(ValueError, match="delay_steps must be >= 0"):
        simulate_receptor_exponential_izhikevich(
            emitter, edges, 10, 1.0, jax.random.PRNGKey(0)
        )


def test_receptor_kernel_rejects_nonzero_delay_steps():
    model = _model_with_delays(delay_steps=3)
    edges = model.params["edge_list"]
    emitter = model.params["emitter"]
    with pytest.raises(ValueError, match="no finite-delay path"):
        simulate_receptor_exponential_izhikevich(
            emitter, edges, 10, 1.0, jax.random.PRNGKey(0)
        )


def test_receptor_kernel_accepts_zero_delay():
    model = _model_with_delays(delay_steps=0)
    edges = model.params["edge_list"]
    emitter = model.params["emitter"]
    out = simulate_receptor_exponential_izhikevich(
        emitter, edges, 10, 1.0, jax.random.PRNGKey(0)
    )
    v = np.asarray(out[0])
    assert v.shape == (10, emitter.v0.shape[0])
    assert bool(np.isfinite(v).all())


def test_zero_edge_model_runs_with_delay_contract_guard():
    """Models with an empty edge list must pass the delay contract guard
    (min()/sum() on a zero-size delay array are vacuous, not violations)."""
    cfg = jtfne.suite2_net1_config(seed=7, n=1, duration_ms=10.0, dt_ms=1.0)
    model = jtfne.construct(cfg)
    sig = model.simulate(_sim(duration_ms=10.0))
    v = np.asarray(sig.V_m)
    assert v.shape == (10, 1)
    assert bool(np.isfinite(v).all())


def test_horizon_exceeding_delay_equals_disconnected_null_bit_exact():
    """Delays beyond the horizon admit no presynaptic arrival, so the run
    must match the disconnected null exactly under the same seed."""
    delayed = _model_with_delays(delay_steps=999).simulate(
        _sim(duration_ms=15.0, backend="edge_list")
    )
    disconnected = _model_with_delays(zero_weights=True).simulate(
        _sim(duration_ms=15.0, backend="edge_list")
    )
    assert np.array_equal(np.asarray(delayed.V_m), np.asarray(disconnected.V_m))
    assert np.array_equal(np.asarray(delayed.spikes), np.asarray(disconnected.spikes))


def test_valid_small_delay_runs_finite_and_differs_from_null():
    """A within-horizon delay must actually route presynaptic spikes."""
    delayed = _model_with_delays(delay_steps=2).simulate(
        _sim(duration_ms=15.0, backend="edge_list")
    )
    disconnected = _model_with_delays(zero_weights=True).simulate(
        _sim(duration_ms=15.0, backend="edge_list")
    )
    v = np.asarray(delayed.V_m)
    assert bool(np.isfinite(v).all())
    assert not np.array_equal(v, np.asarray(disconnected.V_m))
