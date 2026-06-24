"""Dedicated tests for the chunked streaming runner (jaxfne/streaming.py).

test_plasticity_v034.py::test_stdp_simulation_run_smoke already calls
run_stdp_stream, but only with chunk_size_ms covering the entire drive in one
chunk -- the defining behavior of a *streaming* runner (state continuity
across multiple chunks) was never exercised. These tests cover that.
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne


def _build_stdp_inputs(n_neurons=20, n_steps=30, seed=7):
    pos, exc_mask, inh_mask, W = jtfne.make_ei_cloud_network(n_neurons, seed=seed)
    v = jnp.zeros(n_neurons) - 65.0
    u = jnp.zeros(n_neurons)
    s = jnp.zeros(n_neurons)
    trace_pre = jnp.zeros(n_neurons)
    trace_post = jnp.zeros(n_neurons)

    a = jnp.full(n_neurons, 0.02)
    b = jnp.full(n_neurons, 0.2)
    c = jnp.full(n_neurons, -65.0)
    d = jnp.full(n_neurons, 8.0)

    rng = np.random.RandomState(seed)
    stim = jnp.asarray(rng.uniform(0.0, 8.0, size=(n_steps, n_neurons)), dtype=jnp.float32)
    noise = jnp.zeros((n_steps, n_neurons))

    stdp_state = jtfne.STDPState(W=W, trace_pre=trace_pre, trace_post=trace_post)
    plasticity_config = jtfne.STDPPlasticityConfig(A_plus=0.01, A_minus=0.012)
    solver_config = jtfne.SolverConfig(method="euler", dt=0.5)

    return dict(
        v_init=v, u_init=u, s_init=s, stdp_state=stdp_state,
        stim_drive=stim, noise=noise, solver_config=solver_config,
        plasticity_config=plasticity_config, plasticity_scale=0.1,
        exc_mask=exc_mask, inh_mask=inh_mask, a=a, b=b, c=c, d=d,
    )


def test_run_stdp_stream_multi_chunk_matches_single_chunk():
    """Splitting the same drive across multiple chunks must not change the
    final state or trajectory -- chunking is a memory-management detail,
    not a numerical approximation."""
    inputs_single = _build_stdp_inputs()
    inputs_multi = _build_stdp_inputs()  # identical seed -> identical W/stim

    (v1, u1, s1, final1), traj1 = jtfne.run_stdp_stream(
        **inputs_single, chunk_size_ms=30.0, downsample_factor=1,
    )
    (v2, u2, s2, final2), traj2 = jtfne.run_stdp_stream(
        **inputs_multi, chunk_size_ms=5.0, downsample_factor=1,
    )

    assert len(traj1["chunk_summaries"]) == 1
    assert len(traj2["chunk_summaries"]) == 3

    assert jnp.allclose(v1, v2, atol=1e-4)
    assert jnp.allclose(u1, u2, atol=1e-4)
    assert jnp.allclose(s1, s2, atol=1e-4)
    assert jnp.allclose(final1.W, final2.W, atol=1e-4)
    assert jnp.allclose(traj1["vm"], traj2["vm"], atol=1e-4)
    assert jnp.allclose(traj1["spk"], traj2["spk"], atol=1e-4)


def test_run_stdp_stream_chunk_summaries_sequential_and_finite():
    """Chunk summaries must be sequentially indexed and report finite state."""
    inputs = _build_stdp_inputs(n_steps=30)
    (_, _, _, _), traj = jtfne.run_stdp_stream(**inputs, chunk_size_ms=5.0, downsample_factor=1)

    summaries = traj["chunk_summaries"]
    assert [s["chunk_index"] for s in summaries] == list(range(len(summaries)))
    for s in summaries:
        assert s["finite_status"] is True
        assert np.isfinite(s["mean_firing_rate_hz"])
        assert np.isfinite(s["mean_weight"])


def test_run_stdp_stream_downsample_factor_reduces_trajectory_length():
    """downsample_factor must reduce the stored per-chunk trajectory length."""
    inputs = _build_stdp_inputs(n_steps=20)
    (_, _, _, _), traj = jtfne.run_stdp_stream(**inputs, chunk_size_ms=20.0, downsample_factor=4)

    assert traj["vm"].shape[0] == int(np.ceil(20 / 4))


def test_agsdr_legacy_adapter_direct_construction():
    """jtfne.AGSDR() must be directly constructible with default and explicit params."""
    spec = jtfne.AGSDR()
    assert spec.alpha == 0.7
    assert spec.exploration == 0.05
    assert spec.deselect_factor == 2.0

    status = spec.status()
    assert status["optimizer"] == "AGSDR"
    assert status["optimizer_class"] == "blackbox"
    assert status["status"] == "legacy_adapter"

    custom = jtfne.AGSDR(alpha=0.5, exploration=0.2, deselect_factor=3.0)
    custom_status = custom.status()
    assert custom_status["alpha"] == 0.5
    assert custom_status["exploration"] == 0.2
    assert custom_status["deselect_factor"] == 3.0
