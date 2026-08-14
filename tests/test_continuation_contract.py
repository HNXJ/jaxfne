"""Targeted proof matrix for recurrent full-state continuation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne import _pipeline


def _model():
    cfg = jtfne.suite2_net1_config(
        seed=11, n=8, duration_ms=20.0, dt_ms=0.5
    )
    return jtfne.construct(cfg)


def _runtime(*, enable_hdp: bool, noise_scale=None):
    hdp_params = {
        "K_HDP": 0.01,
        "K_ctrl": 0.15,
        "K_w_ctrl": 0.001,
        "tau_0_ms": 20.0,
        "alpha": 0.01,
        "barrier_c": 0.01,
        "barrier_d": 0.01,
        "noise_scale": noise_scale,
    }
    if not enable_hdp:
        hdp_params = {}
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=enable_hdp,
        hdp_params=hdp_params,
    )


def _run_case(*, enable_hdp: bool, noise_scale=None):
    model = _model()
    runtime = _runtime(
        enable_hdp=enable_hdp,
        noise_scale=noise_scale,
    )
    common = dict(
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    full, full_state = jtfne.simulate(
        model, **{**common, "duration_ms": 12.0, "return_state": True}
    )
    first, first_state = jtfne.simulate(
        model, **{**common, "return_state": True}
    )
    second, second_state = jtfne.simulate(
        model,
        **{
            **common,
            "seed": 999,
            "continuation": first_state,
            "return_state": True,
        },
    )
    return full, full_state, first, first_state, second, second_state


def _assert_segmented_matches(full, first, second, full_state, second_state):
    for full_arr, segmented_arr in (
        (full.V_m, jnp.concatenate((first.V_m, second.V_m), axis=0)),
        (full.spikes, jnp.concatenate((first.spikes, second.spikes), axis=0)),
        (full.sources, jnp.concatenate((first.sources, second.sources), axis=0)),
    ):
        assert jnp.array_equal(full_arr, segmented_arr)
    assert jnp.array_equal(full_state.dynamic.v, second_state.dynamic.v)
    assert jnp.array_equal(full_state.dynamic.u, second_state.dynamic.u)
    assert jnp.array_equal(
        full_state.dynamic.prev_spikes, second_state.dynamic.prev_spikes
    )
    assert jnp.array_equal(
        full_state.dynamic.syn_state, second_state.dynamic.syn_state
    )
    assert jnp.array_equal(full_state.dynamic.H, second_state.dynamic.H)
    assert jnp.array_equal(full_state.dynamic.w, second_state.dynamic.w)
    assert second_state.step_index == full_state.step_index == 24


@pytest.mark.parametrize(
    ("enable_hdp", "noise_scale"),
    (
        (True, 0.0),    # C2 deterministic HDP
        (False, None),  # C3 stochastic baseline; baseline default noise is active
        (True, 0.2),    # C4 stochastic HDP
    ),
)
def test_full_state_continuation_matrix(enable_hdp, noise_scale):
    result = _run_case(
        enable_hdp=enable_hdp,
        noise_scale=noise_scale,
    )
    _assert_segmented_matches(
        result[0], result[2], result[4], result[1], result[5]
    )


@pytest.mark.parametrize(
    "ablation",
    ("E_silence", "I_silence", "disconnected_null"),
)
def test_continuation_rejects_unsupported_ablations(ablation):
    with pytest.raises(ValueError, match="does not support ablation"):
        jtfne.simulate(
            _model(),
            duration_ms=1.0,
            dt_ms=0.5,
            seed=17,
            runtime=_runtime(enable_hdp=False),
            ablation=ablation,
            record_fields=False,
            return_state=True,
        )


def test_continuation_accepts_exponential_synaptic_kernel():
    signals, state = jtfne.simulate(
        _model(),
        duration_ms=1.0,
        dt_ms=0.5,
        seed=17,
        runtime=_runtime(enable_hdp=False),
        record_fields=False,
        return_state=True,
    )
    assert signals.V_m.shape[0] == 2
    assert state.step_index == 2


@pytest.mark.parametrize("enable_hdp", (False, True))
def test_continuation_rejects_receptor_exponential_kernel(enable_hdp):
    runtime = _runtime(enable_hdp=enable_hdp, noise_scale=0.0)
    runtime = jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        synaptic_kernel="receptor_exponential",
        enable_hdp=enable_hdp,
        hdp_params=runtime.hdp_params,
    )
    with pytest.raises(ValueError, match="supports only synaptic_kernel"):
        jtfne.simulate(
            _model(),
            duration_ms=1.0,
            dt_ms=0.5,
            seed=17,
            runtime=runtime,
            record_fields=False,
            return_state=True,
        )


def test_deterministic_baseline_continuation_at_step_interface():
    model = _model()
    step_fn, init = _pipeline.compile_step_fn(
        model,
        dt_ms=0.5,
        kernel="baseline",
        noise_scale=0.0,
    )
    schedule_full = jnp.zeros((24, init.dynamic.v.shape[0]), dtype=init.dynamic.v.dtype)
    schedule_half = schedule_full[:12]
    initial = init._replace(prng_key=jax.random.PRNGKey(17))
    full_state, full_outputs = _pipeline.run_continuation(
        step_fn, initial, schedule_full
    )
    first_state, first_outputs = _pipeline.run_continuation(
        step_fn, initial, schedule_half
    )
    second_state, second_outputs = _pipeline.run_continuation(
        step_fn, first_state, schedule_half
    )
    for full_arr, segmented_arr in zip(
        full_outputs,
        tuple(
            jnp.concatenate((first_arr, second_arr), axis=0)
            for first_arr, second_arr in zip(first_outputs, second_outputs)
        ),
    ):
        assert jnp.array_equal(full_arr, segmented_arr)
    assert jnp.array_equal(full_state.dynamic.v, second_state.dynamic.v)
    assert jnp.array_equal(full_state.dynamic.syn_state, second_state.dynamic.syn_state)


def test_partial_state_negative_control_is_deterministic():
    """Only v/u/previous-spikes/synaptic-state are reset; H and w match."""
    model = _model()
    runtime = _runtime(enable_hdp=True, noise_scale=0.0)
    full, _, first, first_state, second, _ = _run_case(
        enable_hdp=True,
        noise_scale=0.0,
    )
    cold_dynamic = _pipeline.dynamic_state_from_model(model)
    partial_state = first_state._replace(
        dynamic=cold_dynamic._replace(
            H=first_state.dynamic.H,
            w=first_state.dynamic.w,
        )
    )
    partial, _ = jtfne.simulate(
        model,
        runtime=runtime,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=999,
        record_sources=True,
        record_fields=False,
        continuation=partial_state,
        return_state=True,
    )
    assert not jnp.array_equal(partial.V_m, second.V_m)


def test_legacy_h_w_initialization_remains_compatible():
    model = _model()
    runtime = _runtime(enable_hdp=True, noise_scale=0.0)
    _, first_state = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    partial_model = model.with_hdp_initial_state(
        H0=first_state.dynamic.H[:],
        w0=first_state.dynamic.w[:],
    )
    assert jnp.array_equal(partial_model.params["hdp_initial_H"], first_state.dynamic.H)
    assert jnp.array_equal(partial_model.params["hdp_initial_w"], first_state.dynamic.w)
    partial = jtfne.simulate(
        partial_model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    assert partial.V_m.shape == (12, model.params["emitter"].n_neurons)


def test_mismatched_prng_continuation_is_a_negative_control():
    full, full_state, first, first_state, second, _ = _run_case(
        enable_hdp=False,
        noise_scale=None,
    )
    bad_state = first_state._replace(prng_key=jax.random.PRNGKey(9999))
    bad, _ = jtfne.simulate(
        _model(),
        duration_ms=6.0,
        dt_ms=0.5,
        seed=17,
        runtime=_runtime(enable_hdp=False),
        record_sources=True,
        record_fields=False,
        continuation=bad_state,
        return_state=True,
    )
    assert not jnp.array_equal(bad.V_m, second.V_m)
    assert not jnp.array_equal(
        jnp.concatenate((first.V_m, bad.V_m), axis=0),
        full.V_m,
    )


def test_continuation_state_is_a_jax_pytree():
    _, state, *_ = _run_case(enable_hdp=True, noise_scale=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert leaves
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(restored.dynamic.v, state.dynamic.v)
    assert jnp.array_equal(restored.prng_key, state.prng_key)


def test_continuation_carrier_preserves_trailing_h_state_shape():
    dynamic = _pipeline.DynamicState(
        v=jnp.zeros((3,)),
        u=jnp.zeros((3,)),
        prev_spikes=jnp.zeros((3,)),
        syn_state=jnp.zeros((2,)),
        H=jnp.zeros((3, 2)),
        w=jnp.zeros((2,)),
    )
    state = jtfne.ContinuationState(
        dynamic=dynamic,
        prng_key=jax.random.PRNGKey(0),
    )
    assert state.dynamic.H.shape == (3, 2)
