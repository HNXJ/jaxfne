"""23-REC-01: strided continuation capture + memory preflight."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._pipeline import (
    compile_step_fn,
    continuation_state_from_model,
    dynamic_state_from_model,
    memory_report,
    run_continuation,
    run_continuation_strided,
)


def _model(**kw):
    cfg = jtfne.suite2_net1_config(seed=1, n=6, duration_ms=20.0, dt_ms=1.0)
    rt = kw.pop("runtime", None)
    if rt is not None:
        cfg = cfg.runtime(**rt)
    return jtfne.construct(cfg)


def _stride_case(*, delays=0, hdp=None, noise=0.0, stride=4, total=20):
    model = _model()
    if delays:
        edges = model.params["edge_list"]
        ds = jnp.full((edges.n_edges,), delays, dtype=jnp.int32)
        object.__setattr__(
            model, "params",
            {**model.params, "edge_list": replace(edges, delay_steps=ds, delay_storage="per_edge")},
        )
    hdp_kwargs = {"record_weight_trace": False}
    if hdp is not None:
        hdp_kwargs.update(hdp)
    step_fn, _ = compile_step_fn(model, dt_ms=1.0, kernel="hdp", **hdp_kwargs)
    n = model.params["emitter"].n_neurons
    sched = jax.random.normal(jax.random.PRNGKey(0), (total, n)) * 2.0
    return model, step_fn, sched, hdp_kwargs


def _states_equal(a, b):
    for x, y in zip(
        jax.tree_util.tree_leaves(a.dynamic), jax.tree_util.tree_leaves(b.dynamic)
    ):
        assert jnp.array_equal(x, y)
    assert jnp.array_equal(a.prng_key, b.prng_key)
    assert int(a.step_index) == int(b.step_index)
    if a.delay_state is None or b.delay_state is None:
        assert a.delay_state is None and b.delay_state is None
    else:
        assert jnp.array_equal(a.delay_state, b.delay_state)


@pytest.mark.parametrize("stride", [2, 4, 7])
def test_strided_frames_match_full_run(stride):
    model, step_fn, sched, _ = _stride_case()
    s0 = continuation_state_from_model(model, seed=3)
    full_state, full_out = run_continuation(step_fn, s0, sched)
    s1 = continuation_state_from_model(model, seed=3)
    end_state, kept, idx = run_continuation_strided(step_fn, s1, sched, stride=stride)
    idx_np = np.asarray(idx)
    assert list(idx_np) == list(range(stride - 1, sched.shape[0], stride)) + (
        [] if sched.shape[0] % stride == 0 else [sched.shape[0] - 1]
    )
    full_np = [np.asarray(o) for o in full_out]
    kept_np = [np.asarray(o) for o in kept]
    for f, k in zip(full_np, kept_np):
        np.testing.assert_array_equal(k, f[idx_np])
    _states_equal(end_state, full_state)


def test_strided_delayed_registered_noisy_exact():
    model, step_fn, sched, _ = _stride_case(
        delays=3,
        hdp={"hdp_rule": "synthetic_presyn_gain",
             "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0}},
        noise=0.5,
        stride=5,
        total=21,
    )
    s0 = continuation_state_from_model(
        model, seed=3, hdp_params={"hdp_rule": "synthetic_presyn_gain"})
    full_state, full_out = run_continuation(step_fn, s0, sched)
    s1 = continuation_state_from_model(
        model, seed=3, hdp_params={"hdp_rule": "synthetic_presyn_gain"})
    end_state, kept, idx = run_continuation_strided(step_fn, s1, sched, stride=5)
    idx_np = np.asarray(idx)
    assert s1.delay_state is not None and end_state.delay_state is not None
    for f, k in zip(full_out, kept):
        np.testing.assert_array_equal(np.asarray(k), np.asarray(f)[idx_np])
    _states_equal(end_state, full_state)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_strided_rejects_bad_stride(bad):
    model, step_fn, sched, _ = _stride_case()
    s0 = continuation_state_from_model(model, seed=3)
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        run_continuation_strided(step_fn, s0, sched, stride=bad)


def test_memory_report_matches_measured():
    model = _model()
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=1,
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    rep = memory_report(model, sim)
    sig = jtfne.simulate(model, sim)
    itemsize = 4
    T, N = 20, 6
    assert rep["components"]["recording.V_m"] == T * N * itemsize
    assert rep["components"]["recording.V_m"] == np.asarray(sig.V_m).nbytes
    assert rep["components"]["recording.spikes"] == np.asarray(sig.spikes).nbytes
    assert rep["components"]["recording.sources"] == np.asarray(sig.sources).nbytes
    assert rep["components"]["delay"] == 0
    dyn = sum(np.asarray(l).nbytes for l in jax.tree_util.tree_leaves(
        dynamic_state_from_model(model)))
    assert rep["components"]["dynamic"] == dyn
    par = sum(np.asarray(l).nbytes for l in jax.tree_util.tree_leaves(model.params))
    assert rep["components"]["persistent"] == par


def test_memory_report_hdp_and_stride():
    cfg = jtfne.suite2_net1_config(seed=1, n=6, duration_ms=20.0, dt_ms=1.0).runtime(
        enable_hdp=True, recurrent_backend="edge_list",
        hdp_params={"hdp_rule": "synthetic_presyn_gain",
                    "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0}})
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    E = int(edges.n_edges)
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=1,
                           runtime=jtfne.RuntimeConfig(
                               recurrent_backend="edge_list", enable_hdp=True,
                               hdp_params={"hdp_rule": "synthetic_presyn_gain",
                                           "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04,
                                                               "gamma": 0.0}}))
    full = memory_report(model, sim)
    assert full["components"]["recording.H_trace"] == 20 * 6 * 4
    assert full["components"]["recording.w_trace"] == 20 * E * 4
    assert "record_weight_trace" in " ".join(full["advice"])
    dec = memory_report(model, sim, {"stride": 4})
    assert dec["kept_frames"] == 5
    assert dec["components"]["recording.V_m"] == 5 * 6 * 4
    assert dec["components"]["recording.w_trace"] == 5 * E * 4
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        memory_report(model, sim, {"stride": 0})


@pytest.mark.parametrize("rule", [None, "synthetic_presyn_gain"])
def test_continuation_weight_toggle_arity_and_equivalence(rule):
    """REC-01 fix: record_weight_trace=False must not crash and must not stack w."""
    model = _model()
    kw: dict = {}
    if rule is not None:
        kw.update({"hdp_rule": rule, "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0}})
    n = model.params["emitter"].n_neurons
    sched = jax.random.normal(jax.random.PRNGKey(0), (12, n)) * 2.0
    hp = {"hdp_rule": rule} if rule is not None else {}
    step_on, _ = compile_step_fn(model, dt_ms=1.0, kernel="hdp",
                                 record_weight_trace=True, **kw)
    step_off, _ = compile_step_fn(model, dt_ms=1.0, kernel="hdp",
                                  record_weight_trace=False, **kw)
    s_on = continuation_state_from_model(model, seed=3, hdp_params=hp)
    s_off = continuation_state_from_model(model, seed=3, hdp_params=hp)
    end_on, out_on = run_continuation(step_on, s_on, sched)
    end_off, out_off = run_continuation(step_off, s_off, sched)
    assert len(out_on) == 5 and len(out_off) == 4
    for a, b in zip(out_on[:4], out_off):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    _states_equal(end_on, end_off)
