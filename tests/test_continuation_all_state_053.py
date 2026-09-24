"""0.5.3 ENGINE item 3: chunked == continuous for every mutable state.

P-010 regression battery: plain simulate() vs chained continuation
segments, bit-exact (no tolerance), including delays in flight,
stochastic membrane + rule noise, chunk counts >> 2, and the
plasticity-off == fixed-W identity.

Decomposition under test: C_t = (X, H, W, B, K, A) with X = v/u/
prev_spikes/syn_state (+V/spikes/sources outputs), B = delay_state,
K = carried prng_key chain, A = aux/b/theta_S.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.hdp_rule import (
    HDPRuleDescriptor,
    HDPRuleUpdate,
    list_registered_hdp_rules,
    register_hdp_rule,
)

N = 8
DT = 0.5


def _model(n=N, seed=11):
    cfg = jtfne.suite2_net1_config(seed=seed, n=n, duration_ms=20.0, dt_ms=DT)
    return jtfne.construct(cfg)


def _delayed(model, delay=3):
    edges = model.params["edge_list"]
    new_edges = replace(
        edges,
        delay_steps=jnp.full(int(edges.n_edges), delay, dtype=jnp.int32),
        delay_storage="per_edge",
    )
    object.__setattr__(model, "params", {**model.params, "edge_list": new_edges})
    return model


def _rt_base(noise=None):
    hp = {} if noise is None else {"noise_scale": noise}
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=False,
        hdp_params=dict(hp),
    )


def _rt_hdp(noise=0.0, **over):
    hp = {
        "K_HDP": 0.01,
        "K_ctrl": 0.15,
        "K_w_ctrl": 0.001,
        "tau_0_ms": 20.0,
        "alpha": 0.01,
        "barrier_c": 0.01,
        "barrier_d": 0.01,
        "noise_scale": noise,
    }
    hp.update(over)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )


def _rt_registered(noise=0.0, **over):
    hp = {
        "hdp_rule": "synthetic_presyn_gain",
        "hdp_rule_params": {"k_h": 0.01, "k_w": 0.02, "gamma": 0.0},
        "noise_scale": noise,
    }
    hp.update(over)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )


def _chain(model, runtime, total_ms, chunk_ms, seed=17):
    """Run total_ms as chained chunks; return (full, parts, states)."""
    common = dict(
        dt_ms=DT,
        seed=seed,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
    )
    full, full_state = jtfne.simulate(model, duration_ms=total_ms, return_state=True, **common)
    parts, states = [], []
    state = None
    n_chunks = int(round(total_ms / chunk_ms))
    for i in range(n_chunks):
        kw = dict(common, duration_ms=chunk_ms, return_state=True)
        if state is None:
            sig, state = jtfne.simulate(model, **kw)
        else:
            sig, state = jtfne.simulate(model, continuation=state, **{**kw, "seed": 1000 + i})
        parts.append(sig)
        states.append(state)
    return full, full_state, parts, states


def _assert_concat_equal(full, parts):
    for name in ("V_m", "spikes", "sources"):
        seg = jnp.concatenate([getattr(p, name) for p in parts], axis=0)
        assert jnp.array_equal(getattr(full, name), seg), name


def _assert_dynamic_equal(a, b):
    for key in ("v", "u", "prev_spikes", "syn_state", "H", "w"):
        assert jnp.array_equal(getattr(a.dynamic, key), getattr(b.dynamic, key)), key
    for key in ("theta_S", "aux", "b"):
        assert jnp.array_equal(getattr(a.dynamic, key), getattr(b.dynamic, key)), key
    assert int(a.step_index) == int(b.step_index)
    if a.delay_state is None:
        assert b.delay_state is None
    else:
        assert jnp.array_equal(a.delay_state, b.delay_state)


# --- P-010: baseline stochastic --------------------------------------------


def test_p010_baseline_default_noise_4chunks_exact():
    """P-010: plain vs 4-chunk chained, default live noise — bit-exact."""
    model = _model()
    full, full_state, parts, states = _chain(model, _rt_base(), 12.0, 3.0)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])


@pytest.mark.parametrize("n_chunks", [2, 3, 4, 6, 8, 12])
def test_baseline_stochastic_chunk_sweep_exact(n_chunks):
    model = _model()
    full, full_state, parts, states = _chain(model, _rt_base(), 12.0, 12.0 / n_chunks)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])


def test_baseline_declared_zero_noise_8chunks_exact():
    model = _model()
    full, full_state, parts, states = _chain(model, _rt_base(noise=0.0), 12.0, 1.5)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])


# --- all mutable state: HDP stochastic, many chunks --------------------------


def test_hdp_stochastic_8chunks_all_state_exact():
    model = _model()
    runtime = _rt_hdp(noise=0.2)
    full, full_state, parts, states = _chain(model, runtime, 12.0, 1.5)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])
    d = model.last_hdp_diagnostics()
    assert d is not None
    assert bool(jnp.array_equal(d["H_final"], states[-1].dynamic.H))
    assert bool(jnp.array_equal(d["w_final"], states[-1].dynamic.w))


def test_hdp_deterministic_8chunks_all_state_exact():
    model = _model()
    full, full_state, parts, states = _chain(model, _rt_hdp(noise=0.0), 12.0, 1.5)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])


# --- delays in flight + live noise --------------------------------------------


def test_delayed_baseline_stochastic_in_flight_exact():
    """Delays in flight (d=3, split mid-propagation) + live noise: exact."""
    model = _delayed(_model(), delay=3)
    full, full_state, parts, states = _chain(model, _rt_base(), 12.0, 3.0)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])
    assert full_state.delay_state is not None
    assert tuple(np.asarray(full_state.delay_state).shape) == (4, N)


def test_delayed_hdp_stochastic_in_flight_exact():
    model = _delayed(_model(), delay=3)
    full, full_state, parts, states = _chain(model, _rt_hdp(noise=0.2), 12.0, 3.0)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])
    d = model.last_hdp_diagnostics()
    assert bool(jnp.array_equal(d["H_final"], states[-1].dynamic.H))
    assert bool(jnp.array_equal(d["w_final"], states[-1].dynamic.w))


def test_delayed_registered_rule_in_flight_exact():
    model = _delayed(_model(), delay=2)
    full, full_state, parts, states = _chain(model, _rt_registered(noise=0.2), 12.0, 3.0)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])


# --- registered rules ------------------------------------------------------------


def test_registered_rule_stochastic_membrane_exact():
    model = _model()
    full, full_state, parts, states = _chain(model, _rt_registered(noise=0.2), 12.0, 3.0)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])
    d = model.last_hdp_diagnostics()
    assert bool(jnp.array_equal(d["aux_final"], states[-1].dynamic.aux))


def _ensure_053_stochastic():
    name = "gen053_stochastic_gain"
    if name in list_registered_hdp_rules():
        return name

    def _step(ctx):
        k_w = jnp.asarray(ctx.rule_params.get("k_w", 0.05), dtype=ctx.H.dtype)
        sigma = jnp.asarray(ctx.rule_params.get("sigma", 0.5), dtype=ctx.H.dtype)
        pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
        sub = jax.random.split(ctx.key, 2)[1]
        dz = sigma * jax.random.normal(sub, shape=pre_sp.shape, dtype=ctx.H.dtype)
        dw = k_w * (pre_sp + dz * pre_sp) * jnp.abs(ctx.w)
        return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H), d_theta={"edge_weight": dw})

    register_hdp_rule(
        HDPRuleDescriptor(
            name=name,
            aux_coords=(),
            aux_layout="none",
            default_params={"k_w": 0.05, "sigma": 0.5},
        ),
        _step,
    )
    return name


def test_stochastic_rule_key_stream_plain_vs_chunked_exact():
    """Per-rule RNG stream: plain (chain schedule) == chained, bit-exact,
    with rule noise reaching W (k_w > 0)."""
    name = _ensure_053_stochastic()
    hp = {
        "hdp_rule": name,
        "hdp_rule_params": {"k_w": 0.05, "sigma": 0.5},
        "noise_scale": 0.0,  # isolate the rule stream from membrane noise
    }
    runtime = jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=True,
        hdp_params=hp,
    )
    model = _model()
    full, full_state = jtfne.simulate(
        model,
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=runtime,
        record_sources=True,
        record_fields=False,
        return_state=True,
    )
    d_full = model.last_hdp_diagnostics()
    parts, states = [], []
    state = None
    for i in range(4):
        kw = dict(
            dt_ms=DT,
            seed=17,
            runtime=runtime,
            record_sources=True,
            record_fields=False,
            duration_ms=3.0,
            return_state=True,
        )
        if state is None:
            sig, state = jtfne.simulate(model, **kw)
        else:
            sig, state = jtfne.simulate(model, continuation=state, **{**kw, "seed": 1000 + i})
        parts.append(sig)
        states.append(state)
    _assert_concat_equal(full, parts)
    _assert_dynamic_equal(full_state, states[-1])
    w_full = np.asarray(d_full["w_trace"])
    assert w_full.shape[0] == 24
    # Rule noise actually moved weights (stream is live, not a null).
    assert not np.allclose(w_full[-1], w_full[0])


# --- plasticity-off == fixed-W ----------------------------------------------------


def test_plasticity_off_engaged_frozen_equals_fixed_w():
    """HDP engaged but frozen (K_HDP=0, K_w_ctrl=0, live H gains) vs
    enable_hdp=False: spikes bit-exact, weights bit-untouched, H live,
    V divergence bounded (characterization, not an equality claim).

    Cause (located, pre-existing kernel arithmetic): the two kernels'
    V bodies differ structurally (HDP carries the ``(drive+sched)*boost``
    form + bound-state clip); identical inputs already differ by 1-2 ulp
    at kernel level with all gains zeroed and noise 0, and the v^2
    nonlinearity amplifies the seed (observed max 1.8e-4 over 24 steps
    at dt 0.5; spikes bit-exact throughout). No tolerance is used on any
    chunked==continuous claim. Same-kernel plasticity-off identity
    (identity params -> baseline kernel) is strict bit-exact (see
    test_identity_* below + item 1).
    """
    frozen = _rt_hdp(noise=0.2, K_HDP=0.0, K_w_ctrl=0.0)
    off = jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        enable_hdp=False,
        hdp_params={"noise_scale": 0.2},
    )
    m1, m2 = _model(), _model()
    s_frozen = jtfne.simulate(
        m1,
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=frozen,
        record_sources=True,
        record_fields=False,
    )
    d_frozen = m1.last_hdp_diagnostics()
    s_off = jtfne.simulate(
        m2,
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=off,
        record_sources=True,
        record_fields=False,
    )
    # Threshold behavior identical; weights bit-untouched (the plasticity
    # claim); H live (H != HDP).
    assert jnp.array_equal(s_frozen.spikes, s_off.spikes)
    np.testing.assert_array_equal(
        np.asarray(d_frozen["w_final"]),
        np.asarray(m1.params["edge_list"].weight),
    )
    assert not np.allclose(np.asarray(d_frozen["H_trace"]), 1.0)
    # V/sources: seeded structural ulp amplified by dynamics (see
    # docstring). Characterization bound with margin over the observed
    # 1.8e-4 max — guards against future large divergence, claims no
    # equality.
    assert float(jnp.max(jnp.abs(s_frozen.V_m - s_off.V_m))) < 1e-3
    assert float(jnp.max(jnp.abs(s_frozen.sources - s_off.sources))) < 1e-3


def test_identity_params_routing_stays_bit_exact_live_noise():
    """Same-kernel plasticity-off identity under live noise: identity HDP
    params route to the baseline kernel, bit-exact vs enable_hdp=False."""
    from jaxfne.hdp_rule import hdp_is_engaged

    null_hp = {
        "K_HDP": 0.0,
        "K_ctrl": 0.0,
        "K_w_ctrl": 0.0,
        "alpha": 0.0,
        "barrier_c": 0.0,
        "barrier_d": 0.0,
        "noise_scale": 0.2,
    }
    m1, m2 = _model(), _model()
    assert not hdp_is_engaged(null_hp, m1.params, enable_hdp=True)
    s1 = jtfne.simulate(
        m1,
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=_rt_hdp(**null_hp),
        record_sources=True,
        record_fields=False,
    )
    s2 = jtfne.simulate(
        m2,
        duration_ms=12.0,
        dt_ms=DT,
        seed=17,
        runtime=jtfne.RuntimeConfig(
            dtype="float32",
            recurrent_backend="edge_list",
            enable_hdp=False,
            hdp_params={"noise_scale": 0.2},
        ),
        record_sources=True,
        record_fields=False,
    )
    assert jnp.array_equal(s1.V_m, s2.V_m)
    assert jnp.array_equal(s1.spikes, s2.spikes)
    assert jnp.array_equal(s1.sources, s2.sources)


# --- direct-kernel bulk contract preserved ------------------------------------------


def test_direct_kernel_bulk_default_preserved_and_chain_opt_in():
    """Direct kernel callers keep the legacy bulk draw (None); the chain
    schedule is opt-in and equals chained single steps."""
    from jaxfne import _pipeline
    from jaxfne.emitters import EdgeList, IzhikevichParams
    from jaxfne.emitters import (
        simulate_edge_recurrent_izhikevich as base_kernel,
    )

    n = 8
    p = IzhikevichParams(
        a=jnp.full((n,), 0.02),
        b=jnp.full((n,), 0.2),
        c=jnp.full((n,), -65.0),
        d=jnp.full((n,), 8.0),
        drive=jnp.full((n,), 6.0),
        sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)),
        v0=jnp.full((n,), -65.0),
        u0=jnp.full((n,), -13.0),
        source_scale=jnp.ones((n,)),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="x",
    )
    rng = np.random.default_rng(0)
    edges = EdgeList(
        pre=jnp.asarray(rng.integers(0, n, 32), jnp.int32),
        post=jnp.asarray(rng.integers(0, n, 32), jnp.int32),
        weight=jnp.asarray(rng.normal(0, 0.3, 32).astype(np.float32)),
        receptor_index=jnp.zeros((32,), jnp.int32),
        tau_ms=jnp.full((32,), 5.0),
        source_calibration_status="x",
    )
    key = jax.random.PRNGKey(17)
    # Legacy bulk default still runs.
    V_bulk, _, _, _ = base_kernel(p, edges, 12, DT, key, noise_scale=0.5)
    assert V_bulk.shape == (12, n)
    # Chain opt-in equals chained single steps bit-exactly.
    sched = _pipeline.continuation_noise_schedule(key, 12, n, jnp.float32)
    V_sched, _, _, _ = base_kernel(p, edges, 12, DT, key, noise_schedule=sched, noise_scale=0.5)
    _, keys = _pipeline._advance_prng_key(key, 12)
    parts, init = [], None
    for i in range(12):
        V1, _, _, d1 = base_kernel(p, edges, 1, DT, keys[i], noise_scale=0.5, init_state=init)
        parts.append(np.asarray(V1))
        init = {k: d1[k] for k in ("v", "u", "prev_spikes", "syn_state")}
    np.testing.assert_array_equal(np.asarray(V_sched), np.concatenate(parts))
    # With noise live the two contracts differ (documents the P-010 repair);
    # with noise 0 they agree.
    assert not np.array_equal(np.asarray(V_bulk), np.asarray(V_sched))
    V_b0, _, _, _ = base_kernel(p, edges, 12, DT, key, noise_scale=0.0)
    V_s0, _, _, _ = base_kernel(p, edges, 12, DT, key, noise_schedule=sched, noise_scale=0.0)
    np.testing.assert_array_equal(np.asarray(V_b0), np.asarray(V_s0))


def test_continuation_key_chain_advances_across_8chunks():
    model = _model()
    runtime = _rt_hdp(noise=0.2)
    _, full_state, _, states = _chain(model, runtime, 12.0, 1.5)
    assert int(full_state.step_index) == 24
    assert int(states[-1].step_index) == 24
    keys = [np.asarray(s.prng_key) for s in states]
    for a, b in zip(keys, keys[1:]):
        assert not np.array_equal(a, b)
