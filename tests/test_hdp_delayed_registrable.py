"""23-DELAY-01: delayed registrable HDP qualification.

Chain under test: event_{t-d} -> H_t -> P -> Theta_t -> I_t.

Covers: zero-delay parity, one-step exact arrival, asymmetric multi-step
exact arrival, direct-kernel chunked vs uninterrupted continuation across an
in-flight event (noise_scale=0), Model continuation-path chunked vs
uninterrupted (per_edge storage), gain-divergence causality, JIT replay.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered
from jaxfne.emitters import EdgeList, IzhikevichParams


def _two_neuron(delay_steps):
    n = 2
    params = IzhikevichParams(
        a=jnp.full((n,), 0.02),
        b=jnp.full((n,), 0.2),
        c=jnp.full((n,), -65.0),
        d=jnp.full((n,), 8.0),
        drive=jnp.zeros((n,)),
        sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)),
        v0=jnp.full((n,), -65.0),
        u0=jnp.full((n,), -13.0),
        source_scale=jnp.ones((n,)),
        labels=("E", "E"),
        layer_labels=("L4", "L4"),
        source_calibration_status="x",
    )
    edges = EdgeList(
        pre=jnp.array([0], dtype=jnp.int32),
        post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([0.2], dtype=jnp.float32),
        receptor_index=jnp.array([0], dtype=jnp.int32),
        tau_ms=jnp.array([2.0], dtype=jnp.float32),
        delay_steps=jnp.array(delay_steps, dtype=jnp.int32),
        source_calibration_status="x",
    )
    return params, edges


def _three_neuron(delays):
    n = 3
    params = IzhikevichParams(
        a=jnp.full(n, 0.02),
        b=jnp.full(n, 0.20),
        c=jnp.full(n, -65.0),
        d=jnp.full(n, 8.0),
        v0=jnp.full(n, -65.0),
        u0=jnp.full(n, 0.20 * -65.0),
        W=jnp.zeros((n, n)),
        drive=jnp.zeros(n),
        source_scale=jnp.ones(n),
        sign=jnp.ones(n),
        labels=tuple(f"E{i}" for i in range(n)),
    )
    edges = EdgeList(
        pre=jnp.array([0, 0, 1], dtype=jnp.int32),
        post=jnp.array([1, 2, 2], dtype=jnp.int32),
        weight=jnp.array([5.0, 5.0, 5.0], dtype=jnp.float32),
        tau_ms=jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32),
        delay_steps=jnp.array(delays, dtype=jnp.int32),
        receptor_index=jnp.zeros(3, dtype=jnp.int32),
    )
    return params, edges


def test_zero_delay_has_no_ring():
    params, edges = _two_neuron([0])
    drv = jnp.zeros((10, 2), dtype=jnp.float32).at[:, 0].set(12.0)
    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 10, 1.0, jax.random.PRNGKey(0),
        drive_schedule=drv, noise_scale=0.0,
        hdp_rule="synthetic_presyn_gain",
        hdp_rule_params={"k_h": 0.5, "k_w": 0.0, "gamma": 0.0},
    )
    assert "delay_state" not in diag


def test_one_step_delay_exact_arrival():
    for d, expect_shift in [(0, 0), (1, 1), (5, 5)]:
        params, edges = _two_neuron([d])
        ns = 30
        drv = jnp.zeros((ns, 2), dtype=jnp.float32).at[:, 0].set(12.0)
        _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
            params, edges, ns, 1.0, jax.random.PRNGKey(0),
            drive_schedule=drv, noise_scale=0.0,
            hdp_rule="synthetic_presyn_gain",
            hdp_rule_params={"k_h": 0.5, "k_w": 0.0, "gamma": 0.0},
        )
        H = np.asarray(diag["H_trace"])[:, 1]
        Sp = np.asarray(spikes)
        t0 = int(np.argmax(Sp[:, 0] > 0.5))
        hon = int(np.argmax(H > 1.0 + 1e-6))
        assert hon == t0 + expect_shift
    params, edges = _two_neuron([1])
    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 10, 1.0, jax.random.PRNGKey(0),
        drive_schedule=jnp.zeros((10, 2)).at[:, 0].set(12.0), noise_scale=0.0,
        hdp_rule="synthetic_presyn_gain",
        hdp_rule_params={"k_h": 0.5, "k_w": 0.0, "gamma": 0.0},
    )
    assert "delay_state" in diag


def test_asymmetric_multi_step_exact_arrival():
    params, edges = _three_neuron((2, 5, 9))
    ns = 60
    drv = jnp.zeros((ns, 3), dtype=jnp.float32).at[:, 0].set(14.0)
    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, ns, 1.0, jax.random.PRNGKey(1),
        drive_schedule=drv, noise_scale=0.0,
        hdp_rule="synthetic_presyn_gain",
        hdp_rule_params={"k_h": 1.0, "k_w": 0.0, "gamma": 0.0},
    )
    H = np.asarray(diag["H_trace"])
    Sp = np.asarray(spikes)
    t0 = int(np.argmax(Sp[:, 0] > 0.5))
    h1 = int(np.argmax(H[:, 1] > 1.0 + 1e-6))
    h2 = int(np.argmax(H[:, 2] > 1.0 + 1e-6))
    assert h1 == t0 + 2
    assert h2 == t0 + 5


def test_chunked_continuation_across_inflight_event():
    params, edges = _three_neuron((2, 9, 5))
    total, chunk = 40, 10
    kw = {"k_h": 0.3, "k_w": 0.05, "gamma": 0.0}
    full_drv = jnp.zeros((total, 3), dtype=jnp.float32).at[:, 0].set(14.0)
    Vf, Sf, _, diagf = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, total, 1.0, jax.random.PRNGKey(7),
        drive_schedule=full_drv, noise_scale=0.0,
        hdp_rule="synthetic_presyn_gain", hdp_rule_params=kw)
    V1, S1, _, diag1 = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, chunk, 1.0, jax.random.PRNGKey(7),
        drive_schedule=full_drv[:chunk], noise_scale=0.0,
        hdp_rule="synthetic_presyn_gain", hdp_rule_params=kw)
    V2, S2, _, diag2 = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, total - chunk, 1.0, jax.random.PRNGKey(7),
        drive_schedule=full_drv[chunk:], noise_scale=0.0,
        hdp_rule="synthetic_presyn_gain", hdp_rule_params=kw,
        init_state=diag1)
    assert np.array_equal(np.asarray(Sf), np.concatenate([np.asarray(S1), np.asarray(S2)], axis=0))
    assert np.allclose(np.asarray(Vf), np.concatenate([np.asarray(V1), np.asarray(V2)], axis=0), atol=1e-5)
    assert np.allclose(np.asarray(diagf["H_trace"]), np.concatenate([np.asarray(diag1["H_trace"]), np.asarray(diag2["H_trace"])], axis=0), atol=1e-5)
    assert np.array_equal(np.asarray(diagf["delay_state"]), np.asarray(diag2["delay_state"]))


def test_gain_divergence_with_delays():
    def run(kw):
        p, e = _three_neuron((2, 5, 9))
        _, _, s, d = simulate_edge_recurrent_izhikevich_hdp_registered(
            p, e, 40, 1.0, jax.random.PRNGKey(3),
            drive_schedule=jnp.zeros((40, 3)).at[:, 0].set(14.0), noise_scale=0.0,
            hdp_rule="synthetic_presyn_gain",
            hdp_rule_params={"k_h": 0.3, "k_w": kw, "gamma": 0.0})
        return np.asarray(s), np.asarray(d["w_final"])
    s1, w1 = run(0.05)
    s2, w2 = run(0.20)
    assert not np.allclose(w1, w2)
    assert not np.allclose(s1, s2)


def test_jit_deterministic_replay_delayed():
    params, edges = _three_neuron((2, 9, 5))
    drv = jnp.zeros((20, 3)).at[:, 0].set(14.0)
    f = jax.jit(lambda: simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 20, 1.0, jax.random.PRNGKey(9),
        drive_schedule=drv, noise_scale=0.0,
        hdp_rule="synthetic_presyn_gain",
        hdp_rule_params={"k_h": 0.3, "k_w": 0.05, "gamma": 0.0})[1])
    assert np.array_equal(np.asarray(f()), np.asarray(f()))


def test_model_continuation_delayed_registered():
    cfg = jtfne.suite2_net1_config(seed=1, n=4, duration_ms=20.0, dt_ms=1.0).runtime(
        enable_hdp=True, recurrent_backend="edge_list",
        hdp_params={"hdp_rule": "synthetic_presyn_gain",
                    "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                    "noise_scale": 0.0})
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    ds = jnp.full((edges.n_edges,), 3, dtype=jnp.int32)
    object.__setattr__(model, "params", {**model.params, "edge_list": replace(edges, delay_steps=ds, delay_storage="per_edge")})
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                             hdp_params={"hdp_rule": "synthetic_presyn_gain",
                                         "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                                         "noise_scale": 0.0})
    s_full, _ = jtfne.simulate(model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt), return_state=True)
    s1, st1 = jtfne.simulate(model, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=5, runtime=rt), return_state=True)
    assert st1.delay_state is not None
    s2, st2 = jtfne.simulate(model, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=99, runtime=rt), continuation=st1, return_state=True)
    assert jnp.array_equal(jnp.concatenate([s1.V_m, s2.V_m]), s_full.V_m)
    assert jnp.array_equal(jnp.concatenate([s1.spikes, s2.spikes]), s_full.spikes)
