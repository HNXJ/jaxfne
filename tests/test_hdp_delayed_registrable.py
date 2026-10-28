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
import pytest

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


def test_model_continuation_delayed_registered_noisy():
    """23-STOCH-01: delay + registered-HDP + stochastic continuation composes."""
    cfg = jtfne.suite2_net1_config(seed=1, n=4, duration_ms=20.0, dt_ms=1.0).runtime(
        enable_hdp=True, recurrent_backend="edge_list",
        hdp_params={"hdp_rule": "synthetic_presyn_gain",
                    "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                    "noise_scale": 0.5})
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    ds = jnp.full((edges.n_edges,), 3, dtype=jnp.int32)
    object.__setattr__(model, "params", {**model.params, "edge_list": replace(edges, delay_steps=ds, delay_storage="per_edge")})
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                             hdp_params={"hdp_rule": "synthetic_presyn_gain",
                                         "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                                         "noise_scale": 0.5})
    s_full, _ = jtfne.simulate(model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt), return_state=True)
    s1, st1 = jtfne.simulate(model, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=5, runtime=rt), return_state=True)
    assert st1.delay_state is not None
    s2, _ = jtfne.simulate(model, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=99, runtime=rt), continuation=st1, return_state=True)
    assert jnp.array_equal(jnp.concatenate([s1.V_m, s2.V_m]), s_full.V_m)
    assert jnp.array_equal(jnp.concatenate([s1.spikes, s2.spikes]), s_full.spikes)


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


# --- SpikeTrainInput reader clamp (HDP-READER-01) -----------------------------
# Explicit per-neuron forced spikes through Model.simulate on the Izhikevich
# edge-list registered-rule kernel: forced bins spike with the membrane held
# at reset, every other bin matches the unforced run, and every other route
# or combination refuses loudly.


def _silent_two_neuron(delay_steps):
    params, edges = _two_neuron(delay_steps)
    return params, edges


def test_forced_spikes_clamp_zero_delay():
    params, edges = _silent_two_neuron([0])
    ns = 10
    drv = jnp.zeros((ns, 2), dtype=jnp.float32)
    kw = dict(hdp_rule="synthetic_presyn_gain",
              hdp_rule_params={"k_h": 0.5, "k_w": 0.0, "gamma": 0.0},
              noise_scale=0.0)
    _, S_base, _, _ = simulate_edge_recurrent_izhikevich_hdp_registered(params, edges, ns, 1.0, jax.random.PRNGKey(0),
                           drive_schedule=drv, **kw)
    assert float(np.asarray(S_base).sum()) == 0.0, "fixture must be silent"
    force = np.zeros((ns, 2), dtype=bool)
    force[3, 0] = True
    force[7, 1] = True
    V, S, sources, _ = simulate_edge_recurrent_izhikevich_hdp_registered(params, edges, ns, 1.0, jax.random.PRNGKey(0),
                            drive_schedule=drv, forced_spikes=force, **kw)
    S = np.asarray(S)
    V = np.asarray(V)
    assert S[3, 0] == 1.0 and S[7, 1] == 1.0
    assert V[3, 0] == -65.0 and V[7, 1] == -65.0
    keep = np.ones((ns, 2), dtype=bool)
    keep[3, 0] = False
    keep[7, 1] = False
    assert np.array_equal(S[keep], np.asarray(S_base)[keep])
    # The forced spike enters the synaptic fan-out: sources move at the
    # forced bins only.
    assert not np.array_equal(np.asarray(sources)[3], np.asarray(sources)[4])


def test_forced_spikes_clamp_delayed():
    params, edges = _silent_two_neuron([2])
    ns = 10
    drv = jnp.zeros((ns, 2), dtype=jnp.float32)
    kw = dict(hdp_rule="synthetic_presyn_gain",
              hdp_rule_params={"k_h": 0.5, "k_w": 0.0, "gamma": 0.0},
              noise_scale=0.0)
    _, S_base, _, _ = simulate_edge_recurrent_izhikevich_hdp_registered(params, edges, ns, 1.0, jax.random.PRNGKey(0),
                           drive_schedule=drv, **kw)
    assert float(np.asarray(S_base).sum()) == 0.0, "fixture must be silent"
    force = np.zeros((ns, 2), dtype=bool)
    force[3, 0] = True
    V, S, _, _ = simulate_edge_recurrent_izhikevich_hdp_registered(params, edges, ns, 1.0, jax.random.PRNGKey(0),
                      drive_schedule=drv, forced_spikes=force, **kw)
    S = np.asarray(S)
    assert S[3, 0] == 1.0
    assert np.asarray(V)[3, 0] == -65.0
    keep = np.ones((ns, 2), dtype=bool)
    keep[3, 0] = False
    assert np.array_equal(S[keep], np.asarray(S_base)[keep])


def _reader_model():
    cfg = jtfne.suite2_net1_config(seed=1, n=4, duration_ms=20.0, dt_ms=1.0).runtime(
        enable_hdp=True, recurrent_backend="edge_list",
        hdp_params={"hdp_rule": "synthetic_presyn_gain",
                    "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                    "noise_scale": 0.0})
    model = jtfne.construct(cfg)
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                             hdp_params={"hdp_rule": "synthetic_presyn_gain",
                                         "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                                         "noise_scale": 0.0})
    return model, rt


def test_spike_train_input_model_level_reader():
    model, rt = _reader_model()
    n = int(model.params["emitter"].n_neurons)
    sim_cfg = jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt)
    plain = jtfne.simulate(model, sim_cfg)
    # An empty clamp is a no-op: bit-exact parity with the plain run.
    empty = jtfne.simulate(model, sim_cfg, jtfne.SpikeTrainInput({}, n_neurons=n))
    assert jnp.array_equal(empty.spikes, plain.spikes)
    assert jnp.array_equal(empty.V_m, plain.V_m)
    # Force neuron 0 at a bin where it is otherwise silent.
    silent = [t for t in range(20) if float(plain.spikes[t, 0]) == 0.0]
    assert silent, "fixture must leave neuron 0 silent somewhere"
    t_force = silent[len(silent) // 2]
    forced = jtfne.simulate(
        model, sim_cfg, jtfne.SpikeTrainInput({0: [float(t_force)]}, n_neurons=n))
    assert float(forced.spikes[t_force, 0]) == 1.0
    assert float(forced.V_m[t_force, 0]) == float(model.params["emitter"].c[0])


def test_spike_train_input_refusals():
    model, rt = _reader_model()
    n = int(model.params["emitter"].n_neurons)
    sim_cfg = jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt)
    with pytest.raises(ValueError, match="neuron 99"):
        jtfne.SpikeTrainInput({99: [1.0]}, n_neurons=n)
    with pytest.raises(ValueError, match="non-finite"):
        jtfne.SpikeTrainInput({0: [float("nan")]}, n_neurons=n)
    with pytest.raises(ValueError, match="beyond"):
        jtfne.simulate(model, sim_cfg, jtfne.SpikeTrainInput({0: [50.0]}, n_neurons=n))
    with pytest.raises(ValueError, match="one spike per bin"):
        jtfne.simulate(model, sim_cfg, jtfne.SpikeTrainInput({0: [5.0, 5.4]}, n_neurons=n))
    with pytest.raises(ValueError, match="poisson_drive"):
        jtfne.simulate(
            model,
            jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt,
                             poisson_drive={"rate_hz": 2.0, "amplitude": 0.5}),
            jtfne.SpikeTrainInput({0: [5.0]}, n_neurons=n))
    with pytest.raises(ValueError, match="ablation"):
        jtfne.simulate(
            model,
            jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt,
                             ablation="E_silence"),
            jtfne.SpikeTrainInput({0: [5.0]}, n_neurons=n))
    # The plain recurrent kernel does not realize the clamp.
    plain_cfg = jtfne.suite2_net1_config(seed=1, n=4, duration_ms=20.0, dt_ms=1.0)
    plain_model = jtfne.construct(plain_cfg)
    plain_rt = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    with pytest.raises(ValueError, match="registered-rule"):
        jtfne.simulate(
            plain_model,
            jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=plain_rt),
            jtfne.SpikeTrainInput({0: [5.0]}, n_neurons=n))
    # A clamp is not a drive schedule and never resolves to one.
    with pytest.raises(ValueError, match="SpikeTrainInput"):
        model._resolve_stimulus_schedule(
            jtfne.SpikeTrainInput({0: [1.0]}, n_neurons=n), sim_cfg, rt)


def test_spike_train_input_floor_binning():
    # Spec bins are [n*dt, (n+1)*dt): floor, with a 1e-9-step epsilon
    # absorbing float-division dust (2.35/0.05 reads 46.999... in float64).
    inp = jtfne.SpikeTrainInput({0: [2.35], 1: [0.6], 2: [2.35 - 1e-6]}, n_neurons=3)
    mask = np.asarray(inp.to_array(60, 0.05))
    assert mask[:, 0].sum() == 1 and int(mask[:, 0].argmax()) == 47
    assert mask[:, 1].sum() == 1 and int(mask[:, 1].argmax()) == 12
    # Clearly below the edge stays in the lower bin ...
    assert mask[:, 2].sum() == 1 and int(mask[:, 2].argmax()) == 46
    # ... while femtosecond-scale dust joins the edge bin (documented eps).
    edge_dust = np.asarray(
        jtfne.SpikeTrainInput({0: [2.35 - 1e-12]}, n_neurons=1).to_array(60, 0.05))
    assert int(edge_dust[:, 0].argmax()) == 47
    # round() was wrong here: 0.6 ms at dt=1.0 is bin 0, and 19.6 ms with
    # 20 steps is the last bin (round() refused it as beyond-duration).
    coarse = np.asarray(
        jtfne.SpikeTrainInput({0: [0.6], 1: [19.6]}, n_neurons=2).to_array(20, 1.0))
    assert int(coarse[:, 0].argmax()) == 0
    assert int(coarse[:, 1].argmax()) == 19
    # The last bin is bins: t = duration - dt/2 lands on step n_steps - 1.
    last = np.asarray(
        jtfne.SpikeTrainInput({0: [19.5]}, n_neurons=1).to_array(20, 1.0))
    assert int(last[:, 0].argmax()) == 19
    # t >= duration is still refused, naming the neuron.
    with pytest.raises(ValueError, match="neuron 0"):
        jtfne.SpikeTrainInput({0: [20.0]}, n_neurons=1).to_array(20, 1.0)
    # A non-sequence times entry is refused as ValueError naming the neuron.
    with pytest.raises(ValueError, match="neuron 0"):
        jtfne.SpikeTrainInput({0: 5.0}, n_neurons=1)
    with pytest.raises(ValueError, match="neuron 1"):
        jtfne.SpikeTrainInput({1: "5.0"}, n_neurons=2)


def test_delayed_registered_traced_offset_jit():
    # Fix 1: the delayed path must shift a concrete range (off + arange(n))
    # so a traced continuation offset compiles under jit, like the
    # zero-delay path and the emitters.py siblings already do.
    params, edges = _three_neuron((2, 9, 5))
    total, chunk = 10, 4
    kw = dict(hdp_rule="synthetic_presyn_gain",
              hdp_rule_params={"k_h": 0.3, "k_w": 0.05, "gamma": 0.0},
              noise_scale=0.0)
    drv = jnp.zeros((total, 3), dtype=jnp.float32).at[:, 0].set(14.0)
    _, _, _, diag1 = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, chunk, 1.0, jax.random.PRNGKey(7),
        drive_schedule=drv[:chunk], **kw)
    assert "continuation_step_offset" in diag1, "fixture must carry a step offset"
    _, S2_ref, _, _ = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, total - chunk, 1.0, jax.random.PRNGKey(7),
        drive_schedule=drv[chunk:], init_state=diag1, **kw)
    concrete = dict(diag1)

    def run(off):
        init = dict(concrete)
        init["continuation_step_offset"] = off
        return simulate_edge_recurrent_izhikevich_hdp_registered(
            params, edges, total - chunk, 1.0, jax.random.PRNGKey(7),
            drive_schedule=drv[chunk:], init_state=init, **kw)[1]

    S2 = np.asarray(jax.jit(run)(jnp.asarray(diag1["continuation_step_offset"])))
    assert np.array_equal(S2, np.asarray(S2_ref))


def _reader_model_jit():
    cfg = jtfne.suite2_net1_config(seed=1, n=4, duration_ms=20.0, dt_ms=1.0).runtime(
        enable_hdp=True, recurrent_backend="edge_list",
        hdp_params={"hdp_rule": "synthetic_presyn_gain",
                    "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                    "noise_scale": 0.0})
    model = jtfne.construct(cfg)
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True, jit=True,
                             hdp_params={"hdp_rule": "synthetic_presyn_gain",
                                         "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                                         "noise_scale": 0.0})
    return model, rt


def test_forced_spike_trials_reuse_compilation():
    # Fix 2: repeated forced-spike trials reuse one compilation; only the
    # raster shape and presence sit in the static cache key, the raster
    # itself rides a traced kernel argument.
    model, rt = _reader_model_jit()
    n = int(model.params["emitter"].n_neurons)
    sim_cfg = jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt)
    r1 = jtfne.simulate(model, sim_cfg, jtfne.SpikeTrainInput({0: [5.0]}, n_neurons=n))
    assert len(model._compiled_cache) == 1, "first forced trial must compile once"
    r2 = jtfne.simulate(model, sim_cfg, jtfne.SpikeTrainInput({1: [9.0]}, n_neurons=n))
    assert len(model._compiled_cache) == 1, "second raster must reuse the compilation"
    assert not bool(jnp.array_equal(r1.spikes, r2.spikes)), "trials must differ"
    assert float(r1.spikes[5, 0]) == 1.0
    assert float(r2.spikes[9, 1]) == 1.0


def test_forced_spike_continuation_segment_relative():
    # Fix 3: SpikeTrainInput composes with return_state/continuation; each
    # segment's spike times are relative to that segment's start, so chunked
    # forced runs reproduce the continuous forced run. The reference also returns
    # its state: XLA compiles return_state=True and False as different programs, which
    # may differ by one float32 ulp in V_m (test_continuation_contract compares the same way).
    model, rt = _reader_model()
    n = int(model.params["emitter"].n_neurons)
    full, _ = jtfne.simulate(
        model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({0: [5.0], 1: [15.0]}, n_neurons=n), return_state=True)
    s1, st1 = jtfne.simulate(
        model, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({0: [5.0]}, n_neurons=n), return_state=True)
    assert st1 is not None
    s2, _ = jtfne.simulate(
        model, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=99, runtime=rt),
        jtfne.SpikeTrainInput({1: [5.0]}, n_neurons=n),
        continuation=st1, return_state=True)
    assert jnp.array_equal(jnp.concatenate([s1.spikes, s2.spikes]), full.spikes)
    assert jnp.array_equal(jnp.concatenate([s1.V_m, s2.V_m]), full.V_m)


def _reader_model_trace_stdp():
    hp = {"hdp_rule": "trace_stdp", "hdp_rule_params": {}, "noise_scale": 0.0}
    cfg = jtfne.suite2_net1_config(seed=1, n=4, duration_ms=40.0, dt_ms=1.0).runtime(
        enable_hdp=True, recurrent_backend="edge_list", hdp_params=dict(hp))
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    ds = jnp.zeros((int(edges.n_edges),), dtype=jnp.int32).at[0].set(2)
    object.__setattr__(model, "params", {**model.params, "edge_list": replace(edges, delay_steps=ds, delay_storage="per_edge")})
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                             hdp_params=dict(hp))
    return model, rt


def test_forced_spike_continuation_hdp_trace_stdp():
    # Fix 3 with HDP on: trace_stdp on a 2 ms-delay edge, forced pre/post
    # spikes in both 20 ms segments with a pre(19 ms)->post(25 ms) pair
    # spanning the boundary. One 40 ms run equals two chained 20 ms runs
    # (second segment times relative to its start) on spikes, H and w.
    model, rt = _reader_model_trace_stdp()
    n = int(model.params["emitter"].n_neurons)
    el = model.params["edge_list"]
    pre_n, post_n = int(np.asarray(el.pre)[0]), int(np.asarray(el.post)[0])
    w0 = np.asarray(el.weight)
    full = jtfne.simulate(
        model, jtfne.simulation(duration_ms=40.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({pre_n: [5.0, 19.0], post_n: [10.0, 25.0]}, n_neurons=n))
    for t, i in ((5, pre_n), (19, pre_n), (10, post_n), (25, post_n)):
        assert float(full.spikes[t, i]) == 1.0, "clamp must force the pair spikes"
    H_full = np.asarray(model._last_hdp_diag["H_final"])
    w_full = np.asarray(model._last_hdp_diag["w_final"])
    Ht_full = np.asarray(model._last_hdp_diag["H_trace"])
    assert Ht_full.shape == (40, n, 2), "trace_stdp must record both traces"
    assert not np.allclose(w_full, w0), "pair must move w (precondition)"
    s1, st1 = jtfne.simulate(
        model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({pre_n: [5.0, 19.0], post_n: [10.0]}, n_neurons=n),
        return_state=True)
    Ht1 = np.asarray(model._last_hdp_diag["H_trace"])
    s2, _ = jtfne.simulate(
        model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=99, runtime=rt),
        jtfne.SpikeTrainInput({post_n: [5.0]}, n_neurons=n),
        continuation=st1, return_state=True)
    H2 = np.asarray(model._last_hdp_diag["H_final"])
    w2 = np.asarray(model._last_hdp_diag["w_final"])
    Ht2 = np.asarray(model._last_hdp_diag["H_trace"])
    assert np.array_equal(np.concatenate([np.asarray(s1.spikes), np.asarray(s2.spikes)], axis=0),
                          np.asarray(full.spikes))
    assert np.array_equal(np.concatenate([Ht1, Ht2], axis=0), Ht_full)
    assert np.array_equal(H2, H_full)
    assert np.array_equal(w2, w_full)


def _reader_model_trace_stdp_masked():
    model, _ = _reader_model_trace_stdp()
    n_edges = int(model.params["edge_list"].n_edges)
    mask = np.ones((n_edges,))
    mask[0] = 0.0
    hp = {"hdp_rule": "trace_stdp", "hdp_rule_params": {}, "noise_scale": 0.0,
          "plasticity_mask": jnp.asarray(mask)}
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                             hdp_params=dict(hp))
    return model, rt


def test_forced_spike_continuation_plasticity_mask():
    # Fix 3 carries plasticity_mask: edge 0 frozen in the single run and in
    # both chained segments; an unmasked edge still moves; chained == single
    # on spikes, H and w_final.
    model, rt = _reader_model_trace_stdp_masked()
    n = int(model.params["emitter"].n_neurons)
    el = model.params["edge_list"]
    pre_n, post_n = int(np.asarray(el.pre)[0]), int(np.asarray(el.post)[0])
    w0 = np.asarray(el.weight)
    full = jtfne.simulate(
        model, jtfne.simulation(duration_ms=40.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({pre_n: [5.0, 19.0], post_n: [10.0, 25.0]}, n_neurons=n))
    w_full = np.asarray(model._last_hdp_diag["w_final"])
    H_full = np.asarray(model._last_hdp_diag["H_final"])
    Ht_full = np.asarray(model._last_hdp_diag["H_trace"])
    assert float(w_full[0]) == float(w0[0]), "masked edge must freeze (single)"
    assert not np.allclose(w_full[1:], w0[1:]), "unmasked edges must move"
    s1, st1 = jtfne.simulate(
        model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({pre_n: [5.0, 19.0], post_n: [10.0]}, n_neurons=n),
        return_state=True)
    Ht1 = np.asarray(model._last_hdp_diag["H_trace"])
    s2, _ = jtfne.simulate(
        model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=99, runtime=rt),
        jtfne.SpikeTrainInput({post_n: [5.0]}, n_neurons=n),
        continuation=st1, return_state=True)
    w2 = np.asarray(model._last_hdp_diag["w_final"])
    H2 = np.asarray(model._last_hdp_diag["H_final"])
    Ht2 = np.asarray(model._last_hdp_diag["H_trace"])
    assert float(w2[0]) == float(w0[0]), "masked edge must freeze (chained)"
    assert np.array_equal(np.concatenate([np.asarray(s1.spikes), np.asarray(s2.spikes)], axis=0),
                          np.asarray(full.spikes))
    assert np.array_equal(np.concatenate([Ht1, Ht2], axis=0), Ht_full)
    assert np.array_equal(H2, H_full)
    assert np.array_equal(w2, w_full)


def _reader_model_trace_stdp_noisy():
    hp = {"hdp_rule": "trace_stdp", "hdp_rule_params": {}, "noise_scale": 0.5}
    cfg = jtfne.suite2_net1_config(seed=1, n=4, duration_ms=40.0, dt_ms=1.0).runtime(
        enable_hdp=True, recurrent_backend="edge_list", hdp_params=dict(hp))
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    ds = jnp.zeros((int(edges.n_edges),), dtype=jnp.int32).at[0].set(2)
    object.__setattr__(model, "params", {**model.params, "edge_list": replace(edges, delay_steps=ds, delay_storage="per_edge")})
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                             hdp_params=dict(hp))
    return model, rt


def test_forced_spike_continuation_noisy_equivalence():
    # Fix 3 under noise: chained forced segments reproduce the single forced
    # run on spikes, H and w_final at fixed seeds; the noise moves V_m
    # against the noise-free run (precondition).
    model, rt = _reader_model_trace_stdp_noisy()
    n = int(model.params["emitter"].n_neurons)
    el = model.params["edge_list"]
    pre_n, post_n = int(np.asarray(el.pre)[0]), int(np.asarray(el.post)[0])
    full = jtfne.simulate(
        model, jtfne.simulation(duration_ms=40.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({pre_n: [5.0, 19.0], post_n: [10.0, 25.0]}, n_neurons=n))
    H_full = np.asarray(model._last_hdp_diag["H_final"])
    w_full = np.asarray(model._last_hdp_diag["w_final"])
    Ht_full = np.asarray(model._last_hdp_diag["H_trace"])
    quiet, _ = _reader_model_trace_stdp()
    q = jtfne.simulate(
        quiet, jtfne.simulation(duration_ms=40.0, dt_ms=1.0, seed=5,
                                runtime=jtfne.RuntimeConfig(
                                    recurrent_backend="edge_list", enable_hdp=True,
                                    hdp_params={"hdp_rule": "trace_stdp",
                                                "hdp_rule_params": {},
                                                "noise_scale": 0.0})),
        jtfne.SpikeTrainInput({pre_n: [5.0, 19.0], post_n: [10.0, 25.0]}, n_neurons=n))
    assert not np.array_equal(np.asarray(full.V_m), np.asarray(q.V_m)), \
        "noise must move the output (precondition)"
    s1, st1 = jtfne.simulate(
        model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt),
        jtfne.SpikeTrainInput({pre_n: [5.0, 19.0], post_n: [10.0]}, n_neurons=n),
        return_state=True)
    Ht1 = np.asarray(model._last_hdp_diag["H_trace"])
    s2, _ = jtfne.simulate(
        model, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=99, runtime=rt),
        jtfne.SpikeTrainInput({post_n: [5.0]}, n_neurons=n),
        continuation=st1, return_state=True)
    assert np.array_equal(np.concatenate([np.asarray(s1.spikes), np.asarray(s2.spikes)], axis=0),
                          np.asarray(full.spikes))
    assert np.array_equal(np.concatenate([Ht1, np.asarray(model._last_hdp_diag["H_trace"])], axis=0),
                          Ht_full)
    assert np.array_equal(np.asarray(model._last_hdp_diag["H_final"]), H_full)
    assert np.array_equal(np.asarray(model._last_hdp_diag["w_final"]), w_full)


def test_registered_plain_path_applies_plasticity_mask():
    model, rt = _reader_model()
    n_edges = int(model.params["edge_list"].n_edges)
    w0 = np.asarray(model.params["edge_list"].weight)
    sim_cfg = jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt)
    jtfne.simulate(model, sim_cfg)
    w_free = np.asarray(model._last_hdp_diag["w_final"])
    assert not np.allclose(w_free, w0), "fixture must learn without a mask"
    frozen_hp = {"hdp_rule": "synthetic_presyn_gain",
                 "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                 "noise_scale": 0.0,
                 "plasticity_mask": jnp.zeros((n_edges,))}
    frozen_rt = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                                    hdp_params=frozen_hp)
    jtfne.simulate(
        model,
        jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=frozen_rt))
    assert np.allclose(np.asarray(model._last_hdp_diag["w_final"]), w0)
