"""23-LAW-01: structured-law exercise of the registrable primitive.

Rule under test: ``eligibility_trace_gain`` — event-driven per-edge
eligibility with H-gated weight drive (aux-carrying structured law, not
an STDP mechanism claim). Exercises the primitive paths the
qualification rule never touches: non-empty ``aux`` cold start,
carry, continuation, and delayed composition.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered
from jaxfne._pipeline import (
    compile_step_fn,
    continuation_state_from_model,
    run_continuation,
)
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.hdp_rule import expected_aux_shape, get_hdp_rule, list_registered_hdp_rules


RULE = "eligibility_trace_gain"
PARAMS = {"k_h": 0.5, "k_w": 0.1, "gamma": 0.0, "tau_e": 5.0}


def _two_neuron(delay_steps=(0,)):
    n = 2
    params = IzhikevichParams(
        a=jnp.full((n,), 0.02), b=jnp.full((n,), 0.2), c=jnp.full((n,), -65.0),
        d=jnp.full((n,), 8.0), drive=jnp.zeros(n), sign=jnp.ones(n),
        W=jnp.zeros((n, n)), v0=jnp.full((n,), -65.0), u0=jnp.full((n,), -13.0),
        source_scale=jnp.ones(n), labels=("E", "E"), layer_labels=("L4", "L4"),
        source_calibration_status="x",
    )
    edges = EdgeList(
        pre=jnp.array([0], dtype=jnp.int32), post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([0.2], dtype=jnp.float32),
        receptor_index=jnp.array([0], dtype=jnp.int32),
        tau_ms=jnp.array([2.0], dtype=jnp.float32),
        delay_steps=jnp.array(delay_steps, dtype=jnp.int32),
        source_calibration_status="x",
    )
    return params, edges


def _manual_trajectory(spikes, w0, *, k_h, k_w, gamma, tau_e, dt):
    H = np.ones(2, dtype=np.float64)
    E = np.zeros(1, dtype=np.float64)
    w = np.array([w0], dtype=np.float64)
    H_trace, E_trace, w_trace = [], [], []
    for t in range(spikes.shape[0]):
        s = spikes[t]
        pre_sp, post_sp = s[0], s[1]
        dH = np.zeros(2)
        dH[1] += k_h * pre_sp
        dH -= gamma * (H - 1.0)
        dE = -E / tau_e + pre_sp * post_sp
        dw = k_w * H[1] * E[0] * abs(w[0])
        H = np.clip(H + dt * dH, 0.1, 10.0)
        E = E + dt * dE
        w[0] = np.clip(abs(w[0]) + dt * dw, 1e-3, 50.0)
        H_trace.append(H.copy())
        E_trace.append(E.copy())
        w_trace.append(w[0])
    return np.asarray(H_trace), np.asarray(E_trace), np.asarray(w_trace)


def test_rule_surface_and_aux_layout():
    assert RULE in list_registered_hdp_rules()
    desc, _ = get_hdp_rule(RULE)
    assert desc.aux_layout == "per_edge"
    assert desc.aux_coords == ("eligibility",)
    assert expected_aux_shape(desc, n_neurons=4, n_edges=9) == (9,)
    with pytest.raises(ValueError, match="aux_layout"):
        from jaxfne.hdp_rule import HDPRuleDescriptor, register_hdp_rule
        register_hdp_rule(
            HDPRuleDescriptor(name="bogus_aux_rule_xyz", aux_layout="per_block"),
            lambda ctx: None,  # never reached; validation fires first
        )


def _two_neuron_drive(n_steps, level=12.0):
    return jnp.zeros((n_steps, 2), dtype=jnp.float32).at[:, :].set(level)


def test_analytic_eligibility_trajectory():
    params, edges = _two_neuron()
    drive = _two_neuron_drive(20)
    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 20, 1.0, jax.random.PRNGKey(2),
        drive_schedule=drive, noise_scale=0.0, hdp_rule=RULE, hdp_rule_params=PARAMS)
    H = np.asarray(diag["H_trace"])
    E = np.asarray(diag["aux_trace"])[:, 0]
    w = np.asarray(diag["w_trace"])[:, 0]
    H_exp, E_exp, w_exp = _manual_trajectory(np.asarray(spikes), 0.2, dt=1.0, **PARAMS)
    assert np.any(np.asarray(spikes)[:, 1] > 0.5)  # coincidence occurred
    np.testing.assert_allclose(H, H_exp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(E, E_exp[:, 0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(w, w_exp, rtol=1e-5, atol=1e-5)
    assert np.asarray(diag["aux_final"]).shape == (1,)


def test_gain_divergence_causal_current():
    def run(k_w):
        p, e = _two_neuron()
        drv = _two_neuron_drive(20)
        _, _, src, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
            p, e, 20, 1.0, jax.random.PRNGKey(0), drive_schedule=drv, noise_scale=0.0,
            hdp_rule=RULE, hdp_rule_params={**PARAMS, "k_w": k_w})
        return np.asarray(src), np.asarray(diag["w_final"])
    s1, w1 = run(0.02)
    s2, w2 = run(0.30)
    assert not np.allclose(w1, w2)
    assert not np.allclose(s1, s2)


def _pulsed_scenario(d, pulse, total=40):
    """Tonic presynaptic drive (pre spikes {2,6,16,...}) + timed post pulse.

    Engineered coincidences (verified): d=0/pulse[14,17) -> post spike 16
    (= pre spike); d=3/pulse[17,20) -> post spike 19 (= pre spike 16 + 3).
    """
    p, e = _two_neuron([d])
    drv = jnp.zeros((total, 2), dtype=jnp.float32).at[:, 0].set(20.0)
    drv = drv.at[pulse[0]:pulse[1], 1].set(30.0)
    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        p, e, total, 1.0, jax.random.PRNGKey(0), drive_schedule=drv, noise_scale=0.0,
        hdp_rule=RULE, hdp_rule_params=PARAMS)
    E = np.asarray(diag["aux_trace"])[:, 0]
    H = np.asarray(diag["H_trace"])[:, 1]
    return E, H


def test_delayed_eligibility_arrival_shift():
    """Delayed pre_sp reaches dH and dE together: eligibility onset shifts by
    exactly d (16 -> 19), nothing before arrival, H arrival shifts likewise."""
    E0, H0 = _pulsed_scenario(0, (14, 17))
    E3, H3 = _pulsed_scenario(3, (17, 20))
    eon0 = int(np.argmax(E0 > 1e-9))
    eon3 = int(np.argmax(E3 > 1e-9))
    assert eon0 == 16
    assert eon3 == 19 == eon0 + 3
    assert np.all(E3[:19] == 0.0)
    hon0 = int(np.argmax(H0 > 1.0 + 1e-6))
    hon3 = int(np.argmax(H3 > 1.0 + 1e-6))
    assert hon3 == hon0 + 3


def test_aux_chunk_continuation_exact():
    params, edges = _two_neuron()
    drv = _two_neuron_drive(20)
    kw = dict(hdp_rule=RULE, hdp_rule_params=PARAMS)
    Vf, Sf, _, diagf = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 20, 1.0, jax.random.PRNGKey(0),
        drive_schedule=drv, noise_scale=0.0, **kw)
    _, _, _, diag1 = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 10, 1.0, jax.random.PRNGKey(0),
        drive_schedule=drv[:10], noise_scale=0.0, **kw)
    assert np.asarray(diag1["aux_final"]).shape == (1,)
    V2, S2, _, diag2 = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 10, 1.0, jax.random.PRNGKey(0),
        drive_schedule=drv[10:], noise_scale=0.0, init_state=diag1, **kw)
    assert jnp.array_equal(V2, Vf[10:])
    assert jnp.array_equal(S2, Sf[10:])
    np.testing.assert_allclose(
        np.asarray(diag2["aux_final"]), np.asarray(diagf["aux_final"]),
        rtol=1e-5, atol=1e-5)


def test_aux_shape_mismatch_rejected():
    params, edges = _two_neuron()
    drv = jnp.zeros((6, 2), dtype=jnp.float32).at[:, 0].set(8.0)
    _, _, _, first = simulate_edge_recurrent_izhikevich_hdp_registered(
        params, edges, 6, 1.0, jax.random.PRNGKey(0),
        drive_schedule=drv, noise_scale=0.0, hdp_rule=RULE, hdp_rule_params=PARAMS)
    bad = dict(first)
    bad["aux_final"] = jnp.zeros((0,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="aux_final must have shape"):
        simulate_edge_recurrent_izhikevich_hdp_registered(
            params, edges, 6, 1.0, jax.random.PRNGKey(0),
            drive_schedule=drv, noise_scale=0.0, hdp_rule=RULE,
            hdp_rule_params=PARAMS, init_state=bad)


def test_model_dispatch_and_continuation_with_aux():
    model = jtfne.construct(
        jtfne.suite2_net1_config(seed=1, n=8).runtime(
            enable_hdp=True, recurrent_backend="edge_list",
            hdp_params={"hdp_rule": RULE, "hdp_rule_params": PARAMS, "noise_scale": 0.0}))
    sig = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=1)
    diag = model.last_hdp_diagnostics()
    assert bool(np.isfinite(np.asarray(sig.V_m)).all())
    full, _ = jtfne.simulate(model, duration_ms=12.0, dt_ms=0.5, seed=6, return_state=True)
    first, st1 = jtfne.simulate(model, duration_ms=6.0, dt_ms=0.5, seed=6, return_state=True)
    assert st1.dynamic.aux.shape[0] == int(model.params["edge_list"].n_edges)
    second, st2 = jtfne.simulate(
        model, duration_ms=6.0, dt_ms=0.5, seed=99, continuation=st1, return_state=True)
    assert jnp.array_equal(jnp.concatenate((first.V_m, second.V_m), axis=0), full.V_m)
    assert jnp.array_equal(st2.dynamic.aux.shape, st1.dynamic.aux.shape)
