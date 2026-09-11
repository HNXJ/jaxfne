"""23-HDP-01: registrable rule qualification and causal effectiveness."""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.hdp_rule import hdp_params_are_identity, list_registered_hdp_rules


def _two_neuron_fixture(*, k_w: float):
    params = IzhikevichParams(
        a=jnp.full((2,), 0.02),
        b=jnp.full((2,), 0.2),
        c=jnp.full((2,), -65.0),
        d=jnp.full((2,), 8.0),
        drive=jnp.array([0.0, 0.0]),
        sign=jnp.ones((2,)),
        W=jnp.zeros((2, 2)),
        v0=jnp.full((2,), -65.0),
        u0=jnp.full((2,), -13.0),
        source_scale=jnp.ones((2,)),
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
        source_calibration_status="x",
    )
    n_steps = 20
    drive = jnp.zeros((n_steps, 2), dtype=jnp.float32)
    drive = drive.at[:, 0].set(8.0)
    return params, edges, drive, {
        "hdp_rule": "synthetic_presyn_gain",
        "hdp_rule_params": {"k_h": 0.5, "k_w": k_w, "gamma": 0.0},
    }


def _manual_rule_trajectory(spikes: np.ndarray, w0: float, *, k_h: float, k_w: float, dt: float):
    H = np.ones(2, dtype=np.float64)
    w = np.array([w0], dtype=np.float64)
    H_trace = []
    w_trace = []
    for t in range(spikes.shape[0]):
        s = spikes[t]
        pre_sp = s[0]
        dw = k_w * H[1] * pre_sp * abs(w[0])
        dH = np.zeros(2)
        dH[1] += k_h * pre_sp
        H = H + dt * dH
        w[0] = np.clip(w[0] + dt * dw, 1e-3, 50.0)
        H_trace.append(H.copy())
        w_trace.append(w[0])
    return np.asarray(H_trace), np.asarray(w_trace)


def test_registered_rule_surface_exists():
    assert "synthetic_presyn_gain" in list_registered_hdp_rules()


def test_p1_ne_p2_implies_different_theta_and_current():
    import jax

    p1, e1, d1, kw1 = _two_neuron_fixture(k_w=0.05)
    p2, e2, d2, kw2 = _two_neuron_fixture(k_w=0.20)
    _, _, src1, diag1 = simulate_edge_recurrent_izhikevich_hdp_registered(
        p1, e1, d1.shape[0], 1.0, jax.random.PRNGKey(0),
        drive_schedule=d1, noise_scale=0.0,
        hdp_rule=kw1["hdp_rule"], hdp_rule_params=kw1["hdp_rule_params"],
    )
    _, _, src2, diag2 = simulate_edge_recurrent_izhikevich_hdp_registered(
        p2, e2, d2.shape[0], 1.0, jax.random.PRNGKey(0),
        drive_schedule=d2, noise_scale=0.0,
        hdp_rule=kw2["hdp_rule"], hdp_rule_params=kw2["hdp_rule_params"],
    )
    assert not np.allclose(np.asarray(diag1["w_final"]), np.asarray(diag2["w_final"]))
    assert not np.allclose(np.asarray(src1), np.asarray(src2))


def test_analytic_H_and_theta_trajectories_match_manual_discrete_rule():
    import jax

    params, edges, drive, kw = _two_neuron_fixture(k_w=0.1)
    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        params,
        edges,
        drive.shape[0],
        1.0,
        jax.random.PRNGKey(2),
        drive_schedule=drive,
        noise_scale=0.0,
        hdp_rule=kw["hdp_rule"],
        hdp_rule_params=kw["hdp_rule_params"],
    )
    H = np.asarray(diag["H_trace"])
    w = np.asarray(diag["w_trace"])[:, 0]
    H_exp, w_exp = _manual_rule_trajectory(
        np.asarray(spikes), 0.2, k_h=0.5, k_w=0.1, dt=1.0
    )
    np.testing.assert_allclose(H, H_exp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(w, w_exp, rtol=1e-5, atol=1e-5)


def test_disabled_identity_routes_to_baseline():
    cfg_off = jtfne.suite2_net1_config(seed=3, n=8).runtime(enable_hdp=False)
    cfg_on = jtfne.suite2_net1_config(seed=3, n=8).runtime(enable_hdp=True)
    assert hdp_params_are_identity({})
    off = jtfne.construct(cfg_off)
    on = jtfne.construct(cfg_on)
    v_off = np.asarray(jtfne.simulate(off, duration_ms=20.0, dt_ms=0.5, seed=4).V_m)
    v_on = np.asarray(jtfne.simulate(on, duration_ms=20.0, dt_ms=0.5, seed=4).V_m)
    assert np.array_equal(v_off, v_on)


def test_model_dispatch_registered_rule():
    model = jtfne.construct(
        jtfne.suite2_net1_config(seed=1, n=8).runtime(
            enable_hdp=True,
            recurrent_backend="edge_list",
            hdp_params={
                "hdp_rule": "synthetic_presyn_gain",
                "hdp_rule_params": {"k_h": 0.01, "k_w": 0.02, "gamma": 0.0},
            },
        )
    )
    sig = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=1)
    diag = model.last_hdp_diagnostics()
    assert diag is not None
    assert bool(np.isfinite(sig.V_m).all())
    assert float(np.max(np.abs(np.asarray(diag["w_final"])))) > 0.0


def test_checkpoint_chunk_continuation_registered_rule():
    model = jtfne.construct(
        jtfne.suite2_net1_config(seed=6, n=8).runtime(
            enable_hdp=True,
            recurrent_backend="edge_list",
            hdp_params={
                "hdp_rule": "synthetic_presyn_gain",
                "hdp_rule_params": {"k_h": 0.02, "k_w": 0.03, "gamma": 0.0},
            },
        )
    )
    full, full_state = jtfne.simulate(
        model, duration_ms=12.0, dt_ms=0.5, seed=6, return_state=True
    )
    first, first_state = jtfne.simulate(
        model, duration_ms=6.0, dt_ms=0.5, seed=6, return_state=True
    )
    second, second_state = jtfne.simulate(
        model,
        duration_ms=6.0,
        dt_ms=0.5,
        seed=99,
        continuation=first_state,
        return_state=True,
    )
    assert jnp.array_equal(
        jnp.concatenate((first.V_m, second.V_m), axis=0), full.V_m
    )
    assert jnp.array_equal(full_state.dynamic.w, second_state.dynamic.w)
    assert jnp.array_equal(full_state.dynamic.H, second_state.dynamic.H)
