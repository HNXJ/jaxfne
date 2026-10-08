"""PCL H-state rule: ``pcl_stdp_h`` matches ``pcl_stdp`` on the Fig. 2 network."""
from __future__ import annotations

import sys

import numpy as np

import jaxfne  # noqa: F401

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

import artifacts.etudes.pcl.pcl_fig2 as F  # noqa: E402
import artifacts.etudes.pcl.pcl_fig2_hdp as K  # noqa: E402
import artifacts.etudes.pcl.pcl_hdp_rule as R  # noqa: E402
import artifacts.etudes.pcl_h.pcl_h_fig2 as KH  # noqa: E402
import artifacts.etudes.pcl_h.pcl_h_rule as RH  # noqa: E402


def _run_both(seed=7):
    R.register()
    RH.register()
    rng = np.random.default_rng(seed)
    trains, total = K.concat([F.make_sample(rng) for _ in range(2)])
    s, amp = 5.0, 30.0
    rp = R.rule_params(np.full(3, F.TAU_LTP), np.full(3, F.W_MAX * s), np.full(3, F.ETA_LTP),
                       np.full(3, F.ETA_BOUND), np.zeros(3), np.zeros(3), np.arange(3), 3)
    w0 = np.full(3, 5.0)
    w_old, n_old = K.run(w0, trains, amp, total, rp, True)
    w_new, n_new = KH.run(w0, trains, amp, total, rp, True)
    return w_old, n_old, w_new, n_new


def test_h_rule_matches_edge_rule_weights_and_spikes():
    w_old, n_old, w_new, n_new = _run_both()
    assert n_old[1:].sum() > 0, "fixture must drive the post neurons"
    np.testing.assert_allclose(w_new, w_old, rtol=1e-5, atol=1e-5)
    assert np.array_equal(n_new, n_old)


def test_h_rule_declares_h_traces_and_per_edge_ltd():
    from jaxfne.hdp_rule import get_hdp_rule
    RH.register()
    desc, _ = get_hdp_rule(RH.NAME)
    assert desc.h_shape == (2,) and tuple(desc.h_coords) == ("pre_trace", "post_trace")
    assert desc.aux_layout == "per_edge" and tuple(desc.aux_coords) == ("pre_snapshot", "ltd_sum")
    assert tuple(desc.theta_targets) == ("edge_weight",)


def _multi_input_run(rule, rp, aux_zeros, h_zeros=None):
    """One post neuron (3) with three plastic inputs (0, 1, 2), independent trains."""
    import jax
    import jax.numpy as jnp
    from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim
    from jaxfne.emitters import EdgeList
    rng = np.random.default_rng(11)
    trains = [np.sort(rng.uniform(0.0, 500.0, rng.poisson(10))) for _ in range(4)]
    amp = 30.0
    sched = K.schedule(trains, amp, 500.0)
    assert not np.array_equal(np.asarray(sched[:, 0]), np.asarray(sched[:, 1]))
    assert not np.array_equal(np.asarray(sched[:, 0]), np.asarray(sched[:, 2]))
    assert not np.array_equal(np.asarray(sched[:, 1]), np.asarray(sched[:, 2]))
    edges = EdgeList(pre=jnp.array([0, 1, 2]), post=jnp.array([3, 3, 3]),
                     weight=-jnp.full(3, 2.0), receptor_index=jnp.ones(3, jnp.int32),
                     tau_ms=jnp.full(3, 2.0), source_calibration_status="x")
    st = {"v": jnp.full(4, -65.0), "u": jnp.full(4, -13.0), "prev_spikes": jnp.zeros(4),
          "syn_state": jnp.zeros(3), "w_final": -jnp.full(3, 2.0), "aux_final": aux_zeros}
    if h_zeros is not None:
        st["H_final"] = h_zeros
    _, spikes, _, diag = sim(K.izh(4), edges, int(sched.shape[0]), K.DT, jax.random.PRNGKey(0),
                            drive_schedule=sched, noise_scale=0.0, init_state=st, hdp_rule=rule,
                            hdp_rule_params=rp, plasticity_mask=jnp.ones(3))
    return np.abs(np.asarray(diag["w_final"])), np.asarray(spikes).sum(0)


def test_h_rule_matches_edge_rule_with_three_inputs():
    import jax.numpy as jnp
    R.register()
    RH.register()
    s = 5.0
    rp = R.rule_params(np.full(3, F.TAU_LTP), np.full(3, F.W_MAX * s), np.full(3, F.ETA_LTP),
                       np.full(3, F.ETA_BOUND), np.full(3, 30.0), np.ones(3), np.zeros(3, int), 1)
    w_old, n_old = _multi_input_run(R.NAME, rp, jnp.zeros((3, 3), jnp.float32))
    w_new, n_new = _multi_input_run(RH.NAME, rp, jnp.zeros((3, 2), jnp.float32),
                                    jnp.zeros((4, 2), jnp.float32))
    assert n_old[3] > 0, "fixture must make the post neuron spike"
    np.testing.assert_allclose(w_new, w_old, rtol=1e-5, atol=1e-5)
    assert np.array_equal(n_new, n_old)


def test_h_rule_refuses_non_uniform_tau():
    import jax.numpy as jnp
    import pytest
    R.register()
    RH.register()
    rp = R.rule_params([7.0, 7.1, 7.0], np.full(3, 250.0), np.full(3, F.ETA_LTP),
                       np.full(3, F.ETA_BOUND), np.zeros(3), np.zeros(3), np.arange(3), 3)
    with pytest.raises(ValueError, match="tau classes"):  # 2 distinct taus, rule holds 1
        _multi_input_run(RH.NAME, rp, jnp.zeros((3, 2), jnp.float32), jnp.zeros((4, 2), jnp.float32))
    with pytest.raises(ValueError, match="uniform tau"):  # 2 classes, but mixed onto one post neuron
        _multi_input_run(RH.NAME2, rp, jnp.zeros((3, 2), jnp.float32), jnp.zeros((4, 3), jnp.float32))


def test_h2_rule_matches_edge_rule_with_two_tau_classes():
    """Neuron 0 projects onto 1 (tau 7) and 2 (tau 40): two pre traces in H, one per class."""
    import jax
    import jax.numpy as jnp
    from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim
    from jaxfne.emitters import EdgeList
    R.register()
    RH.register()
    rng = np.random.default_rng(3)
    trains = [np.sort(rng.uniform(0.0, 600.0, rng.poisson(14))) for _ in range(3)]
    sched = K.schedule(trains, 30.0, 600.0)
    rp = R.rule_params([7.0, 40.0], np.full(2, 250.0), np.full(2, F.ETA_LTP), np.full(2, F.ETA_BOUND),
                       np.zeros(2), np.zeros(2), np.arange(2), 2)
    edges = EdgeList(pre=jnp.array([0, 0]), post=jnp.array([1, 2]), weight=-jnp.full(2, 2.0),
                     receptor_index=jnp.ones(2, jnp.int32), tau_ms=jnp.full(2, 2.0),
                     source_calibration_status="x")

    def go(rule, aux, h=None):
        st = {"v": jnp.full(3, -65.0), "u": jnp.full(3, -13.0), "prev_spikes": jnp.zeros(3),
              "syn_state": jnp.zeros(2), "w_final": -jnp.full(2, 2.0), "aux_final": aux}
        if h is not None:
            st["H_final"] = h
        _, sp, _, d = sim(K.izh(3), edges, int(sched.shape[0]), K.DT, jax.random.PRNGKey(0),
                          drive_schedule=sched, noise_scale=0.0, init_state=st, hdp_rule=rule,
                          hdp_rule_params=rp, plasticity_mask=jnp.ones(2))
        return np.abs(np.asarray(d["w_final"])), np.asarray(sp).sum(0)

    w_old, n_old = go(R.NAME, jnp.zeros((2, 3), jnp.float32))
    w_new, n_new = go(RH.NAME2, jnp.zeros((2, 2), jnp.float32), jnp.zeros((3, 3), jnp.float32))
    assert n_old[1] > 0 and n_old[2] > 0, "fixture must make both post neurons spike"
    assert np.abs(w_old - 2.0).min() > 1e-3, "fixture must move both weights"
    np.testing.assert_allclose(w_new, w_old, rtol=1e-5, atol=1e-5)
    assert np.array_equal(n_new, n_old)


def test_h_rule_tie_pre_and_post_same_step():
    """Same-step pre+post spike: LTP includes this step's pre event, then the snapshot resets it."""
    import jax.numpy as jnp
    from jaxfne.hdp_rule import HDPRuleContext
    R.register()
    RH.register()
    dt, tau, wmax, eta, f, lam = 0.1, 7.0, 50.0, 2.0, 0.02, 30.0
    w = np.array([10.0, 20.0])
    d = float(np.exp(-dt / tau))
    # Consistent states: old A[e] = x[pre[e]] - S[e]; neuron 0 spikes, neuron 1 does not, post 2 spikes.
    x = np.array([1.0, 0.5, 0.0])
    S = np.array([0.4, 0.2])
    aux_old = np.array([[x[0] - S[0], 0.2, 0.7], [x[1] - S[1], 0.1, 0.7]])
    spikes = np.array([1.0, 0.0, 1.0])
    rp = R.rule_params([tau] * 2, [wmax] * 2, [eta] * 2, [f] * 2, [lam] * 2, [1, 1], [0, 0], 1)
    base = dict(v=jnp.zeros(3), u=jnp.zeros(3), spikes=jnp.asarray(spikes), prev_spikes=jnp.zeros(3),
                syn_state=jnp.zeros(2), w=-jnp.asarray(w, jnp.float32),
                pre=jnp.array([0, 1]), post=jnp.array([2, 2]),
                exc_mask=jnp.array([False, False]), dt=jnp.asarray(dt), n_neurons=3,
                rule_params=rp, key=None)
    upd_old = R._step(HDPRuleContext(H=jnp.ones(3), aux=jnp.asarray(aux_old, jnp.float32), **base))
    H = np.zeros((3, 2), np.float32)
    H[:, 0] = x
    H[2, 1] = 0.7
    upd_new = RH._step(HDPRuleContext(H=jnp.asarray(H), aux=jnp.stack(
        [jnp.asarray(S, jnp.float32), jnp.asarray(aux_old[:, 1], jnp.float32)], axis=-1), **base))
    w_old = w + dt * np.asarray(upd_old.d_theta["edge_weight"])
    w_new = w + dt * np.asarray(upd_new.d_theta["edge_weight"])
    np.testing.assert_allclose(w_new, w_old, rtol=1e-5, atol=1e-5)
    # LTP term included this step's pre spike: edge 0 potentiated more than edge 1.
    assert w_new[0] - w[0] > w_new[1] - w[1]
    B_old = aux_old + dt * np.asarray(upd_old.d_aux)
    aux_new = np.asarray([[S[0], aux_old[0, 1]], [S[1], aux_old[1, 1]]]) + dt * np.asarray(upd_new.d_aux)
    np.testing.assert_allclose(aux_new[:, 1], B_old[:, 1], rtol=1e-5, atol=1e-5)
    # Snapshot now holds x1[pre], so the next-step LTP reads 0 on both edges.
    np.testing.assert_allclose(aux_new[:, 0], [(x[0] * d + 1.0), (x[1] * d + 0.0)], rtol=1e-5)
