"""PCL étude: the Fig. 2 inhibitory-STDP gate orders inhibition by predictability."""
from __future__ import annotations

import numpy as np

import artifacts.etudes.pcl.pcl_fig2 as F


def test_inputs_copy_the_declared_fraction():
    t0, *rest = F.make_sample(np.random.default_rng(0))
    for p, tk in zip(F.P_REPLACE, rest):
        copied = np.isin(np.round(tk - F.LAG_MS, 9), np.round(t0, 9)).mean()
        assert abs(copied - p) < 0.15


def test_inhibition_ordered_by_predictability():
    r = F.run_network(0, w0=1.0, n_train=20, n_test=10, epochs=2)
    w, s = r["w_hist"][-1], r["suppression"]
    assert w[0] > w[1] > w[2] and s[0] > s[1] > s[2]


def test_column_flags_and_normalization():
    import jax
    import artifacts.etudes.pcl.pcl_column as C
    W0 = C.init_weights(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    Wf, s, _ = C.run_sequence(W0, key, 2, 1.0, 1.0, (False, False, 1.0), 200)
    assert all(np.array_equal(np.asarray(Wf[k]), np.asarray(W0[k])) for k in W0) and s.sum() > 0
    We, _, _ = C.run_sequence(W0, key, 2, 1.0, 1.0, (True, False, 0.0), 200)
    Wi, _, _ = C.run_sequence(W0, key, 2, 1.0, 1.0, (False, True, 1.0), 200)
    assert not np.array_equal(np.asarray(We["s_exc"]), np.asarray(W0["s_exc"]))
    assert np.array_equal(np.asarray(We["s_dist"]), np.asarray(W0["s_dist"]))
    assert not np.array_equal(np.asarray(Wi["s_dist"]), np.asarray(W0["s_dist"]))
    for name, W in (("s_exc", We), ("s_dist", Wi)):
        lam = C.CONN[name][3] * C.MASKS[name].sum(-1)
        np.testing.assert_allclose(np.asarray(W[name]).sum(-1), lam, rtol=1e-4)


def test_segment_is_cut_along_its_axis(monkeypatch):
    import jax
    import jax.numpy as jnp
    import artifacts.etudes.pcl.pcl_column as C
    monkeypatch.setitem(C.STIM, "kind", "segment")
    monkeypatch.setitem(C.STIM, "length", 4.0)
    cov = jnp.zeros(C.G * C.G, bool)
    inside = cov
    for t in np.arange(400) * C.DT:  # ori 0: horizontal segment swept vertically
        _, inside = C.stimulus_step(jax.random.PRNGKey(0), inside, t, 0, 1.0, -13.0, 2.0)
        cov = cov | inside
    cols = np.nonzero(np.asarray(cov).reshape(C.G, C.G).any(0))[0]
    assert cols.min() == 4 and cols.max() == 7  # ori 0 axis is -x: |-(x - 7.5) - 2| < 2


def test_hdp_rule_matches_paper_update():
    """One step of pcl_stdp: post neuron 2 spikes; edges 0->2 and 1->2 share a group."""
    import jax.numpy as jnp
    from jaxfne.hdp_rule import HDPRuleContext
    import artifacts.etudes.pcl.pcl_hdp_rule as R
    dt, tau, wmax, eta, f, lam = 0.1, 7.0, 50.0, 2.0, 0.02, 30.0
    w = np.array([10.0, 20.0])
    aux = np.array([[0.5, 0.2, 0.8], [0.0, 0.1, 0.8]])  # ltp, ltd, ypost
    spikes = np.array([1.0, 0.0, 1.0])  # pre 0 and post 2 spike this step
    rp = R.rule_params([tau] * 2, [wmax] * 2, [eta] * 2, [f] * 2, [lam] * 2, [1, 1], [0, 0], 1)
    ctx = HDPRuleContext(H=jnp.ones(3), aux=jnp.asarray(aux, jnp.float32), v=jnp.zeros(3), u=jnp.zeros(3),
                         spikes=jnp.asarray(spikes), prev_spikes=jnp.zeros(3), syn_state=jnp.zeros(2),
                         w=-jnp.asarray(w, jnp.float32), pre=jnp.array([0, 1]), post=jnp.array([2, 2]),
                         exc_mask=jnp.array([False, False]), dt=jnp.asarray(dt), n_neurons=3,
                         rule_params=rp, key=None)
    upd = R._step(ctx)
    d = np.exp(-dt / tau)
    A1 = aux[:, 0] * d + np.array([1.0, 0.0])
    B1 = aux[:, 1] + np.array([1.0, 0.0]) * aux[:, 2] * d
    w1 = np.maximum(w + (wmax - w) * f * eta * A1 - w * f * eta * B1, 0)
    w2 = w1 * lam / w1.sum()
    np.testing.assert_allclose(w + dt * np.asarray(upd.d_theta["edge_weight"]), w2, rtol=1e-5)
    aux_next = aux + dt * np.asarray(upd.d_aux)
    np.testing.assert_allclose(aux_next, [[0, 0, 1], [0, 0, 1]], atol=1e-5)


def test_no_learning_keeps_weights():
    net = F.Net(3.0)
    net.run(F.make_sample(np.random.default_rng(1)), learn=False)
    assert np.array_equal(net.w, np.full(3, 3.0))
