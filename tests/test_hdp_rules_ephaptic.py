"""Endogenous ephaptic field rule ``ephaptic_field`` (jaxfne/hdp_rules_ephaptic.py)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne  # noqa: F401  (registers the rule)
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.emitters_lif import make_lif_params, simulate_lif_edge
from jaxfne.hdp_rule import check_hdp_rule_params
from jaxfne.hdp_rules_ephaptic import (
    EPHAPTIC_RULE_NAME,
    ephaptic_field,
    ephaptic_kernel,
    ephaptic_rule_params,
)

N = 40
N_STEPS = 400
DT = 0.1


def _dense(kernel, n):
    M = np.zeros((n, n))
    M[kernel["dst"], kernel["src"]] = kernel["weight"]
    return M


def test_kernel_geometry_and_normalization():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    raw = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=2.5, normalize="none"), 3)
    expected = np.zeros((3, 3))
    expected[0, 1] = expected[1, 0] = np.exp(-0.5)  # d = 1
    expected[1, 2] = expected[2, 1] = np.exp(-2.0)  # d = 2; d(0,2) = 3 > cutoff
    np.testing.assert_allclose(raw, expected, rtol=1e-12)
    row = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=2.5), 3)
    np.testing.assert_allclose(row.sum(axis=1), [1.0, 1.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(row[1], expected[1] / expected[1].sum(), rtol=1e-12)
    assert np.all(np.diag(row) == 0.0)
    lone = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=0.5), 3)
    assert lone.sum() == 0.0  # no pair within cutoff: rows stay 0, no division by 0


def test_field_matches_dense_product():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.0, 100.0, size=(N, 3))
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    assert 0 < len(k["src"]) < N * (N - 1), "fixture must be sparse and non-empty"
    v = rng.normal(-60.0, 5.0, size=N)
    phi = ephaptic_field(jnp.asarray(v), src=k["src"], dst=k["dst"], weight=k["weight"],
                         alpha=0.3, v_rest=-65.0)
    # float32 kernel: the sum over neighbours of (v_j + 65) cancels, so the bound is absolute.
    np.testing.assert_allclose(np.asarray(phi), 0.3 * _dense(k, N) @ (v + 65.0), rtol=1e-5, atol=1e-5)


def test_params_fail_closed():
    with pytest.raises(ValueError, match="unrecognized"):
        check_hdp_rule_params(EPHAPTIC_RULE_NAME, {"alpah": 1.0})
    with pytest.raises(ValueError, match="kernel lacks"):
        ephaptic_rule_params({"src": []}, alpha=1.0, v_rest=0.0)
    with pytest.raises(ValueError, match="v_rest"):
        _lif_run(EPHAPTIC_RULE_NAME, {"alpha": 0.1})


def _network():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 100.0, size=(N, 3))
    pre, post = np.nonzero(rng.random((N, N)) < 0.1)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    weight = np.where(rng.random(pre.size) < 0.8, 2.0, -4.0)
    I_ext = rng.uniform(0.0, 2.0, size=(N_STEPS, N)).astype(np.float64)
    return pos, pre, post, weight, I_ext


def _lif_run(rule, rule_params):
    _, pre, post, weight, I_ext = _network()
    return simulate_lif_edge(
        make_lif_params(("simple",) * N), n_steps=N_STEPS, dt_ms=DT,
        edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay_steps=np.ones(pre.size, dtype=int), I_ext_pA=jnp.asarray(I_ext),
        record_v=True, hdp_rule=rule, hdp_rule_params=rule_params,
    )


def _lif_eph(alpha):
    pos = _network()[0]
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    return _lif_run(EPHAPTIC_RULE_NAME, ephaptic_rule_params(k, alpha=alpha, v_rest=0.0)), k


def test_lif_zero_coupling_equals_no_rule():
    base = _lif_run(None, None)
    eph, _ = _lif_eph(0.0)
    assert np.asarray(base["spikes"]).sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(np.asarray(eph["spikes"]), np.asarray(base["spikes"]))
    np.testing.assert_array_equal(np.asarray(eph["V_mV"]), np.asarray(base["V_mV"]))
    assert np.all(np.asarray(eph["b_final"]) == 0.0)


def test_lif_field_is_generated_by_activity():
    eph, k = _lif_eph(0.05)
    base = _lif_run(None, None)
    V_last = np.asarray(eph["V_mV"])[-1]
    expected = 0.05 * _dense(k, N) @ V_last
    # The LIF kernel runs in float32.
    np.testing.assert_allclose(np.asarray(eph["aux_final"]), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(eph["b_final"]), np.asarray(eph["aux_final"]))
    assert np.abs(expected).max() > 0, "fixture must produce a nonzero field"
    assert not np.array_equal(np.asarray(eph["spikes"]), np.asarray(base["spikes"])), (
        "a nonzero field must change the spikes in this fixture"
    )


@pytest.mark.parametrize("alpha", [-0.05, 0.02, 0.1])
def test_lif_coupling_sweep_finite_and_reproducible(alpha):
    a, _ = _lif_eph(alpha)
    b, _ = _lif_eph(alpha)
    assert np.all(np.isfinite(np.asarray(a["V_mV"])))
    np.testing.assert_array_equal(np.asarray(a["V_mV"]), np.asarray(b["V_mV"]))


def _izh_params(n):
    return IzhikevichParams(
        a=jnp.full((n,), 0.02), b=jnp.full((n,), 0.2), c=jnp.full((n,), -65.0),
        d=jnp.full((n,), 8.0), drive=jnp.full((n,), 5.0), sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)), v0=jnp.full((n,), -65.0), u0=jnp.full((n,), -13.0),
        source_scale=jnp.ones((n,)), labels=("E",) * n, layer_labels=("L4",) * n,
        source_calibration_status="x",
    )


def _izh_run(alpha):
    pos, pre, post, _, _ = _network()
    edges = EdgeList(
        pre=jnp.asarray(pre, jnp.int32), post=jnp.asarray(post, jnp.int32),
        weight=jnp.full((pre.size,), 0.5), receptor_index=jnp.zeros((pre.size,), jnp.int32),
        tau_ms=jnp.full((pre.size,), 2.0), source_calibration_status="x",
    )
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    return simulate_edge_recurrent_izhikevich_hdp_registered(
        _izh_params(N), edges, N_STEPS, DT, jax.random.PRNGKey(0),
        hdp_rule=EPHAPTIC_RULE_NAME,
        hdp_rule_params=ephaptic_rule_params(k, alpha=alpha, v_rest=-65.0),
    )


def test_izhikevich_field_changes_activity_and_zero_is_inert():
    _, s0, _, d0 = _izh_run(0.0)
    _, s1, _, d1 = _izh_run(0.02)
    assert np.asarray(s0).sum() > 0, "fixture must spike"
    assert np.all(np.asarray(d0["aux_final"]) == 0.0)
    assert np.abs(np.asarray(d1["aux_final"])).max() > 0
    assert not np.array_equal(np.asarray(s0), np.asarray(s1))
