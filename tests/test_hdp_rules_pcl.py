"""Core PCL STDP rule (jaxfne/hdp_rules_pcl.py): paper one-pair values, normalization, refusals."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

import jaxfne.hdp_rules_pcl as pcl
from jaxfne.emitters_lif import make_lif_params
from jaxfne.hdp_rule import is_registered_hdp_rule, list_registered_hdp_rules

TAU = 7.0
WMAX = 3.0
F = 0.33
ETA = 0.00077 * 50.0  # Table 2 rate scaled by NORM_FACTOR, as the port passes it
DT = 0.5


def _lif1(n=1):
    p = make_lif_params(("simple",) * n)
    return p._replace(
        **{f: jnp.asarray(getattr(p, f), jnp.float32)
           for f in ("tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV",
                     "tau_rp_ms", "tau_sra_ms", "delta_sra_mV")}
    )


def _params(n, tau=TAU, wmax=WMAX, eta=ETA, f=F, lam=0.0, norm=0.0, group=(0,), n_groups=1):
    from jaxfne.emitters_lif import simulate_lif_edge  # noqa: F401  (kernel import surface)
    return pcl.rule_params(
        tau=np.full(n, tau), w_max=np.full(n, wmax), eta=np.full(n, eta),
        f=np.full(n, f), lam=np.full(n, lam), norm=np.full(n, norm),
        group=np.asarray(group, np.int32), n_groups=n_groups,
    )


def test_rule_registered_and_params_refused():
    assert is_registered_hdp_rule("pcl_stdp"), list_registered_hdp_rules()
    from jaxfne.hdp_rule import check_hdp_rule_params

    check_hdp_rule_params("pcl_stdp", None)
    try:
        check_hdp_rule_params("pcl_stdp", {"no_such_gain": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown rule-param key must raise")


def test_one_pair_ltp_matches_paper_eq():
    """Pre at t=0, post at t=2 ms: dw = (wmax-w) f eta exp(-2/7), norm off."""
    from jaxfne.emitters_lif import simulate_lif_edge

    p = _lif1(2)
    I_ext = np.zeros((5, 2), np.float32)
    I_ext[0, 0] = 30.0  # pre spikes at t = 0
    I_ext[4, 1] = 30.0  # post spikes at t = 2.0 ms
    out = simulate_lif_edge(
        p, n_steps=5, dt_ms=DT,
        edge_pre=np.array([0]), edge_post=np.array([1]),
        edge_weight=np.array([1.0]), edge_delay_steps=np.array([1]),
        I_ext_pA=jnp.asarray(I_ext), V0_mV=jnp.array([0.0, -5.0], jnp.float32),
        hdp_rule="pcl_stdp", hdp_rule_params=_params(1),
    )
    # The pre arrival lands at step 1 (t = 0.5); the trace decays 3 more steps
    # to the post spike at step 4: A = exp(-1.5/7).
    want = 1.0 + (WMAX - 1.0) * F * ETA * math.exp(-1.5 / TAU)
    got = float(np.asarray(out["w_final"])[0])
    assert abs(got - want) < 1e-5, (got, want)


def test_ltp_ltd_pair_matches_paper_eqs():
    """Post at t=0 and t=4, pre arrival at t=2: LTP on exp(-2/7)-decayed trace,
    LTD on the deferred sum; both soft-bound as the paper's Eqs. 4-11."""
    from jaxfne.emitters_lif import simulate_lif_edge

    p = _lif1(2)
    I_ext = np.zeros((9, 2), np.float32)
    I_ext[0, 1] = 30.0  # post spikes at t = 0
    I_ext[8, 1] = 30.0  # post spikes again at t = 4.0 ms
    I_ext[0, 0] = 30.0  # pre spikes at t = 0; arrival lands at step 1 (t = 0.5)
    out = simulate_lif_edge(
        p, n_steps=9, dt_ms=DT,
        edge_pre=np.array([0]), edge_post=np.array([1]),
        edge_weight=np.array([2.0]), edge_delay_steps=np.array([1]),
        I_ext_pA=jnp.asarray(I_ext), V0_mV=jnp.array([0.0, -5.0], jnp.float32),
        hdp_rule="pcl_stdp", hdp_rule_params=_params(1),
    )
    # LTP trace at the second post spike: 7 decay steps from the arrival.
    ltp = (WMAX - 2.0) * F * ETA * math.exp(-3.5 / TAU)
    # Deferred LTD: at the arrival step Y is still 1.0 (the post spike sets it
    # with no same-step decay), so B accumulates one decay: exp(-0.5/7).
    ltd = 2.0 * F * ETA * math.exp(-0.5 / TAU)
    want = 2.0 + ltp - ltd
    got = float(np.asarray(out["w_final"])[0])
    assert abs(got - want) < 1e-5, (got, want, ltp, ltd)


def test_group_l1_normalization_invariant():
    """Two edges, one group, norm on: after the post spike the group sums to lam."""
    from jaxfne.emitters_lif import simulate_lif_edge

    p = make_lif_params(("simple", "simple", "simple"))
    p = p._replace(
        **{f: jnp.asarray(getattr(p, f), jnp.float32)
           for f in ("tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV",
                     "tau_rp_ms", "tau_sra_ms", "delta_sra_mV")}
    )
    I_ext = np.zeros((3, 3), np.float32)
    I_ext[1, 2] = 30.0  # post (cell 2) spikes at step 1; no pre spikes anywhere
    out = simulate_lif_edge(
        p, n_steps=3, dt_ms=DT,
        edge_pre=np.array([0, 1]), edge_post=np.array([2, 2]),
        edge_weight=np.array([3.0, 1.0]), edge_delay_steps=np.array([1, 1]),
        I_ext_pA=jnp.asarray(I_ext),
        hdp_rule="pcl_stdp",
        hdp_rule_params=_params(2, lam=10.0, norm=1.0, group=(0, 0), n_groups=1),
    )
    got = np.asarray(out["w_final"])
    assert abs(float(got.sum()) - 10.0) < 1e-5, got
    assert abs(float(got[0]) - 7.5) < 1e-5 and abs(float(got[1]) - 2.5) < 1e-5, got
    assert bool((got >= 0.0).all())
