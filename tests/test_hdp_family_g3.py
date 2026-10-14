"""Lane 7 (G3): registered HDP rules on the AdEx family kernel.

``simulate_adex_hdp_registered`` (``jaxfne/emitters_adex.py``) runs the same
conductance network as ``simulate_adex`` with per-edge ``|w|`` carried under
any registered rule via ``jaxfne/_hdp_family_kernel.py``. AdEx contexts carry
a zero ``u`` (no recovery counterpart); only ``edge_weight`` rules run here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._hdp_family_kernel import assert_freeze_allclose
from jaxfne.emitters_adex import (
    make_adex_params,
    simulate_adex,
    simulate_adex_hdp_registered,
)
from jaxfne.hdp_rule import (
    HDPRuleDescriptor,
    HDPRuleUpdate,
    list_registered_hdp_rules,
    register_hdp_rule,
)

DT_MS = 0.1
N_STEPS = 100
T_PULSE = 50
W0_NS = 3.0


@pytest.fixture(autouse=True)
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _two_cell_drive(*, pre_pulse: int, post_pulse: int | None = None):
    """One 1e5 pA step forcing a spike on cell 0 (and optionally cell 1)."""
    I_pA = np.zeros((N_STEPS, 2))
    I_pA[pre_pulse, 0] = 1e5
    if post_pulse is not None:
        I_pA[post_pulse, 1] = 1e5
    return I_pA


def _edge_kwargs(*, delay=20):
    return dict(
        edge_pre=[0],
        edge_post=[1],
        edge_weight_nS=[W0_NS],
        edge_receptor=[0],
        edge_delay_steps=[delay],
        receptor_reversal_mV=(0.0,),
        receptor_tau_ms=(2.0,),
    )


def _run_hdp(I_pA, rule, rule_params=None, *, delay=20, **kw):
    return simulate_adex_hdp_registered(
        make_adex_params(("E", "PV")),
        n_steps=N_STEPS,
        dt_ms=DT_MS,
        **_edge_kwargs(delay=delay),
        decay="exp",
        I_ext_pA=I_pA,
        record_v=True,
        hdp_rule=rule,
        hdp_rule_params=rule_params,
        **kw,
    )


def test_adex_hdp_float64_freeze():
    """synthetic_presyn_gain (k_h=0.5, k_w=0.1): one forced pre spike at
    step 50, arrival at step 70 drives H[1] 1.0 -> 1.05 and |w| 3.0 -> 3.03."""
    out = _run_hdp(
        _two_cell_drive(pre_pulse=T_PULSE),
        "synthetic_presyn_gain",
        {"k_h": 0.5, "k_w": 0.1, "gamma": 0.0},
    )
    spikes = np.asarray(out["spikes"])
    assert spikes.dtype == bool
    assert spikes.shape == (N_STEPS, 2)
    assert int(spikes.sum()) == 1, "fixture must force exactly one spike"
    assert spikes[T_PULSE, 0]
    H = np.asarray(out["H_trace"])
    w = np.asarray(out["w_trace_nS"])
    assert H.shape == (N_STEPS, 2) and H.dtype == np.float64
    assert w.shape == (N_STEPS, 1) and w.dtype == np.float64
    assert_freeze_allclose(H[69:72, 1], [1.0, 1.05, 1.05])
    assert_freeze_allclose(H[69:72, 0], [1.0, 1.0, 1.0])
    assert_freeze_allclose(w[69:72, 0], [W0_NS, 3.03, 3.03])
    assert_freeze_allclose(out["H_final"], [1.0, 1.05])
    assert_freeze_allclose(out["w_final_nS"], [3.03])
    assert np.asarray(out["aux_final"]).shape == (0,)
    assert out["hdp_rule"] == "synthetic_presyn_gain"
    assert out["hdp_rule_registered"] is True


def _pair_stdp_step(ctx):
    tau = jnp.asarray(ctx.rule_params.get("tau", 20.0), dtype=ctx.H.dtype)
    a_plus = jnp.asarray(ctx.rule_params.get("A_plus", 0.05), dtype=ctx.H.dtype)
    a_minus = jnp.asarray(ctx.rule_params.get("A_minus", 0.06), dtype=ctx.H.dtype)
    pre_sp = ctx.pre_sp if ctx.pre_sp is not None else ctx.spikes[ctx.pre]
    post_sp = ctx.spikes[ctx.post]
    tr_pre, tr_post = ctx.aux[:, 0], ctx.aux[:, 1]
    d_aux = jnp.stack(
        [-tr_pre / tau + pre_sp, -tr_post / tau + post_sp], axis=1
    )
    dw = a_plus * tr_pre * post_sp - a_minus * tr_post * pre_sp
    return HDPRuleUpdate(
        dH=jnp.zeros_like(ctx.H), d_aux=d_aux, d_theta={"edge_weight": dw}
    )


def _ensure_pair_stdp():
    if "l7_pair_stdp" not in list_registered_hdp_rules():
        register_hdp_rule(
            HDPRuleDescriptor(
                name="l7_pair_stdp",
                theta_targets=("edge_weight",),
                aux_coords=("tr_pre", "tr_post"),
                aux_layout="per_edge",
                default_params={"tau": 20.0, "A_plus": 0.05, "A_minus": 0.06},
            ),
            _pair_stdp_step,
        )


def test_pair_stdp_potentiates_causal_depresses_anticausal():
    """Test-local per-edge-trace STDP: pre->post (30/40) grows |w|,
    post->pre (40/30) shrinks it; aux carries the two traces."""
    _ensure_pair_stdp()
    causal = _run_hdp(
        _two_cell_drive(pre_pulse=30, post_pulse=40), "l7_pair_stdp", delay=1
    )
    anticausal = _run_hdp(
        _two_cell_drive(pre_pulse=40, post_pulse=30), "l7_pair_stdp", delay=1
    )
    for out, pre_t, post_t in (
        (causal, 30, 40),
        (anticausal, 40, 30),
    ):
        spikes = np.asarray(out["spikes"])
        assert spikes[pre_t, 0] and spikes[post_t, 1], (
            "fixture must spike both cells at the forced steps"
        )
    assert np.asarray(causal["aux_final"]).shape == (1, 2)
    w_causal = float(np.asarray(causal["w_final_nS"])[0])
    w_anti = float(np.asarray(anticausal["w_final_nS"])[0])
    assert w_causal > W0_NS, f"causal pair must potentiate: {w_causal}"
    assert w_anti < W0_NS, f"anticausal pair must depress: {w_anti}"


def test_identity_rule_matches_baseline_adex():
    """synthetic_presyn_gain defaults: spikes bit-identical to simulate_adex,
    |w| untouched, H pinned at the RBS reference 1."""
    I_pA = _two_cell_drive(pre_pulse=T_PULSE)
    hdp_out = _run_hdp(I_pA, "synthetic_presyn_gain")
    base_out = simulate_adex(
        make_adex_params(("E", "PV")),
        n_steps=N_STEPS,
        dt_ms=DT_MS,
        **_edge_kwargs(),
        decay="exp",
        I_ext_pA=I_pA,
        record_v=True,
    )
    assert np.array_equal(np.asarray(hdp_out["spikes"]), np.asarray(base_out["spikes"]))
    np.testing.assert_allclose(
        np.asarray(hdp_out["V_mV"]), np.asarray(base_out["V_mV"]), rtol=1e-12, atol=0.0
    )
    assert_freeze_allclose(hdp_out["w_final_nS"], [W0_NS])
    assert_freeze_allclose(hdp_out["H_final"], [1.0, 1.0])


def _bias_probe_step(ctx):
    return HDPRuleUpdate(
        dH=jnp.zeros_like(ctx.H),
        d_theta={"drive_bias": jnp.zeros_like(ctx.H)},
    )


def test_adex_hdp_fails_closed():
    """Unknown rule, unconsumed param key, and drive_bias rules (AdEx keeps
    magnitude-only |w|) all raise ValueError naming the cause."""
    _I = _two_cell_drive(pre_pulse=T_PULSE)
    with pytest.raises(ValueError, match="unknown registered hdp_rule"):
        _run_hdp(_I, "l7_no_such_rule")
    with pytest.raises(ValueError, match="unrecognized keys"):
        _run_hdp(_I, "synthetic_presyn_gain", {"k_h": 0.5, "k_typo": 1.0})
    if "l7_bias_probe" not in list_registered_hdp_rules():
        register_hdp_rule(
            HDPRuleDescriptor(
                name="l7_bias_probe", theta_targets=("drive_bias",)
            ),
            _bias_probe_step,
        )
    with pytest.raises(ValueError, match="drive_bias"):
        _run_hdp(_I, "l7_bias_probe")


def test_new_surface_is_advanced_not_public():
    """Lane 7 entry points are root-reachable without widening __all__."""
    assert hasattr(jtfne, "simulate_adex_hdp_registered")
    assert "simulate_adex_hdp_registered" not in jtfne.__all__
