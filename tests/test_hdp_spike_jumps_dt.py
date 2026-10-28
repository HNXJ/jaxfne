"""Spike terms are per-event jumps, so their effect does not depend on dt.

The two probe rules (``synthetic_presyn_gain``, ``eligibility_trace_gain``) add
``k_h`` to the post-synaptic H once per presynaptic arrival. With ``gamma = 0``
and one arrival, H ends at ``1 + k_h`` at every ``dt_ms``. The checks run through
each kernel the probe rules use: the registered Izhikevich kernel, the LIF step
(``emitters_lif``) and the AdEx family kernel (``_hdp_family_kernel``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne._hdp_family_kernel import apply_registered_rule_update
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.emitters_adex import make_adex_params, simulate_adex_hdp_registered
from jaxfne.emitters_lif import make_lif_params, simulate_lif_edge
from jaxfne.hdp_rule import (
    HDPRuleDescriptor,
    HDPRuleUpdate,
    check_hdp_rule_params,
    expected_aux_shape,
    get_hdp_rule,
    list_registered_hdp_rules,
    register_hdp_rule,
    superpose_hdp_rules,
)

K_H = 0.5
DT_VALUES = (0.1, 0.5)
PROBE_PARAMS = {"k_h": K_H, "k_w": 0.1, "gamma": 0.0}
PROBE_RULES = ("synthetic_presyn_gain", "eligibility_trace_gain")
UNDECLARED_RULE = "w2b_undeclared_jump_probe"
BIAS_DESCRIPTOR = HDPRuleDescriptor(name="w2b_bias_jump_probe", theta_targets=("drive_bias",))
ZERO_RULE = "w2b_zero_probe"
COMPOSITE_RULE = "w2b_synthetic_plus_zero"


def _izhikevich_params() -> IzhikevichParams:
    return IzhikevichParams(
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


def _run_registrable(rule, rule_params, dt):
    """Cell 0 forced to spike at step 0 over a zero-delay edge 0 -> 1."""
    n_steps = 2
    edges = EdgeList(
        pre=jnp.array([0], dtype=jnp.int32),
        post=jnp.array([1], dtype=jnp.int32),
        weight=jnp.array([0.2], dtype=jnp.float32),
        receptor_index=jnp.array([0], dtype=jnp.int32),
        tau_ms=jnp.array([2.0], dtype=jnp.float32),
        source_calibration_status="x",
    )
    forced = np.zeros((n_steps, 2), dtype=np.float32)
    forced[0, 0] = 1.0
    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        _izhikevich_params(), edges, n_steps, dt, jax.random.PRNGKey(0),
        hdp_rule=rule, hdp_rule_params=rule_params,
        forced_spikes=jnp.asarray(forced),
    )
    return np.asarray(spikes), np.asarray(diag["H_final"])


def _run_lif(rule, rule_params, dt):
    """Cell 0 driven at step 0; delay 1 puts its arrival at step 1."""
    n_steps = 10
    I_ext = np.zeros((n_steps, 2), dtype=np.float32)
    I_ext[0, 0] = 30.0
    out = simulate_lif_edge(
        make_lif_params(("simple", "simple")), n_steps=n_steps, dt_ms=dt,
        edge_pre=np.array([0]), edge_post=np.array([1]),
        edge_weight=np.array([5.0]), edge_delay_steps=np.array([1]),
        I_ext_pA=jnp.asarray(I_ext), hdp_rule=rule, hdp_rule_params=rule_params,
    )
    return np.asarray(out["spikes"]), np.asarray(out["H_final"])


def _run_adex(rule, rule_params, dt):
    """One 1e5 pA step on cell 0 at step 10; delay 20 steps puts arrival at step 30."""
    n_steps = 100
    I_pA = np.zeros((n_steps, 2))
    I_pA[10, 0] = 1e5
    out = simulate_adex_hdp_registered(
        make_adex_params(("E", "PV")), n_steps=n_steps, dt_ms=dt,
        edge_pre=[0], edge_post=[1], edge_weight_nS=[3.0], edge_receptor=[0],
        edge_delay_steps=[20], receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0,),
        decay="exp", I_ext_pA=I_pA, record_v=True,
        hdp_rule=rule, hdp_rule_params=rule_params,
    )
    return np.asarray(out["spikes"]), np.asarray(out["H_final"])


RUNNERS = {"registrable": _run_registrable, "lif": _run_lif, "adex": _run_adex}


@pytest.mark.parametrize("dt", DT_VALUES)
@pytest.mark.parametrize("rule", PROBE_RULES)
@pytest.mark.parametrize("kernel", sorted(RUNNERS))
def test_presyn_arrival_adds_k_h_once_at_any_dt(kernel, rule, dt):
    spikes, H = RUNNERS[kernel](rule, PROBE_PARAMS, dt)
    assert int(spikes[:, 0].sum()) == 1, "fixture must spike the pre cell once"
    assert int(spikes[:-1, 0].sum()) == 1, "the spike must arrive inside the window"
    np.testing.assert_allclose(H.reshape(-1)[1], 1.0 + K_H, atol=1e-6)


def _zero_step(ctx):
    return HDPRuleUpdate(dH=jnp.zeros_like(ctx.H))


def _undeclared_step(ctx):
    return HDPRuleUpdate(
        dH=jnp.zeros_like(ctx.H), j_theta={"drive_bias": jnp.zeros_like(ctx.H)}
    )


def _ensure_registered(name, descriptor, step_fn):
    if name not in list_registered_hdp_rules():
        register_hdp_rule(descriptor, step_fn)


@pytest.mark.parametrize("dt", DT_VALUES)
def test_composite_keeps_presyn_jump(dt):
    _ensure_registered(
        ZERO_RULE, HDPRuleDescriptor(name=ZERO_RULE, default_params={}), _zero_step
    )
    if COMPOSITE_RULE not in list_registered_hdp_rules():
        superpose_hdp_rules(COMPOSITE_RULE, ["synthetic_presyn_gain", ZERO_RULE])
    spikes, H = _run_registrable(
        COMPOSITE_RULE, {"synthetic_presyn_gain": PROBE_PARAMS}, dt
    )
    assert int(spikes[:, 0].sum()) == 1, "fixture must spike the pre cell once"
    H = np.asarray(H)
    assert H.shape == (2, 2)
    np.testing.assert_allclose(H[1, 0], 1.0 + K_H, atol=1e-6)
    np.testing.assert_allclose(H[1, 1], 1.0, atol=1e-6)


@pytest.mark.parametrize("kernel", sorted(RUNNERS))
def test_undeclared_jump_key_is_refused(kernel):
    _ensure_registered(
        UNDECLARED_RULE,
        HDPRuleDescriptor(name=UNDECLARED_RULE, default_params={}),
        _undeclared_step,
    )
    with pytest.raises(ValueError, match="undeclared"):
        RUNNERS[kernel](UNDECLARED_RULE, {}, 0.1)


# (rule, exc tag, w0, expected w): the family integrator's |w| path (exc None,
# exc True, exc False with a negative weight) and the eligibility rule's
# zero-aux weight drive.
FAMILY_CASES = (
    ("synthetic_presyn_gain", "none", 3.0, 3.3),
    ("synthetic_presyn_gain", "exc", 3.0, 3.3),
    ("synthetic_presyn_gain", "inh", -3.0, -3.3),
    ("eligibility_trace_gain", "none", 3.0, 3.0),
)


@pytest.mark.parametrize("dt", DT_VALUES)
@pytest.mark.parametrize("rule, exc, w0, w_expected", FAMILY_CASES)
def test_family_integrator_jumps_are_dt_free(rule, exc, w0, w_expected, dt):
    descriptor, rule_step = get_hdp_rule(rule)
    n = 2
    aux_shape = expected_aux_shape(descriptor, n_neurons=n, n_edges=1)
    exc_mask = {
        "none": None,
        "exc": jnp.array([True]),
        "inh": jnp.array([False]),
    }[exc]
    H_n, w_n, aux_n, _ = apply_registered_rule_update(
        descriptor=descriptor, rule_step=rule_step,
        H=jnp.ones((n,), jnp.float32), aux=jnp.zeros(aux_shape, jnp.float32),
        bias=jnp.zeros((n,), jnp.float32), v=jnp.zeros((n,), jnp.float32),
        u=jnp.zeros((n,), jnp.float32), spikes=jnp.ones((n,), jnp.float32),
        prev_spikes=jnp.zeros((n,), jnp.float32),
        syn_state=jnp.zeros((1,), jnp.float32),
        w=jnp.array([w0], jnp.float32), pre=jnp.array([0], jnp.int32),
        post=jnp.array([1], jnp.int32), exc_mask=exc_mask,
        dt=jnp.asarray(dt, jnp.float32), n_neurons=n,
        rule_params=check_hdp_rule_params(rule, PROBE_PARAMS),
        key=jax.random.PRNGKey(0), pre_sp=None, rule_name=rule,
    )
    np.testing.assert_allclose(np.asarray(H_n)[1], 1.0 + K_H, atol=1e-6)
    np.testing.assert_allclose(np.asarray(w_n), [w_expected], atol=1e-6)
    expected_aux = (
        np.ones(aux_shape) if rule == "eligibility_trace_gain" else np.zeros(aux_shape)
    )
    np.testing.assert_allclose(np.asarray(aux_n), expected_aux, atol=1e-6)


def _bias_jump_step(ctx):
    return HDPRuleUpdate(
        dH=jnp.zeros_like(ctx.H),
        j_theta={"drive_bias": jnp.full_like(ctx.H, 0.25)},
    )


@pytest.mark.parametrize("dt", DT_VALUES)
def test_family_integrator_drive_bias_jump_is_dt_free(dt):
    n = 2
    H_n, w_n, _, b_n = apply_registered_rule_update(
        descriptor=BIAS_DESCRIPTOR, rule_step=_bias_jump_step,
        H=jnp.ones((n,), jnp.float32), aux=jnp.zeros((0,), jnp.float32),
        bias=jnp.zeros((n,), jnp.float32), v=jnp.zeros((n,), jnp.float32),
        u=jnp.zeros((n,), jnp.float32), spikes=jnp.zeros((n,), jnp.float32),
        prev_spikes=jnp.zeros((n,), jnp.float32),
        syn_state=jnp.zeros((1,), jnp.float32),
        w=jnp.array([3.0], jnp.float32), pre=jnp.array([0], jnp.int32),
        post=jnp.array([1], jnp.int32), exc_mask=None,
        dt=jnp.asarray(dt, jnp.float32), n_neurons=n,
        rule_params={}, key=jax.random.PRNGKey(0), pre_sp=None,
        rule_name=BIAS_DESCRIPTOR.name,
    )
    np.testing.assert_allclose(np.asarray(b_n), [0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(np.asarray(H_n), [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(w_n), [3.0], atol=1e-6)
