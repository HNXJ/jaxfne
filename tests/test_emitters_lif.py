"""PCL LIF core kernel (jaxfne/emitters_lif.py): single-cell trajectory, floor, refractory, hook, refusals."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from jaxfne.emitters_lif import (
    LIF_CELL_TYPE_DEFAULTS,
    lif_step,
    make_lif_params,
    simulate_lif_edge,
)


_LIF_FLOAT_FIELDS = (
    "tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV",
    "tau_rp_ms", "tau_sra_ms", "delta_sra_mV",
)


def _as_f32(params):
    """Cast every float column of a LIFParams to float32 (labels untouched)."""
    return params._replace(
        **{f: jnp.asarray(getattr(params, f), jnp.float32) for f in _LIF_FLOAT_FIELDS}
    )


def _simple_f32():
    return _as_f32(make_lif_params(("simple",)))


def test_defaults_match_port_spec():
    d = LIF_CELL_TYPE_DEFAULTS["simple"]
    assert (d["tau_m_ms"], d["v_thresh_mV"], d["v_reset_mV"], d["v_min_mV"]) == (18.0, 10.0, -10.0, -20.0)
    assert (d["eta_rp_mV"], d["tau_rp_ms"]) == (10.0, 5.0)
    d = LIF_CELL_TYPE_DEFAULTS["complex"]
    assert (d["tau_m_ms"], d["v_thresh_mV"], d["v_reset_mV"], d["v_min_mV"]) == (50.0, 3.0, -10.0, -20.0)
    assert d["delta_sra_mV"] == 0.0


def test_make_lif_params_refusals():
    try:
        make_lif_params(("pyramidal",))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown cell type must raise")
    try:
        make_lif_params(("simple",), {"simple": {"no_such_field": 1.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown field must raise")
    try:
        make_lif_params(("simple",), {"complex": {"tau_m_ms": 9.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("override for absent type must raise")


def test_membrane_decay_closed_form():
    p = _simple_f32()
    out = simulate_lif_edge(p, n_steps=10, dt_ms=0.5, V0_mV=jnp.array([5.0], jnp.float32))
    got = float(out["V"][0])
    want = 5.0 * math.exp(-5.0 / 18.0)
    assert abs(got - want) < 1e-5, (got, want)
    assert not bool(out["spikes"].any())


def test_input_adds_weight_and_floor():
    p = _simple_f32()
    out = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        I_ext_pA=jnp.array([[7.0]], jnp.float32),
    )
    assert abs(float(out["V"][0]) - 7.0) < 1e-6
    out = simulate_lif_edge(
        p, n_steps=1, dt_ms=0.5,
        I_ext_pA=jnp.array([[-50.0]], jnp.float32),
    )
    assert float(out["V"][0]) == -20.0  # v_min floor, Neuron.cpp:26
    assert not bool(out["spikes"].any())


def test_refractory_subtraction_per_arrival():
    p = _simple_f32()
    V = jnp.array([-10.0], jnp.float32)  # just spiked and reset
    A = jnp.array([0.0], jnp.float32)
    t_last = jnp.array([0.0], jnp.float32)
    (V1, A1, t1), spk = lif_step((V, A, t_last), jnp.array([0.0]), jnp.array([1.0]),
                                 0.5, p, 0.5)
    want = float(-10.0 * math.exp(-0.5 / 18.0) - 10.0 * math.exp(-0.5 / 5.0))
    assert abs(float(V1[0]) - want) < 1e-5, (float(V1[0]), want)
    assert not bool(spk[0])
    # No arrival: decay only, no refractory subtraction (event-driven C++ does nothing).
    (V2, _, _), _ = lif_step((V, A, t_last), jnp.array([0.0]), jnp.array([0.0]),
                             0.5, p, 0.5)
    assert abs(float(V2[0]) - float(-10.0 * math.exp(-0.5 / 18.0))) < 1e-6


def test_threshold_reset_and_adaptation_increment():
    p = _as_f32(make_lif_params(("simple",), {"simple": {"delta_sra_mV": 2.0}}))
    V = jnp.array([9.0], jnp.float32)
    A = jnp.array([0.0], jnp.float32)
    t_last = jnp.array([-1.0e30], jnp.float32)
    (V1, A1, t1), spk = lif_step((V, A, t_last), jnp.array([2.0]), jnp.array([1.0]),
                                 0.5, p, 0.5)
    assert bool(spk[0])
    assert float(V1[0]) == -10.0  # reset
    assert abs(float(A1[0]) - 2.0) < 1e-6  # spike-rate adaptation increment
    assert float(t1[0]) == 0.5


def test_delayed_drive_spikes_postsynaptic_cell():
    p = _as_f32(make_lif_params(("simple", "simple")))
    I_ext = np.zeros((3, 2), np.float32)
    I_ext[0, 0] = 30.0
    out = simulate_lif_edge(
        p, n_steps=3, dt_ms=0.5,
        edge_pre=np.array([0]), edge_post=np.array([1]),
        edge_weight=np.array([12.0]), edge_delay_steps=np.array([1]),
        I_ext_pA=jnp.asarray(I_ext),
    )
    spk = np.asarray(out["spikes"])
    assert spk[0, 0] and not spk[0, 1]
    assert spk[1, 1], spk  # 12 mV arrival crosses threshold 10
    assert not spk[1, 0]


def test_rule_hook_context_and_static_weights():
    from jaxfne.hdp_rule import (
        HDPRuleDescriptor,
        HDPRuleUpdate,
        list_registered_hdp_rules,
        register_hdp_rule,
    )

    seen = []

    def _probe(ctx):
        # Shapes and pre_sp presence are static (visible at trace time); the
        # per-step invocation itself is proven numerically via H_final below
        # (lax.scan traces body once, so a call counter would read 1).
        seen.append((int(ctx.n_neurons), float(ctx.dt), tuple(ctx.pre.shape),
                     tuple(ctx.post.shape), tuple(ctx.spikes.shape),
                     ctx.pre_sp is not None))
        return HDPRuleUpdate(dH=jnp.full_like(ctx.H, 0.4),
                             d_theta={"edge_weight": jnp.zeros_like(ctx.w)})

    name = "test_lif_probe_hook"
    if name not in list_registered_hdp_rules():
        register_hdp_rule(
            HDPRuleDescriptor(name=name, h_coords=("H",), theta_targets=("edge_weight",),
                              scope="node", default_params={}),
            _probe,
        )
    p = _simple_f32()
    w0 = np.array([5.0])
    out = simulate_lif_edge(
        p, n_steps=3, dt_ms=0.5,
        edge_pre=np.array([0]), edge_post=np.array([0]),
        edge_weight=w0, edge_delay_steps=np.array([1]),
        hdp_rule=name,
    )
    assert len(seen) == 1, seen
    n, dt, pre_sh, post_sh, spk_sh, has_pre_sp = seen[0]
    assert (n, pre_sh, post_sh, spk_sh, has_pre_sp) == (1, (1,), (1,), (1,), True)
    assert abs(dt - 0.5) < 1e-12
    assert float(np.asarray(out["w_final"])[0]) == 5.0  # zero drive: weights static
    # dH = 0.4/ms integrated every step: H = 1 + 3 * 0.5 * 0.4 proves the rule
    # observed all three steps.
    assert abs(float(np.asarray(out["H_final"])[0]) - 1.6) < 1e-6


def test_kernel_refusals():
    p = _simple_f32()
    bad = [
        dict(n_steps=0, dt_ms=0.5),
        dict(n_steps=3, dt_ms=0.0),
        dict(n_steps=3, dt_ms=-0.5),
    ]
    for kw in bad:
        try:
            simulate_lif_edge(p, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"must refuse {kw}")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, edge_pre=np.array([0, 0]),
                          edge_post=np.array([0]), edge_weight=np.array([1.0]),
                          edge_delay_steps=np.array([1]))
    except ValueError:
        pass
    else:
        raise AssertionError("ragged edge arrays must raise")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, edge_pre=np.array([0]),
                          edge_post=np.array([0]), edge_weight=np.array([1.0]),
                          edge_delay_steps=np.array([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("delay < 1 must raise")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, hdp_rule="no_such_rule")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown hdp_rule must raise")
    try:
        simulate_lif_edge(p, n_steps=2, dt_ms=0.5, hdp_rule_params={"eta": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("rule params without a rule must raise")
