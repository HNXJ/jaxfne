"""0.5.3 item 5 (ENGINE W2): plasticity controls, adversarial (H5).

Enable / disable / clamp per rule AND per projection. Template: the
K_HDP=0 null (plus K_w_ctrl=0: K_HDP=0 alone leaves the restoring term
live). Clamp holds W exactly (pinned w0 + zero mask entries -> kernel
carries w_next = w with no float ops on those edges). d_H=1 throughout
(H8: no rule requires d_H>1).
"""

from __future__ import annotations

from dataclasses import replace as _replace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne import hdp_network as hn
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp
from jaxfne._hdp_registrable_kernel import (
    simulate_edge_recurrent_izhikevich_hdp_registered,
)
from jaxfne.hdp_rule import HDPRuleDescriptor, HDPRuleUpdate, register_hdp_rule

LEGACY_GAINS = {
    "K_HDP": 0.2,
    "K_ctrl": 0.3,
    "K_w_ctrl": 0.002,
    "alpha": 0.05,
    "tau_0_ms": 5.0,
    "noise_scale": 0.0,
}
RULE = "synthetic_presyn_gain"
RULE_PARAMS = {"k_h": 0.5, "k_w": 0.3, "gamma": 0.0}
W0 = np.array([0.2, 0.4], dtype=np.float32)


def _two_edge_fixture():
    params = IzhikevichParams(
        a=jnp.full((3,), 0.02),
        b=jnp.full((3,), 0.2),
        c=jnp.full((3,), -65.0),
        d=jnp.full((3,), 8.0),
        drive=jnp.array([0.0, 0.0, 0.0]),
        sign=jnp.ones((3,)),
        W=jnp.zeros((3, 3)),
        v0=jnp.full((3,), -65.0),
        u0=jnp.full((3,), -13.0),
        source_scale=jnp.ones((3,)),
        labels=("E", "E", "I"),
        layer_labels=("L4", "L4", "L4"),
        source_calibration_status="x",
    )
    edges = EdgeList(
        pre=jnp.array([0, 0], dtype=jnp.int32),
        post=jnp.array([1, 2], dtype=jnp.int32),
        weight=jnp.array([0.2, 0.4], dtype=jnp.float32),
        receptor_index=jnp.array([0, 0], dtype=jnp.int32),
        tau_ms=jnp.array([2.0, 2.0], dtype=jnp.float32),
        source_calibration_status="x",
    )
    drive = jnp.zeros((40, 3), dtype=jnp.float32).at[:, 0].set(8.0)
    return params, edges, drive


def _run_legacy(**kw):
    p, e, d = _two_edge_fixture()
    v, spk, src, diag = simulate_edge_recurrent_izhikevich_hdp(
        p, e, d.shape[0], 1.0, jax.random.PRNGKey(0), drive_schedule=d, **kw
    )
    return {"v": v, "spikes": spk, "sources": src, "diag": diag}


def _run_registered(**kw):
    p, e, d = _two_edge_fixture()
    v, spk, src, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        p,
        e,
        d.shape[0],
        1.0,
        jax.random.PRNGKey(0),
        drive_schedule=d,
        noise_scale=0.0,
        hdp_rule=RULE,
        hdp_rule_params=dict(RULE_PARAMS),
        **kw,
    )
    return {"v": v, "spikes": spk, "sources": src, "diag": diag}




# --- controls API unit -------------------------------------------------------


def test_enable_validates_and_copies():
    out = hn.enable_plasticity({"K_HDP": 0.01})
    assert out == {"K_HDP": 0.01}
    with pytest.raises(ValueError, match="unrecognized"):
        hn.enable_plasticity({"K_HDP_typo": 0.01})


def test_enable_rejects_stale_mask():
    with pytest.raises(ValueError, match="contradicts"):
        hn.enable_plasticity({"K_HDP": 0.01, "plasticity_mask": [1, 0]})


def test_disable_legacy_zeroes_both_gains():
    out = hn.disable_plasticity({"K_HDP": 0.2, "K_w_ctrl": 0.002, "alpha": 0.05})
    assert out["K_HDP"] == 0.0 and out["K_w_ctrl"] == 0.0
    assert out["alpha"] == 0.05 and "plasticity_mask" not in out


def test_disable_registered_zeroes_k_w():
    out = hn.disable_plasticity(
        {"hdp_rule": RULE, "hdp_rule_params": {"k_h": 0.5, "k_w": 0.3}}
    )
    assert out["hdp_rule_params"]["k_w"] == 0.0
    assert out["hdp_rule_params"]["k_h"] == 0.5


def test_disable_registered_without_k_w_raises():
    register_hdp_rule(
        HDPRuleDescriptor(name="ctl053_no_kw", default_params={"k_b": 0.3}),
        lambda ctx: HDPRuleUpdate(dH=ctx.H * 0.0),
    )
    with pytest.raises(ValueError, match="no 'k_w' coefficient"):
        hn.disable_plasticity({"hdp_rule": "ctl053_no_kw"})


def test_disable_with_mask_keeps_gains():
    out = hn.disable_plasticity({"K_HDP": 0.2}, mask=[0, 1])
    assert out["K_HDP"] == 0.2
    np.testing.assert_array_equal(np.asarray(out["plasticity_mask"]), [0.0, 1.0])


def test_clamp_attaches_mask_and_checks_value():
    out = hn.clamp_plasticity({"K_HDP": 0.2}, mask=[0, 1], value=0.35)
    assert out["K_HDP"] == 0.2
    np.testing.assert_array_equal(np.asarray(out["plasticity_mask"]), [0.0, 1.0])
    for bad in (float("nan"), float("inf"), "frozen"):
        with pytest.raises(ValueError, match="finite float"):
            hn.clamp_plasticity({"K_HDP": 0.2}, mask=[0, 1], value=bad)


def test_pin_projection_weights_exact():
    pinned = hn.pin_projection_weights(
        jnp.array([0.1, 0.2, 0.3]), np.array([True, False, True]), 0.35
    )
    np.testing.assert_array_equal(
        np.asarray(pinned), np.array([0.1, 0.35, 0.3], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="equal length"):
        hn.pin_projection_weights(jnp.array([0.1, 0.2]), np.array([True]), 0.35)


def test_projection_mask_builders():
    np.testing.assert_array_equal(hn.projection_mask(3, [0, 2]), [True, False, True])
    np.testing.assert_array_equal(
        hn.projection_mask(2, values=np.array([1, 0])), [True, False]
    )
    with pytest.raises(ValueError, match="out of range"):
        hn.projection_mask(2, [0, 5])
    with pytest.raises(ValueError, match="shape"):
        hn.projection_mask(3, values=np.array([True, False]))
    with pytest.raises(ValueError, match="exactly one"):
        hn.projection_mask(2)


# --- fixed-W identity proof (dedicated; do not assume) ------------------------


def test_fixedW_identity_proof_disabled_equals_null_bit_for_bit():
    """plasticity-off == the fixed-W path bit-for-bit.

    Three spellings of 'no plasticity' on the identical realized fixture:
    (a) explicit null gains (the pre-0.5.3 K_HDP=0 template path),
    (b) all-zero mask with gains ON, (c) item-5 disable_plasticity output.
    All w_traces are array-equal AND equal to w0; V/spikes/H agree too, so
    the controls change nothing but the intended freeze.
    """
    r_null = _run_legacy(
        K_HDP=0.0, K_w_ctrl=0.0, alpha=0.05, tau_0_ms=5.0, K_ctrl=0.3,
        noise_scale=0.0,
    )
    r_mask = _run_legacy(plasticity_mask=np.array([0, 0]), **LEGACY_GAINS)
    r_ctl = _run_legacy(**hn.disable_plasticity(dict(LEGACY_GAINS)))
    for name, r in (("null", r_null), ("mask", r_mask), ("controls", r_ctl)):
        wt = np.asarray(r["diag"]["w_trace"])
        assert np.array_equal(wt, np.broadcast_to(W0, wt.shape)), name
    for key in ("v", "spikes", "sources"):
        np.testing.assert_array_equal(np.asarray(r_null[key]), np.asarray(r_mask[key]))
        np.testing.assert_array_equal(np.asarray(r_null[key]), np.asarray(r_ctl[key]))
    for key in ("w_trace", "H_trace"):
        np.testing.assert_array_equal(
            np.asarray(r_null["diag"][key]), np.asarray(r_mask["diag"][key])
        )
        np.testing.assert_array_equal(
            np.asarray(r_null["diag"][key]), np.asarray(r_ctl["diag"][key])
        )


def test_H_evolves_while_W_disabled():
    """H != W separation: disabling fixes W exactly while H still moves."""
    r = _run_legacy(
        K_HDP=0.0, K_w_ctrl=0.0, alpha=0.05, tau_0_ms=5.0, K_ctrl=0.3,
        noise_scale=0.0,
    )
    H = np.asarray(r["diag"]["H_trace"])
    assert not np.array_equal(H[0], H[-1]), "H must evolve under drive"
    wt = np.asarray(r["diag"]["w_trace"])
    assert np.array_equal(wt, np.broadcast_to(W0, wt.shape))


# --- clamp-then-verify-exact --------------------------------------------------


def test_clamp_then_verify_exact_legacy():
    r = _run_legacy(plasticity_mask=np.array([0, 1]), **LEGACY_GAINS)
    wt = np.asarray(r["diag"]["w_trace"])
    assert np.array_equal(wt[:, 0], np.full(wt.shape[0], np.float32(0.2)))
    assert not np.array_equal(wt[:, 1], np.full(wt.shape[0], np.float32(0.4)))


def test_clamp_value_pinned_through_controls():
    p, e, d = _two_edge_fixture()
    mask = hn.projection_mask(2, [0])
    kw = hn.clamp_plasticity(dict(LEGACY_GAINS), mask=mask, value=0.35)
    pinned = hn.pin_projection_weights(np.asarray(e.weight), mask, 0.35)
    e2 = _replace(e, weight=jnp.asarray(pinned))
    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        p, e2, d.shape[0], 1.0, jax.random.PRNGKey(0), drive_schedule=d, **kw
    )
    wt = np.asarray(diag["w_trace"])
    assert np.array_equal(wt[:, 1], np.full(wt.shape[0], np.float32(0.35)))
    assert not np.array_equal(wt[:, 0], np.full(wt.shape[0], np.float32(0.2)))


# --- per-projection isolation -------------------------------------------------


def test_per_projection_isolation_legacy():
    r_a = _run_legacy(plasticity_mask=np.array([0, 1]), **LEGACY_GAINS)
    wa = np.asarray(r_a["diag"]["w_trace"])
    assert np.array_equal(wa[:, 0], np.full(wa.shape[0], np.float32(0.2)))
    assert not np.array_equal(wa[:, 1], np.full(wa.shape[0], np.float32(0.4)))
    r_b = _run_legacy(plasticity_mask=np.array([1, 0]), **LEGACY_GAINS)
    wb = np.asarray(r_b["diag"]["w_trace"])
    assert np.array_equal(wb[:, 1], np.full(wb.shape[0], np.float32(0.4)))
    assert not np.array_equal(wb[:, 0], np.full(wb.shape[0], np.float32(0.2)))


def test_per_projection_isolation_registered():
    r_a = _run_registered(plasticity_mask=np.array([0, 1]))
    wa = np.asarray(r_a["diag"]["w_trace"])
    assert np.array_equal(wa[:, 0], np.full(wa.shape[0], np.float32(0.2)))
    assert not np.array_equal(wa[:, 1], np.full(wa.shape[0], np.float32(0.4)))
    r_b = _run_registered(plasticity_mask=np.array([1, 0]))
    wb = np.asarray(r_b["diag"]["w_trace"])
    assert np.array_equal(wb[:, 1], np.full(wb.shape[0], np.float32(0.4)))
    assert not np.array_equal(wb[:, 0], np.full(wb.shape[0], np.float32(0.2)))


def test_disable_registered_fixes_W_exactly():
    kw = hn.disable_plasticity({"hdp_rule": RULE, "hdp_rule_params": dict(RULE_PARAMS)})
    p, e, d = _two_edge_fixture()
    _, _, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
        p, e, d.shape[0], 1.0, jax.random.PRNGKey(0), drive_schedule=d,
        noise_scale=0.0, hdp_rule=kw["hdp_rule"],
        hdp_rule_params=kw["hdp_rule_params"],
    )
    wt = np.asarray(diag["w_trace"])
    assert np.array_equal(wt, np.broadcast_to(W0, wt.shape))


# --- adversarial masks --------------------------------------------------------


@pytest.mark.parametrize("bad", [[1, 0, 0], [[1, 0], [0, 1]]])
def test_legacy_rejects_malformed_mask_shape(bad):
    with pytest.raises(ValueError, match="must have shape"):
        _run_legacy(plasticity_mask=np.asarray(bad), **LEGACY_GAINS)


def test_legacy_rejects_nonfinite_mask():
    with pytest.raises(ValueError, match="finite"):
        _run_legacy(plasticity_mask=np.array([0.0, float("nan")]), **LEGACY_GAINS)


def test_registered_rejects_malformed_mask():
    with pytest.raises(ValueError, match="must have shape"):
        _run_registered(plasticity_mask=np.array([1, 0, 0]))
    with pytest.raises(ValueError, match="finite"):
        _run_registered(plasticity_mask=np.array([float("inf"), 1.0]))


def test_legacy_rejects_mask_with_population_locality():
    with pytest.raises(ValueError, match="refused with population"):
        _run_legacy(
            plasticity_mask=np.array([1, 1]),
            h_state_locality="population",
            h_state_dim=2,
            controller_B=np.eye(2),
            m_ei_edge_mask=np.array([1.0, 0.0]),
            K_HDP=0.0, K_w_ctrl=0.0, noise_scale=0.0,
        )


# --- Model / hdp_network.run level --------------------------------------------


@pytest.fixture(scope="module")
def model8():
    return jtfne.construct(jtfne.suite2_net1_config(seed=1, n=8))


def test_model_simulate_mask_threading(model8):
    n = int(model8.params["edge_list"].n_edges)
    w0 = np.asarray(model8.params["edge_list"].weight)
    mask = np.ones(n)
    mask[: n // 2] = 0.0
    rt = jtfne.RuntimeConfig(
        enable_hdp=True,
        hdp_params={
            "K_HDP": 0.05, "K_ctrl": 0.15, "K_w_ctrl": 0.002,
            "alpha": 0.05, "tau_0_ms": 5.0, "noise_scale": 0.0,
            "plasticity_mask": mask,
        },
    )
    jtfne.simulate(model8, duration_ms=20.0, dt_ms=0.5, seed=4, runtime=rt)
    wt = np.asarray(model8.last_hdp_diagnostics()["w_trace"])
    assert np.array_equal(wt[:, : n // 2], np.broadcast_to(w0[: n // 2], wt[:, : n // 2].shape))
    assert not np.array_equal(
        wt[:, n // 2 :], np.broadcast_to(w0[n // 2 :], wt[:, n // 2 :].shape)
    )


def test_hdp_network_run_rejects_unknown_keys():
    cfg = hn.HDPColumnConfig(n_neurons=8, duration_ms=4.0, dt_ms=0.5, seed=5)
    model = hn.build_model(cfg)
    with pytest.raises(ValueError, match="unrecognized"):
        hn.run(model, cfg, 4.0, 5, hdp_kwargs={"K_HDP_typo": 0.1})


def test_hdp_network_run_disable_fixes_W():
    cfg = hn.HDPColumnConfig(n_neurons=8, duration_ms=10.0, dt_ms=0.5, seed=6)
    model = hn.build_model(cfg)
    w0 = np.asarray(model.params["edge_list"].weight)
    out = hn.run(model, cfg, 10.0, 6, hdp_kwargs=hn.disable_plasticity({"K_HDP": 0.01}))
    wt = np.asarray(out["diagnostics"]["w_trace"])
    assert np.array_equal(wt, np.broadcast_to(w0, wt.shape))
