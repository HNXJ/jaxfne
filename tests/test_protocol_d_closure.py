"""0.4.17-D Protocol D closure at D3."""

from __future__ import annotations

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    _izhikevich_dv_du_recovery_h_k,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_activity_h_k_rbd,
    simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery,
    simulate_edge_recurrent_izhikevich_owned_h_k_delayed,
    simulate_edge_recurrent_izhikevich_static_h_k_recovery,
)
from jaxfne.protocol_d_biological_rbs.d_closure import (
    load_d_closure_receipt,
    validate_d_closure_receipt,
)
from jaxfne.protocol_d_biological_rbs.d3_protocol import D3_SPEC_PATH


def test_d_closure_receipt_frozen():
    receipt = load_d_closure_receipt()
    assert receipt["status"] == "FROZEN"
    assert receipt["protocol_d_closed"] is True
    assert receipt["closed_at_checkpoint"] == "D3"


def test_d_closure_D4_not_authorized():
    receipt = load_d_closure_receipt()
    assert receipt["D4_status"] == "not_authorized"
    validate_d_closure_receipt(receipt)


def test_d_closure_figure_6_ladder():
    ladder = load_d_closure_receipt()["figure_6_ladder"]
    assert ladder["static_H_K_to_X"] == "demonstrated"
    assert ladder["activity_written_H_K_to_distinct_spike_adaptation"] == "not_supported"


def test_d_closure_next_milestone_E():
    receipt = load_d_closure_receipt()
    assert receipt["next_milestone"] == "0.4.17-E"
    assert receipt["next_checkpoint"] == "E0_specification"


def test_d3_protocol_receipt_points_to_closure():
    proto = json.loads((D3_SPEC_PATH.parent / "d3_protocol_receipt.json").read_text(encoding="utf-8"))
    assert proto["protocol_d_closed"] is True
    assert proto["D4_status"] == "not_authorized"
    assert proto["next_milestone"] == "0.4.17-E"


# --- H_K reaches V, u and the synaptic current only through the typed recovery map
# (du = a * (H_K * b * v - u)). Sites: static D1, dynamic D2a, owned E3 (delayed), activity D2b.
# Pattern follows test_protocol_h_rbd_h1c.py (_SITE_CASES to the end).

N_STEPS = 120
DT_MS = 0.5
TAU_K_MS = 100.0
TAU_A_MS = 25.0
SITES = ("static", "dynamic", "owned", "activity")

# Each case routes to one H_K site. Only the owned (E3) kernel accepts edge delays.
SITE_CASES = [
    pytest.param("static", 0, id="static"),
    pytest.param("dynamic", 0, id="dynamic"),
    pytest.param("owned", 0, id="owned_delay0"),
    pytest.param("owned", 2, id="owned_delay2"),
    pytest.param("activity", 0, id="activity"),
]


def _ring(delay=0):
    """Two-neuron ring: neuron 0 drives neuron 1 through a recurrent edge."""
    jdtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.full((2,), -65.0, dtype=jdtype),
        u0=jnp.zeros((2,), dtype=jdtype),
        a=jnp.full((2,), 0.02, dtype=jdtype),
        b=jnp.full((2,), 0.2, dtype=jdtype),
        c=jnp.full((2,), -65.0, dtype=jdtype),
        d=jnp.full((2,), 8.0, dtype=jdtype),
        drive=jnp.zeros((2,), dtype=jdtype),
        sign=jnp.ones((2,), dtype=jdtype),
        W=jnp.zeros((2, 2), dtype=jdtype),
        source_scale=jnp.ones((2,), dtype=jdtype),
        labels=("E", "E"),
        layer_labels=("L4", "L4"),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.asarray([0, 1], dtype=jnp.int32),
        post=jnp.asarray([1, 0], dtype=jnp.int32),
        weight=jnp.asarray([5.0, 5.0], dtype=jdtype),
        receptor_index=jnp.asarray([0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([2.0, 2.0], dtype=jdtype),
    )
    if delay:
        edges = replace(edges, delay_steps=jnp.full((2,), delay, dtype=jnp.int32))
    return params, edges


def _schedule():
    return jnp.zeros((N_STEPS, 2), dtype=jnp.float32).at[10:80, 0].set(35.0)


def _run_site(
    site, h, params, edges, *, n_steps=N_STEPS, drive_schedule=None, owned_dynamic=True
):
    key = jax.random.PRNGKey(0)
    h_arr = jnp.full((2,), h, dtype=jnp.float32)
    common = dict(dtype="float32", drive_schedule=drive_schedule, noise_scale=0.0)
    if site == "static":
        return simulate_edge_recurrent_izhikevich_static_h_k_recovery(
            params, edges, n_steps, DT_MS, key, h_k=h_arr, **common
        )
    if site == "dynamic":
        return simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery(
            params, edges, n_steps, DT_MS, key, h_k0=h_arr, tau_k_ms=TAU_K_MS, **common
        )
    if site == "owned":
        return simulate_edge_recurrent_izhikevich_owned_h_k_delayed(
            params,
            edges,
            n_steps,
            DT_MS,
            key,
            h_k0=h_arr,
            owner_mask=jnp.ones((2,), dtype=jnp.float32),
            tau_k_ms=TAU_K_MS,
            dynamic=owned_dynamic,
            **common,
        )
    if site == "activity":
        return simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
            params,
            edges,
            n_steps,
            DT_MS,
            key,
            h_k0=h_arr,
            tau_a_ms=TAU_A_MS,
            tau_k_ms=TAU_K_MS,
            kappa_ak=0.0,
            **common,
        )
    raise ValueError(site)


def _h_free(params, edges, *, n_steps=N_STEPS, drive_schedule=None):
    return simulate_edge_recurrent_izhikevich(
        params,
        edges,
        n_steps,
        DT_MS,
        jax.random.PRNGKey(0),
        dtype="float32",
        drive_schedule=drive_schedule,
        noise_scale=0.0,
        record_u_trace=True,
    )


def _h_k_record(site, st):
    """H_K per step. The static kernel keeps only its constant H_K_static."""
    if site == "static":
        return np.broadcast_to(np.asarray(st["H_K_static"]), (N_STEPS, 2))
    return np.asarray(st["H_K_trace"])


def _assert_same_run(site_out, free_out):
    v, s, _, st = site_out
    vf, sf, _, sfs = free_out
    np.testing.assert_array_equal(np.asarray(v), np.asarray(vf))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(sf))
    np.testing.assert_array_equal(np.asarray(st["u"]), np.asarray(sfs["u"]))
    if "u_trace" in st:
        np.testing.assert_array_equal(np.asarray(st["u_trace"]), np.asarray(sfs["u_trace"]))


@pytest.mark.parametrize("site, delay", SITE_CASES)
def test_pinned_h_k_one_equals_h_free_kernel(site, delay):
    """H_K = 1 for the whole run: V, spikes and u equal the H-free kernel."""
    params, edges = _ring(delay)
    sched = _schedule()
    site_out = _run_site(site, 1.0, params, edges, drive_schedule=sched)
    assert float(np.sum(np.asarray(site_out[1]))) > 0, "fixture must spike"
    np.testing.assert_array_equal(_h_k_record(site, site_out[3]), np.ones((N_STEPS, 2)))
    _assert_same_run(site_out, _h_free(params, edges, drive_schedule=sched))


@pytest.mark.parametrize("site, delay", SITE_CASES)
def test_moved_h_k_changes_output(site, delay):
    """H_K = 0.8 moves H_K off 1 and changes V or u relative to the H-free kernel."""
    params, edges = _ring(delay)
    sched = _schedule()
    v, _, _, st = _run_site(site, 0.8, params, edges, drive_schedule=sched)
    assert float(np.max(np.abs(_h_k_record(site, st) - 1.0))) > 1e-3, "fixture must move H_K"
    vf, _, _, sfs = _h_free(params, edges, drive_schedule=sched)
    moved_v = not np.array_equal(np.asarray(v), np.asarray(vf))
    moved_u = not np.array_equal(np.asarray(st["u"]), np.asarray(sfs["u"]))
    assert moved_v or moved_u


def test_recovery_map_puts_h_k_only_in_du():
    """The typed map: dv does not depend on H_K; du = a * (H_K * b * v - u)."""
    jdtype = jnp.float32
    v = jnp.asarray([-60.0, -50.0], dtype=jdtype)
    u = jnp.asarray([5.0, -3.0], dtype=jdtype)
    i_ext = jnp.asarray([10.0, 2.0], dtype=jdtype)
    a = jnp.full((2,), 0.02, dtype=jdtype)
    b = jnp.full((2,), 0.2, dtype=jdtype)
    h = jnp.asarray([0.8, 1.3], dtype=jdtype)
    dv_h, du_h = _izhikevich_dv_du_recovery_h_k(v, u, i_ext, a, b, h)
    dv_1, du_1 = _izhikevich_dv_du_recovery_h_k(v, u, i_ext, a, b, jnp.ones_like(h))
    assert not np.allclose(np.asarray(du_h), np.asarray(du_1)), "fixture must move du"
    np.testing.assert_array_equal(np.asarray(dv_h), np.asarray(dv_1))
    expected_du = np.asarray(a) * (np.asarray(h) * np.asarray(b) * np.asarray(v) - np.asarray(u))
    np.testing.assert_allclose(np.asarray(du_h), expected_du, rtol=1e-6)


@pytest.mark.parametrize("site, delay", [("static", 0), ("owned", 0), ("owned", 2)])
def test_fixed_h_k_equals_h_free_with_b_eff(site, delay):
    """Typed map b_eff = H_K * b: fixed H_K = 0.8 equals the H-free kernel with b * 0.8."""
    params, edges = _ring(delay)
    sched = _schedule()
    h = 0.8
    params_eff = replace(params, b=params.b * jnp.float32(h))
    site_out = _run_site(site, h, params, edges, drive_schedule=sched, owned_dynamic=False)
    assert float(np.sum(np.asarray(site_out[1]))) > 0, "fixture must spike"
    np.testing.assert_array_equal(
        _h_k_record(site, site_out[3]), np.full((N_STEPS, 2), h, dtype=np.float32)
    )
    _assert_same_run(site_out, _h_free(params_eff, edges, drive_schedule=sched))


@pytest.mark.parametrize("site", SITES)
def test_first_step_applies_h_k_in_du_only(site):
    """One step at H_K = 0.8: V_1 is the H_K-free dv; u_1 carries H_K * b (closed form)."""
    params, edges = _ring(0)
    h = 0.8
    v1, _, _, st = _run_site(site, h, params, edges, n_steps=1)
    v0 = np.asarray(params.v0, dtype=np.float64)
    u0 = np.asarray(params.u0, dtype=np.float64)
    a = np.asarray(params.a, dtype=np.float64)
    b = np.asarray(params.b, dtype=np.float64)
    dv = 0.04 * v0 * v0 + 5.0 * v0 + 140.0 - u0
    expected_v1 = v0 + DT_MS * dv
    expected_u1 = u0 + DT_MS * a * (h * b * v0 - u0)
    free_u1 = u0 + DT_MS * a * (b * v0 - u0)
    assert float(np.max(np.abs(expected_u1 - free_u1))) > 1e-3, "fixture must move du"
    np.testing.assert_allclose(np.asarray(v1)[0], expected_v1, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(st["u"]), expected_u1, rtol=1e-6)
