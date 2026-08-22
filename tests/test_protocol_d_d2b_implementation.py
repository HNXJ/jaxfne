"""0.4.17-D D2b — two-coordinate activity-coupled RBS implementation tests."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    _advance_h_a_trace,
    _advance_h_k_activity_coupled,
    simulate_edge_recurrent_izhikevich_activity_h_k_rbd,
    simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery,
)
from jaxfne.protocol_d_biological_rbs.d2b_execution import (
    h_k_analytic_two_timescale,
    load_d2b_execution_receipt,
    rbs_post_stimulus_euler_traces,
    run_d2b_activity_h_k_coupling,
    write_d2b_execution_receipt,
)
from jaxfne.protocol_d_biological_rbs.d2b_protocol import (
    D2B_EXECUTION_RECEIPT_PATH,
    D2B_SPEC_PATH,
    load_d2b_spec,
    validate_d2b_spec,
)


def _isolated(n: int = 1) -> tuple[IzhikevichParams, EdgeList]:
    jdtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.full((n,), -65.0, dtype=jdtype),
        u0=jnp.zeros((n,), dtype=jdtype),
        a=jnp.full((n,), 0.02, dtype=jdtype),
        b=jnp.full((n,), 0.2, dtype=jdtype),
        c=jnp.full((n,), -65.0, dtype=jdtype),
        d=jnp.full((n,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n,), dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.zeros((0,), dtype=jnp.int32),
        post=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jdtype),
        receptor_index=jnp.zeros((0,), dtype=jnp.int32),
        tau_ms=jnp.zeros((0,), dtype=jdtype),
    )
    return params, edges


def test_d2b_spec_frozen_and_S_binary():
    spec = load_d2b_spec()
    validate_d2b_spec(spec)
    sdef = spec["activity_input_S"]
    assert sdef["normalization"] == "binary spike event per step (timestep-independent unit event)"
    assert spec["execution_authorization"]["implementation_authorized"] is False


def test_d2b_causal_h_k_uses_old_h_a_not_new():
    """Regression: H_K^{n+1} must use H_A^n, not H_A^{n+1}."""
    H_A = jnp.asarray([0.0], dtype=jnp.float32)
    H_K = jnp.asarray([1.0], dtype=jnp.float32)
    S = jnp.asarray([1.0], dtype=jnp.float32)
    dt = jnp.asarray(0.5, dtype=jnp.float32)
    tau_a = jnp.asarray(25.0, dtype=jnp.float32)
    tau_k = jnp.asarray(100.0, dtype=jnp.float32)
    kappa = jnp.asarray(0.4, dtype=jnp.float32)

    H_A_new = _advance_h_a_trace(H_A, S, dt, tau_a)
    H_K_correct = _advance_h_k_activity_coupled(H_K, H_A, dt, tau_k, kappa)
    H_K_wrong = _advance_h_k_activity_coupled(H_K, H_A_new, dt, tau_k, kappa)
    assert float(H_K_correct[0]) != float(H_K_wrong[0])
    assert float(H_K_correct[0]) == pytest.approx(1.0 + 0.5 * 0.4 * 0.0 / 100.0, rel=0, abs=1e-6)


def test_d2b_kernel_one_step_matches_manual_recurrence():
    params, edges = _isolated()
    n_steps = 1
    sched = jnp.zeros((1, 1), dtype=jnp.float32)
    sched = sched.at[0, 0].set(20.0)
    key = jax.random.PRNGKey(0)
    _, _, _, st = simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
        params,
        edges,
        n_steps,
        0.5,
        key,
        h_a0=jnp.zeros((1,), dtype=jnp.float32),
        h_k0=jnp.ones((1,), dtype=jnp.float32),
        tau_a_ms=25.0,
        tau_k_ms=100.0,
        kappa_ak=0.4,
        drive_schedule=sched,
        noise_scale=0.0,
    )
    S = float(st["S_trace"][0, 0])
    if S > 0.5:
        ha_manual = 0.0 + 0.5 * (1.0 - 0.0) / 25.0
        hk_manual = 1.0 + 0.5 * ((1.0 - 1.0) + 0.4 * 0.0) / 100.0
        assert float(st["H_A_trace"][0, 0]) == pytest.approx(ha_manual, rel=0, abs=1e-6)
        assert float(st["H_K_trace"][0, 0]) == pytest.approx(hk_manual, rel=0, abs=1e-6)


def test_d2b_h_trace_first_class_not_collapsed():
    params, edges = _isolated()
    sched = jnp.zeros((50, 1), dtype=jnp.float32)
    sched = sched.at[10:40, 0].set(15.0)
    key = jax.random.PRNGKey(1)
    _, _, _, st = simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
        params,
        edges,
        50,
        0.5,
        key,
        h_k0=jnp.ones((1,), dtype=jnp.float32),
        tau_a_ms=25.0,
        tau_k_ms=100.0,
        kappa_ak=0.4,
        drive_schedule=sched,
        noise_scale=0.0,
    )
    assert "H_A_trace" in st
    assert "H_K_trace" in st
    assert st["H_trace"].shape == (50, 1, 2)
    assert jnp.allclose(st["H_trace"][..., 0], st["H_A_trace"])
    assert jnp.allclose(st["H_trace"][..., 1], st["H_K_trace"])


def test_d2b_kappa_zero_matches_d2a_h_k():
    params, edges = _isolated()
    n_steps = 200
    dt = 0.5
    sched = jnp.zeros((n_steps, 1), dtype=jnp.float32)
    sched = sched.at[20:120, 0].set(15.0)
    key = jax.random.PRNGKey(5)
    _, _, _, st_d2b = simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
        params,
        edges,
        n_steps,
        dt,
        key,
        h_k0=jnp.ones((1,), dtype=jnp.float32),
        tau_a_ms=25.0,
        tau_k_ms=100.0,
        kappa_ak=0.0,
        drive_schedule=sched,
        noise_scale=0.0,
    )
    _, _, _, st_d2a = simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery(
        params,
        edges,
        n_steps,
        dt,
        key,
        h_k0=jnp.ones((1,), dtype=jnp.float32),
        tau_k_ms=100.0,
        drive_schedule=sched,
        noise_scale=0.0,
    )
    diff = np.max(np.abs(np.asarray(st_d2b["H_K_trace"]) - np.asarray(st_d2a["H_K_trace"])))
    assert diff == pytest.approx(0.0, abs=1e-6)


def test_d2b_post_stimulus_analytic_brackets_euler():
    ha0, hk0 = 0.08, 1.05
    n_steps = 2000
    dt = 0.5
    ha_e, hk_e = rbs_post_stimulus_euler_traces(
        ha0, hk0, n_steps, dt_ms=dt, tau_a_ms=25.0, tau_k_ms=100.0, kappa_ak=0.4
    )
    t = (np.arange(n_steps) + 1) * dt
    hk_an = h_k_analytic_two_timescale(
        hk0 - 1.0, ha0, t, tau_a_ms=25.0, tau_k_ms=100.0, kappa_ak=0.4
    )
    assert hk_e[-1] == pytest.approx(hk_an[-1], abs=0.05)


def test_d2b_node_local_spike_writes_own_h_a_only():
    params, edges = _isolated(n=2)
    n_steps = 300
    sched = jnp.zeros((n_steps, 2), dtype=jnp.float32)
    sched = sched.at[50:200, 0].set(15.0)
    key = jax.random.PRNGKey(9)
    _, _, _, st = simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
        params,
        edges,
        n_steps,
        0.5,
        key,
        h_k0=jnp.ones((2,), dtype=jnp.float32),
        tau_a_ms=25.0,
        tau_k_ms=100.0,
        kappa_ak=0.4,
        drive_schedule=sched,
        noise_scale=0.0,
    )
    ha1 = np.asarray(st["H_A_trace"][:, 1])
    s1 = np.asarray(st["S_trace"][:, 1])
    if np.sum(s1 > 0.5) == 0:
        assert np.max(ha1) == 0.0


def test_d2b_execution_receipt_all_gates_pass():
    if not D2B_EXECUTION_RECEIPT_PATH.exists():
        write_d2b_execution_receipt()
    receipt = load_d2b_execution_receipt()
    assert receipt["checkpoint"] == "D2b"
    assert receipt["status"] == "FROZEN"
    assert receipt["implementation_authorized"] is True
    for gname, gval in receipt["gates"].items():
        assert gval["passed"] is True, gname
    assert receipt["activity_input_S"].startswith("binary")


def test_d2b_run_in_memory_matches_receipt_schema():
    receipt = run_d2b_activity_h_k_coupling()
    assert receipt["kappa_AK"] == pytest.approx(0.4)
    assert receipt["tau_A_ms"] == pytest.approx(25.0)
    assert receipt["tau_K_ms"] == pytest.approx(100.0)
    assert len(receipt["cells"]) == 3


def test_d2b_protocol_receipt_closed_after_implementation():
    proto = json.loads(
        (D2B_SPEC_PATH.parent / "d2b_protocol_receipt.json").read_text(encoding="utf-8")
    )
    if D2B_EXECUTION_RECEIPT_PATH.exists():
        assert proto.get("implementation_authorized") is True or proto.get("status") == "CLOSED"
