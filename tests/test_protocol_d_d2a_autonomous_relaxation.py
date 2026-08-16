"""0.4.17-D D2a — autonomous H_K F1 relaxation tests."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    _advance_h_k_f1_autonomous,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery,
    simulate_edge_recurrent_izhikevich_static_h_k_recovery,
)
from jaxfne.protocol_d_biological_rbs.d2a_execution import (
    h_k_f1_analytic,
    h_k_f1_euler_trace,
    load_d2a_execution_receipt,
    write_d2a_execution_receipt,
)
from jaxfne.protocol_d_biological_rbs.d2a_protocol import (
    D2A_SPEC_PATH,
    d2a_h_k0_values,
    load_d2a_spec,
    validate_d2a_spec,
)


def _isolated() -> tuple[IzhikevichParams, EdgeList]:
    jdtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.asarray([-65.0], dtype=jdtype),
        u0=jnp.zeros((1,), dtype=jdtype),
        a=jnp.asarray([0.02], dtype=jdtype),
        b=jnp.asarray([0.2], dtype=jdtype),
        c=jnp.asarray([-65.0], dtype=jdtype),
        d=jnp.asarray([8.0], dtype=jdtype),
        drive=jnp.zeros((1,), dtype=jdtype),
        sign=jnp.ones((1,), dtype=jdtype),
        W=jnp.zeros((1, 1), dtype=jdtype),
        source_scale=jnp.ones((1,), dtype=jdtype),
        labels=("E",),
        layer_labels=("L4",),
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


def test_d2a_spec_frozen_autonomous_only():
    spec = load_d2a_spec()
    assert spec["checkpoint"] == "D2a"
    assert spec["dynamics_F_H"]["kappa_K"] == 0
    assert spec["d2b_deferred"]["status"] == "specified_not_authorized"


def test_d2a_f1_euler_matches_jax_advance():
    H = jnp.asarray([0.8, 1.2], dtype=jnp.float32)
    dt = jnp.asarray(0.5, dtype=jnp.float32)
    tau = jnp.asarray(100.0, dtype=jnp.float32)
    H_next = _advance_h_k_f1_autonomous(H, dt, tau)
    ref0 = h_k_f1_euler_trace(0.8, 1, dt_ms=0.5, tau_k_ms=100.0)[0]
    ref1 = h_k_f1_euler_trace(1.2, 1, dt_ms=0.5, tau_k_ms=100.0)[0]
    assert float(H_next[0]) == pytest.approx(ref0, rel=0, abs=1e-6)
    assert float(H_next[1]) == pytest.approx(ref1, rel=0, abs=1e-6)


def test_d2a_analytic_bracket_euler():
    h0 = 0.8
    t = np.array([50.0, 500.0, 1500.0])
    analytic = h_k_f1_analytic(h0, t, tau_k_ms=100.0)
    euler = h_k_f1_euler_trace(h0, 3000, dt_ms=0.5, tau_k_ms=100.0)
    idx = (t / 0.5).astype(int) - 1
    for i, ix in enumerate(idx):
        assert euler[ix] == pytest.approx(analytic[i], abs=0.05)


def test_d2a_h_k0_restores_to_one():
    receipt = load_d2a_execution_receipt()
    for cell in receipt["cells"]:
        if cell["H_K0"] != 1.0:
            assert abs(cell["H_K_final"] - 1.0) < 0.02


def test_d2a_g3_baseline_bit_exact():
    receipt = load_d2a_execution_receipt()
    for detail in receipt["gates"]["G3_baseline_invariance"]["details"]:
        assert detail["V_m_bit_exact"] is True
        assert detail["spikes_bit_exact"] is True
        assert detail["H_K_max_deviation"] == 0.0


def test_d2a_dynamic_h_k1_matches_classical():
    params, edges = _isolated()
    n_steps = 200
    dt = 0.5
    sched = jnp.zeros((n_steps, 1), dtype=jnp.float32)
    sched = sched.at[20:180, 0].set(15.0)
    key = jax.random.PRNGKey(3)
    h1 = jnp.ones((1,), dtype=jnp.float32)
    v_c, sp_c, _, _ = simulate_edge_recurrent_izhikevich(
        params, edges, n_steps, dt, key, drive_schedule=sched, noise_scale=0.0
    )
    v_d, sp_d, _, st = simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery(
        params,
        edges,
        n_steps,
        dt,
        key,
        h_k0=h1,
        tau_k_ms=100.0,
        drive_schedule=sched,
        noise_scale=0.0,
    )
    assert float(jnp.max(jnp.abs(v_c - v_d))) == 0.0
    assert float(jnp.max(jnp.abs(sp_c - sp_d))) == 0.0
    assert float(jnp.max(jnp.abs(st["H_K_trace"] - 1.0))) == 0.0


def test_d2a_static_limit_matches_d1_at_fixed_h():
    params, edges = _isolated()
    n_steps = 100
    h08 = jnp.asarray([0.8], dtype=jnp.float32)
    key = jax.random.PRNGKey(4)
    v_s, sp_s, _, _ = simulate_edge_recurrent_izhikevich_static_h_k_recovery(
        params, edges, n_steps, 0.5, key, h_k=h08, noise_scale=0.0
    )
    v_d, sp_d, _, st = simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery(
        params, edges, n_steps, 0.5, key, h_k0=h08, tau_k_ms=1e9, noise_scale=0.0
    )
    # Very slow relaxation: H_K approximately constant over short horizon
    assert float(jnp.max(jnp.abs(v_s - v_d))) < 1e-3


def test_d2a_execution_receipt_all_gates():
    receipt = load_d2a_execution_receipt()
    assert receipt["status"] == "FROZEN"
    assert receipt["n_cells"] == 9
    for gate in receipt["gates"].values():
        assert gate["passed"] is True


def test_d2a_validate_spec():
    validate_d2a_spec()


def test_d2a_sweep_levels():
    assert d2a_h_k0_values() == (0.8, 1.0, 1.2)


def test_d2a_protocol_receipt():
    meta = json.loads((D2A_SPEC_PATH.parent / "d2a_protocol_receipt.json").read_text())
    assert meta["checkpoint"] == "D2a"
    assert meta["status"] == "CLOSED"
