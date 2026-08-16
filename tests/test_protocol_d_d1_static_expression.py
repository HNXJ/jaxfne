"""0.4.17-D D1 — static H_K recovery expression tests."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    _izhikevich_dv_du,
    _izhikevich_dv_du_recovery_h_k,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_static_h_k_recovery,
)
from jaxfne.protocol_d_biological_rbs.d1_execution import (
    load_d1_execution_receipt,
    write_d1_execution_receipt,
)
from jaxfne.protocol_d_biological_rbs.d1_protocol import (
    D1_EXECUTION_RECEIPT_PATH,
    D1_SPEC_PATH,
    d1_h_k_sweep_values,
    load_d1_spec,
    validate_d1_spec,
)


def _isolated() -> tuple[IzhikevichParams, EdgeList]:
    jdtype = jnp.float32
    n = 1
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


def test_d1_spec_frozen_and_coupling():
    spec = load_d1_spec()
    assert spec["status"] == "FROZEN"
    assert spec["implementation"]["coupling"] == "b_eff = H_K * b"
    assert spec["execution_authorization"]["implementation_authorized"] is True


def test_d1_static_sweep_levels():
    assert d1_h_k_sweep_values() == (0.8, 1.0, 1.2)


def test_d1_local_jacobian_receipt():
  v = jnp.asarray([-60.0], dtype=jnp.float32)
  u = jnp.asarray([5.0], dtype=jnp.float32)
  a = jnp.asarray([0.02], dtype=jnp.float32)
  b = jnp.asarray([0.2], dtype=jnp.float32)
  h_k = jnp.asarray([1.1], dtype=jnp.float32)
  du_dot = lambda hk: _izhikevich_dv_du_recovery_h_k(v, u, jnp.zeros_like(v), a, b, hk)[1]
  jac = jax.jacobian(du_dot)(h_k)
  assert float(jac.squeeze()) == pytest.approx(float(a[0] * b[0] * v[0]))


def test_d1_g1_containment_bit_exact():
    params, edges = _isolated()
    n_steps = 200
    dt = 0.5
    sched = jnp.zeros((n_steps, 1), dtype=jnp.float32)
    sched = sched.at[20:180, 0].set(15.0)
    key = jax.random.PRNGKey(0)
    h1 = jnp.ones((1,), dtype=jnp.float32)
    v_c, sp_c, _, _ = simulate_edge_recurrent_izhikevich(
        params, edges, n_steps, dt, key, drive_schedule=sched, noise_scale=0.0
    )
    v_e, sp_e, _, st_e = simulate_edge_recurrent_izhikevich_static_h_k_recovery(
        params, edges, n_steps, dt, key, h_k=h1, drive_schedule=sched, noise_scale=0.0
    )
    assert float(jnp.max(jnp.abs(v_c - v_e))) == 0.0
    assert float(jnp.max(jnp.abs(sp_c - sp_e))) == 0.0
    # u is traced only in extended kernel; at H_K=1 du paths are identical
    dv0, du0 = _izhikevich_dv_du(jnp.asarray([-62.0]), jnp.asarray([1.0]), jnp.asarray([0.0]), params.a, params.b)
    dv1, du1 = _izhikevich_dv_du_recovery_h_k(
        jnp.asarray([-62.0]), jnp.asarray([1.0]), jnp.asarray([0.0]), params.a, params.b, h1
    )
    assert float(du0[0]) == float(du1[0])
    assert jnp.all(np.isfinite(st_e["u_trace"]))


def test_d1_g2_static_h_k_constant():
    params, edges = _isolated()
    n_steps = 100
    h_k = jnp.asarray([0.8], dtype=jnp.float32)
    _, _, _, st = simulate_edge_recurrent_izhikevich_static_h_k_recovery(
        params,
        edges,
        n_steps,
        0.5,
        jax.random.PRNGKey(1),
        h_k=h_k,
        noise_scale=0.0,
    )
    assert float(st["H_K_static"][0]) == pytest.approx(0.8)


def test_d1_h_k_one_matches_classical_du():
    v = jnp.asarray([-55.0], dtype=jnp.float32)
    u = jnp.asarray([2.0], dtype=jnp.float32)
    I = jnp.asarray([10.0], dtype=jnp.float32)
    a = jnp.asarray([0.02], dtype=jnp.float32)
    b = jnp.asarray([0.2], dtype=jnp.float32)
    h1 = jnp.ones((1,), dtype=jnp.float32)
    _, du_c = _izhikevich_dv_du(v, u, I, a, b)
    _, du_e = _izhikevich_dv_du_recovery_h_k(v, u, I, a, b, h1)
    assert float(du_c[0]) == float(du_e[0])


def test_d1_validate_spec_passes():
    validate_d1_spec()


def test_d1_execution_receipt_frozen_gates():
    receipt = load_d1_execution_receipt()
    assert receipt["status"] == "FROZEN"
    assert receipt["gates"]["G1_containment"]["passed"] is True
    assert receipt["gates"]["G2_static_state_integrity"]["passed"] is True
    assert receipt["gates"]["G4_bidirectional_sensitivity"]["evaluated_levels"] == [0.8, 1.0, 1.2]
    assert receipt["n_cells"] == 9
    assert receipt["coupling"] == "b_eff = H_K * b"


def test_d1_all_three_levels_present_per_seed():
    receipt = load_d1_execution_receipt()
    spec = load_d1_spec()
    seeds = spec["simulation_policy"]["seeds"]
    for seed in seeds:
        levels = sorted({c["H_K"] for c in receipt["cells"] if c["seed"] == seed})
        assert levels == [0.8, 1.0, 1.2]


def test_d1_expression_sensitivity_recorded():
    receipt = load_d1_execution_receipt()
    expr = receipt["expression_summary"]
    assert expr["direction_not_preregistered"] is True
    assert "any_u_sensitive" in expr


def test_d1_documentation_clause_present():
    receipt = load_d1_execution_receipt()
    assert "effective relative K-associated recovery" in receipt["documentation_clause"]


def test_d1_protocol_receipt_pointer():
    meta = json.loads((D1_SPEC_PATH.parent / "d1_protocol_receipt.json").read_text())
    assert meta["checkpoint"] == "D1"
    assert meta["status"] == "CLOSED"
