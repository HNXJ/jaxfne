"""0.4.17-E E2 — typed delayed-coupling implementation tests."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.protocol_e_integration.e1_execution import build_e1_configuration
from jaxfne.protocol_e_integration.e2_execution import (
    assert_no_compile_drift,
    attach_provenance_class_delays,
    build_e2_model,
    build_typed_delay_table,
    delay_step_occupancy,
    hierarchy_fingerprint,
    load_e2_execution_receipt,
    run_e2_delayed_coupling,
    verify_delayed_continuation,
    verify_delay_ownership,
    write_e2_execution_receipt,
)
from jaxfne.protocol_e_integration.e2_protocol import (
    E2_EXECUTION_RECEIPT_PATH,
    E2_SPEC_PATH,
    load_e2_spec,
    validate_e2_spec,
)


def test_e2_build_model_attaches_provenance_delays():
    spec = load_e2_spec()
    model = build_e2_model(zero_delays=False)
    counts = assert_no_compile_drift(model, spec)
    assert counts["FF_A1_to_A2"] == 140
    g3 = verify_delay_ownership(model, spec)
    assert g3["passed"]


def test_e2_zero_delay_matches_e1_via_e2_path():
    spec = load_e2_spec()
    zero = build_e2_model(zero_delays=True)
    e1 = jtfne.construct(build_e1_configuration())
    sim = jtfne.Simulation(
        duration_ms=100.0,
        dt_ms=0.5,
        seed=11,
        runtime=jtfne.RuntimeConfig(
            dtype="float32",
            recurrent_backend="edge_list",
            enable_hdp=False,
            hdp_params={"noise_scale": 0.0},
        ),
    )
    a = e1.simulate(sim)
    b = zero.simulate(sim)
    assert np.array_equal(np.asarray(a.V_m), np.asarray(b.V_m))
    assert np.array_equal(np.asarray(a.spikes), np.asarray(b.spikes))


def test_e2_hierarchy_fingerprint_unchanged_by_delay_attachment():
    spec = load_e2_spec()
    base = jtfne.construct(build_e1_configuration())
    assert_no_compile_drift(base, spec)
    pre = hierarchy_fingerprint(base)
    delayed = attach_provenance_class_delays(
        jtfne.construct(build_e1_configuration()),
        {row["edge_class"]: int(row["delay_steps"]) for row in spec["delay_values"]["classes"]},
    )
    post = hierarchy_fingerprint(delayed)
    assert pre["identity_digest"] == post["identity_digest"]
    assert pre["provenance_digest"] == post["provenance_digest"]
    assert pre["edge_topology_digest"] == post["edge_topology_digest"]


def test_e2_delay_step_occupancy_diagnostic():
    model = build_e2_model(zero_delays=False)
    occ = delay_step_occupancy(model)
    assert occ["2"] == 651
    assert occ["4"] == 140
    assert occ["8"] == 140


def test_e2_delayed_continuation_primary_and_inflight():
    model = build_e2_model(zero_delays=False)
    primary = verify_delayed_continuation(
        model, total_ms=1000.0, split_ms=400.0, dt_ms=0.5, seed=11
    )
    inflight = verify_delayed_continuation(
        model, total_ms=1000.0, split_ms=120.0, dt_ms=0.5, seed=11
    )
    assert primary["passed"]
    assert inflight["passed"]
    assert primary["event_timing_exact"]
    assert inflight["event_timing_exact"]


def test_e2_typed_delay_table_matches_frozen_counts():
    spec = load_e2_spec()
    model = build_e2_model(zero_delays=False)
    counts = assert_no_compile_drift(model, spec)
    table = build_typed_delay_table(model, spec, actual_counts=counts)
    assert len(table) == 4
    by_class = {row["edge_class"]: row for row in table}
    assert by_class["local_A1"]["actual_edges"] == 334
    assert by_class["FB_A2_to_A1"]["delay_steps"] == 8


def test_e2_run_gates_and_write_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jaxfne.protocol_e_integration.e2_execution.E2_EXECUTION_RECEIPT_PATH",
        tmp_path / "e2_execution_receipt.json",
    )
    receipt = write_e2_execution_receipt()
    assert receipt["checkpoint"] == "E2"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
    assert receipt["delay_step_occupancy"]["N_d8"] == 140


def test_e2_validate_spec_passes():
    validate_e2_spec()


def test_e2_frozen_execution_receipt_present():
    receipt = load_e2_execution_receipt()
    assert receipt["status"] == "FROZEN"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
    assert "p_local" in receipt
    assert "E1-receipt-derived" in receipt["p_local"]["provenance"]
