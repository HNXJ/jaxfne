"""0.4.17-E E4 — observation-chain composition implementation tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxfne.protocol_e_integration.e2_execution import build_e2_model
from jaxfne.protocol_e_integration.e3_execution import run_e3_kernel
from jaxfne.protocol_e_integration.e4_execution import (
    aggregate_q_by_hierarchy,
    apply_e4_probe,
    build_hierarchy_aware_source_table_summary,
    freeze_e3_trajectory,
    materialize_phi_ref,
    run_all_primary_observations,
    run_e4_neural_only,
    run_e4_observation_composition,
    verify_field_linearity,
    verify_zero_source_operators,
    write_e4_execution_receipt,
)
from jaxfne.protocol_e_integration.e4_protocol import (
    E4_EXECUTION_RECEIPT_PATH,
    load_e4_spec,
    validate_e4_spec,
)


def test_e4_neural_only_matches_e3_null():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    e3 = run_e3_kernel(
        model,
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
        seed=11,
        mode="E3-null",
    )
    e4 = run_e4_neural_only(
        model,
        seed=11,
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    assert np.array_equal(e3["V_m"], e4.V_m)
    assert np.array_equal(e3["sources"], e4.Q)
    assert np.array_equal(e3["H_K_trace"], e4.H_K)


def test_e4_q_conservation_under_hierarchy_aggregation():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    _, conservation = aggregate_q_by_hierarchy(traj)
    assert conservation["passed"]


def test_e4_probe_factorization_shared_phi_distinct_y():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    field = materialize_phi_ref(traj, spec)
    shallow = apply_e4_probe(traj, field, "lfp_contact_shallow", spec=spec)
    deep = apply_e4_probe(traj, field, "lfp_contact_deep", spec=spec)
    assert shallow.phi_ref_hash == deep.phi_ref_hash
    assert shallow.q_hash_before == shallow.q_hash_after == traj.cause_hashes["Q"]
    assert not np.array_equal(shallow.Y, deep.Y)


def test_e4_zero_source_and_linearity_operators():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    assert verify_zero_source_operators(traj, spec=spec)["passed"]
    assert verify_field_linearity(traj, spec=spec)["passed"]


def test_e4_all_primary_observations_preserve_q():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    obs = run_all_primary_observations(traj, spec=spec)
    assert obs["cause_hashes_unchanged"]
    table = build_hierarchy_aware_source_table_summary(traj)
    assert table["conservation"]["passed"]


def test_e4_dynamic_diagnostic_observation_runs():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-dynamic",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    obs = run_all_primary_observations(traj, spec=spec)
    assert obs["cause_hashes_unchanged"]
    assert float(np.max(np.abs(traj.H_K - 1.0))) > 0.0


def test_e4_run_gates_and_write_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jaxfne.protocol_e_integration.e4_execution.E4_EXECUTION_RECEIPT_PATH",
        tmp_path / "e4_execution_receipt.json",
    )
    receipt = write_e4_execution_receipt()
    assert receipt["checkpoint"] == "E4"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
    assert receipt["source_of_truth_hashes"]["Q"]
    assert receipt["e3_dynamic_observation_diagnostic"]["observation_completed"] is True
    assert len(receipt["semantic_status_table"]) == 9


@pytest.mark.skipif(
    not E4_EXECUTION_RECEIPT_PATH.exists(),
    reason="frozen E4 execution receipt not present",
)
def test_e4_frozen_execution_receipt_on_disk():
    from jaxfne.protocol_e_integration.e4_execution import load_e4_execution_receipt

    receipt = load_e4_execution_receipt()
    validate_e4_spec()
    assert receipt["status"] == "FROZEN"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
