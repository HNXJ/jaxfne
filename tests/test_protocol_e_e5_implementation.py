"""0.4.17-E E5 — causal perturbation implementation tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxfne.protocol_e_integration.e2_execution import build_e2_model
from jaxfne.protocol_e_integration.e5_execution import (
    compute_delta_r,
    compute_evidence_gates,
    run_e5_arm_kernel,
    run_e5_arm_trajectory,
    run_e5_causal_perturbation,
    write_e5_receipts,
)
from jaxfne.protocol_e_integration.e5_protocol import (
    E5_EXECUTION_RECEIPT_PATH,
    E5_INTERPRETATION_RECEIPT_PATH,
    load_e5_spec,
    validate_e5_spec,
)


def test_e5_n0_equals_n1_when_gamma_h_disabled():
    spec = load_e5_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    n0 = run_e5_arm_kernel(
        model, arm="N0", seed=11, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"])
    )
    n1 = run_e5_arm_kernel(
        model, arm="N1", seed=11, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"])
    )
    assert np.array_equal(n0["V_m"], n1["V_m"])
    assert np.array_equal(n0["spikes"], n1["spikes"])
    assert np.array_equal(n0["sources"], n1["sources"])


def test_e5_n1_and_d_share_h_k_trajectory():
    spec = load_e5_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    n1 = run_e5_arm_kernel(
        model, arm="N1", seed=11, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"])
    )
    d = run_e5_arm_kernel(
        model, arm="D", seed=11, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"])
    )
    assert np.array_equal(n1["H_K_trace"], d["H_K_trace"])
    assert not np.array_equal(n1["V_m"], d["V_m"])


def test_e5_observation_preserves_q_hash():
    spec = load_e5_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    run = run_e5_arm_trajectory(
        model, arm="D", seed=11, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"])
    )
    assert run.observation["cause_hashes_unchanged"]


def test_e5_delta_r_and_evidence_gates_structure():
    spec = load_e5_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    n1 = run_e5_arm_trajectory(
        model, arm="N1", seed=11, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"])
    )
    d = run_e5_arm_trajectory(
        model, arm="D", seed=11, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"])
    )
    from jaxfne.protocol_e_integration.e5_execution import _population_indices
    from jaxfne.protocol_e_integration.e3_execution import owner_flat_indices

    pop = _population_indices(d.traj.identity_map, owner_flat_indices())
    delta = compute_delta_r(d, n1, pop=pop)
    gates = compute_evidence_gates(delta)
    assert "G_O" in gates and "d_propagation" in gates
    assert delta["Delta_Q"]["L2_norm_difference"] > 0.0


def test_e5_run_and_write_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jaxfne.protocol_e_integration.e5_execution.E5_EXECUTION_RECEIPT_PATH",
        tmp_path / "e5_execution_receipt.json",
    )
    monkeypatch.setattr(
        "jaxfne.protocol_e_integration.e5_execution.E5_INTERPRETATION_RECEIPT_PATH",
        tmp_path / "e5_interpretation_receipt.json",
    )
    out = write_e5_receipts()
    assert out["raw"]["design"]["trajectory_count"] == 9
    assert all(g["passed"] for g in out["raw"]["quality_gates"].values())
    assert out["interpretation"]["aggregate_classification"] in {
        "NO_EFFECT",
        "LOCAL_EXPRESSION",
        "HIERARCHICAL_PROPAGATION",
        "UNRESOLVED",
    }
    assert out["raw"]["sanity_checks"]["N0_equals_N1_neural"][0]["N0_equals_N1_V_m_bit_exact"]


@pytest.mark.skipif(
    not E5_EXECUTION_RECEIPT_PATH.exists(),
    reason="frozen E5 execution receipt not present",
)
def test_e5_frozen_receipts_on_disk():
    from jaxfne.protocol_e_integration.e5_execution import (
        load_e5_execution_receipt,
        load_e5_interpretation_receipt,
    )

    validate_e5_spec()
    raw = load_e5_execution_receipt()
    interp = load_e5_interpretation_receipt()
    assert raw["status"] == "FROZEN"
    assert len(raw["trajectories"]) == 9
    assert interp["feature_freeze"]["policy"].startswith("hard")
    assert all(raw["quality_gates"][g]["passed"] for g in raw["quality_gates"])
