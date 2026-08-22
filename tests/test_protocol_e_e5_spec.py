"""0.4.17-E E5 — causal perturbation specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_e_integration.e5_protocol import (
    E5_SPEC_PATH,
    e5_arm_ids,
    e5_gate_ids,
    e5_result_classes,
    load_e5_spec,
    validate_e5_spec,
)


def test_e5_spec_frozen():
    spec = load_e5_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "E5"


def test_e5_central_question_perturbation_vs_null():
    q = load_e5_spec()["question"].lower()
    assert "perturbation" in q
    assert "mechanism-null" in q or "mechanism null" in q


def test_e5_arms_n0_n1_d():
    arms = {a["id"]: a for a in load_e5_spec()["experimental_arms"]}
    assert e5_arm_ids() == ("N0", "N1", "D")
    assert arms["N0"]["H_K_initial_on_O_H"] == 1.0
    assert arms["N1"]["Gamma_H"] == "identity_disabled"
    assert arms["D"]["Gamma_H"] == "enabled"
    assert arms["N1"]["H_K_initial_on_O_H"] == arms["D"]["H_K_initial_on_O_H"] == 1.2


def test_e5_causal_contrast_d_minus_n1():
    cc = load_e5_spec()["causal_contrast"]
    assert cc["primary"] == "D - N1"
    assert "H_K^N1" in cc["null_invariants"]["H_K_trajectory"] or "H_K^N1(t)" in cc["null_invariants"]["H_K_trajectory"]


def test_e5_owner_matches_e3():
    owner = load_e5_spec()["perturbation_target"]
    assert owner["selector"] == {"area": "A2", "layer": "L5", "cell_type": "E"}
    assert owner["flat_indices"] == [70, 71, 72, 73, 74, 75, 76]


def test_e5_propagation_assay_levels():
    levels = [row["id"] for row in load_e5_spec()["propagation_assay"]["levels"]]
    assert levels == ["X_owner", "X_A2_nonowner", "X_A1", "Q", "Y"]


def test_e5_response_vector_delta_r():
    rv = load_e5_spec()["response_vector"]
    assert rv["contrast"] == "D - N1"
    assert rv["components"] == [
        "Delta_X_owner",
        "Delta_X_A2_nonowner",
        "Delta_X_A1",
        "Delta_Q",
        "Delta_Y",
    ]


def test_e5_result_classification_enum():
    assert e5_result_classes() == (
        "NO_EFFECT",
        "LOCAL_EXPRESSION",
        "HIERARCHICAL_PROPAGATION",
        "UNRESOLVED",
    )
    not_req = load_e5_spec()["success_criteria"]["not_required"]
    assert "HIERARCHICAL_PROPAGATION" in not_req


def test_e5_seeds_fixed():
    assert load_e5_spec()["simulation_policy"]["seeds"] == [11, 12, 13]


def test_e5_zero_architecture_rule():
    rule = load_e5_spec()["scientific_scope"]["architecture_rule"]
    assert "zero" in rule.lower() or "only" in rule.lower()


def test_e5_single_trajectory_observation_workflow():
    wf = load_e5_spec()["observation_workflow"]
    assert wf["simulate_calls_per_seed"] == 3
    assert "never rerun" in wf["rule"].lower() or "post-hoc" in wf["rule"].lower()


def test_e5_gates_g1_through_g10():
    gates = e5_gate_ids()
    assert len(gates) == 10
    assert gates[0] == "G1_arm_isolation"
    assert gates[9] == "G10_no_phenotype_overinterpretation"


def test_e5_implementation_not_authorized_at_spec_freeze():
    auth = load_e5_spec()["execution_authorization"]
    assert auth["specification_only"] is True
    assert auth["implementation_authorized"] is False
    assert auth["next_checkpoint"] == "E5_implementation"


def test_e5_publication_manifest_provenance_note():
    note = load_e5_spec()["publication_evidence_manifest_note"]
    assert "execution_parent_sha" in note
    assert "artifact_commit_sha" in note
    assert "do not retroactively" in note["rule"].lower()


def test_e5_validate_spec_passes():
    validate_e5_spec()


def test_e5_protocol_receipt_closed():
    receipt = json.loads((E5_SPEC_PATH.parent / "e5_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["primary_contrast"] == "D - N1"
    assert receipt["execution_receipt"] == "artifacts/protocol_e_integration/e5_execution_receipt.json"
    assert receipt["interpretation_receipt"] == "artifacts/protocol_e_integration/e5_interpretation_receipt.json"
