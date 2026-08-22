"""0.4.17-E E4 — observation-chain composition specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_e_integration.e4_protocol import (
    E4_SPEC_PATH,
    e4_gate_ids,
    e4_primary_field_ids,
    e4_primary_probe_ids,
    load_e4_spec,
    validate_e4_spec,
)


def test_e4_spec_frozen():
    spec = load_e4_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "E4"


def test_e4_reduction_contract_r_e4_to_e3():
    rc = load_e4_spec()["reduction_contract"]
    assert rc["id"] == "R_E4_to_E3"
    assert "sources_Q" in rc["bit_exact_equalities_required"]
    assert "delay_state_B_t" in rc["bit_exact_equalities_required"]
    assert "H_K_trace" in rc["bit_exact_equalities_required"]


def test_e4_trajectory_invariance_probe_independent():
    traj = load_e4_spec()["trajectory_invariance_contract"]
    assert traj["id"] == "T_E4_probe_independent_neural_source"
    assert set(traj["probe_selection_invariant_fields"]) >= {"Q", "H_K", "delay_state_B_t"}


def test_e4_single_simulate_workflow():
    wf = load_e4_spec()["causal_architecture"]["workflow"]
    assert "simulate once" in wf["display"]
    assert "never re-simulate" in " ".join(wf["steps"]).lower()


def test_e4_inherits_experiment_a_source_semantics():
    inherit = load_e4_spec()["experiment_a_inheritance"]
    assert inherit["protocol_id"] == "experiment_a_v0417_b"
    assert (
        inherit["inherited_source_contract"]["Q_recorded"]
        == "signals.sources_canonical_relative_source"
    )
    assert "integrated-model" in inherit["rule"].lower() or "do not create" in inherit["rule"].lower()


def test_e4_conservative_primary_operators():
    assert e4_primary_field_ids() == ("lfp_ref",)
    assert e4_primary_probe_ids() == (
        "lfp_contact_shallow",
        "lfp_contact_deep",
        "csd_from_lfp_ref",
    )


def test_e4_eeg_meg_deferred_not_primary():
    spec = load_e4_spec()
    deferred = spec["deferred_operators"]["analysis_only_excluded_from_primary_evidence"]
    ids = {row["id"] for row in deferred}
    assert ids >= {"eeg_superficial", "eeg_deep", "meg_relative"}
    assert all(row["semantic"] == "analysis_only" for row in deferred)
    assert "no_EEG_MEG_in_primary_evidence" in spec["explicit_prohibitions"]


def test_e4_native_outputs_include_q_first_class():
    natives = {row["id"]: row for row in load_e4_spec()["native_outputs"]}
    assert natives["Q"]["first_class"] is True
    assert natives["Q"]["semantic"] == "native"
    assert natives["V_m"]["semantic"] == "native"


def test_e4_hierarchy_aware_source_table_uses_e1_identity():
    table = load_e4_spec()["hierarchy_aware_source_table"]
    assert "e1_execution_receipt.json#identity_map" in table["identity_source"]
    assert table["display"] == "Q -> (area, layer, cell_type, t)"


def test_e4_gates_g1_through_g10():
    gates = e4_gate_ids()
    assert len(gates) == 10
    assert gates[0] == "G1_e3_reduction_neural_invariance"
    assert gates[3] == "G4_probe_independence"
    assert gates[9] == "G10_no_phenotype_claim"


def test_e4_neural_path_e3_null():
    spec = load_e4_spec()
    assert spec["neural_execution"]["reference_mode"] == "E3-null"
    assert spec["simulation_policy"]["neural_mode"] == "E3-null"


def test_e4_implementation_not_authorized_at_spec_freeze():
    auth = load_e4_spec()["execution_authorization"]
    assert auth["specification_only"] is True
    assert auth["implementation_authorized"] is False
    assert auth["next_checkpoint"] == "E4_implementation"


def test_e4_validate_spec_passes():
    validate_e4_spec()


def test_e4_protocol_receipt_closed():
    receipt = json.loads((E4_SPEC_PATH.parent / "e4_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["reduction_contract"] == "R_E4_to_E3"
    assert receipt["next_checkpoint"] == "E5_specification"
    assert receipt["execution_receipt"] == "artifacts/protocol_e_integration/e4_execution_receipt.json"
