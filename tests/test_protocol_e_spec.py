"""0.4.17-E protocol integration specification tests (E0 through E5)."""

from __future__ import annotations

import json

import pytest

from jaxfne.protocol_e_integration.e0_1_protocol import (
    E0_1_SPEC_PATH,
    e0_1_ladder_ids,
    load_e0_1_spec,
    validate_e0_1_spec,
)
from jaxfne.protocol_e_integration.e0_protocol import (
    E0_SPEC_PATH,
    PROTOCOL_ID,
    load_e0_spec,
    validate_e0_spec,
)
from jaxfne.protocol_e_integration.e1_protocol import (
    E1_SPEC_PATH,
    load_e1_spec,
)
from jaxfne.protocol_e_integration.e2_protocol import (
    E2_SPEC_PATH,
    e2_delay_class_ids,
    e2_gate_ids,
    load_e2_spec,
)
from jaxfne.protocol_e_integration.e3_protocol import (
    E3_SPEC_PATH,
    e3_gate_ids,
    e3_owner_flat_indices,
    load_e3_spec,
    resolve_owner_indices_from_e1_identity,
    validate_e3_spec,
)
from jaxfne.protocol_e_integration.e4_protocol import (
    E4_SPEC_PATH,
    e4_gate_ids,
    e4_primary_field_ids,
    e4_primary_probe_ids,
    load_e4_spec,
    validate_e4_spec,
)
from jaxfne.protocol_e_integration.e5_protocol import (
    E5_SPEC_PATH,
    e5_arm_ids,
    e5_gate_ids,
    e5_result_classes,
    load_e5_spec,
    validate_e5_spec,
)


_SPEC_CASES = [
    ("E0", load_e0_spec),
    ("E0.1", load_e0_1_spec),
    ("E1", load_e1_spec),
    ("E2", load_e2_spec),
    ("E3", load_e3_spec),
    ("E4", load_e4_spec),
    ("E5", load_e5_spec),
]


@pytest.mark.parametrize("checkpoint,load_spec", _SPEC_CASES)
def test_spec_frozen(checkpoint, load_spec):
    spec = load_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == checkpoint


@pytest.mark.parametrize(
    "validate_spec",
    [
        validate_e0_spec,
        validate_e0_1_spec,
        validate_e3_spec,
        validate_e4_spec,
        validate_e5_spec,
    ],
)
def test_validate_spec_passes(validate_spec):
    validate_spec()


def test_e0_protocol_id():
    assert load_e0_spec()["protocol_id"] == PROTOCOL_ID


def test_e0_composition_question():
    spec = load_e0_spec()
    assert "compose" in spec["question"].lower()
    assert spec["experiment_type"] == "composition_not_phenotype_manufacturing"


def test_e0_primary_compositional_invariants():
    criteria = load_e0_spec()["primary_success_criteria"]
    assert criteria["type"] == "compositional_invariants"
    assert "finite_stable_dynamics" in " ".join(criteria["required"])
    assert "adaptation_phenotype" in criteria["not_required"]


def test_e0_blocks_HDP_and_D4():
    blocks = load_e0_spec()["explicit_blocks"]
    assert blocks["W3_closed_loop_HDP"]["status"] == "unresolved"
    assert blocks["D4_second_RBS_class"]["status"] == "not_authorized"


def test_e0_inherits_D3_mechanism_null_lesson():
    lesson = load_e0_spec()["methodological_inheritance_from_D3"]
    assert "mechanism_null" in lesson["rule"]


def test_e0_implementation_not_authorized():
    auth = load_e0_spec()["execution_authorization"]
    assert auth["implementation_authorized"] is False
    assert auth["specification_only"] is True


def test_e0_protocol_receipt_frozen():
    receipt = json.loads((E0_SPEC_PATH.parent / "e0_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN"
    assert receipt["implementation_authorized"] is False
    assert receipt["next_checkpoint"] == "E0.1_specification"


def test_e0_1_ladder_E1_through_E5():
    assert e0_1_ladder_ids() == ("E1", "E2", "E3", "E4", "E5")
    ladder = load_e0_1_spec()["implementation_ladder"]["ordered_checkpoints"]
    assert ladder[0]["id"] == "E1"
    assert "RBS" in ladder[0]["excludes"][0]


def test_e0_1_reduction_contracts():
    ids = {r["id"] for r in load_e0_1_spec()["reduction_contracts"]}
    assert "R_E2_to_E1" in ids
    assert "R_E3_to_E2" in ids
    assert "R_E4_to_E3" in ids


def test_e0_1_monotonicity_principle():
    mono = load_e0_1_spec()["integration_monotonicity_principle"]
    assert mono["containment"] == "E1 subset E2 subset E3 subset E4"
    assert mono["explicit_reduction_tests_required"] is True


def test_e0_1_D3_phenotype_rule_for_E5():
    rule = load_e0_1_spec()["methodological_inheritance_from_D3"]["E5_phenotype_rule"]
    assert "mechanism-null" in rule


def test_e0_1_heterogeneity_terminology():
    term = load_e0_1_spec()["terminology"]
    assert "parameter sets" in term["heterogeneous_populations"]


def test_e0_1_implementation_not_authorized():
    auth = load_e0_1_spec()["execution_authorization"]
    assert auth["implementation_authorized"] is False
    assert auth["next_checkpoint"] == "E1_implementation"


def test_e0_1_protocol_receipt_frozen():
    receipt = json.loads((E0_1_SPEC_PATH.parent / "e0_1_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN"
    assert receipt["ladder"] == ["E1", "E2", "E3", "E4", "E5"]


def test_e1_two_area_laminar_hierarchy():
    h = load_e1_spec()["hierarchy"]
    assert h["areas"] == ["A1", "A2"]
    assert len(h["laminar_layers_per_area"]) >= 4


def test_e1_ff_fb_ownership_declared():
    conn = load_e1_spec()["connectivity"]
    assert conn["feedforward"]["id"] == "FF"
    assert conn["feedback"]["id"] == "FB"
    assert conn["delay_steps"] == 0


def test_e1_no_rbs_no_delays():
    spec = load_e1_spec()
    assert spec["rbs"]["enabled"] is False
    assert spec["connectivity"]["delay_steps"] == 0


def test_e1_population_heterogeneity_not_emitter_equations():
    sem = load_e1_spec()["hierarchy"]["heterogeneity_semantics"]
    assert "population_parameter" in sem
    banned = set(load_e1_spec()["explicit_prohibitions"])
    assert "no_call_population_heterogeneity_emitter_equation_heterogeneity" in banned


def test_e1_structural_gates_defined():
    gates = load_e1_spec()["gates"]
    assert "G1_construction" in gates
    assert "G5_reproducibility" in gates


def test_e1_implementation_authorized():
    auth = load_e1_spec()["execution_authorization"]
    assert auth["implementation_authorized"] is True


def test_e1_protocol_receipt_closed():
    receipt = json.loads((E1_SPEC_PATH.parent / "e1_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["implementation_authorized"] is True
    assert receipt["status"] == "CLOSED"
    assert receipt["next_checkpoint"] == "E2_specification"


def test_e2_reduction_contract_r_e2_to_e1():
    rc = load_e2_spec()["reduction_contract"]
    assert rc["id"] == "R_E2_to_E1"
    assert "V_m" in rc["bit_exact_equalities_required"]
    assert "edge_provenance_table" in rc["bit_exact_equalities_required"]


def test_e2_delay_classes_match_e1_provenance():
    classes = e2_delay_class_ids()
    assert classes == ("local_A1", "local_A2", "FF_A1_to_A2", "FB_A2_to_A1")


def test_e2_delay_ordering_and_grid_alignment():
    spec = load_e2_spec()
    rows = {r["edge_class"]: r for r in spec["delay_values"]["classes"]}
    assert float(rows["local_A1"]["tau_ms"]) <= float(rows["FF_A1_to_A2"]["tau_ms"])
    assert float(rows["FF_A1_to_A2"]["tau_ms"]) < float(rows["FB_A2_to_A1"]["tau_ms"])
    dt = float(spec["delay_values"]["dt_ms"])
    for row in spec["delay_values"]["classes"]:
        assert int(row["delay_steps"]) == int(round(float(row["tau_ms"]) / dt))


def test_e2_p_local_e1_receipt_derived_not_preregistered():
    p = load_e2_spec()["e1_receipt_derived_constants"]["p_local"]
    assert float(p["value"]) == 0.2
    assert "E1-receipt-derived" in p["provenance"]
    assert "not retroactively" in p["rule"].lower() or "do not retroactively" in p["rule"].lower()


def test_e2_expected_edge_counts_from_e1_receipt():
    counts = load_e2_spec()["e1_receipt_derived_constants"]["expected_edge_class_counts"]
    assert counts["FF_A1_to_A2"] == 140
    assert counts["FB_A2_to_A1"] == 140
    assert sum(counts[k] for k in ("local_A1", "local_A2", "FF_A1_to_A2", "FB_A2_to_A1")) == 931


def test_e2_gates_g1_through_g8():
    gates = e2_gate_ids()
    assert len(gates) == 8
    assert gates[0] == "G1_e1_reduction"
    assert gates[4] == "G5_delayed_continuation"
    assert gates[7] == "G8_no_scientific_overinterpretation"


def test_e2_continuation_segmentation_frozen():
    seg = load_e2_spec()["continuation_segmentation"]
    assert seg["public_state_name"] == "delay_state"
    assert seg["primary_split"]["t_split_ms"] == 400.0
    assert seg["inflight_stress_split"]["t_split_ms"] == 120.0


def test_e2_construction_scaffolding_excluded_from_evidence():
    scaffold = load_e2_spec()["construction_scaffolding"]
    assert scaffold["evidence_status"] == "excluded_from_E2_evidence"


@pytest.mark.parametrize(
    "checkpoint,load_spec",
    [
        ("E2", load_e2_spec),
        ("E3", load_e3_spec),
        ("E4", load_e4_spec),
        ("E5", load_e5_spec),
    ],
)
def test_implementation_not_authorized_at_spec_freeze(checkpoint, load_spec):
    auth = load_spec()["execution_authorization"]
    assert auth["specification_only"] is True
    assert auth["implementation_authorized"] is False
    assert auth["next_checkpoint"] == f"{checkpoint}_implementation"


def test_e2_protocol_receipt_closed():
    receipt = json.loads((E2_SPEC_PATH.parent / "e2_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["implementation_authorized"] is True
    assert receipt["next_checkpoint"] == "E3_specification"
    assert receipt["reduction_contract"] == "R_E2_to_E1"


def test_e3_reduction_contract_r_e3_to_e2():
    rc = load_e3_spec()["reduction_contract"]
    assert rc["id"] == "R_E3_to_E2"
    assert float(rc["reference_state"]["H_K"]) == 1.0
    assert "typed_delay_table" in rc["bit_exact_equalities_required"]


def test_e3_owner_a2_l5_e_exists_in_e1_identity():
    spec = load_e3_spec()
    owner = spec["ownership"]
    assert owner["selector"] == {"area": "A2", "layer": "L5", "cell_type": "E"}
    assert owner["n_nodes"] == 7
    resolved = resolve_owner_indices_from_e1_identity(spec)
    assert resolved == [70, 71, 72, 73, 74, 75, 76]
    assert list(e3_owner_flat_indices(spec)) == resolved


def test_e3_uses_d1_d2a_not_d2b():
    primitive = load_e3_spec()["rbs_primitive"]
    assert "static_null" in primitive["authorized_kernels"]
    assert "dynamic_recovery" in primitive["authorized_kernels"]
    assert primitive["deferred_not_used"]["D2b"] == "activity-written H_A -> H_K coupling"
    assert float(primitive["dynamics_F_H"]["dynamic_mode"]["tau_K_ms"]) == 100.0


def test_e3_execution_modes_null_and_dynamic():
    modes = {m["id"]: m for m in load_e3_spec()["execution_modes"]}
    assert modes["E3-null"]["H_K_initial"] == 1.0
    assert modes["E3-null"]["must_match_e2"] is True
    assert modes["E3-dynamic"]["H_K_initial"] == 1.2


def test_e3_gates_g1_through_g9():
    gates = e3_gate_ids()
    assert len(gates) == 9
    assert gates[0] == "G1_e2_reduction"
    assert gates[6] == "G7_continuation"
    assert gates[8] == "G9_no_phenotype_claim"


def test_e3_combined_state_continuation_carries_h_and_b():
    cont = load_e3_spec()["combined_state_continuation"]
    assert "delay_state B_t" in " ".join(cont["carried_fields"])
    assert cont["inherited_splits_ms"]["inflight_stress"] == 120.0


def test_e3_delay_table_inheritance_unchanged():
    rule = load_e3_spec()["typed_delay_inheritance"]
    assert rule["must_match_exactly"] is True
    assert "e2_execution_receipt" in rule["source_receipt"]


def test_e3_protocol_receipt_closed():
    receipt = json.loads((E3_SPEC_PATH.parent / "e3_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["rbs_owner"]["n_nodes"] == 7
    assert receipt["next_checkpoint"] == "E4_specification"
    assert receipt["execution_receipt"] == "artifacts/protocol_e_integration/e3_execution_receipt.json"


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


def test_e4_protocol_receipt_closed():
    receipt = json.loads((E4_SPEC_PATH.parent / "e4_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["reduction_contract"] == "R_E4_to_E3"
    assert receipt["next_checkpoint"] == "E5_specification"
    assert receipt["execution_receipt"] == "artifacts/protocol_e_integration/e4_execution_receipt.json"


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


def test_e5_publication_manifest_provenance_note():
    note = load_e5_spec()["publication_evidence_manifest_note"]
    assert "execution_parent_sha" in note
    assert "artifact_commit_sha" in note
    assert "do not retroactively" in note["rule"].lower()


def test_e5_protocol_receipt_closed():
    receipt = json.loads((E5_SPEC_PATH.parent / "e5_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["primary_contrast"] == "D - N1"
    assert receipt["execution_receipt"] == "artifacts/protocol_e_integration/e5_execution_receipt.json"
    assert receipt["interpretation_receipt"] == "artifacts/protocol_e_integration/e5_interpretation_receipt.json"
