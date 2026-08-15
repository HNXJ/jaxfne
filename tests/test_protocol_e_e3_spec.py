"""0.4.17-E E3 — RBS composition specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_e_integration.e3_protocol import (
    E3_SPEC_PATH,
    e3_gate_ids,
    e3_owner_flat_indices,
    load_e3_spec,
    resolve_owner_indices_from_e1_identity,
    validate_e3_spec,
)


def test_e3_spec_frozen():
    spec = load_e3_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "E3"


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


def test_e3_implementation_not_authorized_at_spec_freeze():
    auth = load_e3_spec()["execution_authorization"]
    assert auth["specification_only"] is True
    assert auth["implementation_authorized"] is False
    assert auth["next_checkpoint"] == "E3_implementation"


def test_e3_validate_spec_passes():
    validate_e3_spec()


def test_e3_protocol_receipt_closed():
    receipt = json.loads((E3_SPEC_PATH.parent / "e3_protocol_receipt.json").read_text())
    assert receipt["status"] == "CLOSED"
    assert receipt["rbs_owner"]["n_nodes"] == 7
    assert receipt["next_checkpoint"] == "E4_specification"
    assert receipt["execution_receipt"] == "artifacts/protocol_e_integration/e3_execution_receipt.json"
