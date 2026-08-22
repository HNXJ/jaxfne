"""0.4.17-E E2 — typed delayed-coupling specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_e_integration.e2_protocol import (
    E2_SPEC_PATH,
    e2_delay_class_ids,
    e2_gate_ids,
    load_e2_spec,
    validate_e2_spec,
)


def test_e2_spec_frozen():
    spec = load_e2_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "E2"


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


def test_e2_implementation_not_authorized_at_spec_freeze():
    auth = load_e2_spec()["execution_authorization"]
    assert auth["specification_only"] is True
    assert auth["implementation_authorized"] is False
    assert auth["next_checkpoint"] == "E2_implementation"


def test_e2_validate_spec_passes():
    validate_e2_spec()


def test_e2_protocol_receipt_closed():
    receipt = json.loads((E2_SPEC_PATH.parent / "e2_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["implementation_authorized"] is True
    assert receipt["next_checkpoint"] == "E3_specification"
    assert receipt["reduction_contract"] == "R_E2_to_E1"
