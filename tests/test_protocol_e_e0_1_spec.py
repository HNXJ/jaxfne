"""0.4.17-E E0.1 — implementation ladder specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_e_integration.e0_1_protocol import (
    E0_1_SPEC_PATH,
    e0_1_ladder_ids,
    load_e0_1_spec,
    validate_e0_1_spec,
)


def test_e0_1_spec_frozen():
    spec = load_e0_1_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "E0.1"


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


def test_e0_1_validate_spec_passes():
    validate_e0_1_spec()


def test_e0_1_protocol_receipt_frozen():
    receipt = json.loads((E0_1_SPEC_PATH.parent / "e0_1_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN"
    assert receipt["ladder"] == ["E1", "E2", "E3", "E4", "E5"]
