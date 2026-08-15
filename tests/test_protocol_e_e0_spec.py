"""0.4.17-E E0 — integrated composition specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_e_integration.e0_protocol import (
    E0_SPEC_PATH,
    PROTOCOL_ID,
    load_e0_spec,
    validate_e0_spec,
)


def test_e0_spec_frozen():
    spec = load_e0_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "E0"
    assert spec["protocol_id"] == PROTOCOL_ID


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


def test_e0_validate_spec_passes():
    validate_e0_spec()


def test_e0_protocol_receipt_frozen():
    receipt = json.loads((E0_SPEC_PATH.parent / "e0_protocol_receipt.json").read_text())
    assert receipt["status"] == "FROZEN"
    assert receipt["implementation_authorized"] is False
    assert receipt["next_checkpoint"] == "E0_implementation"
