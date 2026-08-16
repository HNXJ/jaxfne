"""0.4.17-E E1 — hierarchy/runtime specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_e_integration.e1_protocol import (
    E1_SPEC_PATH,
    load_e1_spec,
    validate_e1_spec,
)


def test_e1_spec_frozen():
    spec = load_e1_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "E1"


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


def test_e1_validate_spec_passes():
    validate_e1_spec()


def test_e1_protocol_receipt_closed():
    receipt = json.loads((E1_SPEC_PATH.parent / "e1_protocol_receipt.json").read_text())
    assert receipt["implementation_authorized"] is True
    assert receipt["status"] == "CLOSED"
    assert receipt["next_checkpoint"] == "E2_specification"
