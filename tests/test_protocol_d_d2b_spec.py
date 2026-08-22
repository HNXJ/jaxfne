"""0.4.17-D D2b — two-coordinate activity-to-H_K coupling specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_d_biological_rbs.d2b_protocol import (
    D2B_SPEC_PATH,
    load_d2b_spec,
    validate_d2b_spec,
)


def test_d2b_spec_frozen():
    spec = load_d2b_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "D2b"
    assert spec["write_once"] is True


def test_d2b_two_coordinate_vector():
    spec = load_d2b_spec()
    ids = [c["id"] for c in spec["rbs_vector"]["coordinates"]]
    assert ids == ["H_A", "H_K"]


def test_d2b_reference_conventions():
    spec = load_d2b_spec()
    by_id = {c["id"]: c for c in spec["rbs_vector"]["coordinates"]}
    assert by_id["H_A"]["reference"] == 0
    assert by_id["H_K"]["reference"] == 1


def test_d2b_h_a_not_ion():
    spec = load_d2b_spec()
    h_a = next(c for c in spec["rbs_vector"]["coordinates"] if c["id"] == "H_A")
    assert h_a["is_ion"] is False
    assert "activity" in h_a["role"].lower()


def test_d2b_causal_chain():
    chain = load_d2b_spec()["causal_chain"]["display"]
    assert "H_{A" in chain and "H_{K" in chain and "S" in chain


def test_d2b_dynamics_frozen():
    dyn = load_d2b_spec()["dynamics"]
    assert "tau_A" in dyn["H_A"]["display"]
    assert "kappa_AK" in dyn["H_K"]["display"]
    assert dyn["coupling_constant"]["value"] > 0


def test_d2b_tau_ordering():
    ts = load_d2b_spec()["dynamics"]["timescales"]
    assert ts["tau_A_ms"] < ts["tau_K_ms"]


def test_d2b_discrete_causal_ordering():
    steps = load_d2b_spec()["dynamics"]["discrete_update_ordering"]["steps"]
    assert len(steps) == 2
    assert "H_A^n" in steps[1]


def test_d2b_null_kappa_reduces_to_d2a():
    null = load_d2b_spec()["null_hierarchy"]["kappa_AK_zero"]
    assert "D2a" in null["reduction"]


def test_d2b_null_S_zero():
    null = load_d2b_spec()["null_hierarchy"]["S_zero"]
    assert "H_A" in null["reduction"] and "H_K" in null["reduction"]


def test_d2b_analytic_post_stimulus_contract():
    ana = load_d2b_spec()["analytic_post_stimulus_contract"]
    assert "h_K" in ana["deviation_variable"]
    assert "exp" in ana["H_A_solution"]


def test_d2b_success_criteria_state_space_only():
    criteria = load_d2b_spec()["success_criteria_state_space"]
    assert len(criteria) == 7
    assert any("kappa_AK" in c for c in criteria)


def test_d2b_no_firing_direction_preregistered():
    banned = set(load_d2b_spec()["explicit_prohibitions"])
    assert "no_preregister_H_K_up_implies_firing_down" in banned


def test_d2b_implementation_not_authorized():
    auth = load_d2b_spec()["execution_authorization"]
    assert auth["implementation_authorized"] is False
    assert auth["specification_only"] is True


def test_d2b_validate_spec_passes():
    validate_d2b_spec()


def test_d2b_protocol_receipt_closed_after_implementation():
    receipt = json.loads((D2B_SPEC_PATH.parent / "d2b_protocol_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "CLOSED"
    assert receipt["implementation_authorized"] is True
    assert receipt["next_checkpoint"] == "D3"
    assert "execution_receipt" in receipt
