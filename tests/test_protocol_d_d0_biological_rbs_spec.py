"""0.4.17-D D0 — frozen biological RBS specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_d_biological_rbs import (
    D0_SPEC_PATH,
    PROTOCOL_ID,
    d0_first_coordinate_id,
    d0_static_sweep_values,
    load_d0_spec,
    validate_d0_spec,
)


def test_d0_spec_frozen_and_id():
    spec = load_d0_spec()
    assert spec["status"] == "FROZEN"
    assert spec["protocol_id"] == PROTOCOL_ID
    assert spec["checkpoint"] == "D0"
    assert spec["write_once"] is True


def test_d0_central_containment_question():
    spec = load_d0_spec()
    assert "classical emitter" in spec["containment_thesis"]["display"].lower()
    null = spec["containment_thesis"]["formal_null"]
    assert "H_K" in null
    assert "E_classical" in null


def test_d0_dot_w_zero_hdp_off():
    runtime = load_d0_spec()["runtime_contract"]
    assert runtime["dot_W"] == 0
    assert runtime["enable_hdp"] is False


def test_d0_ionic_vector_grammar_one_d1_coordinate():
    assert d0_first_coordinate_id() == "H_K"
    coords = load_d0_spec()["rbs_vector_grammar"]["typed_coordinates"]
    d1 = [c for c in coords if c["d1_implementation"]]
    assert len(d1) == 1
    assert d1[0]["id"] == "H_K"


def test_d0_h_k_effective_channel_not_concentration():
    interp = load_d0_spec()["first_coordinate_H_K"]["physical_interpretation"]
    assert "availability" in interp["is"].lower() or "gain" in interp["is"].lower()
    assert "potassium concentration" in interp["is_not"]
    assert "Nernst potential" in interp["is_not"]


def test_d0_typed_coupling_g_k_eff():
    gamma = load_d0_spec()["first_coordinate_H_K"]["typed_coupling_Gamma"]
    assert "g_K^eff" in gamma["map_display"] or "g_K" in gamma["map_display"]
    assert gamma["normalization"]


def test_d0_static_sweep_preregistered():
    assert d0_static_sweep_values() == (0.8, 1.0, 1.2)
    sweep = load_d0_spec()["static_sweep_before_dynamics"]
    assert "dH_K/dt = 0" in " ".join(sweep["hold_fixed"])


def test_d0_checkpoint_ladder_d4_not_mandatory():
    ladder = load_d0_spec()["checkpoint_ladder"]
    assert ladder["D4"]["mandatory_for_0417"] is False
    assert ladder["D1"]["implementation_authorized"] is False


def test_d0_explicit_prohibitions():
    banned = set(load_d0_spec()["explicit_prohibitions"])
    assert "no_HDP_dot_W" in banned
    assert "no_STDP_coordinate_in_D1" in banned
    assert "no_neurotransmitter_coordinate_in_D1" in banned
    assert "no_implement_all_three_ionic_coordinates_in_D1" in banned


def test_d0_naming_disambiguation_from_delay_protocol():
    dis = load_d0_spec()["naming_disambiguation"]
    assert "edge-delay" in dis["distinct_from"] or "delay" in dis["distinct_from"].lower()


def test_d0_manuscript_c_and_h4_separate():
    fig = load_d0_spec()["manuscript_figure_discipline"]
    assert "protocol_c" in fig
    assert "protocol_h4" in fig
    assert "separate" in fig["rule"].lower()


def test_d0_validate_spec_passes():
    validate_d0_spec()


def test_d0_implementation_not_authorized():
    spec = load_d0_spec()
    assert spec["execution_authorization"]["implementation_authorized"] is False


def test_d0_protocol_receipt_frozen():
    receipt = json.loads((D0_SPEC_PATH.parent / "d0_protocol_receipt.json").read_text())
    assert receipt["checkpoint"] == "D0"
    assert receipt["status"] == "FROZEN"
    assert receipt["implementation_authorized"] is False
    assert receipt["first_coordinate"] == "H_K"
    assert receipt["next_checkpoint"] == "D1"
