"""0.4.17-D D3 — adaptation/recovery phenotype specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_d_biological_rbs.d3_protocol import (
    D3_SPEC_PATH,
    d3_null_arm_ids,
    d3_recovery_interval_ms,
    load_d3_spec,
    validate_d3_spec,
)


def test_d3_spec_frozen():
    spec = load_d3_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "D3"
    assert spec["write_once"] is True


def test_d3_paradigm_four_phases():
    phases = [p["id"] for p in load_d3_spec()["paradigm"]["phases"]]
    assert phases == ["baseline", "repeated_stimulation", "recovery_interval", "rechallenge"]


def test_d3_identical_pulse_train():
    pt = load_d3_spec()["pulse_train"]
    assert pt["identical_pulses"] is True
    assert pt["n_pulses_m"] >= 2
    assert pt["amplitude"] == 15.0


def test_d3_adaptation_index_definition():
    idx = load_d3_spec()["adaptation_index"]
    assert "1 - R_late / R_early" in idx["display"]
    assert idx["interpretation"]["negative"]


def test_d3_recovery_index_secondary():
    rec = load_d3_spec()["recovery_index"]
    assert "R_rechallenge" in rec["display"]
    assert rec["classification_role"] == "secondary; not required for initial ADAPTATION classification"


def test_d3_null_hierarchy_four_arms():
    assert d3_null_arm_ids() == ("N0", "N1", "N2", "D")
    contrast = load_d3_spec()["null_hierarchy"]["primary_contrast"]
    assert contrast["id"] == "D_minus_N2"


def test_d3_recovery_intervals_from_timescales():
    spec = load_d3_spec()
    assert d3_recovery_interval_ms() == (50.0, 100.0, 250.0)
    assert spec["recovery_intervals"]["tau_A_ms"] == 25.0
    assert spec["recovery_intervals"]["tau_K_ms"] == 100.0


def test_d3_three_way_classification():
    labels = load_d3_spec()["classification"]["labels"]
    assert labels == ["ADAPTATION", "NO_ADAPTATION", "UNRESOLVED"]
    forbidden = set(load_d3_spec()["classification"]["forbidden_collapse"])
    assert any("silence" in f for f in forbidden)


def test_d3_frozen_thresholds_preregistered():
    th = load_d3_spec()["frozen_thresholds"]
    assert th["post_hoc_tuning_forbidden"] is True
    assert th["theta_A"] > 0
    assert th["theta_H"] > 0
    assert th["min_mean_R_early"] > 0


def test_d3_mechanism_checks_separate_from_phenotype():
    checks = load_d3_spec()["hidden_state_mechanism_checks"]
    assert "not sufficient" in checks["scope"]
    assert "D2b failure" in checks["informative_null_outcome"]
    rules = " ".join(load_d3_spec()["interpretation_rules"])
    assert "NO_ADAPTATION is a valid" in rules


def test_d3_no_fatigue_formal_mechanism():
    spec = load_d3_spec()
    assert "fatigue" in spec["terminology"]["avoid_as_formal_mechanism"]
    assert "no_fatigue_as_formal_mechanism_label" in spec["explicit_prohibitions"]
    assert "fatigue" not in spec["phenomenon_label"].lower()


def test_d3_implementation_not_authorized():
    auth = load_d3_spec()["execution_authorization"]
    assert auth["implementation_authorized"] is False
    assert auth["specification_only"] is True


def test_d3_validate_spec_passes():
    validate_d3_spec()


def test_d3_protocol_receipt_frozen():
    from jaxfne.protocol_d_biological_rbs.d3_protocol import D3_SPEC_PATH

    receipt = json.loads((D3_SPEC_PATH.parent / "d3_protocol_receipt.json").read_text())
    assert receipt["status"] == "CLOSED"
    assert receipt["protocol_d_closed"] is True
    assert receipt["D4_status"] == "not_authorized"
    assert receipt["next_checkpoint"] == "E0_specification"
