"""0.4.17-C C0 — frozen wave protocol specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_c import PROTOCOL_ID, PROTOCOL_SPEC_PATH, load_protocol_spec


def test_c0_protocol_frozen_and_id():
    spec = load_protocol_spec()
    assert spec["status"] == "FROZEN"
    assert spec["protocol_id"] == PROTOCOL_ID
    assert spec["write_once"] is True


def test_c0_separates_delay_wave_and_feedback():
    sep = load_protocol_spec()["scope_separation"]
    assert sep["C_D"]["status"] == "OUT_OF_SCOPE_FOR_C_IMPLEMENTATION"
    assert sep["C_W"]["status"] == "IN_SCOPE"
    assert sep["C_F"]["status"] == "CLOSED"


def test_c0_estimator_cannot_feed_upstream():
    spec = load_protocol_spec()
    assert spec["causal_hierarchy"]["upstream_feed_prohibited"] is True
    assert spec["causal_hierarchy"]["estimator_symbol"] == "W_hat"


def test_c0_three_way_classification():
    labels = load_protocol_spec()["classification_labels"]
    assert set(labels.keys()) >= {"TRAVELING_WAVE", "NO_WAVE", "UNRESOLVED"}
    assert "UNRESOLVED -> NO_WAVE" in labels["forbidden_collapse"]


def test_c0_synthetic_controls_preregistered():
    spec = load_protocol_spec()
    pos = spec["synthetic_controls"]["positive_planar_wave"]
    assert "v_phase = omega/|k|" in pos["known_quantities"][2] or "v_phase" in str(pos["known_quantities"])
    negs = {c["id"] for c in spec["synthetic_controls"]["negative_controls"]}
    assert "sync_k_zero" in negs
    assert "noise_only" in negs


def test_c0_no_implementation_at_c0():
    receipt = json.loads(
        (PROTOCOL_SPEC_PATH.parent / "c0_wave_protocol_receipt.json").read_text()
    )
    assert receipt["checkpoint"] == "C0"
    assert receipt["implementation_authorized"] is False


def test_c0_excludes_field_feedback_and_figures():
    excluded = set(load_protocol_spec()["excluded_from_0417_C"])
    assert "field_ephaptic_feedback_C_F" in excluded
    assert "manuscript_figure_generation" in excluded


def test_c0_experiment_a_must_not_force_wave():
    assert load_protocol_spec()["experiment_a_integration"]["must_not_force_wave_on_experiment_a"]


def test_c0_operational_wave_requires_more_than_phase_diff():
    op = load_protocol_spec()["operational_wave_definition"]
    assert op["requires_more_than_phase_difference"] is True
