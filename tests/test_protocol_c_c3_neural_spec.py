"""0.4.17-C C3 — frozen neural geometry/delay experiment specification tests."""

from __future__ import annotations

import json

from jaxfne.protocol_c.c3_protocol import (
    C3_SPEC_PATH,
    c3_condition_ids,
    c3_total_cells,
    load_c3_spec,
    validate_c3_spec,
)
from jaxfne.protocol_c.protocol import PROTOCOL_ID


def test_c3_spec_frozen_and_id():
    spec = load_c3_spec()
    assert spec["status"] == "FROZEN"
    assert spec["protocol_id"] == PROTOCOL_ID
    assert spec["checkpoint"] == "C3"
    assert spec["write_once"] is True


def test_c3_causal_chain_primary_x_only():
    spec = load_c3_spec()
    chain = spec["causal_chain"]
    assert chain["primary_substrate"] == "X"
    assert chain["upstream_feed_prohibited"] is True
    assert "delayed" in chain["protocol_d_vs_c"]["protocol_d"].lower()
    assert "propagation" in chain["protocol_d_vs_c"]["protocol_c"].lower()


def test_c3_runtime_hdp_disabled():
    runtime = load_c3_spec()["runtime_contract"]
    assert runtime["enable_hdp"] is False
    assert runtime["enable_homeostasis"] is False
    assert runtime["enable_rbd"] is False


def test_c3_design_includes_geometry_and_delay_shuffle_controls():
    ids = c3_condition_ids()
    assert "ordered_uniform" in ids
    assert "ordered_geometry_derived" in ids
    assert "ordered_delay_shuffled" in ids
    assert "shuffled_geometry_derived" in ids
    assert len(ids) == 6


def test_c3_delay_shuffle_preserves_multiset_rule():
    spec = load_c3_spec()
    ctrl = spec["delay_construction"]["delay_shuffle_control"]
    assert "multiset" in ctrl["preserves"].lower() or "exact_delay_steps_multiset" in ctrl["preserves"]


def test_c3_three_way_classification_not_collapsed():
    interp = load_c3_spec()["per_seed_condition_output_contract"]
    assert "UNRESOLVED" in interp["classification_labels"]
    assert "UNRESOLVED -> NO_WAVE" in interp["forbidden_collapse"]


def test_c3_population_endpoints_p_w_and_p_u():
    pop = load_c3_spec()["population_endpoints"]
    assert pop["primary"]["symbol"] == "p_W"
    assert pop["identifiability"]["symbol"] == "p_U"


def test_c3_directional_conjecture_not_required_for_success():
    conj = load_c3_spec()["directional_conjecture"]
    assert conj["required_for_c3_success"] is False


def test_c3_v_c_diagnostic_not_theoretical_equivalence():
    diag = load_c3_spec()["v_c_diagnostic"]
    assert diag["interpretation"] == "diagnostic_only"
    assert "equals" in diag["forbidden_claim"] or "equals_axonal" in diag["forbidden_claim"]


def test_c3_explicit_prohibitions():
    banned = set(load_c3_spec()["explicit_prohibitions"])
    assert "no_HDP" in banned
    assert "no_W3_W3c_machinery" in banned
    assert "no_ephaptic_or_field_feedback" in banned
    assert "no_H4_M_X_as_wave_evidence" in banned


def test_c3_estimator_bound_to_c1():
    binding = load_c3_spec()["estimator_binding"]
    assert "estimate_traveling_wave" in binding["implementation"]
    assert binding["c1_validation_receipt"].endswith("c1_synthetic_validation_receipt.json")
    assert binding["frequency_band_hz"] == [8.0, 13.0]


def test_c3_total_cells():
    assert c3_total_cells() == 60


def test_c3_validate_spec_passes():
    validate_c3_spec()


def test_c3_prospective_run_not_authorized():
    spec = load_c3_spec()
    assert spec["execution_authorization"]["prospective_run_authorized"] is False


def test_c3_protocol_receipt_closed():
    receipt = json.loads(
        (C3_SPEC_PATH.parent / "c3_protocol_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["checkpoint"] == "C3"
    assert receipt["status"] == "CLOSED"
    assert receipt["prospective_run_executed"] is True
    assert receipt["protocol_c_closed"] is True
    assert receipt["outcome_letter"] == "C"
    assert receipt["next_checkpoint"] == "0.4.17-D"
