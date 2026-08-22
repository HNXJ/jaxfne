"""0.4.17-C3/C4 — prospective execution and interpretation receipt tests."""

from __future__ import annotations

import json

from jaxfne.protocol_c.c3_execution import (
    C3_CONDITION_SUMMARY_PATH,
    C3_EXECUTION_RECEIPT_PATH,
    load_c3_execution_receipt,
)
from jaxfne.protocol_c.c3_protocol import c3_total_cells, load_c3_spec
from jaxfne.protocol_c.c4_interpretation import (
    C4_INTERPRETATION_RECEIPT_PATH,
    load_c4_interpretation_receipt,
    summarize_conditions,
)


def test_c3_execution_receipt_has_60_cells():
    receipt = load_c3_execution_receipt()
    assert receipt["status"] == "FROZEN"
    assert receipt["n_cells"] == 60
    assert receipt["n_cells"] == c3_total_cells()
    assert len(receipt["cells"]) == 60


def test_c3_each_cell_has_full_estimator_contract():
    required = {
        "classification",
        "frequency_hz",
        "wave_vector",
        "direction",
        "phase_velocity",
        "phase_fit_r2",
        "spatial_coherence",
        "null_score",
        "quality_reasons",
        "finite_status",
    }
    for cell in load_c3_execution_receipt()["cells"]:
        est = cell["estimator"]
        for key in required:
            assert key in est
        assert est["classification"] in {"TRAVELING_WAVE", "NO_WAVE", "UNRESOLVED"}


def test_c3_condition_summary_counts_sum_to_10():
    summary = json.loads(C3_CONDITION_SUMMARY_PATH.read_text(encoding="utf-8"))
    for row in summary["per_condition"]:
        assert row["N_TW"] + row["N_NW"] + row["N_U"] == 10
        assert row["p_W"] == row["N_TW"] / 10.0
        assert row["p_U"] == row["N_U"] / 10.0


def test_c4_preregistered_delta_p_w_recorded():
    summary = json.loads(C3_CONDITION_SUMMARY_PATH.read_text(encoding="utf-8"))
    c = summary["contrasts"]
    og = next(r for r in summary["per_condition"] if r["condition_id"] == "ordered_geometry_derived")
    ou = next(r for r in summary["per_condition"] if r["condition_id"] == "ordered_uniform")
    assert c["delta_p_W_ordered_geometry_derived_minus_uniform"] == og["p_W"] - ou["p_W"]


def test_c4_interpretation_receipt_frozen():
    c4 = load_c4_interpretation_receipt()
    assert c4["status"] == "FROZEN"
    assert c4["outcome_letter"] in {"A", "B", "C", "D"}
    assert c4["protocol_c_closed"] is True
    assert c4["evidence_level"]["wave_mechanism"] == "NOT_CLAIMED"


def test_c4_outcome_d_distinct_from_c():
    c4 = load_c4_interpretation_receipt()
    assert c4["outcomes_reference"]["D"] != c4["outcomes_reference"]["C"]


def test_c3_raw_receipt_before_interpretation_paths():
    receipt = load_c3_execution_receipt()
    assert receipt["interpretation_deferred_to"] == "C4"
    assert C3_EXECUTION_RECEIPT_PATH.exists()
    assert C4_INTERPRETATION_RECEIPT_PATH.exists()


def test_summarize_matches_frozen_receipt():
    summary = summarize_conditions(load_c3_execution_receipt())
    frozen = json.loads(C3_CONDITION_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["per_condition"] == frozen["per_condition"]
