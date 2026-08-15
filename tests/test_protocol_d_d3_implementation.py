"""0.4.17-D D3 — adaptation/recovery implementation and interpretation tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxfne.protocol_d_biological_rbs.d3_execution import (
    D3_EXECUTION_RECEIPT_PATH,
    build_d3_drive_schedule,
    build_d3_response_window_overlap_metadata,
    classify_d3_cell,
    compute_adaptation_indices,
    load_d3_execution_receipt,
    run_d3_adaptation_recovery,
    write_d3_execution_receipt,
)
from jaxfne.protocol_d_biological_rbs.d3_interpretation import (
    D3_INTERPRETATION_RECEIPT_PATH,
    build_d3_interpretation,
    load_d3_interpretation_receipt,
    write_d3_interpretation_receipt,
)
from jaxfne.protocol_d_biological_rbs.d3_protocol import load_d3_spec


def test_d3_response_window_overlap_frozen_20ms():
    spec = load_d3_spec()
    meta = build_d3_response_window_overlap_metadata(spec)
    assert meta["response_window_ms"] == 80.0
    assert meta["isi_ms"] == 60.0
    overlaps = meta["train_pulse_overlaps"]
    assert len(overlaps) == 5
    assert all(o["overlap_ms"] == pytest.approx(20.0) for o in overlaps)


def test_d3_drive_schedule_train_and_rechallenge():
    spec = load_d3_spec()
    sched, onsets, rech = build_d3_drive_schedule(spec, T_recovery_ms=100.0)
    assert len(onsets) == 6
    assert onsets[0] == pytest.approx(100.0)
    assert onsets[1] == pytest.approx(160.0)
    assert rech == pytest.approx(540.0)
    assert sched.shape == (2000, 1)
    assert float(np.max(sched)) == pytest.approx(15.0)


def test_d3_adaptation_indices_from_R_values():
    spec = load_d3_spec()
    out = compute_adaptation_indices([4, 4, 3, 2, 1, 1], 3, spec)
    assert out["R_early"] == pytest.approx(4.0)
    assert out["R_late"] == pytest.approx(1.0)
    assert out["A_adapt"] == pytest.approx(0.75)
    assert out["R_recovery"] == pytest.approx((3.0 - 1.0) / (4.0 - 1.0))


def test_d3_classify_unresolved_sparse_early():
    spec = load_d3_spec()
    cell = {"null_arm": "D", "R_early": 0.5, "A_adapt": None, "mechanism": {"mechanism_ok": True}}
    out = classify_d3_cell(cell, spec)
    assert out["classification"] == "UNRESOLVED"


def test_d3_classify_D_requires_mechanism():
    spec = load_d3_spec()
    cell = {
        "null_arm": "D",
        "R_early": 2.0,
        "A_adapt": 0.5,
        "mechanism": {"mechanism_ok": False, "M1_pass": False, "M2_pass": True},
    }
    assert classify_d3_cell(cell, spec)["classification"] == "NO_ADAPTATION"
    cell["mechanism"]["mechanism_ok"] = True
    cell["mechanism"]["M1_pass"] = True
    assert classify_d3_cell(cell, spec)["classification"] == "ADAPTATION"


def test_d3_classify_facilitation_flag():
    spec = load_d3_spec()
    cell = {"null_arm": "N2", "R_early": 3.0, "A_adapt": -0.2}
    out = classify_d3_cell(cell, spec)
    assert out["classification"] == "NO_ADAPTATION"
    assert out["facilitation"] is True


def test_d3_execution_receipt_36_cells():
    if not D3_EXECUTION_RECEIPT_PATH.exists():
        write_d3_execution_receipt()
    ex = load_d3_execution_receipt()
    assert ex["n_cells"] == 36
    assert "response_window_semantics" in ex
    for c in ex["cells"]:
        assert len(c["R_train"]) == 6
        assert "R_early" in c
        assert "classification" in c
        if c["null_arm"] == "D":
            assert "H_A_trace" in c
            assert "H_K_trace" in c
            assert "mechanism" in c


def test_d3_classification_reconstructs_from_raw():
    ex = load_d3_execution_receipt()
    spec = load_d3_spec()
    for c in ex["cells"]:
        rebuilt = classify_d3_cell(c, spec)
        assert rebuilt["classification"] == c["classification"]
        assert rebuilt.get("facilitation") == c.get("facilitation")


def test_d3_interpretation_receipt_questions():
    if not D3_INTERPRETATION_RECEIPT_PATH.exists():
        ex = load_d3_execution_receipt()
        write_d3_interpretation_receipt(ex)
    interp = load_d3_interpretation_receipt()
    assert "Q1_mechanism" in interp["questions"]
    assert "Q2_adaptation" in interp["questions"]
    assert "Q3_recovery" in interp["questions"]
    assert "primary_contrast_D_minus_N2" in interp
    assert interp["d2b_not_invalidated_by_NO_ADAPTATION"] is True


def test_d3_in_memory_run_matches_cell_grid():
    receipt = run_d3_adaptation_recovery()
    assert receipt["n_cells"] == 36
    arms = {c["null_arm"] for c in receipt["cells"]}
    assert arms == {"N0", "N1", "N2", "D"}


def test_d3_protocol_receipt_closed_after_execution():
    from jaxfne.protocol_d_biological_rbs.d3_protocol import D3_SPEC_PATH

    proto = json.loads((D3_SPEC_PATH.parent / "d3_protocol_receipt.json").read_text())
    assert proto["status"] == "CLOSED"
    assert proto["implementation_authorized"] is True
    assert proto["n_cells_executed"] == 36
