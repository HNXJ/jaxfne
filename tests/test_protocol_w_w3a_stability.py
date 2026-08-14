"""Protocol W3a — activity-enabled stability analysis receipts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxfne.w3a_stability_analysis import (
    W3_SILENT_REST_SHA,
    W3aScanConfig,
    export_w3a_stability_receipt,
    run_w3a_stability_scan,
)

RECEIPT = Path("artifacts/protocol_w/w3a_stability/w3a_stability_receipt.json")
SPEC = Path("artifacts/protocol_w/w3a_stability/w3a_stability_spec.json")


def _fast_cfg() -> W3aScanConfig:
    return W3aScanConfig(
        i_tonic_grid=(0.0, 10.0, 20.0),
        burn_in_steps=80,
        sample_steps=120,
        period_search_max=40,
    )


def test_silent_rest_b_hw_zero_at_i_zero():
    out = run_w3a_stability_scan(cfg=_fast_cfg())
    s0 = out["scan_results"][0]
    assert s0["I_tonic"] == 0.0
    assert s0["b_HW"] == pytest.approx(0.0, abs=1e-12)
    assert s0["loop_gain"]["L_HDP"] == pytest.approx(0.0, abs=1e-12)


def test_active_drive_gives_nonzero_b_hw_and_loop_gain():
    out = run_w3a_stability_scan(cfg=_fast_cfg())
    s20 = next(s for s in out["scan_results"] if s["I_tonic"] == 20.0)
    assert s20["syn_active"]
    assert abs(s20["b_HW"]) > 1e-6
    assert s20["loop_gain"]["L_HDP"] > 0.0


def test_w3a_fp_active_not_found_on_nominal_scan():
    out = run_w3a_stability_scan(cfg=_fast_cfg())
    assert not out["gates"]["W3a_FP"]["active_fp_found"]


def test_w3a_po_activated_when_syn_active():
    out = run_w3a_stability_scan(cfg=_fast_cfg())
    active = [s for s in out["scan_results"] if s["w3a_po"].get("activated")]
    assert len(active) >= 1


def test_preserves_silent_rest_sha_reference():
    out = run_w3a_stability_scan(cfg=_fast_cfg())
    assert out["preserved_silent_rest_result_sha"] == W3_SILENT_REST_SHA


def test_frozen_spec_and_receipt_exist():
    assert SPEC.is_file()
    assert RECEIPT.is_file()
    spec = json.loads(SPEC.read_text())
    assert spec["preserved_prerequisite"]["silent_rest_w3_sha"] == W3_SILENT_REST_SHA
    frozen = json.loads(RECEIPT.read_text())
    assert frozen["schema"] == "protocol_w_w3a_stability_receipt.v1"
    assert frozen["gates"]["W3a_FP"]["active_fp_found"] is False


def test_exporter_matches_frozen_receipt_fp_gate():
    live = export_w3a_stability_receipt(cfg=_fast_cfg())
    frozen = json.loads(RECEIPT.read_text())
    assert live["gates"]["W3a_FP"]["active_fp_found"] is False
    assert frozen["gates"]["W3a_FP"]["active_fp_found"] is False
