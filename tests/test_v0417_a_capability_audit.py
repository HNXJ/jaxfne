"""0.4.17-A — frozen capability audit receipt tests."""

from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path("artifacts/audit/v0417_a_capability_audit.json")
BASELINE_SHA = "15f32b3a885dae0d84cb0163bc47c153302898c4"


def test_frozen_capability_audit_baseline_and_status():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "FROZEN"
    assert audit["baseline"]["commit"] == BASELINE_SHA
    assert audit["baseline"]["version"] == "0.4.16"


def test_frozen_capability_audit_panel_counts():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    totals = audit["panel_readiness_map"]["totals"]
    assert sum(totals.values()) == len(audit["panels"])
    assert totals["BLOCKED"] >= 4
    assert audit["figure_6_frozen_ladder"]["closed_HDP_loop"] == "unresolved"


def test_frozen_capability_audit_figure4_eeg_meg_not_ready():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    fig4 = [p for p in audit["panels"] if p["figure"] == 4]
    eeg = next(p for p in fig4 if p["panel"] == "4C")
    meg = next(p for p in fig4 if p["panel"] == "4D")
    assert eeg["status"] == "ANALYSIS_ONLY"
    assert meg["status"] == "ANALYSIS_ONLY"
    calibrated = [p for p in fig4 if p["status"] == "NEEDS_NEW_SCIENCE"]
    assert any(p["panel"] in ("4G", "4H") for p in calibrated)


def test_minimal_delta_excludes_w3_closed_loop():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    excluded = audit["explicitly_not_in_minimal_delta"]
    assert any("W3 closed-loop" in x for x in excluded)
    assert any("W3c" in x for x in excluded)
