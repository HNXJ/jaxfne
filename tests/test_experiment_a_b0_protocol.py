"""0.4.17-B B0 — frozen Experiment A protocol specification tests."""

from __future__ import annotations

import json

import pytest

from jaxfne.experiment_a import PROTOCOL_ID, PROTOCOL_SPEC_PATH, load_protocol_spec

AUDIT_BASELINE = "6117e2bf28156bfa56da6abb4d3ab7ae74abb078"
RELEASE_BASELINE = "15f32b3a885dae0d84cb0163bc47c153302898c4"


def test_b0_protocol_spec_exists_and_frozen():
    spec = load_protocol_spec()
    assert spec["status"] == "FROZEN"
    assert spec["protocol_id"] == PROTOCOL_ID
    assert spec["write_once"] is True


def test_b0_protocol_baseline_links():
    spec = load_protocol_spec()
    assert spec["baseline"]["commit"] == RELEASE_BASELINE
    assert spec["baseline"]["audit_commit"] == AUDIT_BASELINE
    assert spec["baseline"]["version"] == "0.4.16"


def test_b0_causal_architecture_and_anti_drift():
    spec = load_protocol_spec()
    arch = spec["causal_architecture"]
    assert "Q" in arch["first_class_frozen_artifacts"]
    assert "one neural simulate" in arch["anti_drift_rule"].lower()


def test_b0_eeg_meg_analysis_only_not_manuscript():
    spec = load_protocol_spec()
    macro = [p for p in spec["probe_operators_P"] if p["id"].startswith(("eeg_", "meg_"))]
    assert macro, "expected EEG/MEG probe entries"
    for entry in macro:
        assert entry["semantic"] == "analysis_only"
        assert entry.get("manuscript_claim") is False


def test_b0_invariants_predeclared():
    spec = load_protocol_spec()
    ids = {inv["id"] for inv in spec["invariants"]}
    assert "Q_zero_implies_Y_zero" in ids
    assert "F_superposition" in ids
    assert "probe_does_not_mutate_Q" in ids
    assert "B2_Q_invariant_probe_distinct" in ids


def test_b0_excludes_out_of_scope_capabilities():
    spec = load_protocol_spec()
    excluded = set(spec["excluded_from_0417_B"])
    assert "calibrated_EEG_MEG_solver" in excluded
    assert "wave_estimator" in excluded
    assert "multi_area_runtime" in excluded


def test_b0_spec_roundtrip_json():
    raw = json.loads(PROTOCOL_SPEC_PATH.read_text(encoding="utf-8"))
    assert raw["schema"] == "jaxfne.experiment_a.b0_protocol_spec.v1"
    assert raw["checkpoints"]["B0"].startswith("Freeze")
