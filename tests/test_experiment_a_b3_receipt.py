"""0.4.17-B B3 — Experiment A receipt bundle tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaxfne.experiment_a.canonical import freeze_canonical_dataset
from jaxfne.experiment_a.receipt import run_observation_suite

BUNDLE = Path("artifacts/etudes/experiment_a")


def test_b3_observation_suite_levels_and_semantics():
    ds = freeze_canonical_dataset(duration_ms=120.0, package_head="test_b3")
    suite = run_observation_suite(ds)
    assert suite["levels"]["A"]
    assert suite["levels"]["B"]
    assert suite["b2_invariants"]["q_hash_invariant"]
    assert suite["operator_status"]["eeg_superficial"] == "analysis_only"
    assert suite["operator_status"]["lfp_ref"] == "relative_proxy"


@pytest.mark.skipif(not (BUNDLE / "b3_experiment_a_receipt.json").exists(), reason="run runner first")
def test_b3_frozen_receipt_in_repo():
    receipt = json.loads((BUNDLE / "b3_experiment_a_receipt.json").read_text())
    assert receipt["checkpoint"] == "B3"
    assert receipt["status"] == "FROZEN"
    metrics = json.loads((BUNDLE / "metrics.json").read_text())
    assert metrics["q_hash_invariant"] is True
