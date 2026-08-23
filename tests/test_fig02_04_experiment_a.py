"""Coordinated Experiment A publication figures 2–4 tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]

import pytest

from jaxfne.experiment_a.canonical import load_frozen_canonical_dataset
from scripts.internal.publication.fig02_04_protocol import (
    FIG02_04_AUDIT_PATH,
    FIG02_PATH,
    FIG03_PATH,
    FIG04_PATH,
    load_fig02_04_cross_audit,
    load_fig02_04_spec,
    validate_fig02_04_cross_audit,
    validate_fig02_04_receipts,
    validate_fig02_04_spec,
)

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "artifacts" / "etudes" / "experiment_a" / "canonical_source.npz"
B1 = ROOT / "artifacts" / "etudes" / "experiment_a" / "b1_canonical_receipt.json"
GENERATOR = ROOT / "scripts" / "publication_figures" / "fig02_04_experiment_a.py"


def test_fig02_04_spec_frozen():
    spec = load_fig02_04_spec()
    assert spec["status"] == "FROZEN"
    assert len(spec["pec_panel_ids"]) == 3


def test_validate_fig02_04_spec():
    validate_fig02_04_spec()


@pytest.mark.skipif(not NPZ.is_file(), reason="canonical_source.npz not present locally")
def test_load_frozen_canonical_matches_b1_receipt():
    ds = load_frozen_canonical_dataset(npz_path=NPZ, receipt_path=B1)
    receipt = json.loads(B1.read_text(encoding="utf-8"))
    assert ds.cause_hashes["Q"] == receipt["cause_hashes"]["Q"]


@pytest.mark.skipif(not NPZ.is_file(), reason="canonical_source.npz not present locally")
def test_fig02_04_generator_and_cross_audit():
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        env={**os.environ, "PYTHONPATH": str(_SCRIPT_REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        cwd=GENERATOR.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr or result.stdout
    audit = load_fig02_04_cross_audit()
    validate_fig02_04_cross_audit(audit)
    validate_fig02_04_receipts()
    assert FIG02_PATH.is_file() and FIG03_PATH.is_file() and FIG04_PATH.is_file()
    q_set = set(audit["q_hashes_by_figure"].values())
    assert len(q_set) == 1
    receipt = json.loads((ROOT / "artifacts/publication/fig04_generation_receipt.json").read_text(encoding="utf-8"))
    assert receipt.get("eeg_meg_ready_panels") == 0
