"""Figure 5 Protocol C publication tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jaxfne.publication.fig05_protocol import (
    FIG05_AUDIT_PATH,
    FIG05_PATH,
    FIG05_RECEIPT_PATH,
    load_fig05_generation_receipt,
    load_fig05_semantic_audit,
    load_fig05_spec,
    validate_fig05_generation_receipt,
    validate_fig05_semantic_audit,
    validate_fig05_spec,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "publication_figures" / "fig05_protocol_c.py"


def test_fig05_spec_frozen_negative():
    spec = load_fig05_spec()
    assert spec["status"] == "FROZEN"
    assert spec["polarity"] == "NEGATIVE"
    assert spec["frozen_quantities"]["N_U"] == 0


def test_validate_fig05_spec():
    validate_fig05_spec()


def test_fig05_generator_and_audit():
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        cwd=GENERATOR.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    audit = load_fig05_semantic_audit()
    validate_fig05_semantic_audit(audit)
    validate_fig05_generation_receipt()
    assert FIG05_PATH.is_file()
    assert audit["checks"]["zero_unresolved"] is True
    receipt = load_fig05_generation_receipt()
    assert receipt["polarity"] == "NEGATIVE"
    assert receipt["outcome_letter"] == "C"
