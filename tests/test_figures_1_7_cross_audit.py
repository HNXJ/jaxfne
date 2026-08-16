"""Figures 1–7 cross-figure audit tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from jaxfne.publication.cross_figure_audit import (
    load_cross_figure_audit,
    run_cross_figure_audit,
    validate_cross_figure_audit,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "publication_figures" / "figures_1_7_cross_audit.py"


def test_cross_audit_logic_passes():
    audit = run_cross_figure_audit()
    assert audit["status"] == "PASSED"
    assert audit["frozen_scientific_boundaries"]["W3b_unresolved_not_negative"] is True
    assert audit["checks"]["main_figure_evidence_set_complete"] is True


def test_cross_audit_generator():
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        cwd=GENERATOR.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    validate_cross_figure_audit()
    audit = load_cross_figure_audit()
    assert audit["next_checkpoint"] == "publication_reconstruction"
    assert len(audit["figure_provenance"]) == 7
