"""Figure 7 E-integration tests."""

from __future__ import annotations

import subprocess
import os
import sys
from pathlib import Path
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.internal.publication.fig07_evidence import (
    e2_delay_classes,
    e3_owner,
    e5_null_controls,
    e5_propagation_metrics,
    load_fig07_evidence,
)
from scripts.internal.publication.fig07_protocol import (
    FIG07_PATH,
    load_fig07_generation_receipt,
    load_fig07_semantic_audit,
    load_fig07_spec,
    validate_fig07_generation_receipt,
    validate_fig07_semantic_audit,
    validate_fig07_spec,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "publication_figures" / "fig07_e_integration.py"


def test_fig07_spec_frozen():
    spec = load_fig07_spec()
    assert spec["status"] == "FROZEN"
    assert len(spec["pec_panel_ids"]) == 6


def test_frozen_receipt_quantities():
    ev = load_fig07_evidence()
    delays = e2_delay_classes(ev)
    assert [d["tau_ms"] for d in delays] == [1.0, 2.0, 4.0]
    owner = e3_owner(ev)
    assert owner["flat_indices"] == list(range(70, 77))
    nulls = e5_null_controls(ev)
    assert len(nulls["seeds"]) == 3
    prop = e5_propagation_metrics(ev)
    assert prop["classification"] == "HIERARCHICAL_PROPAGATION"


def test_fig07_generator():
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        env={**os.environ, "PYTHONPATH": str(_SCRIPT_REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        cwd=GENERATOR.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr or result.stdout
    validate_fig07_spec()
    validate_fig07_semantic_audit()
    validate_fig07_generation_receipt()
    assert FIG07_PATH.is_file()
    receipt = load_fig07_generation_receipt()
    assert receipt["main_figure_evidence_set"] == "COMPLETE"
    assert receipt["next_checkpoint"] == "figures_1_7_cross_audit"
    audit = load_fig07_semantic_audit()
    assert audit["checks"]["n0_equals_n1_all_seeds"] is True
