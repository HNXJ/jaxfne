"""Figure 6 H/W/D evidence tests."""

from __future__ import annotations

import subprocess
import os
import sys
from pathlib import Path
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]

from jaxfne.publication.fig06_evidence import h4_primary_mx, load_fig06_evidence, w3b_counts
from jaxfne.publication.fig06_protocol import (
    FIG06_PATH,
    load_fig06_generation_receipt,
    load_fig06_semantic_audit,
    load_fig06_spec,
    validate_fig06_generation_receipt,
    validate_fig06_semantic_audit,
    validate_fig06_spec,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "publication_figures" / "fig06_hwd_evidence.py"


def test_fig06_spec_frozen():
    spec = load_fig06_spec()
    assert spec["status"] == "FROZEN"
    assert "E5" in spec["excluded_content"]


def test_receipt_quantities():
    ev = load_fig06_evidence()
    mx = h4_primary_mx(ev)
    assert mx["M_X_long_heterogeneous"] == 0.0
    assert mx["M_X_short_heterogeneous"] > 0
    c = w3b_counts(ev)
    assert c["N_S"] == 0 and c["N_X"] == 1944


def test_fig06_generator():
    result = subprocess.run(
        [sys.executable, str(GENERATOR)],
        env={**os.environ, "PYTHONPATH": str(_SCRIPT_REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        cwd=GENERATOR.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr or result.stdout
    validate_fig06_spec()
    validate_fig06_semantic_audit()
    validate_fig06_generation_receipt()
    assert FIG06_PATH.is_file()
    audit = load_fig06_semantic_audit()
    assert audit["checks"]["d3_no_adaptation"] is True
    receipt = load_fig06_generation_receipt()
    assert receipt["next_checkpoint"] == "figure_7_generation"
