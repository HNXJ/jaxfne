"""Figure 1 grammar publication tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]

import pytest

from jaxfne.publication.fig01_protocol import (
    FIG01_AUDIT_PATH,
    FIG01_FIGURE_PATH,
    FIG01_RECEIPT_PATH,
    FIG01_SPEC_PATH,
    load_fig01_generation_receipt,
    load_fig01_semantic_audit,
    load_fig01_spec,
    validate_fig01_generation_receipt,
    validate_fig01_semantic_audit,
    validate_fig01_spec,
)


def test_fig01_spec_frozen():
    spec = load_fig01_spec()
    assert spec["status"] == "FROZEN"
    assert spec["pec_panel_id"] == "Fig01.grammar"
    assert len(spec["semantic_elements"]) >= 30


def test_fig01_spec_excludes_empirical_content():
    spec = load_fig01_spec()
    excluded = set(spec["excluded_content"])
    assert "E5 propagation evidence" in excluded
    assert "H4/C3/D3 negatives" in excluded


def test_validate_fig01_spec():
    validate_fig01_spec()


@pytest.mark.skipif(not FIG01_FIGURE_PATH.exists(), reason="figure not generated yet")
def test_fig01_figure_and_audit_on_disk():
    assert FIG01_AUDIT_PATH.is_file()
    assert FIG01_RECEIPT_PATH.is_file()
    audit = load_fig01_semantic_audit()
    validate_fig01_semantic_audit(audit)
    receipt = load_fig01_generation_receipt()
    validate_fig01_generation_receipt(receipt)
    assert receipt["next_checkpoint"] == "figures_2_4_generation"
    assert receipt["semantic_audit_status"] == "PASSED"


def test_fig01_generator_runs():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "publication_figures" / "fig01_grammar.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "PYTHONPATH": str(_SCRIPT_REPO_ROOT), "PYTHONIOENCODING": "utf-8"},
        cwd=root / "scripts" / "publication_figures",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert FIG01_FIGURE_PATH.is_file()
    audit = json.loads(FIG01_AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit["status"] == "PASSED"
