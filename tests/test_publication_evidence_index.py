"""Publication evidence consolidation tests."""

from __future__ import annotations

import json

import pytest

from jaxfne.publication.pec_protocol import (
    PEC_CONSOLIDATION_RECEIPT_PATH,
    PEC_SPEC_PATH,
    PUBLICATION_EVIDENCE_INDEX_PATH,
    load_pec_spec,
    load_publication_evidence_index,
    panel_ids,
    validate_pec_spec,
    validate_publication_evidence_index,
    write_pec_consolidation_receipt,
)


def test_pec_spec_frozen():
    spec = load_pec_spec()
    assert spec["status"] == "FROZEN"
    assert spec["checkpoint"] == "PEC"
    assert spec["execution_authorization"]["figure_rendering_authorized"] is False


def test_evidence_index_covers_figures_and_e5():
    index = load_publication_evidence_index()
    figures = {p["figure"] for p in index["panels"]}
    assert {1, 2, 3, 4, 5, 6, 7}.issubset(figures)
    e5 = [p for p in index["panels"] if p["panel_id"].startswith("Fig07.E5")]
    assert len(e5) >= 2
    assert any(p["polarity"] == "POSITIVE" and p["claim_level"] == "DEMONSTRATED" for p in e5)


def test_negative_and_unresolved_panels_present():
    index = load_publication_evidence_index()
    pol = {p["polarity"] for p in index["panels"]}
    assert "NEGATIVE" in pol
    assert "UNRESOLVED" in pol
    h4 = next(p for p in index["panels"] if p["panel_id"] == "Fig06.H4_negative")
    assert h4["claim_level"] == "DEMONSTRATED" and h4["polarity"] == "NEGATIVE"


def test_validate_index_receipt_paths_exist():
    validate_publication_evidence_index()


def test_validate_pec_spec():
    validate_pec_spec()


def test_write_consolidation_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jaxfne.publication.pec_protocol.PEC_CONSOLIDATION_RECEIPT_PATH",
        tmp_path / "pec_consolidation_receipt.json",
    )
    receipt = write_pec_consolidation_receipt(artifact_commit_sha="abc123")
    assert receipt["panel_count"] == len(panel_ids())
    assert receipt["figure_rendering_authorized"] is False


@pytest.mark.skipif(
    not PEC_CONSOLIDATION_RECEIPT_PATH.exists(),
    reason="frozen PEC receipt not present",
)
def test_frozen_pec_receipt_on_disk():
    receipt = json.loads(PEC_CONSOLIDATION_RECEIPT_PATH.read_text())
    assert receipt["status"] == "FROZEN"
    assert receipt["next_checkpoint"] == "figure_1_generation"
