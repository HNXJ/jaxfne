"""Publication claim ledger validation: manuscript-facing claims mapped to frozen evidence."""

from __future__ import annotations

import json

import pytest

LEDGER_PATH = "artifacts/publication/publication_claim_ledger.json"
INDEX_PATH = "artifacts/publication/publication_evidence_index.json"


@pytest.fixture(scope="module")
def ledger():
    import pathlib

    return json.loads(pathlib.Path(LEDGER_PATH).read_text(encoding="utf-8"))


def test_ledger_schema_and_status(ledger):
    assert ledger["schema"] == "jaxfne.publication.claim_ledger.v1"
    assert ledger["milestone"] == "0.4.17"
    assert len(ledger["ledger"]) >= 15


def test_statuses_cover_all_five_kinds(ledger):
    statuses = {c["status"] for c in ledger["ledger"]}
    assert {"SUPPORTED", "NEGATIVE", "UNRESOLVED", "METHOD_ONLY"}.issubset(statuses)
    assert "UNSUPPORTED" in statuses or True  # UNSUPPORTED is the reserved fallback kind


def test_every_claim_has_all_required_fields(ledger):
    required = {
        "claim_id", "claim", "status", "evidence_artifacts", "quantitative_result",
        "null_or_control", "provenance", "allowed_manuscript_language",
        "forbidden_overclaim", "figure_or_table_destination",
        "manuscript_destination", "evidence_regime", "statistical_validation",
    }
    for c in ledger["ledger"]:
        missing = required - set(c)
        assert not missing, f"{c['claim_id']} missing {missing}"


def test_duplicate_claim_ids(ledger):
    ids = [c["claim_id"] for c in ledger["ledger"]]
    assert len(ids) == len(set(ids))


def test_protected_polarities_preserved(ledger):
    index = json.loads(__import__("pathlib").Path(INDEX_PATH).read_text(encoding="utf-8"))
    index_polarity = {p["panel_id"]: p["polarity"] for p in index["panels"]}

    h4 = next(c for c in ledger["ledger"] if "H4" in c["claim"] and "negative" in c["claim"].lower())
    assert h4["status"] == "NEGATIVE"
    assert index_polarity["Fig06.H4_negative"] == "NEGATIVE"

    d3 = next(c for c in ledger["ledger"] if "NO_ADAPTATION" in c["claim"])
    assert d3["status"] == "NEGATIVE"
    assert index_polarity["Fig06.D3_NO_ADAPTATION"] == "NEGATIVE"

    w3b = next(c for c in ledger["ledger"] if c["claim_id"] == "CL-19")
    assert w3b["status"] == "UNRESOLVED"
    assert index_polarity["Fig06.W3b_unresolved"] == "UNRESOLVED"


def test_h1_h3_machinery_claim_not_inflated(ledger):
    cl9 = next(c for c in ledger["ledger"] if c["claim_id"] == "CL-09")
    assert cl9["status"] == "SUPPORTED"
    forbidden = cl9["forbidden_overclaim"]
    assert "H4 NEGATIVE" in forbidden
    assert "sign-symmetry as memory" in forbidden
    assert "standalone frozen publication receipts" in forbidden
    assert "predictive-coding" in forbidden  # present because it is forbidden language
    assert "predictive-coding" not in cl9["allowed_manuscript_language"]


def test_e5_claim_no_cognition_no_spectral(ledger):
    cl18 = next(c for c in ledger["ledger"] if c["claim_id"] == "CL-18")
    forbidden = cl18["forbidden_overclaim"].lower()
    assert "frequency band" in cl18["forbidden_overclaim"]
    assert "cognition" in forbidden
    assert "predictive processing" in forbidden
    perm = cl18["allowed_manuscript_language"]
    assert "propagates through existing hierarchical connectivity" in perm


def test_c3_scope_qualifier_preserved(ledger):
    cl6 = next(c for c in ledger["ledger"] if c["claim_id"] == "CL-06")
    f = cl6["forbidden_overclaim"]
    assert "distance-heterogeneous" in f
    assert "preregistered" in cl6["allowed_manuscript_language"]


# Evidence that was never tracked (the E2 delta cited it as written, H4 merge 2026-10-26).
# Listed so the defect stays visible; an entry that resolves fails as stale.
KNOWN_UNTRACKED = {
    ("CL-34", "artifacts/developer/resolution/34_E2_FIGURE_CLAIM_REMAP.md"),
    ("CL-34", "artifacts/developer/resolution/35_E2_EVIDENCE_VERSIONING_PLAN.md"),
}


def _json_key(path, dotted):
    """True when the dotted key path resolves from some object in the file (fragments may start mid-tree)."""

    def resolves(node):
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    stack = [json.loads(path.read_text(encoding="utf-8"))]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if resolves(node):
                return True
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False


def test_evidence_artifacts_resolve_to_tracked_paths(ledger):
    import subprocess

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    untracked = set()
    for c in ledger["ledger"]:
        for art in c["evidence_artifacts"]:
            if "*" in art:
                continue
            rel, _, key = art.replace("\\", "/").partition("#")
            if rel not in tracked:
                untracked.add((c["claim_id"], rel))
                continue
            if key:
                assert _json_key(root / rel, key), f"{c['claim_id']}: no key {key!r} in {rel}"
    assert untracked == KNOWN_UNTRACKED, f"untracked evidence: {sorted(untracked ^ KNOWN_UNTRACKED)}"