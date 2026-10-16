"""CITATION.cff presence and minimal schema checks."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CFF_PATH = REPO_ROOT / "CITATION.cff"


@pytest.mark.parametrize(
    "needle",
    [
        "cff-version: 1.2.0",
        "title:",
        "repository-code:",
        "https://github.com/HNXJ/jaxfne",
        "family-names: Nejat",
        "type: software",
    ],
)
def test_citation_cff_required_fields(needle: str) -> None:
    text = CFF_PATH.read_text(encoding="utf-8")
    assert needle in text, f"missing in CITATION.cff: {needle!r}"


# Concept DOI minted by Zenodo for v0.4.7 and shared by every later version (zenodo.org API, 2026-10-16).
CONCEPT_DOI = "10.5281/zenodo.21448155"


def test_citation_cff_cites_only_the_concept_doi() -> None:
    text = CFF_PATH.read_text(encoding="utf-8")
    dois = set(re.findall(r"10\.5281/zenodo\.\w+", text))
    assert dois == {CONCEPT_DOI}, f"CITATION.cff must carry the concept DOI and no other Zenodo DOI: {dois}"


def test_citation_docs_have_no_placeholder_doi() -> None:
    for rel in ("docs/citation.md", "README.md"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "zenodo.XXXXXXX" not in text, f"placeholder DOI in {rel}"
        assert CONCEPT_DOI in text, f"concept DOI missing from {rel}"
