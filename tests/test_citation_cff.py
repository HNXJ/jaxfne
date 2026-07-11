"""CITATION.cff presence and minimal schema checks."""

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


def test_citation_cff_no_fabricated_doi() -> None:
    text = CFF_PATH.read_text(encoding="utf-8")
    assert "10.5281/zenodo." not in text, "do not add a placeholder DOI before Zenodo mints one"
