"""Guard for the curated agent API catalog.

`internal_docs/JAXFNE_AGENT_API_CATALOG.md` is a curated lookup table that exists
to stop agents from rediscovering / hand-rolling package-native jaxfne functions
(especially the spectrolaminar pipeline). These checks keep it small, honest,
and free of the wording pitfalls it is meant to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "internal_docs" / "JAXFNE_AGENT_API_CATALOG.md"

SPECTRO_REQUIRED = [
    "make_laminar_column_config",
    "build_laminar_column",
    "simulate_laminar_trials",
    "spectrolaminar_from_trials",
    "summarize_spectrolaminar_similarity",
    "spectrolaminar_suite_3panel",
]

FORBIDDEN_LIKE = ["lfp_like", "csd_like", "eeg_like", "meg_like"]


@pytest.fixture(scope="module")
def text() -> str:
    # internal_docs/ is gitignored (internal-only artifact), so the catalog is
    # absent in clean checkouts / CI. Skip rather than fail when it is not present.
    if not CATALOG.exists():
        pytest.skip(f"internal catalog not present (gitignored): {CATALOG}")
    return CATALOG.read_text()


def _api_rows(text: str) -> list[str]:
    # An API row is a markdown table row whose first cell is a `code` token.
    return [ln for ln in text.splitlines() if re.match(r"^\|\s*`", ln)]


def test_catalog_exists(text):
    assert text.strip()


def test_at_most_100_api_rows(text):
    rows = _api_rows(text)
    assert 0 < len(rows) <= 100, f"expected 1..100 API rows, got {len(rows)}"


def test_contains_spectrolaminar_pipeline(text):
    for fn in SPECTRO_REQUIRED:
        assert fn in text, f"spectrolaminar pipeline fn {fn!r} missing from catalog"


def test_canonical_import_present(text):
    assert "import jaxfne as jtfne" in text


def test_no_like_labels(text):
    lowered = text.lower()
    hits = [tok for tok in FORBIDDEN_LIKE if tok in lowered]
    assert not hits, f"forbidden *_like labels present: {hits}"


def test_no_hardcoded_local_path(text):
    for needle in ("/Users/hamednejat", "/workspace/main"):
        assert needle not in text, f"hardcoded local path {needle!r} present in catalog"
