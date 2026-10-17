"""Docs currency: a published current page states no older minor version as current.

At version N.M.x, results and claims from N.(M-1).x and earlier are regenerated or move to the
Archive section of the nav. A currency phrase ("current", "live", "development tree", "as of",
"Status:") followed by an older N.M version on a published, non-Archive page fails. History
("introduced in 0.2.19", "since 0.4.7") passes.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
CURRENCY = re.compile(
    r"\b(?:current(?:ly)?|live|development tree|as of|status\b)([^\n]{0,40}?)\bv?(\d+)\.(\d+)\.\d+",
    re.IGNORECASE)
# Words between the currency phrase and the version that make it history.
HISTORY = re.compile(r"\b(?:since|introduced|from|added)\b", re.IGNORECASE)


def package_minor() -> tuple[int, int]:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    major, minor = re.search(r'^version = "(\d+)\.(\d+)\.\d+', text, re.MULTILINE).groups()
    return int(major), int(minor)


def _excluded_and_archived() -> set[str]:
    """Docs paths outside the current site: exclude_docs entries and pages under the Archive nav."""
    lines = (REPO / "mkdocs.yml").read_text(encoding="utf-8").splitlines()
    out: set[str] = set()
    block = None
    for line in lines:
        if block == "exclude":
            if line and not line.startswith(" "):
                block = None
            elif line.strip() and not line.strip().startswith("#"):
                out.add(line.strip())
                continue
        if block == "archive":
            indent = len(line) - len(line.lstrip())
            if line.strip() and indent <= archive_indent:
                block = None
            else:
                match = re.search(r":\s*(\S+\.md)\s*$", line)
                if match:
                    out.add(match.group(1))
                continue
        if line.startswith("exclude_docs:"):
            block = "exclude"
        elif line.strip() == "- Archive:":
            block, archive_indent = "archive", len(line) - len(line.lstrip())
    return out


def current_pages() -> list[pathlib.Path]:
    skip = _excluded_and_archived()
    pages = []
    for page in sorted(DOCS.rglob("*.md")):
        rel = page.relative_to(DOCS).as_posix()
        if rel in skip or rel.startswith(("archive/", "reference/releases/", "_static/")) or rel == "reference/changelog.md":
            continue
        pages.append(page)
    return pages


def stale_claims(text: str, current: tuple[int, int]) -> list[str]:
    return [m.group(0) for m in CURRENCY.finditer(text)
            if (int(m.group(2)), int(m.group(3))) < current and not HISTORY.search(m.group(1))]


def test_fixture_lists_split_the_docs():
    skip = _excluded_and_archived()
    assert "archive/tutorials/readouts/tutorial_outputs.md" in skip, "the Archive nav must be read"
    assert "reference/project/zenodo_doi.md" in skip, "exclude_docs must be read"
    assert "index.md" not in skip and len(current_pages()) > 50


def test_rule_flags_an_old_current_claim_and_passes_history():
    assert stale_claims("Status: v0.4.8 (development tree)", (0, 5)) == ["Status: v0.4.8"]
    assert stale_claims("current development tree: 0.4.8", (0, 5))
    assert stale_claims("introduced in jaxfne 0.2.19; since 0.4.7 the default", (0, 5)) == []
    assert stale_claims("current 0.5.0", (0, 5)) == []
    assert stale_claims("**Status.** CANONICAL (0.4.17).", (0, 5))
    assert stale_claims("**Status.** CANONICAL since 0.4.17.", (0, 5)) == []
    assert stale_claims("amplitude_status 0.4.1", (0, 5)) == []


def test_current_pages_state_no_older_minor_as_current():
    current = package_minor()
    found = []
    for page in current_pages():
        for claim in stale_claims(page.read_text(encoding="utf-8"), current):
            found.append(f"{page.relative_to(REPO).as_posix()}: {claim!r}")
    assert not found, (
        f"pages claim a version older than {current[0]}.{current[1]} as current; update them or "
        "move them to the Archive section:\n" + "\n".join(found))
