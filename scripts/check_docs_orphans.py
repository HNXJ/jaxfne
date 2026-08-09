#!/usr/bin/env python3
"""Fail if any docs/*.md file exists on disk but isn't reachable from
mkdocs.yml's nav tree.

mkdocs itself only flags an orphan page as a cosmetic INFO note -- confirmed
2026-07-14: `mkdocs build --strict` does not fail on it, only on broken
links/config errors (see mkdocs.yml's own exclude_docs comment). This script
exists specifically to close that gap and give CI a real, hard-failing gate
for the class of regression fixed 2026-07-08 (16 pages built and orphaned
from nav, only caught by manual inspection at the time).

`exclude_docs` entries in mkdocs.yml (e.g. docs/_generated/version.md, a
machine-written file never meant to be navigable) are honored the same way
mkdocs itself honors them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"


def _collect_nav_paths(nav: object, out: set[str]) -> None:
    """Recursively collect every file path referenced anywhere in mkdocs.yml's
    nav tree (a nested list of dicts/strings/lists)."""
    if isinstance(nav, str):
        out.add(nav)
    elif isinstance(nav, dict):
        for value in nav.values():
            _collect_nav_paths(value, out)
    elif isinstance(nav, list):
        for item in nav:
            _collect_nav_paths(item, out)


def find_orphans() -> list[str]:
    config = yaml.safe_load(MKDOCS_YML.read_text())
    nav_paths: set[str] = set()
    _collect_nav_paths(config.get("nav", []), nav_paths)

    exclude_raw = config.get("exclude_docs", "") or ""
    excluded = {line.strip() for line in exclude_raw.splitlines() if line.strip()}

    all_md = {
        p.relative_to(DOCS_DIR).as_posix()
        for p in DOCS_DIR.rglob("*.md")
    }

    orphans = sorted(all_md - nav_paths - excluded)
    return orphans


def main() -> int:
    orphans = find_orphans()
    if orphans:
        print("Orphan doc pages found (exist on disk, not reachable from mkdocs.yml's nav,")
        print("not in exclude_docs):")
        for path in orphans:
            print(f"  - docs/{path}")
        print(
            "\nEither wire each page into mkdocs.yml's nav tree, or add it to "
            "exclude_docs with a comment explaining why it's intentionally "
            "non-navigable (see the existing _generated/version.md entry)."
        )
        return 1
    print("No orphan doc pages found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
