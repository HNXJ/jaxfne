#!/usr/bin/env python3
"""Harness v2.1 reference checker.

Every repo-relative path referenced in agent-facing markdown (AGENTS.md,
docs/**, skills/**) must exist on disk OR be explicitly marked optional
via an inline `<!-- optional -->` marker in the same paragraph.

Exit codes: 0 clean, 1 broken (unmarked) references found.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOVERNED = [ROOT / "AGENTS.md", ROOT / "docs/for_ai_agents.md",
            *list((ROOT / "skills").rglob("*.md"))]
PREFIXES = ("skills/", "scripts/", "docs/", "artifacts/", "scratch/", ".opencode/", ".cursor/", "tests/", "jaxfne/")
BACKTICK = re.compile(r"`([A-Za-z0-9_.\-/]+(?:/[A-Za-z0-9_.\-]+)*)`")
LINK = re.compile(r"\]\(([^)#]+)\)")


def paragraph_has_optional(text: str, pos: int) -> bool:
    start = max(text.rfind("\n\n", 0, pos), text.rfind("\n#", 0, pos))
    end = text.find("\n\n", pos)
    if end == -1:
        end = len(text)
    para = text[max(start, 0) : end]
    return "<!-- optional" in para


def run(scan) -> int:
    failures = []
    for f in scan:
        if not f.is_file():
            continue
        text = f.read_text(errors="replace")
        for m in list(BACKTICK.finditer(text)) + list(LINK.finditer(text)):
            ref = m.group(1)
            if ref.startswith("#") or ref.startswith("http") or ref.startswith("mailto"):
                continue
            if not ref.startswith(PREFIXES):
                continue
            ref = ref.split("#")[0].rstrip("/")
            if not ref:
                continue
            target = ROOT / ref
            if target.exists():
                continue
            if paragraph_has_optional(text, m.start()):
                continue
            try:
                rel = f.relative_to(ROOT)
            except ValueError:
                rel = Path(f.name)
            failures.append(f"{rel}:{text[: m.start()].count(chr(10)) + 1} -> {ref}")
    if failures:
        for x in failures:
            print("FAIL:", x)
        return 1
    print(f"references OK: all {len(scan)} scanned files resolve or are marked optional")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="scan every docs/*.md (may report pre-existing historical debt)")
    a = ap.parse_args()
    if a.all:
        scan = [ROOT / "AGENTS.md", *list((ROOT / "docs").rglob("*.md")),
                *list((ROOT / "skills").rglob("*.md"))]
    else:
        scan = GOVERNED
    return run(scan)


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())