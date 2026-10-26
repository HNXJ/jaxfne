#!/usr/bin/env python3
"""Harness v2.1 reference checker.

Every repo-relative path referenced in agent-facing markdown (artifacts/AGENTS.md,
docs/**, artifacts/control/skills/**) must be a file git keeps (tracked, or new and not
ignored) or a folder holding one, OR be explicitly marked optional via an inline
`<!-- optional -->` marker in the same paragraph. Gitignored files on disk never count, so
a local run sees what a fresh clone sees.

Exit codes: 0 clean, 1 broken (unmarked) references found.
"""
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
GOVERNED = [
    ROOT / "artifacts" / "context.md",
    ROOT / "artifacts" / "AGENTS.md",
    ROOT / "docs/reference/project/for_ai_agents.md",
    *list((ROOT / "artifacts" / "control" / "skills").rglob("*.md")),
]
PREFIXES = ("artifacts/control/skills/", "scripts/", "docs/", "artifacts/", "scratch/", ".opencode/", ".cursor/", "tests/", "jaxfne/")
BACKTICK = re.compile(r"`([A-Za-z0-9_.\-/]+(?:/[A-Za-z0-9_.\-]+)*)`")
LINK = re.compile(r"\]\(([^)#]+)\)")


def paragraph_has_optional(text: str, pos: int) -> bool:
    start = max(text.rfind("\n\n", 0, pos), text.rfind("\n#", 0, pos))
    end = text.find("\n\n", pos)
    if end == -1:
        end = len(text)
    para = text[max(start, 0) : end]
    return "<!-- optional" in para


def kept_paths(root: Path) -> set[str]:
    """Files git keeps (tracked or new and unignored) and every folder above them."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True, check=True,
    ).stdout.decode("utf-8")
    kept = set()
    for name in filter(None, out.split("\0")):
        path = PurePosixPath(name)
        kept.add(str(path))
        kept.update(str(parent) for parent in path.parents if str(parent) != ".")
    return kept


def run(scan, root: Path = ROOT) -> int:
    kept = kept_paths(root)
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
            if ref in kept:
                continue
            if paragraph_has_optional(text, m.start()):
                continue
            try:
                rel = f.relative_to(root)
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
        scan = [ROOT / "artifacts" / "AGENTS.md", *list((ROOT / "docs").rglob("*.md")),
                *list((ROOT / "artifacts" / "control" / "skills").rglob("*.md"))]
    else:
        scan = GOVERNED
    return run(scan)


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())