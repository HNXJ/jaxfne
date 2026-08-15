#!/usr/bin/env python3
"""Harness v2.1 memory hygiene checker (semantic).

Flags TRANSIENT project state in durable memory files (global MEMORY.md and
any file passed on argv): absolute SHAs, "HEAD is ...", "currently ...",
"next checkpoint is ...", "release candidate ...", "<N> tests passing",
relative time anchors. Durable preferences (e.g. "development work normally
happens on dev", version-lineage statements like "since 0.4.17") are allowed.

Exit codes: 0 clean, 1 transient state found.
"""
import re
import sys
from pathlib import Path

HOME_MEMORY = Path.home() / ".config" / "opencode" / "MEMORY.md"
TRANSIENT = [
    (re.compile(r"\bHEAD is [0-9a-f]{7,}\b", re.I), "HEAD snapshot"),
    (re.compile(r"\b[0-9a-f]{40}\b"), "absolute sha"),
    (re.compile(r"\b(next checkpoint|current checkpoint) is", re.I), "checkpoint pointer"),
    (re.compile(r"\brelease candidate (is|:)", re.I), "release transient"),
    (re.compile(r"\bcaptured? (on|at) \d{4}-\d{2}-\d{2}", re.I), "pinned date"),
    (re.compile(r"\b\d+ tests? (passing|failing)\b", re.I), "test status"),
    (re.compile(r"\byesterday\b|\btoday[, ]|last week\b", re.I), "relative time anchor"),
    (re.compile(r"\bcurrently (open|active|pending| in progress)\b", re.I), "living state"),
]


def check(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    hits = []
    for rx, label in TRANSIENT:
        for m in rx.finditer(text):
            line = text[: m.start()].count("\n") + 1
            hits.append(f"{path}:{line}: transient ({label}): {m.group(0)[:60]!r}")
    return hits


def main() -> int:
    files = [HOME_MEMORY] + [Path(a) for a in sys.argv[1:]]
    hits = []
    for f in files:
        hits += check(f)
    if hits:
        for h in hits:
            print("FAIL:", h)
        return 1
    print(f"memory hygiene OK: {len(files)} file(s) contain only durable, project-neutral content")
    return 0


if __name__ == "__main__":
    sys.exit(main())