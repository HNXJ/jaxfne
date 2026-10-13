#!/usr/bin/env python3
"""Tracked-tree size gate: fail CI when the tracked tree grows back.

Sizes come from git blobs, not the working tree (CRLF on Windows would
differ): ``git ls-files -s`` -> blob ids -> ``git cat-file --batch-check``
object sizes. Index state, so a staged new file counts.

Usage:
    python scripts/check_tree_size.py           # report
    python scripts/check_tree_size.py --check   # exit 1 on violation
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = Path(__file__).resolve().parent / "tree_size_allowlist.json"

GROWTH_HEADROOM = 1.10


def load_allowlist() -> dict:
    return json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def tracked_sizes(repo_root: Path = ROOT) -> dict[str, int]:
    """Return {posix path: blob size in bytes} for every tracked file."""
    ls = subprocess.run(
        ["git", "ls-files", "-s", "-z"],  # -z: paths unquoted, as stored
        capture_output=True,
        encoding="utf-8",
        cwd=str(repo_root),
        check=True,
    )
    blobs: list[str] = []
    paths: list[str] = []
    for entry in ls.stdout.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        blob = meta.split()[1]
        blobs.append(blob)
        paths.append(path)
    if not blobs:
        return {}
    batch = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        input="\n".join(blobs) + "\n",
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        check=True,
    )
    sizes: dict[str, int] = {}
    for path, line in zip(paths, batch.stdout.splitlines()):
        sizes[path] = int(line.split()[2])
    return sizes


def check_tree(
    repo_root: Path = ROOT, allowlist: dict | None = None
) -> tuple[list[str], dict]:
    """Apply rules A/B/C; return (violations, info)."""
    allow = allowlist if allowlist is not None else load_allowlist()
    cap = int(allow["cap_bytes"])
    budget = int(allow["budget_bytes"])
    listed: dict[str, int] = dict(allow.get("files", {}))
    sizes = tracked_sizes(repo_root)
    total = sum(sizes.values())
    over_cap = sum(1 for s in sizes.values() if s > cap)
    violations: list[str] = []
    for path in sorted(sizes):
        size = sizes[path]
        if size > cap and path not in listed:
            violations.append(f"over-cap-unlisted: {path} bytes={size} cap={cap}")
        elif path in listed and size > listed[path] * GROWTH_HEADROOM:
            violations.append(
                f"over-cap-grown: {path} bytes={size} recorded={listed[path]}"
            )
    for path in sorted(listed):
        if path not in sizes:
            violations.append(f"stale-allowlist-entry: {path}")
    if total > budget:
        violations.append(f"over-budget: total={total} budget={budget}")
    budget_files = allow.get("budget_files")
    if budget_files is not None and len(sizes) > int(budget_files):
        violations.append(f"over-file-budget: files={len(sizes)} budget_files={budget_files}")
    info = {
        "files": len(sizes),
        "total": total,
        "budget": budget,
        "over_cap": over_cap,
        "violations": len(violations),
    }
    return violations, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    violations, info = check_tree()
    print(
        f"tree-size: files={info['files']} total={info['total']}"
        f" budget={info['budget']} over_cap={info['over_cap']}"
        f" violations={info['violations']}"
    )
    for v in violations:
        print(v)
    if args.check and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
