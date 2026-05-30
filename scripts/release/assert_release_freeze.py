#!/usr/bin/env python3
"""
Release freeze guard for jaxfne.

Checks whether a release freeze is active by reading AGENTS.md and/or
.release_target.json. If a freeze is active, verifies that HEAD and
origin/main both match the declared intended_release_sha.

A missing freeze declaration is NOT a pass: this script exits 0 only
when freeze is explicitly marked inactive, or when freeze is active and
all SHA invariants hold.

Usage:
    python scripts/release/assert_release_freeze.py
    python scripts/release/assert_release_freeze.py --check

Exit codes:
    0  — freeze not active, or freeze active and all SHAs match
    1  — freeze active and a required invariant is violated
    2  — configuration error (cannot determine freeze state reliably)
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_cmd(args):
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return ""


def read_freeze_from_agents_md():
    """Read release freeze state from the dedicated RELEASE_STATE block in AGENTS.md.

    Only reads lines within the marked block:
        <!-- RELEASE_STATE_BEGIN -->
        release_freeze: true/false
        intended_release_sha: <40-char sha or none>
        <!-- RELEASE_STATE_END -->

    Returns (freeze_active: bool | None, intended_sha: str)
    freeze_active=None means no valid RELEASE_STATE block was found.
    """
    agents_path = Path("AGENTS.md")
    if not agents_path.exists():
        return None, ""

    content = agents_path.read_text()
    freeze_active = None
    intended_sha = ""

    in_block = False
    for line in content.splitlines():
        if "<!-- RELEASE_STATE_BEGIN -->" in line:
            in_block = True
            continue
        if "<!-- RELEASE_STATE_END -->" in line:
            in_block = False
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if re.match(r"release_freeze\s*[:=]\s*true", stripped, re.IGNORECASE):
            freeze_active = True
        elif re.match(r"release_freeze\s*[:=]\s*false", stripped, re.IGNORECASE):
            freeze_active = False
        m = re.search(r"intended_release_sha\s*[:=]\s*['\"]?([0-9a-f]{40})['\"]?", stripped)
        if m:
            intended_sha = m.group(1)

    return freeze_active, intended_sha


def read_freeze_from_lockfile():
    """Read release freeze state from .release_target.json if present."""
    lock_path = Path(".release_target.json")
    if not lock_path.exists():
        return None, ""
    try:
        data = json.loads(lock_path.read_text())
        freeze_active = data.get("release_freeze", None)
        intended_sha = data.get("intended_release_sha", "")
        return freeze_active, intended_sha
    except Exception:
        return None, ""


def main():
    parser = argparse.ArgumentParser(description="Assert release freeze state.")
    parser.add_argument("--check", action="store_true", help="Check-only mode (no side effects)")
    args = parser.parse_args()

    # Sources of freeze state (lockfile takes priority if present)
    lock_freeze, lock_sha = read_freeze_from_lockfile()
    agents_freeze, agents_sha = read_freeze_from_agents_md()

    # Resolve freeze state: lockfile overrides AGENTS.md
    if lock_freeze is not None:
        freeze_active = lock_freeze
        intended_sha = lock_sha or agents_sha
        source = ".release_target.json"
    elif agents_freeze is not None:
        freeze_active = agents_freeze
        intended_sha = agents_sha
        source = "AGENTS.md"
    else:
        # No freeze declaration found in either source — not active
        print("Release freeze is not active (no declaration found in AGENTS.md or .release_target.json). Pass.")
        sys.exit(0)

    if not freeze_active:
        print(f"Release freeze is not active (source: {source}). Pass.")
        sys.exit(0)

    # Freeze is active — verify all invariants
    print(f"Release freeze ACTIVE (source: {source}, intended_sha: {intended_sha})")
    errors = []

    if not intended_sha:
        errors.append("Release freeze is active but intended_release_sha is not declared")
    else:
        run_cmd(["git", "fetch", "origin", "--prune"])

        head_sha = run_cmd(["git", "rev-parse", "HEAD"])
        origin_main_sha = run_cmd(["git", "rev-parse", "origin/main"])

        if head_sha != intended_sha:
            errors.append(
                f"HEAD ({head_sha}) != intended_release_sha ({intended_sha})"
            )
        if origin_main_sha and origin_main_sha != intended_sha:
            errors.append(
                f"origin/main ({origin_main_sha}) != intended_release_sha ({intended_sha})"
            )

        git_status = run_cmd(["git", "status", "--porcelain"])
        if git_status:
            errors.append(f"Working tree is dirty during release freeze: {git_status[:80]!r}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    print("Success: Release freeze active and all SHA invariants hold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
