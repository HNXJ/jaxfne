#!/usr/bin/env python3
"""Report hygiene check tool.

Scans staged and generated files to ensure no AI agent, model names,
local scratch paths, or raw GOAL_COMPLETE tokens leak into public repo
surfaces or generated reports.
"""

from __future__ import annotations

import re
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Forbidden patterns
FORBIDDEN = [
    (re.compile(r"antigravity", re.IGNORECASE), "antigravity (agent name)"),
    (re.compile(r"gemini", re.IGNORECASE), "gemini (model name)"),
    (re.compile(r"claude", re.IGNORECASE), "claude (model/agent name)"),
    (re.compile(r"GOAL_COMPLETE"), "GOAL_COMPLETE (completion token)"),
    (re.compile(r"\bAgent:\s"), "Agent: (agent identifier)"),
]

# Excluded path prefixes (relative to REPO_ROOT, POSIX)
EXCLUDED_PREFIXES = (
    "artifacts/legacy",
    "internal_docs",
    # 2026-07-14: untracked (see .gitignore) but may still be present locally
    # on a machine that had them before the untracking -- PRP/agent working
    # notes, not doc content this check is meant to police.
    "artifacts/developer",
)

# Excluded files
EXCLUDED_FILES = {
    Path("scripts/report_hygiene_check.py"),
    # Governance/context sources intentionally name mirror directories.
    # Material context invariants (grammars, archival marking, source set,
    # router linkage) are enforced by tests/test_agent_context_hygiene.py;
    # the standalone scripts/audit_agent_context.py audit was removed
    # 2026-09-08 as stale/redundant (see repair note).
    Path("tests/test_agent_context_hygiene.py"),
    Path("pyproject.toml"), # references /.claude
}

# Allowed suffixes to scan (avoids scanning large generated HTML/JS visualizations)
ALLOWED_SUFFIXES = {
    ".md",
    ".py",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
    ".ipynb",
}

def get_staged_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Get list of git staged files."""
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root)
        )
        return [repo_root / line for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def staged_renames(repo_root: Path = REPO_ROOT) -> list[tuple[str, str]]:
    """Staged renames as (old, new) repo-relative paths (rename detection on)."""
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-status", "-M"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root)
        )
    except Exception:
        return []
    renames = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].startswith("R"):
            renames.append((parts[1], parts[2]))
    return renames


def _blob_id(repo_root: Path, rev: str) -> str | None:
    """Blob id for a revision path spec, or None when it does not resolve."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--verify", rev],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root)
        )
    except Exception:
        return None
    return res.stdout.strip()


def _blob_bytes(repo_root: Path, rev: str) -> bytes | None:
    """Blob content for a revision path spec, or None when it does not resolve."""
    try:
        res = subprocess.run(
            ["git", "show", rev],
            capture_output=True,
            check=True,
            cwd=str(repo_root)
        )
    except Exception:
        return None
    return res.stdout


def pure_rename_targets(repo_root: Path = REPO_ROOT) -> set[Path]:
    """Staged renames whose bytes are unchanged everywhere.

    A pure rename (``git mv``, R100) carries the HEAD blob under the old name
    into the index under the new name, so its staged blob equals its HEAD
    blob. A rename with staged edits has different blobs and is NOT skipped.
    The worktree must also match: ``git mv a b`` followed by an unstaged edit
    to ``b`` leaves the staged blob equal to the HEAD blob while the file on
    disk carries new bytes, so it stays in the scan set. Unresolvable blobs
    fail closed toward scanning (never skipped).
    """
    skips: set[Path] = set()
    for old, new in staged_renames(repo_root):
        head_blob = _blob_id(repo_root, f"HEAD:{old}")
        staged_blob = _blob_id(repo_root, f":{new}")
        if head_blob is None or head_blob != staged_blob:
            continue
        try:
            worktree_bytes = (repo_root / new).read_bytes()
        except OSError:
            continue
        if _blob_bytes(repo_root, f"HEAD:{old}") == worktree_bytes:
            skips.add(repo_root / new)
    return skips


def collect_scan_paths(repo_root: Path = REPO_ROOT) -> set[Path]:
    """All scan candidates minus unchanged pure renames."""
    scan_paths: set[Path] = set()
    scan_dirs = ["docs", "outputs", "artifacts/tutorials", "examples"]
    for dname in scan_dirs:
        dir_path = repo_root / dname
        if dir_path.is_dir():
            for p in dir_path.rglob("*"):
                if p.is_file():
                    scan_paths.add(p)

    # Gather staged files
    for staged in get_staged_files(repo_root):
        scan_paths.add(staged)

    return scan_paths - pure_rename_targets(repo_root)

def is_excluded(path: Path) -> bool:
    """Check if the given path is excluded from hygiene scanning."""
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return True
    
    rel_posix = rel.as_posix()
    for prefix in EXCLUDED_PREFIXES:
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return True

    # Skip VCS / cache dirs by single segment
    for part in rel.parts[:-1]:
        if part in {".git", ".venv", ".pytest_cache", ".ruff_cache", "site"}:
            return True
            
    if rel in EXCLUDED_FILES:
        return True
        
    return False

def _display_path(path: Path) -> str:
    """Report path relative to the repo, or absolute when outside it."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def scan_file(path: Path) -> list[str]:
    """Scan a single file for forbidden patterns, returning list of violations."""
    if not path.is_file():
        return []
    
    if path.suffix not in ALLOWED_SUFFIXES:
        return []
        
    violations = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, desc in FORBIDDEN:
            # Search line by line for precise reporting
            for line_no, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    # Do not report lines that are clearly testing or referencing exceptions
                    if "EXCLUDED_FILES" in line or "FORBIDDEN" in line:
                        continue
                    violations.append(
                        f"{_display_path(path)}:{line_no} - Found {desc}: {line.strip()[:100]}"
                    )
    except Exception as e:
        violations.append(f"{_display_path(path)} - Error reading: {e}")
    return violations

def main() -> int:
    target_files = [
        p for p in collect_scan_paths(REPO_ROOT) if not is_excluded(p)
    ]
    
    # Scan all target files
    all_violations = []
    for path in sorted(target_files):
        violations = scan_file(path)
        all_violations.extend(violations)

    if all_violations:
        print("=== REPORT HYGIENE VIOLATIONS DETECTED ===", file=sys.stderr)
        for v in all_violations:
            print(v, file=sys.stderr)
        print("==========================================", file=sys.stderr)
        return 1
    
    print("Report hygiene check: CLEAN")
    return 0

if __name__ == "__main__":
    sys.exit(main())
