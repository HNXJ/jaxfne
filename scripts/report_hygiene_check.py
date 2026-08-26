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
    # Governance/context sources intentionally name mirror directories and
    # audit exclusions; scripts/audit_agent_context.py checks those surfaces.
    Path("scripts/audit_agent_context.py"),
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

def get_staged_files() -> list[Path]:
    """Get list of git staged files."""
    try:
        res = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT)
        )
        return [REPO_ROOT / line for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        return []

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
                        f"{path.relative_to(REPO_ROOT)}:{line_no} - Found {desc}: {line.strip()[:100]}"
                    )
    except Exception as e:
        violations.append(f"{path.relative_to(REPO_ROOT)} - Error reading: {e}")
    return violations

def main() -> int:
    # 1. Gather all files in scan areas: docs/, outputs/, artifacts/tutorials/, examples/
    scan_paths: set[Path] = set()
    scan_dirs = ["docs", "outputs", "artifacts/tutorials", "examples"]
    for dname in scan_dirs:
        dir_path = REPO_ROOT / dname
        if dir_path.is_dir():
            for p in dir_path.rglob("*"):
                if p.is_file():
                    scan_paths.add(p)

    # 2. Gather staged files
    for staged in get_staged_files():
        scan_paths.add(staged)

    # Filter out exclusions
    target_files = [p for p in scan_paths if not is_excluded(p)]
    
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
