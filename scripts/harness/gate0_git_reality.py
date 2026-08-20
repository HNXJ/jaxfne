#!/usr/bin/env python3
"""Gate 0: Mechanical verification of Git reality & workspace identity.

Verification sequence:
1. Verify Git root, git-dir, and expected workspace directory.
2. Verify expected repository remote identity (HNXJ/jaxfne).
3. Fetch origin with explicit error handling (fails on network/auth error unless --offline).
4. Inspect local branch, local HEAD, origin/main, origin/dev.
5. Check tracking status: AHEAD, BEHIND, DIVERGED, or SYNCHRONIZED.
6. Inspect dirty-tree state (staged, unstaged, untracked).
7. Verify required authorities exist for active mode.
8. Output structured identity block and exit:
   - 0: PASS
   - 1: FAIL (STALE_LOCAL_STATE, DIVERGED, REMOTE_MISMATCH, FETCH_FAILED, DIRTY_TREE, MISSING_AUTHORITY)
   - 2: REMOTE_STATE_UNVERIFIED (when running explicitly with --offline)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REMOTES = [
    "https://github.com/hnxj/jaxfne.git",
    "https://github.com/HNXJ/jaxfne.git",
    "git@github.com:hnxj/jaxfne.git",
    "git@github.com:HNXJ/jaxfne.git",
]


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip()


def check_gate0(fetch: bool = True, offline: bool = False, strict_clean: bool = False) -> int:
    print("================================================================================")
    print("GATE 0: GIT REALITY & WORKSPACE IDENTITY")
    print("================================================================================")

    # 1. Root & Git Dir
    code, git_dir = run(["git", "rev-parse", "--git-dir"])
    if code != 0:
        print("FAIL[GATE-0]: Not inside a Git repository")
        return 1

    # 2. Remote Identity
    code, remote_url = run(["git", "remote", "get-url", "origin"])
    if code != 0:
        print("FAIL[GATE-0]: Missing 'origin' remote")
        return 1

    remote_clean = remote_url.rstrip("/")
    if not any(remote_clean == exp.rstrip("/") for exp in EXPECTED_REMOTES):
        print(f"FAIL[GATE-0]: Remote mismatch: got '{remote_url}', expected HNXJ/jaxfne")
        return 1

    # 3. Fetch origin with explicit failure capture
    fetch_failed = False
    if fetch and not offline:
        f_code, f_err = run(["git", "fetch", "--all", "--prune"])
        if f_code != 0:
            print(f"FAIL[GATE-0]: git fetch failed: {f_err}")
            fetch_failed = True

    if offline or fetch_failed:
        if not offline:
            print("GATE 0 RESULT: FETCH_FAILED (Cannot verify remote truth)")
            return 1
        print("WARNING: Running in --offline mode; remote state is unverified")

    # 4. Branch & Local HEAD
    code, branch = run(["git", "branch", "--show-current"])
    code, head = run(["git", "rev-parse", "HEAD"])
    _, origin_main = run(["git", "rev-parse", "origin/main"])
    _, origin_dev = run(["git", "rev-parse", "origin/dev"])

    # 5. Tracking / Divergence Analysis
    code_behind, behind_count = run(["git", "rev-list", f"HEAD..origin/{branch}", "--count"])
    code_ahead, ahead_count = run(["git", "rev-list", f"origin/{branch}..HEAD", "--count"])

    n_behind = int(behind_count) if code_behind == 0 else 0
    n_ahead = int(ahead_count) if code_ahead == 0 else 0

    if n_behind > 0 and n_ahead > 0:
        sync_status = f"DIVERGED (+{n_ahead}, -{n_behind})"
    elif n_behind > 0:
        sync_status = f"BEHIND (-{n_behind}) -> STALE_LOCAL_STATE"
    elif n_ahead > 0:
        sync_status = f"AHEAD (+{n_ahead})"
    else:
        sync_status = "SYNCHRONIZED (dev == origin/dev)"

    # 6. Dirty-Tree State
    _, status_short = run(["git", "status", "--porcelain"])
    dirty_lines = [line for line in status_short.splitlines() if line.strip()]
    tracked_dirty = [l for l in dirty_lines if not l.startswith("??")]
    untracked = [l for l in dirty_lines if l.startswith("??")]

    # 7. Required Authorities
    required_paths = [
        ROOT / "docs" / "doctrine" / "tfne_containment_architecture.md",
        ROOT / "artifacts" / "publication" / "publication_evidence_index.json",
        ROOT / "artifacts" / "release" / "v0_4_17_release_receipt.json",
        ROOT / "artifacts" / "issue_log" / "ISSUE_LOG.md",
        ROOT / "scratch" / "CURRENT_TASK.md",
    ]
    missing_authorities = [str(p.relative_to(ROOT)) for p in required_paths if not p.exists()]

    # Display Report
    print(f"Workspace Root:  {ROOT}")
    print(f"Remote URL:      {remote_url} (VERIFIED HNXJ/jaxfne)")
    print(f"Active Branch:   {branch} [{sync_status}]")
    print(f"Local HEAD:      {head}")
    print(f"origin/main:     {origin_main}")
    print(f"origin/dev:      {origin_dev}")
    print(f"Working Tree:    {'CLEAN' if not dirty_lines else f'{len(tracked_dirty)} tracked modified, {len(untracked)} untracked'}")
    print(f"Authorities:     {'ALL 5 PRESENT' if not missing_authorities else f'MISSING: {missing_authorities}'}")

    # Determine Gate 0 Exit
    if missing_authorities:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: FAIL (Missing required authorities: {missing_authorities})")
        print("================================================================================")
        return 1

    if n_behind > 0:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: STALE_LOCAL_STATE ({sync_status})")
        print("================================================================================")
        return 1

    if n_behind > 0 and n_ahead > 0:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: FAIL (Branch has DIVERGED from origin/{branch})")
        print("================================================================================")
        return 1

    if strict_clean and dirty_lines:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: FAIL (Dirty working tree not permitted under strict_clean)")
        print("================================================================================")
        return 1

    if offline:
        print("--------------------------------------------------------------------------------")
        print("GATE 0 RESULT: REMOTE_STATE_UNVERIFIED")
        print("================================================================================")
        return 2

    print("--------------------------------------------------------------------------------")
    print("GATE 0 RESULT: PASS")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 0 Git reality check")
    parser.add_argument("--no-fetch", action="store_true", help="Skip remote fetch")
    parser.add_argument("--offline", action="store_true", help="Explicit offline mode")
    parser.add_argument("--strict-clean", action="store_true", help="Require pristine git status")
    args = parser.parse_args()

    sys.exit(check_gate0(fetch=not args.no_fetch, offline=args.offline, strict_clean=args.strict_clean))
