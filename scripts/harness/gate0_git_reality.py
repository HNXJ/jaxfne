#!/usr/bin/env python3
"""Gate 0: Mechanical verification of Git reality & workspace identity.

Verification sequence:
1. Verify Git root, git-dir, and expected workspace directory.
2. Verify expected repository remote identity (HNXJ/jaxfne).
3. Fetch origin with explicit error handling (fails on network/auth error unless --offline).
4. Inspect local branch, local HEAD, origin/main, origin/dev.
5. Check tracking status: AHEAD, BEHIND, DIVERGED, or SYNCHRONIZED.
6. Inspect dirty-tree state (staged, unstaged, untracked).
7. Verify required authorities exist for the active mode.
8. Output structured identity block and return:
   - 0: PASS
   - 1: FAIL (STALE_LOCAL_STATE, DIVERGED, REMOTE_MISMATCH, FETCH_FAILED, DIRTY_TREE, MISSING_AUTHORITY)
   - 2: REMOTE_STATE_UNVERIFIED (when running explicitly with --offline)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REMOTES = [
    "https://github.com/hnxj/jaxfne.git",
    "https://github.com/HNXJ/jaxfne.git",
    "git@github.com:hnxj/jaxfne.git",
    "git@github.com:HNXJ/jaxfne.git",
]

MODE_AUTHORITIES = {
    "RELEASE": [
        "artifacts/release/v0_4_17_release_receipt.json",
        "artifacts/issue_log/ISSUE_LOG.md",
        "scratch/CURRENT_TASK.md",
    ],
    "RELEASE_PREPARATION": [
        "artifacts/release/v0_4_17_release_receipt.json",
        "artifacts/issue_log/ISSUE_LOG.md",
        "scratch/CURRENT_TASK.md",
    ],
    "PUBLICATION": [
        "artifacts/publication/publication_evidence_index.json",
        "docs/publication/results_reconstruction/results_draft.md",
        "scratch/CURRENT_TASK.md",
    ],
    "SCIENCE": [
        "docs/doctrine/tfne_containment_architecture.md",
        "docs/doctrine/rbs_rbd_hdp.md",
        "scratch/CURRENT_TASK.md",
    ],
    "DOCS": [
        "docs/doctrine/tfne_containment_architecture.md",
        "mkdocs.yml",
        "scratch/CURRENT_TASK.md",
    ],
    "CODE": [
        "artifacts/AGENTS.md",
        "scratch/CURRENT_TASK.md",
    ],
}


def run_git(cmd: list[str], cwd: Path) -> tuple[int, str]:
    res = subprocess.run(["git"] + cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip() if res.returncode == 0 else res.stderr.strip()


def check_gate0(
    root: Path = DEFAULT_ROOT,
    fetch: bool = True,
    offline: bool = False,
    strict_clean: bool = False,
    mode: str | None = None,
    allowed_remotes: list[str] = EXPECTED_REMOTES,
) -> int:
    print("================================================================================")
    print("GATE 0: GIT REALITY & WORKSPACE IDENTITY")
    print("================================================================================")

    # 1. Root & Git Dir
    code, git_dir = run_git(["rev-parse", "--git-dir"], cwd=root)
    if code != 0:
        print("FAIL[GATE-0]: Not inside a Git repository")
        return 1

    # 2. Remote Identity
    code, remote_url = run_git(["remote", "get-url", "origin"], cwd=root)
    if code != 0:
        print("FAIL[GATE-0]: Missing 'origin' remote")
        return 1

    remote_clean = remote_url.rstrip("/")
    if allowed_remotes and not any(remote_clean == exp.rstrip("/") for exp in allowed_remotes):
        print(f"FAIL[GATE-0]: Remote mismatch: got '{remote_url}', expected canonical remote")
        return 1

    # 3. Fetch origin with explicit failure capture
    fetch_failed = False
    if fetch and not offline:
        f_code, f_err = run_git(["fetch", "--all", "--prune"], cwd=root)
        if f_code != 0:
            print(f"FAIL[GATE-0]: git fetch failed: {f_err}")
            fetch_failed = True

    if offline or fetch_failed:
        if not offline:
            print("GATE 0 RESULT: FETCH_FAILED (Cannot verify remote truth)")
            return 1
        print("WARNING: Running in --offline mode; remote state is unverified")

    # 4. Branch & Local HEAD
    code, branch = run_git(["branch", "--show-current"], cwd=root)
    code, head = run_git(["rev-parse", "HEAD"], cwd=root)
    _, origin_main = run_git(["rev-parse", "origin/main"], cwd=root)
    _, origin_dev = run_git(["rev-parse", "origin/dev"], cwd=root)

    # 5. Tracking / Divergence Analysis
    code_behind, behind_count = run_git(["rev-list", f"HEAD..origin/{branch}", "--count"], cwd=root)
    code_ahead, ahead_count = run_git(["rev-list", f"origin/{branch}..HEAD", "--count"], cwd=root)

    n_behind = int(behind_count) if code_behind == 0 and behind_count.isdigit() else 0
    n_ahead = int(ahead_count) if code_ahead == 0 and ahead_count.isdigit() else 0

    if n_behind > 0 and n_ahead > 0:
        sync_status = f"DIVERGED (+{n_ahead}, -{n_behind})"
    elif n_behind > 0:
        sync_status = f"BEHIND (-{n_behind}) -> STALE_LOCAL_STATE"
    elif n_ahead > 0:
        sync_status = f"AHEAD (+{n_ahead})"
    else:
        sync_status = "SYNCHRONIZED"

    # 6. Dirty-Tree State
    _, status_short = run_git(["status", "--porcelain"], cwd=root)
    dirty_lines = [line for line in status_short.splitlines() if line.strip()]
    tracked_dirty = [l for l in dirty_lines if not l.startswith("??")]
    untracked = [l for l in dirty_lines if l.startswith("??")]

    # 7. Mode-Dependent Required Authorities
    task_file = root / "scratch" / "CURRENT_TASK.md"
    detected_mode = mode
    if detected_mode is None and task_file.exists():
        for line in task_file.read_text().splitlines():
            if line.startswith("mode:"):
                detected_mode = line.split(":", 1)[1].strip()
                break
    if detected_mode is None:
        detected_mode = "CODE"

    required_rel_paths = MODE_AUTHORITIES.get(detected_mode, MODE_AUTHORITIES["CODE"])
    missing_authorities = [rel for rel in required_rel_paths if not (root / rel).exists()]

    # Display Report
    print(f"Workspace Root:  {root}")
    print(f"Remote URL:      {remote_url}")
    print(f"Active Branch:   {branch} [{sync_status}]")
    print(f"Local HEAD:      {head}")
    print(f"origin/main:     {origin_main}")
    print(f"origin/dev:      {origin_dev}")
    print(f"Working Tree:    {'CLEAN' if not dirty_lines else f'{len(tracked_dirty)} tracked modified, {len(untracked)} untracked'}")
    print(f"Active Mode:     {detected_mode}")
    print(f"Authorities:     {'ALL REQUIRED PRESENT' if not missing_authorities else f'MISSING: {missing_authorities}'}")

    # Determine Gate 0 Exit (DIVERGED checked BEFORE simple behind)
    if missing_authorities:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: FAIL (Missing required authorities for mode {detected_mode}: {missing_authorities})")
        print("================================================================================")
        return 1

    if n_behind > 0 and n_ahead > 0:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: DIVERGED ({sync_status})")
        print("================================================================================")
        return 1

    if n_behind > 0:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: STALE_LOCAL_STATE ({sync_status})")
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
    parser.add_argument("--mode", type=str, default=None, help="Explicit task mode")
    args = parser.parse_args()

    sys.exit(check_gate0(fetch=not args.no_fetch, offline=args.offline, strict_clean=args.strict_clean, mode=args.mode))
