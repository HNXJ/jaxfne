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
import json
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
        "artifacts/release/current_release_authorities.json",
        "artifacts/issue_log/ISSUE_LOG.md",
        "scratch/CURRENT_TASK.md",
    ],
    "RELEASE_PREPARATION": [
        "artifacts/release/current_release_authorities.json",
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
    ],
}

RELEASE_MODES = frozenset({"RELEASE", "RELEASE_PREPARATION"})
TASK_INIT_HINT = (
    "Initialize scratch/CURRENT_TASK.md from scratch/CURRENT_TASK.example.md "
    "(set mode: RELEASE or other non-CODE mode)."
)


def read_package_version(root: Path) -> str:
    for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("pyproject.toml missing version")


def detect_task_mode(root: Path, explicit_mode: str | None) -> tuple[str, str]:
    """Return (mode, task_file_status) where status is present|absent|explicit."""
    if explicit_mode is not None:
        return explicit_mode, "explicit"
    task_file = root / "scratch" / "CURRENT_TASK.md"
    if task_file.exists():
        for line in task_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("mode:"):
                return line.split(":", 1)[1].strip(), "present"
        return "CODE", "present"
    return "CODE", "absent"


def validate_release_authorities(root: Path, mode: str) -> str | None:
    """Return an error message when RELEASE-mode authority is stale or inconsistent."""
    if mode not in RELEASE_MODES:
        return None
    auth_path = root / "artifacts" / "release" / "current_release_authorities.json"
    if not auth_path.exists():
        return "current_release_authorities.json missing"
    data = json.loads(auth_path.read_text(encoding="utf-8"))
    target = data.get("release_target_version")
    if not target:
        return "release_target_version missing in current_release_authorities.json"
    package_version = read_package_version(root)
    if target != package_version:
        return (
            f"STALE_RELEASE_AUTHORITY: release_target_version={target} "
            f"!= package version={package_version}; update "
            "artifacts/release/current_release_authorities.json before RELEASE work"
        )
    receipt_rel = data.get("release_receipt")
    if receipt_rel:
        receipt_path = root / receipt_rel
        if receipt_path.exists():
            receipt_version = json.loads(receipt_path.read_text(encoding="utf-8")).get("version")
            if receipt_version and receipt_version != target:
                return (
                    f"release receipt version {receipt_version} != "
                    f"release_target_version {target}"
                )
    return None


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
    detected_mode, task_file_status = detect_task_mode(root, mode)
    required_rel_paths = MODE_AUTHORITIES.get(detected_mode, MODE_AUTHORITIES["CODE"])
    missing_authorities = [rel for rel in required_rel_paths if not (root / rel).exists()]
    release_authority_error = validate_release_authorities(root, detected_mode)

    if task_file_status == "absent":
        task_line = (
            "absent (default mode CODE for ordinary work; "
            f"{TASK_INIT_HINT})"
        )
    elif task_file_status == "explicit":
        task_line = "explicit (--mode)"
    else:
        task_line = "present"

    # Display Report
    print(f"Workspace Root:  {root}")
    print(f"Remote URL:      {remote_url}")
    print(f"Active Branch:   {branch} [{sync_status}]")
    print(f"Local HEAD:      {head}")
    print(f"origin/main:     {origin_main}")
    print(f"origin/dev:      {origin_dev}")
    print(f"Working Tree:    {'CLEAN' if not dirty_lines else f'{len(tracked_dirty)} tracked modified, {len(untracked)} untracked'}")
    print(f"Active Mode:     {detected_mode}")
    print(f"Task File:       {task_line}")
    print(f"Authorities:     {'ALL REQUIRED PRESENT' if not missing_authorities else f'MISSING: {missing_authorities}'}")
    if release_authority_error:
        print(f"Release Auth:    FAIL ({release_authority_error})")

    # Determine Gate 0 Exit (DIVERGED checked BEFORE simple behind)
    if missing_authorities:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: FAIL (Missing required authorities for mode {detected_mode}: {missing_authorities})")
        if "scratch/CURRENT_TASK.md" in missing_authorities:
            print(f"INIT REQUIRED: {TASK_INIT_HINT}")
        print("================================================================================")
        return 1

    if release_authority_error:
        print("--------------------------------------------------------------------------------")
        print(f"GATE 0 RESULT: FAIL ({release_authority_error})")
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
