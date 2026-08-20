#!/usr/bin/env python3
"""Gate 0: Mechanical verification of Git reality.

Order of verification:
1. Verify Git root + remote
2. Fetch origin with prune (if network/remote reachable)
3. Check local branch and commit
4. Check remote origin/main and origin/dev
5. Detect STALE_LOCAL_STATE or branch divergence
6. Print structured identity report and exit 0 (clean) or 1 (stale/divergent).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip()


def check_gate0(fetch: bool = True) -> int:
    # 1. Root & Remote
    code, git_dir = run(["git", "rev-parse", "--git-dir"])
    if code != 0:
        print("FAIL[GATE-0]: Not a git repository")
        return 1

    code, remote_url = run(["git", "remote", "get-url", "origin"])
    if code != 0:
        print("FAIL[GATE-0]: Missing 'origin' remote")
        return 1

    # 2. Fetch origin
    if fetch:
        run(["git", "fetch", "--all", "--prune"])

    # 3. Branch & Local HEAD
    code, branch = run(["git", "branch", "--show-current"])
    code, head = run(["git", "rev-parse", "HEAD"])

    # 4. Remote tips
    _, origin_main = run(["git", "rev-parse", "origin/main"])
    _, origin_dev = run(["git", "rev-parse", "origin/dev"])

    # 5. Status / Stale checks
    stale_flags = []
    code, behind_count = run(["git", "rev-list", f"HEAD..origin/{branch}", "--count"])
    if code == 0 and int(behind_count) > 0:
        stale_flags.append(f"STALE_LOCAL_STATE: local '{branch}' is {behind_count} commits behind origin/{branch}")

    code, ahead_count = run(["git", "rev-list", f"origin/{branch}..HEAD", "--count"])
    ahead_str = f"+{ahead_count}" if code == 0 and int(ahead_count) > 0 else ""

    print("================================================================================")
    print("GATE 0: GIT REALITY & WORKSPACE IDENTITY")
    print("================================================================================")
    print(f"Workspace Root:  {ROOT}")
    print(f"Remote URL:      {remote_url}")
    print(f"Active Branch:   {branch} ({ahead_str}{'-' + behind_count if stale_flags else ''})")
    print(f"Local HEAD:      {head}")
    print(f"origin/main:     {origin_main}")
    print(f"origin/dev:      {origin_dev}")

    if stale_flags:
        print("--------------------------------------------------------------------------------")
        for flag in stale_flags:
            print(f"WARNING: {flag}")
        print("--------------------------------------------------------------------------------")
        print("GATE 0 RESULT: STALE_LOCAL_STATE (Sync with origin required)")
        print("================================================================================")
        return 1

    print("GATE 0 RESULT: PASS")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    fetch_opt = "--no-fetch" not in sys.argv
    sys.exit(check_gate0(fetch=fetch_opt))
