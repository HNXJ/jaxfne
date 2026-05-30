#!/usr/bin/env python3
"""
Release target reconciler for jaxfne.

Verifies that origin/main, CI headSha, and the intended release SHA all
agree before any tag repair, GitHub Release edit, TestPyPI, or PyPI upload.

Usage:
    python scripts/release/reconcile_release_target.py \
        --version 0.3.14 \
        --target-sha 1e645118f078ef315935893a8486f21bd2bdacbe

    python scripts/release/reconcile_release_target.py \
        --version 0.3.14 \
        --target-sha 1e645118f078ef315935893a8486f21bd2bdacbe \
        --ci-run-id 12345678

Exit codes:
    0 — all gates pass (release_target_reconciled=true)
    1 — one or more gates fail (release_target_reconciled=false)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(args, check=False):
    """Run a subprocess command and return stdout, or empty string on failure."""
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=check)
        return res.stdout.strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile release target SHA across git, CI, and tag."
    )
    parser.add_argument("--version", required=True, help="Version string (e.g. 0.3.14)")
    parser.add_argument(
        "--target-sha",
        required=True,
        help="Intended release commit SHA (peeled commit, not tag object)",
    )
    parser.add_argument(
        "--ci-run-id",
        default=None,
        help="GitHub Actions run ID for headSha lookup (optional)",
    )
    args = parser.parse_args()

    version = args.version
    intended_sha = args.target_sha
    failure_reasons = []

    # 1. Fetch remote state
    run_cmd(["git", "fetch", "origin", "--prune", "--tags"])

    # 2. origin/main SHA
    origin_main_sha = run_cmd(["git", "rev-parse", "origin/main"])
    if not origin_main_sha:
        failure_reasons.append("Could not resolve origin/main SHA")

    # 3. Local HEAD SHA (not CI — local HEAD is not a substitute for CI headSha)
    local_head_sha = run_cmd(["git", "rev-parse", "HEAD"])

    # 4. CI headSha: resolve from gh CLI if run-id provided, else unknown
    ci_head_sha = "unknown"
    ci_conclusion = "unknown"
    if args.ci_run_id:
        ci_head_sha = run_cmd(
            ["gh", "run", "view", args.ci_run_id, "--json", "headSha", "-q", ".headSha"]
        )
        ci_conclusion = run_cmd(
            ["gh", "run", "view", args.ci_run_id, "--json", "conclusion", "-q", ".conclusion"]
        )
        if not ci_head_sha:
            failure_reasons.append(f"Could not resolve CI headSha for run {args.ci_run_id}")
        if not ci_conclusion:
            ci_conclusion = "unknown"
    else:
        failure_reasons.append(
            "No --ci-run-id provided; ci_head_sha and ci_conclusion cannot be verified"
        )

    # 5. Tag audit: annotated tag object SHA and peeled commit SHA
    tag_ref = f"refs/tags/v{version}"
    tag_object_sha = run_cmd(["git", "ls-remote", "origin", tag_ref])
    tag_object_sha = tag_object_sha.split()[0] if tag_object_sha else ""

    tag_peeled_sha = run_cmd(["git", "ls-remote", "origin", f"{tag_ref}^{{}}"])
    tag_peeled_sha = tag_peeled_sha.split()[0] if tag_peeled_sha else ""

    # 6. Working tree status
    git_status = run_cmd(["git", "status", "--porcelain"])
    working_tree_clean = git_status == ""
    if not working_tree_clean:
        failure_reasons.append(
            f"Working tree is dirty: {git_status[:120]!r}"
        )

    # 7. Gate: origin/main must equal intended_release_sha
    if origin_main_sha and origin_main_sha != intended_sha:
        failure_reasons.append(
            f"origin/main ({origin_main_sha}) != intended_release_sha ({intended_sha})"
        )

    # 8. Gate: CI headSha must equal intended_release_sha (only if known)
    if ci_head_sha not in ("unknown", "") and ci_head_sha != intended_sha:
        failure_reasons.append(
            f"CI headSha ({ci_head_sha}) != intended_release_sha ({intended_sha})"
        )

    # 9. Gate: CI conclusion must be "success" (only if known)
    if ci_conclusion not in ("unknown", "") and ci_conclusion != "success":
        failure_reasons.append(f"CI conclusion is {ci_conclusion!r}, not 'success'")

    # 10. Gate: tag peeled SHA must match intended (if tag exists)
    if tag_peeled_sha and tag_peeled_sha != intended_sha:
        failure_reasons.append(
            f"Tag peeled SHA ({tag_peeled_sha}) != intended_release_sha ({intended_sha})"
        )

    # Reconciliation only true when ALL gates pass
    reconciled = len(failure_reasons) == 0

    report = {
        "version": version,
        "intended_release_sha": intended_sha,
        "origin_main_sha": origin_main_sha,
        "local_head_sha": local_head_sha,
        "ci_head_sha": ci_head_sha,
        "ci_conclusion": ci_conclusion,
        "tag_object_sha": tag_object_sha,
        "tag_peeled_sha": tag_peeled_sha,
        "working_tree_clean": working_tree_clean,
        "release_target_reconciled": reconciled,
        "safe_to_repair_tag": False,
        "safe_to_upload": False,
        "failure_reasons": failure_reasons,
    }

    print(json.dumps(report, indent=2))
    sys.exit(0 if reconciled else 1)


if __name__ == "__main__":
    main()
