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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_test_gate import (
    ATTESTATION_SCHEMA,
    RELEASE_CI_GATE_FAMILIES,
    attestation_path,
)


def _as_exit_code(value: object) -> int:
    """Coerce a recorded exit code, treating anything unparseable as a failure."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1


def _family_execution_evidence(rec: object) -> tuple[bool, str]:
    """Decide whether one ledger record is evidence of successful execution.

    A record only counts when it names a command that actually ran, carries both
    timestamps, and reports a zero exit for every recorded invocation. A family
    that was declared but never executed carries no command and fails here.
    """
    if not isinstance(rec, dict):
        return False, "record is not an object"
    commands = rec.get("commands") or []
    if not commands or not rec.get("command"):
        return False, "declared but not executed (no command recorded)"
    if not rec.get("started") or not rec.get("completed"):
        return False, "missing started/completed timestamps"
    for entry in commands:
        if not isinstance(entry, dict) or _as_exit_code(entry.get("exit_code")) != 0:
            return False, "a recorded command exited non-zero"
    if _as_exit_code(rec.get("exit_code")) != 0:
        return False, f"exit_code={rec.get('exit_code')!r}"
    if rec.get("status") != "PASS":
        return False, f"status={rec.get('status')!r}"
    return True, ""


def verify_pre_release_gate_receipt(
    intended_sha: str,
    *,
    expected_tree_sha: str | None = None,
) -> list[str]:
    """Authorize a release candidate from observed RC-gate execution evidence.

    Enforces, without trusting any stored literal:

      attestation.commit_sha == intended_sha
      observed_pass_families >= RELEASE_CI_GATE_FAMILIES
      every required family has actual successful execution evidence

    ``observed_pass_families`` and ``pre_release_subsumes_ci`` are re-derived
    from the per-family ledger here; the values stored in the attestation are
    cross-checked against that derivation and rejected when they disagree, so a
    hand-edited literal cannot authorize a release.
    """
    path = attestation_path()
    if not path.exists():
        return [
            f"Missing release candidate gate attestation at {path}. "
            "Run 'python scripts/run_test_gate.py rc' on the candidate SHA."
        ]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"Corrupt release candidate gate attestation: {e}"]
    if not isinstance(data, dict):
        return ["Corrupt release candidate gate attestation: top level is not an object"]

    errors: list[str] = []

    if data.get("schema") != ATTESTATION_SCHEMA:
        errors.append(
            f"RC attestation schema {data.get('schema')!r} != expected {ATTESTATION_SCHEMA!r}"
        )

    receipt_sha = data.get("commit_sha", "")
    if receipt_sha != intended_sha:
        errors.append(
            f"RC attestation commit SHA ({receipt_sha}) != intended release SHA ({intended_sha})"
        )

    if data.get("working_tree_clean") is not True:
        errors.append("RC attestation reports a dirty working tree at gate time")

    if expected_tree_sha is not None and data.get("tree_sha") != expected_tree_sha:
        errors.append(
            f"RC attestation tree SHA ({data.get('tree_sha')}) != candidate tree ({expected_tree_sha})"
        )

    records: dict[str, object] = {}
    for rec in data.get("families") or []:
        if isinstance(rec, dict) and rec.get("family"):
            records[str(rec["family"])] = rec

    observed = {fam for fam, rec in records.items() if _family_execution_evidence(rec)[0]}

    for family in sorted(RELEASE_CI_GATE_FAMILIES):
        if family not in records:
            errors.append(
                f"RC attestation has no execution record for required family {family!r}"
            )
            continue
        ok, why = _family_execution_evidence(records[family])
        if not ok:
            errors.append(
                f"Required family {family!r} lacks successful execution evidence: {why}"
            )

    missing = RELEASE_CI_GATE_FAMILIES - observed
    if missing:
        errors.append(
            f"observed_pass_families does not cover required families: {sorted(missing)}"
        )

    stored_observed = set(data.get("observed_pass_families") or [])
    if stored_observed != observed:
        errors.append(
            "Stored observed_pass_families disagrees with the per-family ledger "
            f"(stored-only: {sorted(stored_observed - observed)}, "
            f"ledger-only: {sorted(observed - stored_observed)})"
        )

    derived_subsumes = RELEASE_CI_GATE_FAMILIES.issubset(observed)
    if data.get("pre_release_subsumes_ci") is not derived_subsumes:
        errors.append(
            f"Stored pre_release_subsumes_ci ({data.get('pre_release_subsumes_ci')!r}) "
            f"disagrees with the value derived from observed execution ({derived_subsumes!r})"
        )

    if data.get("status") != "PASS":
        errors.append(f"RC attestation status is {data.get('status')!r}, expected 'PASS'")

    return errors


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

    # 11. Gate: the RC attestation must record observed execution of every
    #     blocking family, on this exact candidate (SHA and tree), with a clean
    #     working tree. Nothing here trusts a stored literal.
    intended_tree_sha = run_cmd(["git", "rev-parse", f"{intended_sha}^{{tree}}"]) or None
    rc_errors = verify_pre_release_gate_receipt(
        intended_sha, expected_tree_sha=intended_tree_sha
    )
    failure_reasons.extend(rc_errors)

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
        "release_candidate_receipt_valid": len(rc_errors) == 0,
        "release_target_reconciled": reconciled,
        "safe_to_repair_tag": reconciled,
        "safe_to_upload": reconciled,
        "failure_reasons": failure_reasons,
    }

    print(json.dumps(report, indent=2))
    sys.exit(0 if reconciled else 1)


if __name__ == "__main__":
    main()
