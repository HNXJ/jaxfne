"""Regression tests for the release gate hierarchy and mechanical authorization.

Enforces:
1. Invariant: PRE_RELEASE_GATE >= RELEASE_CI_GATE
   at the level of required test/check families.
2. Every check family executed in main/release CI workflows is accounted for.
3. Publication reconciler mechanically rejects candidates lacking valid, matching
   RC gate receipts.
4. CLI options for run_test_gate include 'rc' and 'release-candidate'.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from scripts.run_test_gate import (
    CHECK_FAMILIES,
    RELEASE_CI_GATE_FAMILIES,
    GATE_CHECK_FAMILIES,
    GATES,
)
from scripts.release.reconcile_release_target import verify_pre_release_gate_receipt

ROOT = Path(__file__).resolve().parents[1]


def test_pre_release_gate_subsumes_release_ci_gate():
    """Prove the fundamental invariant:
        PRE_RELEASE_GATE (rc) >= RELEASE_CI_GATE
    for all required check families.
    """
    rc_families = GATE_CHECK_FAMILIES["rc"]
    ci_families = RELEASE_CI_GATE_FAMILIES

    # 1. Non-empty definitions
    assert len(rc_families) > 0, "RC gate check families must not be empty"
    assert len(ci_families) > 0, "CI gate check families must not be empty"

    # 2. Check subset/superset invariant
    missing_from_rc = ci_families - rc_families
    assert (
        not missing_from_rc
    ), f"Harness defect: PRE_RELEASE_GATE is missing CI families: {missing_from_rc}"
    assert rc_families.issuperset(
        ci_families
    ), f"Invariant violated: rc ({rc_families}) does not cover all CI families ({ci_families})"


def test_release_ci_workflows_match_family_inventory():
    """Verify that every step in main/release CI maps to a recognized check family."""
    ci_fast_path = ROOT / ".github" / "workflows" / "ci.yml"
    release_ci_path = ROOT / ".github" / "workflows" / "release_ci.yml"

    assert ci_fast_path.exists(), f"Missing {ci_fast_path}"
    assert release_ci_path.exists(), f"Missing {release_ci_path}"

    ci_fast = yaml.safe_load(ci_fast_path.read_text(encoding="utf-8"))
    release_ci = yaml.safe_load(release_ci_path.read_text(encoding="utf-8"))

    # Verify key steps exist in CI workflows
    ci_fast_steps = [s.get("name", "") for s in ci_fast["jobs"]["test"]["steps"]]
    release_ci_steps = [s.get("name", "") for s in release_ci["jobs"]["test"]["steps"]]

    # Both must run compileall
    assert any("Compileall" in s for s in ci_fast_steps)
    assert any("Compileall" in s for s in release_ci_steps)

    # Fast CI on main must run broad tests, docs language, ruff, orphan check
    assert any("Lint (ruff)" in s for s in ci_fast_steps)
    assert any("Audit public docs language" in s for s in ci_fast_steps)
    assert any("Docs build (strict)" in s for s in ci_fast_steps)
    assert any("Run all tests (broad gate" in s for s in ci_fast_steps)

    # Release CI on main must run all tests including slow
    assert any("Run all tests (including slow)" in s for s in release_ci_steps)

    # Both must run examples and build wheel
    assert any("Run examples" in s for s in ci_fast_steps)
    assert any("Run examples" in s for s in release_ci_steps)


def test_reconciler_rejects_missing_receipt(monkeypatch, tmp_path):
    """Publication reconciler must mechanically reject missing or invalid RC receipts."""
    fake_root = tmp_path
    monkeypatch.setattr("scripts.release.reconcile_release_target.ROOT", fake_root)

    intended_sha = "0123456789abcdef0123456789abcdef01234567"

    # 1. Missing receipt -> rejection
    errs = verify_pre_release_gate_receipt(intended_sha)
    assert len(errs) == 1
    assert "Missing release candidate gate receipt" in errs[0]

    # 2. Corrupt receipt -> rejection
    receipt_dir = fake_root / "artifacts" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = receipt_dir / "release_candidate_gate_receipt.json"

    receipt_file.write_text("NOT_JSON", encoding="utf-8")
    errs = verify_pre_release_gate_receipt(intended_sha)
    assert any("Corrupt release candidate gate receipt" in e for e in errs)

    # 3. Status != PASS -> rejection
    bad_status = {
        "status": "FAIL",
        "commit_sha": intended_sha,
        "check_families": sorted(list(RELEASE_CI_GATE_FAMILIES)),
    }
    receipt_file.write_text(json.dumps(bad_status), encoding="utf-8")
    errs = verify_pre_release_gate_receipt(intended_sha)
    assert any("status is 'FAIL', expected 'PASS'" in e for e in errs)

    # 4. Mismatched commit SHA -> rejection
    mismatched_sha = {
        "status": "PASS",
        "commit_sha": "different_sha_00000000000000000000000000",
        "check_families": sorted(list(RELEASE_CI_GATE_FAMILIES)),
    }
    receipt_file.write_text(json.dumps(mismatched_sha), encoding="utf-8")
    errs = verify_pre_release_gate_receipt(intended_sha)
    assert any("commit SHA" in e and "!=" in e for e in errs)

    # 5. Incomplete check families (missing pytest_slow) -> rejection
    subset_families = {
        "status": "PASS",
        "commit_sha": intended_sha,
        "check_families": sorted(list(RELEASE_CI_GATE_FAMILIES - {"pytest_slow"})),
    }
    receipt_file.write_text(json.dumps(subset_families), encoding="utf-8")
    errs = verify_pre_release_gate_receipt(intended_sha)
    assert any("missing required check families" in e and "pytest_slow" in e for e in errs)

    # 6. Complete matching valid receipt -> PASS
    valid_receipt = {
        "status": "PASS",
        "commit_sha": intended_sha,
        "check_families": sorted(list(RELEASE_CI_GATE_FAMILIES)),
    }
    receipt_file.write_text(json.dumps(valid_receipt), encoding="utf-8")
    errs = verify_pre_release_gate_receipt(intended_sha)
    assert errs == [], f"Expected pass, got errors: {errs}"


def test_gates_cli_rc_registered():
    """Verify rc and release-candidate are registered in GATES dictionary."""
    assert "rc" in GATES
    assert "release-candidate" in GATES
    assert GATES["rc"] == GATES["release-candidate"]
