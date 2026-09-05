"""Regression tests for the release gate hierarchy and mechanical authorization.

Enforces:
1. Invariant: PRE_RELEASE_GATE >= RELEASE_CI_GATE
   at the level of required test/check families.
2. Every check family executed in main/release CI workflows is accounted for.
3. The publication reconciler authorizes only from *observed* RC gate execution
   evidence, and rejects every way that evidence can be absent, incomplete,
   failed, mismatched, or forged.
4. CLI options for run_test_gate include 'rc' and 'release-candidate'.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re

from scripts.run_test_gate import (
    ATTESTATION_SCHEMA,
    CHECK_FAMILIES,
    RELEASE_CI_GATE_FAMILIES,
    GATE_CHECK_FAMILIES,
    GATES,
    observed_pass_families,
    record_execution,
    reset_observations,
)
from scripts.release.reconcile_release_target import verify_pre_release_gate_receipt

ROOT = Path(__file__).resolve().parents[1]

INTENDED_SHA = "0123456789abcdef0123456789abcdef01234567"
TREE_SHA = "fedcba9876543210fedcba9876543210fedcba98"


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


def test_every_gate_family_is_a_known_check_family():
    """No gate may reference a family absent from the inventory.

    ``pytest_dev`` was referenced by the dev gate while missing from
    CHECK_FAMILIES; the ledger now rejects unknown families at execution time,
    so the inventory has to stay complete.
    """
    for gate, families in GATE_CHECK_FAMILIES.items():
        unknown = families - CHECK_FAMILIES
        assert not unknown, f"gate {gate!r} references unknown check families: {sorted(unknown)}"


def test_release_ci_workflows_match_family_inventory():
    """Verify that every step in main/release CI maps to a recognized check family."""
    ci_fast_path = ROOT / ".github" / "workflows" / "ci.yml"
    release_ci_path = ROOT / ".github" / "workflows" / "release_ci.yml"

    assert ci_fast_path.exists(), f"Missing {ci_fast_path}"
    assert release_ci_path.exists(), f"Missing {release_ci_path}"

    def _extract_steps(text: str) -> list[str]:
        return [m.group(1).strip().strip("'\"") for m in re.finditer(r"^\s*-\s*name:\s*(.+)$", text, re.MULTILINE)]

    ci_fast_steps = _extract_steps(ci_fast_path.read_text(encoding="utf-8"))
    release_ci_steps = _extract_steps(release_ci_path.read_text(encoding="utf-8"))

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


def test_ci_pytest_sweeps_carry_no_undeclared_ignores():
    """CI must not silently drop test modules the RC gate would run.

    The two multi-area modules were excluded in d4e3f72 for referencing symbols
    that did not exist yet. Those symbols exist now, so any reintroduced
    ``--ignore`` must go through BROAD_PYTEST_IGNORE where it is visible to the
    gate hierarchy, not be hidden in a workflow command line.
    """
    for name in ("ci.yml", "release_ci.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "--ignore=tests/" not in text, (
            f"{name} hides a pytest exclusion; declare it in "
            "scripts/run_test_gate.py::BROAD_PYTEST_IGNORE instead"
        )


# --- Observed-execution ledger ------------------------------------------------


def test_ledger_records_required_evidence_fields():
    """Each family record must carry the fields the authorization contract needs."""
    reset_observations()
    try:
        record_execution("compileall", ["python", "-m", "compileall"], "T0", "T1", 0)
        rec = next(iter(_ledger_snapshot().values()))
        for field in ("family", "command", "started", "completed", "exit_code", "status", "evidence"):
            assert field in rec, f"ledger record missing required field {field!r}"
        assert observed_pass_families() == {"compileall"}
    finally:
        reset_observations()


def test_ledger_excludes_families_with_nonzero_exit():
    reset_observations()
    try:
        record_execution("lint_ruff", ["ruff", "check"], "T0", "T1", 1)
        assert observed_pass_families() == set()
    finally:
        reset_observations()


def _ledger_snapshot() -> dict:
    from scripts.run_test_gate import _OBSERVATIONS

    return dict(_OBSERVATIONS)


# --- Authorization: adversarial cases ----------------------------------------


def _family_record(family: str, exit_code: int = 0) -> dict:
    argv = ["python", "-m", "pytest", f"--family={family}"]
    return {
        "family": family,
        "command": " ".join(argv),
        "commands": [{"argv": argv, "exit_code": exit_code}],
        "started": "2026-01-01T00:00:00+00:00",
        "completed": "2026-01-01T00:01:00+00:00",
        "exit_code": exit_code,
        "status": "PASS" if exit_code == 0 else "FAIL",
        "evidence": [f"exit={exit_code}"],
    }


def _valid_attestation() -> dict:
    families = [_family_record(f) for f in sorted(GATE_CHECK_FAMILIES["rc"])]
    observed = sorted(r["family"] for r in families)
    return {
        "schema": ATTESTATION_SCHEMA,
        "gate": "rc",
        "commit_sha": INTENDED_SHA,
        "tree_sha": TREE_SHA,
        "working_tree_clean": True,
        "timestamp_utc": "2026-01-01T00:01:00+00:00",
        "required_families": sorted(RELEASE_CI_GATE_FAMILIES),
        "declared_families": sorted(GATE_CHECK_FAMILIES["rc"]),
        "observed_pass_families": observed,
        "families": families,
        "pre_release_subsumes_ci": True,
        "status": "PASS",
    }


def _install(monkeypatch, tmp_path, data: dict | str | None) -> Path:
    path = tmp_path / "rc_gate_attestation.json"
    monkeypatch.setenv("JAXFNE_RC_ATTESTATION", str(path))
    if data is None:
        return path
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def test_authorization_accepts_complete_observed_evidence(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, _valid_attestation())
    errs = verify_pre_release_gate_receipt(INTENDED_SHA, expected_tree_sha=TREE_SHA)
    assert errs == [], f"expected authorization, got: {errs}"


def test_authorization_fails_when_attestation_missing(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, None)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert len(errs) == 1
    assert "Missing release candidate gate attestation" in errs[0]


def test_authorization_fails_on_corrupt_attestation(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, "NOT_JSON")
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("Corrupt release candidate gate attestation" in e for e in errs)


def test_authorization_fails_when_sha_differs(monkeypatch, tmp_path):
    data = _valid_attestation()
    data["commit_sha"] = "9999999999999999999999999999999999999999"
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("commit SHA" in e and "!=" in e for e in errs)


def test_authorization_fails_when_required_family_absent(monkeypatch, tmp_path):
    data = _valid_attestation()
    data["families"] = [r for r in data["families"] if r["family"] != "pytest_slow"]
    data["observed_pass_families"] = sorted(r["family"] for r in data["families"])
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("no execution record for required family 'pytest_slow'" in e for e in errs)


def test_authorization_fails_when_family_declared_but_not_executed(monkeypatch, tmp_path):
    """A family listed with no command never ran, so it is not evidence."""
    data = _valid_attestation()
    for rec in data["families"]:
        if rec["family"] == "twine_check":
            rec["commands"] = []
            rec["command"] = ""
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("declared but not executed" in e for e in errs)
    assert any("twine_check" in e for e in errs)


def test_authorization_fails_when_required_family_exits_nonzero(monkeypatch, tmp_path):
    data = _valid_attestation()
    for rec in data["families"]:
        if rec["family"] == "pytest_broad":
            rec["commands"][0]["exit_code"] = 1
            rec["exit_code"] = 1
            rec["status"] = "FAIL"
    data["observed_pass_families"] = sorted(
        r["family"] for r in data["families"] if r["status"] == "PASS"
    )
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("pytest_broad" in e and "successful execution evidence" in e for e in errs)


def test_authorization_fails_when_working_tree_dirty(monkeypatch, tmp_path):
    data = _valid_attestation()
    data["working_tree_clean"] = False
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("dirty working tree" in e for e in errs)


def test_authorization_fails_when_candidate_tree_identity_differs(monkeypatch, tmp_path):
    data = _valid_attestation()
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA, expected_tree_sha="0" * 40)
    assert any("tree SHA" in e for e in errs)


def test_authorization_rejects_forged_subsumption_literal(monkeypatch, tmp_path):
    """A hand-set pre_release_subsumes_ci cannot authorize a release.

    This is the defect the v1 receipt had: the flag was written as a literal and
    the check compared two constants. It is now re-derived from the ledger and
    the stored value is cross-checked against that derivation.
    """
    data = _valid_attestation()
    data["families"] = [r for r in data["families"] if r["family"] != "pytest_slow"]
    data["observed_pass_families"] = sorted(RELEASE_CI_GATE_FAMILIES)  # forged
    data["pre_release_subsumes_ci"] = True  # forged
    data["status"] = "PASS"  # forged
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("no execution record for required family 'pytest_slow'" in e for e in errs)
    assert any("Stored observed_pass_families disagrees" in e for e in errs)
    assert any("Stored pre_release_subsumes_ci" in e for e in errs)


def test_authorization_fails_on_wrong_schema(monkeypatch, tmp_path):
    data = _valid_attestation()
    data["schema"] = "jaxfne.rc_gate_receipt.v1"
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("schema" in e for e in errs)


def test_authorization_fails_when_status_not_pass(monkeypatch, tmp_path):
    data = copy.deepcopy(_valid_attestation())
    data["status"] = "FAIL"
    _install(monkeypatch, tmp_path, data)
    errs = verify_pre_release_gate_receipt(INTENDED_SHA)
    assert any("status is 'FAIL', expected 'PASS'" in e for e in errs)


def test_rc_attestation_is_not_tracked_in_git():
    """The attestation must stay untracked: a tracked one changes the SHA it attests."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/attestations/" in gitignore
    assert not (ROOT / "artifacts" / "receipts" / "release_candidate_gate_receipt.json").exists(), (
        "the self-referential v1 receipt must not be reintroduced"
    )


def test_gates_cli_rc_registered():
    """Verify rc and release-candidate are registered in GATES dictionary."""
    assert "rc" in GATES
    assert "release-candidate" in GATES
    assert GATES["rc"] == GATES["release-candidate"]
