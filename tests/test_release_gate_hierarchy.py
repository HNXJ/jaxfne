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
    BROAD_MARKER_EXPR,
    CHECK_FAMILIES,
    NOTEBOOK_MARKER_EXPR,
    RC_MARKER_EXPRS,
    SLOW_MARKER_EXPR,
    RELEASE_CI_GATE_FAMILIES,
    GATE_CHECK_FAMILIES,
    GATES,
    observed_pass_families,
    record_execution,
    reset_observations,
)
from scripts.release.reconcile_release_target import verify_pre_release_gate_receipt

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

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
# --- Effective test population, not just family names -------------------------
#
# Matching check-family names does NOT prove matching test sets. Release CI on
# main runs `pytest tests` with no marker filter; the RC gate ran only
# "not slow" and "slow and not notebook", so the 30 node ids carrying the
# `notebook` marker were executed by release CI and never by the RC gate. Every
# family name lined up while the invariant was broken underneath. These gates
# check the selection algebra itself.


def _selects(expr: str, *, slow: bool, notebook: bool) -> bool:
    """Evaluate a pytest -m expression over the (slow, notebook) marker algebra."""
    return bool(eval(expr, {"__builtins__": {}}, {"slow": slow, "notebook": notebook}))


def test_rc_marker_selectors_are_exhaustive():
    """The RC sweeps must cover every (slow, notebook) combination.

    Release CI applies no marker filter, so an uncovered combination is a test
    release CI runs and the RC gate does not.
    """
    uncovered = [
        (slow, notebook)
        for slow in (False, True)
        for notebook in (False, True)
        if not any(_selects(e, slow=slow, notebook=notebook) for e in RC_MARKER_EXPRS)
    ]
    assert not uncovered, (
        "RC gate does not select these (slow, notebook) combinations: "
        f"{uncovered}; release CI would execute them and the RC gate would not"
    )


def test_notebook_selector_is_the_gap_the_other_two_leave():
    """The notebook sweep exists precisely to close the broad/slow gap."""
    gap = [
        (slow, notebook)
        for slow in (False, True)
        for notebook in (False, True)
        if not _selects(BROAD_MARKER_EXPR, slow=slow, notebook=notebook)
        and not _selects(SLOW_MARKER_EXPR, slow=slow, notebook=notebook)
    ]
    assert gap == [(True, True)], f"unexpected broad/slow gap: {gap}"
    for slow, notebook in gap:
        assert _selects(NOTEBOOK_MARKER_EXPR, slow=slow, notebook=notebook)


def test_release_ci_pytest_sweep_is_unfiltered():
    """If release CI ever gains a -m filter, the exhaustiveness argument changes.

    The proof that RC covers release rests on release CI selecting *everything*.
    """
    text = (ROOT / ".github" / "workflows" / "release_ci.yml").read_text(encoding="utf-8")
    # Inspect only the arguments after the `pytest` token: `python -m pytest`
    # is the module flag, not a marker filter.
    sweeps = [ln.split("pytest", 1)[1] for ln in text.splitlines() if "python -m pytest" in ln]
    unfiltered = [args for args in sweeps if " -m " not in args]
    assert unfiltered, (
        "release_ci.yml no longer has an unfiltered pytest sweep; the RC "
        "coverage argument rests on release CI selecting everything"
    )
    for args in unfiltered:
        assert "--ignore" not in args, (
            f"release CI sweep gained an --ignore: {args.strip()!r}"
        )


def test_pytest_notebook_is_a_required_family():
    """Release CI executes the notebook set, so it is release-blocking."""
    assert "pytest_notebook" in RELEASE_CI_GATE_FAMILIES
    assert "pytest_notebook" in GATE_CHECK_FAMILIES["rc"]
    assert "pytest_notebook" in GATE_CHECK_FAMILIES["release"]


# --- Branch-protection context uniqueness -------------------------------------


def _workflow_job_names(filename: str) -> list[str]:
    """Top-level job ids in a workflow (two-space indented keys under `jobs:`)."""
    text = (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
    names, in_jobs = [], False
    for line in text.splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if in_jobs:
            if line and not line.startswith(" "):
                break
            if re.fullmatch(r"  [A-Za-z0-9_-]+:", line):
                names.append(line.strip().rstrip(":"))
    return names


SUPPORTED_CI_PYTHONS = {"3.11", "3.14"}

# Parsed with regex, not yaml.safe_load: release_ci.yml installs only
# ".[dev,jaxley]" and pyyaml is declared in the `io` extra, so importing yaml
# here fails on that runner. Guarding the import with skipif would be worse --
# a test that disappears when a dependency is absent is the exact defect
# scripts/check_environment_parity.py exists to catch.
_PY_MATRIX_RE = re.compile(r"^[ 	]*python-version:[ 	]*\[([^\]]*)\][ 	]*$", re.M)
_PY_PIN_RE = re.compile(r"""^[ 	]*python-version:[ 	]*['"]?(\d+\.\d+)['"]?[ 	]*$""", re.M)


def _matrix_pythons(filename: str) -> list[set[str]]:
    """Every `python-version: [...]` matrix list in a workflow."""
    text = (WORKFLOWS / filename).read_text(encoding="utf-8")
    return [
        {v.strip().strip("'\"") for v in match.group(1).split(",") if v.strip()}
        for match in _PY_MATRIX_RE.finditer(text)
    ]


def _pinned_pythons(text: str) -> list[str]:
    """Literal scalar `python-version:` pins (matrix expressions never match)."""
    return _PY_PIN_RE.findall(text)


def test_ci_python_coverage_is_the_declared_two_versions():
    """CI exercises exactly the two declared interpreter lines, and no others.

    Coverage is deliberately the ends of the supported range rather than every
    minor version. 3.14 is included specifically because the local
    release-candidate gate runs on it: without it the gate would certify a
    release on an interpreter CI never exercises.
    """
    for name in ("ci.yml", "release_ci.yml"):
        matrices = _matrix_pythons(name)
        assert matrices, f"{name}: no python-version matrix found"
        for versions in matrices:
            assert versions == SUPPORTED_CI_PYTHONS, (
                f"{name} tests {sorted(versions)}, "
                f"expected exactly {sorted(SUPPORTED_CI_PYTHONS)}"
            )


def test_no_test_workflow_pins_a_retired_interpreter():
    """A workflow that installs the dev extras must use a covered interpreter."""
    checked = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if '".[dev' not in text:
            continue  # build/publish jobs do not need the dev extras
        checked += 1
        for pinned in _pinned_pythons(text):
            assert pinned in SUPPORTED_CI_PYTHONS, (
                f"{path.name} pins Python {pinned}, which CI no longer covers; "
                f"expected one of {sorted(SUPPORTED_CI_PYTHONS)}"
            )
    assert checked, "no workflow installing the dev extras was found"


def test_ci_job_names_are_unique_across_workflows():
    """Required status checks are keyed by job name, so names must not collide.

    `test (3.12)`, `test (3.13)` and `build` were previously emitted by both
    workflows, so a required context could be satisfied by whichever run
    reported last -- enforcement weaker than it appeared.
    """
    fast = _workflow_job_names("ci.yml")
    release = _workflow_job_names("release_ci.yml")
    assert fast and release, f"failed to parse job names: fast={fast} release={release}"
    collisions = set(fast) & set(release)
    assert not collisions, (
        f"job names shared by ci.yml and release_ci.yml: {sorted(collisions)}; "
        "branch protection cannot require one workflow's job specifically"
    )


SUPPORTED_PYTHONS = {"3.11", "3.12", "3.13", "3.14"}
PYTHON_FLOOR = "3.11"
PUBLISH_PYTHON = "3.11"


def test_python_support_policy():
    """Metadata, classifiers, CI endpoints, publish pins, and docs must agree.

    JaxFNE supports Python 3.11-3.14; blocking CI validates the range at its
    3.11 and 3.14 endpoints. 3.12/3.13 are supported but not independently
    CI-tested; 3.10 is unsupported (the jax line itself floors at >=3.11).
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"""^requires-python\s*=\s*["']>=(\d+\.\d+)["']""", text, re.M)
    assert m, "requires-python not found in pyproject.toml"
    assert m.group(1) == PYTHON_FLOOR, (
        f"requires-python floors at {m.group(1)}, expected >={PYTHON_FLOOR}"
    )
    classifiers = set(
        re.findall(r"Programming Language :: Python :: (\d+\.\d+)", text)
    )
    assert classifiers == SUPPORTED_PYTHONS, (
        f"classifiers cover {sorted(classifiers)}, "
        f"expected exactly {sorted(SUPPORTED_PYTHONS)}"
    )
    pub = (WORKFLOWS / "publish.yml").read_text(encoding="utf-8")
    pinned = _pinned_pythons(pub)
    assert pinned, "publish.yml has no python-version pins"
    for version in pinned:
        assert version == PUBLISH_PYTHON, (
            f"publish.yml builds on {version}, expected {PUBLISH_PYTHON} "
            "(inside the tested set)"
        )
    policy = (ROOT / "docs" / "ci_policy.md").read_text(encoding="utf-8")
    assert "JaxFNE supports Python 3.11-3.14." in policy
    assert "at its 3.11 and 3.14 endpoints" in policy
    assert "not independently exercised by the full CI matrix" in policy
    faq = (ROOT / "docs" / "faq.md").read_text(encoding="utf-8")
    assert "requires Python 3.11 or later" in faq
    assert "3.10 or later" not in faq
    contrib = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    assert "(3.11 and 3.14 tested)" in contrib


# --------------------------------------------------------------------------
# Artifact provenance: the publishable bytes must be the validated bytes.
#
# The defect these tests exist to prevent: every gate ran against artifacts
# that were then thrown away, and publish rebuilt from source. Byte-identical
# output would have been luck, not a property -- nothing compared them.
# --------------------------------------------------------------------------

PUBLISH_WORKFLOW = WORKFLOWS / "publish.yml"
RELEASE_WORKFLOW = WORKFLOWS / "release_ci.yml"


def test_publish_workflow_never_builds():
    """Publishing must consume retained artifacts, never produce new ones."""
    text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    commands = [
        line for line in text.splitlines()
        if not line.lstrip().startswith("#") and "python -m build" in line
    ]
    assert commands == [], (
        "publish.yml builds distributions. Bytes produced at publish time have "
        f"passed no gate: {commands}"
    )


def test_publish_workflow_verifies_hashes_before_upload():
    """SHA256 equality with the manifest must gate the upload, not follow it."""
    text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    verify_at = text.find("build_release_manifest.py --verify")
    upload_at = text.find("pypa/gh-action-pypi-publish")
    assert verify_at != -1, "publish.yml does not verify artifact hashes"
    assert upload_at != -1, "publish.yml does not upload"
    assert verify_at < upload_at, "hash verification must precede the first upload"
    assert text.count("build_release_manifest.py --verify") == text.count(
        "pypa/gh-action-pypi-publish"
    ), "every upload job must carry its own hash guard"


def test_release_build_retains_artifacts_and_provenance():
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "name: release-dist" in text, "release-build discards the artifacts it validates"
    assert "if-no-files-found: error" in text
    assert "build_release_manifest.py" in text, "no provenance recorded for the built bytes"
    assert text.count("python -m build") == 1, (
        "release_ci.yml builds more than once; only one build can be the retained one"
    )


def test_build_backend_is_pinned_exactly():
    """A ranged backend silently changes the artifact hash.

    The wheel embeds ``Generator: hatchling <version>``, so resolving a
    different version inside a range produces different bytes with no failure.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requires = re.search(r"^requires = \[(.*?)\]", pyproject, re.M | re.S)
    assert requires is not None
    for spec in re.findall(r'"([^"]+)"', requires.group(1)):
        assert "==" in spec, f"build backend requirement {spec!r} is not pinned exactly"


def test_release_manifest_is_not_committed():
    """A committed manifest cannot certify the commit that contains it."""
    tracked = list(ROOT.rglob("RELEASE_MANIFEST.json"))
    for path in tracked:
        assert "dist" in path.parts or "release_candidate" in path.parts, (
            f"{path} looks committed; a manifest naming its own containing commit "
            "is self-referential"
        )


def test_junit_parity_is_a_required_rc_check_family():
    """The comparator must be required, not merely available."""
    assert "junit_parity" in CHECK_FAMILIES
    assert "junit_parity" in GATE_CHECK_FAMILIES["rc"], (
        "RC-vs-CI per-node comparison is not declared for the rc gate"
    )
    source = (ROOT / "scripts" / "run_test_gate.py").read_text(encoding="utf-8")
    assert 'family="junit_parity"' in source, (
        "junit_parity is declared but never executed; a declared-only family "
        "cannot fail the gate"
    )
    assert "check_junit_parity.py" in source


def test_junit_parity_checker_fails_closed():
    """Every divergence class must fail; only platform differences may pass."""
    from scripts.compare_pytest_junit import compare

    passing = {"t::a": {"outcome": "passed", "reason": ""}}
    assert compare(passing, dict(passing))["pass"] is True

    justified = compare(
        {"t::a": {"outcome": "skipped", "reason": "requires a POSIX shell"}},
        {"t::a": {"outcome": "passed", "reason": ""}},
    )
    assert justified["pass"] is True
    assert justified["outcome_differences"][0]["classification"] == (
        "INTENTIONAL_PLATFORM_DIFFERENCE"
    )

    for rc, ci in (
        ({"t::a": {"outcome": "passed", "reason": ""}}, {}),                      # rc_only
        ({}, {"t::a": {"outcome": "passed", "reason": ""}}),                      # ci_only
        (                                                                          # unknown
            {"t::a": {"outcome": "skipped", "reason": "no idea"}},
            {"t::a": {"outcome": "passed", "reason": ""}},
        ),
        (                                                                          # env defect
            {"t::a": {"outcome": "skipped", "reason": "reportlab not installed"}},
            {"t::a": {"outcome": "passed", "reason": ""}},
        ),
        (                                                                          # failure
            {"t::a": {"outcome": "failure", "reason": "boom"}},
            {"t::a": {"outcome": "failure", "reason": "boom"}},
        ),
    ):
        assert compare(rc, ci)["pass"] is False, f"divergence passed: {rc} vs {ci}"
