"""Nightly Release CI skips a tree that already passed (the tree-check job).

Reads the workflow with yaml.safe_load. pyyaml is installed before any pytest
step in release_ci.yml (the docs-deps step) and in the dev environment.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release_ci.yml"
GUARD = "needs.tree-check.outputs.skip != 'true'"
RUNS_ENDPOINT = "actions/workflows/release_ci.yml/runs"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _jobs() -> dict:
    return _workflow()["jobs"]


def test_tree_check_job_exposes_skip_output():
    jobs = _jobs()
    assert "tree-check" in jobs, "release_ci.yml has no tree-check job"
    outputs = jobs["tree-check"].get("outputs", {})
    assert "skip" in outputs, f"tree-check exposes no skip output: {sorted(outputs)}"
    assert "'false'" in str(outputs["skip"]), (
        "tree-check skip output must default to 'false' when its step is skipped "
        f"(non-schedule events): {outputs['skip']!r}"
    )


def test_every_other_job_needs_tree_check_and_honours_skip():
    jobs = _jobs()
    for name in ("release-test", "release-build", "publication"):
        assert name in jobs, f"required job {name!r} is missing"
    others = [name for name in jobs if name != "tree-check"]
    for name in others:
        job = jobs[name]
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "tree-check" in needs, f"{name} does not need tree-check: {needs}"
        assert GUARD in str(job.get("if", "")), (
            f"{name} if does not guard on the tree-check skip: {job.get('if')!r}"
        )
    # Existing needs are merged, not replaced.
    build_needs = jobs["release-build"]["needs"]
    assert "release-test" in build_needs, f"release-build lost release-test: {build_needs}"


def test_permissions_add_actions_read_and_keep_contents_read():
    perms = _workflow()["permissions"]
    assert perms.get("actions") == "read", f"actions permission missing: {perms}"
    assert perms.get("contents") == "read", f"contents permission lost: {perms}"


def test_runs_endpoint_step_is_guarded_to_schedule_events():
    steps = _jobs()["tree-check"]["steps"]
    calls = [step for step in steps if RUNS_ENDPOINT in str(step.get("run", ""))]
    assert calls, "no tree-check step calls the workflow runs endpoint"
    for step in calls:
        guard = str(step.get("if", ""))
        assert "github.event_name == 'schedule'" in guard, (
            f"runs-endpoint step {step.get('name')!r} is not guarded to schedule "
            f"events: if={guard!r}"
        )


CANCEL = "gh run cancel"


def test_cancel_runs_only_in_a_skip_guarded_step_of_tree_check():
    """A skipped nightly must conclude cancelled, so --status success never selects it."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count(CANCEL) == 1, f"expected one cancel command, found {text.count(CANCEL)}"
    hits = [
        (name, step)
        for name, job in _jobs().items()
        for step in job.get("steps", [])
        if CANCEL in str(step.get("run", ""))
    ]
    assert len(hits) == 1, f"cancel command is not in exactly one step: {len(hits)}"
    name, step = hits[0]
    assert name == "tree-check", f"cancel command sits in job {name!r}, not tree-check"
    guard = str(step.get("if", ""))
    assert "steps.decide.outputs.skip == 'true'" in guard, (
        f"cancel step is not guarded by the skip decision: if={guard!r}"
    )
    assert "echo" in str(step.get("run", "")), "cancel step prints no decision before cancelling"


def test_actions_write_is_job_level_on_tree_check_only():
    jobs = _jobs()
    assert jobs["tree-check"].get("permissions") == {"actions": "write", "contents": "read"}, (
        "tree-check needs job-level actions: write, contents: read: "
        f"{jobs['tree-check'].get('permissions')!r}"
    )
    for name, job in jobs.items():
        if name != "tree-check":
            assert "permissions" not in job, f"{name} sets its own permissions block"
    assert _workflow()["permissions"] == {"contents": "read", "actions": "read"}
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.count("actions: write") == 1, "actions: write must appear exactly once"
