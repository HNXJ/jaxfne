"""The broad gate's code-tree stamp (lane D) and the rc gate's Release CI reuse (lane F).

Each test moves ``rtg.ROOT`` onto a temporary git repository and replaces ``rtg._run`` with a
recorder, so no gate command, pytest sweep, build or network call runs. The families the
recorder sees are what the gate would have executed. New names are reached through the module
(not imported), and each test asserts its precondition first, so it fails on the missing
behaviour rather than passing vacuously.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import scripts.run_test_gate as rtg
from scripts.release.reconcile_release_target import verify_pre_release_gate_receipt

ROOT = Path(__file__).resolve().parents[1]
REPORT_FAMILIES = {family for family, _ in rtg.AUDIT_STEPS}
REUSE_FAMILIES = REPORT_FAMILIES | {"pytest_stack_readers"}
STAMP_RELPATH = Path("artifacts") / "_temp" / "broad_gate_stamp.json"
LANE_TEST = "artifacts/etudes/papers/demo/test_demo.py"
STACK_PATHS = (
    "artifacts/todo_stack.md",
    "artifacts/open_stack.md",
    "artifacts/state.md",
    "artifacts/memory.md",
    "artifacts/goal_stack.md",
    "artifacts/fact_stack.md",
)
LEGS = ("3.11", "3.14")

TREE_FILES = {
    "jaxfne/__init__.py": "VALUE = 1\n",
    "tests/test_a.py": "def test_a():\n    assert True\n",
    "tests/test_stack_reader.py": 'STACK = "artifacts/open_stack.md"\n\n\ndef test_stack_reader():\n    assert True\n',
    "scripts/tool.py": "print('tool')\n",
    "examples/ex.py": "print('example')\n",
    "docs/page.md": "# page\n",
    "dashboard/app.js": "console.log(1)\n",
    "artifacts/notes.md": "notes\n",
    "pyproject.toml": '[project]\nname = "fixture"\nversion = "0.0.0"\n',
    LANE_TEST: "def test_demo():\n    assert True\n",
    ".gitignore": "artifacts/_temp/\nartifacts/attestations/\n",
}
TREE_FILES.update({path: "# stack\n" for path in STACK_PATHS})

# One executed case per reusable family, against the fixed slow set and module map that fixed_suite pins.
# The notebook module is slow-marked as real collection reports it, and it counts as pytest_notebook.
FIXED_SLOW = frozenset({
    "tests/test_a.py::test_slow",
    "tests/test_suite_no1_notebook_execution.py::test_suite_no1_notebook_executes",
})
FIXED_MODULES = {
    "tests.test_a": "tests/test_a.py",
    "tests.test_suite_no1_notebook_execution": "tests/test_suite_no1_notebook_execution.py",
}
BROAD_CASE = '<testcase classname="tests.test_a" name="test_x"/>'
SLOW_CASE = '<testcase classname="tests.test_a" name="test_slow"/>'
NOTEBOOK_CASE = (
    '<testcase classname="tests.test_suite_no1_notebook_execution" name="test_suite_no1_notebook_executes"/>'
)
GREEN_JUNIT = (
    '<testsuites><testsuite name="pytest" tests="4">'
    + BROAD_CASE + SLOW_CASE + NOTEBOOK_CASE
    + '<testcase classname="tests.test_a" name="test_y">'
    '<skipped message="requires a POSIX shell"/></testcase>'
    "</testsuite></testsuites>"
)
RED_JUNIT = (
    '<testsuites><testsuite name="pytest" tests="1">'
    '<testcase classname="tests.test_a" name="test_x"><failure message="boom"/></testcase>'
    "</testsuite></testsuites>"
)
EMPTY_JUNIT = '<testsuites><testsuite name="pytest" tests="0"/></testsuites>'


class _Proc:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _write(path: Path, text: str) -> None:
    """Write LF bytes exactly (no platform newline translation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return res.stdout.strip()


def _commit(repo: Path, message: str, *paths: str) -> str:
    _git(repo, "add", "--", *paths)
    _git(
        repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-q", "-m", message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _commit_on_other_tree(repo: Path) -> str:
    """A commit whose tree differs from HEAD's. HEAD and the working tree end as they were."""
    start = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repo, "checkout", "-q", "-b", "_temp_fixture_other")
    _write(repo / "jaxfne" / "__init__.py", "VALUE = 2\n")
    sha = _commit(repo, "other tree", "jaxfne/__init__.py")
    _git(repo, "checkout", "-q", start)
    return sha


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel, text in TREE_FILES.items():
        _write(root / rel, text)
    _git(root, "init", "-q")
    _commit(root, "fixture", *TREE_FILES)
    monkeypatch.setattr(rtg, "ROOT", root)
    return root


class _Calls(list):
    """Family of each command the gates ran, in order (None when unfamilied); ``cmds`` holds argv."""

    def __init__(self):
        super().__init__()
        self.cmds: list[list[str]] = []

    def clear(self):
        super().clear()
        self.cmds.clear()


@pytest.fixture
def gates(repo, monkeypatch, tmp_path):
    """Route every gate command to a recorder. Yields a _Calls list."""
    calls = _Calls()

    def fake_run(cmd, *, cwd=None, env=None, family=None):
        calls.append(family)
        calls.cmds.append(list(cmd))
        if family is not None:
            now = rtg._utcnow()
            rtg.record_execution(family, list(cmd), now, now, 0)
        if cmd[1:3] == ["-m", "build"]:
            _write(Path(cmd[cmd.index("--outdir") + 1]) / "fixture-0.0.0-py3-none-any.whl", "wheel")

    def fake_lanes(lanes=None):
        calls.append("pytest_lanes")
        calls.cmds.append(["fake-lanes"])
        now = rtg._utcnow()
        rtg.record_execution("pytest_lanes", ["fake-lanes"], now, now, 0)

    monkeypatch.setattr(rtg, "_run", fake_run)
    monkeypatch.setattr(rtg, "gate_pytest_lanes", fake_lanes)
    monkeypatch.setenv("JAXFNE_RC_ATTESTATION", str(tmp_path / "rc_gate_attestation.json"))
    rtg.reset_observations()
    yield calls
    rtg.reset_observations()


def _families(calls: list) -> set[str]:
    return {f for f in calls if f is not None}


def _broad(gates: list, *argv: str) -> set[str]:
    """Run the broad CLI and return the families it executed."""
    gates.clear()
    assert rtg.main(["broad", *argv]) == 0
    return _families(gates)


# --- D: the code-tree stamp ---------------------------------------------------------


def test_first_broad_run_is_full_and_writes_the_stamp(repo, gates):
    ran = _broad(gates)
    assert {"pytest_broad", "pytest_lanes"} <= ran
    stamp = json.loads((repo / STAMP_RELPATH).read_text(encoding="utf-8"))
    assert stamp["schema"] == rtg.BROAD_STAMP_SCHEMA
    assert stamp["key"] == rtg._code_tree_key()


def test_matching_stamp_runs_the_report_audits_and_the_stack_readers(repo, gates, capsys):
    _broad(gates)
    assert (repo / STAMP_RELPATH).exists(), "fixture must write a stamp first"
    key = rtg._code_tree_key()
    ran = _broad(gates)
    assert ran == REUSE_FAMILIES, "a matching stamp runs the report audits and the stack readers only"
    assert not ran & {"compileall", "pytest_broad", "pytest_lanes"}
    assert f"code-tree key {key} matches" in capsys.readouterr().out


def test_stack_readers_are_found_at_run_time_by_git_grep(repo):
    assert rtg._stack_reader_tests() == ["tests/test_stack_reader.py"]


def test_key_excludes_exactly_the_six_state_files():
    """The pathspec is ``.`` minus the six state files; dropping the exclude list fails here."""
    assert set(rtg.STAMP_STATE_FILES) == set(STACK_PATHS)
    assert set(rtg._stamp_pathspec()) == {".", *(f":(exclude){path}" for path in STACK_PATHS)}


def _stack_open_uncommitted(repo: Path, monkeypatch) -> None:
    _write(repo / "artifacts" / "open_stack.md", "# open stack\nchanged\n")


def _stack_todo_committed(repo: Path, monkeypatch) -> None:
    _write(repo / "artifacts" / "todo_stack.md", "- T-1 changed\n")
    _commit(repo, "todo", "artifacts/todo_stack.md")


def _stack_all_six_committed(repo: Path, monkeypatch) -> None:
    for path in STACK_PATHS:
        _write(repo / path, f"# {path} changed\n")
    _commit(repo, "stacks", *STACK_PATHS)


STACK_CHANGES = {
    "open_stack_uncommitted": _stack_open_uncommitted,
    "todo_stack_committed": _stack_todo_committed,
    "all_six_committed": _stack_all_six_committed,
}


@pytest.mark.parametrize("change", list(STACK_CHANGES))
def test_stack_only_change_reuses_the_stamp_and_runs_the_stack_readers(repo, gates, monkeypatch, change):
    _broad(gates)
    assert (repo / STAMP_RELPATH).exists(), "fixture must write a stamp first"
    key = rtg._code_tree_key()
    STACK_CHANGES[change](repo, monkeypatch)
    assert rtg._code_tree_key() == key, "a state-file edit must not change the code-tree key"
    ran = _broad(gates)
    assert ran == REUSE_FAMILIES, f"{change}: the stack readers must run in place of the full sweep"
    cmd = next(c for c in gates.cmds if "tests/test_stack_reader.py" in c)
    assert cmd[1:3] == ["-m", "pytest"] and "tests" not in cmd


def _change_committed_blob(repo: Path, monkeypatch) -> None:
    _write(repo / "jaxfne" / "__init__.py", "VALUE = 2\n")
    _commit(repo, "change", "jaxfne/__init__.py")


def _edit_unstaged(repo: Path, monkeypatch) -> None:
    _write(repo / "scripts" / "tool.py", "print('changed')\n")


def _add_untracked(repo: Path, monkeypatch) -> None:
    _write(repo / "tests" / "test_new.py", "def test_new():\n    assert True\n")


def _change_lane_test(repo: Path, monkeypatch) -> None:
    _write(repo / LANE_TEST, "def test_demo():\n    assert False\n")
    _commit(repo, "lane change", LANE_TEST)


def _change_docs_page(repo: Path, monkeypatch) -> None:
    _write(repo / "docs" / "page.md", "# page changed\n")
    _commit(repo, "docs", "docs/page.md")


def _edit_dashboard_unstaged(repo: Path, monkeypatch) -> None:
    _write(repo / "dashboard" / "app.js", "console.log(2)\n")


def _change_artifact_note(repo: Path, monkeypatch) -> None:
    _write(repo / "artifacts" / "notes.md", "notes changed\n")
    _commit(repo, "note", "artifacts/notes.md")


def _add_same_name_outside_the_list(repo: Path, monkeypatch) -> None:
    """A tracked file with a state file's basename but another path is not excluded."""
    _write(repo / "artifacts" / "archive" / "open_stack.md", "old\n")
    _commit(repo, "archive", "artifacts/archive/open_stack.md")


def _other_interpreter(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(rtg, "_interpreter_id", lambda: "Python 9.9.9\n/elsewhere/python")


def _other_dependency_version(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(rtg, "_installed_versions", lambda: "jax==9.9.9")


CODE_TREE_CHANGES = {
    "committed_blob": _change_committed_blob,
    "unstaged_edit": _edit_unstaged,
    "untracked_file": _add_untracked,
    "lane_test_committed": _change_lane_test,
    "docs_page_committed": _change_docs_page,
    "dashboard_unstaged": _edit_dashboard_unstaged,
    "artifact_note_committed": _change_artifact_note,
    "same_name_outside_the_list": _add_same_name_outside_the_list,
    "other_interpreter": _other_interpreter,
    "other_dependency_version": _other_dependency_version,
}


@pytest.mark.parametrize("change", list(CODE_TREE_CHANGES))
def test_any_other_tracked_change_forces_the_full_run(repo, gates, monkeypatch, change):
    _broad(gates)
    assert (repo / STAMP_RELPATH).exists(), "fixture must write a stamp first"
    CODE_TREE_CHANGES[change](repo, monkeypatch)
    ran = _broad(gates)
    assert {"compileall", "pytest_broad", "pytest_lanes"} <= ran, f"{change} must run everything"


def test_empty_stack_reader_set_runs_the_full_broad_gate(repo, gates, monkeypatch, capsys):
    _broad(gates)
    assert (repo / STAMP_RELPATH).exists(), "fixture must write a stamp first"
    monkeypatch.setattr(rtg, "_stack_reader_tests", lambda: [])
    ran = _broad(gates)
    assert {"compileall", "pytest_broad", "pytest_lanes"} <= ran
    assert "pytest_stack_readers" not in ran
    assert "no tracked test file names a stack file" in capsys.readouterr().out


def test_no_stamp_flag_forces_the_full_run_even_when_the_stamp_matches(repo, gates):
    _broad(gates)
    assert (repo / STAMP_RELPATH).exists(), "fixture must write a stamp first"
    ran = _broad(gates, "--no-stamp")
    assert {"compileall", "pytest_broad", "pytest_lanes"} <= ran


@pytest.mark.parametrize("damage", ["missing", "garbage", "wrong-schema", "wrong-key"])
def test_damaged_or_missing_stamp_runs_everything(repo, gates, damage):
    _broad(gates)
    stamp = repo / STAMP_RELPATH
    assert stamp.exists(), "fixture must write a stamp first"
    if damage == "missing":
        stamp.unlink()
    elif damage == "garbage":
        _write(stamp, "{not json")
    else:
        data = json.loads(stamp.read_text(encoding="utf-8"))
        if damage == "wrong-schema":
            data["schema"] = "jaxfne.some_other_stamp.v9"
        else:
            data["key"] = "0" * 64
        _write(stamp, json.dumps(data))
    ran = _broad(gates)
    assert "pytest_broad" in ran, f"a {damage} stamp must not be trusted"


def test_no_git_answer_runs_everything_and_writes_no_stamp(repo, gates, monkeypatch):
    _broad(gates)
    stamp = repo / STAMP_RELPATH
    assert stamp.exists(), "fixture must write a stamp first"
    stamp.unlink()
    monkeypatch.setattr(rtg, "_code_tree_key", lambda: None)
    ran = _broad(gates)
    assert "pytest_broad" in ran
    assert not stamp.exists(), "without a key there is nothing to stamp"


def test_dirty_tree_runs_everything_and_writes_no_stamp(repo, gates):
    _broad(gates)
    stamp = repo / STAMP_RELPATH
    assert stamp.exists(), "fixture must write a stamp first"
    stamp.unlink()
    _add_untracked(repo, None)
    ran = _broad(gates)
    assert "pytest_broad" in ran
    assert not stamp.exists(), "a run that starts in a dirty tree must not write the stamp"


def test_direct_gate_calls_never_write_the_stamp(repo, gates):
    _broad(gates)
    stamp = repo / STAMP_RELPATH
    stamp.unlink()  # the CLI wrote it; a direct call must not
    rtg.gate_broad()
    assert not stamp.exists()


def test_direct_gate_calls_never_read_the_stamp(repo, gates):
    _broad(gates)
    assert (repo / STAMP_RELPATH).exists(), "fixture must write a stamp first"
    gates.clear()
    rtg.gate_broad()
    assert {"compileall", "pytest_broad", "pytest_lanes"} <= _families(gates)


def test_no_stamp_is_refused_outside_the_broad_gate(gates, capsys):
    with pytest.raises(SystemExit):
        rtg.main(["dev", "--no-stamp"])
    assert "applies to the broad gate only" in capsys.readouterr().err
    assert gates == []


# --- F: Release CI reuse for the rc gate ---------------------------------------------


@pytest.fixture
def fixed_suite(monkeypatch):
    """Pin the slow node ids and the test-module map that the JUnit family check reads."""
    monkeypatch.setattr(rtg, "_slow_node_ids", lambda: FIXED_SLOW, raising=False)
    monkeypatch.setattr(rtg, "_test_modules", lambda: dict(FIXED_MODULES), raising=False)


@pytest.fixture
def release_ci(monkeypatch, fixed_suite):
    """Answer gh for one Release CI run. The test sets the run list and the JUnit it downloads."""
    state = {"runs": [], "legs": LEGS, "junit": GREEN_JUNIT, "download_rc": 0, "error": None}

    def fake_gh(*args):
        if state["error"] is not None:
            raise state["error"]
        if args[:2] == ("run", "list"):
            return _Proc(0, json.dumps(state["runs"]))
        assert args[:2] == ("run", "download"), args
        if state["download_rc"]:
            return _Proc(state["download_rc"], "", "refused")
        dest = Path(args[args.index("--dir") + 1])
        for leg in state["legs"]:
            _write(dest / f"pytest-results-{leg}" / f"pytest-results-{leg}.xml", state["junit"])
        return _Proc(0)

    monkeypatch.setattr(rtg, "_gh", fake_gh)
    return state


def _release_run(repo: Path, **fields) -> dict:
    run = {
        "databaseId": 12345,
        "headSha": _git(repo, "rev-parse", "HEAD"),
        "headBranch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "createdAt": "2026-10-27T01:00:00Z",
        "updatedAt": "2026-10-27T02:00:00Z",
    }
    run.update(fields)
    return run


def _assert_full_rc(gates: list, capsys, reason: str) -> None:
    ran = _families(gates)
    assert {"pytest_broad", "pytest_slow", "pytest_notebook", "junit_parity"} <= ran, (
        "a fallback must run the full rc, including the RC-vs-CI parity check"
    )
    assert reason in capsys.readouterr().out
    att = json.loads(rtg.attestation_path().read_text(encoding="utf-8"))
    assert att["release_ci_reuse"] is None
    assert all("reused_from" not in rec for rec in att["families"])
    assert verify_pre_release_gate_receipt(att["commit_sha"]) == []


def test_same_tree_passing_run_reuses_the_three_pytest_sweeps(repo, gates, release_ci):
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    release_ci["runs"] = [_release_run(repo)]

    assert rtg.main(["rc"]) == 0

    ran = _families(gates)
    assert not ran & set(rtg.RELEASE_CI_REUSABLE_FAMILIES), "reused families were rerun locally"
    assert {
        "environment_parity", "pytest_lanes", "examples_smoke", "docs_build_strict",
        "package_build", "twine_check", "isolated_wheel_smoke",
    } <= ran
    assert "junit_parity" not in ran
    att = json.loads(rtg.attestation_path().read_text(encoding="utf-8"))
    reuse = att["release_ci_reuse"]
    assert (reuse["run_id"], reuse["tree_sha"], reuse["head_sha"]) == ("12345", tree, head)
    assert reuse["families"] == sorted(rtg.RELEASE_CI_REUSABLE_FAMILIES)
    assert reuse["junit"] == ["pytest-results-3.11.xml", "pytest-results-3.14.xml"]
    assert reuse["skipped_locally"] == ["junit_parity"]
    by_family = {rec["family"]: rec for rec in att["families"]}
    for family in rtg.RELEASE_CI_REUSABLE_FAMILIES:
        assert by_family[family]["reused_from"]["run_id"] == "12345"
        assert by_family[family]["status"] == "PASS"
    assert att["status"] == "PASS"
    assert verify_pre_release_gate_receipt(head, expected_tree_sha=tree) == []


def test_run_on_another_tree_falls_back_to_the_full_rc(repo, gates, release_ci, capsys):
    other = _commit_on_other_tree(repo)
    release_ci["runs"] = [_release_run(repo, headSha=other)]
    assert rtg.main(["rc"]) == 0
    _assert_full_rc(gates, capsys, "differs from HEAD^{tree}")


@pytest.mark.parametrize(
    "fields, junit, legs, download_rc, reason",
    [
        ({"conclusion": "failure"}, GREEN_JUNIT, LEGS, 0, "not 'success'"),
        ({"status": "in_progress", "conclusion": ""}, GREEN_JUNIT, LEGS, 0, "not 'success'"),
        ({"event": "pull_request", "headBranch": "feature"}, GREEN_JUNIT, LEGS, 0, "is a pull_request run"),
        ({"event": "push", "headBranch": "dev"}, GREEN_JUNIT, LEGS, 0, "is a push run"),
        ({}, "not xml at all", LEGS, 0, "cannot be read"),
        ({}, GREEN_JUNIT, ("3.11",), 0, "JUnit missing for"),
        ({}, RED_JUNIT, LEGS, 0, "failure(s) or error(s)"),
        ({}, EMPTY_JUNIT, LEGS, 0, "holds no test cases"),
        ({}, GREEN_JUNIT, LEGS, 1, "gh run download failed"),
    ],
    ids=[
        "run-failed", "run-not-completed", "pull-request-run", "push-to-dev-run",
        "junit-unreadable", "junit-leg-missing", "junit-has-failure", "junit-empty",
        "download-refused",
    ],
)
def test_run_that_cannot_stand_in_falls_back_to_the_full_rc(
    repo, gates, release_ci, capsys, fields, junit, legs, download_rc, reason
):
    release_ci["runs"] = [_release_run(repo, **fields)]
    release_ci["junit"], release_ci["legs"], release_ci["download_rc"] = junit, legs, download_rc
    assert rtg.main(["rc"]) == 0
    _assert_full_rc(gates, capsys, reason)


def test_no_successful_run_falls_back_to_the_full_rc(repo, gates, release_ci, capsys):
    release_ci["runs"] = []
    assert rtg.main(["rc"]) == 0
    _assert_full_rc(gates, capsys, "no successful release_ci.yml run")


def test_gh_unavailable_falls_back_to_the_full_rc(repo, gates, release_ci, capsys):
    release_ci["error"] = OSError("gh not found")
    assert rtg.main(["rc"]) == 0
    _assert_full_rc(gates, capsys, "gh is unavailable")


@pytest.mark.parametrize(
    "event, branch, expected",
    [
        ("push", "main", True),
        ("schedule", "main", True),
        ("workflow_dispatch", "feature", True),
        ("pull_request", "main", False),
        ("push", "dev", False),
    ],
)
def test_only_the_unfiltered_release_ci_runs_stand_in(event, branch, expected):
    assert rtg._runs_unfiltered_sweep({"event": event, "headBranch": branch}) is expected


def test_reused_legs_are_the_release_ci_matrix():
    text = (ROOT / ".github" / "workflows" / "release_ci.yml").read_text(encoding="utf-8")
    matrices = re.findall(r"python-version:\s*\[([^\]]*)\]", text)
    assert matrices, "release_ci.yml has no python-version matrix"
    for body in matrices:
        assert tuple(v.strip().strip("'\"") for v in body.split(",")) == rtg.RELEASE_CI_LEGS


def test_reusable_families_are_release_ci_pytest_sweeps_only():
    assert set(rtg.RELEASE_CI_REUSABLE_FAMILIES) <= rtg.RELEASE_CI_GATE_FAMILIES
    assert "pytest_lanes" not in rtg.RELEASE_CI_REUSABLE_FAMILIES, "Release CI runs no paper lanes"
    assert "junit_parity" not in rtg.RELEASE_CI_REUSABLE_FAMILIES


# --- F: the JUnit family check ---------------------------------------------------------


def _leg_junit(tmp_path: Path, body: str) -> list[Path]:
    """One JUnit per matrix leg, every leg holding ``body``."""
    paths = []
    for leg in LEGS:
        path = tmp_path / f"pytest-results-{leg}.xml"
        _write(path, '<testsuites><testsuite name="pytest">' + body + "</testsuite></testsuites>")
        paths.append(path)
    return paths


def _skipped(case: str) -> str:
    return case.replace("/>", '><skipped message="no kernel"/></testcase>')


@pytest.mark.parametrize(
    "node_id, classname, name",
    [
        ("tests/test_x.py::test_y", "tests.test_x", "test_y"),
        ("tests/test_x.py::test_y[p]", "tests.test_x", "test_y[p]"),
        ("tests/test_x.py::test_y[a.b]", "tests.test_x", "test_y[a.b]"),
        ("tests/test_x.py::test_y[c::d]", "tests.test_x", "test_y[c::d]"),
        ("tests/test_x.py::TestC::test_y", "tests.test_x.TestC", "test_y"),
        ("tests/test_x.py::TestC::test_y[p]", "tests.test_x.TestC", "test_y[p]"),
        ("tests/test_x.py::TestO::TestI::test_d", "tests.test_x.TestO.TestI", "test_d"),
    ],
)
def test_node_id_and_junit_key_are_inverse(node_id, classname, name):
    modules = {"tests.test_x": "tests/test_x.py"}
    assert rtg._junit_key(node_id) == (classname, name)
    assert rtg._node_id_from_junit(classname, name, modules) == node_id


def test_junit_classname_with_no_test_module_maps_to_nothing():
    modules = {"tests.test_x": "tests/test_x.py"}
    assert rtg._node_id_from_junit("tests.other", "test_y", modules) is None
    assert rtg._node_id_from_junit("docs.page", "test_z", modules) is None


@pytest.mark.parametrize("family", ["pytest_broad", "pytest_slow", "pytest_notebook"])
def test_family_with_only_skipped_cases_is_not_reused(tmp_path, fixed_suite, family):
    # The notebook case is in the slow set too: with only the slow case skipped, it must still count as
    # pytest_notebook, so pytest_slow stays unexecuted.
    cases = {"pytest_broad": BROAD_CASE, "pytest_slow": SLOW_CASE, "pytest_notebook": NOTEBOOK_CASE}
    body = "".join(_skipped(xml) if name == family else xml for name, xml in cases.items())
    assert "<skipped" in body, "fixture must skip the family's case"
    problem = rtg._release_ci_junit_problem(_leg_junit(tmp_path, body))
    assert problem is not None, f"a {family} family with no executed case must not stand in"
    assert "pytest-results-3.11.xml" in problem and f"no executed {family} case" in problem


def test_one_executed_case_per_family_stands_in(tmp_path, fixed_suite):
    body = BROAD_CASE + SLOW_CASE + NOTEBOOK_CASE
    assert rtg._release_ci_junit_problem(_leg_junit(tmp_path, body)) is None


def test_executed_case_outside_every_test_module_is_not_reused(tmp_path, fixed_suite):
    stray = '<testcase classname="docs.page" name="test_z"/>'
    body = BROAD_CASE + SLOW_CASE + NOTEBOOK_CASE + stray
    problem = rtg._release_ci_junit_problem(_leg_junit(tmp_path, body))
    assert problem is not None and "maps to no test module" in problem


def test_failed_slow_collection_is_not_reused(tmp_path, fixed_suite, monkeypatch):
    def collection_failed():
        raise RuntimeError("slow-test collection exited 2")

    monkeypatch.setattr(rtg, "_slow_node_ids", collection_failed)
    problem = rtg._release_ci_junit_problem(_leg_junit(tmp_path, BROAD_CASE + SLOW_CASE + NOTEBOOK_CASE))
    assert problem is not None and "slow-test node ids unavailable" in problem
