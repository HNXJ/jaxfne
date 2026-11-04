"""Tests for scripts/check_fidelity_fix.py (rule F3 of RESIDUAL_PROTOCOL.md).

Each test builds a tiny git repository in tmp_path, asserts its own precondition,
then runs the script on it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_fidelity_fix.py"

BUGGY_AREA = "def area(r):\n    return 3 * r\n"
FIXED_AREA = "def area(r):\n    return 3 * r * r\n"
TEST_FIX = "from pkg_mod import area\n\n\ndef test_fix():\n    assert area(2) == 12\n"
TEST_PASSES_ON_BUGGY = "from pkg_mod import area\n\n\ndef test_fix():\n    assert area(2) == 6\n"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return proc.stdout.decode("utf-8")


def _init(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "core.autocrlf", "false")
    return repo


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _commit(repo: Path, message: str, *paths: str) -> str:
    _git(repo, "add", *paths)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


def _base(tmp_path: Path) -> tuple[Path, str]:
    repo = _init(tmp_path)
    _write(repo, "pyproject.toml", "[project]\nname = 'fixture'\nversion = '0'\n")
    _write(repo, "pkg_mod.py", BUGGY_AREA)
    base = _commit(repo, "base: area is wrong", "pyproject.toml", "pkg_mod.py")
    return repo, base


def _run(repo: Path, rng: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--range", rng, "--repo", str(repo)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _exec_area(source: str):
    ns: dict = {}
    exec(source, ns)
    return ns["area"]


def test_real_fix_with_failing_test_is_accepted(tmp_path):
    repo, base = _base(tmp_path)
    assert _exec_area(BUGGY_AREA)(2) != 12  # the test fails on the parent code
    assert _exec_area(FIXED_AREA)(2) == 12  # and passes on the fixed code
    _write(repo, "pkg_mod.py", FIXED_AREA)
    _write(repo, "tests/test_m.py", TEST_FIX)
    fix = _commit(
        repo,
        "fix area\n\nFidelity-fix: tests/test_m.py::test_fix\n",
        "pkg_mod.py",
        "tests/test_m.py",
    )
    assert "Fidelity-fix: tests/test_m.py::test_fix" in _git(repo, "log", "-1", "--format=%B", fix)

    proc = _run(repo, f"{base}..{fix}")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERDICT: ACCEPT" in proc.stdout
    assert "# examined: 1 tagged commits of 1" in proc.stdout


def test_planted_fix_whose_test_passes_on_parent_is_revised(tmp_path):
    repo, base = _base(tmp_path)
    assert _exec_area(BUGGY_AREA)(2) == 6  # the planted test passes on the parent code
    _write(repo, "tests/test_m.py", TEST_PASSES_ON_BUGGY)
    planted = _commit(
        repo,
        "fix area\n\nFidelity-fix: tests/test_m.py::test_fix\n",
        "tests/test_m.py",
    )
    assert "Fidelity-fix: tests/test_m.py::test_fix" in _git(repo, "log", "-1", "--format=%B", planted)

    proc = _run(repo, f"{base}..{planted}")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "VERDICT: REVISE" in proc.stdout
    assert "passes on the parent" in proc.stdout
    assert "# examined: 1 tagged commits of 1" in proc.stdout


def test_tag_naming_absent_node_is_revised(tmp_path):
    repo, base = _base(tmp_path)
    assert "def test_missing" not in TEST_FIX  # the named node does not exist in the file
    _write(repo, "pkg_mod.py", FIXED_AREA)
    _write(repo, "tests/test_m.py", TEST_FIX)
    tagged = _commit(
        repo,
        "fix area\n\nFidelity-fix: tests/test_m.py::test_missing\n",
        "pkg_mod.py",
        "tests/test_m.py",
    )
    assert "Fidelity-fix: tests/test_m.py::test_missing" in _git(repo, "log", "-1", "--format=%B", tagged)

    proc = _run(repo, f"{base}..{tagged}")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "VERDICT: REVISE" in proc.stdout
    assert "test_missing" in proc.stdout
    assert "# examined: 1 tagged commits of 1" in proc.stdout


def test_untagged_commits_are_ignored(tmp_path):
    repo, base = _base(tmp_path)
    _write(repo, "pkg_mod.py", FIXED_AREA)
    first = _commit(repo, "fix area without a tag", "pkg_mod.py")
    _write(repo, "tests/test_m.py", TEST_FIX)
    second = _commit(repo, "add a test without a tag", "tests/test_m.py")
    for sha in (first, second):
        assert "Fidelity-fix:" not in _git(repo, "log", "-1", "--format=%B", sha)

    proc = _run(repo, f"{base}..{second}")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERDICT: ACCEPT" in proc.stdout
    assert "# examined: 0 tagged commits of 2" in proc.stdout


def test_bad_range_is_usage_error(tmp_path):
    repo, _ = _base(tmp_path)
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "-q", "no-such-ref^{commit}"],
        capture_output=True,
    )
    assert probe.returncode != 0  # the ref really is absent

    proc = _run(repo, "no-such-ref..HEAD")

    assert proc.returncode == 2, proc.stdout + proc.stderr
