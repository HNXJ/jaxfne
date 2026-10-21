"""The release notes ``scripts/release_body.py`` writes, and the check that refuses them.

Each refusal is driven here: a check that cannot refuse would let a release publish notes
that disagree with ``pyproject.toml`` or quote a changelog section that does not exist.
Every fixture builds the tree the case is named after and asserts its precondition.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from scripts.release_body import build_body, changelog_section, release_metadata

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release_body.py"


def _tree(tmp_path: pathlib.Path, version: str, changelog: str) -> pathlib.Path:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "widget-lib"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.9"\n'
        "classifiers = [\n"
        '    "Programming Language :: Python :: 3.9",\n'
        '    "Programming Language :: Python :: 3.10",\n'
        '    "Programming Language :: Python :: 3.11",\n'
        "]\n"
        "[project.optional-dependencies]\n"
        'alpha = ["numpy"]\n'
        'beta = ["scipy"]\n',
        encoding="utf-8",
    )
    (tmp_path / "docs" / "reference").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "reference" / "changelog.md").write_text(changelog, encoding="utf-8")
    return tmp_path


CHANGELOG = (
    "# Changelog\n\n"
    "## [Unreleased]\n\n"
    "## v7.1.0 (2026-10-01)\n\n"
    "### Fixed\n\n- A widget no longer wobbles.\n\n"
    "## v7.0.0 (2026-09-01)\n\n"
    "### Added\n\n- Widgets.\n"
)


def test_the_body_tables_state_what_the_tree_declares(tmp_path):
    root = _tree(tmp_path, "7.1.0", CHANGELOG)
    assert 'version = "7.1.0"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "## v7.1.0 (2026-10-01)" in CHANGELOG, "fixture must carry the version section"
    body, violations = build_body(root, "v7.1.0")

    assert violations == [] and body is not None
    assert "| package | widget-lib |" in body, body
    assert "| version | 7.1.0 |" in body, body
    assert "| python | >=3.9 (3.9, 3.10, 3.11) |" in body, body
    assert "| extras | alpha, beta |" in body, body
    assert "`pip install widget-lib==7.1.0`" in body, body
    assert "## v7.1.0 (2026-10-01)" in body, body
    assert "- A widget no longer wobbles." in body, body
    assert "7.0.0" not in body and "Unreleased" not in body, body


def test_a_candidate_section_renders_but_fails_the_dated_check(tmp_path):
    root = _tree(tmp_path, "7.1.0", CHANGELOG.replace("## v7.1.0 (2026-10-01)", "## v7.1.0 (candidate)"))
    body, violations = build_body(root, "v7.1.0")
    assert violations == [] and body is not None, "the body still renders a candidate section"

    checked, failures = build_body(root, "v7.1.0", require_dated=True)
    assert checked is None
    assert len(failures) == 1 and "must be dated before release" in failures[0], failures
    assert "candidate" in failures[0], failures


def test_a_dated_section_passes_the_dated_check(tmp_path):
    root = _tree(tmp_path, "7.1.0", CHANGELOG)
    body, violations = build_body(root, "v7.1.0", require_dated=True)
    assert violations == [] and body is not None


def test_this_repositorys_body_matches_its_own_metadata():
    metadata = release_metadata(REPO_ROOT)
    assert metadata.name == "jaxfne", "fixture is the live repository"
    changelog = (REPO_ROOT / "docs" / "reference" / "changelog.md").read_text(encoding="utf-8")
    assert f"## v{metadata.version}" in changelog, "the live changelog must carry the version"
    body, violations = build_body(REPO_ROOT, f"v{metadata.version}")
    assert violations == [], violations
    assert f"pip install jaxfne=={metadata.version}" in body


def test_a_tag_naming_another_version_is_refused(tmp_path):
    root = _tree(tmp_path, "7.1.0", CHANGELOG)
    body, violations = build_body(root, "v7.2.0")
    assert body is None
    assert any("v7.2.0" in v and "7.1.0" in v for v in violations), violations


def test_a_version_without_a_changelog_section_is_refused(tmp_path):
    root = _tree(tmp_path, "7.2.0", CHANGELOG)
    assert "## v7.2.0" not in CHANGELOG, "fixture must lack the version section"
    body, violations = build_body(root, "v7.2.0")
    assert body is None
    assert violations == ["docs/reference/changelog.md has no non-empty section headed "
                          "'## v7.2.0 ...'"], violations


def test_an_empty_section_is_no_section():
    assert changelog_section("## v1.0.0 (2026-01-01)\n\n## v0.9.0 (2025-01-01)\n- x\n",
                             "1.0.0") is None
    assert changelog_section("## v1.0.0 (2026-01-01)\n- y\n", "1.0.0") == (
        "## v1.0.0 (2026-01-01)\n- y")
    # A version that is a prefix of another is not that other's section.
    assert changelog_section("## v1.0.0.1 (2026-01-01)\n- z\n", "1.0.0") is None


def test_the_check_never_touches_network_or_gh():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("subprocess", "urllib", "socket", "os.system", '"gh"', "'gh'"):
        assert token not in source, f"scripts/release_body.py must not use {token}"


class TestTheCommand:
    def _run(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                              text=True, timeout=120)

    def test_a_refused_body_exits_1_and_prints_no_body(self, tmp_path):
        root = _tree(tmp_path, "7.1.0", CHANGELOG)
        result = self._run("--tag", "v7.2.0", "--root", str(root))
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert "release body:" in result.stderr, result.stderr
        assert result.stdout == "", result.stdout

    def test_check_passes_on_a_matching_tag_and_fails_otherwise(self, tmp_path):
        root = _tree(tmp_path, "7.1.0", CHANGELOG)
        passing = self._run("--tag", "v7.1.0", "--check", "--root", str(root))
        assert passing.returncode == 0, (passing.stdout, passing.stderr)
        assert "PASS:" in passing.stdout, passing.stdout
        failing = self._run("--tag", "v7.2.0", "--check", "--root", str(root))
        assert failing.returncode == 1, (failing.stdout, failing.stderr)

    def test_check_refuses_a_candidate_section(self, tmp_path):
        root = _tree(tmp_path, "7.1.0",
                     CHANGELOG.replace("## v7.1.0 (2026-10-01)", "## v7.1.0 (candidate)"))
        assert "## v7.1.0 (candidate)" in (root / "docs" / "reference" / "changelog.md").read_text(encoding="utf-8")
        rendered = self._run("--tag", "v7.1.0", "--root", str(root))
        assert rendered.returncode == 0, (rendered.stdout, rendered.stderr)
        checked = self._run("--tag", "v7.1.0", "--check", "--root", str(root))
        assert checked.returncode == 1, (checked.stdout, checked.stderr)
        assert "must be dated before release" in checked.stderr, checked.stderr
        assert checked.stdout == "", checked.stdout

    def test_a_passing_body_is_written_with_lf_endings(self, tmp_path):
        root = _tree(tmp_path, "7.1.0", CHANGELOG)
        result = self._run("--tag", "v7.1.0", "--root", str(root))
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "\r" not in result.stdout
        assert result.stdout == build_body(root, "v7.1.0")[0]
