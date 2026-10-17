"""Changelog fragments: one file per change, assembled into ``docs/reference/changelog.md``.

Two claims carry the design, and each has a test built from its own case:

* two changes made on parallel branches merge without conflict, where two edits under
  ``## [Unreleased]`` do not -- the control is the point, since a merge test that cannot fail
  says nothing about the layout;
* the file's mixed CRLF/LF endings survive assembly byte for byte: the tools read
  ``docs/reference/changelog.md`` as bytes and insert without changing any existing byte.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.assemble_changelog import (
    CATEGORIES,
    FRAGMENT_DIR,
    FragmentError,
    assemble,
    main,
    read_fragments,
    render_section,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_default_fragment_directory_is_the_one_directory():
    # read_fragments returns [] for a missing directory, so a stale default would leave every
    # fragment unassembled without an error. The path is asserted, not inferred from the gate.
    assert FRAGMENT_DIR == ROOT / "artifacts" / "changelog.d"
    assert FRAGMENT_DIR.is_dir(), f"{FRAGMENT_DIR} is not a directory"


def _changelog(unreleased_body: bytes = b"\n", older: bytes = b"") -> bytes:
    return b"# Changelog\n\n## [Unreleased]\n" + unreleased_body + older


OLDER = b"## v0.1.0 (2026-09-01)\n\n### Added\n\n- old\n"


def _write(directory: Path, name: str, body: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(body)
    return path


def test_assembly_inserts_a_dated_section_below_unreleased(tmp_path):
    changelog = _changelog(older=OLDER)
    assert b"## [Unreleased]\n" in changelog, "fixture needs the marker line"
    assert b"## v0.1.1" not in changelog, "fixture must not already carry the version"
    fragments_dir = tmp_path / "artifacts" / "changelog.d"
    _write(fragments_dir, "m.added.md", b"- new thing\n")
    _write(fragments_dir, "a.fixed.md", b"- a fix\n")
    fragments = read_fragments(fragments_dir)
    assert [p.name for _, _, p in fragments] == ["m.added.md", "a.fixed.md"]

    assembled = assemble(changelog, "0.1.1", "2026-09-02", fragments)

    assert assembled == (
        b"# Changelog\n\n## [Unreleased]\n"
        b"\n## v0.1.1 (2026-09-02)\n\n### Added\n\n- new thing\n\n### Fixed\n\n- a fix\n\n"
        + OLDER
    )


def test_an_empty_fragment_directory_is_refused(tmp_path):
    changelog = _changelog(older=OLDER)
    assert list(tmp_path.iterdir()) == [], "fixture must hold no fragments"
    with pytest.raises(FragmentError, match="no fragments"):
        assemble(changelog, "0.1.1", "2026-09-02", read_fragments(tmp_path))


def test_existing_bytes_are_kept_byte_for_byte_including_crlf(tmp_path):
    crlf = (b"# Changelog\r\n\r\n## [Unreleased]\r\n\r\n"
            b"## v0.1.0 (2026-09-01)\r\n\r\n### Added\n\r\n- old\r\n")
    assert crlf.count(b"\r") > 0, "fixture must carry CRLF lines"
    assert b"### Added\n" in crlf, "fixture must mix in an LF-only line"
    fragments_dir = tmp_path / "artifacts" / "changelog.d"
    _write(fragments_dir, "a.added.md", b"- new\n")
    fragments = read_fragments(fragments_dir)
    section = render_section("0.1.1", "2026-09-02", fragments).encode("utf-8")

    assembled = assemble(crlf, "0.1.1", "2026-09-02", fragments)

    marker_end = crlf.index(b"## [Unreleased]") + len(b"## [Unreleased]\r\n")
    head, tail = crlf[:marker_end], crlf[marker_end:]
    cut = tail.find(b"\n## ")
    gap, rest = tail[:cut + 1], tail[cut + 1:]
    assert assembled == head + gap + section + rest
    assert b"\r" in gap and b"\r" in rest, "fixture must bracket the insertion with CRLF"
    assert assembled.count(section) == 1, "the section is inserted exactly once"
    assert assembled.replace(section, b"", 1) == crlf, "every original byte survives"


def test_a_released_version_is_refused(tmp_path):
    changelog = _changelog(older=OLDER)
    assert b"## v0.1.0 (2026-09-01)" in changelog, "fixture must carry the released section"
    with pytest.raises(FragmentError, match="already has a section"):
        assemble(changelog, "0.1.0", "2026-09-02", [("added", "- x\n", Path("x"))])


def test_a_version_prefix_of_a_released_version_is_not_a_match():
    changelog = _changelog(older=b"\n## v0.1.0.1 (2026-09-01)\n\n### Added\n\n- old\n")
    assembled = assemble(changelog, "0.1.0", "2026-09-02", [("added", "- x\n", Path("x"))])
    assert b"## v0.1.0 (2026-09-02)" in assembled


def test_hand_written_unreleased_lines_are_refused():
    changelog = _changelog(unreleased_body=b"\n- pending\n", older=OLDER)
    assert b"- pending" in changelog, "fixture must carry the hand-written line"
    with pytest.raises(FragmentError, match="hand-written"):
        assemble(changelog, "0.1.1", "2026-09-02", [("added", "- x\n", Path("x"))])


def test_a_missing_unreleased_line_is_refused():
    changelog = b"# Changelog\n\n" + OLDER
    assert b"[Unreleased]" not in changelog, "fixture must lack the marker"
    with pytest.raises(FragmentError, match="no '## \\[Unreleased\\]'"):
        assemble(changelog, "0.1.1", "2026-09-02", [("added", "- x\n", Path("x"))])


@pytest.mark.parametrize("name, body, message", [
    ("note.md", b"- x\n", "is named"),
    ("note.improved.md", b"- x\n", "unknown category"),
    ("note.added.md", b"", "empty"),
    ("note.added.md", b"plain prose\n", "bullet"),
    ("note.added.md", b"- x\n\n### Fixed\n", "headings"),
    ("note.added.md", b"- x\r\n", "LF"),
])
def test_a_malformed_fragment_raises_rather_than_being_skipped(tmp_path, name, body, message):
    _write(tmp_path, name, body)
    with pytest.raises(FragmentError, match=message):
        read_fragments(tmp_path)


def test_fragment_categories_are_the_keep_a_changelog_headings_the_file_uses():
    assert set(CATEGORIES) == {"added", "changed", "deprecated", "removed", "fixed"}
    assert list(CATEGORIES.values()) == ["Added", "Changed", "Deprecated", "Removed", "Fixed"]


_GIT_ENV = {k: v for k, v in os.environ.items()
            if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")}


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=jaxfne-test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=", "-c", "core.autocrlf=false", *args],
        capture_output=True, text=True, env=_GIT_ENV,
    )


def _two_branches(repo: Path, change_a, change_b) -> subprocess.CompletedProcess:
    """Apply ``change_a`` and ``change_b`` on two branches off one commit; merge b into a."""
    assert _git(repo.parent, "init", "-q", str(repo)).returncode == 0
    (repo / "docs" / "reference").mkdir(parents=True)
    (repo / "docs" / "reference" / "changelog.md").write_bytes(
        b"# Changelog\n\n## [Unreleased]\n\n## v0.1.0 (2026-09-01)\n\n### Added\n\n- old\n")
    for step in (("add", "docs/reference/changelog.md"), ("commit", "-q", "-m", "base"),
                 ("branch", "-M", "base"), ("checkout", "-q", "-b", "b")):
        assert _git(repo, *step).returncode == 0, step
    change_b(repo)
    assert _git(repo, "add", "-A").returncode == 0
    assert _git(repo, "commit", "-q", "-m", "b").returncode == 0
    assert _git(repo, "checkout", "-q", "-b", "a", "base").returncode == 0
    change_a(repo)
    assert _git(repo, "add", "-A").returncode == 0
    assert _git(repo, "commit", "-q", "-m", "a").returncode == 0
    return _git(repo, "merge", "--no-edit", "b")


def _fragment(name, text):
    def write(repo: Path) -> None:
        (repo / "artifacts" / "changelog.d").mkdir(parents=True, exist_ok=True)
        (repo / "artifacts" / "changelog.d" / name).write_bytes(text.encode("utf-8"))
    return write


def _under_unreleased(text):
    def write(repo: Path) -> None:
        path = repo / "docs" / "reference" / "changelog.md"
        old = path.read_bytes().decode("utf-8")
        path.write_bytes(old.replace("## [Unreleased]\n", f"## [Unreleased]\n\n{text}\n", 1)
                         .encode("utf-8"))
    return write


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@needs_git
def test_two_parallel_fragments_merge_without_conflict(tmp_path):
    repo = tmp_path / "repo"
    merged = _two_branches(repo, _fragment("gamma.added.md", "- gamma\n"),
                           _fragment("delta.fixed.md", "- delta\n"))
    assert merged.returncode == 0, merged.stdout + merged.stderr

    changelog = (repo / "docs" / "reference" / "changelog.md").read_bytes()
    assembled = assemble(changelog, "0.1.1", "2026-09-02",
                         read_fragments(repo / "artifacts" / "changelog.d"))
    text = assembled.decode("utf-8")
    assert "### Added\n\n- gamma\n\n### Fixed\n\n- delta\n" in text
    assert "## v0.1.0 (2026-09-01)" in text


@needs_git
def test_the_same_two_changes_written_under_unreleased_conflict(tmp_path):
    """The control: without fragments, the same parallel work collides."""
    merged = _two_branches(tmp_path / "repo", _under_unreleased("- gamma"),
                           _under_unreleased("- delta"))
    assert merged.returncode != 0
    assert "CONFLICT" in merged.stdout + merged.stderr


def test_the_command_writes_the_section_and_consumes_its_fragments(tmp_path, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    original = b"# Changelog\n\n## [Unreleased]\n\n" + OLDER
    changelog.write_bytes(original)
    fragments = tmp_path / "artifacts" / "changelog.d"
    _write(fragments, "a.added.md", b"- new\n")
    _write(fragments, "README.md", b"How to write a fragment.\n")
    common = ["--version", "0.1.1", "--date", "2026-09-02",
              "--changelog", str(changelog), "--fragments", str(fragments)]

    assert main([*common, "--dry-run"]) == 0
    assert changelog.read_bytes() == original, "a dry run changes nothing"
    assert (fragments / "a.added.md").exists()
    assert capsys.readouterr().out == "## v0.1.1 (2026-09-02)\n\n### Added\n\n- new\n\n"

    assert main(common) == 0
    assert changelog.read_bytes() == (
        b"# Changelog\n\n## [Unreleased]\n\n## v0.1.1 (2026-09-02)\n\n### Added\n\n- new\n\n"
        + OLDER)
    assert sorted(p.name for p in fragments.iterdir()) == ["README.md"]


def test_the_committed_fragments_render():
    """A malformed fragment fails here, on the commit that adds it, not at release."""
    fragments = read_fragments(ROOT / "artifacts" / "changelog.d")
    for _, _, path in fragments:
        assert path.parent == ROOT / "artifacts" / "changelog.d"
    if fragments:
        render_section("999.0.0", "2099-01-01", fragments)
