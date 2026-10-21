"""A pure rename (``git mv``, R100) must not flag unchanged bytes.

Fail-first test: ``scripts/report_hygiene_check.py`` scans staged files, so a
staged pure rename is scanned even though its bytes are unchanged. The gate
must skip files whose staged blob equals their HEAD blob under the old name
(``git diff --cached --name-status -M``); a renamed file with edits is still
scanned.
"""

import subprocess
from pathlib import Path

from scripts.report_hygiene_check import (
    collect_scan_paths,
    pure_rename_targets,
    scan_file,
)


def _git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
        env={
            "GIT_AUTHOR_NAME": "hygiene-test",
            "GIT_AUTHOR_EMAIL": "hygiene-test@example.com",
            "GIT_COMMITTER_NAME": "hygiene-test",
            "GIT_COMMITTER_EMAIL": "hygiene-test@example.com",
        },
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(["init"], repo)
    (repo / "notes.md").write_text("plain content\n", encoding="utf-8")
    _git(["add", "notes.md"], repo)
    _git(["commit", "-m", "fixture"], repo)


def test_pure_rename_is_skipped(tmp_path):
    """Staged R100 with identical bytes is not scanned."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(["mv", "notes.md", "renamed.md"], repo)

    assert (repo / "renamed.md") in pure_rename_targets(repo)
    assert (repo / "renamed.md") not in collect_scan_paths(repo)


def test_renamed_with_edits_is_still_scanned(tmp_path):
    """A rename that also changes bytes stays in the scan set."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(["mv", "notes.md", "renamed.md"], repo)
    with (repo / "renamed.md").open("a", encoding="utf-8") as handle:
        handle.write("this line mentions claude\n")
    _git(["add", "renamed.md"], repo)

    assert (repo / "renamed.md") not in pure_rename_targets(repo)
    assert (repo / "renamed.md") in collect_scan_paths(repo)
    assert any("Found claude" in hit for hit in scan_file(repo / "renamed.md"))


def test_rename_with_unstaged_edit_is_still_scanned(tmp_path):
    """``git mv`` plus an unstaged edit escapes neither the index nor the scan."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(["mv", "notes.md", "renamed.md"], repo)
    with (repo / "renamed.md").open("a", encoding="utf-8") as handle:
        handle.write("this line mentions claude\n")

    assert (repo / "renamed.md") not in pure_rename_targets(repo)
    assert (repo / "renamed.md") in collect_scan_paths(repo)
    assert any("Found claude" in hit for hit in scan_file(repo / "renamed.md"))
