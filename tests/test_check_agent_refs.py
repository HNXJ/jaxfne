"""check_agent_refs resolves references the way a fresh clone does: gitignored files never count."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "harness"))

import check_agent_refs  # noqa: E402


def _repo(tmp_path: Path, doc: str) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("artifacts/_temp/\n", encoding="utf-8")
    (tmp_path / "artifacts" / "_temp").mkdir(parents=True)
    (tmp_path / "artifacts" / "_temp" / "CURRENT_TASK.md").write_text("mode: CODE\n", encoding="utf-8")
    (tmp_path / "artifacts" / "tracked.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "artifacts" / "AGENTS.md").write_text(doc, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "artifacts/tracked.md", ".gitignore"], check=True)
    return tmp_path


def test_a_gitignored_path_on_disk_is_a_broken_reference(tmp_path):
    root = _repo(tmp_path, "Temp lives in `artifacts/_temp/CURRENT_TASK.md`.\n")
    assert (root / "artifacts" / "_temp" / "CURRENT_TASK.md").is_file(), "fixture must hold the ignored file"
    assert check_agent_refs.run([root / "artifacts" / "AGENTS.md"], root=root) == 1


def test_an_optional_marker_excuses_a_gitignored_path(tmp_path):
    root = _repo(tmp_path, "Temp lives in `artifacts/_temp/`. <!-- optional -->\n")
    assert check_agent_refs.run([root / "artifacts" / "AGENTS.md"], root=root) == 0


def test_tracked_files_their_folders_and_new_unignored_files_resolve(tmp_path):
    root = _repo(tmp_path, "See `artifacts/tracked.md`, `artifacts/` and `artifacts/new.md`.\n")
    (root / "artifacts" / "new.md").write_text("y\n", encoding="utf-8")
    assert check_agent_refs.run([root / "artifacts" / "AGENTS.md"], root=root) == 0
