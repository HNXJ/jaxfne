"""sync_skills --check is honored: clean tree passes, --check+--update is refused."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "sync_skills", REPO / "scripts" / "harness" / "sync_skills.py"
)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def test_check_passes_on_clean_tree(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sync_skills.py", "--check"])
    assert ss.main() == 0


def test_check_and_update_are_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sync_skills.py", "--check", "--update"])
    with pytest.raises(SystemExit) as exc:
        ss.main()
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def _run(script, *args, cwd=None):
    import subprocess

    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, cwd=cwd,
    )


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_memory_hygiene_flags_a_short_sha(tmp_path):
    f = tmp_path / "m.md"
    f.write_text("Evidence: run on 226a7952 failed.\n", encoding="utf-8")
    r = _run("harness/check_memory_hygiene.py", str(f))
    assert r.returncode == 1 and "short sha" in r.stdout


def test_memory_hygiene_scans_the_project_memory():
    mh = _load("mh", "harness/check_memory_hygiene.py")
    assert mh.PROJECT_MEMORY.is_file() and mh.PROJECT_MEMORY.name == "memory.md"


def test_release_validation_says_when_nothing_was_examined(tmp_path):
    ok = _run("release/validate_release_artifacts.py", "--version", "0",
              "--allow-missing-dist", cwd=tmp_path)
    assert ok.returncode == 0 and "NOT EXAMINED" in ok.stdout
    assert _run("release/validate_release_artifacts.py", "--version", "0",
                cwd=tmp_path).returncode == 1


def test_harness_integrity_names_the_absent_global_manifest(monkeypatch, capsys, tmp_path):
    hi = _load("hi", "harness/check_harness_integrity.py")
    monkeypatch.setattr(hi, "GLOBAL_MANIFEST", tmp_path / "absent.json")
    monkeypatch.setattr(sys, "argv", ["check_harness_integrity.py"])
    assert hi.main() == 0
    assert "global absent" in capsys.readouterr().out
