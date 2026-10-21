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
