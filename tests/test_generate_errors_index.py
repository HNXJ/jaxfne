"""Errors index gate: the ERRORS-INDEX block of docs/reference/errors.md matches jaxfne/."""

from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "generate_errors_index", REPO / "scripts" / "generate_errors_index.py"
)
gei = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gei)


def test_collect_examines_both_exported_classes():
    tables = gei.collect()
    rows = [row for table in tables.values() for row in table]
    assert rows, "fixture: jaxfne/ must hold literal raise sites"
    assert list(tables) == sorted(tables), "modules must sort by path"
    names_by_module = {m: {name for _, name, _ in table} for m, table in tables.items()}
    assert "TFNEError" in names_by_module["jaxfne/tfne.py"]
    assert "EpistemicRefusal" in names_by_module["jaxfne/fields/probes.py"]


def test_render_is_deterministic_and_rows_sort_by_line():
    tables = gei.collect()
    assert gei.render(tables) == gei.render(gei.collect())
    for table in tables.values():
        assert table == sorted(table)


def test_repository_errors_index_current():
    assert gei.main(["--check"]) == 0


def test_check_fails_on_planted_stale_block(tmp_path, monkeypatch):
    text = gei.PAGE.read_text(encoding="utf-8")
    before, rest = text.split(gei.START, 1)
    _, after = rest.split(gei.END, 1)
    page = tmp_path / "errors.md"
    monkeypatch.setattr(gei, "PAGE", page)
    page.write_text(before + gei.START + "\n\nplanted stale block\n\n" + gei.END + after, encoding="utf-8")
    assert "planted stale block" in page.read_text(encoding="utf-8"), "fixture must plant a stale block"
    assert gei.main(["--check"]) == 1
    page.write_text(text, encoding="utf-8")
    assert gei.main(["--check"]) == 0


def test_check_fails_without_markers(tmp_path, monkeypatch):
    page = tmp_path / "errors.md"
    page.write_text("# Errors\n", encoding="utf-8")
    monkeypatch.setattr(gei, "PAGE", page)
    assert gei.main(["--check"]) == 1


def test_write_is_idempotent(tmp_path, monkeypatch):
    page = tmp_path / "errors.md"
    page.write_bytes(gei.PAGE.read_bytes())
    monkeypatch.setattr(gei, "PAGE", page)
    assert gei.main(["--write"]) == 0
    first = page.read_bytes()
    assert gei.main(["--write"]) == 0
    assert page.read_bytes() == first == gei.PAGE.read_bytes()
