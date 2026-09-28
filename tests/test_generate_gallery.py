"""Gallery index gate: the ATLAS-INDEX block of docs/gallery.md matches the manifests."""

from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "generate_gallery", REPO / "scripts" / "generate_gallery.py"
)
gg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gg)


def test_check_fails_on_stale_index(tmp_path, monkeypatch):
    entries, problems = gg.collect()
    assert not problems and len(entries) > 1
    text = gg.GALLERY.read_text(encoding="utf-8")
    before, after = text.split(gg.START)[0], text.split(gg.END)[1]
    gallery = tmp_path / "gallery.md"
    monkeypatch.setattr(gg, "GALLERY", gallery)
    gallery.write_text(before + gg.render(entries) + after, encoding="utf-8")
    assert gg.main(["--check"]) == 0
    gallery.write_text(before + gg.render(entries[1:]) + after, encoding="utf-8")  # one atlas missing
    assert gg.main(["--check"]) == 1


def test_repository_gallery_index_current():
    assert gg.main(["--check"]) == 0
