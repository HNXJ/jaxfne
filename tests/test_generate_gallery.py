"""Gallery index gate: the ATLAS-INDEX block of docs/start/gallery.md matches the manifests."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

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


def test_write_is_idempotent(tmp_path, monkeypatch):
    gallery = tmp_path / "gallery.md"
    gallery.write_text(gg.GALLERY.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(gg, "GALLERY", gallery)
    assert gg.main(["--write"]) == 0
    first = gallery.read_bytes()
    assert gg.main(["--write"]) == 0
    assert gallery.read_bytes() == first


def _static_without_html(tmp_path):
    """A docs/_static copy holding only the tracked manifests and registry, as in a fresh clone."""
    static = tmp_path / "_static"
    sources = sorted(gg.STATIC.glob("atlas/**/manifest.json")) + sorted(
        gg.STATIC.glob("atlas_three_area/manifest.json")) + [gg.REGISTRY]
    for src in sources:
        dst = static / src.relative_to(gg.STATIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    assert not list(static.rglob("*.html")), "fixture must hold no atlas HTML"
    return static


def test_collect_needs_no_built_html(tmp_path, monkeypatch):
    static = _static_without_html(tmp_path)
    monkeypatch.setattr(gg, "STATIC", static)
    monkeypatch.setattr(gg, "REGISTRY", static / "atlas_registry.json")
    entries, problems = gg.collect()
    registry = json.loads((static / "atlas_registry.json").read_text(encoding="utf-8"))
    assert problems == [] and len(entries) == len(registry["atlases"])


def test_collect_flags_a_panel_outside_the_registry(tmp_path, monkeypatch):
    static = _static_without_html(tmp_path)
    manifest_path = static / "atlas" / "two_neuron_ei" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_state"].lower() in gg.LISTED_STATES, "fixture atlas must be listed"
    manifest["panels"].append({"file": "extra.html", "panel": "extra", "status": "OBSERVED"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(gg, "STATIC", static)
    monkeypatch.setattr(gg, "REGISTRY", static / "atlas_registry.json")
    _, problems = gg.collect()
    assert len(problems) == 1 and "extra.html" in problems[0], problems


def test_repository_gallery_index_current():
    assert gg.main(["--check"]) == 0


def test_check_and_write_are_mutually_exclusive(tmp_path, monkeypatch):
    gallery = tmp_path / "gallery.md"
    gallery.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(gg, "GALLERY", gallery)
    with pytest.raises(SystemExit) as exc:
        gg.main(["--check", "--write"])
    assert exc.value.code == 2
    assert gallery.read_text(encoding="utf-8") == "stale"
