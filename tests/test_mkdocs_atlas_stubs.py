"""mkdocs hook: stand-ins exactly for registry atlas files the checkout lacks."""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("mkdocs")
from mkdocs.config import load_config  # noqa: E402
from mkdocs.structure.files import Files, get_files  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


def _on_files(files, config):
    """Run on_files through mkdocs.yml's plugin collection, as a build does."""
    return config.plugins.on_files(files, config=config)


def _registry_uris():
    registry = json.loads((REPO / "docs/_static/atlas_registry.json").read_text(encoding="utf-8"))
    return {f"{pathlib.Path(e['dir']).relative_to('docs').as_posix()}/{name}"
            for e in registry["atlases"] for name in set(e["panels"]) | {"index.html"}}


def test_stand_ins_cover_exactly_the_unbuilt_registry_files():
    config = load_config(str(REPO / "mkdocs.yml"))
    files = get_files(config)
    atlas = _registry_uris()
    assert len(atlas) == 200, "fixture: the registry declares 200 atlas files"
    # A fresh clone: no atlas HTML among the documentation files.
    fresh = Files([f for f in files if f.src_uri not in atlas])
    assert not any(fresh.get_file_from_path(u) for u in atlas), "fixture must lack every atlas file"
    out = _on_files(fresh, config)
    added = {f.src_uri for f in out if f.generated_by}
    assert added == atlas
    assert out.get_file_from_path("_static/atlas/two_neuron_ei/typo.html") is None


def test_no_stand_in_for_a_built_file():
    config = load_config(str(REPO / "mkdocs.yml"))
    atlas = sorted(_registry_uris())
    files = get_files(config)
    built = Files([f for f in files if f.src_uri not in atlas[1:]])
    out = _on_files(built, config)
    added = {f.src_uri for f in out if f.generated_by}
    assert atlas[0] not in added and added == set(atlas[1:])
