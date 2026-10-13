"""mkdocs hook: stand-ins exactly for registry atlas files the checkout lacks."""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("mkdocs")
from mkdocs.config import load_config  # noqa: E402
from mkdocs.structure.files import File, Files, get_files  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent


def _on_files(files, config):
    """Run on_files through mkdocs.yml's plugin collection, as a build does."""
    return config.plugins.on_files(files, config=config)


def _registry_uris(section="atlases"):
    """URIs of the build products one registry section declares: atlas panels or etude HTML."""
    registry = json.loads((REPO / "docs/_static/atlas_registry.json").read_text(encoding="utf-8"))
    uris = set()
    for e in registry[section]:
        names = set(e["html"]) if section == "etudes" else set(e["panels"]) | {"index.html"}
        uris |= {f"{pathlib.Path(e['dir']).relative_to('docs').as_posix()}/{name}" for name in names}
    return uris


def test_stand_ins_cover_exactly_the_unbuilt_registry_files():
    config = load_config(str(REPO / "mkdocs.yml"))
    files = get_files(config)
    atlas = _registry_uris()
    assert len(atlas) == 200, "fixture: the registry declares 200 atlas files"
    etudes = _registry_uris("etudes")
    assert len(etudes) == 25, "fixture: the registry declares 25 etude HTML files"
    products = atlas | etudes
    # A fresh clone: no build product among the documentation files.
    fresh = Files([f for f in files if f.src_uri not in products])
    assert not any(fresh.get_file_from_path(u) for u in products), "fixture must lack every build product"
    out = _on_files(fresh, config)
    added = {f.src_uri for f in out if f.generated_by}
    assert added == products
    assert out.get_file_from_path("_static/atlas/two_neuron_ei/typo.html") is None


def _with_built(config, built, tmp_path):
    """The docs files with only `built` build products present, written under tmp_path."""
    products = _registry_uris() | _registry_uris("etudes")
    files = [f for f in get_files(config) if f.src_uri not in products]
    for uri in built:
        (tmp_path / uri).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / uri).write_text("<p>built</p>", encoding="utf-8")
        files.append(File(uri, str(tmp_path), config["site_dir"], config["use_directory_urls"]))
    out = Files(files)
    assert {u for u in products if out.get_file_from_path(u)} == set(built), "fixture: built set"
    return out


def test_a_stand_in_fails_the_build_on_read_the_docs(monkeypatch, tmp_path):
    config = load_config(str(REPO / "mkdocs.yml"))
    products = sorted(_registry_uris() | _registry_uris("etudes"))
    one_missing = _with_built(config, products[1:], tmp_path)
    monkeypatch.setenv("READTHEDOCS", "True")
    with pytest.raises(Exception, match="1 atlas or etude files would be published as stand-ins"):
        _on_files(one_missing, config)


def test_no_stand_in_for_a_built_file(tmp_path):
    config = load_config(str(REPO / "mkdocs.yml"))
    products = sorted(_registry_uris() | _registry_uris("etudes"))
    out = _on_files(_with_built(config, products[:1], tmp_path), config)
    added = {f.src_uri for f in out if f.generated_by}
    assert products[0] not in added and added == set(products[1:])
