"""mkdocs hook: each moved page's old URL forwards to its new page."""

from __future__ import annotations

import importlib.util
import pathlib
import posixpath
import re

import pytest

pytest.importorskip("mkdocs")
from mkdocs.config import load_config  # noqa: E402
from mkdocs.exceptions import PluginError  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("mkdocs_redirects", REPO / "scripts" / "mkdocs_redirects.py")
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def test_page_url_matches_mkdocs():
    assert hook.page_url("index.md", True) == ""
    assert hook.page_url("guides/index.md", True) == "guides/"
    assert hook.page_url("guides/hdp.md", True) == "guides/hdp/"
    assert hook.page_url("guides/hdp.md", False) == "guides/hdp.html"


def test_repository_map_points_from_gone_pages_to_live_ones():
    config = load_config(str(REPO / "mkdocs.yml"))
    maps = config["extra"]["redirect_maps"]
    assert len(maps) > 50, "redirect_maps must be read"
    docs = pathlib.Path(config["docs_dir"])
    assert not [old for old in maps if (docs / old).exists()]
    assert not [new for new in maps.values() if not (docs / new).exists()]


# Frozen files cite docs paths from before the 2026-10-16 move and cannot be edited. A page the
# docs build excludes has no redirect page, so its move is recorded here instead.
FROZEN_CITATION_MOVES = {
    "docs/publication/evidence_consolidation.md": "docs/theory/publication/evidence_consolidation.md",
}


def test_frozen_files_cite_docs_paths_that_resolve():
    import json

    config = load_config(str(REPO / "mkdocs.yml"))
    maps = config["extra"]["redirect_maps"]
    frozen = json.loads((REPO / "artifacts/publication/frozen_manifest.json").read_text(encoding="utf-8"))
    cited = set()
    for rel in frozen["files"]:
        path = REPO / rel
        if path.suffix in (".md", ".json") and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            cited |= {m.rstrip(".,)") for m in re.findall(r"(?<![\w/.-])(docs/[\w./-]+\.md)", text)}
    assert cited, "the frozen files must cite docs paths (scan reads nothing otherwise)"
    unresolved = []
    for old in sorted(cited):
        if (REPO / old).exists():
            continue
        new = maps.get(old[len("docs/"):])
        target = f"docs/{new}" if new else FROZEN_CITATION_MOVES.get(old)
        if target is None or not (REPO / target).exists():
            unresolved.append(old)
    assert not unresolved, unresolved
    assert all((REPO / new).exists() for new in FROZEN_CITATION_MOVES.values())


def _build(tmp_path, maps, use_dirs=True):
    docs, site = tmp_path / "docs", tmp_path / "site"
    for new in maps.values():
        (docs / new).parent.mkdir(parents=True, exist_ok=True)
        (docs / new).write_text("# page\n", encoding="utf-8")
    config = {"extra": {"redirect_maps": maps}, "site_dir": str(site), "docs_dir": str(docs),
              "use_directory_urls": use_dirs}
    hook.on_post_build(config)
    return site


@pytest.mark.parametrize("use_dirs", [True, False])
def test_redirect_page_resolves_to_the_new_url(tmp_path, use_dirs):
    maps = {"hdp.md": "guides/state/hdp.md", "api/core.md": "reference/api/core/core.md",
            "tutorials/index.md": "index.md"}
    site = _build(tmp_path, maps, use_dirs)
    for old, new in maps.items():
        old_url, new_url = hook.page_url(old, use_dirs), hook.page_url(new, use_dirs)
        out = site / (old_url + "index.html" if old_url.endswith("/") else old_url)
        href = re.search(r'url=([^"]+)"', out.read_text(encoding="utf-8")).group(1)
        base = old_url if old_url.endswith("/") else posixpath.dirname(old_url)
        landed = posixpath.normpath(posixpath.join("/" + base, href))
        assert landed == posixpath.normpath("/" + new_url), (old, href)


def test_refuses_a_live_page_at_the_old_path_and_a_missing_target(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# a\n", encoding="utf-8")
    config = {"site_dir": str(tmp_path / "site"), "docs_dir": str(docs), "use_directory_urls": True}
    with pytest.raises(PluginError, match="a page exists"):
        hook.on_post_build({**config, "extra": {"redirect_maps": {"a.md": "a.md"}}})
    with pytest.raises(PluginError, match="no such page"):
        hook.on_post_build({**config, "extra": {"redirect_maps": {"b.md": "c.md"}}})
