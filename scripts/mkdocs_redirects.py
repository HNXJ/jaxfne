"""mkdocs hook: a redirect page at each moved page's old URL.

The map is `extra.redirect_maps` in mkdocs.yml (old page -> new page, both relative to docs/), kept
since the 2026-10-16 docs reorganization so published Read the Docs links keep resolving. Each old
URL gets a small HTML page that forwards to the new one; an old URL that a real page now occupies
is refused.
"""

from __future__ import annotations

import posixpath
from pathlib import Path

from mkdocs.exceptions import PluginError

PAGE = (
    "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<title>Moved</title><link rel=\"canonical\" href=\"{href}\">"
    "<meta http-equiv=\"refresh\" content=\"0; url={href}\">"
    "<script>location.replace(\"{href}\" + location.hash)</script></head>"
    "<body><p>This page moved to <a href=\"{href}\">{href}</a>.</p></body></html>\n"
)


def page_url(src: str, use_directory_urls: bool) -> str:
    """Site-relative URL of a markdown page, as mkdocs builds it."""
    stem = src[:-3]
    if not use_directory_urls:
        return stem + ".html"
    if stem == "index" or stem.endswith("/index"):
        return stem[: -len("index")]
    return stem + "/"


def on_post_build(config, **kwargs):
    maps = (config.get("extra") or {}).get("redirect_maps") or {}
    site = Path(config["site_dir"])
    docs = Path(config["docs_dir"])
    use_dirs = config["use_directory_urls"]
    for old, new in maps.items():
        if (docs / old).exists():
            raise PluginError(f"redirect from {old}: a page exists there")
        if not (docs / new).exists():
            raise PluginError(f"redirect to {new}: no such page")
        old_url, new_url = page_url(old, use_dirs), page_url(new, use_dirs)
        out = site / (old_url + "index.html" if old_url.endswith("/") or not old_url else old_url)
        base = old_url if old_url.endswith("/") else posixpath.dirname(old_url)
        href = posixpath.relpath(new_url or ".", base or ".")
        if new_url.endswith("/") and not href.endswith("/"):
            href += "/"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(PAGE.format(href=href), encoding="utf-8")
