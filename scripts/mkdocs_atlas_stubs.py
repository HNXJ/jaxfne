"""mkdocs hook: a stand-in page for each atlas panel, etude HTML file and registry copy this checkout has not built.

The atlas and etude HTML are build products of scripts/build_docs_assets.py --build (Read the
Docs runs it before mkdocs), as are the registry "copies" figures it copies from artifacts/etudes.
A checkout without them would fail `mkdocs build --strict` on every
link to them, so this hook adds a stand-in for each file that docs/_static/atlas_registry.json
declares and the docs tree lacks. A link to a file outside the registry still fails strict validation,
and on Read the Docs any stand-in fails the build.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.atlas_stubs")

STUB = (
    "<!doctype html><meta charset=\"utf-8\"><title>Atlas panel</title>"
    "<p>This atlas panel is built by <code>python scripts/build_docs_assets.py --build</code>.</p>"
)

# A valid 1x1 PNG: the stand-in for a copy dest (a PNG figure) the checkout lacks.
PNG_STUB = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def on_files(files, config):
    docs = Path(config["docs_dir"])
    registry = json.loads((docs / "_static" / "atlas_registry.json").read_text(encoding="utf-8"))
    added = 0
    for entry in registry["atlases"] + registry["etudes"]:
        rel_dir = Path(entry["dir"]).relative_to("docs").as_posix()
        names = entry["html"] if "html" in entry else set(entry["panels"]) | {"index.html"}
        for name in sorted(set(names)):
            uri = f"{rel_dir}/{name}"
            if files.get_file_from_path(uri) is None:
                files.append(File.generated(config, uri, content=STUB))
                added += 1
    for entry in registry.get("copies", []):
        uri = Path(entry["dest"]).relative_to("docs").as_posix()
        if files.get_file_from_path(uri) is None:
            files.append(File.generated(config, uri, content=PNG_STUB))
            added += 1
    log.info("atlas and etude stand-ins: %d (0 when the build products are built)", added)
    if added and os.environ.get("READTHEDOCS") == "True":
        raise PluginError(f"{added} atlas or etude files would be published as stand-ins; "
                          "Read the Docs runs build_docs_assets.py --build first")
    return files
