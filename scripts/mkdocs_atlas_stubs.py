"""mkdocs hook: a stand-in page for each atlas panel this checkout has not built.

The atlas HTML is a build product of scripts/build_docs_assets.py --build (Read the Docs runs
it before mkdocs). A checkout without it would fail `mkdocs build --strict` on every atlas
link, so this hook adds a stand-in for each file that docs/_static/atlas_registry.json declares
and the docs tree lacks. A link to a file outside the registry still fails strict validation,
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


def on_files(files, config):
    docs = Path(config["docs_dir"])
    registry = json.loads((docs / "_static" / "atlas_registry.json").read_text(encoding="utf-8"))
    added = 0
    for entry in registry["atlases"]:
        rel_dir = Path(entry["dir"]).relative_to("docs").as_posix()
        for name in sorted(set(entry["panels"]) | {"index.html"}):
            uri = f"{rel_dir}/{name}"
            if files.get_file_from_path(uri) is None:
                files.append(File.generated(config, uri, content=STUB))
                added += 1
    log.info("atlas stand-ins: %d (0 when the atlas HTML is built)", added)
    if added and os.environ.get("READTHEDOCS") == "True":
        raise PluginError(f"{added} atlas files would be published as stand-ins; "
                          "Read the Docs runs build_docs_assets.py --build first")
    return files
