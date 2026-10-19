#!/usr/bin/env python3
"""Generate the gallery atlas index from committed manifests (Batch B6).

`docs/start/gallery.md` is output, not source, between the markers
``<!-- ATLAS-INDEX:START -->`` / ``<!-- ATLAS-INDEX:END -->``. Only atlases
whose manifest reads VALIDATED or CANONICAL are listed — GENERATED outputs
never enter the gallery. Every link is checked against docs/_static/atlas_registry.json,
the record of the panels the docs build writes.

Usage:
    python scripts/generate_gallery.py --check   # verify gallery matches manifests
    python scripts/generate_gallery.py --write   # rewrite the index block
"""

from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"
GALLERY = ROOT / "docs" / "start" / "gallery.md"
START = "<!-- ATLAS-INDEX:START -->"
END = "<!-- ATLAS-INDEX:END -->"
LISTED_STATES = ("validated", "canonical")


REGISTRY = STATIC / "atlas_registry.json"


def _unbuilt(rel: str, panels: list[dict], registry: dict) -> list[str]:
    """Panel files the docs build does not write for this atlas.

    The HTML is a build product (scripts/build_docs_assets.py), so links are checked
    against the tracked registry rather than files on disk.
    """
    built = registry.get(f"docs/_static/{rel}")
    return [p["file"] for p in panels] if built is None else [
        p["file"] for p in panels if p["file"] not in built]


def collect() -> tuple[list[dict], list[str]]:
    entries, problems = [], []
    registry = {e["dir"]: set(e["panels"])
                for e in json.loads(REGISTRY.read_text(encoding="utf-8"))["atlases"]}
    paths = sorted(STATIC.glob("atlas/**/manifest.json"))
    for manifest_path in paths:
        rel = manifest_path.parent.relative_to(STATIC).as_posix()
        slug = rel
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            problems.append(f"{slug}: unreadable manifest ({exc})")
            continue
        state = str(manifest.get("figure_state", "generated")).lower()
        if state not in LISTED_STATES:
            continue
        missing = _unbuilt(rel, manifest.get("panels", []), registry)
        if missing:
            problems.append(f"{slug}: panel files outside the registry {missing}")
            continue
        entries.append({"slug": slug, "base": f"../_static/{rel}/",
                        "manifest": manifest, "state": state})
    # The three-area hierarchy lives beside atlas/ rather than inside it.
    for manifest_path in sorted(STATIC.glob("atlas_three_area/manifest.json")):
        rel = manifest_path.parent.relative_to(STATIC).as_posix()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            problems.append(f"{rel}: unreadable manifest ({exc})")
            continue
        state = str(manifest.get("figure_state", "generated")).lower()
        if state not in LISTED_STATES:
            continue
        missing = _unbuilt(rel, manifest.get("panels", []), registry)
        if missing:
            problems.append(f"{rel}: panel files outside the registry {missing}")
            continue
        entries.append({"slug": rel, "base": f"../_static/{rel}/",
                        "manifest": manifest, "state": state})
    return entries, problems


def render(entries: list[dict]) -> str:
    lines = [START, "",
             f"Evidence index of validated atlases ({len(entries)} models). "
             "Generated from committed `manifest.json` files — do not edit by hand; "
             "run `python scripts/generate_gallery.py --write`.", ""]
    for e in entries:
        slug, m, state = e["slug"], e["manifest"], e["state"]
        base = e["base"]
        lines.append(f"### `{slug}` — {m.get('title', slug)}")
        lines.append("")
        lines.append(f"State `{state}` · N={m.get('n_neurons')} · "
                     f"edges={m.get('n_edges')} · steps={m.get('n_steps')} · "
                     f"config `{str(m.get('config_hash'))[:12]}` · "
                     f"jaxfne {m.get('jaxfne_version')}")
        lines.append("")
        links = " · ".join(
            f"[{p['panel']}]({base}{p['file']})" for p in m["panels"])
        # Links are relative to docs/start/gallery.md; raw HTML is served unrewritten from the
        # page URL /start/gallery/, one level deeper.
        lines.append(
            f'<iframe class="jx-frame" src="../{base}index.html" loading="lazy" '
            f'title="{m.get("title", "atlas")} dashboard"></iframe>')
        lines.append("")
        lines.append(f"[Open full page]({base}index.html) · {links}")
        lines.append("")
    lines.append(END)
    return "\n".join(lines) + "\n"


def current_block() -> str | None:
    text = GALLERY.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return None
    return text.split(START)[1].split(END)[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    entries, problems = collect()
    if problems:
        print("gallery problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    block = render(entries)
    if args.write:
        text = GALLERY.read_text(encoding="utf-8")
        if START not in text or END not in text:
            raise SystemExit("gallery.md lacks ATLAS-INDEX markers")
        before = text.split(START)[0]
        after = text.split(END)[1]
        if after.startswith("\n"):  # block ends with END's line break; keep one so writes are idempotent
            after = after[1:]
        GALLERY.write_text(before + block + after, encoding="utf-8")
        print(f"gallery rewritten: {len(entries)} atlases")
        return 0
    expected = block.split(START)[1].split(END)[0]  # parsed as current_block() parses the file
    if current_block() != expected:
        print("gallery.md index is stale against the manifests; run "
              "`python scripts/generate_gallery.py --write`")
        return 1
    print(f"gallery: {len(entries)} validated atlases indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
