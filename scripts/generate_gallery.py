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
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"
GALLERY = ROOT / "docs" / "start" / "gallery.md"
START = "<!-- ATLAS-INDEX:START -->"
END = "<!-- ATLAS-INDEX:END -->"
LISTED_STATES = ("validated", "canonical")


REGISTRY = STATIC / "atlas_registry.json"

# Public figure site: every still and "Interactive version" link points here
# (published by the Dashboard workflow's figures job from --site output).
FIGURES_URL = "https://hnxj.github.io/jaxfne/figures/"
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")


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


def _figure_slug(rel: str) -> str:
    """Registry atlas dir -> figure-site slug (mirrors build_docs_assets)."""
    if rel == "atlas":
        return "readme"
    if rel == "atlas_three_area":
        return "three_area"
    assert rel.startswith("atlas/"), f"gallery entry outside atlas/: {rel}"
    return rel[len("atlas/"):]


def _representative_file(panels: list[dict]) -> str:
    """One panel standing for the whole atlas: raster.html, else the first
    sorted raster*, else the first sorted panel file."""
    avail = [p["file"] for p in panels if str(p.get("status", "")).upper() == "AVAILABLE"]
    names = avail or [p["file"] for p in panels]
    if "raster.html" in names:
        return "raster.html"
    raster = sorted(n for n in names if n.startswith("raster"))
    if raster:
        return raster[0]
    return sorted(names)[0]


def render(entries: list[dict]) -> str:
    lines = [START, "",
             f"Evidence index of validated atlases ({len(entries)} models). "
             "Generated from committed `manifest.json` files — do not edit by hand; "
             "run `python scripts/generate_gallery.py --write`.", ""]
    for e in entries:
        slug, m, state = e["slug"], e["manifest"], e["state"]
        fslug = _figure_slug(slug)
        stem = _representative_file(m["panels"])[:-len(".html")]
        title = str(m.get("title", slug))
        lines.append(f"### `{slug}` — {title}")
        lines.append("")
        lines.append(f"State `{state}` · N={m.get('n_neurons')} · "
                     f"edges={m.get('n_edges')} · steps={m.get('n_steps')} · "
                     f"config `{str(m.get('config_hash'))[:12]}` · "
                     f"jaxfne {m.get('jaxfne_version')}")
        lines.append("")
        lines.append(f"![{title}]({FIGURES_URL}{fslug}/{stem}.png)")
        lines.append(f"*{title}: spike raster. "
                     f"[Interactive version]({FIGURES_URL}{fslug}/{stem}.html)*")
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
        raw = GALLERY.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n")
        if START not in text or END not in text:
            raise SystemExit("gallery.md lacks ATLAS-INDEX markers")
        before = text.split(START)[0]
        after = text.split(END)[1]
        if after.startswith("\n"):  # block ends with END's line break; keep one so writes are idempotent
            after = after[1:]
        newline = "\r\n" if b"\r\n" in raw else "\n"
        GALLERY.write_bytes((before + block + after).replace("\n", newline).encode("utf-8"))
        print(f"gallery rewritten: {len(entries)} atlases")
        return 0
    # Never compare SHA values: the recorded git HEAD moves with unrelated
    # commits while the block stays current.
    expected = SHA_RE.sub("0" * 40, block.split(START)[1].split(END)[0])
    if SHA_RE.sub("0" * 40, current_block() or "") != expected:
        print("gallery.md index is stale against the manifests; run "
              "`python scripts/generate_gallery.py --write`")
        return 1
    print(f"gallery: {len(entries)} validated atlases indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
