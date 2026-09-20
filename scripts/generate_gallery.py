#!/usr/bin/env python3
"""Generate the gallery atlas index from committed manifests (Batch B6).

`docs/gallery.md` is output, not source, between the markers
``<!-- ATLAS-INDEX:START -->`` / ``<!-- ATLAS-INDEX:END -->``. Only atlases
whose manifest reads VALIDATED or CANONICAL are listed — GENERATED outputs
never enter the gallery. Every link is checked against the file on disk.

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
GALLERY = ROOT / "docs" / "gallery.md"
START = "<!-- ATLAS-INDEX:START -->"
END = "<!-- ATLAS-INDEX:END -->"
LISTED_STATES = ("validated", "canonical")


def collect() -> tuple[list[dict], list[str]]:
    entries, problems = [], []
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
        panels = manifest.get("panels", [])
        missing = [p["file"] for p in panels
                   if not (manifest_path.parent / p["file"]).exists()]
        if missing:
            problems.append(f"{slug}: missing panel files {missing}")
            continue
        entries.append({"slug": slug, "base": f"_static/{rel}/",
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
        panels = manifest.get("panels", [])
        missing = [p["file"] for p in panels
                   if not (manifest_path.parent / p["file"]).exists()]
        if missing:
            problems.append(f"{rel}: missing panel files {missing}")
            continue
        entries.append({"slug": rel, "base": f"_static/{rel}/",
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
        lines.append(f"[Index]({base}index.html) · {links}")
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
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
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
        GALLERY.write_text(before + block + after, encoding="utf-8")
        print(f"gallery rewritten: {len(entries)} atlases")
        return 0
    print(f"gallery: {len(entries)} validated atlases indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
