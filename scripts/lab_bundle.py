#!/usr/bin/env python3
"""Pack/unpack the Labyrinth `.lab/` graph nodes as one tracked bundle.

`artifacts/.lab/` holds one tracked JSON file per graph node. Tracking all of
them as one file (`artifacts/.lab/bundle.json`) keeps the tree small without
losing a byte: pack stores each file's exact UTF-8 text, unpack writes it back
byte-exact (line endings included).

    python scripts/lab_bundle.py --pack          # write artifacts/.lab/bundle.json
    python scripts/lab_bundle.py --unpack DIR    # restore every file into DIR
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "pyproject.toml").is_file() or not (ROOT / "jaxfne").is_dir():
    print("lab_bundle: repo-root guard failed (not the jaxfne repo root)", file=sys.stderr)
    sys.exit(2)

DEFAULT_LAB = ROOT / "artifacts" / ".lab"
BUNDLE_NAME = "bundle.json"
FORMAT = "lab-bundle-1"

# Mirrors the gitignored artifacts/.lab patterns in .gitignore: these live in
# .lab but are not tracked nodes, plus the bundle itself.
SKIP_NAMES = ("bundle.json", "suggestions.json", "metrics.json")


def iter_node_files(lab_dir: Path) -> list[Path]:
    """Every tracked-style *.json directly in lab_dir: no subdirectories, none
    of the gitignored names."""
    files = []
    for path in sorted(lab_dir.glob("*.json")):
        if path.name in SKIP_NAMES or fnmatch.fnmatch(path.name, "plan-*.json"):
            continue
        files.append(path)
    return files


def pack(lab_dir: Path = DEFAULT_LAB) -> Path:
    files = {p.name: p.read_bytes().decode("utf-8") for p in iter_node_files(lab_dir)}
    out = lab_dir / BUNDLE_NAME
    out.write_text(
        json.dumps({"format": FORMAT, "files": files},
                   indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return out


def unpack(dest: Path, lab_dir: Path = DEFAULT_LAB) -> list[Path]:
    bundle = json.loads((lab_dir / BUNDLE_NAME).read_bytes().decode("utf-8"))
    if bundle["format"] != FORMAT:
        raise ValueError(f"unknown bundle format: {bundle['format']!r}")
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in sorted(bundle["files"]):
        target = dest / name
        target.write_bytes(bundle["files"][name].encode("utf-8"))
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pack", action="store_true", help="write artifacts/.lab/bundle.json")
    group.add_argument("--unpack", type=Path, metavar="DIR", help="restore every file into DIR")
    args = parser.parse_args()

    if args.pack:
        out = pack()
        print(f"packed {len(iter_node_files(DEFAULT_LAB))} files -> {out}")
    else:
        written = unpack(args.unpack)
        print(f"unpacked {len(written)} files -> {args.unpack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
