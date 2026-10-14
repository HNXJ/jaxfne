#!/usr/bin/env python3
"""Pack/unpack the Labyrinth `.lab/` graph nodes as one tracked bundle.

git tracks the graph nodes as one file, `artifacts/.lab/bundle.json`; loose node
files in `artifacts/.lab/` stay local (gitignored). Pack stores each node file's
exact UTF-8 text, unpack writes it back byte-exact (line endings included).
Pack refuses to write fewer nodes than the bundle holds: on a fresh clone the
loose files are absent, so unpack them first.

    python scripts/lab_bundle.py --unpack artifacts/.lab   # restore the node files
    python scripts/lab_bundle.py --pack                    # write artifacts/.lab/bundle.json
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


def read_bundle(path: Path) -> dict:
    """The bundle's {name: text} map; ValueError when it is not a lab bundle."""
    try:
        bundle = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not UTF-8 JSON: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT:
        raise ValueError(f"{path.name}: not a {FORMAT} bundle")
    files = bundle.get("files")
    if not isinstance(files, dict) or not all(isinstance(t, str) for t in files.values()):
        raise ValueError(f"{path.name}: files must map names to text")
    return files


def pack(lab_dir: Path = DEFAULT_LAB) -> Path:
    files = {p.name: p.read_bytes().decode("utf-8") for p in iter_node_files(lab_dir)}
    out = lab_dir / BUNDLE_NAME
    if out.is_file():
        held = len(read_bundle(out))
        if len(files) < held:
            raise ValueError(f"pack would write {len(files)} nodes over a bundle of {held}; "
                             "unpack into the lab directory first")
    out.write_text(
        json.dumps({"format": FORMAT, "files": files},
                   indent=1, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return out


def unpack(dest: Path, lab_dir: Path = DEFAULT_LAB) -> list[Path]:
    files = read_bundle(lab_dir / BUNDLE_NAME)
    bad = [n for n in files
           if Path(n).name != n or not n.endswith(".json") or n.lower() == BUNDLE_NAME]
    if bad:
        raise ValueError(f"bundle names that are not plain node file names: {sorted(bad)}")
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in sorted(files):
        target = dest / name
        target.write_bytes(files[name].encode("utf-8"))
        written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pack", action="store_true", help="write artifacts/.lab/bundle.json")
    group.add_argument("--unpack", type=Path, metavar="DIR", help="restore every file into DIR")
    args = parser.parse_args()

    try:
        if args.pack:
            out = pack()
            print(f"packed {len(iter_node_files(DEFAULT_LAB))} files -> {out}")
        else:
            written = unpack(args.unpack)
            print(f"unpacked {len(written)} files -> {args.unpack}")
    except ValueError as exc:
        print(f"lab_bundle: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
