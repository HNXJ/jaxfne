#!/usr/bin/env python3
"""Docs atlas assets: regenerate the atlases and check them against a tracked registry.

The registry (docs/_static/atlas_registry.json) holds what must not drift for each
atlas: the realized configuration (config_hash, sizes, grid, seed), the panel
statuses, and the figure state with its promotion history. The registry is the
record; the HTML is a build product, and --check reads manifests only.

Modes:
    --registry-write        rebuild the registry from the tracked manifests
    --build [--slug a,b]    regenerate atlases in fresh processes; compare each fresh
                            manifest with the registry; on a match write the registry's
                            figure state and promotion into it; then rewrite gallery.md
    --check                 compare the tracked manifests with the registry; generate nothing

Slugs are registry slugs: the atlas/<slug> directories, ``three_area`` for
atlas_three_area, and ``readme`` for the top-level atlas/ (the canonical README atlas,
built by generate_readme_atlas.py --html-only).

Usage:
    python scripts/build_docs_assets.py --registry-write
    python scripts/build_docs_assets.py --build --slug two_neuron_ei,three_area
    python scripts/build_docs_assets.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC = ROOT / "docs" / "_static"
REGISTRY = STATIC / "atlas_registry.json"
SCHEMA = "atlas_registry.v1"
# Fields a fresh build must reproduce exactly.
COMPARED_FIELDS = ("config_hash", "n_neurons", "n_edges", "n_steps", "seed", "duration_ms", "dt_ms")
# Figure state the registry supplies to a fresh manifest; --check also verifies them on disk.
STATE_FIELDS = ("figure_state", "promotion")
README_SLUG = "readme"
THREE_AREA_SLUG = "three_area"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def discover() -> list[Path]:
    """Manifest paths of every tracked atlas: atlas/**, then atlas_three_area/."""
    return sorted(STATIC.glob("atlas/**/manifest.json")) + sorted(
        STATIC.glob("atlas_three_area/manifest.json"))


def _slug(atlas_dir: Path) -> str:
    if atlas_dir == STATIC / "atlas":
        return README_SLUG
    if atlas_dir == STATIC / "atlas_three_area":
        return THREE_AREA_SLUG
    return atlas_dir.relative_to(STATIC / "atlas").as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict) -> None:
    """Write as the generators do: indent 2, CRLF, no final newline (the tracked style)."""
    path.write_bytes(json.dumps(manifest, indent=2).replace("\n", "\r\n").encode("utf-8"))


def load_registry() -> dict:
    if not REGISTRY.exists():
        raise SystemExit(f"registry missing: {rel(REGISTRY)}; run --registry-write")
    registry = read_json(REGISTRY)
    if registry.get("schema") != SCHEMA:
        raise SystemExit(f"{rel(REGISTRY)}: schema {registry.get('schema')!r} != {SCHEMA!r}")
    for key in ("slug", "dir"):
        values = [e[key] for e in registry["atlases"]]
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise SystemExit(f"{rel(REGISTRY)}: duplicate {key} entries {dupes}")
    return registry


def entry_from_manifest(manifest_path: Path) -> dict:
    manifest = read_json(manifest_path)
    needed = (*COMPARED_FIELDS, *STATE_FIELDS, "panels")
    missing = [k for k in needed if k not in manifest]
    missing += [k for k in COMPARED_FIELDS if k in manifest and manifest[k] is None]
    if missing:
        raise SystemExit(f"{rel(manifest_path)}: tracked manifest lacks {missing}; "
                         "the registry cannot be built from it")
    return {
        "slug": _slug(manifest_path.parent),
        "dir": rel(manifest_path.parent),
        **{k: manifest[k] for k in COMPARED_FIELDS},
        "panels": {p["file"]: p["status"] for p in manifest["panels"]},
        "figure_state": manifest["figure_state"],
        "promotion": manifest["promotion"],
    }


def write_registry() -> int:
    entries = [entry_from_manifest(p) for p in discover()]
    doc = {
        "schema": SCHEMA,
        "built_from": "tracked manifests (build_docs_assets.py --registry-write)",
        "compared_fields": list(COMPARED_FIELDS),
        "atlases": entries,
    }
    REGISTRY.write_bytes((json.dumps(doc, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8"))
    print(f"registry: {len(entries)} atlases -> {rel(REGISTRY)}")
    return 0


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def compare(entry: dict, manifest: dict) -> list[str]:
    """Problems of a manifest against its registry entry; each names its field."""
    problems = []
    for field in COMPARED_FIELDS:
        if field not in manifest:
            problems.append(f"{field}: missing from the manifest")
        elif manifest[field] != entry[field]:
            problems.append(f"{field}: registry {_short(entry[field])} != manifest {_short(manifest[field])}")
    fresh = {p["file"]: p["status"] for p in manifest.get("panels", [])}
    reg = entry["panels"]
    only_reg = sorted(set(reg) - set(fresh))
    only_man = sorted(set(fresh) - set(reg))
    if only_reg or only_man:
        problems.append(f"panels: file set differs (registry only {only_reg}, manifest only {only_man})")
    for name in sorted(set(reg) & set(fresh)):
        if fresh[name] != reg[name]:
            problems.append(f"panels[{name}].status: registry {reg[name]!r} != manifest {fresh[name]!r}")
    return problems


def compare_state(entry: dict, manifest: dict) -> list[str]:
    """Figure state of a tracked manifest against the registry (--check only)."""
    problems = []
    for field in STATE_FIELDS:
        if manifest.get(field) != entry[field]:
            problems.append(f"{field}: registry {_short(entry[field])} != "
                            f"tracked manifest {_short(manifest.get(field))}")
    return problems


def apply_state(manifest: dict, entry: dict) -> dict:
    """Write the registry's figure state and promotion history into a fresh manifest."""
    manifest["figure_state"] = entry["figure_state"]
    manifest["promotion"] = list(entry["promotion"])
    return manifest


def check_registry(registry: dict, manifest_paths: list[Path]) -> tuple[list[str], int]:
    """Compare each registry entry with its tracked manifest and flag unregistered manifests.

    Returns the problems and the number of registry entries examined.
    """
    on_disk = {rel(p.parent): p for p in manifest_paths}
    entries = {e["dir"]: e for e in registry["atlases"]}
    problems = [f"{d}: manifest on disk has no registry entry"
                for d in sorted(set(on_disk) - set(entries))]
    for d, entry in sorted(entries.items()):
        if d not in on_disk:
            problems.append(f"{d}: registry entry has no manifest on disk")
            continue
        manifest = read_json(on_disk[d])
        problems += [f"{d}: {p}" for p in compare(entry, manifest) + compare_state(entry, manifest)]
    return problems, len(entries)


# Relative .html links: a markdown link target, or a raw HTML src/href attribute.
MD_HTML_LINK = re.compile(r"\]\(<?([^)\s#?<>]+\.html)|^ {0,3}\[[^\]]+\]:\s*<?([^>\s#?]+\.html)",
                          re.IGNORECASE | re.MULTILINE)
RAW_HTML_LINK = re.compile(r"""(?:src|href)\s*=\s*["']([^"'#?]+\.html)""", re.IGNORECASE)
ATLAS_DIRS = (STATIC / "atlas", STATIC / "atlas_three_area")


def link_base(page: Path, raw_html: bool) -> Path:
    """Directory a docs link resolves from.

    mkdocs rewrites markdown links relative to the source file. A raw HTML attribute
    reaches the browser unrewritten and resolves from the page URL, which with directory
    URLs is docs/<a>/<page>/ for docs/<a>/<page>.md and docs/<a>/ for docs/<a>/index.md.
    """
    if raw_html and page.name != "index.md":
        return page.with_suffix("")
    return page.parent


def html_links(text: str):
    """(offset, link, raw_html) for each relative .html link in a docs page."""
    for pattern, raw_html in ((MD_HTML_LINK, False), (RAW_HTML_LINK, True)):
        for match in pattern.finditer(text):
            link = next(g for g in match.groups() if g)
            if not re.match(r"^[a-z][a-z0-9+.-]*:|^/", link):
                yield match.start(), link, raw_html


def built_files(entry: dict) -> set[str]:
    """HTML files a build writes into one atlas directory: its panels and index.html."""
    return set(entry["panels"]) | {"index.html"}


def check_links(registry: dict, pages: dict[Path, str]) -> tuple[list[str], int, int]:
    """Every relative .html link in the docs pages must resolve.

    A link into an atlas directory names a build product, so it is checked against the
    registry; any other link must name a file on disk. Returns the problems, the links
    examined and, of those, the atlas links.
    """
    entries = {e["dir"]: e for e in registry["atlases"]}
    problems, examined, atlas_links = [], 0, 0
    for page, text in pages.items():
        for offset, link, raw_html in html_links(text):
            examined += 1
            target = Path(os.path.normpath(link_base(page, raw_html) / link))
            if any(d == target.parent or d in target.parents for d in ATLAS_DIRS):
                atlas_links += 1
                entry = entries.get(rel(target.parent))
                ok = entry is not None and target.name in built_files(entry)
                why = "is not an atlas file the build writes"
            else:
                ok, why = target.is_file(), "names no file"
            if not ok:
                kind = "raw HTML" if raw_html else "markdown"
                line = text.count("\n", 0, offset) + 1
                problems.append(f"{rel(page)}:{line}: {kind} link {link} {why}")
    return problems, examined, atlas_links


def docs_pages() -> dict[Path, str]:
    paths = sorted(p for p in (ROOT / "docs").rglob("*.md") if STATIC not in p.parents)
    return {p: p.read_text(encoding="utf-8") for p in paths}


def check() -> int:
    registry = load_registry()
    problems, examined = check_registry(registry, discover())
    for p in problems:
        print(f"  - {p}")
    print(f"registry check: {examined} atlases, {len(problems)} problems")
    link_problems, links, atlas_links = check_links(registry, docs_pages())
    for p in link_problems:
        print(f"  - {p}")
    print(f"docs links: {links} .html links ({atlas_links} atlas), {len(link_problems)} problems")
    return 1 if problems or link_problems or examined == 0 or atlas_links == 0 else 0


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p)
    env["PYTHONUTF8"] = "1"
    return env


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), env=_child_env(), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def _tail(text: str, n: int = 6) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return lines[-n:]


def _generator_cmd(slug: str) -> list[str]:
    if slug == README_SLUG:
        return [sys.executable, str(ROOT / "scripts" / "generate_readme_atlas.py"), "--html-only"]
    return [sys.executable, str(ROOT / "scripts" / "generate_doc_page_atlases.py"), "--slug", slug]


def _repo_guard() -> None:
    # P-025: generators put the repo root first on sys.path; an installed jaxfne must not win.
    import jaxfne

    assert "site-packages" not in jaxfne.__file__, (
        "P-025: expected the repo jaxfne on sys.path, not site-packages")


def build(slugs: list[str]) -> int:
    registry = load_registry()
    by_slug = {e["slug"]: e for e in registry["atlases"]}
    wanted = slugs or [e["slug"] for e in registry["atlases"]]
    unknown = [s for s in wanted if s not in by_slug]
    if unknown:
        raise SystemExit(f"unknown slugs {unknown}; the registry holds {sorted(by_slug)}")
    _repo_guard()
    counts = {"OK": 0, "MISMATCH": 0, "ERROR": 0}
    # One process per generator: JAX start-up dominates a per-slug process.
    page_slugs = [s for s in wanted if s != README_SLUG]
    batches = ([[README_SLUG]] if README_SLUG in wanted else []) + ([page_slugs] if page_slugs else [])
    started = time.time()
    output = ""
    process_of: dict[str, subprocess.CompletedProcess] = {}
    for batch in batches:
        cmd = _generator_cmd(batch[0]) if batch == [README_SLUG] else [
            sys.executable, str(ROOT / "scripts" / "generate_doc_page_atlases.py"), "--slug", ",".join(batch)]
        proc = _run(cmd)
        output += proc.stdout + proc.stderr
        process_of.update({s: proc for s in batch})
    secs = time.time() - started
    print(f"generators: {secs:.1f}s for {len(wanted)} atlases")
    for slug in wanted:
        entry = by_slug[slug]
        manifest_path = ROOT / entry["dir"] / "manifest.json"
        detail: list[str] = []
        failed = f"[{slug}] FAILED" in output
        if failed:
            verdict, detail = "ERROR", [ln for ln in output.splitlines() if ln.startswith(f"[{slug}] FAILED")]
        elif not manifest_path.exists() or manifest_path.stat().st_mtime < started:
            proc = process_of[slug]
            verdict = "ERROR"
            detail = [f"manifest not rewritten; generator process exited rc={proc.returncode}"]
            detail += [f"  {ln}" for ln in _tail(proc.stderr or proc.stdout)]
        else:
            fresh = read_json(manifest_path)
            detail = compare(entry, fresh)
            if detail:
                verdict = "MISMATCH"
                detail.append("fresh manifest left on disk without registry state")
            else:
                write_manifest(manifest_path, apply_state(fresh, entry))
                verdict = "OK"
        counts[verdict] += 1
        print(f"{entry['dir']}  {verdict}")
        for line in detail:
            print(f"    {line}")
    print(f"atlases: {counts['OK']} ok, {counts['MISMATCH']} mismatch, {counts['ERROR']} error")
    if counts["MISMATCH"] or counts["ERROR"]:
        print("gallery: not rewritten (an atlas failed)")
        return 1
    gallery = _run([sys.executable, str(ROOT / "scripts" / "generate_gallery.py"), "--write"])
    gallery_tail = _tail(gallery.stdout + gallery.stderr, 1)
    print(f"gallery: rc={gallery.returncode} {gallery_tail[0] if gallery_tail else ''}")
    return 0 if gallery.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--registry-write", action="store_true",
                      help="rebuild the registry from the tracked manifests")
    mode.add_argument("--build", action="store_true",
                      help="regenerate atlases and compare them with the registry")
    mode.add_argument("--check", action="store_true",
                      help="compare tracked manifests with the registry; generate nothing")
    parser.add_argument("--slug", default="", help="comma-separated registry slugs for --build (default: all)")
    args = parser.parse_args(argv)
    if args.slug and not args.build:
        parser.error("--slug applies to --build only")
    if args.registry_write:
        return write_registry()
    if args.check:
        return check()
    return build([s.strip() for s in args.slug.split(",") if s.strip()])


if __name__ == "__main__":
    raise SystemExit(main())
