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
                            figure state and promotion into it. A full build (no --slug)
                            also runs each etude generator once and checks its HTML.
                            Then rewrite gallery.md
    --check                 compare the tracked manifests with the registry; generate nothing
    --site OUT [--only a,b] build the standalone figure site: every atlas panel, every
                            etude panel and the docs visuals as OUT/<slug>/<panel>.html
                            (standalone page of the panel's Figure) plus .png stills
                            and a manifest.json. Never writes under docs/;
                            --build/--check are unchanged
    --docs-figures-check    fail on any docs figure iframe outside the allowlist
                            or any docs link to a docs/_static figure page

Slugs are registry slugs: the atlas/<slug> directories, ``three_area`` for
atlas_three_area, and ``readme`` for the top-level atlas/ (the canonical README atlas,
built by generate_readme_atlas.py --html-only), plus the four registry etude slugs
(``experiment_a``, ``hdp_controllability_reachability``,
``multiscale_observation``, ``heterogeneous_emitters``).

Usage:
    python scripts/build_docs_assets.py --registry-write
    python scripts/build_docs_assets.py --build --slug two_neuron_ei,three_area
    python scripts/build_docs_assets.py --check
    python scripts/build_docs_assets.py --docs-figures-check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
ETUDE_KEYS = ("slug", "dir", "generator", "html")

# Standalone figure site (Phase A of the docs-figures plan): every atlas panel
# and the docs visuals as OUT/<slug>/<panel>.html (full standalone page of the
# panel's Plotly figure, plotly.js from the CDN) plus OUT/<slug>/<panel>.png
# (the same Figure through exporters._write_image). PNG raster size is fixed;
# the kaleido scale is its default of 1.
FIG_PNG_WIDTH = 1000
FIG_PNG_HEIGHT = 760
FIG_PNG_SCALE = 1
VISUAL_SLUGS = ("column_network", "area_graph_n20")
SITE_SCHEMA = "figsite.v1"

# Public figure site: every docs still and "Interactive version" link points
# here (published by the Dashboard workflow's figures job from --site output).
FIGURES_URL = "https://hnxj.github.io/jaxfne/figures/"

# Etude slug -> mode key in scripts/rerun_etude_figures.py MODES, except
# experiment_a, whose panels come from scripts/plot_frozen_etude_panels.py.
ETUDE_SITE_MODES = {
    "experiment_a": "experiment_a",
    "hdp_controllability_reachability": "hdp_mcc3",
    "multiscale_observation": "multiscale",
    "heterogeneous_emitters": "heterogeneous",
}


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
    if "etudes" not in registry:
        raise SystemExit(f"{rel(REGISTRY)}: no etudes list; restore it from git")
    for e in registry["etudes"]:
        missing = [k for k in ETUDE_KEYS if k not in e]
        if missing:
            raise SystemExit(f"{rel(REGISTRY)}: etude {e.get('slug')!r} lacks {missing}")
    for key in ("slug", "dir"):
        values = [e[key] for e in registry["atlases"] + registry["etudes"]]
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            raise SystemExit(f"{rel(REGISTRY)}: duplicate {key} entries {dupes}")
    copies = registry.get("copies", [])
    if not isinstance(copies, list):
        raise SystemExit(f"{rel(REGISTRY)}: copies is not a list: {copies!r}")
    seen = set()
    for entry in copies:
        if not isinstance(entry, dict) or "dest" not in entry or "source" not in entry:
            raise SystemExit(f"{rel(REGISTRY)}: malformed copies entry {entry!r}; need dest and source")
        if not entry["dest"].startswith("docs/"):
            raise SystemExit(f"{rel(REGISTRY)}: copies dest outside docs/ {entry!r}")
        if entry["dest"] in seen:
            raise SystemExit(f"{rel(REGISTRY)}: duplicate copies dest {entry['dest']!r}")
        seen.add(entry["dest"])
        if not (ROOT / entry["source"]).is_file():
            raise SystemExit(f"{rel(REGISTRY)}: copies source missing on disk {entry!r}")
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
    # Etudes and copies are declared by hand, not read from manifests: keep the tracked lists.
    tracked = read_json(REGISTRY) if REGISTRY.exists() else {}
    doc = {
        "schema": SCHEMA,
        "built_from": "tracked manifests (build_docs_assets.py --registry-write)",
        "compared_fields": list(COMPARED_FIELDS),
        "atlases": entries,
        "etudes": tracked.get("etudes", []),
    }
    if "copies" in tracked:
        doc["copies"] = tracked["copies"]
    REGISTRY.write_bytes((json.dumps(doc, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8"))
    print(f"registry: {len(entries)} atlases, {len(doc['etudes'])} etudes -> {rel(REGISTRY)}")
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
    """HTML files a build writes into one directory: an etude's listed html, or an atlas's panels and index.html."""
    if "html" in entry:
        return set(entry["html"])
    return set(entry["panels"]) | {"index.html"}


def unwritten(entry: dict, since: float) -> list[str]:
    """Registry HTML files of one directory that are absent or older than ``since``."""
    atlas_dir = ROOT / entry["dir"]
    return sorted(name for name in built_files(entry)
                  if not (atlas_dir / name).is_file() or (atlas_dir / name).stat().st_mtime < since)


def check_links(registry: dict, pages: dict[Path, str]) -> tuple[list[str], int, int]:
    """Every relative .html link in the docs pages must resolve.

    A link into an atlas or registered etude directory names a build product, so it is
    checked against the registry; any other link must name a file on disk. Returns the
    problems, the links examined and, of those, the atlas links.
    """
    entries = {e["dir"]: e for e in registry["atlases"] + registry["etudes"]}
    etude_dirs = [ROOT / e["dir"] for e in registry["etudes"]]
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
            elif any(d == target.parent or d in target.parents for d in etude_dirs):
                entry = entries.get(rel(target.parent))
                ok = entry is not None and target.name in built_files(entry)
                why = "is not an etude file the build writes"
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
    copy_problems = check_copies(registry)
    for p in copy_problems:
        print(f"  - {p}")
    print(f"copies: {len(registry.get('copies', []))} checked, {len(copy_problems)} problems")
    failed = bool(problems or link_problems or copy_problems)
    # atlas_links is informational only: after the docs-figures rewrite every
    # figure link is an absolute figure-site URL, so zero relative atlas links
    # is the expected steady state, not a failure.
    return 1 if failed or examined == 0 else 0


def run_copies(registry: dict) -> int:
    """Copy each registry source to its dest byte-exact. Returns the number of errors."""
    copies = registry.get("copies", [])
    errors = 0
    for entry in copies:
        try:
            dest = ROOT / entry["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / entry["source"], dest)
        except OSError as exc:
            errors += 1
            print(f"  ERROR {entry['dest']}: copy from {entry['source']} failed: {exc}")
    if errors:
        print(f"copies: {len(copies) - errors} ok, {errors} error")
    else:
        print(f"copies: {len(copies)} ok")
    return errors


def check_copies(registry: dict) -> list[str]:
    """Dests that exist but differ from their source.

    A missing dest is an unbuilt product, not an error, as with atlas HTML.
    """
    problems = []
    for entry in registry.get("copies", []):
        dest = ROOT / entry["dest"]
        if dest.is_file() and dest.read_bytes() != (ROOT / entry["source"]).read_bytes():
            problems.append(f"ERROR {entry['dest']}: differs from {entry['source']}")
    return problems


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


def _etude_cmd(entry: dict) -> list[str]:
    script, *args = entry["generator"]
    return [sys.executable, str(ROOT / script), *args]


def run_etudes(registry: dict) -> int:
    """Run each etude generator once. An etude is OK when it exits 0 and writes every listed HTML file.

    Returns the number of etudes in ERROR.
    """
    errors = 0
    for entry in registry["etudes"]:
        manifest_path = ROOT / entry["dir"] / "manifest.json"
        # The generators reset the figure state; the tracked manifest's state is kept, as for atlases.
        state = {k: read_json(manifest_path)[k] for k in STATE_FIELDS}
        started = time.time()
        proc = _run(_etude_cmd(entry))
        secs = time.time() - started
        if proc.returncode != 0:
            verdict = "ERROR"
            detail = [f"generator rc={proc.returncode}"]
            detail += [f"  {ln}" for ln in _tail(proc.stderr or proc.stdout)]
        elif stale := unwritten(entry, started):
            verdict, detail = "ERROR", [f"HTML not written by this build: {stale}"]
        else:
            write_manifest(manifest_path, apply_state(read_json(manifest_path), state))
            verdict, detail = "OK", []
        if verdict == "ERROR":
            errors += 1
        print(f"{entry['dir']}  {verdict}  ({secs:.1f}s)")
        for line in detail:
            print(f"    {line}")
    print(f"etudes: {len(registry['etudes']) - errors} ok, {errors} error")
    return errors


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
        elif stale := unwritten(entry, started):
            verdict, detail = "ERROR", [f"HTML not written by this build: {stale}"]
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
    if slugs:
        etude_errors = 0
        print("etudes: skipped (--slug builds atlases only)")
    else:
        etude_errors = run_etudes(registry)
    copy_errors = run_copies(registry)
    if counts["MISMATCH"] or counts["ERROR"]:
        print("gallery: not rewritten (an atlas failed)")
        return 1
    gallery = _run([sys.executable, str(ROOT / "scripts" / "generate_gallery.py"), "--write"])
    gallery_tail = _tail(gallery.stdout + gallery.stderr, 1)
    print(f"gallery: rc={gallery.returncode} {gallery_tail[0] if gallery_tail else ''}")
    return 0 if gallery.returncode == 0 and not etude_errors and not copy_errors else 1


def site_slugs(registry: dict) -> list[str]:
    """Every atlas slug in build order, then every etude slug, then the docs visuals."""
    return ([e["slug"] for e in registry["atlases"]] + [e["slug"] for e in registry["etudes"]]
            + list(VISUAL_SLUGS))


def pages_showing(target: str, pages: dict[Path, str],
                  figure_urls: tuple[str, ...] = ()) -> list[str]:
    """Docs pages holding an .html link that resolves to ``target`` (repo-relative).

    ``figure_urls`` are absolute figure-site URLs for the same figure; after the
    docs-figures rewrite the pages hold those instead of the relative link.
    """
    hits = []
    for page, text in pages.items():
        if any(url in text for url in figure_urls):
            hits.append(rel(page))
            continue
        for _offset, link, raw_html in html_links(text):
            if rel(Path(os.path.normpath(link_base(page, raw_html) / link))) == target:
                hits.append(rel(page))
                break
    return sorted(hits)


def _git_head() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _site_figure_source(slug: str):
    """(kind, title, model, signals, render_kw, doc_dir) for one site slug.

    Atlas slugs re-simulate their registry spec; the visuals reuse their
    generator's figure builder. Nothing is written here.
    """
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    if slug in VISUAL_SLUGS:
        import generate_docs_visuals as gv

        fig = gv.FIGURES[slug]()
        title = str(getattr(fig.layout.title, "text", "") or slug)
        return ("visual", title, None, None, {}, "docs/_static/visuals", {f"{slug}.html": fig})
    import generate_doc_page_atlases as dpa

    if slug == README_SLUG:
        import generate_readme_atlas as gra

        model, signals = gra.build_canonical_model()
        title = gra.CANONICAL_TITLE
        kw: dict = {"duration_ms": gra.CANONICAL_DURATION_MS,
                    "dt_ms": gra.CANONICAL_DT_MS, "seed": gra.CANONICAL_SEED}
    else:
        title, model, signals, kw = dpa.SPECS[slug]()
    return ("atlas", title, model, signals, kw, "", {})


def _build_site_slug(slug: str, out: Path, registry: dict, pages: dict[Path, str]) -> list[dict]:
    """Write OUT/<slug>/<panel>.html + .png; return the slug's manifest entries."""
    from jaxfne.vis.atlas_suite import render_atlas
    from jaxfne.vis.exporters import _write_image

    kind, title, model, signals, kw, doc_dir, visual_figs = _site_figure_source(slug)
    by_slug = {e["slug"]: e for e in registry["atlases"]}
    slug_dir = out / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if kind == "visual":
        panels = [{"file": f"{slug}.html", "panel": slug, "status": "AVAILABLE"}]
        figures = visual_figs
        doc_targets = {f"{slug}.html": f"{doc_dir}/{slug}.html"}
        wrapped: dict[str, Path] = {}
    else:
        tmpdir = tempfile.TemporaryDirectory()
        manifest, figures = render_atlas(model, signals, out_dir=tmpdir.name, title=title,
                                         return_figures=True, **kw)
        panels = manifest["panels"]
        entry = by_slug[slug]
        doc_targets = {p["file"]: f"{entry['dir']}/{p['file']}" for p in panels}
        wrapped = {p["file"]: Path(tmpdir.name) / p["file"] for p in panels}
    for panel in panels:
        name = panel["file"]
        stem = name[:-len(".html")] if name.endswith(".html") else name
        fig = figures.get(name)
        html_rel = f"{slug}/{name}"
        label = panel.get("panel", stem)
        docs = pages_showing(doc_targets[name], pages)
        if fig is None:
            # No Figure to stand alone: keep the existing wrapped omission page.
            shutil.copyfile(wrapped[name], slug_dir / name)
            entries.append({"slug": slug, "panel": stem, "html": html_rel,
                            "title": label, "status": "omitted", "docs_pages": docs})
            continue
        fig.write_html(str(slug_dir / name), include_plotlyjs="cdn", full_html=True)
        fig.update_layout(width=FIG_PNG_WIDTH, height=FIG_PNG_HEIGHT)
        # Kaleido's own default scale is 1 == FIG_PNG_SCALE; _write_image only
        # retries its teardown failure (P-011).
        _write_image(fig, str(slug_dir / f"{stem}.png"))
        if kind == "visual":
            label = title
        entries.append({"slug": slug, "panel": stem, "html": html_rel,
                        "png": f"{slug}/{stem}.png", "title": label,
                        "status": "available", "docs_pages": docs})
    if kind != "visual":
        tmpdir.cleanup()
    return entries


def _build_site_etude_slug(slug: str, out: Path, registry: dict,
                          pages: dict[Path, str]) -> list[dict]:
    """Write OUT/<slug>/<panel>.html + .png from the etude's own figure builders.

    experiment_a replays its frozen bundle (no resimulation); the other etudes
    rerun their deterministic figure modes with the figures handed back instead
    of only written to disk.
    """
    from jaxfne.vis.exporters import _write_image

    slug_dir = out / slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    figures: dict = {}
    tmpdir = tempfile.TemporaryDirectory()
    try:
        if slug == "experiment_a":
            from scripts.plot_frozen_etude_panels import panels_experiment_a

            panels = panels_experiment_a(ROOT / "artifacts" / "etudes" / "experiment_a",
                                         Path(tmpdir.name), figures=figures)
        else:
            from scripts.rerun_etude_figures import MODES

            key = ETUDE_SITE_MODES[slug]
            built_slug, fn = MODES[key]
            assert built_slug == slug, f"etude site mode {key} builds {built_slug}, not {slug}"
            panels = fn(Path(tmpdir.name), figures=figures)
        entry = {e["slug"]: e for e in registry["etudes"]}[slug]
        entries: list[dict] = []
        for panel in panels:
            name = panel["file"]
            stem = name[:-len(".html")] if name.endswith(".html") else name
            fig = figures.get(name)
            label = panel.get("panel", stem)
            docs = pages_showing(
                f"{entry['dir']}/{name}", pages,
                (f"{FIGURES_URL}{slug}/{stem}.html", f"{FIGURES_URL}{slug}/{stem}.png"))
            if fig is None:
                # No Figure to stand alone: keep the existing wrapped omission page.
                shutil.copyfile(Path(tmpdir.name) / name, slug_dir / name)
                entries.append({"slug": slug, "panel": stem, "html": f"{slug}/{name}",
                                "title": label, "status": "omitted", "docs_pages": docs})
                continue
            fig.write_html(str(slug_dir / name), include_plotlyjs="cdn", full_html=True)
            fig.update_layout(width=FIG_PNG_WIDTH, height=FIG_PNG_HEIGHT)
            # Kaleido's own default scale is 1 == FIG_PNG_SCALE; _write_image only
            # retries its teardown failure (P-011).
            _write_image(fig, str(slug_dir / f"{stem}.png"))
            entries.append({"slug": slug, "panel": stem, "html": f"{slug}/{name}",
                            "png": f"{slug}/{stem}.png", "title": label,
                            "status": "available", "docs_pages": docs})
        return entries
    finally:
        tmpdir.cleanup()


def check_site_figures(site: Path) -> list[str]:
    """Every non-omitted figure needs its html and a non-empty png."""
    manifest = read_json(Path(site) / "manifest.json")
    problems = []
    for fig in manifest.get("figures", []):
        label = f"{fig.get('slug')}/{fig.get('panel')}"
        html = Path(site) / str(fig.get("html", ""))
        if not html.is_file():
            problems.append(f"{label}: html missing at {fig.get('html')}")
        if fig.get("status") == "omitted":
            continue
        png = Path(site) / str(fig.get("png", ""))
        if not png.is_file():
            problems.append(f"{label}: png missing at {fig.get('png')}")
        elif png.stat().st_size == 0:
            problems.append(f"{label}: png empty at {fig.get('png')}")
    return problems


# Docs-figures migration steady state: no figure iframe may remain (the four
# unmapped pages below are left as-is), and no relative link may point at a
# figure page (atlas, etude or visual dirs); those live on the figure site.
ALLOWLISTED_FIGURE_IFRAMES = {
    "docs/assets/interactive/v037_source_column_3d.html",
    "docs/assets/interactive/izhikevich_single_emitter_explorer.html",
    "docs/_static/jaxley_interop/vm_dark.html",
    "docs/assets/showcases/v1_v4_pfc_network3d.html",
}
_IFRAME_SRC_RE = re.compile(r"<iframe\b[^>]*\bsrc\s*=\s*[\"']([^\"'#?]+?\.html)",
                            re.IGNORECASE | re.DOTALL)


def docs_figures_check() -> int:
    """Fail on a non-allowlisted docs iframe or a relative figure-page link."""
    registry = load_registry()
    figure_dirs = ({e["dir"] for e in registry["atlases"]}
                   | {e["dir"] for e in registry["etudes"]} | {"docs/_static/visuals"})
    pages = docs_pages()
    iframe_hits = link_hits = 0
    problems: list[str] = []
    for page in sorted(pages):
        text = pages[page]
        for m in _IFRAME_SRC_RE.finditer(text):
            src = m.group(1).strip()
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:|^/", src) or not src.endswith(".html"):
                continue
            target = rel(Path(os.path.normpath(link_base(page, True) / src)))
            iframe_hits += 1
            if target not in ALLOWLISTED_FIGURE_IFRAMES:
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{rel(page)}:{line}: iframe {src} not in allowlist")
        for off, link, raw in html_links(text):
            target = Path(os.path.normpath(link_base(page, raw) / link))
            if rel(target.parent) not in figure_dirs or not link.endswith(".html"):
                continue
            link_hits += 1
            line = text.count("\n", 0, off) + 1
            problems.append(f"{rel(page)}:{line}: figure link {link} must be a figure-site URL")
    for p in problems:
        print(f"  - {p}")
    print(f"docs-figures-check: {iframe_hits} iframes, {link_hits} figure links, "
          f"{len(problems)} problems in {len(pages)} pages")
    return 1 if problems else 0


def check_cited_figures(entries: list[dict], slugs: list[str],
                        pages: dict[Path, str]) -> list[str]:
    """Every figure-site URL a docs page cites for a built slug must be an available file.

    The completeness check alone passed a build whose rerun etudes were all
    omitted while the docs showed their stills.
    """
    held = {e["html"] for e in entries if e["status"] == "available"}
    held |= {e["png"] for e in entries if e["status"] == "available"}
    url = re.compile(re.escape(FIGURES_URL) + r"(?P<slug>[^/)\s\"']+)/(?P<file>[^)\s\"'#?]+)")
    problems = []
    for page, text in pages.items():
        for m in url.finditer(text):
            target = f"{m['slug']}/{m['file']}"
            if m["slug"] in slugs and target not in held:
                problems.append(f"{rel(page)}: cites {target}, which this build did not produce")
    return sorted(set(problems))


def build_site(site: str | Path, only: list[str]) -> int:
    """Build the standalone figure site into OUT (never under docs/)."""
    registry = load_registry()
    known = site_slugs(registry)
    wanted = only or known
    unknown = [s for s in wanted if s not in known]
    if unknown:
        raise SystemExit(f"unknown site slugs {unknown}; known slugs {known}")
    _repo_guard()
    out = Path(site)
    pages = docs_pages()
    build_sha = _git_head()
    etude_slugs = {e["slug"] for e in registry["etudes"]}
    entries: list[dict] = []
    errors = 0
    for slug in wanted:
        try:
            if slug in etude_slugs:
                entries.extend(_build_site_etude_slug(slug, out, registry, pages))
            else:
                entries.extend(_build_site_slug(slug, out, registry, pages))
            print(f"{slug}: {len([e for e in entries if e['slug'] == slug])} figures")
        except Exception as exc:  # noqa: BLE001 -- one slug must not kill the site
            errors += 1
            print(f"[{slug}] SITE ERROR: {exc!r}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps({"schema": SITE_SCHEMA, "build_sha": build_sha,
                    "figures": entries}, indent=2) + "\n", encoding="utf-8")
    problems = check_site_figures(out) + check_cited_figures(entries, wanted, pages)
    for p in problems:
        print(f"  - {p}")
    print(f"site: {len(entries)} figures, {errors} errors, "
          f"{len(problems)} completeness problems -> {out}")
    return 1 if errors or problems else 0


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
    mode.add_argument("--site", metavar="OUT", default=None,
                      help="build the standalone figure site (HTML + PNG + manifest) into OUT")
    mode.add_argument("--docs-figures-check", action="store_true",
                      help="fail on a non-allowlisted docs iframe or relative figure link")
    parser.add_argument("--slug", default="", help="comma-separated registry slugs for --build (default: all)")
    parser.add_argument("--only", default="",
                        help="comma-separated site slugs for --site (default: all atlases + etudes + visuals)")
    args = parser.parse_args(argv)
    if args.slug and not args.build:
        parser.error("--slug applies to --build only")
    if args.only and args.site is None:
        parser.error("--only applies to --site only")
    if args.registry_write:
        return write_registry()
    if args.check:
        return check()
    if args.docs_figures_check:
        return docs_figures_check()
    if args.site is not None:
        return build_site(args.site, [s.strip() for s in args.only.split(",") if s.strip()])
    return build([s.strip() for s in args.slug.split(",") if s.strip()])


if __name__ == "__main__":
    raise SystemExit(main())
