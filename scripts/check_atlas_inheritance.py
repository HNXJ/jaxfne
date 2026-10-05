#!/usr/bin/env python3
"""Mechanical inheritance check: Atlas source stage objects vs registry (AT-00-R3).

Parses per-stage declared object sets from the canonical Atlas source
(artifacts/project_sources/8_atlas.md:61-65, the S-section array block) and
diffs them against the registry's STAGE_COMPONENTS
(artifacts/atlas/at_manifest.py:30-64) plus the registry's own
inheritance_check (at_manifest.py:566-588).

Parser is conservative: it accepts only the explicit ``\\begin{array}``
block containing all five stage headers (S_1, S_{2-4}, S_{5-7}, S_{8-9},
S_{10}) with comma-separated object lists. Later rows use "+" (incremental
over the previous stage), so sets are accumulated. Tokens map through the
explicit ALIASES table below; anything unmapped is a named diff, never
guessed. One descriptive phrase ("large-scale composition", S10 row) is
allowlisted as non-component prose: the registry's S10 adds only G, D over
S8-9, so treating that phrase as a component would false-fail. If the doc
has no such machine-readable block, the check reports that as the finding
instead of guessing.

Usage:
    python scripts/check_atlas_inheritance.py [--check]
        [--source artifacts/project_sources/8_atlas.md] [--root .]
Exit 0 + "inheritance VALID" on match; non-zero + named diffs otherwise
(exit 2 when the source has no parseable stage/object structure).
Read-only: never edits the doc or the registry.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not (ROOT / "pyproject.toml").is_file() or not (ROOT / "jaxfne").is_dir():
    print("inheritance INVALID: repo-root guard failed (not the jaxfne repo root)")
    sys.exit(2)

import jaxfne  # noqa: E402  (repo-root import, asserted below)

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

import artifacts.atlas.at_manifest as M  # noqa: E402

DEFAULT_SOURCE = "artifacts/project_sources/8_atlas.md"
STAGE_ORDER = ["S1", "S2-4", "S5-7", "S8-9", "S10"]

# Raw source token (after LaTeX normalization) -> canonical component name.
ALIASES: dict[str, str] = {
    "X": "X",
    "Q": "Q",
    "Phi": "Phi",
    "W": "W",
    "B": "B",
    "r": "r",
    "interaction": "interaction",
    "N": "N",
    "rho": "rho",
    "collective state": "collective_state",
    "area hierarchy": "area_hierarchy",
    "long delays": "long_delays",
    "G": "G",
    "D": "D",
}

# Source phrases that are descriptive prose, not components. Only entry:
# "large-scale composition" (S10 row) has no registry counterpart by design
# (registry S10 adds exactly G, D over S8-9).
DESCRIPTIVE: set[str] = {"large-scale composition"}

_STAGE_RE = re.compile(r"S_?\{?(\d+(?:-\d+)?)\}?")
_TEXT_RE = re.compile(r"\\text\s*\{([^}]*)\}")
_MATHBF_RE = re.compile(r"\\mathbf\s*\{?\s*([A-Za-z])\s*\}?")
_ARRAY_RE = re.compile(r"\\begin\{array\}.*?\\end\{array\}", re.DOTALL)


def _normalize_fragment(frag: str) -> str:
    frag = re.sub(r"\\end\{array\}", " ", frag)
    frag = re.sub(r"\\begin\{array\}\{[^}]*\}", " ", frag)
    frag = _TEXT_RE.sub(r"\1", frag)
    frag = _MATHBF_RE.sub(r"\1", frag)
    frag = frag.replace("\\Phi", "Phi").replace("\\rho", "rho")
    frag = frag.replace("\\,", " ").replace("\\;", " ").replace("\\:", " ")
    return frag.replace("$", " ")


def parse_source_sets(doc_path: Path) -> tuple[dict[str, set[str]], list[str]]:
    """Parse cumulative per-stage object sets from the Atlas source doc.

    Returns (sets, notes). Raises ValueError with a REPORT-style message
    when the doc has no machine-readable stage/object structure.
    Notes name descriptive phrases skipped and the anchor line used.
    """
    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"AT-00-R3: cannot read source doc {doc_path}: {exc!r}")
    candidates = _ARRAY_RE.findall(text)
    block = next(
        (b for b in candidates if "S_1" in b and "S_{10}" in b and "S_{2-4}" in b),
        "",
    )
    if not block:
        raise ValueError(
            "AT-00-R3: no machine-readable stage/object structure in "
            f"{doc_path}: no \\begin{{array}} block with explicit stage "
            "headers S_1..S_{10} and object lists (cf. 8_atlas.md:61-65). "
            "Add a machine-readable stage/object table before any check can exist."
        )
    new_tokens: dict[str, list[str]] = {}
    unmapped: list[str] = []
    notes: list[str] = []
    for row in re.split(r"\\\\", block):
        m = _STAGE_RE.search(row)
        if not m:
            continue
        stage = f"S{m.group(1)}"
        if stage not in STAGE_ORDER or stage in new_tokens:
            continue
        frag = row[m.end() :]
        frag = frag.split("&")[-1]
        toks: list[str] = []
        for raw in _normalize_fragment(frag).split(","):
            tok = raw.strip().lstrip("+").strip().rstrip(".").strip()
            if not tok:
                continue
            if tok in DESCRIPTIVE:
                notes.append(f"{stage}: descriptive phrase skipped {tok!r}")
                continue
            if tok in ALIASES:
                toks.append(ALIASES[tok])
            else:
                unmapped.append(f"{stage}: {tok!r}")
        new_tokens[stage] = toks
    missing = [s for s in STAGE_ORDER if s not in new_tokens]
    if missing:
        raise ValueError(
            f"AT-00-R3: stage/object block unparseable in {doc_path}: "
            f"stages missing explicit object lists: {missing}. "
            "Add a machine-readable stage/object table before any check can exist."
        )
    if unmapped:
        raise ValueError(
            f"AT-00-R3: unmapped source tokens in {doc_path} (not guessed): "
            f"{unmapped}. Extend ALIASES explicitly or allowlist as descriptive."
        )
    sets: dict[str, set[str]] = {}
    acc: set[str] = set()
    for stage in STAGE_ORDER:
        acc = acc | set(new_tokens[stage])
        sets[stage] = set(acc)
    anchor = next(
        (i + 1 for i, ln in enumerate(text.splitlines()) if "S_1" in ln and "X,Q" in ln),
        -1,
    )
    notes.append(f"source anchor 8_atlas.md:{anchor} (S-section array block)")
    return sets, notes


def diff_against_registry(source_sets: dict[str, set[str]]) -> list[str]:
    """Diff parsed source sets against STAGE_COMPONENTS + registry entries."""
    errors: list[str] = []
    for stage in STAGE_ORDER:
        if stage not in M.STAGE_COMPONENTS:
            errors.append(f"stage {stage}: missing from registry STAGE_COMPONENTS")
            continue
        want = set(M.STAGE_COMPONENTS[stage])
        got = source_sets[stage]
        if got != want:
            errors.append(
                f"stage {stage}: source-only {sorted(got - want)}, "
                f"registry-only {sorted(want - got)}"
            )
    for v in M.inheritance_check():
        errors.append(f"registry inheritance: {v}")
    return errors


def check(source: Path, root: Path) -> tuple[list[str], dict[str, set[str]], list[str]]:
    """Full check. Returns (errors, source_sets, notes); empty errors = VALID."""
    src = source if source.is_absolute() else root / source
    try:
        sets, notes = parse_source_sets(src)
    except ValueError as exc:
        return [str(exc)], {}, []
    return diff_against_registry(sets), sets, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atlas source-vs-registry inheritance check")
    parser.add_argument(
        "--check", action="store_true", help="CI spelling: validate, exit code only"
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Atlas source doc path")
    parser.add_argument("--root", default=".", help="repo root")
    args = parser.parse_args(argv)

    root = Path(args.root)
    errors, sets, notes = check(Path(args.source), root)
    if errors:
        no_struct = any("no machine-readable" in e or "unparseable" in e for e in errors)
        print(f"inheritance INVALID ({len(errors)} diffs):")
        for err in errors:
            print(f"  - {err}")
        return 2 if no_struct else 1
    print("inheritance VALID")
    if not args.check:
        for stage in STAGE_ORDER:
            print(f"  {stage}: {sorted(sets[stage])}")
        for note in notes:
            print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
