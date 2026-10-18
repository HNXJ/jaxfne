#!/usr/bin/env python3
"""Record-integrity gate: past records may move but their bytes may not change.

A record is any file in one of these classes under ``artifacts/``:

- ``receipts``: a release receipt with a final JSON ``status`` (``published``
  or ``sealed_published``, case-insensitive; observed non-final values such as
  ``C_R_IDENTIFIED_AWAITING_RELEASE_AUTHORIZATION`` and
  ``gates_pass_not_sealed`` are not records) or a ``*_seal_notes*.md`` file;
- ``archive``: anything under ``artifacts/archive/``;
- ``legacy``: anything under ``artifacts/legacy/``;
- ``dated``: any path containing a date (``20YY-MM-DD``);
- ``prereg``: any path containing ``prereg`` (case-insensitive);
- ``benchmark``: ``artifacts/**/benchmark/agent_tasks_*.json``;
- ``frozen``: every path key in ``artifacts/publication/frozen_manifest.json``.

``NON_RECORD_PATHS`` below excludes class matches that are not records. The
2026-10-15 enumeration leaves exactly the three live ``.py`` scripts: each is
a generator or reader of records, never a record itself, and each entry cites
the code line that shows it. Prospective-run executor outputs under a
preregistration (``v2_runs/rep_*.json``, confirmatory receipts, ``e2d_table.*``,
amendment receipts) are records and stay pinned. The dated class holds no live
script and no file current code writes, so it contributes no entry.

The pinned state in ``scripts/record_hashes.json`` maps each record path to
its content sha256, plus per-class counts. A pinned path that exists with
different bytes fails (``changed-record``: this catches a swap of two records
inside one class, which a hash multiset cannot see). A pinned path that is
missing passes only as a move: its hash sits at a target the class discovery
itself matches for at least one of the record's pinned classes, outside
``NON_RECORD_PATHS`` and outside the pinned paths (reported as
``moved-record`` so ``--add`` can repin it). Any other destination fails as
``declassified-move``: a target no class matches, an excluded script, another
pinned path, or a discovered path sharing no pinned class with the record.
When the hash is nowhere it is ``deleted-record``. A live record pinned
nowhere is ``unpinned-hash``. Per-class live counts must equal the pinned
counts: below the pin is a deletion or a demotion (e.g. a manifest key removed
around the file), above is unpinned; both fail as ``class-count-mismatch``
naming the class.

``--add`` takes explicit paths, runs the check first, and refuses when any
violation or unlisted move exists beyond the unpinned-hash or moved entries
for exactly the listed paths. So ``mv a b``, edit b, ``--add b`` is refused
(``a`` is missing and its hash is nowhere). Nothing is written on refusal.

Usage:
    python scripts/check_record_integrity.py                 # report
    python scripts/check_record_integrity.py --check         # exit 1 on violation
    python scripts/check_record_integrity.py --add PATH ...  # pin listed records
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHES_PATH = Path(__file__).resolve().parent / "record_hashes.json"

# Receipt JSON statuses that mark a record as final (compared case-insensitively).
FINAL_STATUSES = {"published", "sealed_published"}

DATE_IN_NAME = re.compile(r"20\d{2}-\d{2}-\d{2}")

CLASSES = (
    "receipts",
    "archive",
    "legacy",
    "dated",
    "prereg",
    "benchmark",
    "frozen",
)

# Class matches that are live code operating on records, never records.
# Each entry cites the line proving it generates or reads records:
NON_RECORD_PATHS = frozenset(
    [
        # Writes its receipt: e2a_blinded_executor.py:159 (out_path.write_text).
        "artifacts/e2/preregistration/E2a_search/e2a_blinded_executor.py",
        # Reads receipts to audit the freeze: freeze_audit.py:6 (receipt_path.read_bytes).
        "artifacts/e2/preregistration/E2a_search/freeze_audit.py",
        # Writes receipts and rate windows: e2b_v1_executor.py:413 and
        # e2b_v1_executor.py:427 (write_text), e2b_v1_executor.py:354
        # (savez_compressed).
        "artifacts/e2/preregistration/E2b_confirmatory/e2b_v1_executor.py",
    ]
)


def sha256_bytes(data: bytes) -> str:
    """Content hash of one record (raw bytes; moves keep the digest)."""
    return hashlib.sha256(data).hexdigest()


def _frozen_paths(repo_root: Path = ROOT) -> set[str]:
    """Path keys of the publication frozen manifest (posix, repo-relative)."""
    manifest = repo_root / "artifacts" / "publication" / "frozen_manifest.json"
    try:
        return set(json.loads(manifest.read_text(encoding="utf-8"))["files"])
    except (OSError, ValueError, KeyError, AttributeError):
        return set()


def _is_final_receipt(path: Path) -> bool:
    """True when the receipt JSON carries a final ``status``.

    Fail-closed: a receipt that cannot be parsed, or carries no ``status``,
    is not a record (a corrupt or half-written receipt must not be pinned).
    """
    try:
        status = json.loads(path.read_text(encoding="utf-8")).get("status", "")
    except (OSError, ValueError, AttributeError):
        return False
    return str(status).strip().lower() in FINAL_STATUSES


def _classes_of_rel(rel: str, frozen_keys: set[str]) -> list[str]:
    """Path-predicate classes of one repo-relative posix path.

    Mirrors the path-based half of discovery. Receipt globs match by name;
    final-status filtering needs file content, so a missing pinned file that
    matches a receipt glob still counts as ``receipts`` (it was pinned as one).
    """
    classes: list[str] = []
    base = rel.rsplit("/", 1)[-1]
    if rel.startswith("artifacts/") and (
        fnmatch.fnmatch(base, "v*_release_receipt.json")
        or fnmatch.fnmatch(base, "*_seal_notes*.md")
    ):
        classes.append("receipts")
    if rel.startswith("artifacts/archive/"):
        classes.append("archive")
    if rel.startswith("artifacts/legacy/"):
        classes.append("legacy")
    if DATE_IN_NAME.search(rel):
        classes.append("dated")
    if "prereg" in rel.lower():
        classes.append("prereg")
    if fnmatch.fnmatch(rel, "artifacts/**/benchmark/agent_tasks_*.json"):
        classes.append("benchmark")
    if rel in frozen_keys:
        classes.append("frozen")
    return classes


def discover_records_by_class(repo_root: Path = ROOT) -> dict[str, list[Path]]:
    """Return the sorted record paths under ``artifacts/``, keyed by class."""
    base = repo_root / "artifacts"
    frozen = _frozen_paths(repo_root)
    by_class: dict[str, list[Path]] = {name: [] for name in CLASSES}
    if not base.is_dir():
        return by_class
    for pattern, needs_final in (
        ("**/v*_release_receipt.json", True),
        ("**/*_seal_notes*.md", False),
    ):
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            if needs_final and not _is_final_receipt(path):
                continue
            by_class["receipts"].append(path)
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(repo_root).as_posix()
        if rel in NON_RECORD_PATHS:
            continue
        for name in _classes_of_rel(rel, frozen):
            if name != "receipts" and name != "frozen":
                by_class[name].append(path)
    for rel in sorted(frozen):
        if rel in NON_RECORD_PATHS:
            continue
        path = repo_root / rel
        if path.is_file():
            by_class["frozen"].append(path)
    for name in CLASSES:
        by_class[name] = sorted(by_class[name])
    return by_class


def discover_records(repo_root: Path = ROOT) -> list[Path]:
    """Return the sorted union of all record paths under ``artifacts/``."""
    seen: dict[str, Path] = {}
    for paths in discover_records_by_class(repo_root).values():
        for path in paths:
            seen[path.relative_to(repo_root).as_posix()] = path
    return [seen[rel] for rel in sorted(seen)]


def _load_pinned(hashes_path: Path) -> dict[str, dict]:
    """Pinned ``{"paths": {rel: sha}, "counts": {class: n}}`` (fail-closed)."""
    try:
        raw = json.loads(hashes_path.read_text(encoding="utf-8"))
        paths = raw.get("paths")
        counts = raw.get("counts")
        if not isinstance(paths, dict) or not isinstance(counts, dict):
            return {"paths": {}, "counts": {}}
        return {
            "paths": {str(r): str(h) for r, h in paths.items()},
            "counts": {str(c): int(n) for c, n in counts.items()},
        }
    except (OSError, ValueError, AttributeError, TypeError):
        return {"paths": {}, "counts": {}}


def load_pinned() -> dict[str, dict]:
    """Pinned path map and per-class counts (fail-closed on unreadable)."""
    return _load_pinned(HASHES_PATH)


def pinned_counts_for(paths: dict[str, str], frozen_keys: set[str]) -> dict[str, int]:
    """Per-class counts of pinned paths (missing files keep their rel class)."""
    counts = {name: 0 for name in CLASSES}
    for rel in paths:
        for name in _classes_of_rel(rel, frozen_keys):
            counts[name] += 1
    return counts


def _hash_locations_all(repo_root: Path) -> dict[str, list[str]]:
    """Map sha256 to sorted paths for every file under artifacts/.

    Unlike the move scan this includes pinned and excluded paths, so bytes
    laundered onto another pinned record or an excluded script are still
    found; callers split valid move targets from violations.
    """
    base = repo_root / "artifacts"
    found: dict[str, list[str]] = {}
    if not base.is_dir():
        return found
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(repo_root).as_posix()
        try:
            digest = sha256_bytes(path.read_bytes())
        except OSError:
            continue
        found.setdefault(digest, []).append(rel)
    return found


def _is_valid_move_target(
    new: str,
    old_classes: list[str],
    frozen_keys: set[str],
    pinned_rels: set[str],
) -> bool:
    """True when ``new`` keeps the record inside its pinned classes.

    The target must be unpinned, outside ``NON_RECORD_PATHS``, and matched by
    the class discovery for at least one of the record's pinned classes. A
    frozen-manifest move counts ``frozen`` among the record's classes when the
    manifest now points at the new path.
    """
    if new in NON_RECORD_PATHS or new in pinned_rels:
        return False
    new_classes = _classes_of_rel(new, frozen_keys)
    if new in frozen_keys and "frozen" not in old_classes:
        old_classes = [*old_classes, "frozen"]
    return any(name in new_classes for name in old_classes)


def _analyze(repo_root: Path, pinned: dict[str, dict]) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (violations, moves) for one tree against pinned paths.

    - ``changed-record``: a pinned path exists with different bytes (names the
      path with pinned and live hashes; a within-class swap yields two);
    - ``unlisted-record``: the bytes are unchanged but the path is no longer
      discovered (e.g. a frozen manifest key removed around the file);
    - ``deleted-record``: a pinned path is gone and its hash is at no
      other path (names the class);
    - ``declassified-move``: the hash sits at a path that is no valid move
      target (outside every pinned class, an excluded script, another pinned
      path, or a discovered path sharing no pinned class);
    - ``unpinned-hash``: a live record pinned nowhere (names the file);
    - ``class-count-mismatch``: per-class live count differs from the pinned
      count (below is a deletion or demotion, above is unpinned);
    - moves ``(old, new)``: a missing pinned path whose hash sits at a valid
      target; reported, not violations.
    """
    paths: dict[str, str] = pinned.get("paths", {})
    counts: dict[str, int] = pinned.get("counts", {})
    frozen_keys = _frozen_paths(repo_root)
    by_class = discover_records_by_class(repo_root)
    live = {
        p.relative_to(repo_root).as_posix(): sha256_bytes(p.read_bytes())
        for ps in by_class.values()
        for p in ps
    }
    violations: list[str] = []
    moves: list[tuple[str, str]] = []
    locations: dict[str, list[str]] | None = None
    for old in sorted(paths):
        want = paths[old]
        if old in live:
            if live[old] != want:
                violations.append(f"changed-record: {old} pinned={want} live={live[old]}")
            continue
        disk = repo_root / old
        if disk.is_file():
            try:
                disk_hash = sha256_bytes(disk.read_bytes())
            except OSError:
                disk_hash = None
            if disk_hash == want:
                violations.append(
                    f"unlisted-record: {old} sha256={want}"
                    " (bytes unchanged but path is no longer a record)"
                )
            elif disk_hash is not None:
                violations.append(f"changed-record: {old} pinned={want} live={disk_hash}")
            continue
        if locations is None:
            locations = _hash_locations_all(repo_root)
        old_classes = _classes_of_rel(old, frozen_keys)
        found = [rel for rel in locations.get(want, []) if rel != old]
        targets = [
            rel
            for rel in found
            if rel not in paths and _is_valid_move_target(rel, old_classes, frozen_keys, set(paths))
        ]
        for new in targets:
            moves.append((old, new))
        for rel in found:
            if rel in targets:
                continue
            if rel in paths and live.get(rel, None) == paths[rel]:
                # An unchanged pinned path holding identical bytes does not
                # rescue the missing record; the deletion stands below.
                continue
            violations.append(
                f"declassified-move: {old} -> {rel} sha256={want}"
                " (not a class-preserving move target)"
            )
        if not found or all(rel in paths and live.get(rel, None) == paths[rel] for rel in found):
            classes = _classes_of_rel(old, frozen_keys)
            label = f" class={classes[0]}" if classes else ""
            violations.append(f"deleted-record:{label} {old} sha256={want}")
    moved_new = {new for _, new in moves}
    for rel in sorted(live):
        if rel in paths or rel in moved_new:
            continue
        violations.append(f"unpinned-hash: {rel} sha256={live[rel]}")
    for name in CLASSES:
        live_n = len(by_class[name])
        pin_n = counts.get(name, 0)
        if live_n != pin_n:
            violations.append(f"class-count-mismatch: class={name} pinned={pin_n} live={live_n}")
    return violations, moves


def find_violations(repo_root: Path = ROOT, pinned: dict[str, dict] | None = None) -> list[str]:
    """Human-readable violations for one tree (moves excluded)."""
    owned = pinned if pinned is not None else load_pinned()
    violations, _ = _analyze(repo_root, owned)
    return violations


def check_tree(repo_root: Path = ROOT, pinned: dict[str, dict] | None = None) -> list[str]:
    """All violations (changed, moved-away, or deleted bytes) for one tree."""
    return find_violations(repo_root, pinned)


def _allowed_for_add(
    violations: list[str],
    moves: list[tuple[str, str]],
    listed: set[str],
) -> tuple[bool, list[str]]:
    """True when every finding is covered by exactly the listed paths."""
    blocking: list[str] = []
    deferred: list[str] = []
    for violation in violations:
        if violation.startswith("unpinned-hash:"):
            rel = violation.split("unpinned-hash: ", 1)[1].split(" sha256=", 1)[0]
            if rel in listed:
                continue
        elif violation.startswith("class-count-mismatch:"):
            # A count delta caused only by the listed adds/repins reconciles
            # when the merge recomputes counts; judge it after the rest.
            deferred.append(violation)
            continue
        blocking.append(violation)
    for old, new in moves:
        if new not in listed:
            blocking.append(f"unlisted-move: {old} -> {new} (not in --add paths)")
    if blocking:
        blocking.extend(deferred)
    return (not blocking), blocking


def plan_add(
    repo_root: Path, rels: list[str], pinned: dict[str, dict]
) -> tuple[list[str], dict[str, dict], list[str]]:
    """Validate explicit ``--add`` paths; return (ok, merged, refusals).

    The check runs first: any violation or move beyond the unpinned-hash or
    moved entries for exactly the listed paths refuses the whole ``--add``
    (so ``mv a b``, edit b, ``--add b`` is refused: ``a`` is missing and its
    hash is nowhere). On any refusal the merged state is still returned but
    the caller must not write it.
    """
    violations, moves = _analyze(repo_root, pinned)
    ok, blocking = _allowed_for_add(violations, moves, set(rels))
    if not ok:
        return [], pinned, [f"refused: {line}" for line in blocking]
    frozen_keys = _frozen_paths(repo_root)
    by_class = discover_records_by_class(repo_root)
    discovered = {p.relative_to(repo_root).as_posix() for ps in by_class.values() for p in ps}
    move_dest = {new: old for old, new in moves}
    merged_paths = dict(pinned.get("paths", {}))
    done: list[str] = []
    refusals: list[str] = []
    for rel in rels:
        if rel in move_dest:
            if rel not in discovered:
                refusals.append(f"refused: {rel} is not a record in any class")
                continue
            old = move_dest[rel]
            digest = merged_paths.pop(old)
            merged_paths[rel] = digest
            done.append(f"repinned-move: {old} -> {rel} sha256={digest}")
            continue
        path = repo_root / rel
        if not path.is_file():
            refusals.append(f"refused: {rel} is not a file in the working tree")
            continue
        if rel not in discovered:
            refusals.append(f"refused: {rel} is not a record in any class")
            continue
        digest = sha256_bytes(path.read_bytes())
        if merged_paths.get(rel) == digest:
            done.append(f"pinned: {rel} sha256={digest} (already pinned)")
        else:
            merged_paths[rel] = digest
            member = sorted(
                name
                for name, ps in by_class.items()
                if rel in {p.relative_to(repo_root).as_posix() for p in ps}
            )
            done.append(f"pinned: class={'+'.join(member)} {rel} sha256={digest}")
    if refusals:
        return [], pinned, refusals
    stored_counts = pinned.get("counts", {})
    # Recount over the merged paths (accepted move-sources replaced by their
    # dests), not the pinned set: a repinned move carries its classes with it,
    # while a manifest-key drop with no accepted move still shows as lost.
    recounted = pinned_counts_for(merged_paths, frozen_keys)
    demoted = sorted(
        name for name in CLASSES if stored_counts.get(name, 0) > recounted.get(name, 0)
    )
    if demoted:
        # A manifest-key drop demotes pinned records without touching them, so
        # the deferred count mismatch alone would reconcile into the rewrite.
        # A demotion is never resolved by --add: restore the keys first.
        detail = ", ".join(
            f"{name} pinned={stored_counts.get(name, 0)} live-keys={recounted.get(name, 0)}"
            for name in demoted
        )
        return (
            [],
            pinned,
            [
                f"refused: pinned records lost class {detail};"
                " restore the manifest keys before --add"
            ],
        )
    merged = {
        "paths": merged_paths,
        "counts": pinned_counts_for(merged_paths, frozen_keys),
    }
    return done, merged, []


def add_paths(rels: list[str], repo_root: Path = ROOT, hashes_path: Path = HASHES_PATH) -> int:
    """Pin explicit record paths; refuse when the tree is otherwise dirty."""
    pinned = _load_pinned(hashes_path)
    ok, merged, refusals = plan_add(repo_root, rels, pinned)
    for line in refusals:
        print(line)
    if refusals:
        print(f"record-integrity: refused={len(refusals)} pinned=0 (nothing written)")
        return 1
    hashes_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for line in ok:
        print(line)
    print(f"record-integrity: added-paths={len(rels)} pinned-paths={len(merged['paths'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--add", nargs="+", metavar="PATH", default=None)
    args = ap.parse_args(argv)

    if args.add is not None:
        return add_paths(args.add)

    by_class = discover_records_by_class()
    violations, moves = _analyze(ROOT, load_pinned())
    counts = " ".join(f"{name}={len(by_class[name])}" for name in CLASSES)
    totals = load_pinned()["counts"]
    print(
        f"record-integrity: {counts}"
        f" records={sum(len(v) for v in by_class.values())}"
        f" pinned={sum(totals.values())} violations={len(violations)}"
        f" moves={len(moves)}"
    )
    for old, new in moves:
        print(f"moved-record: {old} -> {new}")
    for violation in violations:
        print(violation)
    if args.check and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
