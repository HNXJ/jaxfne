"""Gate for docs/theory/doctrine/definition_map.md (C3 definition, symbol and test triples).

Every Symbol other than `none` must import; every Test other than `none` must name
an existing file that defines the test. Rows with a `none` in Symbol or Test are
gaps and must match KNOWN_GAPS exactly, so a new gap fails until it is listed.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = REPO_ROOT / "docs" / "theory" / "doctrine" / "definition_map.md"

MIN_ROWS = 15

KNOWN_GAPS = frozenset({"D28"})

# Rows whose Symbol has a path segment starting with "_": internal realizations.
INTERNAL_ONLY = frozenset({"D31", "D32"})

_HEADER = ["Id", "Definition", "Source", "Symbol", "Test"]


def _strip_ticks(cell: str) -> str:
    return cell.strip().strip("`").strip()


def _parse_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in MAP_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells == _HEADER or set("".join(cells)) <= set("-: "):
            continue
        assert len(cells) == len(_HEADER), f"row has {len(cells)} cells: {line!r}"
        row = dict(zip(_HEADER, cells))
        row["Symbol"] = _strip_ticks(row["Symbol"])
        row["Test"] = _strip_ticks(row["Test"])
        rows.append(row)
    return rows


def _resolve_symbol(dotted: str):
    """Import the longest importable module prefix, then getattr the rest."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        module_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
                continue
            raise
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"no importable prefix for {dotted!r}")


def _test_defined(ref: str) -> str | None:
    """Return None if ref is a defined test, else a reason string."""
    if "::" not in ref:
        return f"Test {ref!r} lacks '::'"
    file_part, name = ref.split("::", 1)
    if not file_part.startswith("tests/"):
        return f"Test file {file_part!r} is not under tests/"
    path = REPO_ROOT / file_part
    if not path.is_file():
        return f"Test file {file_part!r} does not exist"
    text = path.read_text(encoding="utf-8")
    if not re.search(rf"^\s*def {re.escape(name)}\(", text, flags=re.MULTILINE):
        return f"no 'def {name}' in {file_part!r}"
    return None


def test_map_parses_with_minimum_rows():
    rows = _parse_rows()
    assert len(rows) >= MIN_ROWS, f"{len(rows)} rows, need at least {MIN_ROWS}"
    ids = [r["Id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate Ids: {sorted(ids)}"


def test_every_symbol_imports():
    failures = []
    checked = 0
    for row in _parse_rows():
        if row["Symbol"] == "none":
            continue
        checked += 1
        try:
            _resolve_symbol(row["Symbol"])
        except Exception as exc:  # noqa: BLE001 - report every failure at once
            failures.append(f"{row['Id']} {row['Symbol']}: {type(exc).__name__}: {exc}")
    assert checked > 0, "no non-none symbols were examined"
    assert not failures, "unresolvable symbols:\n" + "\n".join(failures)


def test_every_test_is_defined():
    failures = []
    checked = 0
    for row in _parse_rows():
        if row["Test"] == "none":
            continue
        checked += 1
        reason = _test_defined(row["Test"])
        if reason is not None:
            failures.append(f"{row['Id']}: {reason}")
    assert checked > 0, "no non-none tests were examined"
    assert not failures, "undefined tests:\n" + "\n".join(failures)


def test_gap_rows_match_known_gaps():
    gap_ids = {
        r["Id"] for r in _parse_rows() if r["Symbol"] == "none" or r["Test"] == "none"
    }
    new_gaps = sorted(gap_ids - KNOWN_GAPS)
    stale = sorted(KNOWN_GAPS - gap_ids)
    assert not new_gaps, f"new gap rows not in KNOWN_GAPS: {new_gaps}"
    assert not stale, f"KNOWN_GAPS entries no longer gaps: {stale}"


def _has_private_segment(symbol: str) -> bool:
    return any(part.startswith("_") for part in symbol.split("."))


def test_private_symbol_rows_match_internal_only():
    private_ids = {
        r["Id"]
        for r in _parse_rows()
        if r["Symbol"] != "none" and _has_private_segment(r["Symbol"])
    }
    public_ids = {
        r["Id"]
        for r in _parse_rows()
        if r["Symbol"] != "none" and not _has_private_segment(r["Symbol"])
    }
    new_private = sorted(private_ids - INTERNAL_ONLY)
    stale = sorted(INTERNAL_ONLY - private_ids)
    assert not new_private, f"private-symbol rows not in INTERNAL_ONLY: {new_private}"
    assert not stale, f"INTERNAL_ONLY entries whose symbol is now public or absent: {stale}"
    assert not (INTERNAL_ONLY & public_ids), "INTERNAL_ONLY lists a public symbol"
