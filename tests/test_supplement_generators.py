"""Supplement generators S1 and S5: outputs match the definition map and the identity test."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "scripts/publication_figures/supplement.py"
DEFINITION_MAP = REPO / "docs/theory/doctrine/definition_map.md"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("supplement", GENERATOR)
    assert spec is not None and spec.loader is not None, f"cannot load {GENERATOR}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


supplement = _load_generator()


def _map_rows_read_directly() -> dict[str, tuple[str, str]]:
    """Id -> (symbol cell, test cell), read from the map without the generator's parser."""
    rows: dict[str, tuple[str, str]] = {}
    for line in DEFINITION_MAP.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\| D\d+ \|", line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows[cells[0]] = (cells[3], cells[4])
    return rows


def test_s1_lists_every_map_row_with_its_status(tmp_path: Path) -> None:
    expected = _map_rows_read_directly()
    assert expected, "the map has no D rows; the comparison would pass vacuously"

    counts = supplement.write_s1(tmp_path)

    md_rows: dict[str, list[str]] = {}
    for line in (tmp_path / "S1_notation.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| D"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            md_rows[cells[0]] = cells
    assert set(md_rows) == set(expected), "S1 rows differ from the map's D rows"

    statuses: dict[str, str] = {}
    for row_id, (symbol, test) in expected.items():
        named = all(cell.strip("`") not in ("", "none") for cell in (symbol, test))
        want = "mapped" if named else "open"
        assert md_rows[row_id][2:5] == [symbol, test, want], (
            f"{row_id}: S1 row {md_rows[row_id][2:5]} != map symbol/test with status {want!r}"
        )
        statuses[row_id] = want

    assert set(statuses.values()) == {"mapped", "open"}, "need at least one mapped and one open row"
    assert counts["rows"] == len(expected)
    assert counts["mapped"] == sum(s == "mapped" for s in statuses.values())
    assert counts["open"] == sum(s == "open" for s in statuses.values())


@pytest.fixture(scope="module")
def s5_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("s5")
    supplement.write_s5(out)
    return out


@pytest.fixture(scope="module")
def s5_payload(s5_out: Path) -> dict:
    return json.loads((s5_out / "S5_reduction_identity.json").read_text(encoding="utf-8"))


def test_s5_covers_the_live_registry(s5_payload: dict) -> None:
    # The core-rule filter is owned by the identity test module, which S5 itself calls.
    live = set(supplement.load_identity_module()._core_rule_names())
    assert live, "the registry snapshot is empty"
    assert {row["rule"] for row in s5_payload["rules"]} == live


def test_s5_identity_numbers_are_zero(s5_payload: dict) -> None:
    rows = s5_payload["rules"]
    assert rows, "no rules examined"
    for row in rows:
        assert row["spike_mismatches_zero_drive"] == 0, row["rule"]
        assert row["max_abs_dw_zero_drive"] == 0.0, row["rule"]
        assert row["max_abs_dw_all_false_mask"] == 0.0, row["rule"]


def test_s5_every_falsifier_is_above_zero(s5_payload: dict) -> None:
    rows = s5_payload["rules"]
    assert rows, "no rules examined"
    for row in rows:
        assert row["max_abs_dw_nonzero_params"] > 0.0, row["rule"]


def test_s5_markdown_has_a_row_per_rule_and_the_panel_is_a_png(
    s5_out: Path, s5_payload: dict
) -> None:
    md_lines = (s5_out / "S5_reduction_identity.md").read_text(encoding="utf-8").splitlines()
    table = [line for line in md_lines if line.startswith("| ")]
    assert len(table) == len(s5_payload["rules"]) + 1, "header plus one row per rule"
    png = (s5_out / "S5_reduction_identity.png").read_bytes()
    assert png.startswith(PNG_MAGIC), "S5 panel is not a PNG"
