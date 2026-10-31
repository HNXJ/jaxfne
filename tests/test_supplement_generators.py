"""Supplement generators S1, S2 and S5: outputs match the map, the calibration
transforms and the identity test."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
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

    # D28 closed the last open row (2026-10-28): "open" appears only while a row names no symbol.
    assert set(statuses.values()) <= {"mapped", "open"} and "mapped" in statuses.values()
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


# --- S2: relative quantities and calibration transforms ---------------------


def test_s2_lists_at_least_one_calibration_transform(tmp_path: Path) -> None:
    counts = supplement.write_s2(tmp_path)
    assert counts["rows"] >= 1, "S2 must list at least one public calibration transform"
    payload = json.loads((tmp_path / "S2_calibration.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == counts["rows"]
    md = (tmp_path / "S2_calibration.md").read_text(encoding="utf-8")
    assert md.startswith("# S2.")
    for row in payload["rows"]:
        assert row["code_path"], row["name"]
        assert row["doctrine"], row["name"]


def test_s2_every_row_check_passes(tmp_path: Path) -> None:
    counts = supplement.write_s2(tmp_path)
    assert counts["rows"] >= 1
    assert counts["checked"] == counts["rows"]
    assert counts["failed"] == 0
    payload = json.loads((tmp_path / "S2_calibration.json").read_text(encoding="utf-8"))
    for row in payload["rows"]:
        assert row["passed"], row["name"]
        assert row["input"] is not None and row["expected"] is not None, row["name"]
        assert np.allclose(
            np.asarray(row["output"], dtype=float),
            np.asarray(row["expected"], dtype=float),
            rtol=1e-12,
            atol=0.0,
        ), row["name"]


def test_s2_perturbed_transform_fails_and_main_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import jaxfne.fields as fields_mod

    real_apply = fields_mod.apply_calibration

    def scaled_by_two(obj, transform):
        out = real_apply(obj, transform)
        swaps = {}
        for f in dataclasses.fields(out):
            v = getattr(out, f.name)
            if hasattr(v, "dtype") and hasattr(v, "shape"):
                swaps[f.name] = v * 2
        assert swaps, "perturbed fake must scale a numeric payload"
        return dataclasses.replace(out, **swaps)

    monkeypatch.setattr(fields_mod, "apply_calibration", scaled_by_two)
    counts = supplement.write_s2(tmp_path)
    assert counts["rows"] >= 1
    assert counts["failed"] == counts["rows"], "a x2 perturbation must fail every row"
    monkeypatch.setattr(supplement, "write_s5", lambda out: [])
    assert supplement.main(["--out", str(tmp_path)]) == 1
    assert f"S2 rows={counts['rows']}" in capsys.readouterr().out
