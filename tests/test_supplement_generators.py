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


# --- S14: the scored GSDR P3 fit ------------------------------------------------


def _s14_score() -> dict:
    return json.loads(
        (REPO / "artifacts" / "etudes" / "papers" / "gsdr2026" / "run" / "p3_fit_score.json")
        .read_text(encoding="utf-8")
    )


def test_s14_lists_all_five_seeds_with_the_json_verdict(tmp_path: Path) -> None:
    score = _s14_score()
    assert len(score["rows"]) == 5, "the scored fit ran five optimizer seeds"

    counts = supplement.write_s14(tmp_path)

    assert counts["rows"] == 5 and counts["passing"] == score["seeds_passing"] == 0
    payload = json.loads((tmp_path / "S14_gsdr_fit.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == score["verdict"] == "FAIL", "the verdict comes from the score JSON"
    assert payload["seeds_passing"] == 0
    assert [row["seed"] for row in payload["rows"]] == [row["seed"] for row in score["rows"]]
    for want, got in zip(score["rows"], payload["rows"]):
        for key in ("loss_g0", "loss_g30", "kappa_mean", "kappa_min", "kappa_max",
                    "rate_mean_hz", "G2", "G3"):
            assert got[key] == want[key], (want["seed"], key)

    md_lines = (tmp_path / "S14_gsdr_fit.md").read_text(encoding="utf-8").splitlines()
    assert md_lines[0].startswith("# S14.")
    assert "**Verdict: FAIL**" in "\n".join(md_lines)
    table = [line for line in md_lines if line.startswith("| ")]
    assert len(table) == len(payload["rows"]) + 1, "header plus one row per seed"
    for row in score["rows"]:
        assert any(line.startswith(f"| {row['seed']} |") for line in table), row["seed"]


def test_s14_a_four_pass_synthetic_score_passes(tmp_path: Path) -> None:
    score = _s14_score()
    rows = [dict(row, G2=True, G3=True) for row in score["rows"][:4]]
    rows.append(dict(score["rows"][4], G2=False, G3=True))
    synthetic = tmp_path / "synthetic_p3_fit_score.json"
    synthetic.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    (tmp_path / "s14").mkdir()

    counts = supplement.write_s14(tmp_path / "s14", score_path=synthetic)

    assert counts["rows"] == 5 and counts["passing"] == 4
    payload = json.loads((tmp_path / "s14" / "S14_gsdr_fit.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert "**Verdict: PASS**" in (tmp_path / "s14" / "S14_gsdr_fit.md").read_text(encoding="utf-8")


# --- S12: the scored Vogels P3 v4 (segment B) -----------------------------------


def _s12_score() -> dict:
    return json.loads(
        (REPO / "artifacts" / "etudes" / "papers" / "vogels2011" / "run" / "p3_v4_score.json")
        .read_text(encoding="utf-8")
    )


def test_s12_lists_five_seeds_and_recomputes_the_json_verdict(tmp_path: Path) -> None:
    score = _s12_score()
    assert score["seeds"] == [4, 5, 6, 7, 8], "the signed run scored seeds 4-8"
    assert score["verdict"].startswith("off"), "fixture: the signed result is off"

    counts = supplement.write_s12(tmp_path)

    assert counts["rows"] == 5 and counts["verdict"] == "off"
    payload = json.loads((tmp_path / "S12_vogels_p3.json").read_text(encoding="utf-8"))
    assert payload["failed"] == ["R-E"], payload["failed"]
    assert payload["R-E"]["z_hz"] == score["R-E"]["z_hz"]
    assert payload["O-DROP"]["ratio"] == score["O-DROP"]["ratio"]
    for row in payload["rows"]:
        s = str(row["seed"])
        assert row["b_hz_s"] == score["CONV"]["per_seed"][s]["b_hz_s"], s
        assert row["re_hz"] == score["R-E"]["per_seed_z_hz"][s], s
        assert row["converged"] is True, s
    md = (tmp_path / "S12_vogels_p3.md").read_text(encoding="utf-8")
    assert md.splitlines()[0].startswith("# S12.")
    assert "**Verdict: off (R-E)**" in md
    table = [line for line in md.splitlines() if line.startswith("| ")]
    assert len(table) == 5 + 1, "header plus one row per seed"


def test_s12_an_in_band_synthetic_score_closes_and_a_mismatch_is_refused(tmp_path: Path) -> None:
    score = _s12_score()
    closing = dict(score, verdict="closes")
    closing["R-E"] = dict(score["R-E"], z_hz=3.3)
    closing["O-DROP"] = dict(score["O-DROP"], ratio=score["O-DROP"]["before_hz"] / 3.3)
    path = tmp_path / "closing.json"
    path.write_text(json.dumps(closing), encoding="utf-8")
    (tmp_path / "a").mkdir()
    assert supplement.write_s12(tmp_path / "a", score_path=path)["verdict"] == "closes"

    unconverged = json.loads(json.dumps(closing))
    unconverged["CONV"]["per_seed"]["6"].update(b_hz_s=-0.05, se_hz_s=0.005)
    bad = tmp_path / "unconverged.json"
    bad.write_text(json.dumps(unconverged), encoding="utf-8")
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="disagrees"):
        supplement.write_s12(tmp_path / "b", score_path=bad)


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
