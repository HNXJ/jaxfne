"""F7 validation figure: renders the passing criteria from the score JSONs, recomputes
each pass, and refuses a plotted criterion that does not pass."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "scripts/publication_figures/fig07_validation.py"
PCL_SCORE = REPO / "artifacts/etudes/papers/pcl/prereg/scores/fig3_v4_full.json"
VOGELS_SCORE = REPO / "artifacts/etudes/papers/vogels2011/run/p3_v4_score.json"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fig07_validation", GENERATOR)
    assert spec is not None and spec.loader is not None, f"cannot load {GENERATOR}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fig07() -> ModuleType:
    return _load_generator()


def test_render_writes_png_and_json_values_equal_the_source_fields(
    fig07: ModuleType, tmp_path: Path
) -> None:
    pcl = _read(PCL_SCORE)
    vog = _read(VOGELS_SCORE)
    assert pcl["t1_simple_pass"] and pcl["t1_complex_pass"], "fixture: PCL T1 passes"
    assert vog["seeds"] == [4, 5, 6, 7, 8] and vog["O-DROP"]["pass"], "fixture: seeds 4-8, O-DROP passes"

    counts = fig07.write_fig07(tmp_path)

    assert counts == {"criteria": 11, "passing": 11}
    assert (tmp_path / "F7_validation.png").read_bytes().startswith(PNG_MAGIC), "F7 is not a PNG"
    out = _read(tmp_path / "F7_validation.json")
    rows = {row["id"]: row for row in out["criteria"]}
    assert len(rows) == len(out["criteria"]) == 11, "one row per plotted criterion"
    assert all(row["pass"] is True for row in out["criteria"]), "F7 carries no failing criterion"

    for kind in ("simple", "complex"):
        field = f"t1_r_ori_{kind}"
        assert rows[field]["value"] == pcl[field], field
        assert rows[field]["threshold"] == max(
            2.0, pcl[f"{field}_null99"], pcl[f"{field}_npref99"]
        ), field
    for field in ("median_X_simple", "median_X_complex", "t4_diff"):
        assert rows[field]["value"] == pcl[field], field
    assert rows["median_X_simple"]["threshold"] == 0.15
    assert rows["median_X_complex"]["threshold"] == 0.5 * pcl["median_X_simple"]
    assert rows["t4_diff"]["threshold"] == pcl["t4_diff_null99"]

    for seed in vog["seeds"]:
        row = rows[f"CONV seed {seed}"]
        conv = vog["CONV"]["per_seed"][str(seed)]
        assert row["terms"]["b_hz_s"] == conv["b_hz_s"], seed
        assert row["terms"]["se_hz_s"] == conv["se_hz_s"], seed
        assert row["value"] == pytest.approx(abs(conv["b_hz_s"]) / conv["se_hz_s"], rel=1e-12)
        assert row["threshold"] == 2.101
    assert rows["O-DROP ratio"]["value"] == vog["O-DROP"]["ratio"]
    assert rows["O-DROP ratio"]["threshold"] == 2.0


def test_a_synthetic_pcl_t1_below_threshold_raises(fig07: ModuleType, tmp_path: Path) -> None:
    pcl = _read(PCL_SCORE)
    assert pcl["t1_r_ori_simple"] >= 2.0, "fixture: the real simple T1 statistic passes"
    pcl["t1_r_ori_simple"] = 1.5
    synthetic = tmp_path / "pcl_t1_below.json"
    synthetic.write_text(json.dumps(pcl), encoding="utf-8")

    with pytest.raises(ValueError, match="t1_r_ori_simple does not pass"):
        fig07.write_fig07(tmp_path / "out", pcl_path=synthetic)


def test_a_synthetic_vogels_seed_with_t3_and_small_b_raises(
    fig07: ModuleType, tmp_path: Path
) -> None:
    vog = _read(VOGELS_SCORE)
    assert "4" in vog["CONV"]["per_seed"], "fixture: seed 4 is scored"
    vog["CONV"]["per_seed"]["4"].update(b_hz_s=0.02, se_hz_s=0.02 / 3.0)
    synthetic = tmp_path / "vogels_t3.json"
    synthetic.write_text(json.dumps(vog), encoding="utf-8")

    with pytest.raises(ValueError, match="CONV seed 4 does not pass"):
        fig07.write_fig07(tmp_path / "out", vogels_path=synthetic)
