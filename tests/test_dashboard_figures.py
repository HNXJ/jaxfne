"""scripts/build_dashboard_figures.py: light and dark copies of each chart, drawn independently."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_dashboard_figures.py"
DATA = REPO / "dashboard" / "data" / "replications.json"


def load_script():
    spec = importlib.util.spec_from_file_location("build_dashboard_figures_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_chart():
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    charts = [c for p in doc["papers"] for c in p.get("charts", [])]
    assert charts, "fixture must have a chart"
    return charts[0]


def test_every_chart_has_a_light_and_a_dark_name():
    mod = load_script()
    doc = json.loads(DATA.read_text(encoding="utf-8"))
    names = [(name, dark) for name, _, dark in mod.charts(doc)]
    assert names, "fixture must have charts"
    light = {n[: -len(".png")] for n, d in names if not d}
    dark = {n[: -len(".dark.png")] for n, d in names if d}
    assert light == dark


def test_chart_hash_ignores_receipt_digests_and_follows_drawn_values():
    mod = load_script()
    chart = first_chart()
    base = mod.chart_hash(chart)
    with_digest = json.loads(json.dumps(chart))
    assert "test_receipt" not in with_digest, "fixture key must be new"
    with_digest["test_receipt"] = [{"path": "x.json", "sha256": "0" * 64}]
    without_digest = json.loads(json.dumps(with_digest))
    del without_digest["test_receipt"][0]["sha256"]
    assert mod.chart_hash(with_digest) == mod.chart_hash(without_digest)
    assert mod.chart_hash(without_digest) != base, "fixture must change a drawn value"


def test_a_dark_draw_does_not_restyle_the_next_light_draw(tmp_path):
    mod = load_script()
    chart = first_chart()
    mod.draw(chart, tmp_path / "light_first.png")
    mod.draw(chart, tmp_path / "dark.png", dark=True)
    mod.draw(chart, tmp_path / "light_after.png")
    first = (tmp_path / "light_first.png").read_bytes()
    assert first != (tmp_path / "dark.png").read_bytes(), "fixture must draw a different dark figure"
    assert (tmp_path / "light_after.png").read_bytes() == first
