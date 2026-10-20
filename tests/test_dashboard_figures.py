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


def test_a_dark_draw_does_not_restyle_the_next_light_draw(tmp_path):
    mod = load_script()
    chart = first_chart()
    mod.draw(chart, tmp_path / "light_first.png")
    mod.draw(chart, tmp_path / "dark.png", dark=True)
    mod.draw(chart, tmp_path / "light_after.png")
    first = (tmp_path / "light_first.png").read_bytes()
    assert first != (tmp_path / "dark.png").read_bytes(), "fixture must draw a different dark figure"
    assert (tmp_path / "light_after.png").read_bytes() == first
