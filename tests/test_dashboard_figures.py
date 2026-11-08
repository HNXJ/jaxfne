"""scripts/build_dashboard_figures.py: light and dark copies of each chart, drawn independently;
scripts/dashboard_suites.py: suite PNGs and receipts checked by hash, raster column orders proven."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from matplotlib.figure import Figure  # noqa: E402

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


def test_committed_figures_and_suites_are_fresh():
    mod = load_script()
    assert mod.dashboard_suites.SUITES, "fixture must declare suite panels"
    assert mod.check(json.loads(DATA.read_text(encoding="utf-8"))) == 0


def _fake_suite(tmp_path, monkeypatch):
    """One declared panel with one receipt under a fake warehouse, its PNGs and its manifest entry."""
    ds = load_script().dashboard_suites
    wh, figs = tmp_path / "wh", tmp_path / "figs"
    wh.mkdir()
    figs.mkdir()
    monkeypatch.setitem(ds.ROOTS, "warehouse", wh)
    panel = ds.Panel("paper", "panel", "Title", ["warehouse/r*.json"], None)
    monkeypatch.setattr(ds, "SUITES", [panel])
    (wh / "r1.json").write_text('{"a": 1}', encoding="utf-8")
    pngs = {}
    for name in ds.png_names(panel):
        (figs / name).write_bytes(name.encode())
        pngs[name] = hashlib.sha256(name.encode()).hexdigest()
    entry = {"paper": "paper", "id": "panel", "png_sha256": pngs,
             "sources": {"warehouse/r1.json": hashlib.sha256(b'{"a": 1}').hexdigest()}}
    return ds, wh, figs, entry


def test_suite_check_flags_edited_pngs_changed_receipts_and_new_receipts(tmp_path, monkeypatch):
    ds, wh, figs, entry = _fake_suite(tmp_path, monkeypatch)
    assert ds.check([entry], figs) == ([], 1, 0)

    png = figs / "suite-paper-panel.dark.png"
    png.write_bytes(b"edited")
    assert any("differs from its recorded sha256" in p for p in ds.check([entry], figs)[0])
    png.write_bytes(png.name.encode())

    (wh / "r2.json").write_text("{}", encoding="utf-8")
    assert any("receipt set differs" in p for p in ds.check([entry], figs)[0])
    (wh / "r2.json").unlink()

    (wh / "r1.json").write_text('{"a": 2}', encoding="utf-8")
    ds.sha256_file.__defaults__[0].clear()  # the digest cache lives for one process run
    assert any("changed since the render" in p for p in ds.check([entry], figs)[0])

    (wh / "r1.json").unlink()  # receipts absent, as in CI: the PNGs are still checked
    assert ds.check([entry], figs) == ([], 0, 1)
    assert any("not rendered" in p for p in ds.check([], figs)[0])


def test_potjans_raster_refuses_a_column_order_its_rates_do_not_prove(tmp_path):
    ds = load_script().dashboard_suites
    spikes = np.zeros((20000, 3), dtype=bool)
    spikes[::200, :2] = True  # cells 0-1: one spike per 20 ms, 50 Hz (2 s at dt 0.1 ms)
    spikes[::400, 2] = True  # cell 2: 25 Hz
    run = {"seed": 4, "file": "s.npz", "dt_ms": 0.1, "duration_ms": 2000.0, "scale": 0.1, "n": 3,
           "pops": {"L23E": {"n": 2, "rate_hz": 50.0}, "L23PV": {"n": 1, "rate_hz": 25.0}}}
    (tmp_path / "seed4").mkdir()
    np.savez(tmp_path / "seed4" / "s.npz", spikes=spikes)
    manifest = tmp_path / "seed4" / "manifest.json"
    files = [manifest, tmp_path / "seed4" / "s.npz"]
    manifest.write_text(json.dumps({"runs": [run]}), encoding="utf-8")
    assert "every 10th cell" in ds.pd_raster(Figure(), files, ds.THEMES[False])

    run["pops"] = {"L23PV": {"n": 1, "rate_hz": 25.0}, "L23E": {"n": 2, "rate_hz": 50.0}}
    manifest.write_text(json.dumps({"runs": [run]}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reproduce its rate_hz"):
        ds.pd_raster(Figure(), files, ds.THEMES[False])


def test_caminos_counts_refuses_columns_that_do_not_match_spikes_kept(tmp_path):
    ds = load_script().dashboard_suites
    counts = np.tile(np.array([[1, 5, 2]], dtype=np.int32), (12, 1))
    np.save(tmp_path / "c.npy", counts)
    (tmp_path / "caminos_scored").mkdir()
    manifest = tmp_path / "caminos_scored" / "manifest.json"
    manifest.write_text(json.dumps({"runs": [{"seed": 1, "p": 0.07, "file": "c.npy", "dt": 1.0}]}), encoding="utf-8")
    score = tmp_path / "caminos_scored" / "score.json"
    files = [manifest, score, tmp_path / "c.npy"]
    kept = {"seed": 1, "p": 0.07, "spikes_kept": {"E": 10, "PV": 50, "SOM": 20}}
    score.write_text(json.dumps({"per_run": [kept]}), encoding="utf-8")
    assert ds.caminos_counts(files, 1, 0.07)[0].shape == (12, 3)

    kept["spikes_kept"] = {"E": 50, "PV": 10, "SOM": 20}
    score.write_text(json.dumps({"per_run": [kept]}), encoding="utf-8")
    with pytest.raises(ValueError, match="spikes_kept"):
        ds.caminos_counts(files, 1, 0.07)
