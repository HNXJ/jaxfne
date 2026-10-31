"""docs/start/replications.md is generated from dashboard/sources.json and lists every paper."""

import json
from pathlib import Path

from scripts.generate_replications_page import DASHBOARD, OUT, SOURCES, main, render


def test_page_is_current():
    assert main(["--check"]) == 0


def test_every_paper_links_its_dashboard_page():
    papers = json.loads(SOURCES.read_text(encoding="utf-8"))["papers"]
    assert papers, "fixture must list papers"
    text = OUT.read_text(encoding="utf-8")
    for p in papers:
        assert f"({DASHBOARD}#model/{p['id']})" in text, p["id"]


def test_pipe_in_a_title_does_not_split_the_row():
    sources = {"papers": [{"id": "x", "title": "A | B", "citation": "C", "class": "reproduced (conditional)",
                           "release": "0.5.6", "stage": "P5", "verdict": "reproduced"}]}
    row = [ln for ln in render(sources).splitlines() if "#model/x" in ln][0]
    assert row.count(" | ") == 7


def test_replica_column_shows_the_score_or_a_dash():
    base = {"title": "T", "citation": "C", "class": "in progress", "release": "0.5.9", "stage": "P3"}
    doc = {"papers": [dict(base, id="x", replica={"score": 38}), dict(base, id="y", replica={"score": None}),
                      dict(base, id="z")]}
    rows = {ln.split("#model/")[1][0]: ln for ln in render(doc).splitlines() if "#model/" in ln}
    assert "| 38/100 |" in rows["x"], rows["x"]
    assert "| P3 | — |" in rows["y"] and "| P3 | — |" in rows["z"], (rows["y"], rows["z"])


def test_page_columns_mirror_the_three_verdicts():
    doc = {"papers": [
        {"id": "x", "title": "T", "citation": "C", "class": "mixed", "release": "0.5.7",
         "stage": "P6", "verdict": "mixed", "verdict_note": "R4 missed",
         "verdict_summary": {"kind": "paper", "paper": "mixed",
                             "library": {"stage": "P6", "verdict": "pass"}, "methods": None}},
        {"id": "y", "title": "U", "citation": "D", "class": "core method", "release": "0.5.8",
         "stage": None, "verdict": None,
         "verdict_summary": {"kind": "core_method", "paper": None,
                             "library": None, "methods": "pass"}},
    ]}
    rows = [ln for ln in render(doc).splitlines() if "#model/" in ln]
    assert len(rows) == 2
    assert "P6 PASS" in rows[0] and "mixed" in rows[0], "the paper and library verdicts share the row"
    assert rows[1].count(" | ") == 7 and "pass" in rows[1], "the core-method row shows its methods verdict"


def test_page_shows_the_derived_library_and_methods_verdicts():
    # sources.json carries no derived verdicts; the page takes them from the dashboard data.
    root = Path(__file__).resolve().parents[1]
    derived = json.loads((root / "dashboard" / "data" / "replications.json").read_text(encoding="utf-8"))["papers"]
    text = OUT.read_text(encoding="utf-8")
    rows = {ln for ln in text.splitlines() if "#model/" in ln}
    with_lib = [p for p in derived if (p.get("verdict_summary") or {}).get("library")]
    with_methods = [p for p in derived if (p.get("verdict_summary") or {}).get("methods")]
    assert with_lib and with_methods, "fixture must hold a library and a methods verdict"
    for p in with_lib:
        lib = p["verdict_summary"]["library"]
        row = next(r for r in rows if f"#model/{p['id']})" in r)
        assert f"{lib['stage']} {str(lib['verdict']).upper()}" in row, p["id"]
    for p in with_methods:
        row = next(r for r in rows if f"#model/{p['id']})" in r)
        assert row.rstrip().endswith(f"| {p['verdict_summary']['methods']} |"), p["id"]


def test_stage_and_verdict_wording_agree_across_the_records():
    # One verdict per paper across receipt, registry, dashboard and docs (open stack O-61):
    # the CAMINOS lag is a finding (CALIBRATION section 10), the Queue rows carry the stage the
    # calibration records state, and no dashboard open item calls an attempted library path unattempted.
    root = Path(__file__).resolve().parents[1]
    papers = {p["id"]: p for p in json.loads(SOURCES.read_text(encoding="utf-8"))["papers"]}
    caminos = root / "artifacts" / "etudes" / "papers" / "caminos"
    ledger = (caminos / "LEDGER.md").read_text(encoding="utf-8")
    r2 = next(ln for ln in ledger.splitlines() if ln.startswith("| CAM-R2 "))
    assert "finding: needs X" in r2 and "reproduced" not in r2.split("|")[4]
    assert "lag" not in papers["caminos"]["summary"].split("reproduce")[0][-40:]
    assert not any("P6 library path not attempted" in o for o in papers["caminos"]["open"])
    queue = (root / "artifacts" / "etudes" / "meta" / "REPLICATIONS.md").read_text(encoding="utf-8")
    examined = 0
    for pid, key in (("potjans2014", "| Potjans and Diesmann 2014"), ("vogels2011", "| Vogels TP"),
                     ("mejias2016", "| Mejias, Murray")):
        row = next(ln for ln in queue.splitlines() if ln.startswith(key))
        assert row.split(" | ")[2].strip() == papers[pid]["stage"], pid
        calib = (root / "artifacts" / "etudes" / "papers" / pid / "CALIBRATION.md").read_text(encoding="utf-8")
        assert f"Status: {papers[pid]['stage']} " in calib, pid
        examined += 1
    assert examined == 3


def test_page_is_in_the_nav():
    nav = (Path(__file__).resolve().parents[1] / "mkdocs.yml").read_text(encoding="utf-8")
    assert "      - Replications: start/replications.md\n" in nav.replace("\r\n", "\n")
