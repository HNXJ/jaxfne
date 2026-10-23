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
    assert row.count(" | ") == 6


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
    assert rows[1].count(" | ") == 6 and "pass" in rows[1], "the core-method row shows its methods verdict"


def test_page_is_in_the_nav():
    nav = (Path(__file__).resolve().parents[1] / "mkdocs.yml").read_text(encoding="utf-8")
    assert "      - Replications: start/replications.md\n" in nav.replace("\r\n", "\n")
