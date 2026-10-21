"""Tests for the `studies` key of dashboard/data/replications.json.

The study list is the table under '## Research studies (study list)' in docs/reference/references.md,
parsed by parse_studies in scripts/build_dashboard_data.py. DASH_GEN_BUILD_SCRIPT may name another
generator; it is used to show that these tests fail against the baseline script.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get("DASH_GEN_BUILD_SCRIPT", REPO / "scripts" / "build_dashboard_data.py"))
REFERENCES = REPO / "docs" / "reference" / "references.md"
DATA = REPO / "dashboard" / "data" / "replications.json"
HEADING = "## Research studies (study list)"
ALLOWED = {"reproduced", "reproduced (conditional)", "mixed", "in progress", "queued", "not started"}


def load_generator():
    spec = importlib.util.spec_from_file_location("build_dashboard_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def study_table(where_cells):
    """A references page holding one study table row per Where cell (each row has a DOI)."""
    lines = ["# References", "", HEADING, "", "| Reference | Result compared | Where |", "|---|---|---|"]
    for i, where in enumerate(where_cells):
        lines.append(f"| Author {i}. Title. *Journal* 1, 1. [doi:10.1000/x{i}](https://doi.org/10.1000/x{i}) | Result {i} | {where} |")
    lines += ["", "## Core references", "", "| Reference | Use |", "|---|---|", "| Other row. | Not a study |", ""]
    return "\n".join(lines)


def doi_rows_counted_by_text(markdown):
    """Rows under the heading that start with '| ' and carry a DOI, counted here without the generator."""
    lines = markdown.splitlines()
    start = lines.index(HEADING) + 1
    count = 0
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.startswith("| ") and "doi.org/" in line:
            count += 1
    return count


def test_parser_returns_at_least_40_rows():
    mod = load_generator()
    rows = mod.parse_studies(REFERENCES.read_text(encoding="utf-8"))
    assert len(rows) >= 40, f"got {len(rows)} study rows"


def test_every_status_is_in_the_allowed_set():
    mod = load_generator()
    rows = mod.parse_studies(REFERENCES.read_text(encoding="utf-8"))
    statuses = {r["status"] for r in rows}
    assert statuses, "no statuses were examined"
    assert statuses <= ALLOWED, f"unexpected statuses {sorted(statuses - ALLOWED)}"


def test_row_count_equals_independent_count_of_doi_rows():
    mod = load_generator()
    text = REFERENCES.read_text(encoding="utf-8")
    rows = mod.parse_studies(text)
    counted = doi_rows_counted_by_text(text)
    assert counted > 0, "no DOI rows found under the heading"
    assert len(rows) == counted, f"parsed {len(rows)} rows, text holds {counted} DOI rows"


def test_unknown_status_raises_and_names_it():
    mod = load_generator()
    good = study_table(["Not started."] * 40)
    assert len(mod.parse_studies(good)) == 40, "fixture must parse before the bad row is added"
    bad = study_table(["Not started."] * 39 + ["Published. [Dashboard](https://example.org)"])
    with pytest.raises(mod.BuildError, match="Published"):
        mod.parse_studies(bad)


def test_fewer_than_40_rows_raises():
    mod = load_generator()
    with pytest.raises(mod.BuildError, match="39 rows"):
        mod.parse_studies(study_table(["Not started."] * 39))


def test_parenthetical_after_status_is_a_note_not_part_of_the_status():
    mod = load_generator()
    rows = mod.parse_studies(study_table(["In progress (pilot). [Dashboard](https://example.org)"] + ["Not started."] * 39))
    first = rows[0]
    assert first["status"] == "in progress"
    assert first["status_note"] == "pilot"
    assert first["where"] == "[Dashboard](https://example.org)"
    assert first["doi"] == "10.1000/x0"


def test_conditional_is_part_of_the_status_and_has_no_note():
    mod = load_generator()
    rows = mod.parse_studies(study_table(["Reproduced (conditional). [Dashboard](https://example.org)"] + ["Not started."] * 39))
    assert rows[0]["status"] == "reproduced (conditional)"
    assert rows[0]["status_note"] is None


def test_committed_data_holds_studies_equal_to_a_fresh_parse():
    mod = load_generator()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    fresh = mod.parse_studies(REFERENCES.read_text(encoding="utf-8"))
    assert "studies" in data, "replications.json has no studies key"
    assert data["studies"] == fresh
