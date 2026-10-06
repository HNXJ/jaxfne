"""AT-00-R5: Atlas figure column order is fixed and checked."""

from __future__ import annotations

from pathlib import Path

from artifacts.atlas.figure_columns import COLUMNS, check_figure_spec

ROOT = Path(__file__).resolve().parents[1]
SRC = "artifacts/atlas/results/at10_n20_055.json"


def _spec(cols, src=SRC):
    return {"id": "F", "panels": [{"column": c, "source": src} for c in cols]}


def test_order_is_the_declared_seven():
    assert COLUMNS == (
        "structure", "dynamics", "state_plasticity", "source", "field",
        "observation", "computation",
    )


def test_conforming_subset_passes_and_existing_source_resolves():
    assert check_figure_spec(_spec(["structure", "source", "computation"]), root=ROOT) == []


def test_out_of_order_unknown_missing_and_empty_are_flagged():
    assert any("out of order" in d for d in check_figure_spec(_spec(["field", "structure"])))
    assert any("unknown column" in d for d in check_figure_spec(_spec(["lfp"])))
    assert any("missing" in d for d in check_figure_spec(_spec(["structure"], "nope.json"), root=ROOT))
    assert check_figure_spec({"id": "F", "panels": []}) == ["F: no panels"]
