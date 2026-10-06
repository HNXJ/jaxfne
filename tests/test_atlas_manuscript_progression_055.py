"""AT-00-R6: manuscript progression order is fixed and checked."""

from __future__ import annotations

import json
import re
from pathlib import Path

from artifacts.atlas.manuscript_progression import SIM_ORDER, STAGES, check_progression

ROOT = Path(__file__).resolve().parents[1]


def _outline():
    return [{"stage": n, "sims": list(s)} for n, s in STAGES]


def test_progression_covers_every_coverage_sim_once_in_order():
    rows = json.loads((ROOT / "artifacts/programme/atlas_coverage.json").read_text())["requirements"]
    sims = {r["sim"] for r in rows if re.fullmatch(r"S\d+", r["sim"])}
    assert sims == set(SIM_ORDER) and len(SIM_ORDER) == len(set(SIM_ORDER)) == 10
    assert SIM_ORDER == tuple(f"S{i}" for i in range(1, 11))


def test_matching_outline_passes_and_swapped_or_missing_fail():
    assert check_progression(_outline()) == []
    swapped = _outline()
    swapped[1], swapped[2] = swapped[2], swapped[1]
    assert check_progression(swapped)
    missing = _outline()
    missing[3]["sims"] = ["S8"]
    assert any("adaptive" in d for d in check_progression(missing))
