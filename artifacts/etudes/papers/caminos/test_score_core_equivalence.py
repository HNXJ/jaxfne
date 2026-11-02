"""score_core_equivalence: inside, outside with its distance, and the null probability."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_core_equivalence as sce  # noqa: E402


def _row(value, seed, base):
    return {"fraction": value, "seed": seed, "f_pk_hz": base, "band_power_25_45": 10 * base,
            "max_e_rate_hz": base + 1, "rates_Hz": {"E": base, "PV": 2 * base, "SOM": 3 * base}}


ORACLE = [_row(0.4, s, b) for s, b in ((1, 10.0), (2, 11.0), (3, 12.0))] + [_row(0.9, 1, 5.0)]


def test_core_equal_to_oracle_is_inside_everywhere():
    cells = sce.seed_range_equivalence(ORACLE, [r for r in ORACLE if r["fraction"] == 0.4])
    assert len(cells) == len(sce.METRICS), "fixture must share one value only"
    assert all(c["inside"] and c["distance"] == 0.0 for c in cells)


def test_core_above_range_reports_distance():
    core = [_row(0.4, s, 13.0) for s in (1, 2, 3)]
    cells = {c["metric"]: c for c in sce.seed_range_equivalence(ORACLE, core)}
    f = cells["f_pk_hz"]
    assert not f["inside"]
    assert f["distance"] == pytest.approx(1.0, rel=1e-12)
    assert f["distance_range_widths"] == pytest.approx(0.5, rel=1e-12)
    assert f["distance_pct_of_oracle_mean"] == pytest.approx(100.0 / 11.0, rel=1e-12)
    assert cells["rate_SOM"]["distance"] == pytest.approx(3.0, rel=1e-12)


def test_core_below_range_reports_positive_distance():
    core = [_row(0.4, s, 9.0) for s in (1, 2, 3)]
    f = {c["metric"]: c for c in sce.seed_range_equivalence(ORACLE, core)}["f_pk_hz"]
    assert not f["inside"] and f["distance"] == pytest.approx(1.0, rel=1e-12)


def test_null_inside_probability_bounds_and_seed():
    p = sce.null_inside_probability(3, 3, draws=50_000)
    assert 0.5 < p < 0.95
    assert sce.null_inside_probability(3, 3, draws=50_000) == p
    assert sce.null_inside_probability(3, 1, draws=50_000) < p, "one core draw spreads more than a mean of 3"
