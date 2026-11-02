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


def _six(value, bases):
    return [_row(value, s, b) for s, b in enumerate(bases, start=1)]


def test_welch_marks_a_large_offset_systematic_after_holm():
    oracle = _six(0.9, (10.0, 10.1, 9.9, 10.05, 9.95, 10.0))
    core = _six(0.9, (12.0, 12.1, 11.9, 12.05, 11.95, 12.0))
    cells = {c["metric"]: c for c in sce.welch_cells(oracle, core, [0.9])}
    f = cells["f_pk_hz"]
    assert f["diff"] == pytest.approx(2.0, rel=1e-12)
    assert f["ci95"][0] > 0 and f["systematic"] and f["p_holm"] >= f["p"]


def test_welch_same_draws_are_not_systematic():
    draws = (10.0, 10.1, 9.9, 10.05, 9.95, 10.0)
    cells = sce.welch_cells(_six(0.9, draws), _six(0.9, draws), [0.9])
    assert all(c["diff"] == 0.0 and c["p"] == pytest.approx(1.0) and not c["systematic"] for c in cells)


def test_welch_holm_never_lowers_p():
    oracle = _six(0.7, (10.0, 10.3, 9.8, 10.1, 9.9, 10.2))
    core = _six(0.7, (10.2, 10.4, 9.9, 10.3, 10.0, 10.1))
    cells = sce.welch_cells(oracle, core, [0.7])
    assert all(c["p_holm"] >= c["p"] for c in cells)


def test_null_inside_probability_bounds_and_seed():
    p = sce.null_inside_probability(3, 3, draws=50_000)
    assert 0.5 < p < 0.95
    assert sce.null_inside_probability(3, 3, draws=50_000) == p
    assert sce.null_inside_probability(3, 1, draws=50_000) < p, "one core draw spreads more than a mean of 3"
