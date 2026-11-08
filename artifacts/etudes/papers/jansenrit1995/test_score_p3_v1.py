"""Scorer for Jansen-Rit P3 v1: a profile matching the prereg passes; each criterion fails on its own break."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_p3_v1 as sc  # noqa: E402

# (f_peak Hz, osc, SD mV, mean mV) per C, shaped like the pilot table in the prereg.
GOOD = {68.0: (4.5, 0.24, 0.43, 10.5), 128.0: (10.5, 0.75, 0.75, 7.8), 135.0: (10.5, 0.91, 1.34, 7.6),
        270.0: (5.0, 0.73, 11.8, -5.2), 675.0: (2.75, 0.50, 36.5, -22.0), 1350.0: (4.5, 0.22, 0.42, -11.9)}


def rec(prof):
    rows = [{"C": c, "finite": True, "f_peak_hz": f, "osc_index": o, "eeg_std_mv": sd, "eeg_mean_mv": m}
            for c, (f, o, sd, m) in prof.items()]
    return {"seed": 0, "git_sha": "x", "rows": rows}


def test_good_profile_passes():
    r = sc.score_seed(rec(GOOD))
    assert r["pass"] and r["classes"] == list(sc.E1_EXPECTED)


def edit(prof, c, field, value):
    prof = copy.deepcopy(prof)
    row = list(prof[c])
    row[field] = value
    prof[c] = tuple(row)
    return prof


@pytest.mark.parametrize("c, field, value, crit", [
    (68.0, 1, 0.5, "E1"),      # noise at 68 becomes "other" (osc >= 0.4, f 4.5 Hz, SD 0.43 mV)
    (128.0, 1, 0.3999, "E1"),  # just under the 0.4 threshold: alpha at 128 becomes noise
    (135.0, 1, 0.70, "E2"),    # osc(135) below osc(128)
    (675.0, 0, 5.5, "E3"),     # no slowing
    (135.0, 3, 11.0, "E4"),    # EEG level out of 5-10 mV
    (270.0, 2, 4.99, "E1"),    # spike-wave SD just under 5 mV
])
def test_each_criterion_fails_alone_on_its_break(c, field, value, crit):
    r = sc.score_seed(rec(edit(GOOD, c, field, value)))
    assert r["criteria"][crit] is False and r["pass"] is False
    assert all(v for k, v in r["criteria"].items() if k != crit), r["criteria"]


@pytest.mark.parametrize("c, field, value", [
    (128.0, 1, 0.4),    # osc exactly 0.4 is oscillatory (noise is osc < 0.4)
    (128.0, 0, 8.0),    # alpha band 8-13 Hz inclusive
    (135.0, 0, 13.0),
    (270.0, 2, 5.0),    # spike-wave SD >= 5 mV
    (135.0, 3, 5.0),    # E4 5 <= mean <= 10 mV
    (135.0, 3, 10.0),
])
def test_boundaries_pass(c, field, value):
    assert sc.score_seed(rec(edit(GOOD, c, field, value)))["pass"] is True


def test_missing_or_duplicate_c_row_is_refused():
    prof = dict(GOOD)
    del prof[1350.0]
    with pytest.raises(KeyError):
        sc.score_seed(rec(prof))
    r = rec(GOOD)
    r["rows"].append(dict(r["rows"][0]))
    with pytest.raises(KeyError):
        sc.score_seed(r)
