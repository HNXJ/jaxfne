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


@pytest.mark.parametrize("c, field, value, crit", [
    (135.0, 1, 0.3, "E1"),     # alpha at 135 becomes noise
    (135.0, 1, 0.70, "E2"),    # osc(135) below osc(128)
    (675.0, 0, 5.5, "E3"),     # no slowing
    (135.0, 3, 11.0, "E4"),    # EEG level out of 5-10 mV
    (270.0, 2, 3.0, "E1"),     # spike-wave SD below 5 mV
])
def test_each_criterion_fails_on_its_break(c, field, value, crit):
    prof = copy.deepcopy(GOOD)
    row = list(prof[c])
    row[field] = value
    prof[c] = tuple(row)
    r = sc.score_seed(rec(prof))
    assert r["criteria"][crit] is False and r["pass"] is False


def test_missing_c_row_is_refused():
    prof = dict(GOOD)
    del prof[1350.0]
    with pytest.raises(KeyError):
        sc.score_seed(rec(prof))
