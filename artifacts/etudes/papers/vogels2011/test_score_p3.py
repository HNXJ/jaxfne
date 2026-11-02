"""Tests for the P3 v4 signed scorer (CONV, R-E, O-DROP over seeds 4-8).

Five synthetic receipts stand in for the paper-scale scored seeds; the
scorer reads fields only, never the rasters, so the verdict logic runs in
seconds.
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from artifacts.etudes.papers.vogels2011 import score_p3 as S

SHA_A = "aa" * 32
SHA_B = "bb" * 32


def _receipt(seed, *, z=3.0, before=120.0, rates=None, status="signed",
             sha=SHA_A, w=2.0):
    if rates is None:
        rates = [z + 0.05 * math.sin(i) for i in range(150)]
    return {
        "seed": int(seed), "n": 10000, "T_s": 150.0, "status": status,
        "prereg": "artifacts/etudes/papers/vogels2011/prereg/P3_v4_signed.md",
        "prereg_sha256": sha,
        "e_rate_per_s_hz": list(rates),
        "R-E": {"y_hz": 3.0, "z_hz": float(z), "e": abs(z - 3.0) / 3.0},
        "O-DROP": {"before_hz": float(before), "z_hz": float(z),
                   "ratio": float(before) / float(z)},
        "W-IE": {"w_ie_final_mean_nS": float(w)},
        "AI": {"label": "irregular", "cv_isi": 0.05, "kappa": 0.3},
    }


def _write_receipts(d, receipts):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for r in receipts:
        (d / ("score_receipt_seed%d.json" % r["seed"])).write_text(
            json.dumps(r, sort_keys=True))


def _sloped_rates(slope=0.05):
    return [8.0 - slope * i for i in range(150)]


def test_closes_when_all_hold(tmp_path):
    """Five clean receipts close the lane (P3 v4: CONV, R-E, O-DROP)."""
    d = tmp_path / "clean"
    _write_receipts(d, [_receipt(s) for s in (4, 5, 6, 7, 8)])
    out = tmp_path / "verdict.json"
    assert S.main([str(d), str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["verdict"] == "closes", "clean seeds must close"
    assert got["CONV"]["pass"] is True
    assert got["R-E"]["pass"] is True
    assert got["O-DROP"]["pass"] is True


def test_one_nonconverged_seed_flips_verdict(tmp_path):
    """One sloped seed breaks CONV and the lane (P3 v4 item 3)."""
    recs = [_receipt(s) for s in (4, 5, 6, 7, 8)]
    recs[2]["e_rate_per_s_hz"] = _sloped_rates()
    d = tmp_path / "slope"
    _write_receipts(d, recs)
    out = tmp_path / "verdict.json"
    assert S.main([str(d), str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["CONV"]["per_seed"]["6"]["converged"] is False
    assert got["CONV"]["pass"] is False
    assert got["verdict"] != "closes", "a non-converged seed flips the lane"


def test_high_rate_flips_re(tmp_path):
    """Seed-mean e 0.41 breaks R-E and the lane (P3 v4, e <= 0.20)."""
    d = tmp_path / "high"
    _write_receipts(d, [_receipt(s, z=3.0 * 1.41) for s in (4, 5, 6, 7, 8)])
    out = tmp_path / "verdict.json"
    assert S.main([str(d), str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["R-E"]["e"] == pytest.approx(0.41, rel=1e-9)
    assert got["R-E"]["pass"] is False
    assert got["verdict"] != "closes", "R-E outside the band flips the lane"


def test_missing_seed_refused(tmp_path):
    """Four receipts are refused, never scored (P3 v4: seeds 4-8)."""
    d = tmp_path / "miss"
    _write_receipts(d, [_receipt(s) for s in (4, 5, 6, 7)])
    with pytest.raises(SystemExit):
        S.main([str(d), str(tmp_path / "v.json")])


def test_duplicate_seed_refused(tmp_path):
    """Two receipts for one seed are refused (P3 v4: seeds 4-8, once each)."""
    d = tmp_path / "dup"
    _write_receipts(d, [_receipt(s) for s in (4, 5, 6, 7, 8)])
    (d / "score_receipt_seed4_copy.json").write_text(
        (d / "score_receipt_seed4.json").read_text())
    with pytest.raises(SystemExit):
        S.main([str(d), str(tmp_path / "v.json")])


def test_mixed_sha_refused(tmp_path):
    """A mixed-prereg-SHA set is refused (P3 v4: one frozen protocol)."""
    recs = [_receipt(s) for s in (4, 5, 6, 7, 8)]
    recs[0]["prereg_sha256"] = SHA_B
    d = tmp_path / "mix"
    _write_receipts(d, recs)
    with pytest.raises(SystemExit):
        S.main([str(d), str(tmp_path / "v.json")])


def test_unsigned_refused(tmp_path):
    """A non-signed receipt is refused (P3 v4: signed only)."""
    recs = [_receipt(s) for s in (4, 5, 6, 7, 8)]
    recs[1]["status"] = "unscored pilot"
    d = tmp_path / "uns"
    _write_receipts(d, recs)
    with pytest.raises(SystemExit):
        S.main([str(d), str(tmp_path / "v.json")])


def test_short_series_refused(tmp_path):
    """A T < 150 s receipt is refused by the scorer (window is 130-150 s)."""
    recs = [_receipt(s) for s in (4, 5, 6, 7, 8)]
    recs[3]["e_rate_per_s_hz"] = recs[3]["e_rate_per_s_hz"][:20]
    d = tmp_path / "short"
    _write_receipts(d, recs)
    with pytest.raises(SystemExit):
        S.main([str(d), str(tmp_path / "v.json")])
