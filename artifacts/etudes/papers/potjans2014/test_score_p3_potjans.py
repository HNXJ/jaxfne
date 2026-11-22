"""Tests for the P3_v2 signed scorer: five seed manifests to the lane verdict."""

import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

from artifacts.etudes.papers.potjans2014 import score_p3 as S

Y = {"L23E": 0.86, "L4E": 4.45, "L5E": 7.59, "L6E": 1.09}
R_POPS = {"R-L23E": "L23E", "R-L4E": "L4E", "R-L5E": "L5E", "R-L6E": "L6E"}
POPS = ("L23E", "L23PV", "L4E", "L4PV", "L5E", "L5PV", "L6E", "L6PV")
LAYERS = ("L23", "L4", "L5", "L6")


def _passing_rates():
    """E rates on the paper values, I rates above E in every layer."""
    rates = dict(Y)
    for layer in LAYERS:
        rates[layer + "PV"] = 2.0 * Y[layer + "E"]
    return rates


def _manifest(seed, rates, cvs, status="signed", sha="abc123"):
    rows = {}
    for row, pop in sorted(R_POPS.items()):
        y, z = Y[pop], float(rates[pop])
        e = abs(z - y) / y
        rows[row] = {"y_hz": y, "z_hz": z, "e": e, "pass": bool(e <= 0.20)}
    oi = {layer: {"e_hz": float(rates[layer + "E"]),
                  "i_hz": float(rates[layer + "PV"]),
                  "pass": bool(rates[layer + "PV"] > rates[layer + "E"])}
          for layer in LAYERS}
    return {"git_sha": sha,
            "runs": [{"seed": seed, "status": status,
                      "file": "spikes_seed%d_scored.npz" % seed,
                      "estimands_p3v2": {
                          "rates_hz": {p: float(rates[p]) for p in POPS},
                          "R": rows, "O-IE": oi,
                          "O-ORD": {"pass": True},
                          "CV": {"cv_isi": {p: float(cvs[p]) for p in POPS},
                                 "pass": True},
                          "SYN_fano_3ms": {p: 1.0 for p in POPS}}}]}


def _write_set(tmp_path, rates_by_seed=None, cvs=None, status="signed",
               sha="abc123", seeds=(4, 5, 6, 7, 8), sha_by_seed=None):
    run_dir = tmp_path / "run"
    for seed in seeds:
        rates = dict(rates_by_seed[seed]) if rates_by_seed else _passing_rates()
        one_cvs = dict(cvs) if cvs else {p: 0.9 for p in POPS}
        one_sha = sha_by_seed[seed] if sha_by_seed else sha
        one_status = status[seed] if isinstance(status, dict) else status
        dest = run_dir / ("seed%d" % seed)
        dest.mkdir(parents=True)
        (dest / "manifest.json").write_text(
            json.dumps(_manifest(seed, rates, one_cvs, one_status, one_sha)))
    return run_dir


def test_closes_when_all_rows_hold(tmp_path):
    """Five in-band synthetic manifests score "closes"."""
    run_dir = _write_set(tmp_path)
    out = tmp_path / "score.json"
    assert S.main([str(run_dir), str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "closes", "every primary row holds"
    for row in ("R-L23E", "R-L4E", "R-L5E", "R-L6E", "O-IE", "O-ORD", "CV"):
        assert doc["rows"][row]["pass"] is True, "row %s holds" % row
    assert "SYN_fano_3ms" in doc["secondary"], "secondary reported"
    assert len(doc["inputs_sha256"]) == 64, "inputs sha256 recorded"


def test_one_off_row_flips_verdict_only(tmp_path):
    """One R row out of band flips only that row and the verdict."""
    bad = _passing_rates()
    bad["L5E"] = Y["L5E"] * 1.5
    run_dir = _write_set(tmp_path, rates_by_seed={s: bad for s in (4, 5, 6, 7, 8)})
    out = tmp_path / "score.json"
    assert S.main([str(run_dir), str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "off", "one primary row off makes the result off"
    assert doc["rows"]["R-L5E"]["pass"] is False, "R-L5E outside its band"
    for row in ("R-L23E", "R-L4E", "R-L6E", "O-IE", "O-ORD", "CV"):
        assert doc["rows"][row]["pass"] is True, "row %s still holds" % row


def test_missing_seed_refused(tmp_path):
    """A set missing seed 8 is refused."""
    run_dir = _write_set(tmp_path, seeds=(4, 5, 6, 7))
    with pytest.raises(SystemExit, match="missing"):
        S.main([str(run_dir), str(tmp_path / "score.json")])


def test_mixed_sha_refused(tmp_path):
    """A set with two code SHAs is refused."""
    run_dir = _write_set(tmp_path, sha_by_seed={4: "abc123", 5: "abc123",
                                                6: "abc123", 7: "abc123",
                                                8: "def456"})
    with pytest.raises(SystemExit, match="mixed"):
        S.main([str(run_dir), str(tmp_path / "score.json")])


def test_unsigned_manifest_refused(tmp_path):
    """A set holding a non-signed run is refused."""
    run_dir = _write_set(tmp_path, status={4: "signed", 5: "signed",
                                           6: "unscored pilot", 7: "signed",
                                           8: "signed"})
    with pytest.raises(SystemExit, match="signed"):
        S.main([str(run_dir), str(tmp_path / "score.json")])
