"""Amendment v8.2 tests (PREREG_P3_v8.md): relative C4, C8 amplitude 2, one-SHA, clean tree.

Synthetic and fast: no scored runs, no F: paths. Every test uses a symbol added
for v8, so on the baseline implementation it fails (AttributeError).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import types

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_IZH_PATH = REPO_ROOT / "artifacts" / "etudes" / "izhikevich2003" / "run_izh.py"
DT_MS = 0.1  # kernel dt of the Fig. 2 runs; 10 ms = 100 rows
TEST_SHA = "a" * 40
TEST_SHA_8 = "b" * 40


def _load_run_izh() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_izh_v8", RUN_IZH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows_from_gaps(gaps, start=0):
    rows = [start]
    for g in gaps:
        rows.append(rows[-1] + int(g))
    return rows


def test_v8_c8_protocol_amplitude_2_while_v7_stays_5():
    r = _load_run_izh()
    v8 = r.expected_fig2_v8_settings()["C8"]
    assert v8["segments"] == [[10.0, 0.0], [1.0, 2.0], [311.0, 0.0]]
    assert v8["event_row"] == 110 and v8["n_steps"] == 3220
    assert r.V8_C8_AMPLITUDE == 2.0
    v7 = r.FIG2_RUNS["C8"]
    assert v7[1][1][1] == 5.0 and v7[2] == 110 and v7[3] == 3220


def test_v8_c4_relative_pass():
    r = _load_run_izh()
    c4 = _rows_from_gaps([100] + [150] * 13)  # 15 spikes, 50 Hz, ratio 1.5
    c1 = _rows_from_gaps([100, 200, 300])  # ratio 3.0
    c5 = _rows_from_gaps([100, 200, 400])  # ratio 4.0
    got = r.eval_c4_v8(c4, c1, c5, DT_MS, 0, 0, 0)
    assert got["numbers"]["rate_hz"] == pytest.approx(50.0)
    assert got["numbers"]["isi_ratio"] == pytest.approx(1.5)
    assert got["numbers"]["c1_isi_ratio"] == pytest.approx(3.0)
    assert got["numbers"]["c5_isi_ratio"] == pytest.approx(4.0)
    assert got["verdict"] == "pass"


def test_v8_c4_strict_equality_fails():
    r = _load_run_izh()
    c4 = _rows_from_gaps([100] * 7 + [300] * 7)  # 15 spikes, 50 Hz, ratio 3.0
    c1 = _rows_from_gaps([100, 200, 300])  # ratio 3.0
    c5 = _rows_from_gaps([100, 200, 400])  # ratio 4.0
    got = r.eval_c4_v8(c4, c1, c5, DT_MS, 0, 0, 0)
    assert got["numbers"]["isi_ratio"] == pytest.approx(got["numbers"]["c1_isi_ratio"])
    assert got["verdict"] == "fail"  # strictly below is required


def test_v8_c4_none_comparator_fails():
    r = _load_run_izh()
    c4 = _rows_from_gaps([100] + [150] * 13)  # valid: 50 Hz, ratio 1.5
    c1 = [0, 100]  # 2 spikes: no ratio
    c5 = _rows_from_gaps([100, 200, 400])
    got = r.eval_c4_v8(c4, c1, c5, DT_MS, 0, 0, 0)
    assert got["numbers"]["c1_isi_ratio"] is None
    assert got["verdict"] == "fail"


def _write_v7_fig2(r, d, sha, with_c5=True):
    runs = {}
    rows_by_id = {
        "C1": _rows_from_gaps([100, 200, 300], start=100),
        "C4": _rows_from_gaps([100] + [150] * 13, start=100),
        "C5": _rows_from_gaps([100, 200, 400], start=100),
    }
    expected = r.expected_fig2_settings()
    for rid, rows in rows_by_id.items():
        if rid == "C5" and not with_c5:
            continue
        np.savez(d / f"{rid}.npz", spike_rows=np.asarray(rows, dtype=np.int64))
        runs[rid] = dict(expected[rid], file=f"{rid}.npz", n_spikes=len(rows))
    (d / "manifest.json").write_text(
        json.dumps({"code_sha": sha, "sensitivity": None, "runs": runs}))


def _write_v8_fig2(r, d, sha):
    expected = r.expected_fig2_v8_settings()["C8"]
    # Damped 20 Hz oscillation starting at the pulse end (row 110), flat before.
    idx = np.arange(1110) - 110
    t = np.where(idx >= 0, idx, 0) * 1e-4
    osc = 2.0 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 20.0 * t)
    v = -62.5 + np.where(idx >= 0, osc, 0.0)
    np.savez(d / "C8.npz", spike_rows=np.asarray([], dtype=np.int64), v=v)
    runs = {"C8": dict(expected, file="C8.npz", n_spikes=0)}
    (d / "manifest.json").write_text(
        json.dumps({"code_sha": sha, "amendment": "v8.2",
                    "sensitivity": None, "runs": runs}))


def _write_v7_score(d, sha):
    criteria = {rid: {"verdict": "pass", "numbers": {"n": 1}}
                for rid in ("C1", "C2", "C3", "C5", "C6", "C7")}
    score = {"prereg": "PREREG_P3.md signed v7", "run_code_shas": [sha],
             "fig2": {"criteria": criteria},
             "fig3": {"overall": {"N1": "pass", "N2": "fail",
                                  "N3": "pass", "N4": "pass"}}}
    (d / "v7_score.json").write_text(json.dumps(score))
    return d / "v7_score.json"


def test_v8_missing_c5_run_makes_c4_incomplete(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA, with_c5=False)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    score = _write_v7_score(tmp_path, TEST_SHA)
    rep = r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")
    assert rep["v7_sha"] == TEST_SHA and rep["v8_sha"] == TEST_SHA_8
    assert rep["fig2"]["criteria"]["C4"]["verdict"] == "incomplete"
    assert rep["fig2"]["criteria"]["C1"] == {"verdict": "pass", "numbers": {"n": 1}}
    assert rep["fig2"]["overall"] == "incomplete"
    assert rep["fig3"]["overall"]["N2"] == "fail"  # Fig. 3 copied unchanged


def test_v8_scores_c4_and_c8_on_synthetic_runs(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    score = _write_v7_score(tmp_path, TEST_SHA)
    rep = r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")
    assert rep["fig2"]["criteria"]["C4"]["verdict"] == "pass"
    assert rep["fig2"]["criteria"]["C8"]["verdict"] == "pass"
    assert rep["amendment"] == "v8.2"


def test_v8_one_sha_mismatch_refuses(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    score = _write_v7_score(tmp_path, "c" * 40)  # not the v7 dir's SHA
    with pytest.raises(SystemExit):
        r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")


def test_v8_missing_v8_sha_refuses(tmp_path):
    r = _load_run_izh()
    v7d, v8d = tmp_path / "v7", tmp_path / "v8"
    v7d.mkdir()
    v8d.mkdir()
    _write_v7_fig2(r, v7d, TEST_SHA)
    _write_v8_fig2(r, v8d, TEST_SHA_8)
    (v8d / "manifest.json").write_text(json.dumps({"sensitivity": None, "runs": {}}))
    score = _write_v7_score(tmp_path, TEST_SHA)
    with pytest.raises(SystemExit):
        r.cmd_v8(score, v7d, v8d, tmp_path / "v8.json")


def test_v8_tree_dirty_refuses_fig2_v8(tmp_path, monkeypatch):
    r = _load_run_izh()
    assert r._v8_tree_dirty(" M artifacts/etudes/izhikevich2003/run_izh.py")
    assert not r._v8_tree_dirty("?? scratch/new-file.txt\n")
    assert not r._v8_tree_dirty("")
    monkeypatch.setattr(r, "_git_status_porcelain",
                        lambda: " M artifacts/etudes/izhikevich2003/run_izh.py")
    with pytest.raises(SystemExit) as exc:
        r.cmd_fig2_v8(tmp_path / "v8")
    assert exc.value.code == 2


def test_v8_failed_git_status_reads_dirty(monkeypatch):
    r = _load_run_izh()
    failed = types.SimpleNamespace(returncode=128, stdout="")
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **k: failed)
    assert r._v8_tree_dirty()

    def _raise(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(r.subprocess, "run", _raise)
    assert r._v8_tree_dirty()
