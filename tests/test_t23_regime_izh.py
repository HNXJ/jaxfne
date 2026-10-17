"""T-23 part 2: activity-regime labels on the Izhikevich 2003 network scorer.

``score_network_seed`` defaults to the whole network as ``"all"``;
``score_fig3_dir`` passes E = cols 0-799, I = cols 800-999 per
``build_fig3_network`` and rolls the labels up per seed. N1-N4 are untouched.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import types

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_IZH_PATH = REPO_ROOT / "artifacts" / "etudes" / "papers" / "izhikevich2003" / "run_izh.py"


def _load_run_izh() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_izh_t23", RUN_IZH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_defaults_to_whole_network_all():
    r = _load_run_izh()
    n_steps, dt_ms = 2000, 0.5
    cells = [[] for _ in range(10)]
    for j in range(10):
        cells[j] = [500 + 100 * k + j for k in range(10)]
    got = r.score_network_seed(cells, n_steps, dt_ms, 4.0, (8.0, 13.0))
    regime = got["activity_regime"]
    assert set(regime["populations"]) == {"all"}
    assert regime["labels"] == {"all": regime["populations"]["all"]["label"]}
    assert regime["window_steps"] == [400, n_steps]  # 200 ms discard at 0.5 ms
    assert regime["bin_ms"] == 5.0 and regime["dt_ms"] == dt_ms
    assert got["N1"]["numbers"]["mean_rate_hz"] > 0.0  # scored numbers intact


def test_seed_silent_labels_silent():
    r = _load_run_izh()
    got = r.score_network_seed([[]] * 10, 2000, 0.5, 4.0, (8.0, 13.0))
    assert got["activity_regime"]["labels"] == {"all": "silent"}
    assert got["N1"]["numbers"]["mean_rate_hz"] == 0.0  # scored numbers intact


def _write_fig3_seed(fig_dir, seed, e_rows, i_rows, n_steps=20400):
    """One synthetic Fig. 3 seed: E cells (cols 0-799) spike, I cells may not."""
    srows, scols = [], []
    for j in range(800):
        for row in e_rows:
            srows.append(row)
            scols.append(j)
    for j in range(800, 1000):
        for row in i_rows:
            srows.append(row)
            scols.append(j)
    np.savez_compressed(
        fig_dir / f"{seed}.npz",
        spike_rows=np.asarray(srows, dtype=np.int64),
        spike_cols=np.asarray(scols, dtype=np.int64),
        n_steps=np.int64(n_steps),
        n_cells=np.int64(1000),
        dt_ms=np.float64(0.5),
    )
    return len(srows)


def _write_fig3_manifest(fig_dir, n_spikes_by_seed):
    manifest = {
        "prereg": "T-23 synthetic",
        "code_sha": "t23",
        "dtype": "float32",
        "dt_ms": 0.5,
        "T_s": 10.2,
        "seeds": [1, 2, 3, 4, 5],
        "sensitivity": None,
        "runs": {
            str(seed): {"seed": seed, "file": f"{seed}.npz", "n_steps": 20400,
                        "dt_ms": 0.5, "dtype": "float32", "n_spikes": n_spikes}
            for seed, n_spikes in n_spikes_by_seed.items()
        },
    }
    (fig_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_fig3_dir_reports_e_and_i_labels(tmp_path):
    r = _load_run_izh()
    e_rows = [500 + 200 * k for k in range(40)]  # E fires; I is silent
    n_spikes = {seed: _write_fig3_seed(tmp_path, seed, e_rows, []) for seed in (1, 2, 3, 4, 5)}
    _write_fig3_manifest(tmp_path, n_spikes)
    got = r.score_fig3_dir(tmp_path, 4.0, (8.0, 13.0))
    assert got["complete"] is True
    for seed in ("1", "2", "3", "4", "5"):
        pops = got["seeds"][seed]["activity_regime"]["populations"]
        assert set(pops) == {"E", "I"}
        assert pops["E"]["active_fraction"] == 1.0  # the E/I split is applied
        assert pops["I"]["active_fraction"] == 0.0
        assert got["seeds"][seed]["activity_regime"]["labels"]["I"] == "silent"
    assert set(got["activity_regime_labels"]) == {"1", "2", "3", "4", "5"}
    assert got["activity_regime_labels"]["1"] == got["seeds"]["1"]["activity_regime"]["labels"]
    assert "N1" in got["seeds"]["1"] and "N1" in got["overall"]  # scored keys intact


def test_fig3_dir_incomplete_seed_labels_none(tmp_path):
    r = _load_run_izh()
    e_rows = [500 + 200 * k for k in range(40)]
    n_spikes = {seed: len(e_rows) * 800 for seed in (1, 2, 3, 4, 5)}
    for seed in (1, 2, 3, 4):  # seed 5 entry exists but its file is missing
        _write_fig3_seed(tmp_path, seed, e_rows, [])
    _write_fig3_manifest(tmp_path, n_spikes)
    got = r.score_fig3_dir(tmp_path, 4.0, (8.0, 13.0))
    assert got["seeds"]["5"]["verdict"] == "incomplete"
    assert got["activity_regime_labels"]["5"] is None
    assert got["activity_regime_labels"]["1"] is not None
