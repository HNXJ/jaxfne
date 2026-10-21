"""T-23 part 2: activity-regime labels on the Destexhe 2009 scorer.

``cmd_score`` reports the whole-network regime under ``"all"`` (the raster
stores no RS/FS split) alongside the untouched scored numbers.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
from contextlib import redirect_stdout

import numpy as np

from artifacts.etudes.papers.destexhe2009 import run_destexhe as R

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _score_raster(tmp_path, spikes, dt_ms=0.1):
    """Score a synthetic raster through the CLI scorer; return its report."""
    with open(tmp_path / "spikes_seed1_t.npz", "wb") as f:
        np.savez(f, spikes=np.asarray(spikes, dtype=bool), dt_ms=np.float64(dt_ms))
    args = argparse.Namespace(seed=1, tag="t", out=str(tmp_path), report=None)
    buf = io.StringIO()
    with redirect_stdout(buf):
        R.cmd_score(args)
    return json.loads(buf.getvalue())


def _spiking_raster():
    """4 cells, 2 s at dt 0.1 ms: every cell fires regularly; must not be silent."""
    steps, n = 20000, 4
    spikes = np.zeros((steps, n), dtype=bool)
    for j in range(n):
        spikes[np.arange(100 + j, steps, 500), j] = True
    assert int(spikes.sum()) > 100, "fixture must spike"
    return spikes


def test_score_reports_whole_network_regime(tmp_path):
    rep = _score_raster(tmp_path, _spiking_raster())
    regime = rep["activity_regime"]
    assert set(regime["populations"]) == {"all"}
    assert regime["labels"] == {"all": regime["populations"]["all"]["label"]}
    assert regime["labels"]["all"] != "silent", "spiking raster must not label silent"
    assert regime["window_steps"] == [0, 20000]
    assert regime["bin_ms"] == 5.0 and regime["dt_ms"] == 0.1
    assert rep["n"] == 4 and rep["steps"] == 20000  # scored fields intact


def test_score_silent_raster_labels_silent(tmp_path):
    rep = _score_raster(tmp_path, np.zeros((20000, 4), dtype=bool))
    regime = rep["activity_regime"]
    assert regime["labels"] == {"all": "silent"}
    assert regime["populations"]["all"]["active_fraction"] == 0.0
    assert rep["alive_last_s"] is False  # scored field intact
