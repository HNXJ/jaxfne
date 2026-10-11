"""Acceptance tests for the PCL event-exact reference (port stage A). Spec: artifacts/etudes/pcl/PORT_SPEC.md."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root(start: Path) -> Path:
    for cand in (start, *start.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError(f"no pyproject.toml above {start}")


_ROOT = str(_repo_root(HERE))
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)
if str(HERE) not in sys.path:
    sys.path.insert(1, str(HERE))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pcl_event_ref as ref  # noqa: E402

SHAPES = Path("F:/warehouse/pcl/code/Predictive-Coding-Light_code/PCL_C++/data/events/shapes.h5")
TOL = 1e-12


def _write_config_dir(root: Path, *, network=None, simple=None, complex_=None) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    files = (
        ("network_config.json", network if network is not None else ref.DEFAULT_NETWORK),
        ("simple_cell_config.json", simple if simple is not None else ref.DEFAULT_SIMPLE),
        ("complex_cell_config.json", complex_ if complex_ is not None else ref.DEFAULT_COMPLEX),
    )
    for name, obj in files:
        (root / "configs" / name).write_text(json.dumps(obj), encoding="utf-8")
    return root


def _spike_setup(net: "ref.PCLEventRef") -> None:
    """Depth 5 at position (0,0) spikes on one event at pixel (157, 100); others stay sub-threshold."""
    d = np.arange(net.ND)
    net.w_s[:] = 0.0
    net.w_s[:, 0, 0, 0] = 0.5  # read by position (1,0) at x_rel = 0
    net.w_s[:, 0, 7, 0] = 0.5  # read by position (0,0) at x_rel = 7
    net.w_s[5, 0, 7, 0] = 40.0  # depth 5 at (0,0) crosses VTHRESH = 30
    net.wi_s[:] = 0.3 + 0.01 * d[:, None] + 0.001 * d[None, :]


def test_a_single_simple_cell_membrane_five_events():
    """Decay, refractory and adaptation terms, spike, reset and negative clamp, against hand arithmetic."""
    raw = copy.deepcopy(ref.DEFAULT_SIMPLE)
    raw["VTHRESH"] = 1.5  # threshold chosen so event 3 spikes
    raw["ETA_SRA"] = 2.0  # DELTA_SRA: adaptation increment per spike (Config.cpp:75)
    cfg = ref.CellConfig("simple", raw)
    cell = ref.Cell(cfg.VTHRESH)
    events = [(1000, 0.4), (3000, 0.5), (5000, 0.9), (7000, 0.3), (9000, 0.2)]
    got_spk, got_v, got_a = [], [], []
    for t, w in events:
        got_spk.append(ref.excite_simple(cell, cfg, t, w))
        got_v.append(cell.V)
        got_a.append(cell.A)

    e = math.exp
    tau_m, tau_sra, tau_rp = 18000.0, 100000.0, 5000.0  # us, from 18, 100 and 5 ms
    # Event 1, t=1000: V starts at 0, no refractory term yet.
    v1 = 0.4
    # Event 2, t=3000: decay over 2000 us, then +0.5.
    v2 = v1 * e(-2000 / tau_m) + 0.5
    # Event 3, t=5000: decay, then +0.9. The pre-reset value is above 1.5, so the cell spikes.
    v3_pre = v2 * e(-2000 / tau_m) + 0.9
    assert v3_pre > 1.5, "fixture must spike at event 3"
    v3, a3 = -10.0, 2.0  # reset to VRESET; A = 0 + DELTA_SRA
    # Event 4, t=7000: V and A decay over 2000 us; refractory 10*exp(-(7000-5000)/5000); A subtracted.
    a4 = a3 * e(-2000 / tau_sra)
    v4 = v3 * e(-2000 / tau_m) + (0.3 - 10.0 * e(-2000 / tau_rp) - a4)
    # Event 5, t=9000: decay over 2000 us; refractory 10*exp(-4000/5000); clamp at -20.
    a5 = a4 * e(-2000 / tau_sra)
    v5_pre = v4 * e(-2000 / tau_m) + (0.2 - 10.0 * e(-4000 / tau_rp) - a5)
    assert v5_pre < -20.0, "fixture must reach the negative clamp at event 5"
    v5 = -20.0

    assert got_spk == [False, False, True, False, False]
    for got, want in zip(got_v, [v1, v2, v3, v4, v5]):
        assert abs(got - want) < TOL
    assert abs(got_a[2] - a3) < TOL and abs(got_a[3] - a4) < TOL and abs(got_a[4] - a5) < TOL
    assert cell.spikes == [5000]


def test_b_local_inhibition_reaches_only_the_same_position():
    """A spike at depth 5 subtracts Winh[d, 5] from depths d != 5 at the same (x, y), and nowhere else."""
    net = ref.PCLEventRef(learning=False)  # frozen weights, so the spike does not move the inputs
    _spike_setup(net)
    wi = net.wi_s.copy()
    net.process_event(1000, 157, 100, 0)  # pixel 157 is covered by positions (0,0) and (1,0)
    assert net.s_cells[5].spikes == [1000], "fixture must spike at depth 5 of position (0,0)"
    p10 = net.NY * net.ND  # index of position (1, 0) at depth 0
    for k in range(net.ND):
        if k == 5:
            assert abs(net.s_cells[5].V - net.scfg.VRESET) < TOL
        else:
            assert abs(net.s_cells[k].V - (0.5 - wi[k, 5])) < TOL  # same (x, y): inhibited
        assert net.s_cells[p10 + k].spikes == []
        assert abs(net.s_cells[p10 + k].V - 0.5) < TOL  # other position, same pixel: not inhibited


def test_c_stdp_one_synapse_matches_hand_formula():
    """One excitatory synapse: LTP and LTD with the soft bounds, clip, and L1 normalisation to 50."""
    net = ref.PCLEventRef()
    k = 3
    net.w_s[k][...] = 0.25  # 2*10*10 synapses x 0.25 = 50 = NORM_FACTOR
    net.wi_s[k][...] = 1500.0 / 64
    cell = net.s_cells[k]  # position (0,0), depth 3
    cell.ev.append((100, 0, 0, 0))  # one event on synapse (polarity 0, x 0, y 0) at t=100
    cell.t_spk = 200
    cell.t_last_spk = 50
    net.weight_update_simple(k)

    e = math.exp
    e_ltp = 0.00077 * 50.0  # Neuron.cpp:31
    e_ltd = -0.00021 * 50.0  # Neuron.cpp:32
    w = 0.25
    d_ltp = (3.0 - w) * 0.33 * 1.0 * e_ltp * 1.0 * e(-(200 - 100) / 7000.0)  # tau_LTP = 7 ms
    d_ltd = (w * 0.33) * e_ltd * 1.0 * e(-(100 - 50) / 7000.0)  # tau_LTD = 7 ms; soft bound on pre-LTP w
    w_new = w + d_ltp + d_ltd
    assert w_new > 0, "fixture must not clip"
    scale = 50.0 / (199 * w + w_new)
    W = net.w_s[k]
    assert abs(W[0, 0, 0] - w_new * scale) < TOL
    others = np.delete(W.reshape(-1), 0)
    assert np.all(np.abs(others - w * scale) < TOL)
    assert abs(float(W.sum()) - 50.0) < TOL
    assert len(cell.ev) == 0


_RUNS: dict = {}


def _shapes_runs():
    """Two seeded runs on the first 50,000 events of shapes.h5 (computed once)."""
    if "runs" not in _RUNS:
        ev = ref.load_cpp_events(SHAPES, n_events=50000)
        runs = []
        for _ in range(2):
            net = ref.PCLEventRef(seed=0)
            t0 = time.perf_counter()
            net.run(ev["t"], ev["x"], ev["y"], ev["p"])
            elapsed = time.perf_counter() - t0
            runs.append({"trains": net.spike_trains(), "n": net.n_spikes(), "sec": elapsed,
                         "events": int(ev["t"].shape[0])})
        _RUNS["runs"] = runs
    return _RUNS["runs"]


def test_d_determinism_first_50000_events_shapes():
    run_a, run_b = _shapes_runs()
    assert run_a["events"] == 50000
    assert run_a["n"] > 0, "fixture must spike"
    assert run_a["trains"] == run_b["trains"]
    assert run_a["n"] == run_b["n"]


def test_e_wall_time_for_50000_events(capsys):
    run = _shapes_runs()[0]
    per_1e4 = run["sec"] / run["events"] * 1e4
    with capsys.disabled():
        print(f"\n[e] 50000 events: {run['sec']:.2f} s, {per_1e4:.3f} s per 10^4 events, "
              f"{run['n']} spikes")
    assert run["events"] == 50000


def test_f_loader_rebases_time_and_keeps_camera_zero():
    with h5py.File(SHAPES, "r") as f:
        first = int(f["events"]["t"][0])
    raw = ref.load_cpp_events(SHAPES, n_events=50000, rebase=False)
    ev = ref.load_cpp_events(SHAPES, n_events=50000)
    assert ev["t"].dtype == np.int64
    assert int(ev["t"][0]) == 0
    assert np.array_equal(raw["t"] - first, ev["t"])
    assert np.all(ev["c"] == 0)
    assert ev["t"].shape[0] == 50000, "fixture: first 50000 events are all camera 0"


def test_g_threshold_adaptation_runs_at_the_measurement_interval(tmp_path):
    """ETA_TA is read (Config.cpp:73) and applied at each 100 ms measurement (NetworkHandle.cpp:138)."""
    simple = copy.deepcopy(ref.DEFAULT_SIMPLE)
    simple["ETA_TA"] = 0.1
    net = ref.PCLEventRef(_write_config_dir(tmp_path / "net", simple=simple))
    assert net.scfg.ETA_SR == 0.1
    net.process_event(0, 150, 100, 0)  # no update: 0 - 0 is not above 100 ms
    net.process_event(350000, 150, 100, 0)  # one update with dt = 350 ms, rate 0 < 0.75
    assert net.n_spikes() == 0, "fixture must not spike"
    want = 30.0 - 0.1 * (1 - math.exp(0.0 - 0.75))  # Neuron.cpp:381
    thr_s = np.array([c.thr for c in net.s_cells])
    assert np.all(np.abs(thr_s - want) < TOL)
    assert all(c.thr == 3.0 for c in net.c_cells), "complex ETA_SR is 0 (NeuronConfig default)"


def test_h_learning_flag_freezes_weights_on_a_spike():
    """Positive control: learning on changes the weights at a spike. Learning off leaves them unchanged."""
    on = ref.PCLEventRef(learning=True)
    _spike_setup(on)
    w0 = on.w_s.copy()
    on.process_event(1000, 157, 100, 0)
    assert on.s_cells[5].spikes == [1000], "fixture must spike"
    assert not np.array_equal(on.w_s, w0), "positive control: learning must change weights"

    off = ref.PCLEventRef(learning=False)
    _spike_setup(off)
    w0 = off.w_s.copy()
    wi0 = off.wi_s.copy()
    off.process_event(1000, 157, 100, 0)
    assert off.s_cells[5].spikes == [1000]
    assert np.array_equal(off.w_s, w0)
    assert np.array_equal(off.wi_s, wi0)


def test_i_lateral_and_topdown_paths_run_with_normalised_maps(tmp_path):
    """Smoke and invariant check only: no C++ oracle exists for these maps in this stage."""
    net_raw = copy.deepcopy(ref.DEFAULT_NETWORK)
    net_raw["layerInhibitions"] = [["local", "lateral"], ["local", "topdown"]]
    net = ref.PCLEventRef(_write_config_dir(tmp_path / "lat", network=net_raw))
    assert int(net.lat_mask[0].sum()) == 24 * 64, "window +-4 minus own position, all depths"
    assert np.allclose(net.LAT.sum(axis=1), net.scfg.LATERAL_NORM_FACTOR, rtol=1e-12)
    assert np.allclose(net.TD.sum(axis=1), net.scfg.TOPDOWN_NORM_FACTOR, rtol=1e-12)

    _spike_setup(net)
    net.w_c[:, 0, 0, 5] = 10.0  # complex (0,0,*) read depth 5 strongly, so they spike
    net.process_event(1000, 157, 100, 0)
    assert net.s_cells[5].spikes == [1000], "fixture must spike"
    assert sum(len(c.spikes) for c in net.c_cells) > 0, "fixture: complex cells must spike"
    assert len(net.s_cells[5].td) > 0, "top-down events from complex spikes reach the simple cell"
    assert (1000, 5) in list(net.s_cells[(1 * net.NY + 0) * net.ND].lat), "lateral event reaches (1,0,0)"
