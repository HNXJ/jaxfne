"""Acceptance tests for the time-binned JAX port (port stage B). Spec: artifacts/etudes/pcl/PORT_SPEC.md.

Gate L1-G2 is test (b): per-cell simple and complex spike-count Pearson r >= 0.95 and totals within
5 % of stage A, at dt = 100 us, with no missed cascade crossings. The tolerances are not loosened;
the gate fails and is marked xfail(strict) with the measured values, so stage B is not the training
engine. dt = 50, 200 and 500 us are printed only. Test (g) is an exact check on one event per bin.
"""

from __future__ import annotations

import copy
import json
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

import h5py  # noqa: E402,F401
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pcl_binned_jax as bj  # noqa: E402
import pcl_event_ref as ref  # noqa: E402

SHAPES = Path("F:/warehouse/pcl/code/Predictive-Coding-Light_code/PCL_C++/data/events/shapes.h5")
N_SHAPES = 200000
N_SMALL = 20000
# Declared in the test: simple VTHRESH lowered from 30 (DefaultConfig.cpp). Stage A with learning
# off gives 737 simple spikes on these 200k events at 6.0 (measured in the gate run). 6.0 is above
# MIN_THRESH = 4. The threshold was chosen from a learning-on probe, before any binned run.
SIMPLE_VTHRESH = 6.0
GATE_R = 0.95
GATE_TOTAL = 0.05


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


def _pearson(u, v) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.std() == 0.0 or v.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


def _rel(total_b: int, total_a: int) -> float:
    return (total_b - total_a) / total_a


@pytest.fixture(scope="module")
def shapes_dir(tmp_path_factory) -> Path:
    simple = copy.deepcopy(ref.DEFAULT_SIMPLE)
    simple["VTHRESH"] = SIMPLE_VTHRESH
    return _write_config_dir(tmp_path_factory.mktemp("shapes") / "net", simple=simple)


_RUNS: dict = {}


def _stage_a_shapes(cdir: Path) -> dict:
    """Stage A on the first 200,000 events, once (cached): counts per cell and wall time."""
    if "A" not in _RUNS:
        ev = ref.load_cpp_events(SHAPES, n_events=N_SHAPES)
        net = ref.PCLEventRef(cdir, seed=0, learning=False)
        t0 = time.perf_counter()
        net.run(ev["t"], ev["x"], ev["y"], ev["p"])
        sec = time.perf_counter() - t0
        _RUNS["A"] = {
            "counts_s": np.array([len(c.spikes) for c in net.s_cells], dtype=np.int64),
            "counts_c": np.array([len(c.spikes) for c in net.c_cells], dtype=np.int64),
            "sec": sec,
            "events": int(ev["t"].shape[0]),
        }
    return _RUNS["A"]


def test_a_single_simple_neuron_binned_equals_event_exact(tmp_path):
    """One simple neuron, one event per bin at bin times: binned spike times equal stage A exactly."""
    net_raw = copy.deepcopy(ref.DEFAULT_NETWORK)
    net_raw["layerSizes"] = [[1, 1, 1], [1, 1, 1]]
    net_raw["neuronSizes"] = [[[10, 10, 1]], [[4, 4, 1]]]
    net_raw["neuronOverlap"] = [[3, 3, 1], [3, 3, 1]]
    simple = copy.deepcopy(ref.DEFAULT_SIMPLE)
    simple["VTHRESH"] = 8.0  # declared: default 30 gives one spike in these 400 events
    cdir = _write_config_dir(tmp_path / "one", network=net_raw, simple=simple)
    dt = 100
    k = np.arange(1, 401, dtype=np.int64)
    t = k * dt  # one event per bin, at the bin time
    x = 150 + (k * 7) % 10  # inside the receptive field of the single simple neuron
    y = 100 + (k * 3) % 10
    p = k % 2

    a = ref.PCLEventRef(cdir, seed=0, learning=False)
    a.run(t, x, y, p)
    want = a.spike_trains()[0][0]
    assert len(want) >= 2, "fixture must spike"

    b = bj.PCLBinned(cdir, seed=0, learning=False, dt_us=dt)
    got = b.run(t, x, y, p, record=True)["spike_times_s"][0]
    assert got == want, f"binned spikes {got} differ from event-exact {want}"


@pytest.mark.xfail(strict=True, reason=(
    "L1-G2 fails at dt=100 us (2026-10-11: simple r 0.874, complex r 0.594): 30 % of bins hold "
    "several events at one position and their order is lost. Simple r reaches 0.979 at dt=10 us; "
    "complex never reaches 0.95. Decision: L1 trains on stage A (event-exact, as fast on CPU)."))
def test_b_shapes_first_200k_events_binned_vs_event_exact(shapes_dir, capsys):
    """Gate L1-G2 on shapes.h5 at dt = 100 us: simple and complex r >= 0.95, totals within 5 %, no misses."""
    ev = ref.load_cpp_events(SHAPES, n_events=N_SHAPES)
    assert ev["t"].shape[0] == N_SHAPES, "fixture: first 200000 events are all camera 0"
    ref_run = _stage_a_shapes(shapes_dir)
    tot_a = int(ref_run["counts_s"].sum())
    tot_ac = int(ref_run["counts_c"].sum())
    assert tot_a >= 200, f"fixture must give >= 200 simple spikes, got {tot_a}"

    rows = {}
    for dt in (100, 50, 200, 500):
        b = bj.PCLBinned(shapes_dir, seed=0, learning=False, dt_us=dt)
        frac = b.multi_event_fraction(ev["t"], ev["x"], ev["y"])
        t0 = time.perf_counter()
        out = b.run(ev["t"], ev["x"], ev["y"], ev["p"])
        sec = time.perf_counter() - t0
        rows[dt] = {
            "r": _pearson(ref_run["counts_s"], out["counts_s"]),
            "rel": _rel(int(out["counts_s"].sum()), tot_a),
            "r_c": _pearson(ref_run["counts_c"], out["counts_c"]),
            "rel_c": _rel(int(out["counts_c"].sum()), tot_ac),
            "tot_b": int(out["counts_s"].sum()),
            "tot_bc": int(out["counts_c"].sum()),
            "miss_s": out["missed_s"],
            "miss_c": out["missed_c"],
            "multi": frac,
            "sec": sec,
        }
    with capsys.disabled():
        print(f"\n[b] stage A: {tot_a} simple spikes, {tot_ac} complex, "
              f"{ref_run['sec']:.2f} s on {ev['t'].shape[0]} events "
              f"({ref_run['sec'] / ev['t'].shape[0] * 1e5:.2f} s per 1e5 events)")
        for dt, row in rows.items():
            tag = "GATE" if dt == 100 else "print"
            m = row["multi"]
            print(f"[b] {tag} dt={dt:3d} us: simple r={row['r']:.4f} total={row['tot_b']} "
                  f"(rel {row['rel'] * 100:+.2f} %); complex r={row['r_c']:.4f} "
                  f"total={row['tot_bc']} (rel {row['rel_c'] * 100:+.2f} %); "
                  f"missed s={row['miss_s']} c={row['miss_c']}; "
                  f"multi-event bins {m['multi_bins']}/{m['nonempty_bins']} "
                  f"({m['fraction'] * 100:.2f} %); B {row['sec']:.2f} s "
                  f"({row['sec'] / ev['t'].shape[0] * 1e5:.2f} s per 1e5 events, includes JIT compile)")
    row = rows[100]
    assert row["r"] >= GATE_R, f"gate L1-G2: simple r {row['r']:.4f} < {GATE_R} at dt=100 us"
    assert abs(row["rel"]) <= GATE_TOTAL, (
        f"gate L1-G2: simple total differs by {row['rel'] * 100:+.2f} % at dt=100 us")
    assert row["r_c"] >= GATE_R, f"gate L1-G2: complex r {row['r_c']:.4f} < {GATE_R} at dt=100 us"
    assert abs(row["rel_c"]) <= GATE_TOTAL, (
        f"gate L1-G2: complex total differs by {row['rel_c'] * 100:+.2f} % at dt=100 us")
    assert row["miss_s"] == 0 and row["miss_c"] == 0, (
        f"cascade rounds missed crossings: simple {row['miss_s']}, complex {row['miss_c']}")


def test_g_one_event_per_bin_matches_event_exact_and_local_inhibition_is_active(tmp_path):
    """One event per bin (no shared position): simple spike times equal stage A, local inhibition on.

    Complex VTHRESH is raised to 1e6 so the complex layer never spikes; its visit order is not tested
    here. Positive control: stage A with local inhibition off gives different simple spike times.
    """
    simple = copy.deepcopy(ref.DEFAULT_SIMPLE)
    simple["VTHRESH"] = SIMPLE_VTHRESH
    cplx = copy.deepcopy(ref.DEFAULT_COMPLEX)
    cplx["VTHRESH"] = 1.0e6
    cdir = _write_config_dir(tmp_path / "g", simple=simple, complex_=cplx)
    # dt = 1 us: TAU_M = 18 us and TAU_RP = 5 us (ref.DEFAULT_SIMPLE), so a bin of 100 us erases the
    # membrane and refractory state between events; at 100 us spacing no simple cell can spike.
    dt = 1
    ev = ref.load_cpp_events(SHAPES, n_events=N_SMALL)
    t = (np.arange(ev["t"].shape[0], dtype=np.int64) + 1) * dt  # one event per bin, at the bin time

    a = ref.PCLEventRef(cdir, seed=0, learning=False)
    a.run(t, ev["x"], ev["y"], ev["p"])
    want_s, want_c = a.spike_trains()
    assert sum(len(w) for w in want_s) >= 50, "fixture must spike"
    assert sum(len(w) for w in want_c) == 0, "fixture: complex layer must not spike"

    b = bj.PCLBinned(cdir, seed=0, learning=False, dt_us=dt)
    assert b.multi_event_fraction(t, ev["x"], ev["y"])["fraction"] == 0.0, "fixture: one event per bin"
    out = b.run(t, ev["x"], ev["y"], ev["p"], record=True)
    assert out["missed_s"] == 0 and out["missed_c"] == 0
    assert out["spike_times_s"] == [list(w) for w in want_s], "simple spike times differ from stage A"
    assert int(out["counts_c"].sum()) == 0

    net_raw = copy.deepcopy(ref.DEFAULT_NETWORK)
    net_raw["layerInhibitions"] = [[], ["local"]]  # positive control: no local inhibition, simple layer
    cdir_off = _write_config_dir(tmp_path / "g_off", network=net_raw, simple=simple, complex_=cplx)
    a_off = ref.PCLEventRef(cdir_off, seed=0, learning=False)
    a_off.run(t, ev["x"], ev["y"], ev["p"])
    assert a_off.spike_trains()[0] != want_s, "positive control: local inhibition must change the spikes"


def test_c_binned_is_deterministic(shapes_dir):
    """Two fresh runs on the same events give identical counts, spike times and state."""
    ev = ref.load_cpp_events(SHAPES, n_events=N_SMALL)
    runs = []
    for _ in range(2):
        b = bj.PCLBinned(shapes_dir, seed=0, learning=False, dt_us=100)
        out = b.run(ev["t"], ev["x"], ev["y"], ev["p"], record=True)
        runs.append((b, out))
    (b1, o1), (b2, o2) = runs
    assert int(o1["counts_s"].sum()) > 0, "fixture must spike"
    assert np.array_equal(o1["counts_s"], o2["counts_s"])
    assert np.array_equal(o1["counts_c"], o2["counts_c"])
    assert o1["spike_times_s"] == o2["spike_times_s"]
    for key in b1.state:
        assert np.array_equal(np.asarray(b1.state[key]), np.asarray(b2.state[key])), key


def test_d_all_state_arrays_are_float32(shapes_dir):
    """Every state array is float32, with and without learning; x64 is off."""
    import jax
    import jax.numpy as jnp

    assert not bool(getattr(jax.config, "jax_enable_x64", False))
    assert jnp.zeros(1).dtype == jnp.float32
    ev = ref.load_cpp_events(SHAPES, n_events=2000)
    for learning in (False, True):
        b = bj.PCLBinned(shapes_dir, seed=0, learning=learning, dt_us=100)
        b.run(ev["t"], ev["x"], ev["y"], ev["p"])
        assert len(b.state) > 0
        for key, arr in b.state.items():
            assert arr.dtype == jnp.float32, f"learning={learning}: {key} is {arr.dtype}"


def test_e_learning_path_runs_and_keeps_invariants(shapes_dir):
    """Smoke (not in the packet): learning on changes the weights; L1 sums and signs hold."""
    ev = ref.load_cpp_events(SHAPES, n_events=N_SMALL)
    b = bj.PCLBinned(shapes_dir, seed=0, learning=True, dt_us=100)
    w0 = np.asarray(b.state["Ws"]).copy()
    out = b.run(ev["t"], ev["x"], ev["y"], ev["p"])
    assert int(out["counts_s"].sum()) > 0, "fixture must spike"
    ws = np.asarray(b.state["Ws"])
    assert np.all(np.isfinite(ws)) and np.all(ws >= 0.0)
    assert not np.array_equal(ws, w0), "positive control: learning must change weights"
    spiked = out["counts_s"].reshape(b.NX, b.NY, b.ND).sum(axis=(0, 1)) > 0
    sums = ws.reshape(b.ND, -1).sum(axis=1)
    assert np.allclose(sums[spiked], 50.0, rtol=1e-4), "depths that updated are L1-normalised to 50"
    wc = np.asarray(b.state["Wc"])
    assert np.all(np.isfinite(wc)) and np.all(wc >= 0.0)


def test_f_lateral_and_topdown_smoke_and_refusal(tmp_path):
    """Smoke (not in the packet): lateral and top-down run on a small network; their learning refuses."""
    net_raw = copy.deepcopy(ref.DEFAULT_NETWORK)
    net_raw["layerSizes"] = [[9, 9, 4], [6, 6, 2]]
    net_raw["neuronSizes"] = [[[10, 10, 1]], [[4, 4, 4]]]
    net_raw["neuronOverlap"] = [[3, 3, 1], [3, 3, 4]]
    net_raw["layerInhibitions"] = [["local", "lateral"], ["local", "topdown"]]
    simple = copy.deepcopy(ref.DEFAULT_SIMPLE)
    simple["VTHRESH"] = SIMPLE_VTHRESH
    cdir = _write_config_dir(tmp_path / "lat", network=net_raw, simple=simple)
    with pytest.raises(NotImplementedError):
        bj.PCLBinned(cdir, seed=0, learning=True, dt_us=100)
    ev = ref.load_cpp_events(SHAPES, n_events=N_SMALL)
    a = ref.PCLEventRef(cdir, seed=0, learning=False)
    a.run(ev["t"], ev["x"], ev["y"], ev["p"])
    b = bj.PCLBinned(cdir, seed=0, learning=False, dt_us=100)
    assert b.flags.lat and b.flags.td, "fixture: both maps enabled"
    out = b.run(ev["t"], ev["x"], ev["y"], ev["p"])
    for key, arr in b.state.items():
        assert np.all(np.isfinite(np.asarray(arr))), key
    a_s = np.array([len(c.spikes) for c in a.s_cells])
    print(f"\n[f] lateral+topdown small net: stage A simple {int(a_s.sum())}, "
          f"complex {sum(len(c.spikes) for c in a.c_cells)}; "
          f"B simple {int(out['counts_s'].sum())}, complex {int(out['counts_c'].sum())} "
          f"r_simple={_pearson(a_s, out['counts_s']):.4f} (not gated); "
          f"missed s={out['missed_s']} c={out['missed_c']}")
    assert int(out["counts_s"].sum()) > 0, "fixture must spike"
