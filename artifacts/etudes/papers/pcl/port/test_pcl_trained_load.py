"""Acceptance tests for loading the trained PCL network and the numba run path (port stage A).

Spec: artifacts/etudes/pcl/PORT_SPEC.md. Reference: pcl_event_ref.py (pure Python), pcl_event_ref_numba.py.
The lateral and top-down maps (<n>li.npy, <n>tdi.npy) of the trained network load in the emulated
libstdc++ order (libstdcxx_umap.py; pcl_event_ref.py docstring), so the full-network tests run unmarked.
"""

from __future__ import annotations

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

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pcl_event_ref as ref  # noqa: E402
import pcl_event_ref_numba as nb  # noqa: E402

CSTDP3 = Path("F:/warehouse/pcl/data/PCL_datasets_and_networks/add_simulations_cSTDP/cSTDP3")
TEST7 = Path("F:/warehouse/pcl/data/PCL_datasets_and_networks/grating/counterphase_gratingComplex/"
             "test7/0/0/0/0.npz")
SHAPES = Path("F:/warehouse/pcl/code/Predictive-Coding-Light_code/PCL_C++/data/events/shapes.h5")
# Trained weights with the lateral and top-down maps dropped (test ablation, not the authors' network).
ABLATE = [["static"], ["static"]]


def _grating_events(n: int):
    """First n events of test7/0/0/0/0.npz (arr_0..arr_4 = t, x, y, p, c)."""
    with np.load(TEST7) as z:
        t = z["arr_0"][:n].astype(np.int64)
        x = z["arr_1"][:n].astype(np.int64)
        y = z["arr_2"][:n].astype(np.int64)
        p = z["arr_3"][:n].astype(np.int64)
        c = z["arr_4"][:n]
    assert t.shape[0] == n, "fixture: grating file must hold n events"
    assert np.all(c == 0), "fixture: the first events must be camera 0"
    return t, x, y, p


_SHAPES: dict = {}


def _shapes_events(n: int):
    key = ("ev", n)
    if key not in _SHAPES:
        _SHAPES[key] = ref.load_cpp_events(SHAPES, n_events=n)
    ev = _SHAPES[key]
    return ev["t"], ev["x"], ev["y"], ev["p"]


def _assert_same_state(a: "ref.PCLEventRef", b: "ref.PCLEventRef") -> None:
    assert a.spike_trains() == b.spike_trains(), "spike trains differ between the pure and numba paths"
    assert a.n_spikes() == b.n_spikes()
    for ca, cb in zip(a.s_cells + a.c_cells, b.s_cells + b.c_cells):
        assert ca.V == cb.V and ca.thr == cb.thr and ca.rate == cb.rate and ca.count == cb.count
        assert ca.tlast == cb.tlast and ca.t_spk == cb.t_spk and ca.t_last_spk == cb.t_last_spk


def _run_both(factory, t, x, y, p):
    """Pure-Python and numba runs of two fresh networks from the same factory; returns (py, nb, s, s)."""
    py = factory()
    nbn = factory()
    nb.warm_up()  # compile outside the timed region
    t0 = time.perf_counter()
    py.run(t, x, y, p)
    sec_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    nb.run_numba(nbn, t, x, y, p)
    sec_nb = time.perf_counter() - t0
    return py, nbn, sec_py, sec_nb


# ----- (a) loading ------------------------------------------------------------------------------

def test_a_trained_arrays_have_expected_shapes_and_nonnegative_weights():
    net = ref.PCLEventRef(CSTDP3, learning=False, layer_inhibitions=ABLATE)
    assert net.w_s.shape == (64, 2, 10, 10)
    assert net.wi_s.shape == (64, 64)
    assert net.w_c.shape == (32, 4, 4, 64)
    assert net.wi_c.shape == (32, 32)
    for name in ("w_s", "wi_s", "w_c", "wi_c"):
        arr = getattr(net, name)
        assert np.all(np.isfinite(arr)), name
        assert np.all(arr >= 0), f"{name} has negative weights"
    assert net.scfg.VTHRESH == 10.0 and net.scfg.TAU_SRA == 18000.0  # trained simple config, in us


def test_a_round_trip_three_named_files_and_state():
    net = ref.PCLEventRef(CSTDP3, learning=False, layer_inhibitions=ABLATE)
    w = CSTDP3 / "weights"
    checks = [
        (w / "0" / "3.npy", net.w_s[3]),  # simple shared excitatory, depth 3
        (w / "0" / "12lli.npy", net.wi_s[12]),  # simple shared local inhibition, depth 12
        (w / "1" / "1100.npy", net.w_c[1100 % 32]),  # complex cell 1100 shares depth 12
    ]
    for path, got in checks:
        raw = np.load(path, allow_pickle=False)
        assert raw.dtype == np.float64, path.name
        assert np.array_equal(raw.reshape(got.shape), got), f"round trip differs for {path}"
    st = json.loads((w / "0" / "5.json").read_text(encoding="utf-8"))
    assert net.s_cells[5].rate == float(st["spiking_rate"])
    assert net.s_cells[5].decay == float(st["learning_decay"])


def test_a_full_trained_network_loads():
    net = ref.PCLEventRef(CSTDP3, learning=False)
    assert net.lat_on and net.td_on
    assert net.LAT.shape == (net.NS, net.NS) and net.TD.shape == (net.NS, net.NC)
    w = CSTDP3 / "weights" / "0"
    for n in (0, 2600):  # a corner cell and an interior cell: each file holds exactly its map's values
        li, tdi = np.load(w / f"{n}li.npy"), np.load(w / f"{n}tdi.npy")
        assert np.array_equal(np.sort(net.LAT[n, net.lat_mask[n]]), np.sort(li)), f"li of cell {n}"
        assert np.array_equal(np.sort(net.TD[n, net.td_mask[n]]), np.sort(tdi)), f"tdi of cell {n}"
        assert np.all(net.LAT[n, ~net.lat_mask[n]] == 0) and np.all(net.TD[n, ~net.td_mask[n]] == 0)


# ----- (b) grating stimulus, trained weights ----------------------------------------------------

def test_b_cstdp3_full_network_on_100k_grating_events_matches_numba(capsys):
    t, x, y, p = _grating_events(100000)
    py, nbn, sec_py, sec_nb = _run_both(
        lambda: ref.PCLEventRef(CSTDP3, learning=False), t, x, y, p)
    _assert_same_state(py, nbn)
    n_simple = sum(len(c.spikes) for c in py.s_cells)
    n_complex = sum(len(c.spikes) for c in py.c_cells)
    assert n_simple > 0 and n_complex > 0, "fixture must spike in both layers"
    with capsys.disabled():
        print(f"\n[b-full] cSTDP3 full network, 1e5 grating events: simple spikes {n_simple}, "
              f"complex spikes {n_complex}")
        print(f"[b-full] pure {sec_py:.2f} s per 1e5 events; numba {sec_nb:.3f} s per 1e5 events")


def test_b_trained_weights_ablated_100k_grating_pure_vs_numba(capsys):
    """Trained excitatory and local weights; lateral and top-down maps dropped from the config."""
    t, x, y, p = _grating_events(100000)
    py, nbn, sec_py, sec_nb = _run_both(
        lambda: ref.PCLEventRef(CSTDP3, learning=False, layer_inhibitions=ABLATE), t, x, y, p)
    _assert_same_state(py, nbn)
    n_simple = sum(len(c.spikes) for c in py.s_cells)
    n_complex = sum(len(c.spikes) for c in py.c_cells)
    assert n_simple > 0, "fixture must spike (simple)"
    with capsys.disabled():
        print(f"\n[b] ablated cSTDP3, 1e5 grating events: simple spikes {n_simple}, complex spikes {n_complex}")
        print(f"[b] pure {sec_py:.2f} s per 1e5 events; numba {sec_nb:.3f} s per 1e5 events")


# ----- (c) default config, shapes ---------------------------------------------------------------

def test_c_default_config_50k_shapes_pure_vs_numba(capsys):
    t, x, y, p = _shapes_events(50000)
    py, nbn, sec_py, sec_nb = _run_both(
        lambda: ref.PCLEventRef(seed=0, learning=False), t, x, y, p)
    _assert_same_state(py, nbn)
    assert py.n_spikes() > 0, "fixture must spike"
    with capsys.disabled():
        print(f"\n[c] default config, 5e4 shapes events: pure {sec_py / 0.5:.2f} s per 1e5 events; "
              f"numba {sec_nb / 0.5:.3f} s per 1e5 events; spikes {py.n_spikes()}")


# ----- (d) every inhibition branch, random weights ----------------------------------------------

def _scaled_all_branches(seed: int = 1) -> "ref.PCLEventRef":
    """Random seeded weights scaled by 5 (unscaled ones do not spike in 2e4 events)."""
    full = [["local", "lateral"], ["local", "topdown"]]
    net = ref.PCLEventRef(seed=seed, learning=False, layer_inhibitions=full)
    net.w_s *= 5.0
    net.w_c *= 5.0
    return net


def test_d_all_inhibition_branches_pure_vs_numba(capsys):
    """Lateral, top-down and local inhibition all on, random seeded weights, learning off."""
    t, x, y, p = _shapes_events(20000)
    py, nbn, sec_py, sec_nb = _run_both(_scaled_all_branches, t, x, y, p)
    _assert_same_state(py, nbn)
    n_simple = sum(len(c.spikes) for c in py.s_cells)
    n_complex = sum(len(c.spikes) for c in py.c_cells)
    # Simple spikes drive the lateral and forward paths; complex spikes drive top-down (fixture check).
    assert n_simple > 0 and n_complex > 0, "fixture must spike in both layers"
    assert py.lat_on and py.td_on and py.local_s and py.local_c
    with capsys.disabled():
        print(f"\n[d] all branches, 2e4 events: pure {sec_py:.2f} s; numba {sec_nb:.3f} s; "
              f"simple {n_simple}, complex {n_complex}")


def test_e_numba_refuses_learning_on():
    net = ref.PCLEventRef(seed=0, learning=True)
    with pytest.raises(NotImplementedError):
        nb.run_numba(net, np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64),
                     np.zeros(1, np.int64))


def test_f_static_alias_reads_as_local():
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=[["static"], ["static"]])
    assert net.local_s and net.local_c and not net.lat_on and not net.td_on


# ----- (g) T-25: real trained PCL bytes through with_trained_weights gates -----
def test_g_authors_trained_bytes_refused_as_foreign_w0_with_native_positive_control():
    """T-25: the port has no NeuronalTensor bridge, so PCL trained weights are
    foreign to every jaxfne circuit. Real trained bytes must be refused by
    ``with_trained_weights`` (shape, then sign), while a native HDP run binds
    with provenance (positive control that the gate path works)."""
    import jaxfne as jtfne

    nt = jtfne.neuronal_tensor
    # Real trained bytes: float64, finite, nonnegative (PCL convention).
    raw = np.load(CSTDP3 / "weights" / "0" / "3.npy", allow_pickle=False)
    assert raw.dtype == np.float64, "fixture: trained file must be float64"
    assert np.all(np.isfinite(raw)) and np.all(raw >= 0), "fixture: trained bytes"
    tensor = nt.make_minimal_ei_tensor()
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=0.5))
    el = model.params["edge_list"]
    pre = np.asarray(el.pre).reshape(-1)
    post = np.asarray(el.post).reshape(-1)
    # The PCL file holds 200 weights; the minimal circuit compiles fewer.
    w_foreign = raw.reshape(-1)
    assert w_foreign.shape[0] != pre.shape[0], "fixture: foreign length must differ"
    with pytest.raises(ValueError, match="same length"):
        nt.with_trained_weights(
            tensor, {"pre": pre, "post": post, "w": w_foreign}, seed=0)
    # The same real bytes cut to edge count: nonnegative values on inhibitory
    # edges flip native signs, so they are refused as foreign, never coerced.
    rows = model.neuron_table()
    src_is_e = np.asarray([r["cell_type"] == "E" for r in rows], dtype=bool)
    assert not bool(np.all(src_is_e[pre])), "fixture must compile inhibitory edges"
    w_cut = np.asarray(np.resize(w_foreign, pre.shape[0]), dtype=np.float64)
    with pytest.raises(ValueError, match="sign"):
        nt.with_trained_weights(
            tensor, {"pre": pre, "post": post, "w": w_cut}, seed=0)
    # Positive control: a native HDP run's final weights bind with provenance.
    jtfne.simulate(
        model,
        runtime=jtfne.RuntimeConfig(
            enable_hdp=True,
            hdp_params=dict(K_HDP=0.2, K_ctrl=0.3, K_w_ctrl=0.002,
                            alpha=0.05, tau_0_ms=5.0, noise_scale=0.0),
        ),
        duration_ms=20.0, dt_ms=0.5, seed=7,
    )
    trained = nt.with_trained_weights(tensor, model)
    assert trained.w0_provenance.origin == "trained"
    assert (trained.w0_provenance.parent_digest
            == nt._tensor_identity_digest(tensor))
