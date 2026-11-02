"""R2 harness tests: the event oracle on a C++-layout network folder (port stage R2).

A synthetic 2x2x2 + 1x1x2 network with local-only inhibition, one 40 mV weight that
spikes on every event at pixel (1,1), and a 24-event camera-0 file. (a) runs the
oracle from two copies and compares exact, with the spike check on the run result;
(b) perturbs one weight by 1e-6 before the run and requires a nonzero diff with exact
False; (c) requires a missing count_spike to be refused; (i) requires the save to
mirror the C++ reset (count 0, empty train) and the (2, 1, 1, W, H) layer-0 shape.
Each run asserts its spikes first, so a silent fixture cannot pass.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ROOT = str(Path(__file__).resolve().parents[5])
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)
if str(HERE) not in sys.path:
    sys.path.insert(1, str(HERE))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pcl_event_ref as ref  # noqa: E402
import run_r2_reference as r2  # noqa: E402

N_EVENTS = 24
NB_PASS = 1


def _synthetic_network() -> dict:
    return {
        "nbCameras": 1,
        "neuron1Synapses": 1,
        "sharingType": "patch",
        "neuronType": ["SimpleCell", "ComplexCell"],
        "layerInhibitions": [["local"], ["local"]],
        "interLayerConnections": [[-1], [0]],
        "layerPatches": [[[0], [0], [0]], [[0], [0], [0]]],
        "layerSizes": [[2, 2, 2], [1, 1, 2]],
        "neuronSizes": [[[2, 2, 1]], [[2, 2, 2]]],
        "neuronOverlap": [[1, 1, 1], [0, 0, 0]],
        "neuronInhibitionRange": [1, 1],
        "vfWidth": 4,
        "vfHeight": 4,
        "measurementInterval": 100,
    }


def _make_base(root: Path) -> Path:
    """A loadable C++-layout folder whose depth-0 cell at (1,1) spikes on pixel (1,1)."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "network_config.json").write_text(
        json.dumps(_synthetic_network()), encoding="utf-8")
    (root / "configs" / "simple_cell_config.json").write_text(
        json.dumps(copy.deepcopy(ref.DEFAULT_SIMPLE)), encoding="utf-8")
    (root / "configs" / "complex_cell_config.json").write_text(
        json.dumps(copy.deepcopy(ref.DEFAULT_COMPLEX)), encoding="utf-8")
    net = ref.PCLEventRef(root, seed=0)
    net.w_s[:] = 0.0
    net.w_s[0, 1, 0, 0] = 40.0  # polarity 1 at offset (0,0): position (1,1), depth 0
    net.w_s[0, 0, 0, 0] = 10.0  # inert second mass: L1 normalisation then preserves ratios,
    # so a pre-run perturbation survives learning instead of normalising away
    net.w_c[:] = 0.0  # complex cells stay silent; the simple spike train carries the test
    r2.save_network(net, root)
    return root


def _make_events(path: Path) -> Path:
    t = 500000 + 1000 * np.arange(N_EVENTS, dtype=np.int64)
    with h5py.File(path, "w") as f:
        g = f.create_group("events")
        g.create_dataset("t", data=t)
        g.create_dataset("x", data=np.full(N_EVENTS, 1, dtype=np.int64))
        g.create_dataset("y", data=np.full(N_EVENTS, 1, dtype=np.int64))
        g.create_dataset("p", data=np.ones(N_EVENTS, dtype=np.int64))
        g.create_dataset("c", data=np.zeros(N_EVENTS, dtype=np.int64))
    return path


def test_a_two_oracle_runs_from_copies_compare_exact(tmp_path):
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    a = r2.copy_network(base, tmp_path / "a")
    b = r2.copy_network(base, tmp_path / "b")
    assert r2.main(["oracle", "--net", str(a), "--events", str(events),
                    "--nb-pass", str(NB_PASS), "--seed", "0",
                    "--out", str(tmp_path / "a_receipt.json")]) == 0
    rb = r2.run_oracle(b, events, NB_PASS, seed=0)
    assert rb["n_events"] == N_EVENTS and rb["n_spikes"] > 0, "fixture must spike"
    cmp = r2.compare(a, b)
    assert cmp["exact"] is True
    assert cmp["0"]["total_spikes_a"] == cmp["0"]["total_spikes_b"] == 0  # reset on save
    assert cmp["0"]["max_abs_weight_diff"] == 0.0
    assert cmp["1"]["exact"] is True  # silent complex layer agrees too


def test_b_pre_run_perturbation_breaks_exact(tmp_path):
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    c = r2.copy_network(base, tmp_path / "c")
    d = r2.copy_network(base, tmp_path / "d")
    w = d / "weights" / "0" / "0.npy"
    arr = np.load(w)
    arr[1, 0, 0] += 1e-6
    np.save(w, arr)
    rc = r2.run_oracle(c, events, NB_PASS, seed=0)
    rd = r2.run_oracle(d, events, NB_PASS, seed=0)
    assert rc["n_spikes"] > 0 and rd["n_spikes"] > 0, "fixture must spike"
    cmp = r2.compare(c, d)
    assert cmp["exact"] is False
    assert cmp["0"]["max_abs_weight_diff"] > 0.0


def test_c_missing_count_spike_refused(tmp_path):
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    e = r2.copy_network(base, tmp_path / "e")
    f = r2.copy_network(base, tmp_path / "f")
    re = r2.run_oracle(e, events, NB_PASS, seed=0)
    r2.run_oracle(f, events, NB_PASS, seed=0)
    assert re["n_spikes"] > 0, "fixture must spike"
    p = e / "weights" / "0" / "3.json"
    state = json.loads(p.read_text(encoding="utf-8"))
    del state["count_spike"]
    p.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="count_spike"):
        r2.compare(e, f)


def test_d_copy_refuses_existing_dst_and_nb_pass_refused(tmp_path):
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    dst = r2.copy_network(base, tmp_path / "g")
    with pytest.raises(FileExistsError):
        r2.copy_network(base, dst)
    with pytest.raises(ValueError, match="nb_pass"):
        r2.run_oracle(dst, events, 0, seed=0)


def test_f_nb_pass_feeds_the_file_again_shifted(tmp_path):
    """nb_pass is the C++ nbPass: pass k repeats the file shifted by k * (last - first)."""
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    one = r2.copy_network(base, tmp_path / "one")
    two = r2.copy_network(base, tmp_path / "two")
    r1 = r2.run_oracle(one, events, 1, seed=0)
    r2_ = r2.run_oracle(two, events, 2, seed=0)
    assert r1["n_spikes"] > 0, "fixture must spike"
    assert r2_["n_events"] == 2 * r1["n_events"] == 2 * N_EVENTS
    span = 1000 * (N_EVENTS - 1)
    assert r2_["last_spike_t"] > span, "the second pass must spike after the first pass's span"


def test_q_nb_pass_shift_spans_the_raw_file_every_camera(tmp_path):
    """The pass shift is the raw file's span, camera-1 edges included (NetworkHandle.cpp:479-516).

    Camera-1 events 5 ms before the first and 50 ms after the last camera-0 event: the raw
    span is 78 ms, the camera-0 span 23 ms. Pass 2's last camera-0 event lands at
    28 ms + 78 ms; a camera-0 span would put every pass-2 spike at or before 28 + 23 ms.
    """
    t0 = 500000 + 1000 * np.arange(N_EVENTS, dtype=np.int64)
    t = np.concatenate([[t0[0] - 5000], t0, [t0[-1] + 50000]])
    c = np.concatenate([[1], np.zeros(N_EVENTS, dtype=np.int64), [1]])
    path = tmp_path / "ev_cam1.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("events")
        g.create_dataset("t", data=t)
        g.create_dataset("x", data=np.full(t.shape[0], 1, dtype=np.int64))
        g.create_dataset("y", data=np.full(t.shape[0], 1, dtype=np.int64))
        g.create_dataset("p", data=np.ones(t.shape[0], dtype=np.int64))
        g.create_dataset("c", data=c)
    net = r2.copy_network(_make_base(tmp_path / "base"), tmp_path / "two")
    res = r2.run_oracle(net, path, 2, seed=0)
    assert res["n_events"] == 2 * N_EVENTS and res["n_spikes"] > 0, "fixture must spike"
    assert 28000 + 23000 < res["last_spike_t"] <= 28000 + 78000


def test_h_init_writes_a_loadable_seeded_folder(tmp_path):
    base = _make_base(tmp_path / "base")
    a = r2.init_network(base / "configs", tmp_path / "ia", seed=3)
    b = r2.init_network(base / "configs", tmp_path / "ib", seed=3)
    c = r2.init_network(base / "configs", tmp_path / "ic", seed=4)
    assert (a / "weights" / "0" / "0.npy").is_file(), "the C++ loads weights only when 0.npy exists"
    assert r2.compare(a, b)["exact"] is True
    assert r2.compare(a, c)["0"]["max_abs_weight_diff"] > 0.0, "the seed must change the init"
    with pytest.raises(FileExistsError):
        r2.init_network(base / "configs", a, seed=3)


def test_e_compare_cli_writes_receipt(tmp_path):
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    a = r2.copy_network(base, tmp_path / "a")
    b = r2.copy_network(base, tmp_path / "b")
    r2.run_oracle(a, events, NB_PASS, seed=0)
    r2.run_oracle(b, events, NB_PASS, seed=0)
    out = tmp_path / "receipt.json"
    assert r2.main(["compare", "--a", str(a), "--b", str(b), "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["tool"] == "run_r2_reference" and rec["subcommand"] == "compare"
    assert len(rec["code_sha"]) == 40
    assert len(rec["inputs"]["a"]["sha256"]) == 64
    assert rec["inputs"]["b"]["n_files"] == rec["inputs"]["a"]["n_files"] > 0
    assert rec["comparison"]["exact"] is True


def test_i_save_mirrors_cpp_reset_and_shape(tmp_path):
    """The save mirrors the C++ reset: (2, 1, 1, W, H) layer-0 weights, count 0, train []."""
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    net = r2.copy_network(base, tmp_path / "net")
    res = r2.run_oracle(net, events, NB_PASS, seed=0)
    assert res["n_spikes"] > 0, "fixture must spike"
    for k in range(2):  # ND of the synthetic network; SX = SY = 2
        arr = np.load(net / "weights" / "0" / f"{k}.npy")
        assert arr.shape == (2, 1, 1, 2, 2), f"layer-0 weight {k}: {arr.shape}"
    for layer, n in (("0", 8), ("1", 2)):  # NX*NY*ND and CX*CY*CD of the fixture
        for idx in range(n):
            state = json.loads((net / "weights" / layer / f"{idx}.json").read_text(encoding="utf-8"))
            assert state["count_spike"] == 0, f"{layer}/{idx}.json count"
            assert state["spike_train"] == [], f"{layer}/{idx}.json train"


def test_j_exact_is_weights_only_with_counts_reset(tmp_path):
    """`exact` decides on weights only; counts are reported, never gating."""
    base = _make_base(tmp_path / "base")
    a = r2.copy_network(base, tmp_path / "a")
    b = r2.copy_network(base, tmp_path / "b")
    for side in (a, b):
        p = side / "weights" / "0" / "3.json"
        state = json.loads(p.read_text(encoding="utf-8"))
        state["count_spike"] = 5  # nonzero on both sides: the old count leg
        p.write_text(json.dumps(state), encoding="utf-8")
    cmp = r2.compare(a, b)
    assert cmp["exact"] is True, "equal weights compare exact with nonzero counts"
    assert cmp["0"]["total_spikes_a"] == cmp["0"]["total_spikes_b"] == 5
    assert cmp["0"]["counts_reset"] is False, "nonzero counts are not a reset"
    assert cmp["counts_reset"] is False
    cmp0 = r2.compare(a, a)
    assert cmp0["counts_reset"] is False


def test_k_saved_folders_compare_exact_with_counts_reset(tmp_path):
    """Saved folders (counts reset to 0) compare exact and report counts_reset true."""
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    a = r2.copy_network(base, tmp_path / "a")
    b = r2.copy_network(base, tmp_path / "b")
    ra = r2.run_oracle(a, events, NB_PASS, seed=0)
    r2.run_oracle(b, events, NB_PASS, seed=0)
    assert ra["n_spikes"] > 0, "fixture must spike"
    cmp = r2.compare(a, b)
    assert cmp["exact"] is True
    assert cmp["counts_reset"] is True, "saved folders carry the C++ reset"


def test_l_oracle_cli_writes_receipt(tmp_path):
    """The oracle subcommand writes a receipt carrying the run result."""
    base = _make_base(tmp_path / "base")
    events = _make_events(tmp_path / "ev.h5")
    net = r2.copy_network(base, tmp_path / "net")
    out = tmp_path / "receipt.json"
    assert r2.main(["oracle", "--net", str(net), "--events", str(events),
                    "--nb-pass", str(NB_PASS), "--seed", "0",
                    "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["tool"] == "run_r2_reference" and rec["subcommand"] == "oracle"
    assert len(rec["code_sha"]) == 40
    assert len(rec["inputs"]["events"]["sha256"]) == 64
    assert len(rec["inputs"]["net"]["sha256"]) == 64
    assert rec["inputs"]["net"]["n_files"] > 0
    assert rec["result"]["n_spikes"] > 0, "receipt must hold the run's spikes"
    assert rec["result"]["n_events"] == N_EVENTS


def _make_npz_events(path: Path, *, n: int = N_EVENTS, x: int = 1, y: int = 1,
                     spacing: int = 1000, t0: int = 500000, cam1: tuple = (),
                     pol: int = 1) -> Path:
    """npz with the training key layout and dtypes (int64/int16/int16/int8/int8)."""
    t = (t0 + spacing * np.arange(n, dtype=np.int64)).astype(np.int64)
    c = np.zeros(n, dtype=np.int8)
    for i in cam1:
        c[i] = 1
    np.savez(path,
             arr_0=t,
             arr_1=np.full(n, x, dtype=np.int16),
             arr_2=np.full(n, y, dtype=np.int16),
             arr_3=np.full(n, pol, dtype=np.int8),
             arr_4=c)
    return path


def test_m_npz_loader_round_trips_layout_and_refuses(tmp_path):
    """The npz loader maps arr_0..arr_4 to the oracle's arrays; wrong keys/dtypes are refused."""
    p = _make_npz_events(tmp_path / "ev.npz", n=10, cam1=(2, 7))
    ev = r2.load_events(p)
    assert ev["t"].tolist() == [1000 * k for k in (0, 1, 3, 4, 5, 6, 8, 9)]
    assert ev["t"][0] == 0, "rebased to the first event, as the C++ feeds"
    assert ev["x"].tolist() == [1] * 8 and ev["y"].tolist() == [1] * 8
    assert ev["p"].tolist() == [1] * 8 and ev["c"].tolist() == [0] * 8
    bad = tmp_path / "bad_keys.npz"
    np.savez(bad, arr_0=np.zeros(4, dtype=np.int64), arr_1=np.zeros(4, dtype=np.int16))
    with pytest.raises(ValueError, match="keys"):
        r2.load_events(bad)
    wrong = tmp_path / "wrong_dtype.npz"
    np.savez(wrong,
             arr_0=np.zeros(4, dtype=np.float64),
             arr_1=np.zeros(4, dtype=np.int16), arr_2=np.zeros(4, dtype=np.int16),
             arr_3=np.zeros(4, dtype=np.int8), arr_4=np.zeros(4, dtype=np.int8))
    with pytest.raises(ValueError, match="arr_0"):
        r2.load_events(wrong)


def _chain_pair(tmp_path):
    base = _make_base(tmp_path / "base")
    a = _make_npz_events(tmp_path / "a.npz", n=N_EVENTS)
    b = _make_npz_events(tmp_path / "b.npz", n=30, spacing=777, t0=900000)
    return base, a, b


def test_n_chain_equals_continuous_with_reset(tmp_path):
    """chain over [A, B] equals one oracle run over A, the reset map, then B.

    The continuous arm saves after A and reloads (the C++ handoff: learned weights with
    reset neuron state), then runs B. Weights agree bitwise float64; per-file spike counts
    agree with the chain receipts.
    """
    base, a, b = _chain_pair(tmp_path)
    assert r2.main(["chain", "--net", str(base), "--events", str(a), str(b),
                    "--arm", "oracle", "--out", str(tmp_path / "out"),
                    "--seed", "0"]) == 0
    recs = [json.loads((tmp_path / "out" / f"{k}_{s}.json").read_text(encoding="utf-8"))
            for k, s in ((0, "a"), (1, "b"))]
    assert [r["arm"] for r in recs] == ["oracle", "oracle"]
    assert len(recs[0]["code_sha"]) == 40 and len(recs[0]["events"]["sha256"]) == 64
    assert recs[0]["n_events"] == N_EVENTS and recs[1]["n_events"] == 30
    cont = ref.PCLEventRef(base, seed=0, learning=True)
    per = []
    for ep in (a, b):
        ev = r2.load_training_npz(ep)
        before = (sum(len(c.spikes) for c in cont.s_cells),
                  sum(len(c.spikes) for c in cont.c_cells))
        cont.run(ev["t"], ev["x"], ev["y"], ev["p"])
        after = (sum(len(c.spikes) for c in cont.s_cells),
                 sum(len(c.spikes) for c in cont.c_cells))
        per.append((after[0] - before[0], after[1] - before[1]))
        r2.save_network(cont, tmp_path / "s")
        if not (tmp_path / "s" / "configs").is_dir():
            shutil.copytree(base / "configs", tmp_path / "s" / "configs")
        cont = ref.PCLEventRef(tmp_path / "s", seed=0, learning=True)
    assert per[0][0] > 0 and per[1][0] > 0, "both files must spike"
    for rec, (ns, nc) in zip(recs, per):
        assert (rec["spikes_simple"], rec["spikes_complex"]) == (ns, nc)
    cmp = r2.compare(tmp_path / "out" / "1_b", tmp_path / "s")
    assert cmp["exact"] is True
    assert cmp["0"]["max_abs_weight_diff"] == 0.0, "bitwise float64"
    assert cmp["1"]["max_abs_weight_diff"] == 0.0, "bitwise float64"


def test_o_chain_without_reset_differs(tmp_path):
    """Negative control: carrying the whole state across the boundary changes file B's spikes.

    Both arms feed B on the same continued clock; only the reset differs. Precondition: V or
    the threshold is nonzero at the boundary, so the check can fail.
    """
    base, a, _ = _chain_pair(tmp_path)
    b = _make_npz_events(tmp_path / "b5.npz", n=5, spacing=777, t0=900000)
    ea, eb = r2.load_training_npz(a), r2.load_training_npz(b)
    off = int(ea["t"][-1]) + 1000
    tb = eb["t"] + off
    reset_arm, kept_arm = (ref.PCLEventRef(base, seed=0, learning=True) for _ in range(2))
    for net in (reset_arm, kept_arm):
        net.run(ea["t"], ea["x"], ea["y"], ea["p"])
    assert any(cell.V != 0.0 or cell.thr != float(ref.DEFAULT_SIMPLE["VTHRESH"])
               for cell in reset_arm.s_cells), "V or the threshold must be nonzero at the boundary"
    r2._reset_neurons(reset_arm)
    reset_arm._last_update = kept_arm._last_update = off  # same clock: only the reset differs
    before_r = (sum(len(c.spikes) for c in reset_arm.s_cells),
                sum(len(c.spikes) for c in reset_arm.c_cells))
    before_k = (sum(len(c.spikes) for c in kept_arm.s_cells),
                sum(len(c.spikes) for c in kept_arm.c_cells))
    reset_arm.run(tb, eb["x"], eb["y"], eb["p"])
    kept_arm.run(tb, eb["x"], eb["y"], eb["p"])
    got_r = (sum(len(c.spikes) for c in reset_arm.s_cells) - before_r[0],
             sum(len(c.spikes) for c in reset_arm.c_cells) - before_r[1])
    got_k = (sum(len(c.spikes) for c in kept_arm.s_cells) - before_k[0],
             sum(len(c.spikes) for c in kept_arm.c_cells) - before_k[1])
    assert got_r != got_k, "the chain without the reset must differ on file B"


def _make_lat_base(root: Path) -> Path:
    """A loadable folder with lateral and top-down inhibition and seeded init weights.

    Dense seeded weights: test_p requires the scan replay to match the oracle within rtol
    1e-12 in weights and exactly in spikes. Hand-zeroed sparse weights were reported to
    leave a gap of about 0.4% in w_s with identical spikes; that gap is the scan kernel's,
    not the chain's (issue P-053). Lateral and top-down on means the chain's scan-layout
    scatter runs.
    """
    net = _synthetic_network()
    net["layerInhibitions"] = [["local", "lateral"], ["local", "topdown"]]
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "network_config.json").write_text(
        json.dumps(net), encoding="utf-8")
    (root / "configs" / "simple_cell_config.json").write_text(
        json.dumps(copy.deepcopy(ref.DEFAULT_SIMPLE)), encoding="utf-8")
    (root / "configs" / "complex_cell_config.json").write_text(
        json.dumps(copy.deepcopy(ref.DEFAULT_COMPLEX)), encoding="utf-8")
    r2.save_network(ref.PCLEventRef(root, seed=0), root)
    return root


def test_p_core_arm_equals_oracle_arm_on_chain(tmp_path):
    """The core arm chained through the PCL boundary map equals the oracle arm's chain.

    Weights at rtol 1e-12, per-file spike counts exact, on a drive whose kernel replay
    matches the oracle (see _make_lat_base). Seeded init weights move under learning,
    so the comparison is not vacuous.
    """
    pytest.importorskip("numba", reason="the core arm loads run_tuning, which needs numba (not a jaxfne dependency)")
    base = _make_lat_base(tmp_path / "base")
    a = _make_npz_events(tmp_path / "a.npz", n=60, spacing=100)
    b = _make_npz_events(tmp_path / "b.npz", n=40, spacing=137, t0=900000)
    ro = r2.chain(base, [a, b], arm="oracle", out_dir=tmp_path / "out_o", seed=0)
    rc = r2.chain(base, [a, b], arm="core", out_dir=tmp_path / "out_c", seed=0)
    for ko, kc in zip(ro, rc):
        assert (ko["spikes_simple"], ko["spikes_complex"]) == \
            (kc["spikes_simple"], kc["spikes_complex"])
        assert ko["n_events"] == kc["n_events"] > 0, "both files must feed"
        assert ko["last_spike_t"] == kc["last_spike_t"], "both arms on the raw file clock"
    assert ro[0]["spikes_simple"] > 0 and ro[1]["spikes_simple"] > 0, "fixture must spike"
    w0 = np.load(tmp_path / "base" / "weights" / "0" / "0.npy")
    w1 = np.load(tmp_path / "out_o" / "1_b" / "weights" / "0" / "0.npy")
    assert not np.array_equal(w0, w1), "fixture must move w_s under learning"
    cmp = r2.compare(tmp_path / "out_o" / "1_b", tmp_path / "out_c" / "1_b")
    assert cmp["exact"] is True
