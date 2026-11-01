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
import sys
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
