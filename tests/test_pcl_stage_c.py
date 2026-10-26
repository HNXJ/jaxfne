"""Stage C gate: core PCL tensor at frozen weights vs the event oracle (S1) on 20 clips.

Acceptance (packet): per-cell spike-count correlation >= 0.95 and total within
5 %, at a recorded dt. Clip set: test7 ori=0, sf 0..9, phase 0, trials 0/1
(20 clips; inputs and S1 oracle outputs verified present on disk 2026-10-14).

Status: the gate PASSES on the tensor route (CALIBRATION.md Step 9):
`build_s1_scan_tensor` + `EventScanDrive` through
`construct -> Model -> simulate` reproduces the S1 oracle per-cell exact on
all 20 clips (committed receipt
`artifacts/etudes/papers/pcl/stage_c_tensor_route_20clip.json`).
`test_stage_c_tensor_route_matches_oracle_on_slice` replays the first 300
cam0 events of clip (0,0) live through the tensor route and checks the
binned spike matrix against the oracle spike times; the receipt test checks
the committed file against the pins below. The oracle record pins the
comparison target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import jaxfne as jtfne

def _warehouse_roots(*suffix: str) -> Path:
    """First existing warehouse spelling (Windows `F:/` vs WSL `/mnt/f`)."""
    cands = [Path("F:/warehouse/pcl", *suffix), Path("/mnt/f/warehouse/pcl", *suffix)]
    for cand in cands:
        if cand.exists():
            return cand
    return cands[0]


IN_ROOT = _warehouse_roots("data", "PCL_datasets_and_networks/grating/"
                           "counterphase_gratingComplex/test7")
S1_ROOT = _warehouse_roots("runs", "fig3_cSTDP3_S1")
NETWORK = _warehouse_roots("data", "PCL_datasets_and_networks/"
                           "add_simulations_cSTDP/cSTDP3")
RECEIPT = (Path(__file__).resolve().parents[1] / "artifacts" / "etudes"
           / "papers" / "pcl" / "stage_c_tensor_route_20clip.json")

CLIPS = [(f, s) for f in range(10) for s in (0, 1)]  # ori 0, sf 0..9, phase 0

# Oracle S1 outputs on the 20 clips, measured 2026-10-14 (read-only probe):
# ~830k camera-0 events and ~1.82 s per clip.
ORACLE_SIMPLE_TOTAL = 418042
ORACLE_COMPLEX_TOTAL = 365087
N_SIMPLE, N_COMPLEX = 5184, 1152

GATE_R = 0.95
GATE_TOTAL = 0.05

pytestmark = pytest.mark.skipif(
    not (IN_ROOT / "0" / "0" / "0" / "0.npz").is_file()
    or not (S1_ROOT / "0" / "0" / "0" / "0.npz").is_file()
    or not (NETWORK / "configs" / "network_config.json").is_file()
    or not (NETWORK / "weights" / "0" / "0.npy").is_file(),
    reason="Stage C needs the warehouse test7 inputs, S1 oracle outputs "
    "and the frozen cSTDP3 network",
)


def _clip_counts(f: int, s: int) -> tuple[np.ndarray, np.ndarray, int]:
    out = np.load(S1_ROOT / "0" / str(f) / "0" / f"{s}.npz", allow_pickle=False)
    sc = out["simple_counts"].astype(np.int64)
    cc = out["complex_counts"].astype(np.int64)
    assert sc.shape == (N_SIMPLE,) and cc.shape == (N_COMPLEX,)
    return sc, cc, int(out["n_events"])


@pytest.mark.data_gated
def test_stage_c_oracle_20clip_record():
    """Pin the S1 oracle side of the Stage C comparison (frozen cSTDP3, learning off)."""
    tot_s = tot_c = 0
    for f, s in CLIPS:
        sc, cc, n_events = _clip_counts(f, s)
        with np.load(IN_ROOT / "0" / str(f) / "0" / f"{s}.npz",
                     allow_pickle=False) as z:
            keep = z["arr_4"] == 0
        assert n_events == int(keep.sum()), f"clip {(f, s)}: oracle fed another event set"
        assert int(sc.sum()) > 0 and int(cc.sum()) > 0, f"clip {(f, s)} must spike"
        tot_s += int(sc.sum())
        tot_c += int(cc.sum())
    assert tot_s == ORACLE_SIMPLE_TOTAL, f"simple total moved: {tot_s}"
    assert tot_c == ORACLE_COMPLEX_TOTAL, f"complex total moved: {tot_c}"


DT_MS = 0.5


def _trained_scan_case(n_events: int = 300):
    """Tensor-route scan drive for the first ``n_events`` cam0 events of clip (0,0).

    Same transcription as the 20-clip gate scripts: trained cSTDP3 weights
    read straight from the warehouse network (the port's
    ``PCLEventRef._load_weights`` layout, read here with numpy only, no
    oracle import), geometry from ``network_config.json`` with the port's
    offset/cover formulas, scan wiring on the ``EventScanDrive`` paradigm
    object, positions + resolution on the tensor. Lateral/top-down maps run
    as zeros, which is the frozen S1 run's effective setting (per-cell exact
    on all 20 clips); the config's ``layerInhibitions`` is pinned so a
    network swap fails loudly instead of silently.
    """
    from jaxfne.pcl_topology import (
        EventScanDrive,
        build_s1_scan_tensor,
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
        scan_inhibition_triplets,
    )

    with open(NETWORK / "configs" / "network_config.json",
              encoding="utf-8") as fh:
        net = json.load(fh)
    assert net["layerInhibitions"] == [["static", "lateral", "topdown"],
                                       ["static", "topdown"]], (
        "frozen cSTDP3 inhibition setting moved; the zero-map scan "
        "transcription below is only valid for it")
    NX, NY, ND = (int(v) for v in net["layerSizes"][0])
    CX, CY, CD = (int(v) for v in net["layerSizes"][1])
    SX, SY = (int(v) for v in net["neuronSizes"][0][0][:2])
    CSX, CSY = (int(v) for v in net["neuronSizes"][1][0][:2])
    s_ov = [int(v) for v in net["neuronOverlap"][0]]
    c_ov = [int(v) for v in net["neuronOverlap"][1]]
    p0, p1 = net["layerPatches"][0], net["layerPatches"][1]
    s_ox = [int(p0[0][0]) + i * (SX - s_ov[0]) for i in range(NX)]
    s_oy = [int(p0[1][0]) + j * (SY - s_ov[1]) for j in range(NY)]
    c_ox = [int(p1[0][0]) + i * (CSX - c_ov[0]) for i in range(CX)]
    c_oy = [int(p1[1][0]) + j * (CSY - c_ov[1]) for j in range(CY)]
    vw, vh = int(net["vfWidth"]), int(net["vfHeight"])
    assert [int(v) for v in net["neuronInhibitionRange"]] == [4, 4]
    # Complex cells covering simple position (i, j), ascending complex index.
    c_cover = {
        (i, j): [(cx, cy) for cx in range(CX) for cy in range(CY)
                 if c_ox[cx] <= i < c_ox[cx] + CSX
                 and c_oy[cy] <= j < c_oy[cy] + CSY]
        for i in range(NX) for j in range(NY)
    }
    NS, NC = NX * NY * ND, CX * CY * CD

    def _w(path: Path, size: int) -> np.ndarray:
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.float64 or arr.size != size:
            raise ValueError(
                f"{path}: need float64 with {size} values, got "
                f"{arr.dtype} {arr.shape}")
        return arr.reshape(-1)

    d0 = NETWORK / "weights" / "0"
    w_s = np.stack([_w(d0 / f"{k}.npy", 2 * SX * SY).reshape(2, SX, SY)
                    for k in range(ND)])
    wi_s = np.stack([_w(d0 / f"{k}lli.npy", ND) for k in range(ND)])
    d1 = NETWORK / "weights" / "1"
    w_c = np.zeros((CD, CSX, CSY, ND))
    wi_c = np.zeros((CD, CD))
    seen = set()
    for c in range(NC):
        cz = c % CD
        w = _w(d1 / f"{c}.npy", CSX * CSY * ND).reshape(CSX, CSY, ND)
        wi = _w(d1 / f"{c}lli.npy", CD)
        if cz not in seen:
            seen.add(cz)
            w_c[cz], wi_c[cz] = w, wi
        else:
            assert np.array_equal(w_c[cz], w) and np.array_equal(wi_c[cz], wi), (
                f"complex weight files of depth {cz} disagree (cell {c})")
    tables = event_scan_tables(
        nx=NX, ny=NY, nd=ND, cx=CX, cy=CY, cd=CD,
        origins_x=[float(v) for v in s_ox],
        origins_y=[float(v) for v in s_oy],
        size=(SX, SY), weights=w_s, c_cover=c_cover,
        c_ox=[int(v) for v in c_ox], c_oy=[int(v) for v in c_oy],
        c_size=(CSX, CSY), c_weights=w_c, vw=vw, vh=vh,
        inhib_range=(4, 4),
        lateral_on=True, lateral_weights=np.zeros((NS, NS)),
        topdown_on=True, topdown_weights=np.zeros((NC, NS)),
        activation=True)
    groups = np.zeros(NS + NC, dtype=np.int64)
    for i in range(NX):
        for j in range(NY):
            groups[(i * NY + j) * ND:(i * NY + j + 1) * ND] = i * NY + j
    for cx in range(CX):
        for cy in range(CY):
            groups[NS + (cx * CY + cy) * CD:NS + (cx * CY + cy + 1) * CD] = (
                NX * NY + cx * CY + cy)
    pre, post, mag = scan_inhibition_triplets(
        NX, NY, ND, CX, CY, CD, NS, wi_s, wi_c)
    blocks = local_inhibition_blocks(groups, li_pre=pre, li_post=post,
                                     li_weight=mag)

    z = np.load(IN_ROOT / "0" / "0" / "0" / "0.npz", allow_pickle=False)
    t_raw = z["arr_0"].astype(np.int64)
    keep = z["arr_4"] == 0
    t_raw = t_raw[keep]
    assert bool(np.all(np.diff(t_raw) >= 0)), "clip file must be time-ordered"
    x = z["arr_1"].astype(np.int64)[keep]
    y = z["arr_2"].astype(np.int64)[keep]
    p = z["arr_3"].astype(np.int64)[keep]
    t0 = int(t_raw[0])
    t = (t_raw - t0)[:n_events]
    x, y, p = x[:n_events], y[:n_events], p[:n_events]
    t_last = int(t.max())
    n_steps = int(t_last // 500) + 1 + 3
    events = event_scan_events(t, x, y, p, dt_ms=DT_MS, n_steps=n_steps,
                               vw=vw, vh=vh)
    drive = EventScanDrive(events, tables, blocks, groups,
                           dt_ms=DT_MS, n_steps=n_steps)
    return drive, build_s1_scan_tensor(ns=NS, nc=NC), n_steps, t0, t_last


def _causal_step_mask(n_steps: int, t_last: int) -> np.ndarray:
    """Steps whose whole 500 us bin ends at or before the last sliced event.

    An oracle spike inside such a bin depends only on events strictly
    before ``t_last``, which the time-ordered slice contains in full; later
    bins may need events past the slice and are excluded on both sides.
    """
    return ((np.arange(n_steps) + 1) * 500 <= t_last)


def _oracle_slice_matrix(n_steps: int, t0: int, t_last: int) -> np.ndarray:
    """Bin the S1 oracle spike times of clip (0,0) onto the slice grid."""
    out = np.load(S1_ROOT / "0" / "0" / "0" / "0.npz", allow_pickle=False)
    mat = np.zeros((n_steps, N_SIMPLE + N_COMPLEX), dtype=bool)
    for base, kc, kt in ((0, "simple_times_cell", "simple_times_us"),
                         (N_SIMPLE, "complex_times_cell",
                          "complex_times_us")):
        cells = out[kc].astype(np.int64)
        ts = out[kt].astype(np.int64) - t0
        m = (ts >= 0) & (ts < n_steps * 500)
        mat[(ts[m] // 500), base + cells[m]] = True
    mat[~_causal_step_mask(n_steps, t_last)] = False
    return mat


@pytest.mark.data_gated
def test_stage_c_tensor_route_matches_oracle_on_slice():
    """Step 9: the tensor route replays clip (0,0)'s first 300 events exactly.

    ``NeuronalTensor -> construct -> Model -> simulate`` with the scan
    x-source paradigm must equal the S1 oracle spike-for-spike (binned on
    the 500 us grid) on this suite-feasible slice; ``V_m`` stays all-NaN.
    """
    import jaxfne.neuronal_tensor as nt

    drive, tensor, n_steps, t0, t_last = _trained_scan_case()
    model = jtfne.construct(
        tensor, nt.RuntimeConfiguration(
            seed=0, duration_ms=n_steps * DT_MS, dt_ms=DT_MS,
            emitter="lif"))
    signals = model.simulate(
        jtfne.Simulation(
            duration_ms=n_steps * DT_MS, dt_ms=DT_MS, seed=1,
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
        paradigm=drive)
    got = np.asarray(signals.spikes)
    mask = _causal_step_mask(n_steps, t_last)
    assert int(got[mask].sum()) > 0, "fixture must spike"
    assert np.array_equal(
        got[mask], _oracle_slice_matrix(n_steps, t0, t_last)[mask])
    assert bool(np.isnan(np.asarray(signals.V_m)).all()), "V_m must be NaN"


@pytest.mark.data_gated
def test_stage_c_tensor_route_20clip_receipt():
    """Step 9: the committed 20-clip receipt matches the pins and the gate."""
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["resolution"] == "event_scan"
    assert receipt["dt_ms"] == DT_MS
    assert receipt["oracle"] == {"simple_total": ORACLE_SIMPLE_TOTAL,
                                 "complex_total": ORACLE_COMPLEX_TOTAL}
    assert receipt["tensor"] == receipt["oracle"], "tensor totals moved"
    assert receipt["verdict"] == "PASS"
    clips = receipt["clips"]
    assert len(clips) == len(CLIPS) == 20
    for row, (f, s) in zip(clips, CLIPS):
        assert (row["f"], row["s"]) == (f, s)
        assert row["tensor_simple"] == row["oracle_simple"]
        assert row["tensor_complex"] == row["oracle_complex"]
        assert row["exact_cells"] == row["n_cells"] == N_SIMPLE + N_COMPLEX
        assert row["r_simple"] >= GATE_R
        assert row["r_complex"] >= GATE_R
        assert row["r_both"] >= GATE_R
    assert receipt["first_spike_us_exact"]["co_spiking"][0] == \
        receipt["first_spike_us_exact"]["co_spiking"][1] > 0
    assert receipt["first_spike_us_exact"]["silent_agree"][0] == \
        receipt["first_spike_us_exact"]["silent_agree"][1]
