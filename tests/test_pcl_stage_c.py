"""Stage C gate: core PCL tensor at frozen weights vs the event oracle (S1) on 20 clips.

Acceptance (packet): per-cell spike-count correlation >= 0.95 and total within
5 %, at a recorded dt. Clip set: test7 ori=0, sf 0..9, phase 0, trials 0/1
(20 clips; inputs and S1 oracle outputs verified present on disk 2026-10-14).

Status: the gate is recorded but not passed. The full-network core cascade has
no mapping onto the lif edge kernel (CORE_GAP in
artifacts/etudes/pcl/port/run_tuning.py:86-94; CALIBRATION.md step-6 outcome):
the lif route refuses paradigm drive (no DVS feed path), the DVS cover-table
feed and the complex-layer pooling have no core expression, and the executed
local-inhibition cap is whole-population while the oracle cascade is
per-position depth-sequential. The oracle record below pins the comparison
target; the core half of the gate is xfail(strict) until the mapping lands.
"""

from __future__ import annotations

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
    or not (S1_ROOT / "0" / "0" / "0" / "0.npz").is_file(),
    reason="Stage C needs the warehouse test7 inputs and S1 oracle outputs",
)


def _clip_counts(f: int, s: int) -> tuple[np.ndarray, np.ndarray, int]:
    out = np.load(S1_ROOT / "0" / str(f) / "0" / f"{s}.npz", allow_pickle=False)
    sc = out["simple_counts"].astype(np.int64)
    cc = out["complex_counts"].astype(np.int64)
    assert sc.shape == (N_SIMPLE,) and cc.shape == (N_COMPLEX,)
    return sc, cc, int(out["n_events"])


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


@pytest.mark.xfail(strict=True, reason=(
    "CORE_GAP: lif route refuses paradigm drive, so DVS clips cannot feed a "
    "core tensor; cover-table feed and complex pooling have no core mapping"))
def test_stage_c_core_tensor_matches_oracle_on_20_clips():
    """Acceptance shape: frozen-weights lif tensor vs oracle, r >= 0.95, total within 5 %.

    First runnable step today: a DVS-fed core run needs paradigm drive on the
    lif route, which is refused by name. The refusal below is the observed
    blocker; the correlation/total comparison runs once the mapping lands.
    """
    import jaxfne.neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="stage-c-probe",
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L4", n_neurons=4,
            neuron_types=[nt.NeuronType.make("E")])])])
    model = jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, duration_ms=2.0, dt_ms=0.5, emitter="lif"))
    schedule = jtfne.stimulus_schedule(
        [{"onset_ms": 0.0, "duration_ms": 2.0, "amplitude": 1.0}],
        n_neurons=4)
    signals = model.simulate(
        jtfne.Simulation(duration_ms=2.0, dt_ms=0.5, seed=1), paradigm=schedule)
    assert signals is not None  # unreachable: paradigm is refused on lif
