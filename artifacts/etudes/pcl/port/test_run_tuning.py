"""S1 switch for the Fig. 3 run (P3_fig3.md Sensitivity S1): --inhibition {full,local-only}.

Fast synthetic-network tests: no trained weights, no test7 files. The synthetic net uses
the default layout with lateral and top-down inhibition enabled, so LAT/TD are nonzero
random maps, exactly what local-only must zero.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pcl_event_ref as ref  # noqa: E402
import pcl_event_ref_numba as nb  # noqa: E402
import run_tuning as rt  # noqa: E402

INH = [["local", "lateral"], ["local", "topdown"]]
# Reference from the pre-change code (seed 0, 400-event burst below): full-path counts.
REF_SIMPLE_N, REF_COMPLEX_N = 7, 28
REF_SIMPLE_T_SUM, REF_COMPLEX_T_SUM = 30260, 103780


def _burst() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = 400
    t = (np.arange(n, dtype=np.int64) + 1) * 20  # 20 us spacing at one pixel
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    return t, x, y, p


def _counts(net) -> tuple[int, int, int, int]:
    s = sum(len(c.spikes) for c in net.s_cells)
    c = sum(len(c.spikes) for c in net.c_cells)
    st = sum(s for cell in net.s_cells for s in cell.spikes)
    ct = sum(s for cell in net.c_cells for s in cell.spikes)
    return s, c, st, ct


def test_local_only_zeroes_what_run_numba_consumes():
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    assert net.lat_on and net.td_on
    assert float(np.abs(net.LAT).sum()) > 0.0
    assert float(np.abs(net.TD).sum()) > 0.0
    full = copy.deepcopy(net)
    wi_before = net.wi_s.copy()

    rt.apply_inhibition_mode(net, "local-only")

    # The arrays run_numba passes to the kernel (pcl_event_ref_numba.py:258-267).
    lat_kernel = np.ascontiguousarray(net.LAT, dtype=np.float64)
    td_kernel = np.ascontiguousarray(net.TD.T, dtype=np.float64)
    assert np.all(lat_kernel == 0.0)
    assert np.all(td_kernel == 0.0)
    # Flags, masks and local weights are untouched.
    assert net.lat_on and net.td_on
    assert bool((net.lat_mask != 0).any()) and bool((net.td_mask != 0).any())
    assert np.array_equal(net.wi_s, wi_before)

    # Behavioural: the zeroed maps change the run on a short event slice.
    t, x, y, p = _burst()
    nb.run_numba(full, t, x, y, p)
    nb.run_numba(net, t, x, y, p)
    assert _counts(net) != _counts(full)


def test_full_path_unchanged():
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    rt.apply_inhibition_mode(net, "full")
    nb.run_numba(net, *_burst())
    assert _counts(net) == (REF_SIMPLE_N, REF_COMPLEX_N, REF_SIMPLE_T_SUM, REF_COMPLEX_T_SUM)


def test_out_dir_mismatch_refused(tmp_path):
    full_dir = tmp_path / "fig3_cSTDP3"
    full_dir.mkdir()
    (full_dir / "manifest.json").write_text(json.dumps({"inhibition": "full"}), encoding="utf-8")
    try:
        rt.check_out_dir(full_dir, "local-only")
    except ValueError:
        pass
    else:
        raise AssertionError("local-only into a full manifest dir must be refused")
    rt.check_out_dir(full_dir, "full")  # matching mode passes

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text(json.dumps({"out": str(legacy)}), encoding="utf-8")
    try:
        rt.check_out_dir(legacy, "local-only")
    except ValueError:
        pass
    else:
        raise AssertionError("legacy (key-less) manifest means the full run and must refuse local-only")


def test_default_out_dirs():
    assert rt.resolve_out_dir(None, "local-only") == rt.OUT_S1
    assert rt.resolve_out_dir(None, "full") == rt.OUT_DIR
    assert rt.OUT_S1 != rt.OUT_DIR
    assert rt.resolve_out_dir(str(rt.OUT_DIR), "local-only") == rt.OUT_DIR  # explicit wins
