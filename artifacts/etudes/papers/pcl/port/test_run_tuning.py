"""Inhibition switch for the Fig. 3 run: --inhibition {full,local-only,no-local,no-local-simple}.

local-only is the S1 sensitivity (P3_fig3.md); no-local (local flags off) and no-local-simple
(the simple layer's local flag off) are diagnostics.

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
_ROOT = str(Path(__file__).resolve().parents[5])
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pytest  # noqa: E402

import pcl_event_ref as ref  # noqa: E402

pytest.importorskip("numba", reason="the numba port needs numba (not a jaxfne dependency)")
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


def test_no_local_clears_both_flags_and_keeps_maps():
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    assert net.local_s and net.local_c and net.lat_on and net.td_on
    lat_before, td_before = net.LAT.copy(), net.TD.copy()
    wi_before = net.wi_s.copy()

    rt.apply_inhibition_mode(net, "no-local")

    assert not net.local_s and not net.local_c
    assert net.lat_on and net.td_on
    assert np.array_equal(net.LAT, lat_before) and np.array_equal(net.TD, td_before)
    assert float(np.abs(net.LAT).sum()) > 0.0 and float(np.abs(net.TD).sum()) > 0.0
    assert np.array_equal(net.wi_s, wi_before)


def test_no_local_changes_what_run_numba_consumes():
    full = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    nolocal = copy.deepcopy(full)
    rt.apply_inhibition_mode(nolocal, "no-local")

    nb.run_numba(full, *_burst())
    nb.run_numba(nolocal, *_burst())
    assert _counts(full)[0] > 0, "fixture must spike"
    assert _counts(nolocal) != _counts(full)


def test_no_local_simple_clears_only_the_simple_flag():
    full = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    mixed = copy.deepcopy(full)
    rt.apply_inhibition_mode(mixed, "no-local-simple")
    assert not mixed.local_s and mixed.local_c and mixed.lat_on and mixed.td_on
    assert np.array_equal(mixed.LAT, full.LAT) and np.array_equal(mixed.TD, full.TD)
    assert np.array_equal(mixed.wi_s, full.wi_s)

    nb.run_numba(full, *_burst())
    nb.run_numba(mixed, *_burst())
    assert _counts(full)[0] > 0, "fixture must spike"
    assert _counts(mixed) != _counts(full)
    assert rt.resolve_out_dir(None, "no-local-simple") == rt.OUT_NOLOCAL_SIMPLE


def test_unknown_mode_refused():
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    try:
        rt.apply_inhibition_mode(net, "nolocal")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown inhibition mode must be refused")
    assert net.local_s and net.local_c  # a refused mode changes nothing


def test_no_local_refused_in_full_run_dir(tmp_path, monkeypatch):
    scored = tmp_path / "runs" / "fig3_cSTDP3"
    monkeypatch.setitem(rt.MODE_DIRS, "full", scored)
    try:
        rt.check_out_dir(scored, "no-local")
    except ValueError:
        pass
    else:
        raise AssertionError("no-local into the full run's dir must be refused")


def test_no_local_manifest_mismatch_refused(tmp_path):
    d = tmp_path / "fig3_cSTDP3_nolocal"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"inhibition": "full"}), encoding="utf-8")
    try:
        rt.check_out_dir(d, "no-local")
    except ValueError:
        pass
    else:
        raise AssertionError("no-local into a full manifest dir must be refused")
    (d / "manifest.json").write_text(json.dumps({"inhibition": "no-local"}), encoding="utf-8")
    rt.check_out_dir(d, "no-local")  # matching mode passes


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
    assert rt.resolve_out_dir(None, "full") == Path("F:/warehouse/pcl/runs/fig3_cSTDP3")
    assert rt.resolve_out_dir(None, "local-only") == Path("F:/warehouse/pcl/runs/fig3_cSTDP3_S1")
    assert rt.resolve_out_dir(None, "no-local") == Path("F:/warehouse/pcl/diag_runs/fig3_cSTDP3_nolocal")
    assert rt.resolve_out_dir(None, "no-local-simple") == Path(
        "F:/warehouse/pcl/diag_runs/fig3_cSTDP3_nolocal_simple")
    assert len({rt.OUT_DIR, rt.OUT_S1, rt.OUT_NOLOCAL, rt.OUT_NOLOCAL_SIMPLE}) == 4
    assert rt.resolve_out_dir(str(rt.OUT_DIR), "local-only") == rt.OUT_DIR  # explicit wins


def test_every_mode_refused_in_another_modes_default_dir(tmp_path, monkeypatch):
    dirs = {m: tmp_path / m for m in rt.MODE_DIRS}  # fresh, manifest-less dirs
    monkeypatch.setattr(rt, "MODE_DIRS", dirs)
    for mode in dirs:
        for owner, d in dirs.items():
            if owner == mode:
                rt.check_out_dir(d, mode)  # its own default dir passes
                continue
            try:
                rt.check_out_dir(d, mode)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{mode} into {owner}'s default dir must be refused")
    detour = dirs["local-only"] / ".." / "full"  # another spelling of the full run's dir
    try:
        rt.check_out_dir(detour, "no-local")
    except ValueError:
        pass
    else:
        raise AssertionError("a '..' spelling of another mode's dir must be refused")


def test_engine_flag_defaults_to_numba():
    args = rt._build_parser().parse_args(["--subset", "pilot"])
    assert args.engine == "numba"
    args = rt._build_parser().parse_args(["--subset", "pilot", "--engine", "core"])
    assert args.engine == "core"


def test_core_matches_numba_all_modes():
    """ENGINE RULE: numba-vs-core equality pin on the synthetic seed-0 burst.

    Each inhibition mode replays the same 400-event feed through
    run_core_events; counts, per-spike times and cell ids must equal the
    numba oracle exactly (the scan carries the network's own cell
    parameters, so the default 30 mV simple threshold applies on both
    sides). Slow (~10 s per mode: one construct + simulate each).
    """
    for mode in ("full", "local-only", "no-local", "no-local-simple"):
        net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
        rt.apply_inhibition_mode(net, mode)
        t, x, y, p = _burst()
        nbn = copy.deepcopy(net)
        nb.run_numba(nbn, t, x, y, p)
        sc = np.array([len(c.spikes) for c in nbn.s_cells], dtype=np.int32)
        cc = np.array([len(c.spikes) for c in nbn.c_cells], dtype=np.int32)
        es_t, es_o = rt._gather(nbn.s_cells)
        ec_t, ec_o = rt._gather(nbn.c_cells)
        got = rt.run_core_events(net, t, x, y, p)
        assert np.array_equal(got["simple_counts"], sc), mode
        assert np.array_equal(got["complex_counts"], cc), mode
        assert np.array_equal(got["simple_times_us"], es_t), mode
        assert np.array_equal(got["simple_times_cell"], es_o), mode
        assert np.array_equal(got["complex_times_us"], ec_t), mode
        assert np.array_equal(got["complex_times_cell"], ec_o), mode
        assert int(got["n_events"]) == 400, mode


def test_core_lif_params_follow_network_config():
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    simple, complex_ = rt._core_lif_params(net)
    assert simple["v_thresh_mV"] == float(net.scfg.VTHRESH) == 30.0
    assert complex_["v_thresh_mV"] == float(net.ccfg.VTHRESH) == 3.0
    assert simple["tau_m_ms"] == float(net.scfg.TAU_M) / 1000.0 == 18.0
    assert complex_["tau_m_ms"] == float(net.ccfg.TAU_M) / 1000.0 == 50.0
    assert "tau_sra_ms" not in complex_  # the oracle carries no complex adaptation term
    assert complex_["delta_sra_mV"] == 0.0


def test_core_multispike_matches_numba():
    """A file in which cells spike twice replays exactly: counts, per-spike times and cells."""
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    n = 2000
    t = (np.arange(n, dtype=np.int64) + 1) * 20
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    nbn = copy.deepcopy(net)
    nb.run_numba(nbn, t, x, y, p)
    s_max = max(len(c.spikes) for c in nbn.s_cells)
    assert s_max == 2, f"fixture must multi-spike, got max {s_max}"
    sc = np.array([len(c.spikes) for c in nbn.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in nbn.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(nbn.s_cells)
    ec_t, ec_o = rt._gather(nbn.c_cells)
    got = rt.run_core_events(net, t, x, y, p)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    assert int(got["n_events"]) == n


def test_core_unsorted_and_empty_refused():
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    t, x, y, p = _burst()
    rev = slice(None, None, -1)
    try:
        rt.run_core_events(net, t[rev], x[rev], y[rev], p[rev])
    except ValueError as exc:
        assert "time-ordered" in str(exc)
    else:
        raise AssertionError("unsorted events must be refused")
    try:
        rt.run_core_events(net, t[:0], x[:0], y[:0], p[:0])
    except ValueError:
        pass
    else:
        raise AssertionError("an empty slice must be refused")


def test_core_interval_update_matches_numba():
    """The periodic rate and threshold update replays exactly across three intervals.

    Four 100-event bursts (20 us apart) start 120 ms apart, so the feed spans more than
    three 100 ms measurement intervals and each burst after the first starts past the
    interval since the last update. The default config carries ETA_TA 0, which never
    moves a threshold, so this network's simple eta is set to 2 mV. Counts, per-spike
    times and cells, and the final thresholds and rates must equal the numba run exactly.
    """
    net = ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH))
    net.scfg.ETA_SR = 2.0
    mi = int(net.measurement_interval)
    t = np.concatenate([k * 120000 + 1000 + np.arange(100, dtype=np.int64) * 20
                        for k in range(4)])
    x = np.full(t.shape[0], 170, dtype=np.int64)
    y = np.full(t.shape[0], 130, dtype=np.int64)
    p = np.zeros(t.shape[0], dtype=np.int64)
    assert int(t[-1]) - int(t[0]) >= 3 * mi, "fixture must span three measurement intervals"
    nbn = copy.deepcopy(net)
    thr0 = np.array([c.thr for c in nbn.s_cells + nbn.c_cells])
    nb.run_numba(nbn, t, x, y, p)
    assert sum(len(c.spikes) for c in nbn.s_cells) > 0, "fixture must spike"
    thr_nb = np.array([c.thr for c in nbn.s_cells + nbn.c_cells])
    rate_nb = np.array([c.rate for c in nbn.s_cells + nbn.c_cells])
    assert bool(np.any(thr_nb != thr0)), "fixture must move a threshold"
    sc = np.array([len(c.spikes) for c in nbn.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in nbn.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(nbn.s_cells)
    ec_t, ec_o = rt._gather(nbn.c_cells)
    got = rt.run_core_events(net, t, x, y, p)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    assert np.array_equal(got["final_threshold_mV"], thr_nb)
    assert np.array_equal(got["final_rate_hz"], rate_nb)
    assert int(got["n_events"]) == t.shape[0]


def _check_simple_stdp_against_oracle(cap, monkeypatch, *, min_buffered):
    """Simple excitatory buffer STDP replays the pure-Python oracle exactly.

    Lateral and top-down are off and the complex population does not learn. The
    2000-event multi-spike fixture of test_core_multispike_matches_numba runs through
    the oracle (learning on) and through run_core_events(learning=True). Spike counts,
    times and cells must be equal; the final w_s and wi_s agree at rtol 1e-12. With
    ``cap`` below the oracle's default the deques drop entries, so the test checks that
    some cell received more than ``min_buffered`` events between two of its spikes.
    """
    monkeypatch.setattr(ref, "CAP", cap)
    net = ref.PCLEventRef(seed=0, learning=True, layer_inhibitions=[["local"], ["local"]])
    net.ccfg.STDP_LEARNING = "off"
    w_init = net.w_s.copy()
    wi_init = net.wi_s.copy()
    n = 2000
    t = (np.arange(n, dtype=np.int64) + 1) * 20
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    oracle = copy.deepcopy(net)
    oracle.run(t, x, y, p)
    s_max = max(len(c.spikes) for c in oracle.s_cells)
    assert s_max >= 2, f"fixture must multi-spike, got max {s_max}"
    assert not np.array_equal(oracle.w_s, w_init), "fixture must move w_s"
    assert not np.array_equal(oracle.wi_s, wi_init), "fixture must move wi_s"
    buffered = 0
    for cell in oracle.s_cells:
        for s0, s1 in zip(cell.spikes[:-1], cell.spikes[1:]):
            buffered = max(buffered, int(np.sum((t > s0) & (t <= s1))))
    assert buffered > min_buffered, f"fixture must buffer more than {min_buffered} events, got {buffered}"
    sc = np.array([len(c.spikes) for c in oracle.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in oracle.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(oracle.s_cells)
    ec_t, ec_o = rt._gather(oracle.c_cells)
    got = rt.run_core_events(net, t, x, y, p, learning=True)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    np.testing.assert_allclose(got["w_s"], oracle.w_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_s"], oracle.wi_s, rtol=1e-12, atol=0)


def test_core_simple_stdp_matches_oracle(monkeypatch):
    _check_simple_stdp_against_oracle(1000, monkeypatch, min_buffered=0)


def test_core_simple_stdp_matches_oracle_cap3(monkeypatch):
    _check_simple_stdp_against_oracle(3, monkeypatch, min_buffered=3)


def test_core_inhibitory_simple_mode_matches_oracle(monkeypatch):
    """Simple mode "inhibitory" with lateral and top-down off changes no matrix.

    The oracle's simple update sits in its excitatory/all gate (pcl_event_ref.py:548-574),
    and its inhibitory gate (:575) does nothing with lateral and top-down off. So the
    oracle's w_s and wi_s stay at their initial values, and the replay with learning=True
    must match the oracle's spikes exactly and its final matrices exactly. Same
    2000-event multi-spike fixture as the excitatory test.
    """
    monkeypatch.setattr(ref, "CAP", 1000)
    net = ref.PCLEventRef(seed=0, learning=True, layer_inhibitions=[["local"], ["local"]])
    net.ccfg.STDP_LEARNING = "off"
    net.scfg.STDP_LEARNING = "inhibitory"
    w_init = net.w_s.copy()
    wi_init = net.wi_s.copy()
    n = 2000
    t = (np.arange(n, dtype=np.int64) + 1) * 20
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    oracle = copy.deepcopy(net)
    oracle.run(t, x, y, p)
    s_max = max(len(c.spikes) for c in oracle.s_cells)
    assert s_max >= 2, f"fixture must multi-spike, got max {s_max}"
    assert np.array_equal(oracle.w_s, w_init), "inhibitory mode must not move w_s"
    assert np.array_equal(oracle.wi_s, wi_init), "inhibitory mode must not move wi_s"
    sc = np.array([len(c.spikes) for c in oracle.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in oracle.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(oracle.s_cells)
    ec_t, ec_o = rt._gather(oracle.c_cells)
    got = rt.run_core_events(net, t, x, y, p, learning=True)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    assert np.array_equal(got["w_s"], oracle.w_s)
    assert np.array_equal(got["wi_s"], oracle.wi_s)


def _spikes_between(spike_lists, s0, s1):
    """Spikes of the listed cells strictly between two times (a buffer's possible entries)."""
    total = 0
    for sp in spike_lists:
        arr = np.asarray(sp, dtype=np.int64)
        total += int(np.sum((arr > s0) & (arr < s1)))
    return total


def _check_inhibitory_simple_against_oracle(cap, monkeypatch, *, min_buffered):
    """Simple "all" with lateral and top-down on replays the oracle's inhibitory update exactly.

    The complex population learns excitatory only. The 2000-event multi-spike fixture runs
    through the oracle (learning on) and through run_core_events(learning=True). Spikes must
    be equal; w_s, wi_s, w_c, wi_c, lat_w and td_w agree at rtol 1e-12. lat_w and td_w are in
    the scan's layout: lat_w[n, j] = LAT[lat_idx[n, j], n] and td_w[c, j] = TD[td_idx[c, j], c].
    Preconditions: LAT and TD move; some simple cell gets lateral and top-down entries in one
    interval between two of its spikes; with ``cap`` below the default, some cell receives more
    than ``min_buffered`` entries of one kind between two of its spikes (so the cap drops some).
    """
    monkeypatch.setattr(ref, "CAP", cap)
    net = ref.PCLEventRef(seed=0, learning=True,
                          layer_inhibitions=[["local", "lateral"], ["local", "topdown"]])
    net.ccfg.STDP_LEARNING = "excitatory"
    net.scfg.STDP_LEARNING = "all"
    assert net.lat_on and net.td_on, "fixture must carry lateral and top-down"
    w_init = net.w_s.copy()
    lat_init = net.LAT.copy()
    td_init = net.TD.copy()
    n = 2000
    t = (np.arange(n, dtype=np.int64) + 1) * 20
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    oracle = copy.deepcopy(net)
    oracle.run(t, x, y, p)
    s_max = max(len(c.spikes) for c in oracle.s_cells)
    assert s_max >= 2, f"fixture must multi-spike, got max {s_max}"
    assert not np.array_equal(oracle.w_s, w_init), "fixture must move w_s"
    assert not np.array_equal(oracle.LAT, lat_init), "fixture must move LAT"
    assert not np.array_equal(oracle.TD, td_init), "fixture must move TD"
    lat_max = 0
    td_max = 0
    both = False
    for m, cell in enumerate(oracle.s_cells):
        lat_src = [oracle.s_cells[int(q)].spikes for q in np.flatnonzero(oracle.lat_mask[:, m])]
        td_src = [oracle.c_cells[int(c)].spikes for c in np.flatnonzero(oracle.td_mask[m, :])]
        for s0, s1 in zip(cell.spikes[:-1], cell.spikes[1:]):
            n_lat = _spikes_between(lat_src, s0, s1)
            n_td = _spikes_between(td_src, s0, s1)
            lat_max = max(lat_max, n_lat)
            td_max = max(td_max, n_td)
            both = both or (n_lat > 0 and n_td > 0)
    assert both, "fixture must give a simple cell lateral and top-down entries before a spike"
    assert max(lat_max, td_max) > min_buffered, (
        f"fixture must buffer more than {min_buffered} entries of one kind, "
        f"got lateral {lat_max}, top-down {td_max}")
    sc = np.array([len(c.spikes) for c in oracle.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in oracle.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(oracle.s_cells)
    ec_t, ec_o = rt._gather(oracle.c_cells)
    got = rt.run_core_events(net, t, x, y, p, learning=True)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    np.testing.assert_allclose(got["w_s"], oracle.w_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_s"], oracle.wi_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["w_c"], oracle.w_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_c"], oracle.wi_c, rtol=1e-12, atol=0)
    tables = rt._core_tables(net)[0]
    lat_idx, lat_n = tables["lat_idx"], tables["lat_n"]
    td_idx, td_n = tables["td_idx"], tables["td_n"]
    lat_ok = np.arange(lat_idx.shape[1])[None, :] < lat_n[:, None]
    td_ok = np.arange(td_idx.shape[1])[None, :] < td_n[:, None]
    lat_want = np.where(lat_ok, oracle.LAT[lat_idx, np.arange(lat_idx.shape[0])[:, None]], 0.0)
    td_want = np.where(td_ok, oracle.TD[td_idx, np.arange(td_idx.shape[0])[:, None]], 0.0)
    np.testing.assert_allclose(np.asarray(got["lat_w"])[lat_ok], lat_want[lat_ok],
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(got["td_w"])[td_ok], td_want[td_ok],
                               rtol=1e-12, atol=0)


def test_core_inhibitory_lateral_topdown_matches_oracle(monkeypatch):
    _check_inhibitory_simple_against_oracle(1000, monkeypatch, min_buffered=0)


def test_core_inhibitory_lateral_topdown_matches_oracle_cap3(monkeypatch):
    _check_inhibitory_simple_against_oracle(3, monkeypatch, min_buffered=3)


def _complex_buffered_forwards(oracle):
    """Largest number of buffered complex events at a complex spike, counted from the spike logs.

    A complex cell's ``ev`` deque gains one entry per simple spike whose covering complex
    positions include the cell's position (oracle :708-720), and it clears at each complex
    spike. The entries between two spikes s0 < s1 are the simple spikes in (s0, s1] whose
    cover holds the cell's position (the entry of the spike at s1 is appended before its excite).
    """
    fwd = []  # (time, complex positions covering that simple position)
    for n, cell in enumerate(oracle.s_cells):
        pos = n // oracle.ND
        cover = set(oracle.c_cover[(pos // oracle.NY, pos % oracle.NY)])
        fwd.extend((ts, cover) for ts in cell.spikes)
    best = 0
    for c, cell in enumerate(oracle.c_cells):
        cp = c // oracle.CD
        where = (cp // oracle.CY, cp % oracle.CY)
        for s0, s1 in zip(cell.spikes[:-1], cell.spikes[1:]):
            best = max(best, sum(1 for ts, cov in fwd if s0 < ts <= s1 and where in cov))
    return best


def _check_complex_stdp_against_oracle(cap, monkeypatch, *, min_buffered):
    """Complex excitatory buffer STDP replays the pure-Python oracle exactly.

    The complex population learns in mode "excitatory" (ComplexNeuron::weightUpdate,
    pcl_event_ref.py:608-646); the simple population learns as in the simple test, and lateral
    and top-down are off. The 2000-event multi-spike fixture runs through the oracle and through
    run_core_events(learning=True). Spike counts, times and cells must be equal; the final w_s,
    wi_s, w_c and wi_c agree at rtol 1e-12. With ``cap`` below the oracle's default the complex
    deques drop entries, so the test checks that some complex cell had more than ``min_buffered``
    buffered events at one of its spikes.
    """
    monkeypatch.setattr(ref, "CAP", cap)
    net = ref.PCLEventRef(seed=0, learning=True, layer_inhibitions=[["local"], ["local"]])
    net.ccfg.STDP_LEARNING = "excitatory"
    wc_init = net.w_c.copy()
    wic_init = net.wi_c.copy()
    n = 2000
    t = (np.arange(n, dtype=np.int64) + 1) * 20
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    oracle = copy.deepcopy(net)
    oracle.run(t, x, y, p)
    c_max = max(len(c.spikes) for c in oracle.c_cells)
    assert c_max >= 2, f"fixture must complex-spike twice, got max {c_max}"
    assert not np.array_equal(oracle.w_c, wc_init), "fixture must move w_c"
    buffered = _complex_buffered_forwards(oracle)
    assert buffered > min_buffered, f"fixture must buffer more than {min_buffered} events, got {buffered}"
    sc = np.array([len(c.spikes) for c in oracle.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in oracle.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(oracle.s_cells)
    ec_t, ec_o = rt._gather(oracle.c_cells)
    got = rt.run_core_events(net, t, x, y, p, learning=True)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    np.testing.assert_allclose(got["w_s"], oracle.w_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_s"], oracle.wi_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["w_c"], oracle.w_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_c"], oracle.wi_c, rtol=1e-12, atol=0)


def test_core_complex_stdp_matches_oracle(monkeypatch):
    _check_complex_stdp_against_oracle(1000, monkeypatch, min_buffered=0)


def test_core_complex_stdp_matches_oracle_cap3(monkeypatch):
    _check_complex_stdp_against_oracle(3, monkeypatch, min_buffered=3)


def test_core_complex_stdp_interval_matches_oracle(monkeypatch):
    """Complex excitatory STDP with the periodic update: thresholds and rates agree.

    The four-burst feed of test_core_interval_update_matches_numba (ETA_SR 2 mV, measurement
    interval 100 ms) runs with complex excitatory learning. Step 1 of the complex update resets
    each complex cell's counter at its spike (pcl_event_ref.py:140), so the periodic rate and
    threshold see only the spikes since that reset. Spike counts, times and cells must be equal,
    the final w_c and wi_c agree at rtol 1e-12, and the final thresholds and rates agree at
    rtol 1e-12. The fixture must move a threshold and a rate, and a complex cell must spike twice.
    """
    monkeypatch.setattr(ref, "CAP", 1000)
    net = ref.PCLEventRef(seed=0, learning=True, layer_inhibitions=[["local"], ["local"]])
    net.scfg.ETA_SR = 2.0
    net.ccfg.STDP_LEARNING = "excitatory"
    t = np.concatenate([k * 120000 + 1000 + np.arange(100, dtype=np.int64) * 20
                        for k in range(4)])
    x = np.full(t.shape[0], 170, dtype=np.int64)
    y = np.full(t.shape[0], 130, dtype=np.int64)
    p = np.zeros(t.shape[0], dtype=np.int64)
    wc_init = net.w_c.copy()
    oracle = copy.deepcopy(net)
    thr0 = np.array([c.thr for c in oracle.s_cells + oracle.c_cells])
    oracle.run(t, x, y, p)
    thr_or = np.array([c.thr for c in oracle.s_cells + oracle.c_cells])
    rate_or = np.array([c.rate for c in oracle.s_cells + oracle.c_cells])
    assert bool(np.any(thr_or != thr0)), "fixture must move a threshold"
    assert bool(np.any(rate_or != 0.0)), "fixture must move a rate"
    assert max(len(c.spikes) for c in oracle.c_cells) >= 2, "fixture must complex-spike twice"
    assert not np.array_equal(oracle.w_c, wc_init), "fixture must move w_c"
    sc = np.array([len(c.spikes) for c in oracle.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in oracle.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(oracle.s_cells)
    ec_t, ec_o = rt._gather(oracle.c_cells)
    got = rt.run_core_events(net, t, x, y, p, learning=True)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    np.testing.assert_allclose(got["final_threshold_mV"], thr_or, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["final_rate_hz"], rate_or, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["w_s"], oracle.w_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_s"], oracle.wi_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["w_c"], oracle.w_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_c"], oracle.wi_c, rtol=1e-12, atol=0)


def _check_complex_same_pass_against_oracle(cap, monkeypatch, *, min_pass_max):
    """Several complex cells of one position spiking in one simple-spike pass replay exactly.

    With the complex local inhibition zeroed and the complex row scaled by 20, the cells of a
    position cross threshold on the same forward. The second cell's loc window then reads the
    first cell's push from that pass, and at CAP 3 one pass pushes more than CAP entries to the
    group. A recorder on the oracle counts complex spikes per (simple-spike pass, position);
    the fixture must hold a pass with at least ``min_pass_max`` spikes at one position.
    Spikes must be equal and the final w_c, wi_c, w_s and wi_s agree at rtol 1e-12.
    """
    monkeypatch.setattr(ref, "CAP", cap)
    net = ref.PCLEventRef(seed=0, learning=True, layer_inhibitions=[["local"], ["local"]])
    net.ccfg.STDP_LEARNING = "excitatory"
    net.wi_c[:] = 0.0
    net.w_c *= 20.0
    n = 300
    t = (np.arange(n, dtype=np.int64) + 1) * 20
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    oracle = copy.deepcopy(net)
    per_pass = {}
    orig_s, orig_c = oracle._on_simple_spike, oracle._on_complex_spike
    state = {"pass": 0}

    def on_simple(n_, i, j, k, t_):
        state["pass"] += 1
        return orig_s(n_, i, j, k, t_)

    def on_complex(c, cx, cy, cz, t_):
        key = (state["pass"], cx, cy)
        per_pass[key] = per_pass.get(key, 0) + 1
        return orig_c(c, cx, cy, cz, t_)

    oracle._on_simple_spike = on_simple
    oracle._on_complex_spike = on_complex
    oracle.run(t, x, y, p)
    most = max(per_pass.values(), default=0)
    assert most >= min_pass_max, f"fixture must spike {min_pass_max} cells of one position in one pass, got {most}"
    assert sum(1 for v in per_pass.values() if v >= 2) > 0, "fixture must hold a multi-spike pass"
    sc = np.array([len(c.spikes) for c in oracle.s_cells], dtype=np.int32)
    cc = np.array([len(c.spikes) for c in oracle.c_cells], dtype=np.int32)
    es_t, es_o = rt._gather(oracle.s_cells)
    ec_t, ec_o = rt._gather(oracle.c_cells)
    got = rt.run_core_events(net, t, x, y, p, learning=True)
    assert np.array_equal(got["simple_counts"], sc)
    assert np.array_equal(got["complex_counts"], cc)
    assert np.array_equal(got["simple_times_us"], es_t)
    assert np.array_equal(got["simple_times_cell"], es_o)
    assert np.array_equal(got["complex_times_us"], ec_t)
    assert np.array_equal(got["complex_times_cell"], ec_o)
    np.testing.assert_allclose(got["w_s"], oracle.w_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_s"], oracle.wi_s, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["w_c"], oracle.w_c, rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["wi_c"], oracle.wi_c, rtol=1e-12, atol=0)


def test_core_complex_same_pass_matches_oracle(monkeypatch):
    _check_complex_same_pass_against_oracle(1000, monkeypatch, min_pass_max=2)


def test_core_complex_same_pass_matches_oracle_cap3(monkeypatch):
    _check_complex_same_pass_against_oracle(3, monkeypatch, min_pass_max=4)


def test_build_s1_scan_tensor_lif_partial_refused():
    from jaxfne.pcl_topology import build_s1_scan_tensor

    simple, _ = rt._core_lif_params(
        ref.PCLEventRef(seed=0, learning=False, layer_inhibitions=copy.deepcopy(INH)))
    try:
        build_s1_scan_tensor(ns=8, nc=4, lif_simple=simple)
    except ValueError as exc:
        assert "together" in str(exc)
    else:
        raise AssertionError("a partial LIF declaration must be refused")
    ten = build_s1_scan_tensor(ns=8, nc=4)  # undeclared still builds
    assert [nt.name for lay in ten.areas[0].layers for nt in lay.neuron_types] == [
        "simple", "complex"]


def test_core_smoke_z_from_tree_core():
    """ENGINE RULE: this z comes from the tree's jaxfne core, never the oracle."""
    import jax.numpy as jnp

    import jaxfne
    from jaxfne.emitters_lif import make_lif_params, simulate_lif_edge

    root = Path(rt._ROOT).resolve()
    assert Path(jaxfne.__file__).resolve().is_relative_to(root), (
        f"must use the tree jaxfne, got {jaxfne.__file__}")
    p = make_lif_params(("simple", "simple"))
    p = p._replace(**{f: jnp.asarray(getattr(p, f), jnp.float32) for f in (
        "tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV",
        "tau_rp_ms", "tau_sra_ms", "delta_sra_mV")})
    drive = np.zeros((3, 2), np.float32)
    drive[0, 0] = 30.0
    out = simulate_lif_edge(
        p, n_steps=3, dt_ms=0.5,
        edge_pre=np.array([0]), edge_post=np.array([1]),
        edge_weight=np.array([12.0]), edge_delay_steps=np.array([1]),
        I_ext_pA=jnp.asarray(drive),
    )
    spk = np.asarray(out["spikes"])
    assert spk[0, 0] and not spk[0, 1]
    assert spk[1, 1], spk  # 12 mV arrival crosses threshold 10
