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


def test_core_full_run_refused_names_gap():
    old = rt._ENGINE
    rt._ENGINE = "core"
    try:
        try:
            rt._run_one(("missing-src.npz", "missing-dst.npz"))
        except NotImplementedError as exc:
            assert "cascade" in str(exc)
        else:
            raise AssertionError("engine core must refuse the full-network run")
        try:
            rt._init_worker("full", "core")
        except NotImplementedError as exc:
            assert "cascade" in str(exc)
        else:
            raise AssertionError("engine core must refuse the worker init")
    finally:
        rt._ENGINE = old
    assert rt._ENGINE == old


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
