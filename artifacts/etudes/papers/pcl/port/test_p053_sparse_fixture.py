"""P-053 sparse-weight fixture: the core arm against the oracle under hand-zeroed LAT and TD.

Binding criteria: P053_sparse_fixture_criteria.md, with Amendment 1 (P5 engagement, the w_s
control, VACUOUS). Per variant and seed, the same network (learning on, lateral and top-down on)
takes the same LAT/TD edit before either arm runs. The oracle runs on a deep copy, the core arm
on the network (read, never mutated). Spikes are exact; w_s and wi_s agree at rtol 1e-12, atol 0.
Nothing here fixes the core arm or the oracle.
"""

from __future__ import annotations

import copy
import functools
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

pytest.importorskip("numba", reason="the core arm needs numba (not a jaxfne dependency)")
import run_tuning as rt  # noqa: E402
import test_run_tuning as trt  # noqa: E402  (reused: INH, _counts)

INH = trt.INH
SEEDS = (0, 1, 2)
VARIANTS = ("V0", "V1", "V2", "V3")
N_EVENTS = 170  # criteria: 60 gives no spike; 170 is the smallest value probed that passes (result file)
RTOL = 1e-12
SPIKE_KEYS = ("simple_counts", "complex_counts", "simple_times_us", "simple_times_cell",
              "complex_times_us", "complex_times_cell")


def _burst(n: int):
    """n camera-0 events at one pixel, 20 us apart, the first at 20 us (as the sibling fixtures)."""
    t = (np.arange(n, dtype=np.int64) + 1) * 20
    x = np.full(n, 170, dtype=np.int64)
    y = np.full(n, 130, dtype=np.int64)
    p = np.zeros(n, dtype=np.int64)
    return t, x, y, p


def _one_per_row(M: np.ndarray) -> None:
    """Keep the largest-magnitude entry of each numpy row (first index on a tie), zero the rest."""
    keep = np.argmax(np.abs(M), axis=1)
    rows = np.arange(M.shape[0])
    vals = M[rows, keep].copy()
    M[:] = 0.0
    M[rows, keep] = vals


def _random_half(M: np.ndarray, seed) -> None:
    """Zero a seeded random half of the nonzero entries (exactly size // 2 of them)."""
    nz = np.flatnonzero(M)
    drop = np.random.default_rng(seed).choice(nz, size=nz.size // 2, replace=False)
    M.flat[drop] = 0.0


def _apply(net, variant: str, seed: int) -> None:
    """The variant's LAT/TD edit, applied to the network before either arm runs."""
    assert net.lat_on and net.td_on, "fixture must carry lateral and top-down"
    if variant == "V0":
        return
    if variant == "V1":
        _one_per_row(net.LAT)
        _one_per_row(net.TD)
    elif variant == "V2":
        _random_half(net.LAT, [seed, 0])
        _random_half(net.TD, [seed, 1])
    elif variant == "V3":
        net.LAT[:] = 0.0
        net.TD[:] = 0.0
    elif variant == "LAT0":  # reporting only: lateral removed, top-down intact
        net.LAT[:] = 0.0
    elif variant == "TD0":  # reporting only: top-down removed, lateral intact
        net.TD[:] = 0.0
    else:
        raise ValueError(f"unknown variant {variant!r}")


def _build(seed: int, variant: str):
    """A fresh network for one variant; deterministic in (seed, variant)."""
    net = ref.PCLEventRef(seed=seed, learning=True, layer_inhibitions=copy.deepcopy(INH))
    _apply(net, variant, seed)
    return net


def _rel(a, b) -> dict:
    """Max relative difference |a-b|/|b| (absolute |a-b| where b is 0), its index, and the tolerance check."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.abs(b)
    rel = np.abs(a - b) / np.where(den > 0, den, 1.0)
    idx = tuple(int(k) for k in np.unravel_index(int(np.argmax(rel)), rel.shape))
    return {"max_rel": float(rel.max()), "index": idx,
            "within": bool(np.allclose(a, b, rtol=RTOL, atol=0))}


def _oracle_arrays(net) -> dict:
    """The oracle's spikes in the core arm's output layout (counts, times, cells)."""
    out = {"simple_counts": np.array([len(c.spikes) for c in net.s_cells], dtype=np.int32),
           "complex_counts": np.array([len(c.spikes) for c in net.c_cells], dtype=np.int32)}
    out["simple_times_us"], out["simple_times_cell"] = rt._gather(net.s_cells)
    out["complex_times_us"], out["complex_times_cell"] = rt._gather(net.c_cells)
    return out


def _spikes_equal(a, b) -> bool:
    """Counts, times and cells of both populations, exact."""
    return all(np.array_equal(a[k], b[k]) for k in SPIKE_KEYS)


@functools.lru_cache(maxsize=None)
def _both(seed: int, variant: str):
    """Oracle (on a deep copy) and core arm (on the network) for one variant; built once and shared."""
    net = _build(seed, variant)
    pre = {"lat_nz": int(np.count_nonzero(net.LAT)), "td_nz": int(np.count_nonzero(net.TD)),
           "w_init": net.w_s.copy(), "wi_init": net.wi_s.copy()}
    t, x, y, p = _burst(N_EVENTS)
    oracle_net = copy.deepcopy(net)
    oracle_net.run(t, x, y, p)
    o = _oracle_arrays(oracle_net)
    o.update(pre)
    o.update(w_s=oracle_net.w_s.copy(), wi_s=oracle_net.wi_s.copy(),
             n_simple=int(sum(len(c.spikes) for c in oracle_net.s_cells)),
             n_complex=int(sum(len(c.spikes) for c in oracle_net.c_cells)))
    del oracle_net
    got = rt.run_core_events(net, t, x, y, p, learning=True)
    return o, got


def _measure(seed: int, variant: str) -> dict:
    """Core arm against oracle for one variant and seed: spikes, max relative w_s and wi_s."""
    o, got = _both(seed, variant)
    return {"n_simple": o["n_simple"], "n_complex": o["n_complex"],
            "spikes_equal": _spikes_equal(o, got),
            "ws": _rel(got["w_s"], o["w_s"]), "wi": _rel(got["wi_s"], o["wi_s"]),
            "w_moved": not np.array_equal(o["w_s"], o["w_init"]),
            "wi_moved": not np.array_equal(o["wi_s"], o["wi_init"]),
            "lat_nz": o["lat_nz"], "td_nz": o["td_nz"]}


@functools.lru_cache(maxsize=None)
def _p5(seed: int) -> dict:
    """P5 (Amendment 1), oracle only: V0 and V3 differ in spikes or in w_s/wi_s beyond rtol 1e-12."""
    a, _ = _both(seed, "V0")
    b, _ = _both(seed, "V3")
    spikes = _spikes_equal(a, b)
    ws = _rel(b["w_s"], a["w_s"])
    wi = _rel(b["wi_s"], a["wi_s"])
    engaged = (not spikes) or (not ws["within"]) or (not wi["within"])
    return {"engaged": engaged, "spikes_equal": spikes, "ws": ws, "wi": wi}


def _line(variant: str, seed: int, rep: dict) -> str:
    return (f"P053 {variant} seed {seed}: spikes {rep['n_simple']}s/{rep['n_complex']}c "
            f"equal={rep['spikes_equal']}; w_s max_rel={rep['ws']['max_rel']:.3e} at {rep['ws']['index']}; "
            f"wi_s max_rel={rep['wi']['max_rel']:.3e} at {rep['wi']['index']}")


@pytest.mark.parametrize("seed", SEEDS)
def test_p053_v0_dense_control_preconditions(seed):
    """V0 (no zeroing): the fixture spikes, moves w_s and wi_s, and LAT and TD are nonzero."""
    rep = _measure(seed, "V0")
    assert rep["n_simple"] > 0, f"fixture must spike, got {rep['n_simple']} simple spikes"
    assert rep["w_moved"], "fixture must move w_s"
    assert rep["wi_moved"], "fixture must move wi_s"
    assert rep["lat_nz"] > 0 and rep["td_nz"] > 0, "fixture must carry nonzero LAT and TD"


@pytest.mark.parametrize("seed", SEEDS)
def test_p053_p5_engagement(seed):
    """P5 (Amendment 1): V0 vs V3, oracle only. Not engaged means this seed's V1 to V3 are VACUOUS."""
    p = _p5(seed)
    print(f"P053 P5 seed {seed}: V0 vs V3 spikes_equal={p['spikes_equal']}; "
          f"w_s max_rel={p['ws']['max_rel']:.3e} at {p['ws']['index']}; "
          f"wi_s max_rel={p['wi']['max_rel']:.3e} at {p['wi']['index']}; engaged={p['engaged']}")
    if not p["engaged"]:
        pytest.skip(f"VACUOUS: P5 not engaged for seed {seed} (V0 and V3 identical)")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_p053_core_matches_oracle(variant, seed):
    """Per variant and seed: spikes exact, w_s and wi_s within rtol 1e-12 (atol 0)."""
    rep = _measure(seed, variant)
    print(_line(variant, seed, rep))
    if variant != "V0" and not _p5(seed)["engaged"]:
        pytest.skip(f"VACUOUS: P5 not engaged for seed {seed}; not counted")
    assert rep["spikes_equal"], f"{variant} seed {seed}: spikes differ: {_line(variant, seed, rep)}"
    assert rep["ws"]["within"], f"{variant} seed {seed}: w_s differs: {_line(variant, seed, rep)}"
    assert rep["wi"]["within"], f"{variant} seed {seed}: wi_s differs: {_line(variant, seed, rep)}"


def _core_only(seed: int, perturb):
    """The core arm alone on V0, with ``perturb`` applied to its network only."""
    net = _build(seed, "V0")
    perturb(net)
    t, x, y, p = _burst(N_EVENTS)
    return rt.run_core_events(net, t, x, y, p, learning=True)


def _moved_index(o) -> tuple:
    """The w_s element the oracle moved most (largest |w_s - w_init|)."""
    delta = np.abs(o["w_s"] - o["w_init"])
    return tuple(int(k) for k in np.unravel_index(int(np.argmax(delta)), delta.shape))


def _scale_w(idx):
    def perturb(net) -> None:
        net.w_s[idx] *= 1.004
    return perturb


@pytest.mark.parametrize("seed", SEEDS)
def test_p053_negative_control_ws_detects(seed):
    """Control (Amendment 1): +0.4% on one w_s element the oracle moved, core arm input only.

    The control detects only if the measured w_s max relative difference is at least 1e-6
    (owner ruling); the measured value is reported.
    """
    o, _ = _both(seed, "V0")
    idx = _moved_index(o)
    assert o["w_s"][idx] != o["w_init"][idx], "control element must be one the oracle moved"
    got = _core_only(seed, _scale_w(idx))
    ws = _rel(got["w_s"], o["w_s"])
    print(f"P053 control seed {seed}: w_s[{idx}] oracle {o['w_s'][idx]:.12g} moved from "
          f"{o['w_init'][idx]:.12g}; perturbed input +0.4%; spikes_equal={_spikes_equal(o, got)}; "
          f"w_s max_rel={ws['max_rel']:.3e} at {ws['index']}; within={ws['within']}")
    assert ws["max_rel"] >= 1e-6, (
        f"negative control detects nothing on w_s (seed {seed}): max_rel {ws['max_rel']:.3e} < 1e-6")


def _perturb_one_lat(net) -> None:
    """Core arm only: the largest-magnitude nonzero LAT entry scaled by 1.004 (+0.4%)."""
    i = int(np.argmax(np.abs(net.LAT)))
    net.LAT.flat[i] *= 1.004


@pytest.mark.parametrize("seed", SEEDS)
def test_p053_lat_engagement_measurement(seed):
    """Run-1 LAT perturbation (+0.4% on the largest LAT entry, core arm only): reported, not a control."""
    o, _ = _both(seed, "V0")
    got = _core_only(seed, _perturb_one_lat)
    ws = _rel(got["w_s"], o["w_s"])
    wi = _rel(got["wi_s"], o["wi_s"])
    print(f"P053 LAT engagement seed {seed}: spikes_equal={_spikes_equal(o, got)}; "
          f"w_s max_rel={ws['max_rel']:.3e} at {ws['index']}; "
          f"wi_s max_rel={wi['max_rel']:.3e} at {wi['index']}")
    assert np.isfinite(ws["max_rel"]) and np.isfinite(wi["max_rel"])


PATHWAY_VARIANTS = ("LAT0", "TD0")  # REPORTING ONLY (owner ruling): no gate, no verdict


@functools.lru_cache(maxsize=None)
def _oracle_only(seed: int, variant: str) -> dict:
    """The oracle alone on one variant, in the layout of _oracle_arrays, plus final w_s and wi_s."""
    net = _build(seed, variant)
    t, x, y, p = _burst(N_EVENTS)
    net.run(t, x, y, p)
    o = _oracle_arrays(net)
    o.update(w_s=net.w_s.copy(), wi_s=net.wi_s.copy())
    return o


def _pathway(seed: int, variant: str) -> dict:
    """One pathway alone (oracle only) against V0: spike times, w_s and wi_s, and whether it engages."""
    a = _oracle_only(seed, "V0")
    b = _oracle_only(seed, variant)
    spikes = _spikes_equal(a, b)
    times_a = np.concatenate([a["simple_times_us"], a["complex_times_us"]]).astype(np.float64)
    times_b = np.concatenate([b["simple_times_us"], b["complex_times_us"]]).astype(np.float64)
    if times_a.shape == times_b.shape and times_a.size > 0:
        times = _rel(times_b, times_a)
    else:
        times = {"max_rel": None, "index": None,
                 "within": None, "count": (times_a.size, times_b.size)}
    ws = _rel(b["w_s"], a["w_s"])
    wi = _rel(b["wi_s"], a["wi_s"])
    engaged = (not spikes) or (not ws["within"]) or (not wi["within"])
    return {"n_simple": int(b["simple_counts"].sum()), "n_complex": int(b["complex_counts"].sum()),
            "spikes_equal": spikes, "times": times, "ws": ws, "wi": wi, "engaged_alone": engaged}


@pytest.mark.parametrize("variant", PATHWAY_VARIANTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_p053_pathway_isolation_reporting(variant, seed):
    """REPORTING ONLY (owner ruling): one pathway alone vs V0; no gate, no verdict change."""
    r = _pathway(seed, variant)
    tm = r["times"]
    tm_txt = (f"max_rel={tm['max_rel']:.3e} at {tm['index']}" if tm["max_rel"] is not None
              else f"spike-time arrays differ in length {tm['count']}")
    print(f"P053 pathway {variant} seed {seed} (REPORTING ONLY): vs V0 spikes {r['n_simple']}s/"
          f"{r['n_complex']}c equal={r['spikes_equal']}; spike times {tm_txt}; "
          f"w_s max_rel={r['ws']['max_rel']:.3e} at {r['ws']['index']}; "
          f"wi_s max_rel={r['wi']['max_rel']:.3e} at {r['wi']['index']}; "
          f"engaged_alone={r['engaged_alone']}")
    assert np.isfinite(r["ws"]["max_rel"]) and np.isfinite(r["wi"]["max_rel"])
