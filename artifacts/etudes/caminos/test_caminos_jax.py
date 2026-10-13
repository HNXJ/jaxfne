"""Stage-A G0 tests for the CAMINOS JAX AdEx network (fast, CPU).

G0: single AdEx cells under 500 ms step currents through
``caminos_jax.adex_step`` (float64) versus an independent float64 NumPy
loop written here from Eqs. 15-16 (never calling ``caminos_jax`` for the
reference trajectory). Spike times must match within one dt.

Step-current levels (pA), chosen so the three levels give 0 / few / many
spikes (observed counts in brackets, single cell, 500 ms, float64):
E:   0 (0), 200 (2), 400 (9).
PV:  0 (0), 150 (4), 600 (43).
SOM: 0 (0), 150 (3), 450 (10).
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
while _repo_root.parent != _repo_root and not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import caminos_jax as cj

DT = 1e-4  # s, reference Euler step (PORT_SPEC Methods)
NSTEPS = 5000  # 500 ms
LEVELS = {"E": (0.0, 200.0, 400.0), "PV": (0.0, 150.0, 600.0), "SOM": (0.0, 150.0, 450.0)}
EXPECTED_COUNTS = {"E": (0, 2, 9), "PV": (0, 4, 43), "SOM": (0, 3, 10)}


def _run_jax(kind, i_pa, dtype):
    """Drive one cell through caminos_jax.adex_step; return spike steps."""
    p = {k: jnp.asarray(v, dtype=dtype) for k, v in cj.cell_params(kind).items()}
    v = jnp.asarray(p["EL_V"])
    w = jnp.asarray(0.0, dtype=dtype)
    r = jnp.asarray(0, jnp.int32)
    cur = jnp.asarray(i_pa * 1e-12, dtype=dtype)
    state = (v, w, r)
    spikes = []
    for t in range(NSTEPS):
        state, spk = cj.adex_step(state, cur, p, np.float64(DT))
        if bool(spk):
            spikes.append(t)
    return spikes


def _run_ref(kind, i_pa):
    """Independent float64 scalar loop from Eqs. 15-16; return spike steps."""
    p = {k: float(v) for k, v in cj.cell_params(kind).items()}
    v = p["EL_V"]
    w = 0.0
    ref = 0
    cur = i_pa * 1e-12
    n_ref = int(round(p["Tref_s"] / DT))
    v_thr = p["VT_V"] + 5.0 * p["Delta_V"]
    spikes = []
    for t in range(NSTEPS):
        if ref <= 0:
            arg = (v - p["VT_V"]) / p["Delta_V"]
            arg = arg if arg < 20.0 else 20.0
            dv = (p["gL_S"] * (p["EL_V"] - v)
                  + p["gL_S"] * p["Delta_V"] * np.exp(arg)
                  - w + cur) / p["C_F"] * DT
            v_cand = v + dv
            if v_cand >= v_thr:
                spikes.append(t)
                v = p["Vreset_V"]
                w = w + (p["a_S"] * (v - p["EL_V"]) - w) / p["tauw_s"] * DT + p["b_A"]
                ref = n_ref
            else:
                v = v_cand
                w = w + (p["a_S"] * (v - p["EL_V"]) - w) / p["tauw_s"] * DT
        else:
            v = p["Vreset_V"]
            w = w + (p["a_S"] * (v - p["EL_V"]) - w) / p["tauw_s"] * DT
            ref -= 1
    return spikes


@pytest.mark.parametrize("kind", ["E", "PV", "SOM"])
def test_g0_spike_times_match_reference(kind):
    """G0: adex_step (float64) matches the independent reference within 1 dt."""
    for level, (i_pa, expected) in enumerate(zip(LEVELS[kind], EXPECTED_COUNTS[kind])):
        got = _run_jax(kind, i_pa, jnp.float64)
        ref = _run_ref(kind, i_pa)
        assert len(got) == expected, (kind, i_pa, len(got))
        assert len(ref) == expected, (kind, i_pa, len(ref))
        assert len(got) == len(ref), (kind, i_pa)
        if got:
            assert max(abs(g - r) for g, r in zip(got, ref)) <= 1, (kind, i_pa)
        if level == 0:
            assert got == []
        else:
            assert len(got) > 0, (kind, i_pa, "fixture must spike")


@pytest.mark.parametrize("kind", ["E", "PV", "SOM"])
def test_f32_matches_f64_counts(kind):
    """Float32 adex_step gives the same spike count as float64 (middle level)."""
    i_pa = LEVELS[kind][1]
    got32 = _run_jax(kind, i_pa, jnp.float32)
    got64 = _run_jax(kind, i_pa, jnp.float64)
    assert len(got32) == len(got64) == EXPECTED_COUNTS[kind][1], kind


def test_network_smoke():
    """Small net (NE=800, NPV=75, NSOM=75, p=0.07, T=0.3 s): finite,
    non-negative, key-sensitive and deterministic counts."""
    kw = dict(p=0.07, NE=800, NPV=75, NSOM=75)
    c1, _ = cj.simulate(jax.random.key(0), 0.3, **kw)
    c2, _ = cj.simulate(jax.random.key(1), 0.3, **kw)
    c1b, _ = cj.simulate(jax.random.key(0), 0.3, **kw)
    c1 = np.asarray(c1)
    assert c1.shape == (3000, 3)
    assert np.all(np.isfinite(c1))
    assert np.all(c1 >= 0)
    assert np.all(c1.sum(axis=0) > 0), c1.sum(axis=0)
    assert not np.array_equal(np.asarray(c1), np.asarray(c2))
    assert np.array_equal(np.asarray(c1), np.asarray(c1b))


def test_network_structure():
    """PV->SOM and SOM->SOM absent; no autapses; E->E fraction ~= p."""
    net = cj.build_network(jax.random.key(0))
    assert ("PV", "SOM") not in net["blocks"]
    assert ("SOM", "SOM") not in net["blocks"]
    ee = np.asarray(net["blocks"][("E", "E")])
    pp = np.asarray(net["blocks"][("PV", "PV")])
    assert not ee.diagonal().any()
    assert not pp.diagonal().any()
    assert abs(ee.mean() - 0.07) < 0.01, ee.mean()


@pytest.mark.parametrize("src", ["E", "PV", "SOM"])
def test_route_inputs_signs(src):
    """E spikes raise gE only; PV and SOM spikes raise gI only (Table 2 reversal potentials)."""
    sizes = {"E": 40, "PV": 10, "SOM": 10}
    net = cj.build_network(jax.random.key(3), p=0.5, NE=40, NPV=10, NSOM=10)
    q = {k: 1.0 for k in ("q_ee", "q_ep", "q_es", "q_pe", "q_pp", "q_se", "q_sp", "q_ext")}
    lo = {"E": 0, "PV": 40, "SOM": 50}[src]
    arr = np.zeros(60, bool)
    arr[lo:lo + sizes[src]] = True
    ge, gi = cj.route_inputs(net["blocks"], q, jnp.asarray(arr), jnp.zeros(50, jnp.int32),
                             sizes, jnp.float32)
    ge, gi = np.asarray(ge), np.asarray(gi)
    if src == "E":
        assert ge.sum() > 0 and gi.sum() == 0
    else:
        assert gi.sum() > 0 and ge.sum() == 0


def test_delay_ring_twenty_steps():
    """Docstring: an arrival read from the ring enters the V update of the
    same step, 20 steps (2 ms at dt = 0.1 ms) after the presynaptic spike
    step. ``simulate`` exposes no conductance trace (only per-step counts
    plus final gE/gI), so this reproduces the ring indexing from the
    docstring and checks the read step through the real ``route_inputs``:
    a spike pushed at step 5 first raises a postsynaptic conductance at
    step 25."""
    dt = float(cj.PARAMS["dt_s"])
    n_delay = max(int(round(float(cj.PARAMS["tau_d_s"]) / dt)), 1)
    assert n_delay == 20, n_delay
    sizes = {"E": 2, "PV": 1, "SOM": 1}
    net = cj.build_network(jax.random.key(0), p=1.0, NE=2, NPV=1, NSOM=1)
    ee = np.asarray(net["blocks"][("E", "E")])
    assert bool(ee[0, 1]), ee  # E0 -> E1 present (p = 1, no autapse)
    q = {
        "q_ee": jnp.asarray(cj.PARAMS["Q_E_E_S"], jnp.float32),
        "q_ep": jnp.asarray(cj.PARAMS["Q_E_PV_S"], jnp.float32),
        "q_es": jnp.asarray(cj.PARAMS["Q_E_SOM_S"], jnp.float32),
        "q_pe": jnp.asarray(cj.PARAMS["Q_PV_E_S"], jnp.float32),
        "q_pp": jnp.asarray(cj.PARAMS["Q_PV_PV_S"], jnp.float32),
        "q_se": jnp.asarray(cj.PARAMS["Q_SOM_E_S"], jnp.float32),
        "q_sp": jnp.asarray(cj.PARAMS["Q_SOM_PV_S"], jnp.float32),
        "q_ext": jnp.asarray(cj.PARAMS["Q_ext_S"], jnp.float32),
    }
    n_tot = 4
    ring = np.zeros((n_delay, n_tot), bool)
    ering = np.zeros((n_delay, 3), np.int32)  # NE + NPV external counts
    spike = np.zeros(n_tot, bool)
    spike[0] = True
    head = 0
    first = None
    for t in range(45):
        arr = ring[head].copy()
        ge, _ = cj.route_inputs(net["blocks"], q, jnp.asarray(arr),
                                jnp.asarray(ering[head]), sizes, jnp.float32)
        ge = np.asarray(ge)
        if t == 5:
            ring[head] = spike  # push presynaptic spike of step 5
        else:
            ring[head] = np.zeros(n_tot, bool)
        ering[head] = np.zeros(3, np.int32)
        if ge.sum() > 0 and first is None:
            first = t
            assert t == 25, t
            assert ge[1] > 0, ge  # E0 -> E1 conductance arrives here
        head = (head + 1) % n_delay
    assert first == 25, first


def test_refractory_clamp_fifty_steps():
    """adex_step under a strong constant current: consecutive spikes are at
    least round(Tref/dt) + 1 steps apart and V equals Vreset on each of the
    50 steps after a spike (docstring clamp)."""
    dt = float(cj.PARAMS["dt_s"])
    n_ref = int(round(float(cj.PARAMS["E_Tref_s"]) / dt))
    assert n_ref == 50, n_ref
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in cj.cell_params("E").items()}
    vreset = float(p["Vreset_V"])
    cur = jnp.asarray(3000.0 * 1e-12, dtype=jnp.float64)
    state = (jnp.asarray(p["EL_V"]), jnp.asarray(0.0, dtype=jnp.float64),
             jnp.asarray(0, jnp.int32))
    spikes = []
    vs = []
    for t in range(2000):
        state, spk = cj.adex_step(state, cur, p, dt)
        vs.append(float(state[0]))
        if bool(spk):
            spikes.append(t)
    assert len(spikes) >= 2, "fixture must spike repeatedly"
    assert min(np.diff(spikes)) >= n_ref + 1, (spikes[:6], min(np.diff(spikes)))
    vs = np.asarray(vs)
    for s in spikes:
        if s + n_ref >= len(vs):
            break
        seg = vs[s + 1:s + 1 + n_ref]
        assert np.all(seg == vreset), (s, seg[:6])


def test_external_drive_mean_within_3se():
    """External drive: summed input per E cell per step is Poisson(0.224);
    mean over 10 s of steps within 3 SE (same jax.random.poisson call path
    as ``simulate``)."""
    lam_live = (float(cj.PARAMS["ext_sources"]) * float(cj.PARAMS["ext_rate_Hz"])
                * float(cj.PARAMS["dt_s"]))
    assert lam_live == pytest.approx(0.224), lam_live
    steps = int(round(10.0 / float(cj.PARAMS["dt_s"])))
    assert steps == 100000, steps
    n_e = 5
    x = jax.random.poisson(jax.random.key(0), lam_live, shape=(steps, n_e))
    jax.block_until_ready(x)
    x = np.asarray(x, dtype=np.float64)
    se = float(np.sqrt(0.224 / x.size))
    assert abs(x.mean() - 0.224) <= 3 * se, (x.mean(), se)


# --- Opt-in variants S3/S4/S5 + exploratory ext_sources (defaults unchanged) ---

PIN_SEED = 7  # small seeded defaults run, recorded from the base file
PIN_P = 0.07
PIN_NE, PIN_NPV, PIN_NSOM = 40, 8, 8
PIN_T = 0.05
PIN_SHAPE = (500, 3)
PIN_SUMS = [113, 32, 2]
PIN_SHA = "661385b0217f1f90daf4a5b40b62596087bad21381dab119eeb1468a13920de2"
PIN_FIRST_SPIKE_STEP = 84
PIN_ROWS_84_86 = [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
PIN_ROWS_100_102 = [[0, 0, 0], [0, 1, 0], [1, 0, 0]]


def test_defaults_match_base_counts():
    """(a) explicit-default variants give the base-code counts bit-for-bit."""
    import hashlib
    counts, _ = cj.simulate(jax.random.key(PIN_SEED), PIN_T, p=PIN_P,
                            NE=PIN_NE, NPV=PIN_NPV, NSOM=PIN_NSOM,
                            indegree="bernoulli", ext="private",
                            step_order="default", ext_sources="fixed")
    counts = np.asarray(counts)
    assert counts.shape == PIN_SHAPE, counts.shape
    assert counts.dtype == np.int32, counts.dtype
    assert counts.sum(axis=0).tolist() == PIN_SUMS, counts.sum(axis=0)
    assert hashlib.sha256(counts.tobytes()).hexdigest() == PIN_SHA
    nz = np.flatnonzero(counts.sum(axis=1))
    assert int(nz[0]) == PIN_FIRST_SPIKE_STEP, int(nz[0])
    assert counts[84:87].tolist() == PIN_ROWS_84_86, counts[84:87].tolist()
    assert counts[100:103].tolist() == PIN_ROWS_100_102, counts[100:103].tolist()


def test_fixed_indegree_exact_per_post_cell():
    """(b) S3: every column of every block has exactly round(p*N_pre)
    partners; E->E and PV->PV diagonals are empty."""
    p, NE, NPV, NSOM = 0.3, 20, 6, 6
    net = cj.build_network(jax.random.key(11), p=p, NE=NE, NPV=NPV, NSOM=NSOM,
                           indegree="fixed")
    assert ("PV", "SOM") not in net["blocks"]
    assert ("SOM", "SOM") not in net["blocks"]
    sizes = {"E": NE, "PV": NPV, "SOM": NSOM}
    for (pre, post), blk in net["blocks"].items():
        blk = np.asarray(blk)
        assert blk.shape == (sizes[pre], sizes[post]), (pre, post, blk.shape)
        assert blk.dtype == bool, (pre, post, blk.dtype)
        k = int(round(p * sizes[pre]))
        assert bool((blk.sum(axis=0) == k).all()), (pre, post, k, blk.sum(axis=0))
    ee = np.asarray(net["blocks"][("E", "E")])
    pp = np.asarray(net["blocks"][("PV", "PV")])
    assert not ee.diagonal().any()
    assert not pp.diagonal().any()


def test_shared_ext_mean_and_correlation():
    """(c) S4: mean per-cell drive within 3 SE of p*560*4*dt; two cells
    sharing trains have positive count correlation."""
    import jax as _jax
    dt = float(cj.PARAMS["dt_s"])
    p, n_src, rate = 0.07, 560, 4.0
    n_tgt, steps = 6, 30000
    counts, ext_map = cj.shared_external_counts(_jax.random.key(5), steps, n_src,
                                                n_tgt, p, rate, dt)
    _jax.block_until_ready(counts)
    counts = np.asarray(counts, dtype=np.float64)
    m = np.asarray(ext_map)
    assert m.shape == (n_src, n_tgt) and m.dtype == bool
    mu = p * n_src * rate * dt
    se = float(np.sqrt(mu / counts.size))
    assert abs(counts.mean() - mu) <= 3 * se, (counts.mean(), mu, se)
    overlap = m.astype(np.int32).T @ m.astype(np.int32)
    np.fill_diagonal(overlap, 0)
    i, j = np.unravel_index(int(np.argmax(overlap)), overlap.shape)
    assert int(overlap[i, j]) > 0, "fixture must share a train"
    cc = np.corrcoef(counts[:, i], counts[:, j])[0, 1]
    assert cc > 0, (i, j, int(overlap[i, j]), cc)


def test_brian2_step_order_ring_twenty_one():
    """(d) S5 ring: the delay ring is 21 steps under step_order='brian2'
    (20 by default); a spike pushed at step 5 first raises a postsynaptic
    conductance at step 26, in the style of test_delay_ring_twenty_steps."""
    dt = float(cj.PARAMS["dt_s"])
    assert cj.delay_steps(dt, "default") == 20, cj.delay_steps(dt, "default")
    n_delay = cj.delay_steps(dt, "brian2")
    assert n_delay == 21, n_delay
    sizes = {"E": 2, "PV": 1, "SOM": 1}
    net = cj.build_network(jax.random.key(0), p=1.0, NE=2, NPV=1, NSOM=1)
    ee = np.asarray(net["blocks"][("E", "E")])
    assert bool(ee[0, 1]), ee  # E0 -> E1 present (p = 1, no autapse)
    q = {
        "q_ee": jnp.asarray(cj.PARAMS["Q_E_E_S"], jnp.float32),
        "q_ep": jnp.asarray(cj.PARAMS["Q_E_PV_S"], jnp.float32),
        "q_es": jnp.asarray(cj.PARAMS["Q_E_SOM_S"], jnp.float32),
        "q_pe": jnp.asarray(cj.PARAMS["Q_PV_E_S"], jnp.float32),
        "q_pp": jnp.asarray(cj.PARAMS["Q_PV_PV_S"], jnp.float32),
        "q_se": jnp.asarray(cj.PARAMS["Q_SOM_E_S"], jnp.float32),
        "q_sp": jnp.asarray(cj.PARAMS["Q_SOM_PV_S"], jnp.float32),
        "q_ext": jnp.asarray(cj.PARAMS["Q_ext_S"], jnp.float32),
    }
    n_tot = 4
    ring = np.zeros((n_delay, n_tot), bool)
    ering = np.zeros((n_delay, 3), np.int32)  # NE + NPV external counts
    spike = np.zeros(n_tot, bool)
    spike[0] = True
    head = 0
    first = None
    for t in range(5 + n_delay + 20):
        arr = ring[head].copy()
        ge, _ = cj.route_inputs(net["blocks"], q, jnp.asarray(arr),
                                jnp.asarray(ering[head]), sizes, jnp.float32)
        ge = np.asarray(ge)
        if t == 5:
            ring[head] = spike  # push presynaptic spike of step 5
        else:
            ring[head] = np.zeros(n_tot, bool)
        ering[head] = np.zeros(3, np.int32)
        if ge.sum() > 0 and first is None:
            first = t
            assert t == 5 + n_delay, t
            assert ge[1] > 0, ge  # E0 -> E1 conductance arrives here
        head = (head + 1) % n_delay
    assert first == 5 + n_delay == 26, first


def test_brian2_refractory_clamp_forty_nine():
    """(d) S5 clamp: adex_step under step_order='brian2' clamps 49 steps
    (50 by default), in the style of test_refractory_clamp_fifty_steps."""
    dt = float(cj.PARAMS["dt_s"])
    assert cj.refractory_steps(float(cj.PARAMS["E_Tref_s"]), dt, "default") == 50
    n_ref = cj.refractory_steps(float(cj.PARAMS["E_Tref_s"]), dt, "brian2")
    assert n_ref == 49, n_ref
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in cj.cell_params("E").items()}
    vreset = float(p["Vreset_V"])
    cur = jnp.asarray(3000.0 * 1e-12, dtype=jnp.float64)
    state = (jnp.asarray(p["EL_V"]), jnp.asarray(0.0, dtype=jnp.float64),
             jnp.asarray(0, jnp.int32))
    spikes = []
    vs = []
    for t in range(2000):
        state, spk = cj.adex_step(state, cur, p, dt, "brian2")
        vs.append(float(state[0]))
        if bool(spk):
            spikes.append(t)
    assert len(spikes) >= 2, "fixture must spike repeatedly"
    assert min(np.diff(spikes)) >= n_ref + 1, (spikes[:6], min(np.diff(spikes)))
    vs = np.asarray(vs)
    for s in spikes:
        if s + n_ref >= len(vs):
            break
        seg = vs[s + 1:s + 1 + n_ref]
        assert np.all(seg == vreset), (s, seg[:6])


def test_pne_ext_sources_lam():
    """(e) exploratory ext_sources='pNE': the source count at p = 0.02 is
    round(0.02*8000) = 160, so lam is 160*4*dt."""
    assert cj.ext_source_count(0.07, 8000, "fixed") == 560
    assert cj.ext_source_count(0.02, 8000, "pNE") == 160
    lam = cj.external_lam(0.02, 8000, float(cj.PARAMS["dt_s"]), "pNE")
    assert lam == pytest.approx(160 * 4.0 * float(cj.PARAMS["dt_s"])), lam


def test_variants_simulate_smoke():
    """Every variant runs end to end on a small net (finite, non-negative,
    right shape); the all-variants run differs from the default run."""
    kw = dict(p=0.07, NE=40, NPV=8, NSOM=8)
    c0, _ = cj.simulate(jax.random.key(7), 0.02, **kw)
    cv, _ = cj.simulate(jax.random.key(7), 0.02, indegree="fixed", ext="shared",
                        step_order="brian2", ext_sources="pNE", **kw)
    c0 = np.asarray(c0)
    cv = np.asarray(cv)
    assert c0.shape == (200, 3) and cv.shape == (200, 3)
    assert np.all(np.isfinite(cv)) and np.all(cv >= 0)
    assert not np.array_equal(c0, cv)
    for one in (dict(indegree="fixed"), dict(ext="shared"),
                dict(step_order="brian2"), dict(ext_sources="pNE")):
        c1, _ = cj.simulate(jax.random.key(7), 0.02, **kw, **one)
        assert np.asarray(c1).shape == (200, 3), one


def test_run_driver_variant_flags():
    """run_caminos.py exposes one flag per variant (defaults off); the file
    name gains a tag per non-default variant; the manifest records them."""
    import run_caminos
    ap = run_caminos.build_parser()
    d = ap.parse_args(["--seeds", "0", "--out", "x"])
    assert (d.indegree, d.ext, d.step_order, d.ext_sources) == (
        "bernoulli", "private", "default", "fixed")
    v = ap.parse_args(["--seeds", "0", "--out", "x", "--indegree", "fixed",
                       "--ext", "shared", "--step-order", "brian2",
                       "--ext-sources", "pNE"])
    assert (v.indegree, v.ext, v.step_order, v.ext_sources) == (
        "fixed", "shared", "brian2", "pNE")
    ref = run_caminos.run_filename(1, 0.07, "float32", 1e-4)
    assert ref == "counts_seed1_p0.07_float32_dt1e-04.npy", ref
    tag = run_caminos.run_filename(1, 0.07, "float32", 1e-4, "fixed", "shared",
                                   "brian2", "pNE")
    assert tag != ref and tag.endswith(".npy"), tag
    for bit in ("indeg-fixed", "ext-shared", "order-brian2", "src-pNE"):
        assert bit in tag, tag
