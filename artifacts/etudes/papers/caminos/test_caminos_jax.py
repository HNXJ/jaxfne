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

_repo_root = Path(__file__).resolve().parents[4]
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


# --- P6 library-path gates (PREREG_P6.md, B0 route (c)) ---

def _p6api():
    """Import caminos_api (function-level so each gate fails on the baseline)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import caminos_api

    return caminos_api


@pytest.fixture(scope="module")
def _p6_model():
    api = _p6api()
    import jaxfne as jtfne

    return api, jtfne.construct(api.build_spec(0.07, 1))


def test_p6_gate1_parameter_identity(_p6_model):
    """Gate 1: every PORT_SPEC table value read back from the realized model, exact."""
    api, model = _p6_model
    from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args

    labels = tuple(model.params["emitter"].labels)
    assert len(labels) == 9500, len(labels)
    assert {t: labels.count(t) for t in ("E", "PV", "SOM")} == {
        "E": 8000, "PV": 750, "SOM": 750}
    first = {t: labels.index(t) for t in ("E", "PV", "SOM")}
    # PORT_SPEC Table 1 in jaxfne units (mV, ms, pF, nS, pA).
    want = {
        "E": {"C_pF": 200.0, "gL_nS": 10.0, "EL_mV": -65.0, "VT_mV": -50.0,
              "Delta_mV": 2.0, "Vreset_mV": -65.0, "tauw_ms": 500.0,
              "a_nS": 4.0, "b_pA": 50.0, "Tref_ms": 5.0},
        "PV": {"C_pF": 200.0, "gL_nS": 10.0, "EL_mV": -65.0, "VT_mV": -50.0,
               "Delta_mV": 0.5, "Vreset_mV": -65.0, "tauw_ms": 500.0,
               "a_nS": 0.0, "b_pA": 0.0, "Tref_ms": 5.0},
        "SOM": {"C_pF": 200.0, "gL_nS": 10.0, "EL_mV": -55.0, "VT_mV": -50.0,
                "Delta_mV": 2.0, "Vreset_mV": -65.0, "tauw_ms": 500.0,
                "a_nS": 4.0, "b_pA": 80.0, "Tref_ms": 5.0},
    }
    for ctype, fields in want.items():
        for field, value in fields.items():
            got = float(np.asarray(getattr(model.params["emitter"], field))[first[ctype]])
            assert got == value, (ctype, field, got, value)
    # PORT_SPEC Table 2: receptors, weights, delay.
    st = model.static["adex"]
    assert tuple(st["receptor_tau_ms"]) == (2.0, 8.0), st["receptor_tau_ms"]
    assert tuple(st["receptor_reversal_mV"]) == (0.0, -80.0)
    assert tuple(st["receptor_names"]) == ("ampa", "gaba")
    rules = {r["name"]: r for r in model.cfg.metadata["circuit"]["connections"]}
    assert sorted(rules) == ["e_e", "e_pv", "e_som", "pv_e", "pv_pv", "som_e", "som_pv"]
    want_rules = {
        "e_e": ("E", "E", 2.0, "ampa"), "e_pv": ("E", "PV", 3.0, "ampa"),
        "e_som": ("E", "SOM", 3.0, "ampa"), "pv_e": ("PV", "E", 5.0, "gaba"),
        "pv_pv": ("PV", "PV", 5.0, "gaba"), "som_e": ("SOM", "E", 5.0, "gaba"),
        "som_pv": ("SOM", "PV", 5.0, "gaba"),
    }
    for name, (pre, post, weight, mech) in want_rules.items():
        r = rules[name]
        assert r["source"] == {"cell_type": pre}, (name, r["source"])
        assert r["target"] == {"cell_type": post}, (name, r["target"])
        assert r["probability"] == 0.07, (name, r["probability"])
        assert r["weight"] == weight, (name, r["weight"])
        assert r["mechanism"] == mech, (name, r["mechanism"])
        assert r["delay_ms"] == 2.0, (name, r["delay_ms"])
        assert r["sampling"] == "bernoulli", (name, r.get("sampling"))
    assert int(st["edge_seed"]) == 1, st["edge_seed"]
    # Drive: round(p*NE) sources at 4 Hz to E and PV, seeds from seed 1.
    sim = api.build_sim(0.07, 1, 0.5)
    assert sim.dt_ms == 0.1 and sim.seed == 0, (sim.dt_ms, sim.seed)
    drive = sim.poisson_drive
    assert drive == {"n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5,
                     "mechanism": "ampa", "target_cell_types": ["E", "PV"],
                     "delay_ms": 2.0, "seed": 1}, drive
    dk = adex_external_drive_kwargs(model, sim)
    assert dk["ext_lambda"] == 560 * 4.0 * 0.1 / 1000.0, dk["ext_lambda"]
    assert dk["ext_weight_nS"] == 2.5 and dk["ext_receptor"] == 0, dk
    assert dk["ext_delay_steps"] == 20, dk["ext_delay_steps"]
    assert np.asarray(dk["ext_targets"]).tolist() == list(range(8750)), (
        len(dk["ext_targets"]))
    kw = adex_kernel_args(model, sim.dt_ms)
    assert kw["decay"] == "euler" and kw["step_order"] == "default", (
        kw["decay"], kw["step_order"])
    assert bool((np.asarray(kw["edge_delay_steps"]) == 20).all()), (
        np.unique(kw["edge_delay_steps"]))


def test_p6_gate2_realized_connectivity(_p6_model):
    """Gate 2: at p 0.07 seed 1 each block within 4 SD, no autapses, no PV->SOM/SOM->SOM."""
    _, model = _p6_model
    st = model.static["adex"]
    labels = np.asarray(model.params["emitter"].labels)
    pre = np.asarray(st["pre"])
    post = np.asarray(st["post"])
    assert pre.size > 0, "fixture must realize edges"
    assert not bool((pre == post).any()), "autapses realized"
    pairs = set(zip(labels[pre].tolist(), labels[post].tolist()))
    assert ("PV", "SOM") not in pairs and ("SOM", "SOM") not in pairs, pairs
    sizes = {"E": 8000, "PV": 750, "SOM": 750}
    blocks = [("E", "E"), ("E", "PV"), ("E", "SOM"), ("PV", "E"),
              ("PV", "PV"), ("SOM", "E"), ("SOM", "PV")]
    got = {(a, b): int(((labels[pre] == a) & (labels[post] == b)).sum())
           for a, b in blocks}
    assert sum(got.values()) == pre.size, "every edge must sit in one of the seven blocks"
    for (a, b), count in got.items():
        n = sizes[a] * sizes[b]
        mean = 0.07 * n
        sd = (n * 0.07 * 0.93) ** 0.5
        assert abs(count - mean) <= 4 * sd, (a, b, count, mean, sd)


def test_p6_route_c_identity():
    """B0 route (c): record_v=False spikes equal model.simulate() spikes exactly."""
    api = _p6api()
    import jaxfne as jtfne
    from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args
    from jaxfne.emitters_adex import simulate_adex

    cfg = (
        jtfne.Configuration()
        .network(name="caminos-small", n=60,
                 cell_types={"E": 0.8, "PV": 0.1, "SOM": 0.1})
        .emitter(family="adex", adex_decay="euler", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": api.TAU_E_MS, "reversal_mV": api.REV_E_MV})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": api.TAU_I_MS, "reversal_mV": api.REV_I_MV})
    )
    for name, pre, post, weight, mechanism in api.RULES:
        cfg = cfg.connections(
            name=name, source={"cell_type": pre}, target={"cell_type": post},
            probability=0.2, weight=weight, mechanism=mechanism,
            delay_ms=api.DELAY_MS, sampling="bernoulli")
    cfg = cfg.connectivity(edge_seed=3)
    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=api.DT_MS,
                           poisson_drive=api.drive_spec(0.2, 3))
    signals = model.simulate(sim)
    assert int(np.asarray(signals.spikes).sum()) > 0, "fixture must spike"
    runtime_cfg = sim.resolved_runtime
    kwargs = adex_kernel_args(model, sim.dt_ms, dtype=runtime_cfg.actual_dtype)
    drive_kwargs = adex_external_drive_kwargs(model, sim)
    out = simulate_adex(**kwargs, **drive_kwargs, n_steps=sim.n_steps,
                        dt_ms=sim.dt_ms, record_v=False)
    assert "V_mV" not in out, sorted(out)
    np.testing.assert_array_equal(np.asarray(out["spikes"]),
                                  np.asarray(signals.spikes))


def test_p6_c1_fixed_drive_reaches_the_kernel(_p6_model):
    """C1 control: --ext-sources fixed gives the kernel 560 sources' rate (pNE gives p*NE)."""
    api, model = _p6_model
    from jaxfne._construct_adex import adex_external_drive_kwargs

    # ext_lambda = n_sources * rate_hz * dt_ms / 1000 = n_sources * 4e-4
    low = adex_external_drive_kwargs(model, api.build_sim(0.02, 1, 0.01, "pNE"))
    fixed = adex_external_drive_kwargs(model, api.build_sim(0.02, 1, 0.01, "fixed"))
    assert float(np.asarray(low["ext_lambda"]).ravel()[0]) == pytest.approx(160 * 4e-4, rel=1e-6)
    assert float(np.asarray(fixed["ext_lambda"]).ravel()[0]) == pytest.approx(560 * 4e-4, rel=1e-6)
    with pytest.raises(ValueError):
        api.drive_spec(0.02, 1, "other")


def test_p6_gate4_converter_known_answer(tmp_path):
    """Gate 4: synthetic raster -> per-step sums; score() reads the converted run."""
    api = _p6api()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import caminos_analysis as ca

    rng = np.random.default_rng(0)
    raster = rng.random((3000, 60)) < 0.01
    raster[10, :] = False
    raster[10, 0] = True
    raster[10, 40] = True
    raster[10, 50] = True
    sizes = {"E": 40, "PV": 10, "SOM": 10}
    counts = api.spikes_to_counts(raster, sizes)
    assert counts.dtype == np.int32 and counts.shape == (3000, 3), (
        counts.dtype, counts.shape)
    assert counts[10].tolist() == [1, 1, 1], counts[10].tolist()
    np.testing.assert_array_equal(
        counts,
        np.stack([raster[:, :40].sum(axis=1), raster[:, 40:50].sum(axis=1),
                  raster[:, 50:].sum(axis=1)], axis=1).astype(np.int32))
    # Full-length synthetic runs (12 s): score() analyses them, and two seeds merge.
    full = np.tile(counts, (40, 1))
    assert full.shape[0] == ca.REF_STEPS, "fixture must be 12 s at dt 1e-4"
    for seed in (1, 2):
        api.write_run(tmp_path, p=0.07, seed=seed, T_s=12.0, counts=full, sizes=sizes,
                      wall_s=0.0, start_time="t", sha="a" * 40)
    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert sorted(e["seed"] for e in manifest["runs"]) == [1, 2]
    result = ca.score(str(tmp_path), ext_sources="pNE")
    analysed = sorted(r["seed"] for r in result["per_run"])
    assert analysed == [1, 2], result["per_run"]
    assert result["not_scored"] == [], result["not_scored"]
    with pytest.raises(SystemExit):  # same file again
        api.write_run(tmp_path, p=0.07, seed=1, T_s=12.0, counts=full, sizes=sizes,
                      wall_s=0.0, start_time="t", sha="a" * 40)
    with pytest.raises(SystemExit):  # another code SHA into the same directory
        api.write_run(tmp_path, p=0.07, seed=3, T_s=12.0, counts=full, sizes=sizes,
                      wall_s=0.0, start_time="t", sha="b" * 40)


# --- Cell Reports 2025 parameter set (CELLREP_PARAMS, notebook cell 1) ---

_PARAMS_A508B24C = {
    "E_VT_V": -50e-3,
    "E_Delta_V": 2e-3,
    "E_Tref_s": 5e-3,
    "E_tauw_s": 500e-3,
    "E_a_S": 4e-9,
    "E_b_A": 50e-12,
    "E_C_F": 200e-12,
    "E_gL_S": 10e-9,
    "E_EL_V": -65e-3,
    "E_Vreset_V": -65e-3,
    "PV_VT_V": -50e-3,
    "PV_Delta_V": 0.5e-3,
    "PV_Tref_s": 5e-3,
    "PV_tauw_s": 500e-3,
    "PV_a_S": 0.0,
    "PV_b_A": 0.0,
    "PV_C_F": 200e-12,
    "PV_gL_S": 10e-9,
    "PV_EL_V": -65e-3,
    "PV_Vreset_V": -65e-3,
    "SOM_VT_V": -50e-3,
    "SOM_Delta_V": 2e-3,
    "SOM_Tref_s": 5e-3,
    "SOM_tauw_s": 500e-3,
    "SOM_a_S": 4e-9,
    "SOM_b_A": 80e-12,
    "SOM_C_F": 200e-12,
    "SOM_gL_S": 10e-9,
    "SOM_EL_V": -55e-3,
    "SOM_Vreset_V": -65e-3,
    "tau_d_s": 2e-3,
    "tauE_s": 2e-3,
    "tauI_s": 8e-3,
    "ErevE_V": 0.0,
    "ErevI_V": -80e-3,
    "Q_PV_E_S": 5e-9,
    "Q_SOM_E_S": 5e-9,
    "Q_PV_PV_S": 5e-9,
    "Q_SOM_PV_S": 5e-9,
    "Q_E_E_S": 2e-9,
    "Q_E_PV_S": 3e-9,
    "Q_E_SOM_S": 3e-9,
    "Q_PV_SOM_S": 0.0,
    "Q_SOM_SOM_S": 0.0,
    "Q_ext_S": 2.5e-9,
    "NE": 8000,
    "NPV": 750,
    "NSOM": 750,
    "p": 0.07,
    "ext_sources": 560,
    "ext_rate_Hz": 4.0,
    "dt_s": 1e-4,
}


def test_cellrep_params_match_notebook():
    """CELLREP_PARAMS pins every oracle value from the authors' notebook
    cell 1 (each message names the notebook cell; exact line numbers were
    unreadable from this sandbox, so all cite cell 1)."""
    P = cj.CELLREP_PARAMS
    dt = float(cj.CELLREP_PARAMS["dt_s"])
    assert P["NE"] == 8000, "nb cell 1: NE"
    assert P["NPV"] == 800, "nb cell 1: NPV"
    assert P["NSOM"] == 600, "nb cell 1: NSOM nominal 600"
    assert P["p"] == 0.017, "nb cell 1: p"
    assert P["p_E_PV_mult"] == 2.0, "nb cell 1: E->PV at 2p"
    assert P["Q_PV_E_S"] == 6e-9, "nb cell 1: Q PV -> E"
    assert P["Q_SOM_E_S"] == 5e-9, "nb cell 1: Q SOM -> E"
    assert P["Q_PV_PV_S"] == 6e-9, "nb cell 1: Q PV -> PV"
    assert P["Q_SOM_PV_S"] == 5e-9, "nb cell 1: Q SOM -> PV"
    assert P["Q_E_E_S"] == 1.25e-9, "nb cell 1: Q E -> E"
    assert P["Q_E_PV_S"] == 3.75e-9, "nb cell 1: Q E -> PV"
    assert P["Q_E_SOM_S"] == 2.5e-9, "nb cell 1: Q E -> SOM"
    assert P["Q_PV_SOM_S"] == 0.0, "nb cell 1: Q PV -> SOM absent"
    assert P["Q_SOM_SOM_S"] == 0.0, "nb cell 1: Q SOM -> SOM absent"
    assert P["Q_ext_S"] == 1.25e-9, "nb cell 1: Q Ext -> E, PV"
    assert P["tauE_E_s"] == 5e-3, "nb cell 1: excitatory decay onto E"
    assert P["tauE_PV_s"] == 1e-3, "nb cell 1: excitatory decay onto PV"
    assert P["tauE_SOM_s"] == 2e-3, "nb cell 1: excitatory decay onto SOM"
    assert P["tauI_s"] == 5e-3, "nb cell 1: GABA decay"
    assert "tauE_s" not in P, "nb cell 1: no single excitatory decay"
    want_adex = {
        "E": {"EL_V": -60e-3, "Delta_V": 2e-3, "a_S": 4e-9, "b_A": 130e-12},
        "PV": {"EL_V": -60e-3, "Delta_V": 0.5e-3, "a_S": 0.0, "b_A": 0.0},
        "SOM": {"EL_V": -55e-3, "Delta_V": 1.5e-3, "a_S": 4e-9, "b_A": 25e-12},
    }
    for kind, fields in want_adex.items():
        for field, value in fields.items():
            assert P[kind + "_" + field] == value, ("nb cell 1: Table 1", kind, field)
    for kind in ("E", "PV", "SOM"):
        assert P[kind + "_VT_V"] == -50e-3, ("nb cell 1: Table 1 VT", kind)
        assert P[kind + "_Tref_s"] == 5e-3, ("nb cell 1: Table 1 Tref", kind)
        assert P[kind + "_tauw_s"] == 500e-3, ("nb cell 1: Table 1 tau_w", kind)
        assert P[kind + "_C_F"] == 200e-12, ("nb cell 1: Table 1 C", kind)
        assert P[kind + "_gL_S"] == 10e-9, ("nb cell 1: Table 1 gL", kind)
        assert P[kind + "_Vreset_V"] == -65e-3, ("nb cell 1: Table 1 Vreset", kind)
    assert P["ErevI_V"] == -80e-3, "nb cell 1: E_rev inhibitory"
    assert P["ErevE_V"] == 0.0, "nb cell 1: E_rev excitatory"
    assert P["tau_d_s"] == 0.0, "nb cell 1: no synaptic delay"
    assert cj.delay_steps(dt, "default", "cellrep") == 1, "nb cell 1: zero delay -> 1-step ring"
    assert P["ext_sources"] == 136 == int(0.017 * 8000), "nb cell 1: N_ext = int(p * NE)"
    assert P["ext_rate_Hz"] == 4.0, "nb cell 1: 4 Hz to E and PV"
    assert cj.ext_source_count(0.017, 8000, "fixed", "cellrep") == 136, "nb cell 1: source count"
    assert cj.external_lam(0.017, 8000, dt, "fixed", "cellrep") == 136 * 4.0 * dt, (
        "nb cell 1: drive lam")
    assert cj.cell_params("E", "cellrep")["EL_V"] == -60e-3, "nb cell 1: EL (E)"
    assert cj.cell_params("SOM", "cellrep")["Delta_V"] == 1.5e-3, "nb cell 1: Delta (SOM)"
    net = cj.build_network(jax.random.key(11), p=0.3, NE=20, NPV=6, NSOM=6,
                           indegree="fixed", param_set="cellrep")
    assert ("PV", "SOM") not in net["blocks"], "nb cell 1: PV->SOM absent"
    assert ("SOM", "SOM") not in net["blocks"], "nb cell 1: SOM->SOM absent"
    assert (net["NE"], net["NPV"], net["NSOM"]) == (20, 6, 6), "cellrep sizes pass through"
    ee = np.asarray(net["blocks"][("E", "E")])
    epv = np.asarray(net["blocks"][("E", "PV")])
    assert bool((ee.sum(axis=0) == 6).all()), "nb cell 1: E->E at p (round(0.3*20)=6)"
    assert bool((epv.sum(axis=0) == 12).all()), "nb cell 1: E->PV at 2p (round(0.6*20)=12)"


def test_cellrep_default_unchanged():
    """The default set equals its a508b24c values, pinned literally."""
    assert cj.PARAMS == _PARAMS_A508B24C, "default PARAMS must equal a508b24c"
    assert cj._params_for("default") is cj.PARAMS, "default selection is PARAMS"
    assert cj._params_for("cellrep") is cj.CELLREP_PARAMS, "cellrep selection is CELLREP_PARAMS"
    assert "tauE_s" in cj.PARAMS, "default keeps its single excitatory decay"
    dt = float(cj.PARAMS["dt_s"])
    assert cj.delay_steps(dt) == 20, "default delay ring still 20 steps"
    assert cj.ext_source_count(0.07, 8000, "fixed") == 560, "default source count still 560"
    net = cj.build_network(jax.random.key(0), NE=40, NPV=8, NSOM=8)
    assert (net["NE"], net["NPV"], net["NSOM"], net["p"]) == (40, 8, 8, 0.07), "default sizes/p"
    assert ("PV", "SOM") not in net["blocks"] and ("SOM", "SOM") not in net["blocks"], (
        "default absent blocks")


# --- Opt-in manipulations: silencing, spike replacement, PV/SOM swap (MANIP) ---

MANIP_KW = dict(p=0.07, NE=200, NPV=40, NSOM=40)
MANIP_SEED = 3
# Baseline per-step counts [steps, 3] recorded from the UNEDITED file at
# 878160ae: cj.simulate(jax.random.key(11), 0.05, p=0.07, NE=200, NPV=40,
# NSOM=40) on the unedited file (x64 on, as this test module; sums [441, 135, 53]).
_MANIP_PIN = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 2, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0], [0, 0, 0], [1, 1, 0], [1, 2, 0], [2, 1, 0], [4, 0, 0], [4, 0, 0], [1, 0, 0], [1, 2, 0], [4, 1, 0], [2, 0, 0], [1, 0, 0], [3, 1, 0], [4, 2, 0], [2, 0, 0], [5, 1, 0], [0, 1, 0], [3, 2, 0], [3, 1, 0], [5, 1, 0], [7, 0, 0], [3, 1, 0], [4, 0, 0], [2, 0, 0], [3, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0], [4, 1, 0], [4, 0, 0], [5, 2, 0], [2, 1, 0], [3, 0, 0], [6, 2, 0], [4, 0, 0], [4, 0, 0], [1, 0, 0], [1, 0, 0], [6, 1, 0], [6, 0, 0], [2, 0, 0], [2, 2, 0], [3, 1, 0], [2, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [4, 0, 0], [1, 0, 0], [1, 0, 0], [5, 1, 0], [1, 1, 0], [2, 0, 0], [5, 0, 0], [1, 0, 0], [3, 0, 0], [2, 1, 0], [2, 1, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0], [3, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 3], [2, 0, 0], [2, 1, 1], [2, 0, 1], [1, 0, 1], [2, 0, 1], [1, 0, 2], [1, 0, 0], [0, 0, 0], [1, 0, 3], [0, 1, 3], [0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 2], [0, 0, 2], [1, 0, 1], [0, 1, 2], [0, 1, 2], [1, 0, 1], [0, 2, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [2, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [3, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [3, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 0], [2, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0], [3, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [3, 0, 0], [0, 0, 0], [1, 1, 0], [2, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [3, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0], [2, 0, 0], [2, 1, 0], [2, 0, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0], [1, 1, 0], [2, 1, 0], [2, 0, 0], [2, 1, 0], [0, 0, 0], [0, 2, 0], [1, 1, 0], [2, 1, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [3, 0, 0], [3, 1, 0], [1, 0, 0], [3, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [3, 0, 0], [0, 0, 0], [1, 1, 0], [3, 0, 0], [0, 0, 0], [1, 2, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [4, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 0, 0], [1, 2, 0], [0, 0, 0], [0, 0, 0], [5, 0, 0], [4, 0, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0], [2, 1, 0], [2, 0, 0], [2, 0, 0], [0, 0, 0], [4, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [2, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], [2, 2, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [1, 2, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0], [3, 0, 0], [2, 0, 0], [0, 0, 0], [4, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0], [0, 0, 0], [4, 2, 0], [2, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 2, 1], [2, 0, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1], [3, 0, 0], [0, 0, 1], [0, 0, 1], [2, 1, 1], [0, 1, 0], [2, 1, 0], [0, 1, 0], [3, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [2, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 2, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 2], [2, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 1, 0]]


def test_default_unchanged():
    """MANIP: the default path (silence/replace/swap all off) still gives
    the per-step counts pinned from the unedited file at 878160ae."""
    counts, _ = cj.simulate(jax.random.key(11), 0.05, **MANIP_KW)
    assert np.asarray(counts).tolist() == _MANIP_PIN


def test_silence_pv_all_no_pv_spikes():
    """MANIP silencing: all PV cells silenced -> zero PV spikes, E active."""
    counts, _ = cj.simulate(jax.random.key(MANIP_SEED), 0.2, silence={"PV": 1.0},
                            **MANIP_KW)
    c = np.asarray(counts)
    assert int(c[:, 0].sum()) > 0, "fixture precondition: E must spike"
    assert int(c[:, 1].sum()) == 0, int(c[:, 1].sum())


def test_silence_fraction_counts():
    """MANIP silencing: round(0.5*40)=20 SOM cells silenced (via the exposed
    mask helper); fully silenced cells emit no spikes; half-silencing cuts
    the SOM output but does not remove it."""
    sizes = {"E": 200, "PV": 40, "SOM": 40}
    mask = cj.manip_mask(jax.random.key(MANIP_SEED), {"SOM": 0.5}, sizes, "silence")
    m = np.asarray(mask)
    assert int(m.sum()) == 20, int(m.sum())
    assert int(m[:240].sum()) == 0, "only SOM cells may be masked"
    c0, _ = cj.simulate(jax.random.key(MANIP_SEED), 0.2, **MANIP_KW)
    c1, _ = cj.simulate(jax.random.key(MANIP_SEED), 0.2, silence={"SOM": 0.5}, **MANIP_KW)
    c2, _ = cj.simulate(jax.random.key(MANIP_SEED), 0.2, silence={"SOM": 1.0}, **MANIP_KW)
    s0 = int(np.asarray(c0)[:, 2].sum())
    s1 = int(np.asarray(c1)[:, 2].sum())
    s2 = int(np.asarray(c2)[:, 2].sum())
    assert s0 > 0, "fixture precondition: SOM spikes without silencing"
    assert s2 == 0, "fully silenced cells emit no spikes"
    assert 0 < s1 < s0, (s0, s1, s2)


def test_replace_rate_matches_target():
    """MANIP replacement: all PV spikes replaced by Poisson trains at
    20 Hz per cell; population rate within 15% of the target."""
    counts, _ = cj.simulate(jax.random.key(5), 2.0, replace={"PV": 1.0},
                            replace_rate_hz={"PV": 20.0}, **MANIP_KW)
    c = np.asarray(counts)
    rate = float(c[:, 1].sum()) / (2.0 * 40)  # spikes/s per PV cell
    assert rate > 0, rate
    assert abs(rate - 20.0) <= 0.15 * 20.0, rate


def test_replace_requires_rate():
    """MANIP validation: replace without replace_rate_hz, a fraction outside
    [0, 1] and an unknown type key are all refused with ValueError."""
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(5), 0.05, replace={"PV": 0.5}, **MANIP_KW)
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(5), 0.05, replace={"PV": 1.5},
                    replace_rate_hz={"PV": 20.0}, **MANIP_KW)
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(5), 0.05, silence={"E": 0.5}, **MANIP_KW)


def test_swap_exchanges_pv_som_params():
    """MANIP swap: the per-neuron arrays built through the swap path give PV
    cells the SOM values and SOM cells the PV values for the seven Table 1
    numbers (VT, EL, Delta, Tref, tau_w, a, b); Vreset, C, gL keep their own
    type; the kwarg runs end to end."""
    sizes = {"E": 4, "PV": 3, "SOM": 5}
    par = cj._concat_params(sizes, jnp.float32, "default", swap=True)
    par0 = cj._concat_params(sizes, jnp.float32, "default")
    lo, hi = 4, 7  # PV block is rows 4:7, SOM block rows 7:12
    for name in ("VT_V", "EL_V", "Delta_V", "Tref_s", "tauw_s", "a_S", "b_A"):
        pv = np.asarray(par[name])[lo:hi]
        som = np.asarray(par[name])[hi:]
        assert np.all(pv == cj.PARAMS["SOM_" + name]), (name, pv)
        assert np.all(som == cj.PARAMS["PV_" + name]), (name, som)
    for name in ("Vreset_V", "C_F", "gL_S"):
        pv = np.asarray(par[name])[lo:hi]
        som = np.asarray(par[name])[hi:]
        assert np.all(pv == cj.PARAMS["PV_" + name]), (name, pv)
        assert np.all(som == cj.PARAMS["SOM_" + name]), (name, som)
        assert np.array_equal(np.asarray(par0[name]), np.asarray(par[name])), name
    for name in ("EL_V", "Delta_V", "a_S", "b_A"):  # types differ on these four
        assert not np.array_equal(np.asarray(par0[name]), np.asarray(par[name])), name
    c, _ = cj.simulate(jax.random.key(1), 0.02, NE=40, NPV=8, NSOM=8, swap=True)
    assert np.asarray(c).shape == (200, 3)


# --- LANE-A opt-in Cell Reports mechanisms (record, onset, coupling, SOM drive, schedule) ---

LANEA_KW = dict(p=0.07, NE=40, NPV=8, NSOM=8)
LANEA_SEED = 7
LANEA_T = 0.2
LANEA_STEPS = 2000
LANEA_NTOT = 56


def test_record_raster_matches_counts():
    """A1: record='raster' returns the boolean spike raster of shape
    (steps, NE+NPV+NSOM); its per-type sums equal the counts run for the
    same key and parameters; other record values are refused."""
    c, f0 = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, **LANEA_KW)
    r, f1 = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, record="raster",
                        **LANEA_KW)
    c, r = np.asarray(c), np.asarray(r)
    assert r.dtype == bool and r.shape == (LANEA_STEPS, LANEA_NTOT), (r.dtype, r.shape)
    assert [int(r[:, :40].sum()), int(r[:, 40:48].sum()), int(r[:, 48:].sum())] == (
        c.sum(axis=0).tolist())
    np.testing.assert_array_equal(np.asarray(f1["V"]), np.asarray(f0["V"]))
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, record="spikes", **LANEA_KW)


def test_silence_onset_delays_mask():
    """A2: with silence={'PV': 1.0} and onset 0.1 s, PV spikes before step
    round(0.1/dt) = 1000 are kept (identical to the no-silence run) and PV
    spikes after it are 0; an onset without silence is refused."""
    onset_step = int(round(0.1 / float(cj.PARAMS["dt_s"])))
    assert onset_step == 1000, onset_step
    c0, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, **LANEA_KW)
    assert int(np.asarray(c0)[:onset_step, 1].sum()) > 0, (
        "precondition: PV spikes before the onset without silence")
    c1, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T,
                        silence={"PV": 1.0}, silence_onset_s=0.1, **LANEA_KW)
    c1 = np.asarray(c1)
    assert c1[:onset_step].tolist() == np.asarray(c0)[:onset_step].tolist()
    assert int(c1[:onset_step, 1].sum()) > 0
    assert int(c1[onset_step:, 1].sum()) == 0
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, silence_onset_s=0.1,
                    **LANEA_KW)


def test_coupling_changes_som_spikes():
    """A3: PV->SOM at 2 nS changes the SOM count; zero weights reproduce the
    default run bit-for-bit (connectivity and RNG untouched); unknown keys
    are refused."""
    kw = dict(p=0.07, NE=200, NPV=40, NSOM=40)
    c0, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, **kw)
    assert int(np.asarray(c0)[:, 2].sum()) > 0, "precondition: default SOM spikes"
    c1, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T,
                        coupling={"PV_SOM": 2e-9}, **kw)
    assert int(np.asarray(c1)[:, 2].sum()) != int(np.asarray(c0)[:, 2].sum())
    c2, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T,
                        coupling={"PV_SOM": 0.0}, **kw)
    np.testing.assert_array_equal(np.asarray(c2), np.asarray(c0))
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, coupling={"E_E": 1e-9},
                    **kw)


def test_som_ext_drive_raises_som_spikes():
    """A4: a 4 Hz SOM drive gives more SOM spikes than 0 Hz with the same
    key; 0 Hz reproduces the default run bit-for-bit."""
    c0, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, **LANEA_KW)
    s0 = int(np.asarray(c0)[:, 2].sum())
    assert s0 >= 0, "precondition: 0 Hz SOM count is defined"
    c0b, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, som_ext_rate_hz=0.0,
                         **LANEA_KW)
    np.testing.assert_array_equal(np.asarray(c0b), np.asarray(c0))
    c4, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, som_ext_rate_hz=4.0,
                        **LANEA_KW)
    assert int(np.asarray(c4)[:, 2].sum()) > s0, (
        s0, int(np.asarray(c4)[:, 2].sum()))
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, som_ext_rate_hz=-1.0,
                    **LANEA_KW)


def test_ext_rate_schedule_zero_silences_feedforward():
    """A5: an all-zero schedule with no other drive leaves E and PV silent
    (the constant 4 Hz run drives E); a wrong-length schedule is refused."""
    c0, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T, **LANEA_KW)
    assert int(np.asarray(c0)[:, 0].sum()) > 0, "precondition: E spikes at 4 Hz"
    cz, _ = cj.simulate(jax.random.key(LANEA_SEED), LANEA_T,
                        ext_rate_schedule=jnp.zeros((LANEA_STEPS,)), **LANEA_KW)
    cz = np.asarray(cz)
    assert int(cz[:, 0].sum()) == 0, int(cz[:, 0].sum())
    assert int(cz[:, 1].sum()) == 0, int(cz[:, 1].sum())
    with pytest.raises(ValueError):
        cj.simulate(jax.random.key(LANEA_SEED), LANEA_T,
                    ext_rate_schedule=jnp.zeros((LANEA_STEPS + 1,)), **LANEA_KW)
