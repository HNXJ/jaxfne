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
    api.write_run(tmp_path, p=0.07, seed=1, T_s=0.3, counts=counts, sizes=sizes,
                  wall_s=0.0, start_time="t", sha="a" * 40)
    result = ca.score(str(tmp_path), ext_sources="pNE")
    assert isinstance(result, dict), result
    assert "counts_seed1_p0.07_float32_dt1e-04.npy" not in result["not_scored"], result
