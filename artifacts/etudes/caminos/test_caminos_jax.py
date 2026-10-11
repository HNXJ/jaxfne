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
