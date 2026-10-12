"""Tests for the AdEx conductance kernel in ``jaxfne/emitters_adex.py``.

Oracle: ``_oracle_adex_step`` is a float64 NumPy transcription of ``adex_step`` in
``artifacts/etudes/caminos/caminos_jax.py`` of the reference worktree (branch
``l2-caminos``, commit 1e768f9d; SI units, Eqs. 15-16 of Tahvili, Vinck, di Volo 2026,
PLoS Comput Biol). Its cell parameters are typed from ``PORT_SPEC.md`` Table 1 in SI
units, independently of the kernel's mV/pA table.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters_adex import (
    ADEX_CELL_TYPE_DEFAULTS,
    AdExParams,
    adex_step,
    make_adex_params,
    simulate_adex,
)

DT_MS = 0.1
DT_S = 1e-4
TREF_MS = 5.0
N_STEPS_STEP = 2200  # 20 ms at 0 pA, then 200 ms at 300 pA, dt 0.1 ms

# PORT_SPEC.md Table 1 (PLoS Comput Biol 2026), SI units.
_SI_TABLE = {
    "E": {"VT": -50e-3, "Delta": 2e-3, "Tref": 5e-3, "tauw": 500e-3, "a": 4e-9,
          "b": 50e-12, "C": 200e-12, "gL": 10e-9, "EL": -65e-3, "Vreset": -65e-3},
    "PV": {"VT": -50e-3, "Delta": 0.5e-3, "Tref": 5e-3, "tauw": 500e-3, "a": 0.0,
           "b": 0.0, "C": 200e-12, "gL": 10e-9, "EL": -65e-3, "Vreset": -65e-3},
    "SOM": {"VT": -50e-3, "Delta": 2e-3, "Tref": 5e-3, "tauw": 500e-3, "a": 4e-9,
            "b": 80e-12, "C": 200e-12, "gL": 10e-9, "EL": -55e-3, "Vreset": -65e-3},
}
_ORACLE_KEYS = ("VT", "Delta", "Tref", "tauw", "a", "b", "C", "gL", "EL", "Vreset")

# Two cells: index 0 = E, index 1 = PV. E->PV uses receptor 0 (E, 0 mV, tau 2 ms);
# PV->E uses receptor 1 (I, -80 mV, tau 8 ms). Delay 20 steps (2 ms).
NETWORK_EDGES = dict(
    edge_pre=[0, 1],
    edge_post=[1, 0],
    edge_weight_nS=[3.0, 5.0],
    edge_receptor=[0, 1],
    edge_delay_steps=[20, 20],
)
NETWORK_RECEPTORS = dict(receptor_reversal_mV=(0.0, -80.0), receptor_tau_ms=(2.0, 8.0))


@pytest.fixture(autouse=True)
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _oracle_params(labels):
    return {
        k: np.array([_SI_TABLE[lab][k] for lab in labels], dtype=np.float64)
        for k in _ORACLE_KEYS
    }


def _oracle_adex_step(V, w, ref, I_in, p, dt, step_order="default"):
    """float64 NumPy transcription of caminos_jax.adex_step (SI units, vector cells)."""
    if step_order not in ("default", "brian2"):
        raise ValueError(f"step_order must be 'default' or 'brian2', got {step_order!r}")
    v_thr = p["VT"] + 5.0 * p["Delta"]
    arg = np.minimum((V - p["VT"]) / p["Delta"], 20.0)
    dV = (p["gL"] * (p["EL"] - V) + p["gL"] * p["Delta"] * np.exp(arg) - w + I_in) / p["C"] * dt
    dw_act = (p["a"] * (V - p["EL"]) - w) / p["tauw"] * dt
    dw_ref = (p["a"] * (p["Vreset"] - p["EL"]) - w) / p["tauw"] * dt
    n_ref = np.round(p["Tref"] / dt).astype(np.int64)
    if step_order == "brian2":
        n_ref = n_ref - 1
    active = ref <= 0
    v_cand = V + dV
    spk = active & (v_cand >= v_thr)
    V_new = np.where(spk, p["Vreset"], np.where(active, v_cand, p["Vreset"]))
    w_new = np.where(active, w + dw_act + np.where(spk, p["b"], 0.0), w + dw_ref)
    ref_new = np.where(spk, n_ref, np.maximum(ref - 1, 0))
    return V_new, w_new, ref_new, spk


def _oracle_cells(labels, I_A, dt):
    p = _oracle_params(labels)
    V = p["EL"].copy()
    w = np.zeros(len(labels))
    ref = np.zeros(len(labels), dtype=np.int64)
    Vs, ws, spks = [], [], []
    for I_t in I_A:
        V, w, ref, spk = _oracle_adex_step(V, w, ref, I_t, p, dt)
        Vs.append(V)
        ws.append(w)
        spks.append(spk)
    return np.array(Vs), np.array(ws), np.array(spks)


def _oracle_network(labels, edges, n_steps, I_A, decay, step_order):
    """Hand-coded loop: decay, add arrivals, current, AdEx step, record spikes.

    Arrivals are read from a list of past spike vectors, not a ring buffer, so the
    oracle does not share the kernel's indexing.
    """
    pre, post, w_nS, rec, delay, rev_mV, tau_ms = edges
    p = _oracle_params(labels)
    rev = np.array(rev_mV, dtype=np.float64) * 1e-3
    tau = np.array(tau_ms, dtype=np.float64) * 1e-3
    decay_r = np.exp(-DT_S / tau) if decay == "exp" else 1.0 - DT_S / tau
    extra = 1 if step_order == "brian2" else 0
    lag = [d + extra for d in delay]
    N = len(labels)
    g = np.zeros((N, len(rev)))
    V = p["EL"].copy()
    w = np.zeros(N)
    ref = np.zeros(N, dtype=np.int64)
    hist, Vs, spks = [], [], []
    for t in range(n_steps):
        arrived = [
            bool(hist[t - lag[e]][pre[e]]) if t - lag[e] >= 0 else False
            for e in range(len(pre))
        ]
        g = g * decay_r
        for e in range(len(pre)):
            if arrived[e]:
                g[post[e], rec[e]] += abs(w_nS[e]) * 1e-9
        I_in = (g * (rev[None, :] - V[:, None])).sum(axis=1) + I_A[t]
        V, w, ref, spk = _oracle_adex_step(V, w, ref, I_in, p, DT_S, step_order)
        hist.append(spk)
        Vs.append(V)
        spks.append(spk)
    return np.array(Vs), np.array(spks)


def _kernel_cells(params, I_pA, step_order="default"):
    """Single cells through ``adex_step``: one scan step per row of ``I_pA``."""

    def body(state, I_t):
        state, spk = adex_step(state, I_t, params, DT_MS, step_order)
        return state, (state[0], state[1], state[2], spk)

    state0 = (
        jnp.asarray(params.EL_mV),
        jnp.zeros_like(params.EL_mV),
        jnp.zeros(params.EL_mV.shape, dtype=jnp.int32),
    )
    _, (V, w, ref, spk) = jax.lax.scan(body, state0, jnp.asarray(I_pA))
    return np.asarray(V), np.asarray(w), np.asarray(ref), np.asarray(spk)


def _step_current(n_cells):
    I_pA = np.zeros((N_STEPS_STEP, n_cells))
    I_pA[200:] = 300.0
    return I_pA


# 1. Single cells under a step current match the oracle.
def test_single_cells_match_oracle_under_step_current():
    labels = ("E", "PV", "SOM")
    I_pA = _step_current(len(labels))
    V_mV, w_pA, _, spk = _kernel_cells(make_adex_params(labels), I_pA)
    V_o, w_o, spk_o = _oracle_cells(labels, I_pA * 1e-12, DT_S)
    for j, lab in enumerate(labels):
        assert spk[:, j].sum() > 0, f"{lab} must spike under 300 pA"
    np.testing.assert_array_equal(spk, spk_o)
    np.testing.assert_allclose(V_mV, V_o * 1e3, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(w_pA, w_o * 1e12, rtol=1e-6, atol=1e-9)


# 2. Refractory hold lasts round(Tref/dt) steps by default and one fewer under brian2.
@pytest.mark.parametrize("step_order,n_hold", [("default", 50), ("brian2", 49)])
def test_refractory_hold_length(step_order, n_hold):
    assert round(TREF_MS / DT_MS) - (1 if step_order == "brian2" else 0) == n_hold
    labels = ("E", "PV", "SOM")
    params = make_adex_params(labels)
    V_mV, _, ref, spk = _kernel_cells(params, _step_current(3), step_order)
    for j, lab in enumerate(labels):
        s = int(np.argmax(spk[:, j]))
        assert spk[s, j], f"{lab} must spike under 300 pA"
        vreset = float(params.Vreset_mV[j])
        np.testing.assert_array_equal(V_mV[s + 1 : s + 1 + n_hold, j], vreset)
        assert ref[s + n_hold - 1, j] == 1
        assert ref[s + n_hold, j] == 0


# 3. Two-cell E->PV and PV->E network matches the hand-coded oracle loop.
@pytest.mark.parametrize("decay", ["exp", "euler"])
@pytest.mark.parametrize("step_order", ["default", "brian2"])
def test_two_cell_network_matches_oracle_loop(decay, step_order):
    labels = ("E", "PV")
    I_pA = _step_current(len(labels))
    out = simulate_adex(
        make_adex_params(labels),
        n_steps=N_STEPS_STEP,
        dt_ms=DT_MS,
        **NETWORK_EDGES,
        **NETWORK_RECEPTORS,
        decay=decay,
        step_order=step_order,
        I_ext_pA=I_pA,
        record_v=True,
    )
    spk = np.asarray(out["spikes"])
    V_mV = np.asarray(out["V_mV"])
    assert spk.sum(axis=0).min() > 0, "both cells must spike in the network run"
    edges = (
        [0, 1],
        [1, 0],
        [3.0, 5.0],
        [0, 1],
        [20, 20],
        (0.0, -80.0),
        (2.0, 8.0),
    )
    V_o, spk_o = _oracle_network(labels, edges, N_STEPS_STEP, I_pA * 1e-12, decay, step_order)
    np.testing.assert_array_equal(spk, spk_o)
    np.testing.assert_allclose(V_mV, V_o * 1e3, rtol=1e-6, atol=1e-9)


def _pulse_pair(pre_idx, post_idx, w_nS, rev_mV, tau_ms, t0=50, n_steps=100):
    """Same network with and without one edge; a one-step current forces the pre spike."""
    labels = ("E", "PV")
    I_pA = np.zeros((n_steps, len(labels)))
    I_pA[t0, pre_idx] = 1e5
    params = make_adex_params(labels)
    base = dict(n_steps=n_steps, dt_ms=DT_MS, I_ext_pA=I_pA, decay="exp", record_v=True)
    with_edge = simulate_adex(
        params,
        **base,
        edge_pre=[pre_idx],
        edge_post=[post_idx],
        edge_weight_nS=[w_nS],
        edge_receptor=[0],
        edge_delay_steps=[20],
        receptor_reversal_mV=(rev_mV,),
        receptor_tau_ms=(tau_ms,),
    )
    no_edge = simulate_adex(params, **base)
    return with_edge, no_edge


# 3b. Sign of the PSP: depolarizing for E->PV, hyperpolarizing for PV->E.
def test_epsp_is_depolarizing_for_E_to_PV():
    with_edge, no_edge = _pulse_pair(0, 1, 3.0, 0.0, 2.0)
    assert np.asarray(with_edge["spikes"])[50, 0], "E must spike at the forced pulse"
    V_w = np.asarray(with_edge["V_mV"])[:, 1]
    V_n = np.asarray(no_edge["V_mV"])[:, 1]
    np.testing.assert_array_equal(V_w[:70], V_n[:70])  # arrival at step 50 + 20
    assert V_w[70] > V_n[70]


def test_ipsp_is_hyperpolarizing_for_PV_to_E():
    with_edge, no_edge = _pulse_pair(1, 0, 5.0, -80.0, 8.0)
    assert np.asarray(with_edge["spikes"])[50, 1], "PV must spike at the forced pulse"
    V_w = np.asarray(with_edge["V_mV"])[:, 0]
    V_n = np.asarray(no_edge["V_mV"])[:, 0]
    np.testing.assert_array_equal(V_w[:70], V_n[:70])
    assert V_w[70] < V_n[70]


# 4. A spike at step t changes g_post first at step t + d (d + 1 under brian2).
@pytest.mark.parametrize("step_order,extra", [("default", 0), ("brian2", 1)])
def test_delay_sets_first_conductance_change(step_order, extra):
    t0, D = 50, 20
    I_pA = np.zeros((t0 + 40, 2))
    I_pA[t0, 0] = 1e5  # forces the E spike at step t0

    def run(n):
        return simulate_adex(
            make_adex_params(("E", "PV")),
            n_steps=n,
            dt_ms=DT_MS,
            edge_pre=[0],
            edge_post=[1],
            edge_weight_nS=[3.0],
            edge_receptor=[0],
            edge_delay_steps=[D],
            receptor_reversal_mV=(0.0,),
            receptor_tau_ms=(2.0,),
            step_order=step_order,
            decay="exp",
            I_ext_pA=I_pA[:n],
        )

    n_before = t0 + D + extra  # steps 0 .. t0+D+extra-1: arrival not yet read
    before = run(n_before)
    after = run(n_before + 1)
    assert np.asarray(after["spikes"])[t0, 0], "E must spike at the forced step"
    assert float(before["g"][1, 0]) == 0.0
    assert float(after["g"][1, 0]) == pytest.approx(3.0, rel=1e-12)


# 5. Two edges from one pre to one post add their conductances; |w| is used.
@pytest.mark.parametrize(
    "weights,expected",
    [([3.0], 3.0), ([3.0, 3.0], 6.0), ([-3.0], 3.0)],
)
def test_multiplicity_adds_conductance(weights, expected):
    t0, D = 5, 20
    I_pA = np.zeros((t0 + D + 1, 2))
    I_pA[t0, 0] = 1e5
    k = len(weights)
    out = simulate_adex(
        make_adex_params(("E", "PV")),
        n_steps=t0 + D + 1,
        dt_ms=DT_MS,
        edge_pre=[0] * k,
        edge_post=[1] * k,
        edge_weight_nS=weights,
        edge_receptor=[0] * k,
        edge_delay_steps=[D] * k,
        receptor_reversal_mV=(0.0,),
        receptor_tau_ms=(2.0,),
        decay="exp",
        I_ext_pA=I_pA,
    )
    assert float(out["g"][1, 0]) == pytest.approx(expected, rel=1e-12)


# 5b. Mixed delays 1..8, three receptors, many edges: equal to the independent loop.
@pytest.mark.parametrize("step_order", ["default", "brian2"])
def test_mixed_delays_and_receptors_match_oracle_loop(step_order):
    rng = np.random.default_rng(7)
    labels = ("E",) * 8 + ("PV",) * 2 + ("SOM",) * 2
    n, n_edges, n_steps = len(labels), 80, 1500
    pre = rng.integers(0, n, n_edges).tolist()
    post = rng.integers(0, n, n_edges).tolist()
    w = rng.uniform(0.5, 4.0, n_edges).tolist()
    rec = rng.integers(0, 3, n_edges).tolist()
    delay = rng.integers(1, 9, n_edges).tolist()
    assert min(delay) == 1 and max(delay) == 8, "fixture must span delays 1..8"
    assert set(rec) == {0, 1, 2}, "fixture must use all three receptors"
    rev, tau = (0.0, -80.0, -70.0), (2.0, 8.0, 20.0)
    I_pA = np.zeros((n_steps, n))
    I_pA[50:] = rng.uniform(250.0, 400.0, n)
    out = simulate_adex(
        make_adex_params(labels), n_steps=n_steps, dt_ms=DT_MS, edge_pre=pre, edge_post=post,
        edge_weight_nS=w, edge_receptor=rec, edge_delay_steps=delay,
        receptor_reversal_mV=rev, receptor_tau_ms=tau, decay="euler", step_order=step_order,
        I_ext_pA=I_pA, record_v=True,
    )
    spk = np.asarray(out["spikes"])
    assert spk.sum() > 20, "fixture must spike"
    V_o, spk_o = _oracle_network(
        labels, (pre, post, w, rec, delay, rev, tau), n_steps, I_pA * 1e-12, "euler", step_order
    )
    np.testing.assert_array_equal(spk, spk_o)
    np.testing.assert_allclose(np.asarray(out["V_mV"]), V_o * 1e3, rtol=1e-6, atol=1e-9)


# 6. Each refusal raises ValueError naming the argument.
_BASE = dict(
    n_steps=10,
    dt_ms=DT_MS,
    edge_pre=[0],
    edge_post=[1],
    edge_weight_nS=[3.0],
    edge_receptor=[0],
    edge_delay_steps=[20],
    receptor_reversal_mV=(0.0,),
    receptor_tau_ms=(2.0,),
    decay="exp",
)
REFUSALS = [
    (dict(decay="euler", receptor_tau_ms=(0.1,)), "decay='euler'"),
    (dict(edge_delay_steps=[0]), "edge_delay_steps"),
    (dict(edge_delay_steps=[2.5]), "edge_delay_steps"),
    (dict(edge_post=[1, 0]), "edge_post"),
    (dict(edge_receptor=[1]), "edge_receptor"),
    (dict(receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0, 8.0)), "receptor_reversal_mV"),
    (dict(receptor_tau_ms=(0.0,)), "receptor_tau_ms"),
    (dict(decay="linear"), "decay"),
    (dict(step_order="fancy"), "step_order"),
    (dict(edge_weight_nS=None), "edge_weight_nS"),
    (dict(edge_pre=[5]), "edge_pre"),
    (dict(I_ext_pA=np.zeros(3)), "I_ext_pA"),
    (dict(V0_mV=np.zeros(3)), "V0_mV"),
    (dict(n_steps=0), "n_steps"),
    (dict(dt_ms=0.0), "dt_ms"),
]


@pytest.mark.parametrize("override,name", REFUSALS)
def test_refusals_name_the_argument(override, name):
    with pytest.raises(ValueError, match=name):
        simulate_adex(make_adex_params(("E", "PV")), **{**_BASE, **override})


# 7. Unknown cell types and override fields raise ValueError; overrides are consumed.
def test_unknown_cell_type_raises_and_names_it():
    with pytest.raises(ValueError, match="'X'"):
        make_adex_params(("E", "X"))


def test_unknown_override_field_raises_and_names_it():
    with pytest.raises(ValueError, match="gain"):
        make_adex_params(("E",), {"E": {"gain": 2.0}})


def test_override_for_unknown_cell_type_raises():
    with pytest.raises(ValueError, match="'X'"):
        make_adex_params(("E",), {"X": {"C_pF": 1.0}})


def test_override_is_consumed_including_zero():
    p = make_adex_params(("E", "PV"), {"PV": {"C_pF": 100.0}, "E": {"b_pA": 0.0}})
    np.testing.assert_array_equal(np.asarray(p.C_pF), [200.0, 100.0])
    np.testing.assert_array_equal(np.asarray(p.b_pA), [0.0, 0.0])


def test_defaults_table():
    assert set(ADEX_CELL_TYPE_DEFAULTS) == {"E", "PV", "SOM"}
    p = make_adex_params(("E", "PV", "SOM"))
    assert isinstance(p, AdExParams)
    assert p.labels == ("E", "PV", "SOM")
    np.testing.assert_array_equal(np.asarray(p.EL_mV), [-65.0, -65.0, -55.0])


def test_dtype_follows_params():
    jax.config.update("jax_enable_x64", False)
    try:
        params = make_adex_params(("E",))
        out = simulate_adex(params, n_steps=5, dt_ms=DT_MS, decay="exp", record_v=True)
        assert out["V_mV"].dtype == np.float32
    finally:
        jax.config.update("jax_enable_x64", True)
