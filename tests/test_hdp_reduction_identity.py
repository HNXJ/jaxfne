"""C5 (S5): H/HDP reduction identity for every core registered HDP rule.

At zero weight drive a rule reduces to the HDP-free kernel: spikes are identical
and weights stay at their initial values. With an all-False plasticity mask the
weights stay fixed. With nonzero parameters at least one weight changes (the
falsifier). Izhikevich rules run on the edge-list kernel; istdp_vogels2011 runs
on the AdEx kernel.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne
from jaxfne import hdp_rules_pcl
from jaxfne._hdp_registrable_kernel import (
    simulate_edge_recurrent_izhikevich_hdp_registered,
)
from jaxfne.emitters import EdgeList, IzhikevichParams, simulate_edge_recurrent_izhikevich
from jaxfne.emitters_adex import make_adex_params, simulate_adex, simulate_adex_hdp_registered
from jaxfne.hdp_rule import get_hdp_rule, list_registered_hdp_rules
from jaxfne.istdp_vogels2011 import ISTDP_RULE_NAME

assert "site-packages" not in jaxfne.__file__, (
    "imported jaxfne is the installed copy; run pytest from the worktree root"
)

# Izhikevich fixture: 12 cells, 3 seeded out-edges each. Weights are dyadic so the
# float32 carry stores them exactly. Regular-spiking values follow
# tests/test_plasticity_controls_053.py.
IZH_N_NEURONS = 12
IZH_N_STEPS = 300
IZH_DT_MS = 1.0
IZH_DRIVE_LO = 5.0  # input to the weakest cell, native units
IZH_DRIVE_HI = 7.0  # input to the strongest cell; the ramp staggers spike times
IZH_OUT_DEGREE = 3
IZH_WEIGHT_GRID = (0.5, 1.0, 1.5)
IZH_GRAPH_SEED = 7
IZH_KEY_SEED = 0
RS_A = 0.02
RS_B = 0.2
RS_C_MV = -65.0
RS_D = 8.0
RS_U0 = -13.0
EDGE_TAU_MS = 2.0  # receptor time constant; matches the plasticity-controls fixture

# AdEx fixture: 6 excitatory and 4 PV cells, every PV cell projects to every E cell,
# plus an E ring. Holding current and pulses follow
# tests/test_istdp_vogels2011.py::_two_edge_run.
ADEX_N_E = 6
ADEX_N_PV = 4
ADEX_N_STEPS = 300
ADEX_DT_MS = 0.1
ADEX_DELAY_STEPS = 20  # 2 ms at dt 0.1 ms
ADEX_WEIGHT_NS = 3.0
ADEX_HOLD_PA = -500.0  # blocks spontaneous spikes
ADEX_PULSE_PA = 1e5  # one-step pulse that forces one spike
ADEX_PV_PULSE_STEPS = (30, 90, 150, 210)
ADEX_E_LAG_STEPS = 10  # E cells pulse this many steps after the PV cells
ADEX_RECEPTOR_TAU_MS = 2.0
ADEX_SEED = 0

# Explicit nonzero settings for rules whose descriptor defaults are all zero.
SYNTHETIC_NONZERO = {"k_h": 0.5, "k_w": 0.3, "gamma": 0.0}
ELIGIBILITY_NONZERO = {"k_h": 0.5, "k_w": 0.3, "gamma": 0.0, "tau_e": 20.0}
PCL_TAU_MS = 20.0
PCL_W_MAX = 5.0
PCL_F = 1.0
PCL_LAM = 1.5
PCL_ETA = 0.1

@dataclass(frozen=True)
class Net:
    """One small network with the HDP-free runner and the registered-rule runner."""

    n_neurons: int
    n_edges: int
    pre: np.ndarray  # (n_edges,) presynaptic cell ids
    post: np.ndarray  # (n_edges,) postsynaptic cell ids
    w0: np.ndarray  # (n_edges,) initial |w|, the weight carry at step 0
    run_base: Callable[[], np.ndarray]  # -> spikes (T, N) bool, HDP-free kernel
    run_rule: Callable[
        [str, Mapping[str, Any] | None, np.ndarray | None], dict[str, np.ndarray]
    ]  # -> {"spikes": (T, N) bool, "w": (n_edges,), "H": (N,) or (N, d_H)}


@dataclass(frozen=True)
class Case:
    """How one registered rule is exercised."""

    family: str  # "izhikevich" or "adex": the kernel the rule runs on
    zero: Callable[[Net], dict[str, Any]]  # zero weight-drive setting
    nonzero: Callable[[Net], dict[str, Any] | None]  # None = descriptor defaults
    h_moves_at_zero: bool  # expected: H changes at zero weight drive


def _izh_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(IZH_GRAPH_SEED)
    pre: list[int] = []
    post: list[int] = []
    w0: list[float] = []
    for src in range(IZH_N_NEURONS):
        others = [dst for dst in range(IZH_N_NEURONS) if dst != src]
        for dst in rng.choice(others, size=IZH_OUT_DEGREE, replace=False):
            pre.append(src)
            post.append(int(dst))
            w0.append(float(rng.choice(IZH_WEIGHT_GRID)))
    return (
        np.asarray(pre, dtype=np.int32),
        np.asarray(post, dtype=np.int32),
        np.asarray(w0, dtype=np.float32),
    )


@functools.cache
def _izh_net() -> Net:
    pre, post, w0 = _izh_edges()
    n = IZH_N_NEURONS
    params = IzhikevichParams(
        a=jnp.full((n,), RS_A),
        b=jnp.full((n,), RS_B),
        c=jnp.full((n,), RS_C_MV),
        d=jnp.full((n,), RS_D),
        drive=jnp.asarray(np.linspace(IZH_DRIVE_LO, IZH_DRIVE_HI, n), dtype=jnp.float32),
        sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)),
        v0=jnp.full((n,), RS_C_MV),
        u0=jnp.full((n,), RS_U0),
        source_scale=jnp.ones((n,)),
        labels=("E",) * n,
        layer_labels=("L4",) * n,
        source_calibration_status="x",
    )
    edges = EdgeList(
        pre=jnp.asarray(pre),
        post=jnp.asarray(post),
        weight=jnp.asarray(w0),
        receptor_index=jnp.zeros(pre.shape, dtype=jnp.int32),
        tau_ms=jnp.full(pre.shape, EDGE_TAU_MS, dtype=jnp.float32),
        source_calibration_status="x",
    )
    key = jax.random.PRNGKey(IZH_KEY_SEED)

    def run_base() -> np.ndarray:
        _, spikes, _, _ = simulate_edge_recurrent_izhikevich(
            params, edges, IZH_N_STEPS, IZH_DT_MS, key, noise_scale=0.0
        )
        return np.asarray(spikes) > 0.5

    def run_rule(
        name: str,
        rule_params: Mapping[str, Any] | None,
        mask: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
            params,
            edges,
            IZH_N_STEPS,
            IZH_DT_MS,
            key,
            noise_scale=0.0,
            hdp_rule=name,
            hdp_rule_params=rule_params,
            plasticity_mask=mask,
        )
        return {
            "spikes": np.asarray(spikes) > 0.5,
            "w": np.asarray(diag["w_final"]),
            "H": np.asarray(diag["H_final"]),
        }

    return Net(
        n_neurons=n,
        n_edges=int(pre.shape[0]),
        pre=pre,
        post=post,
        w0=w0,
        run_base=run_base,
        run_rule=run_rule,
    )


def _adex_drive() -> np.ndarray:
    """(T, N) current in pA: holding current everywhere, one forced spike per pulse.

    Each cell pulses one step after the previous one, so spike times differ on
    every edge and a pre/post index mix-up changes the pairing.
    """
    drive = np.full((ADEX_N_STEPS, ADEX_N_E + ADEX_N_PV), ADEX_HOLD_PA)
    for t_pv in ADEX_PV_PULSE_STEPS:
        for p in range(ADEX_N_PV):
            drive[t_pv + p, ADEX_N_E + p] = ADEX_PULSE_PA
        for e in range(ADEX_N_E):
            drive[t_pv + ADEX_E_LAG_STEPS + e, e] = ADEX_PULSE_PA
    return drive


@functools.cache
def _adex_net() -> Net:
    e_ids = range(ADEX_N_E)
    pv_ids = range(ADEX_N_E, ADEX_N_E + ADEX_N_PV)
    pre: list[int] = []
    post: list[int] = []
    for pv in pv_ids:
        for e in e_ids:
            pre.append(pv)
            post.append(e)
    for e in e_ids:
        pre.append(e)
        post.append((e + 1) % ADEX_N_E)
    n_edges = len(pre)
    params = make_adex_params(("E",) * ADEX_N_E + ("PV",) * ADEX_N_PV)
    w0 = np.full(n_edges, ADEX_WEIGHT_NS)
    kw: dict[str, Any] = {
        "n_steps": ADEX_N_STEPS,
        "dt_ms": ADEX_DT_MS,
        "edge_pre": pre,
        "edge_post": post,
        "edge_weight_nS": w0.tolist(),
        "edge_receptor": [0] * n_edges,
        "edge_delay_steps": [ADEX_DELAY_STEPS] * n_edges,
        "receptor_reversal_mV": (0.0,),
        "receptor_tau_ms": (ADEX_RECEPTOR_TAU_MS,),
        "decay": "exp",
        "I_ext_pA": _adex_drive(),
        "record_v": False,
    }

    def run_base() -> np.ndarray:
        out = simulate_adex(params, **kw)
        return np.asarray(out["spikes"]) > 0.5

    def run_rule(
        name: str,
        rule_params: Mapping[str, Any] | None,
        mask: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        out = simulate_adex_hdp_registered(
            params,
            **kw,
            hdp_rule=name,
            hdp_rule_params=rule_params,
            plasticity_mask=mask,
            seed=ADEX_SEED,
        )
        return {
            "spikes": np.asarray(out["spikes"]) > 0.5,
            "w": np.asarray(out["w_final_nS"]),
            "H": np.asarray(out["H_final"]),
        }

    return Net(
        n_neurons=ADEX_N_E + ADEX_N_PV,
        n_edges=n_edges,
        pre=np.asarray(pre),
        post=np.asarray(post),
        w0=w0,
        run_base=run_base,
        run_rule=run_rule,
    )


def _net(family: str) -> Net:
    if family == "izhikevich":
        return _izh_net()
    if family == "adex":
        return _adex_net()
    raise ValueError(f"unknown family {family!r}")


def _pcl_params(net: Net, *, eta: float, norm: float) -> dict[str, Any]:
    """Per-edge pcl_stdp parameters; groups are the postsynaptic cells."""
    e = net.n_edges
    return hdp_rules_pcl.rule_params(
        tau=np.full(e, PCL_TAU_MS),
        w_max=np.full(e, PCL_W_MAX),
        eta=np.full(e, eta),
        f=np.full(e, PCL_F),
        lam=np.full(e, PCL_LAM),
        norm=np.full(e, norm),
        group=net.post,
        n_groups=net.n_neurons,
    )


# Every entry of the core registry. A rule added to the package without an entry
# fails test_case_table_covers_live_registry.
CASES: dict[str, Case] = {
    "synthetic_presyn_gain": Case(
        family="izhikevich",
        zero=lambda net: {"k_h": 0.5, "k_w": 0.0, "gamma": 0.0},
        nonzero=lambda net: dict(SYNTHETIC_NONZERO),
        h_moves_at_zero=True,
    ),
    "eligibility_trace_gain": Case(
        family="izhikevich",
        zero=lambda net: {"k_h": 0.5, "k_w": 0.0, "gamma": 0.0, "tau_e": 20.0},
        nonzero=lambda net: dict(ELIGIBILITY_NONZERO),
        h_moves_at_zero=True,
    ),
    "trace_ltp": Case(
        family="izhikevich",
        zero=lambda net: {"A_plus": 0.0},
        nonzero=lambda net: None,
        h_moves_at_zero=True,
    ),
    "trace_ltd": Case(
        family="izhikevich",
        zero=lambda net: {"A_minus": 0.0},
        nonzero=lambda net: None,
        h_moves_at_zero=True,
    ),
    "trace_stdp": Case(
        family="izhikevich",
        zero=lambda net: {"A_plus": 0.0, "A_minus": 0.0},
        nonzero=lambda net: None,
        h_moves_at_zero=True,
    ),
    hdp_rules_pcl.NAME: Case(
        family="izhikevich",
        zero=lambda net: _pcl_params(net, eta=0.0, norm=0.0),
        nonzero=lambda net: _pcl_params(net, eta=PCL_ETA, norm=1.0),
        h_moves_at_zero=False,
    ),
    ISTDP_RULE_NAME: Case(
        family="adex",
        zero=lambda net: {"eta": 0.0},
        nonzero=lambda net: None,
        h_moves_at_zero=False,
    ),
}


def _core_rule_names() -> tuple[str, ...]:
    """Core registry at import: rules whose step function ships in the package.

    test_hdp_v_floor.py registers a test-only rule at import. Excluding by module
    keeps the check independent of collection order. Rules registered inside test
    bodies run after collection, so the snapshot does not see them; the supplement
    S5 generator calls this at run time, so a rule built by ``superpose_hdp_rules``
    (a closure defined in ``jaxfne.hdp_rule``) is excluded by its qualname.
    """
    return tuple(
        name
        for name in list_registered_hdp_rules()
        if get_hdp_rule(name)[1].__module__.startswith("jaxfne.")
        and "<locals>" not in get_hdp_rule(name)[1].__qualname__
    )


_LIVE_RULES = _core_rule_names()


def _case(name: str) -> Case:
    assert name in CASES, f"no case for registered rule {name!r}; add it to CASES"
    return CASES[name]


def _assert_cells_active(base: np.ndarray, net: Net) -> None:
    """Precondition: the HDP-free run spikes, and some edge has an active pre and post cell."""
    assert base.sum() > 0, "fixture must spike"
    active = base.any(axis=0)
    assert bool(np.any(active[net.pre] & active[net.post])), (
        "no edge with an active presynaptic and postsynaptic cell"
    )


@pytest.mark.parametrize("name", _LIVE_RULES)
def test_zero_drive_reduces_to_base_model(name: str) -> None:
    case = _case(name)
    net = _net(case.family)
    base = net.run_base()
    _assert_cells_active(base, net)
    out = net.run_rule(name, case.zero(net), None)
    np.testing.assert_array_equal(
        out["spikes"].astype(np.int64),
        base.astype(np.int64),
        err_msg="spikes differ from the HDP-free kernel at zero weight drive",
    )
    np.testing.assert_array_equal(
        out["w"], net.w0, err_msg="weights moved at zero weight drive"
    )
    h_moves = not np.array_equal(out["H"], np.ones_like(out["H"]))
    assert h_moves == case.h_moves_at_zero, (
        f"H moves = {h_moves} at zero weight drive; expected {case.h_moves_at_zero}"
    )


@pytest.mark.parametrize("name", _LIVE_RULES)
def test_all_false_mask_freezes_weights(name: str) -> None:
    case = _case(name)
    net = _net(case.family)
    _assert_cells_active(net.run_base(), net)
    free = net.run_rule(name, case.nonzero(net), None)
    assert bool(np.any(free["w"] != net.w0)), (
        "unmasked run must move a weight, or the mask identity is vacuous"
    )
    masked = net.run_rule(name, case.nonzero(net), np.zeros(net.n_edges, dtype=bool))
    np.testing.assert_array_equal(
        masked["w"], net.w0, err_msg="all-False plasticity mask left weights moving"
    )


@pytest.mark.parametrize("name", _LIVE_RULES)
def test_nonzero_parameters_change_a_weight(name: str) -> None:
    case = _case(name)
    net = _net(case.family)
    _assert_cells_active(net.run_base(), net)
    out = net.run_rule(name, case.nonzero(net), None)
    change = float(np.max(np.abs(out["w"] - net.w0)))
    assert change > 0.0, f"no weight changed at nonzero parameters: max |dw| = {change!r}"


def test_case_table_covers_live_registry() -> None:
    assert _LIVE_RULES, "the registry snapshot is empty"
    missing = sorted(set(_LIVE_RULES) - set(CASES))
    stale = sorted(set(CASES) - set(_LIVE_RULES))
    assert not missing and not stale, (
        f"rules without a case: {missing}; cases without a rule: {stale}"
    )


def test_runtime_superposed_rule_is_not_a_core_rule() -> None:
    # Release CI 40433612421: a superposed rule left registered by an earlier test
    # reached the S5 supplement as a core rule without a case.
    from jaxfne import hdp_rule
    from jaxfne.hdp_rule import superpose_hdp_rules

    name = "_reduction_identity_probe_sup"
    superpose_hdp_rules(name, ["trace_ltp", "trace_ltd"])
    try:
        assert name in list_registered_hdp_rules(), "fixture must register the rule"
        assert name not in _core_rule_names()
        assert set(_core_rule_names()) == set(_LIVE_RULES)
    finally:
        hdp_rule._REGISTRY.pop(name, None)
