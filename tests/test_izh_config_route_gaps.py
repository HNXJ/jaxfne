"""Izhikevich configuration-route gaps (packet 9C-E1): P-046, P-048 and P-050.

P-046: ``connectivity_mode="explicit"`` realizes only the declared rules' edges, and the
receipt counts are true. P-048: ``cell_params`` declares the initial state (``v0``, ``u0``)
that the simulation reads. ``{}`` still selects every cell. P-050: the explicit duplicate
check is no longer quadratic in the edge count.

Every assertion reads the realized object (``params["emitter"]``, ``params["edge_list"]``,
the simulate accessor, ``model.cfg.metadata``), never the declared input. Fixtures use n=40.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import _resolved_edge_weight

N = 40  # cell_types E 0.8 / I 0.2 -> 32 E cells and 8 I cells
E_WEIGHT = 0.37  # asymmetric, non-neutral declared weights
I_WEIGHT = 0.21
# Plain-route dense default: N*(N-1) off-diagonal pairs at weight 0.5/sqrt(N).
DEFAULT_WEIGHT = 0.5 / np.sqrt(N)
V0_I = [-41.5, -12.25, -30.0, -55.5, -20.75, -8.5, -61.25, -35.0]  # one per I cell
U0_I = [3.5, -7.25, 0.5, 2.0, -1.5, 4.75, -6.0, 1.25]  # one per I cell
U0_E_SCALAR = -9.5


def _base():
    return (
        jtfne.configuration()
        .network(name="gap", kind="balanced_ei_population", n=N, cell_types={"E": 0.8, "I": 0.2})
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 5.0, "reversal_mV": 0.0})
    )


def _finish(cfg):
    return (
        cfg.emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy",
               gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )


def _ei_cfg(*, mode=None, mechanism=True):
    kw = {"mechanism": "ampa"} if mechanism else {}
    cfg = (
        _base()
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=1.0, weight=E_WEIGHT, **kw)
        .connections(name="e_i", source={"cell_type": "E"}, target={"cell_type": "I"},
                     probability=1.0, weight=I_WEIGHT, **kw)
    )
    if mode is not None:
        cfg = cfg.update_metadata(connectivity_mode=mode)
    return _finish(cfg)


def _labels(model):
    return np.asarray(model.params["emitter"].labels)


def _cell_counts(model):
    labels = _labels(model)
    n_e, n_i = int(np.sum(labels == "E")), int(np.sum(labels == "I"))
    assert n_e > 0 and n_i > 0, "fixture must hold both E and I cells"
    return n_e, n_i


def _edges(model):
    edges = model.params["edge_list"]
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    w = np.asarray(_resolved_edge_weight(edges, jnp.float32, model.params["emitter"]),
                   dtype=np.float64)
    return edges, pre, post, w


# ----------------------------------------------------------------- P-046 explicit topology

@pytest.mark.parametrize("mechanism", [True, False])
def test_explicit_mode_realizes_only_the_declared_edges(mechanism):
    model = jtfne.construct(_ei_cfg(mode="explicit", mechanism=mechanism))
    n_e, n_i = _cell_counts(model)
    declared = n_e * (n_e - 1) + n_e * n_i
    assert declared == 1248, "fixture composition: 32 E and 8 I cells"
    edges, pre, post, w = _edges(model)
    assert int(edges.n_edges) == declared
    labels = _labels(model)
    assert np.all(labels[pre] == "E"), "every realized edge starts at a declared E source"
    assert np.all(pre != post), "self edges are dropped"
    aw = np.abs(w)
    assert np.count_nonzero(np.isclose(aw, E_WEIGHT, rtol=1e-6)) == n_e * (n_e - 1)
    assert np.count_nonzero(np.isclose(aw, I_WEIGHT, rtol=1e-6)) == n_e * n_i
    W = np.asarray(model.params["emitter"].W)
    assert W.size == 0 or not np.any(W), "explicit W must carry no default coupling"
    assert model.cfg.metadata["connectivity_compilation"] == {
        "connectivity_mode": "explicit",
        "default_edge_count": 0,
        "declared_rule_edge_count": declared,
        "total_compiled_edge_count": declared,
    }


def test_default_mode_keeps_the_dense_coupling_and_writes_no_receipt():
    model = jtfne.construct(_ei_cfg(mode=None))
    edges, _, _, w = _edges(model)
    # Baseline values: 1560 dense default pairs plus 1248 declared edges.
    assert int(edges.n_edges) == 2808
    aw = np.abs(w)
    assert np.count_nonzero(np.isclose(aw, DEFAULT_WEIGHT, rtol=1e-6)) == 1560
    assert np.count_nonzero(np.isclose(aw, E_WEIGHT, rtol=1e-6)) == 992
    assert np.count_nonzero(np.isclose(aw, I_WEIGHT, rtol=1e-6)) == 256
    assert "connectivity_compilation" not in model.cfg.metadata


# ------------------------------------------------------------- empty selector still valid

@pytest.mark.parametrize("mechanism", [True, False])
def test_empty_selector_still_selects_every_cell(mechanism):
    kw = {"mechanism": "ampa"} if mechanism else {}
    cfg = _finish(
        _base()
        .connections(name="all_targets", source={"cell_type": "E"}, target={},
                     probability=1.0, weight=E_WEIGHT, **kw)
        .update_metadata(connectivity_mode="explicit")
    )
    model = jtfne.construct(cfg)
    n_e, n_i = _cell_counts(model)
    n_total = n_e + n_i
    # Every E source reaches every other cell (self edges dropped): n_e * (n_total - 1).
    # The rule's own compiled count is checked, not the total edge list (P-046 is separate).
    rule = model.cfg.metadata["circuit"]["connections"][0]
    assert int(rule["compiled_n_edges"]) == n_e * (n_total - 1) == 1248
    receipt = model.cfg.metadata["connectivity_compilation"]
    assert receipt["declared_rule_edge_count"] == 1248


# ------------------------------------------------------------ P-048 declared initial state

def test_undeclared_u0_keeps_the_preset_value():
    # Guard: with no cell_params the default u0 is unchanged (E -13, I +6.5 for these presets).
    model = jtfne.construct(_finish(_base().runtime(random_v0=False)))
    em = model.params["emitter"]
    labels = _labels(model)
    np.testing.assert_array_equal(np.asarray(em.u0)[labels == "E"], np.float32(-13.0))
    np.testing.assert_array_equal(np.asarray(em.u0)[labels == "I"], np.float32(6.5))


def test_declared_v0_and_u0_are_realized_per_cell_on_the_plain_route():
    cfg = _finish(
        _base()
        .cell_params({"cell_type": "I"}, {"v0": V0_I, "u0": U0_I})
        .cell_params({"cell_type": "E"}, {"u0": U0_E_SCALAR})
        .runtime(random_v0=False)
    )
    model = jtfne.construct(cfg)
    em = model.params["emitter"]
    labels = _labels(model)
    i_idx = np.flatnonzero(labels == "I")
    e_idx = np.flatnonzero(labels == "E")
    assert i_idx.size == len(V0_I), "fixture must give one declared value per I cell"
    np.testing.assert_array_equal(np.asarray(em.v0)[i_idx], np.asarray(V0_I, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(em.u0)[i_idx], np.asarray(U0_I, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(em.u0)[e_idx], np.float32(U0_E_SCALAR))
    np.testing.assert_array_equal(np.asarray(em.v0)[e_idx], np.float32(-65.0))


def test_declared_v0_is_refused_while_random_v0_is_on():
    cfg = _finish(_base().cell_params({"cell_type": "I"}, {"v0": V0_I}))
    with pytest.raises(ValueError, match=re.escape("runtime(random_v0=False)")):
        jtfne.construct(cfg)


def test_declared_u0_is_realized_while_random_v0_is_on():
    cfg = _finish(_base().cell_params({"cell_type": "I"}, {"u0": U0_I}))
    model = jtfne.construct(cfg)
    em = model.params["emitter"]
    i_idx = np.flatnonzero(_labels(model) == "I")
    assert i_idx.size == len(U0_I), "fixture must give one declared value per I cell"
    np.testing.assert_array_equal(np.asarray(em.u0)[i_idx], np.asarray(U0_I, dtype=np.float32))
    assert np.ptp(np.asarray(em.v0)) > 0.0, "random_v0 must still draw v0 when only u0 is declared"


def test_declared_initial_state_changes_the_simulated_run():
    sim = jtfne.Simulation(duration_ms=200.0, dt_ms=0.5, seed=0,
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    plain = jtfne.construct(_finish(_base().runtime(random_v0=False)))
    declared = jtfne.construct(_finish(
        _base()
        .cell_params({"cell_type": "I"}, {"v0": V0_I, "u0": U0_I})
        .runtime(random_v0=False)
    ))
    base = plain.simulate(sim)
    dec = declared.simulate(sim)
    i_idx = np.flatnonzero(_labels(plain) == "I")
    assert int(np.asarray(base.spikes).sum()) > 0, "fixture must spike"
    assert not np.array_equal(np.asarray(base.V_m)[:, i_idx], np.asarray(dec.V_m)[:, i_idx]), \
        "declared v0/u0 must reach the simulated membrane state"


def test_declared_v0_and_u0_are_realized_on_the_population_route():
    def column_cfg(**params_by_type):
        cfg = jtfne.configuration().column(name="c1", layers=["L4"], n=N)
        for ct, params in params_by_type.items():
            cfg = cfg.cell_params({"cell_type": ct}, params)
        return _finish(cfg.runtime(random_v0=False))

    labels = _labels(jtfne.construct(column_cfg()))
    types = [t for t in sorted(set(labels.tolist())) if t != "E" and int(np.sum(labels == t)) > 1]
    assert types, "fixture must hold a non-E cell type with more than one cell"
    ct = types[0]
    idx = np.flatnonzero(labels == ct)
    v0 = [-30.0 - 2.5 * k for k in range(idx.size)]
    u0 = [1.5 + 0.5 * k for k in range(idx.size)]
    em = jtfne.construct(column_cfg(**{ct: {"v0": v0, "u0": u0}})).params["emitter"]
    np.testing.assert_array_equal(np.asarray(em.v0)[idx], np.asarray(v0, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(em.u0)[idx], np.asarray(u0, dtype=np.float32))


def test_explicit_duplicate_check_is_not_quadratic():
    """P-050: 400k distinct identities clear in well under the quadratic cost (about 90 s)."""
    import time
    from types import SimpleNamespace

    from jaxfne._construct_core import _reject_duplicate_explicit_edges

    n = 400_000
    idx = np.arange(n, dtype=np.int64)
    edges = SimpleNamespace(n_edges=n, pre=idx // 1000, post=idx % 1000,
                            receptor_index=np.zeros(n, dtype=np.int64))
    t0 = time.perf_counter()
    _reject_duplicate_explicit_edges(edges)
    assert time.perf_counter() - t0 < 10.0
    # The refusal still holds: one repeated identity outside an exempt block raises.
    dup = SimpleNamespace(n_edges=n + 1, pre=np.r_[edges.pre, 0], post=np.r_[edges.post, 0],
                          receptor_index=np.zeros(n + 1, dtype=np.int64))
    with pytest.raises(ValueError, match="duplicate executable synaptic identities"):
        _reject_duplicate_explicit_edges(dup)
    # A repeat inside one exempt (multinomial) block is the declared outcome.
    _reject_duplicate_explicit_edges(dup, exempt_ranges=[(0, n + 1)])
