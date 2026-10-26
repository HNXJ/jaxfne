"""Construct-path ``weight_range`` and ``Simulation.gaussian_drive`` (packet 9C-E1).

``weight_range`` draws each edge's magnitude U[lo, hi) on the mechanism-aware
compile path and is read back through the accessor simulate uses. ``gaussian_drive``
sets a per-cell stochastic-current coefficient on the Izhikevich routes. Every
refusal is pinned by its message.

The default Izhikevich construct couples E and I through a dense default W (the edge
list is derived from it), so a "disconnected" run is the ``disconnected_null``
ablation, which zeroes that coupling at simulate time. Continuation refuses
ablations, so its test checks only that the declared noise reaches the E cells.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import _resolved_edge_weight

DISCONNECTED = "disconnected_null"
ZERO = {"std_by_cell_type": {"E": 0.0, "I": 0.0}}


def _gauss(e_std, i_std=0.0):
    return {"std_by_cell_type": {"E": e_std, "I": i_std}}


def _ei_model():
    cfg = (
        jtfne.configuration()
        .network(name="gauss_ei", kind="balanced_ei_population", n=40,
                 cell_types={"E": 0.5, "I": 0.5})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    return jtfne.construct(cfg)


def _cell_idx(model):
    labels = np.asarray(model.params["emitter"].labels)
    e_idx = np.flatnonzero(labels == "E")
    i_idx = np.flatnonzero(labels == "I")
    assert e_idx.size > 0 and i_idx.size > 0, "fixture must hold both E and I cells"
    return e_idx, i_idx


def _sim(gauss, *, runtime=None, duration=200.0, ablation=None):
    if runtime is None:
        runtime = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    return jtfne.Simulation(duration_ms=duration, dt_ms=0.5, seed=0, runtime=runtime,
                            gaussian_drive=gauss, ablation=ablation)


# ---------------------------------------------------------------- weight_range

def _range_cfg(edge_seed):
    return (
        jtfne.configuration()
        .network(name="wr", kind="balanced_ei_population", n=100,
                 cell_types={"E": 0.5, "I": 0.5})
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 5.0, "reversal_mV": 0.0})
        .connections(name="ee_range", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.8, mechanism="ampa", weight_range=(0.2, 0.7))
        .connections(name="ei_const", source={"cell_type": "E"}, target={"cell_type": "I"},
                     probability=0.8, mechanism="ampa", weight=0.3)
        .connectivity(edge_seed=edge_seed)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )


def _rule_blocks(model):
    """Weights of the two declared rules, read from their trailing edge blocks.

    The default E-I coupling is built first; the declared rules are appended in
    declaration order (``_construct_core``), so their edges are the trailing
    ``compiled_n_edges`` entries of the built edge list.
    """
    emitter = model.params["emitter"]
    edges = model.params["edge_list"]
    labels = np.asarray(emitter.labels)
    rows = model.cfg.metadata["circuit"]["connections"]
    assert [r["name"] for r in rows] == ["ee_range", "ei_const"], "fixture must declare both rules"
    counts = [int(r["compiled_n_edges"]) for r in rows]
    n_edges = int(edges.n_edges)
    start = n_edges - sum(counts)
    mid = start + counts[0]
    rng, const = slice(start, mid), slice(mid, n_edges)
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    assert np.all(labels[pre[rng]] == "E") and np.all(labels[post[rng]] == "E"), \
        "range block must hold only E to E edges"
    assert np.all(labels[pre[const]] == "E") and np.all(labels[post[const]] == "I"), \
        "constant block must hold only E to I edges"
    w = np.asarray(_resolved_edge_weight(edges, jnp.float32, emitter), dtype=np.float64)
    return w[rng], w[const]


def test_weight_range_draws_uniform_magnitudes_per_edge():
    w_range, w_const = _rule_blocks(jtfne.construct(_range_cfg(edge_seed=0)))
    assert w_range.size >= 1000, "fixture must give the range rule at least 1000 edges"
    assert np.all((w_range >= 0.2) & (w_range < 0.7))
    assert np.ptp(w_range) > 0.0, "range weights must not all be equal"
    se = w_range.std(ddof=1) / np.sqrt(w_range.size)
    assert abs(w_range.mean() - 0.45) <= 3.0 * se
    # The second rule carries no weight_range and keeps one constant weight.
    assert w_const.size > 0, "fixture must give the constant rule edges"
    np.testing.assert_allclose(w_const, 0.3, rtol=1e-6)


def test_weight_range_depends_on_edge_seed():
    w_a, _ = _rule_blocks(jtfne.construct(_range_cfg(edge_seed=0)))
    w_b, _ = _rule_blocks(jtfne.construct(_range_cfg(edge_seed=1)))
    assert w_a.size > 0 and w_b.size > 0, "fixture must give range edges for both seeds"
    assert w_a.shape != w_b.shape or not np.allclose(w_a, w_b)


def _rule_cfg(**kw):
    return (
        jtfne.configuration()
        .network(name="wr_bad", kind="balanced_ei_population", n=20,
                 cell_types={"E": 0.5, "I": 0.5})
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 5.0, "reversal_mV": 0.0})
        .connections(name="bad", source={"cell_type": "E"}, target={"cell_type": "E"},
                     mechanism="ampa", **kw)
    )


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"weight": 0.5, "weight_range": (0.2, 0.7)}, "with weight="),
        ({"weight_std": 0.1, "weight_range": (0.2, 0.7)}, "with weight_std="),
        ({"weight_range": 0.5}, "2-sequence"),
        ({"weight_range": (0.1, 0.2, 0.3)}, "2-sequence"),
        ({"weight_range": (0.1, float("nan"))}, "finite"),
        ({"weight_range": (True, 0.5)}, "finite"),
        ({"weight_range": (-0.1, 0.5)}, "lo must be >= 0"),
        ({"weight_range": (0.5, 0.5)}, "lo < hi"),
        ({"weight_range": (0.7, 0.2)}, "lo < hi"),
    ],
)
def test_weight_range_refusals_name_the_rule(kw, match):
    with pytest.raises(ValueError, match=match) as exc:
        _rule_cfg(**kw)
    assert "'bad'" in str(exc.value)


def test_weight_range_refused_on_sign_only_fallback():
    # No mechanism on the rule: construct takes the sign-only path, which has no
    # per-edge draw, so it must refuse the range rather than fall back to 1/sqrt(n).
    cfg = (
        jtfne.configuration()
        .network(name="wr_so", kind="balanced_ei_population", n=20,
                 cell_types={"E": 0.5, "I": 0.5})
        .connections(name="bad", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.5, weight_range=(0.2, 0.7))
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    with pytest.raises(ValueError, match="weight_range") as exc:
        jtfne.construct(cfg)
    assert "'bad'" in str(exc.value)


# ----------------------------------------------------------- gaussian_drive

def test_gaussian_drive_edge_route_moves_only_the_declared_type():
    model = _ei_model()
    e_idx, i_idx = _cell_idx(model)
    zero = model.simulate(_sim(ZERO, ablation=DISCONNECTED))
    g3 = model.simulate(_sim(_gauss(3.0), ablation=DISCONNECTED))
    g6 = model.simulate(_sim(_gauss(6.0), ablation=DISCONNECTED))
    V0, V3, V6 = (np.asarray(s.V_m) for s in (zero, g3, g6))
    assert np.asarray(zero.spikes)[:, e_idx].sum() > 0, "fixture must spike"
    np.testing.assert_allclose(V3[:, i_idx], V0[:, i_idx], rtol=1e-12)
    assert not np.allclose(V3[:, e_idx], V0[:, e_idx])
    assert not np.allclose(V6[:, e_idx], V3[:, e_idx])
    np.testing.assert_allclose(V6[:, i_idx], V0[:, i_idx], rtol=1e-12)
    assert g3.metadata["gaussian_drive"] == {"std_by_cell_type": {"E": 3.0, "I": 0.0}}
    undeclared = model.simulate(_sim(None, ablation=DISCONNECTED))
    assert "gaussian_drive" not in undeclared.metadata, "default path must record no receipt"


def test_gaussian_drive_dense_default_route_moves_only_the_declared_type():
    model = _ei_model()
    e_idx, i_idx = _cell_idx(model)
    dense = jtfne.RuntimeConfig(recurrent_backend="dense")
    zero = model.simulate(_sim(ZERO, runtime=dense, ablation=DISCONNECTED))
    g3 = model.simulate(_sim(_gauss(3.0), runtime=dense, ablation=DISCONNECTED))
    V0, V3 = np.asarray(zero.V_m), np.asarray(g3.V_m)
    assert np.asarray(zero.spikes)[:, e_idx].sum() > 0, "fixture must spike"
    np.testing.assert_allclose(V3[:, i_idx], V0[:, i_idx], rtol=1e-12)
    assert not np.allclose(V3[:, e_idx], V0[:, e_idx])


def test_gaussian_drive_continuation_route_reaches_the_e_cells():
    # Continuation refuses ablations, so the default E-I coupling is live and the I
    # cells move with the E cells; this checks only that the declared noise arrives.
    model = _ei_model()
    e_idx, _ = _cell_idx(model)
    zero, _ = model.simulate(_sim(ZERO, duration=100.0), return_state=True)
    g3, _ = model.simulate(_sim(_gauss(3.0), duration=100.0), return_state=True)
    V0, V3 = np.asarray(zero.V_m), np.asarray(g3.V_m)
    assert np.asarray(zero.spikes)[:, e_idx].sum() > 0, "fixture must spike"
    assert not np.allclose(V3[:, e_idx], V0[:, e_idx])


def test_gaussian_drive_homeostatic_route_moves_only_the_declared_type():
    model = _ei_model()
    e_idx, i_idx = _cell_idx(model)
    homeo = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_homeostasis=True)
    zero = model.simulate(_sim(ZERO, runtime=homeo, ablation=DISCONNECTED))
    g3 = model.simulate(_sim(_gauss(3.0), runtime=homeo, ablation=DISCONNECTED))
    V0, V3 = np.asarray(zero.V_m), np.asarray(g3.V_m)
    assert "homeostasis" in g3.metadata, "fixture must engage the homeostatic route"
    np.testing.assert_allclose(V3[:, i_idx], V0[:, i_idx], rtol=1e-12)
    assert not np.allclose(V3[:, e_idx], V0[:, e_idx])


def test_gaussian_drive_hdp_route_reaches_the_e_cells():
    model = _ei_model()
    e_idx, _ = _cell_idx(model)
    hdp = jtfne.RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True,
        hdp_params={"alpha": 0.05, "gamma": 0.5, "K_ctrl": 0.15, "K_HDP": 0.01,
                    "tau_0_ms": 5.0})
    zero = model.simulate(_sim(ZERO, runtime=hdp, ablation=DISCONNECTED))
    g3 = model.simulate(_sim(_gauss(3.0), runtime=hdp, ablation=DISCONNECTED))
    V0, V3 = np.asarray(zero.V_m), np.asarray(g3.V_m)
    assert "hdp" in g3.metadata, "fixture must engage the HDP route"
    assert np.asarray(zero.spikes)[:, e_idx].sum() > 0, "fixture must spike"
    assert not np.allclose(V3[:, e_idx], V0[:, e_idx])


def test_gaussian_drive_net_m_admits_zero_and_refuses_nonzero():
    cfg = (
        jtfne.configuration()
        .network(name="nm", kind="balanced_ei_population", n=40,
                 cell_types={"E": 0.5, "I": 0.5})
        .emitter(family="izhikevich", izhikevich_update_order="net_m")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    model = jtfne.construct(cfg)

    def run(gauss, hdp_params=None):
        runtime = jtfne.RuntimeConfig(recurrent_backend="dense", hdp_params=hdp_params)
        return jtfne.Simulation(duration_ms=10.0, dt_ms=1.0, seed=0, runtime=runtime,
                                gaussian_drive=gauss)

    # net_m demands an explicit noise_scale=0.0 (existing rule); compare against that.
    base = model.simulate(run(None, hdp_params={"noise_scale": 0.0}))
    zero = model.simulate(run(ZERO))
    np.testing.assert_allclose(np.asarray(zero.V_m), np.asarray(base.V_m), rtol=1e-12)
    with pytest.raises(ValueError, match="update_order='net_m'"):
        model.simulate(run(_gauss(3.0)))


def test_gaussian_drive_refused_on_receptor_exponential_route():
    model = _ei_model()
    runtime = jtfne.RuntimeConfig(recurrent_backend="edge_list",
                                  synaptic_kernel="receptor_exponential")
    sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.5, seed=0, runtime=runtime,
                           gaussian_drive=_gauss(3.0))
    with pytest.raises(ValueError, match="receptor_exponential") as exc:
        model.simulate(sim)
    assert "simulate_receptor_exponential_izhikevich" in str(exc.value)


def test_gaussian_drive_refused_with_hdp_noise_scale():
    model = _ei_model()
    runtime = jtfne.RuntimeConfig(recurrent_backend="edge_list",
                                  hdp_params={"noise_scale": 0.5})
    sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.5, seed=0, runtime=runtime,
                           gaussian_drive=_gauss(3.0))
    with pytest.raises(ValueError, match="hdp_params\\['noise_scale'\\]"):
        model.simulate(sim)


def test_gaussian_drive_refused_by_simulate_batch():
    model = _ei_model()
    sim = _sim(_gauss(3.0), duration=10.0)
    with pytest.raises(ValueError, match="simulate_batch does not consume Simulation.gaussian_drive"):
        model.simulate_batch(sim, n_seeds=2)


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"std_by_cell_type": {"E": float("inf"), "I": 0.0}}, "finite number >= 0"),
        ({"std_by_cell_type": {"E": -1.0, "I": 0.0}}, "finite number >= 0"),
        ({"std_by_cell_type": {"E": True, "I": 0.0}}, "finite number >= 0"),
        ({"std_by_cell_type": {"E": 1.0, "I": 0.0}, "seed": 3}, "not recognised"),
        ({"E": 1.0}, "not recognised"),
        ({}, "requires the key 'std_by_cell_type'"),
        ({"std_by_cell_type": {}}, "non-empty mapping"),
        ([1.0], "must be a mapping"),
    ],
)
def test_gaussian_drive_declaration_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        jtfne.Simulation(duration_ms=10.0, dt_ms=0.5, seed=0, gaussian_drive=bad)


@pytest.mark.parametrize(
    "stds, match",
    [
        ({"E": 1.0}, "misses model cell type"),
        ({"E": 1.0, "I": 0.0, "PV": 2.0}, "names unknown cell type"),
    ],
)
def test_gaussian_drive_mapping_must_name_exactly_the_model_cell_types(stds, match):
    model = _ei_model()
    runtime = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.5, seed=0, runtime=runtime,
                           gaussian_drive={"std_by_cell_type": stds})
    with pytest.raises(ValueError, match=match):
        model.simulate(sim)


def test_gaussian_drive_refused_on_adex_route():
    cfg = (
        jtfne.configuration()
        .network(name="adx", n=20, cell_types={"E": 0.5, "PV": 0.5})
        .emitter(family="adex", adex_decay="exp")
    )
    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, gaussian_drive=_gauss(3.0))
    with pytest.raises(ValueError, match="emitter route 'adex'"):
        model.simulate(sim)
