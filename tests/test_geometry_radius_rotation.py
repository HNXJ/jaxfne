"""Rotation invariance of the localized sampler under an explicit Euclidean radius (C4).

The grid sampler in ``_candidate_pairs_localized`` buckets on an axis-aligned
grid, so rotating the population can move corner candidates across bucket
walls. Declaring ``spatial_radius`` keeps only pairs within that Euclidean
distance of the post neuron, which makes the candidate set rotation invariant.
The Gaussian weight already uses distance only. The radius is opt-in; without
it the sampler follows the previous path.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from collections import Counter

import jax
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.connectivity import (
    _LOCALIZED_CANDIDATE_POOL_CAP,
    _candidate_pairs_localized,
    compile_connection_rules,
)

POS_SEED = 11
KEY_SEED = 0
N_PRE = 60
N_POST = 20
SIGMA = 0.5
RADIUS = 4.0
MAX_IN_DEGREE = 6
ROT_DEG = 37.0


def _population():
    """60 pre and 20 post ids in a plane, fixed seed, disjoint id ranges."""
    rng = np.random.default_rng(POS_SEED)
    pre_xy = rng.uniform(0.0, 10.0, size=(N_PRE, 2))
    post_xy = rng.uniform(2.0, 8.0, size=(N_POST, 2))
    pre_ids = list(range(N_PRE))
    post_ids = [1000 + i for i in range(N_POST)]
    positions = {}
    for i, nid in enumerate(pre_ids):
        positions[nid] = np.append(pre_xy[i], 0.0)
    for i, nid in enumerate(post_ids):
        positions[nid] = np.append(post_xy[i], 0.0)
    return pre_ids, post_ids, positions


def _sample(pre_ids, post_ids, positions, **kw):
    params = dict(max_in_degree=MAX_IN_DEGREE, spatial_sigma=SIGMA,
                  allow_self=False, key=jax.random.PRNGKey(KEY_SEED))
    params.update(kw)
    return sorted(_candidate_pairs_localized(pre_ids, post_ids, positions, **params))


def _rotation_matrix(deg):
    theta = np.deg2rad(deg)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]])


def test_rotation_about_origin_keeps_radius_edge_set():
    pre_ids, post_ids, positions = _population()
    # The uniform pre-filter must stay out of the picture: a post pool is a
    # subset of the pre ids, so 60 pre ids keep every pool under the cap.
    assert len(pre_ids) < _LOCALIZED_CANDIDATE_POOL_CAP

    dists = np.array([float(np.linalg.norm(positions[a] - positions[b]))
                      for a in pre_ids for b in post_ids])
    assert float(np.abs(dists - RADIUS).min()) > 1e-9, (
        "fixture must keep every pair off the radius boundary")

    base = _sample(pre_ids, post_ids, positions, spatial_radius=RADIUS)
    assert len(base) > 0, "fixture must wire edges"
    assert max(Counter(p for _, p in base).values()) <= MAX_IN_DEGREE
    plain_base = _sample(pre_ids, post_ids, positions)

    rot = _rotation_matrix(ROT_DEG)
    turned = {nid: rot @ positions[nid] for nid in positions}

    # Sensitivity: without the radius the same rotation changes the edge set,
    # so this test fails on the sampler without the radius cutoff.
    assert _sample(pre_ids, post_ids, turned) != plain_base
    assert _sample(pre_ids, post_ids, turned, spatial_radius=RADIUS) == base


def test_translation_keeps_radius_edge_set():
    pre_ids, post_ids, positions = _population()
    assert len(pre_ids) < _LOCALIZED_CANDIDATE_POOL_CAP
    base = _sample(pre_ids, post_ids, positions, spatial_radius=RADIUS)
    assert len(base) > 0, "fixture must wire edges"
    moved = {nid: positions[nid] + np.array([3.5, -2.25, 1.0]) for nid in positions}
    assert _sample(pre_ids, post_ids, moved, spatial_radius=RADIUS) == base


def test_pre_id_permutation_keeps_radius_edge_set():
    pre_ids, post_ids, positions = _population()
    assert len(pre_ids) < _LOCALIZED_CANDIDATE_POOL_CAP
    base = _sample(pre_ids, post_ids, positions, spatial_radius=RADIUS)
    assert len(base) > 0, "fixture must wire edges"
    order = np.random.default_rng(3).permutation(len(pre_ids))
    shuffled = [pre_ids[i] for i in order]
    assert _sample(shuffled, post_ids, positions, spatial_radius=RADIUS) == base


# Plumbing: every new key changes the realized object, or it is refused.


def _positioned_neurons(n=10):
    return [{"neuron_id": i, "area": "X", "layer": "L", "cell_type": "E",
             "x": float(i), "y": 0.0, "z": 0.0} for i in range(n)]


_MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0, "sign": 1.0}}]


def _localized_rule(**kw):
    rule = {"name": "rec", "source": {"area": "X"}, "target": {"area": "X"},
            "weight": 0.1, "mechanism": "ampa", "allow_self_connections": True,
            "max_in_degree": 9, "spatial_sigma": 5.0}
    rule.update(kw)
    return [rule]


def test_declared_radius_reduces_compiled_edges():
    plain = compile_connection_rules(
        _positioned_neurons(), _localized_rule(), _MECHS, seed=0)
    assert plain.n_edges == 90
    cut = compile_connection_rules(
        _positioned_neurons(), _localized_rule(spatial_radius=2.5), _MECHS, seed=0)
    assert 0 < cut.n_edges < plain.n_edges


def test_radius_without_max_in_degree_refuses():
    rule = {"name": "rec", "source": {"area": "X"}, "target": {"area": "X"},
            "weight": 0.1, "mechanism": "ampa",
            "spatial_sigma": 1.0, "spatial_radius": 0.5}
    with pytest.raises(ValueError, match="spatial_radius needs max_in_degree"):
        compile_connection_rules(_positioned_neurons(), [rule], _MECHS, seed=0)


@pytest.mark.parametrize("bad", [0.0, -2.0, float("nan"), float("inf")])
def test_radius_must_be_finite_and_positive(bad):
    with pytest.raises(ValueError, match="spatial_radius"):
        compile_connection_rules(
            _positioned_neurons(), _localized_rule(spatial_radius=bad), _MECHS, seed=0)


_MECH = dict(name="ampa", kind="exponential", params={"tau_ms": 2.0, "receptor": "AMPA"})


def _localized_cfg(**extra):
    return (jtfne.Configuration()
            .runtime(seed=1, dtype="float32", duration_ms=20.0, dt_ms=0.5)
            .column(name="c", layers=["L4"], n=200)
            .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
            .connectivity(p_connect=0.0)
            .mechanisms(**_MECH)
            .connections(name="rec", source={}, target={}, mechanism="ampa", weight=0.03,
                         max_in_degree=10, spatial_sigma=0.1, **extra)
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes"])
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))


def _rule_entry(cfg, name="rec"):
    rules = cfg.metadata["circuit"]["connections"]
    return next(r for r in rules if r["name"] == name)


def test_builder_records_radius_as_float():
    entry = _rule_entry(_localized_cfg(spatial_radius=0.25))
    assert entry["spatial_radius"] == 0.25
    assert isinstance(entry["spatial_radius"], float)
    assert "spatial_radius" not in _rule_entry(_localized_cfg())


def test_declared_radius_reduces_column_edges():
    plain = jtfne.construct(_localized_cfg())
    pos = np.asarray(plain.params["positions"], dtype=float)
    n = pos.shape[0]
    plain_edges = int(plain.params["edge_list"].n_edges)
    assert plain_edges == n * 10, "fixture must wire the full in-degree"
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    radius = float(np.quantile(d[np.triu_indices(n, k=1)], 0.02))
    assert radius > 0.0, "fixture must have spread"
    cut = jtfne.construct(_localized_cfg(spatial_radius=radius))
    cut_edges = int(cut.params["edge_list"].n_edges)
    assert 0 <= cut_edges < plain_edges


def test_sign_only_route_refuses_radius():
    cfg = (jtfne.Configuration()
           .runtime(seed=1, dtype="float32", duration_ms=20.0, dt_ms=0.5)
           .column(name="c", layers=["L4"], n=200)
           .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
           .connectivity(p_connect=0.0)
           .connections(name="rec", source={}, target={}, weight=0.03,
                        max_in_degree=10, spatial_radius=0.25)
           .set_emitter("izhikevich", "cortical_eig")
           .probes(["spikes"])
           .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
    with pytest.raises(ValueError, match="spatial_radius"):
        jtfne.construct(cfg)


def test_adex_route_refuses_radius():
    cfg = (jtfne.Configuration()
           .network(name="x", n=60, cell_types={"E": 0.8, "PV": 0.1, "SOM": 0.1})
           .emitter(family="adex", adex_decay="exp")
           .mechanisms(name="ampa", kind="receptor",
                       params={"tau_ms": 2.0, "reversal_mV": 0.0})
           .connections(name="r", source={"cell_type": "E"}, target={"cell_type": "E"},
                        probability=0.1, weight=0.5, mechanism="ampa", delay_ms=2.0,
                        spatial_radius=0.1))
    with pytest.raises(ValueError, match="spatial_radius"):
        jtfne.construct(cfg)


def _scan_tensor(**conn_kw):
    kw = dict(mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
              local_inhibition={"resolution": "event_scan"})
    kw.update(conn_kw)
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E", **kw)
    return nt.NeuronalTensor(
        name="lif-scan-radius",
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(
                name="L4", n_neurons=6,
                neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")])],
            inter_connections=[conn])])


def test_lif_scan_route_refuses_radius():
    tensor = _scan_tensor(spatial_radius=0.5)
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=20.0, dt_ms=0.5, emitter="lif")
    with pytest.raises(ValueError, match="spatial_radius"):
        jtfne.construct(tensor, runtime)


def _probe_tensor(**conn_kw):
    kw = dict(max_in_degree=5, spatial_sigma=1.0)
    kw.update(conn_kw)
    layer = nt.Layer(
        name="L4", n_neurons=20,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")])
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=2.0, H=1.0), **kw)
    area = nt.Area(name="V1", layers=[layer], inter_connections=[conn])
    return nt.NeuronalTensor(areas=[area], name="radius_probe")


def _tensor_edges(model):
    el = model.params["edge_list"]
    return set(zip(np.asarray(el.pre).tolist(), np.asarray(el.post).tolist()))


def test_tensor_radius_reduces_realized_edges():
    base = nt.construct_neuronal_tensor(
        _probe_tensor(), seed=0, duration_ms=100.0, dt_ms=0.5)
    e0 = _tensor_edges(base)
    assert len(e0) > 0, "fixture must wire edges"
    pos = np.asarray(base.params["positions"], dtype=float)
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    radius = float(np.quantile(d[np.triu_indices(n, k=1)], 0.05))
    assert radius > 0.0, "fixture must have spread"
    cut = nt.construct_neuronal_tensor(
        _probe_tensor(spatial_radius=radius), seed=0, duration_ms=100.0, dt_ms=0.5)
    e1 = _tensor_edges(cut)
    assert 0 <= len(e1) < len(e0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "0.5", True])
def test_tensor_declaration_refuses_bad_radius(bad):
    with pytest.raises(ValueError, match="spatial_radius"):
        nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="E",
            mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
            max_in_degree=5, spatial_radius=bad)


def test_tensor_wire_refuses_radius_without_degree():
    tensor = _probe_tensor(max_in_degree=None, spatial_radius=0.5)
    with pytest.raises(ValueError, match="spatial_radius needs max_in_degree"):
        nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=100.0, dt_ms=0.5)


def test_tensor_radius_round_trips_through_save_load(tmp_path):
    tensor = _probe_tensor(spatial_radius=0.75)
    path = tmp_path / "radius_probe.json"
    nt.save_neuronal_tensor(tensor, path)
    back = nt.load(path)
    assert back.areas[0].inter_connections[0].spatial_radius == 0.75
    plain = _probe_tensor()
    entry = plain.to_dict()["areas"][0]["inter_connections"][0]
    assert "spatial_radius" not in entry
