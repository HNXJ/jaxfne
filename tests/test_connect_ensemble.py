"""``jtfne.connect`` — fuse constructed Models into one ensemble Model.

``construct`` builds one model; ``connect`` joins several. The merge is purely
structural and stays on the sparse edge_list backend: members' neurons, emitter
params, positions, and recurrence are concatenated with index offsets, optional
cross-model edges are compiled, and truth gates never escalate.
"""
import dataclasses

import numpy as np
import pytest

import jaxfne as jtfne


def _col(name, n, seed, *, layers=("L2/3", "L4"), dt_ms=0.5):
    cfg = (
        jtfne.Configuration()
        .runtime(duration_ms=40, dt_ms=dt_ms, seed=seed)
        .column(name, layers=list(layers), n=n)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _unpack(ens):
    el = ens.params["edge_list"]
    pre = np.asarray(el.pre); post = np.asarray(el.post); w = np.asarray(el.weight)
    nt = ens.neuron_table()
    model = np.asarray([r["model"] for r in nt])
    return ens.params["emitter"], el, pre, post, w, model, nt


# --- structural merge ---------------------------------------------------------

def test_connect_requires_two_models():
    with pytest.raises(ValueError, match="at least two"):
        jtfne.connect(_col("V1", 10, 0))


def test_connect_rejects_non_model():
    with pytest.raises(TypeError, match="not a Model"):
        jtfne.connect(_col("V1", 10, 0), {"not": "a model"})


def test_connect_n_total_is_sum_of_members():
    m1, m2 = _col("V1", 30, 0), _col("V2", 24, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    assert ens.params["emitter"].n_neurons == (
        m1.params["emitter"].n_neurons + m2.params["emitter"].n_neurons
    )


def test_connect_W_is_placeholder_edge_list_backend():
    ens = jtfne.connect(_col("V1", 20, 0), _col("V2", 20, 1), namespace=("A", "B"))
    assert tuple(ens.params["emitter"].W.shape) == (0, 0)
    assert "edge_list" in ens.params


def test_connect_index_integrity():
    em, el, pre, post, w, model, nt = _unpack(
        jtfne.connect(_col("V1", 25, 0), _col("V2", 20, 1), namespace=("A", "B"))
    )
    assert pre.min() >= 0 and post.min() >= 0
    assert int(pre.max()) < em.n_neurons and int(post.max()) < em.n_neurons


def test_connect_preserves_member_internal_edges():
    m1, m2 = _col("V1", 25, 0), _col("V2", 20, 1)
    n1 = m1.params["edge_list"].n_edges
    n2 = m2.params["edge_list"].n_edges
    em, el, pre, post, w, model, nt = _unpack(jtfne.connect(m1, m2, namespace=("A", "B")))
    # no cross edges declared -> all edges are internal to one member, offset-preserved
    assert int(((model[pre] == 0) & (model[post] == 0)).sum()) == n1
    assert int(((model[pre] == 1) & (model[post] == 1)).sum()) == n2
    assert int(((model[pre] == 0) & (model[post] == 1)).sum()) == 0


def test_connect_model_label_in_neuron_table():
    ens = jtfne.connect(_col("V1", 12, 0), _col("V2", 8, 1), namespace=("A", "B"))
    nt = ens.neuron_table()
    assert {r["model"] for r in nt} == {0, 1}
    assert sum(r["model"] == 0 for r in nt) == 12
    assert sum(r["model"] == 1 for r in nt) == 8


# --- cross-model edges --------------------------------------------------------

def _ff(probability=1.0, weight=0.5, sign="excitatory", source_extra=None):
    src = {"model": 0, "area": "V1", "layers": ["L2/3"]}
    if source_extra:
        src.update(source_extra)
    return [dict(source=src, target={"model": 1, "area": "V2", "layers": ["L4"]},
                 probability=probability, weight=weight, sign=sign)]


def test_connect_cross_edges_route_correctly():
    em, el, pre, post, w, model, nt = _unpack(jtfne.connect(
        _col("V1", 40, 0), _col("V2", 40, 1), namespace=("A", "B"),
        edges=_ff(source_extra={"cell_type": "E"})))
    cross = (model[pre] == 0) & (model[post] == 1)
    assert int(cross.sum()) > 0
    layer = np.asarray([r["layer"] for r in nt])
    cell = np.asarray([r["cell_type"] for r in nt])
    assert set(layer[pre[cross]]) <= {"L2", "L3", "L2/3"}   # source superficial (merged or split)
    assert set(layer[post[cross]]) == {"L4"}           # target L4
    assert set(cell[pre[cross]]) == {"E"}              # source E only


def test_connect_cross_edge_probability_zero_no_edges():
    em, el, pre, post, w, model, nt = _unpack(jtfne.connect(
        _col("V1", 30, 0), _col("V2", 30, 1), namespace=("A", "B"),
        edges=_ff(probability=0.0)))
    assert int(((model[pre] == 0) & (model[post] == 1)).sum()) == 0


def test_connect_cross_edge_sign_excitatory_positive():
    em, el, pre, post, w, model, nt = _unpack(jtfne.connect(
        _col("V1", 30, 0), _col("V2", 30, 1), namespace=("A", "B"), edges=_ff(sign="excitatory")))
    cross = (model[pre] == 0) & (model[post] == 1)
    assert bool((w[cross] > 0).all())


def test_connect_cross_edge_sign_inhibitory_negative():
    em, el, pre, post, w, model, nt = _unpack(jtfne.connect(
        _col("V1", 30, 0), _col("V2", 30, 1), namespace=("A", "B"), edges=_ff(sign="inhibitory")))
    cross = (model[pre] == 0) & (model[post] == 1)
    assert bool((w[cross] < 0).all())


def test_connect_cross_edge_signed_uses_presynaptic_sign():
    # source spans all cell types -> "signed" weight follows presynaptic neuron sign
    em, el, pre, post, w, model, nt = _unpack(jtfne.connect(
        _col("V1", 40, 0), _col("V2", 40, 1), namespace=("A", "B"), edges=_ff(sign="signed")))
    cross = (model[pre] == 0) & (model[post] == 1)
    cell = np.asarray([r["cell_type"] for r in nt])
    assert bool((w[cross][cell[pre[cross]] == "E"] > 0).all())
    inh = cell[pre[cross]] != "E"
    if inh.any():
        assert bool((w[cross][inh] < 0).all())


def test_connect_cross_rule_no_match_flagged():
    ens = jtfne.connect(
        _col("V1", 20, 0), _col("V2", 20, 1), namespace=("A", "B"),
        edges=[dict(source={"model": 0, "area": "V1"}, target={"model": 1, "area": "GHOST"},
                    probability=1.0)])
    rule = ens.cfg.metadata["circuit"]["connections"][-1]
    assert rule["status"] == "compiled_no_matching_edges"
    assert rule["compiled_n_edges"] == 0
    assert ens.cfg.metadata["ensemble"]["cross_model_edges"] == 0


# --- simulate / tune integration ---------------------------------------------

def test_connect_simulate_stays_finite():
    ens = jtfne.connect(
        _col("V1", 30, 0), _col("V2", 24, 1), namespace=("A", "B"),
        edges=_ff(source_extra={"cell_type": "E"}))
    sig = jtfne.simulate(ens, duration_ms=40, dt_ms=0.5, seed=2)
    assert np.isfinite(np.asarray(sig.get("vm"))).all()
    assert np.isfinite(np.asarray(sig.get("spk"))).all()


# --- truth gates --------------------------------------------------------------

def test_connect_truth_gates_not_escalated():
    ens = jtfne.connect(_col("V1", 12, 0), _col("V2", 12, 1), namespace=("A", "B"))
    md = ens.cfg.metadata
    assert md["claim_level"] == "computational_scaffold"
    assert md["field_solver_status"] == "linear_solver"
    assert md["physical_amplitude_calibrated"] is False


# --- reconciliation guards ----------------------------------------------------

def test_connect_dt_mismatch_raises_strict():
    with pytest.raises(ValueError, match="dt_ms"):
        jtfne.connect(_col("V1", 10, 0, dt_ms=0.5), _col("V2", 10, 1, dt_ms=0.25),
                      namespace=("A", "B"))


def test_connect_area_collision_raises_without_namespace():
    with pytest.raises(ValueError, match="collision"):
        jtfne.connect(_col("V1", 10, 0), _col("V1", 10, 1))


def test_connect_namespace_disambiguates_collision():
    ens = jtfne.connect(_col("V1", 10, 0), _col("V1", 10, 1), namespace=("A", "B"))
    areas = {r["area"] for r in ens.neuron_table()}
    assert areas == {"A/V1", "B/V1"}


def test_connect_n_contacts_mismatch_raises_strict():
    m1 = _col("V1", 10, 0)
    m2 = _col("V2", 10, 1)
    m2 = dataclasses.replace(m2, static={**m2.static, "n_contacts": 32})
    with pytest.raises(ValueError, match="n_contacts"):
        jtfne.connect(m1, m2, namespace=("A", "B"))


def test_connect_namespace_wrong_length_raises():
    with pytest.raises(ValueError, match="one entry per model"):
        jtfne.connect(_col("V1", 10, 0), _col("V2", 10, 1), namespace=("A",))


# --- layout & associativity ---------------------------------------------------

def test_connect_offset_x_separates_member_positions():
    ens = jtfne.connect(_col("V1", 20, 0), _col("V2", 20, 1), namespace=("A", "B"),
                        layout="offset_x")
    pos = np.asarray(ens.params["positions"])
    model = np.asarray([r["model"] for r in ens.neuron_table()])
    x0 = pos[model == 0, 0]; x1 = pos[model == 1, 0]
    assert x1.min() >= x0.max()           # member 1 placed to the right of member 0


def test_connect_three_models_associative_counts():
    a, b, c = _col("V1", 12, 0), _col("V2", 12, 1), _col("V3", 12, 2)
    flat = jtfne.connect(a, b, c, namespace=("A", "B", "C"))
    nested = jtfne.connect(jtfne.connect(a, b, namespace=("A", "B")), c, namespace=("AB", "C"))
    assert flat.params["emitter"].n_neurons == nested.params["emitter"].n_neurons == 36
    assert flat.params["edge_list"].n_edges == nested.params["edge_list"].n_edges


def test_connect_ensemble_metadata_recorded():
    ens = jtfne.connect(_col("V1", 15, 0), _col("V2", 10, 1), namespace=("A", "B"), name="vis_hierarchy")
    meta = ens.cfg.metadata["ensemble"]
    assert meta["n_models"] == 2
    assert meta["model_sizes"] == [15, 10]
    assert meta["n_neurons_total"] == 25
    assert meta["name"] == "vis_hierarchy"
    assert meta["layout"] == "offset_x"
