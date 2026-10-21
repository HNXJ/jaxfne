"""Plural selector keys on the mechanism-aware compile path.

``.connections()`` documents ``layer``/``layers`` and ``cell_type``/``cell_types``
selectors. The mechanism-aware compiler (``connectivity.compile_connection_rules``)
matched each key as an exact neuron-row field, so ``layers``/``cell_types`` matched
no neuron and the rule compiled zero edges without a message. These tests build
the same rule with singular and plural keys and require identical edges.
"""

import numpy as np

import jaxfne as jtfne


def _model(source, target=None, mechanism=True):
    cfg = (
        jtfne.configuration()
        .population(100, neurons={"A": 1, "B": 1}, layers=["A", "B"], name="col")
        .cell_types({"E": 0.5, "SST": 0.5})
        .layer_fractions(
            {"A": (0.0, 0.5), "B": (0.5, 1.0)},
            {"A": {"E": 0.5, "SST": 0.5}, "B": {"E": 0.5, "SST": 0.5}},
        )
        .set_emitter("izhikevich", "cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", modes=["spikes"], n_contacts=4)
    )
    kw = {}
    if mechanism:
        cfg = cfg.mechanisms(name="syn", kind="exp_synapse", params={"tau_ms": 20.0})
        kw["mechanism"] = "syn"
    cfg = cfg.connections(
        name="r", source=source, target=target or {"cell_type": "E"}, weight=0.1,
        sign="inhibitory", probability=0.5, **kw,
    )
    return jtfne.construct(cfg.update_metadata(connectivity_mode="explicit"))


def _edges(model):
    el = model.params["edge_list"]
    nt = model.neuron_table()
    layer = np.array([r["layer"] for r in nt])
    ctype = np.array([r["cell_type"] for r in nt])
    pre, post = np.asarray(el.pre), np.asarray(el.post)
    return pre, post, layer, ctype


def test_layers_list_matches_single_layer_on_mechanism_path():
    pre_s, post_s, layer, ctype = _edges(_model({"cell_type": "SST", "layer": "A"}))
    pre_p, post_p, _, _ = _edges(_model({"cell_type": "SST", "layers": ["A"]}))
    assert len(pre_s) > 0, "fixture must compile edges from layer A"
    assert np.all(layer[pre_p] == "A") and np.all(ctype[pre_p] == "SST")
    np.testing.assert_array_equal(pre_p, pre_s)
    np.testing.assert_array_equal(post_p, post_s)


def test_layers_list_is_a_union():
    pre_all, _, _, _ = _edges(_model({"cell_type": "SST"}))
    pre_ab, _, layer, _ = _edges(_model({"cell_type": "SST", "layers": ["A", "B"]}))
    assert set(layer[pre_ab]) == {"A", "B"}
    np.testing.assert_array_equal(pre_ab, pre_all)


def test_layers_list_restricts():
    pre_s, post_s, _, _ = _edges(_model({"cell_type": "SST", "layer": "A"}))
    pre_p, post_p, _, _ = _edges(_model({"cell_type": "SST", "layers": ["A", "absent"]}))
    assert len(pre_s) > 0, "fixture must compile edges from layer A"
    np.testing.assert_array_equal(pre_p, pre_s)
    np.testing.assert_array_equal(post_p, post_s)


def test_cell_types_list_on_target():
    _, post_s, _, ctype = _edges(_model({"layer": "A"}, target={"cell_type": "E"}))
    _, post_p, _, _ = _edges(_model({"layer": "A"}, target={"cell_types": ["E"]}))
    assert len(post_s) > 0 and np.all(ctype[post_p] == "E")
    np.testing.assert_array_equal(post_p, post_s)


def test_scan_path_matches_plural_keys():
    from jaxfne.connectivity import _select_indices

    rows = [{"neuron_id": i, "layer": lyr, "cell_type": ct}
            for i, (lyr, ct) in enumerate([("A", "E"), ("A", "SST"), ("B", "SST"), ("C", "SST")])]
    got = _select_indices(rows, {"cell_types": ["SST"], "layers": ["A", "C"]}, allow_empty=False, label="t")
    assert got == [1, 3]
    rows = [{"neuron_id": i, "layer": lyr} for i, lyr in enumerate(["L23", "L4", "L"])]
    assert _select_indices(rows, {"layers": "L23"}, allow_empty=False, label="t") == [0]


def test_singular_and_plural_together_refused():
    import pytest

    with pytest.raises(ValueError, match="layer"):
        _model({"cell_type": "SST", "layer": "A", "layers": ["A"]})
