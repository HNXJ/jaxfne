"""Structural closure tests for NeuronalTensor connectivity semantics."""

from __future__ import annotations

import json

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt


def _layer() -> nt.Layer:
    return nt.Layer(
        name="L1",
        n_neurons=4,
        neuron_types=[
            nt.NeuronType.make("E", fraction=0.5),
            nt.NeuronType.make("PV", fraction=0.5),
        ],
    )


def _model(tensor: nt.NeuronalTensor):
    return jtfne.construct(
        tensor,
        jtfne.RuntimeConfiguration(seed=0, duration_ms=2.0, dt_ms=0.5),
    )


def _pairs(model) -> set[tuple[int, int]]:
    edges = model.params["edge_list"]
    return {
        (int(pre), int(post))
        for pre, post in zip(np.asarray(edges.pre), np.asarray(edges.post))
    }


def test_unspecified_tensor_connectivity_uses_defaults(tmp_path):
    tensor = nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[_layer()])])

    model = _model(tensor)
    path = tmp_path / "unspecified.json"
    nt.save_neuronal_tensor(tensor, path)
    reloaded = nt.load_neuronal_tensor(path)

    assert tensor.connectivity_mode == "unspecified"
    assert reloaded.connectivity_mode == "unspecified"
    assert model.params["edge_list"].n_edges == 12
    assert model.cfg.metadata["connectivity_compilation"] == {
        "connectivity_mode": "unspecified",
        "default_edge_count": 12,
        "declared_rule_edge_count": 0,
        "total_compiled_edge_count": 12,
    }


def test_explicit_single_rule_has_no_hidden_defaults():
    tensor = nt.NeuronalTensor(
        areas=[
            nt.Area(
                name="V1",
                layers=[_layer()],
                inter_connections=[
                    nt.InterConnection("L1", "E", "L1", "PV", "AMPA")
                ],
            )
        ],
        area_connections=[],
    )

    model = _model(tensor)
    rows = model.neuron_table()
    edges = model.params["edge_list"]
    expected = {
        (pre, post)
        for pre, pre_row in enumerate(rows)
        for post, post_row in enumerate(rows)
        if pre_row["cell_type"] == "E" and post_row["cell_type"] == "PV"
    }

    assert tensor.connectivity_mode == "explicit"
    assert edges.n_edges == 4
    assert _pairs(model) == expected
    assert not np.any(np.asarray(edges.pre) == np.asarray(edges.post))
    assert model.cfg.metadata["connectivity_compilation"]["default_edge_count"] == 0
    assert model.cfg.metadata["connectivity_compilation"]["total_compiled_edge_count"] == 4


def test_explicit_empty_tensor_connectivity_compiles_zero_edges(tmp_path):
    tensor = nt.NeuronalTensor(
        areas=[
            nt.Area(name="V1", layers=[_layer()], inter_connections=[])
        ],
        area_connections=[],
    )

    model = _model(tensor)
    path = tmp_path / "explicit-empty.json"
    nt.save_neuronal_tensor(tensor, path)
    reloaded = nt.load_neuronal_tensor(path)

    assert tensor.connectivity_mode == "explicit"
    assert reloaded.connectivity_mode == "explicit"
    assert model.params["edge_list"].n_edges == 0
    assert model.cfg.metadata["connectivity_compilation"] == {
        "connectivity_mode": "explicit",
        "default_edge_count": 0,
        "declared_rule_edge_count": 0,
        "total_compiled_edge_count": 0,
    }


def test_legacy_empty_connection_arrays_keep_default_compatibility(tmp_path):
    tensor = nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[_layer()])])
    payload = tensor.to_dict()
    payload.pop("connectivity_mode")
    for area in payload["areas"]:
        area.pop("connectivity_mode")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = nt.load_neuronal_tensor(path)

    assert reloaded.connectivity_mode == "unspecified"


def test_contradictory_unspecified_mode_is_rejected():
    with pytest.raises(ValueError, match="cannot be marked as unspecified"):
        nt.Area(name="V1", layers=[_layer()], inter_connections=[], connectivity_mode="unspecified")
    with pytest.raises(ValueError, match="cannot be marked as unspecified"):
        nt.NeuronalTensor(
            areas=[nt.Area(name="V1", layers=[_layer()])],
            area_connections=[],
            connectivity_mode="unspecified",
        )


def test_duplicate_explicit_synaptic_identity_is_rejected():
    connection = nt.InterConnection("L1", "E", "L1", "PV", "AMPA")
    tensor = nt.NeuronalTensor(
        areas=[
            nt.Area(
                name="V1",
                layers=[_layer()],
                inter_connections=[connection, connection],
            )
        ]
    )

    with pytest.raises(ValueError, match="duplicate executable synaptic identities"):
        _model(tensor)


def test_distinct_parallel_mechanisms_preserve_typed_edges():
    tensor = nt.NeuronalTensor(
        areas=[
            nt.Area(
                name="V1",
                layers=[_layer()],
                inter_connections=[
                    nt.InterConnection(
                        "L1",
                        "E",
                        "L1",
                        "PV",
                        "AMPA",
                        static=nt.StaticParams(dT_ms=2.0),
                    ),
                    nt.InterConnection(
                        "L1",
                        "E",
                        "L1",
                        "PV",
                        "NMDA",
                        static=nt.StaticParams(dT_ms=100.0),
                    ),
                ],
            )
        ]
    )

    model = _model(tensor)
    edges = model.params["edge_list"]

    assert edges.n_edges == 8
    assert len(
        {
            (int(pre), int(post), int(receptor))
            for pre, post, receptor in zip(
                np.asarray(edges.pre),
                np.asarray(edges.post),
                np.asarray(edges.receptor_index),
            )
        }
    ) == 8
    assert set(np.asarray(edges.tau_ms).tolist()) == {2.0, 100.0}


def test_explicit_cross_area_topology_has_no_implicit_within_area_edges():
    layers = [
        nt.Layer(
            name="L1",
            n_neurons=2,
            neuron_types=[nt.NeuronType.make("E", fraction=1.0)],
        )
    ]
    tensor = nt.NeuronalTensor(
        areas=[nt.Area(name="V1", layers=layers), nt.Area(name="V4", layers=layers)],
        area_connections=[
            nt.AreaConnection("V1", "L1", "E", "V4", "L1", "E", mechanism="AMPA")
        ],
    )

    model = _model(tensor)
    rows = model.neuron_table()
    edges = model.params["edge_list"]

    assert edges.n_edges == 4
    assert all(
        rows[int(pre)]["area"] == "V1" and rows[int(post)]["area"] == "V4"
        for pre, post in zip(np.asarray(edges.pre), np.asarray(edges.post))
    )


def test_postconstruction_area_assignment_updates_authoritative_mode():
    layers = [
        nt.Layer(
            name="L1",
            n_neurons=2,
            neuron_types=[nt.NeuronType.make("E", fraction=1.0)],
        )
    ]
    tensor = nt.NeuronalTensor(
        areas=[nt.Area(name="V1", layers=layers), nt.Area(name="V4", layers=layers)]
    )
    tensor.area_connections = [
        nt.AreaConnection("V1", "L1", "E", "V4", "L1", "E", mechanism="AMPA")
    ]

    model = _model(tensor)

    assert tensor.connectivity_mode == "explicit"
    assert model.params["edge_list"].n_edges == 4
    assert model.cfg.metadata["connectivity_compilation"]["default_edge_count"] == 0


def test_removing_inferred_explicit_area_restores_unspecified_defaults():
    area = nt.Area(
        name="V1",
        layers=[_layer()],
        inter_connections=[nt.InterConnection("L1", "E", "L1", "PV", "AMPA")],
    )
    tensor = nt.NeuronalTensor(areas=[area])
    tensor.areas.clear()
    tensor.areas.append(nt.Area(name="V1", layers=[_layer()]))

    model = _model(tensor)

    assert model.cfg.metadata["connectivity_mode"] == "unspecified"
    assert model.params["edge_list"].n_edges == 12


def test_explicit_tensor_preserves_identity_geometry_sign_and_receptor():
    layer = nt.Layer(
        name="L4",
        n_neurons=4,
        neuron_types=[
            nt.NeuronType.make("E", fraction=0.5),
            nt.NeuronType.make("PV", fraction=0.5),
        ],
        geometry=nt.Geometry3D(
            x_range=(0.1, 0.2),
            y_range=(0.3, 0.4),
            z_range=(0.5, 0.6),
        ),
    )
    tensor = nt.NeuronalTensor(
        areas=[
            nt.Area(
                name="V1",
                layers=[layer],
                pose=nt.Pose3D(translation=(1.0, 2.0, 3.0)),
                inter_connections=[
                    nt.InterConnection(
                        "L4",
                        "E",
                        "L4",
                        "PV",
                        "AMPA",
                        static=nt.StaticParams(dT_ms=2.0),
                    ),
                    nt.InterConnection(
                        "L4",
                        "PV",
                        "L4",
                        "E",
                        "GABA_A",
                        static=nt.StaticParams(dT_ms=5.0),
                    ),
                ],
            )
        ]
    )

    model = _model(tensor)
    rows = model.neuron_table()
    positions = np.asarray(model.params["positions"])
    edges = model.params["edge_list"]

    assert all(row["area"] == "V1" and row["layer"] == "L4" for row in rows)
    assert {row["cell_type"] for row in rows} == {"E", "PV"}
    np.testing.assert_array_equal(
        positions,
        np.asarray([[row["x"], row["y"], row["z"]] for row in rows]),
    )
    assert np.all((positions[:, 0] >= 1.1) & (positions[:, 0] <= 1.2))
    assert np.all((positions[:, 1] >= 2.3) & (positions[:, 1] <= 2.4))
    assert np.all((positions[:, 2] >= 3.5) & (positions[:, 2] <= 3.6))

    labels = np.asarray([row["cell_type"] for row in rows])
    signs = np.asarray(edges.weight)
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    assert np.all(signs[(labels[pre] == "E") & (labels[post] == "PV")] > 0)
    assert np.all(signs[(labels[pre] == "PV") & (labels[post] == "E")] < 0)
    assert set(np.asarray(edges.tau_ms).tolist()) == {2.0, 5.0}
