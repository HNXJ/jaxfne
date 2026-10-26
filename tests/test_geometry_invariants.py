"""Geometry invariants (S3, C4): round trip, rigid motion, identity through flattening.

Small model: 20 to 30 neurons, 100 ms, one seed. The rigid motion goes through the
public augment API (``GeometryTransform`` with ``PoseEdit``). The connection rule is
distance-only (localized sampling: ``max_in_degree`` with ``spatial_sigma``).

One case is an open finding and is marked xfail(strict=True): an arbitrary rigid
translation changes the realized edge set of the distance-only rule (inferred cause:
the sampler's absolute spatial buckets).
"""
import json
from collections import Counter

import numpy as np
import pytest
from numpy.testing import assert_allclose

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.augment import AugmentationSpec, GeometryTransform, PoseEdit, augment
from jaxfne.emitters import CELL_TYPE_SIGN

DUR_MS = 100.0
DT_MS = 0.5
SEED = 0


def _probe_tensor(translation=(1.0, 2.0, 3.0)):
    """One area, 20 neurons, one distance-only (localized) E -> PV connection."""
    layer = nt.Layer(
        name="L4",
        n_neurons=20,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=2.0, H=1.0),
        max_in_degree=5, spatial_sigma=1.0,
    )
    area = nt.Area(
        name="V1", layers=[layer], inter_connections=[conn],
        pose=nt.Pose3D(translation=translation),
    )
    return nt.NeuronalTensor(areas=[area], name="geometry_invariance_probe")


def _nested_tensor():
    """Two areas, three layers, 30 neurons; the type fractions give exact integer counts."""
    v1 = nt.Area(
        name="V1",
        layers=[
            nt.Layer(name="L2", n_neurons=12, neuron_types=[
                nt.NeuronType.make("E", fraction=0.75),
                nt.NeuronType.make("PV", fraction=0.25),
            ]),
            nt.Layer(name="L4", n_neurons=10, neuron_types=[
                nt.NeuronType.make("E", fraction=0.5),
                nt.NeuronType.make("PV", fraction=0.5),
            ]),
        ],
        inter_connections=[nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="PV",
            mechanism="AMPA", static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=2.0, H=1.0), delay_ms=1.0,
        )],
        pose=nt.Pose3D(translation=(0.0, 0.0, 0.0)),
    )
    v4 = nt.Area(
        name="V4",
        layers=[nt.Layer(name="L4", n_neurons=8, neuron_types=[
            nt.NeuronType.make("E", fraction=0.5),
            nt.NeuronType.make("PV", fraction=0.5),
        ])],
        pose=nt.Pose3D(translation=(20.0, 0.0, 0.0)),
    )
    return nt.NeuronalTensor(areas=[v1, v4], name="geometry_invariance_nested")


def _build(tensor):
    return nt.construct_neuronal_tensor(tensor, seed=SEED, duration_ms=DUR_MS, dt_ms=DT_MS)


def _edge_set(model):
    el = model.params["edge_list"]
    return set(zip(np.asarray(el.pre).tolist(), np.asarray(el.post).tolist()))


def _spikes(model):
    return np.asarray(jtfne.simulate(model, duration_ms=DUR_MS, dt_ms=DT_MS, seed=SEED).spikes)


def _rigid_move(tensor, rotation_deg, translation):
    """Add a rotation and a translation to area V1's pose via the augment API."""
    spec = AugmentationSpec(transforms=[GeometryTransform(pose_edits=[
        PoseEdit(area="V1", rotation_deg=rotation_deg, translation=translation),
    ])])
    moved, record = augment(tensor, spec)
    assert record.realized_order == ("G",)
    return moved


def _positions(model):
    return np.asarray(model.params["positions"], dtype=np.float64)


# (a) round trip: serialize, load, rebuild; count, labels, signs and positions kept.
def test_round_trip_keeps_count_labels_signs_and_positions(tmp_path):
    tensor = _nested_tensor()
    path = tmp_path / "geometry_roundtrip.json"
    nt.save_neuronal_tensor(tensor, path)
    loaded = nt.load(path)
    # JSON turns tuples into lists; normalize both sides through JSON before comparing.
    assert json.loads(json.dumps(loaded.to_dict())) == json.loads(json.dumps(tensor.to_dict()))

    base = _build(tensor)
    back = _build(loaded)
    base_rows = base.neuron_table()
    back_rows = back.neuron_table()
    assert len(base_rows) == 30, "fixture must hold 30 neurons"
    assert len(back_rows) == len(base_rows)
    assert [r["cell_type"] for r in back_rows] == [r["cell_type"] for r in base_rows]
    assert [r["area"] for r in back_rows] == [r["area"] for r in base_rows]
    assert np.array_equal(np.asarray(back.params["emitter"].sign),
                          np.asarray(base.params["emitter"].sign))
    base_pos = _positions(base)
    assert np.ptp(base_pos[:, 0]) > 15.0, "fixture must place two areas apart"
    assert_allclose(_positions(back), base_pos, rtol=1e-12)


# (b1) rigid motion: pairwise distances kept, and positions equal the rigid map of the base.
def test_rigid_motion_keeps_pairwise_distances_and_maps_positions():
    base_t = _probe_tensor(translation=(1.0, 2.0, 3.0))
    translation = (2.0, -2.0, -3.0)
    rotation = 30.0
    moved_t = _rigid_move(base_t, rotation, translation)

    base_pos = _positions(_build(base_t))
    moved_pos = _positions(_build(moved_t))
    assert base_pos.shape == (20, 3)
    assert np.max(np.abs(moved_pos - base_pos)) > 1.0, "fixture must move positions"

    # Expected: local = base - (1, 2, 3); moved = R(rotation) @ local + (1, 2, 3) + translation.
    theta = np.deg2rad(rotation)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]])
    local = base_pos - np.array([1.0, 2.0, 3.0])
    expected = local @ rot.T + np.array([1.0, 2.0, 3.0]) + np.array(translation)
    assert_allclose(moved_pos, expected, rtol=1e-5, atol=1e-5)

    d0 = np.linalg.norm(base_pos[:, None, :] - base_pos[None, :, :], axis=-1)
    d1 = np.linalg.norm(moved_pos[:, None, :] - moved_pos[None, :, :], axis=-1)
    assert float(d0.max()) > 0.1, "fixture must have spread"
    # Positions are float32 on the model; distances tolerate that rounding.
    assert_allclose(d1, d0, rtol=1e-5, atol=1e-6)


# (b2) rigid motion, distance-only rule: edge set and spikes kept under the same seed.
# The translation is one spatial bucket (3 * spatial_sigma = 3 mm). Inferred mechanism (not
# traced in code): the bucket neighbourhood maps onto itself, so the sampler sees the same
# candidates in the same order as the base model.
def test_bucket_aligned_rigid_motion_keeps_localized_edges_and_spikes():
    base = _build(_probe_tensor(translation=(1.0, 2.0, 3.0)))
    moved = _build(_rigid_move(_probe_tensor(translation=(1.0, 2.0, 3.0)), 0.0, (3.0, 0.0, 0.0)))
    e0 = _edge_set(base)
    s0 = _spikes(base)
    assert len(e0) > 0, "fixture must wire edges"
    assert s0.sum() > 0, "fixture must spike"
    assert _edge_set(moved) == e0
    assert np.array_equal(_spikes(moved), s0)


# (b3) open finding: an arbitrary rigid translation changes the localized edge set.
@pytest.mark.xfail(
    strict=True,
    reason=(
        "localized sampling (connectivity._candidate_pairs_localized) anchors its grid "
        "buckets to absolute coordinates; a rigid translation of (2, -2, -3) mm changes the "
        "realized edge set (observed: 27 of 50 edges differ; the spike comparison is not "
        "reached; inferred: the cloud crosses a bucket boundary and the candidate order changes)"
    ),
)
def test_arbitrary_rigid_translation_keeps_localized_edges_and_spikes():
    base = _build(_probe_tensor(translation=(1.0, 2.0, 3.0)))
    moved = _build(_rigid_move(_probe_tensor(translation=(1.0, 2.0, 3.0)), 0.0, (2.0, -2.0, -3.0)))
    e0 = _edge_set(base)
    s0 = _spikes(base)
    assert len(e0) > 0, "fixture must wire edges"
    assert s0.sum() > 0, "fixture must spike"
    assert _edge_set(moved) == e0
    assert np.array_equal(_spikes(moved), s0)


# (c) identity kept through flattening: area, layer, cell type, sign and position per neuron.
def test_flattened_neurons_keep_area_layer_label_and_sign():
    tensor = _nested_tensor()
    model = _build(tensor)
    rows = model.neuron_table()
    emitter = model.params["emitter"]
    assert len(rows) == 30 == int(emitter.n_neurons), "fixture must hold 30 neurons"

    expected = Counter({
        ("V1", "L2", "E"): 9, ("V1", "L2", "PV"): 3,
        ("V1", "L4", "E"): 5, ("V1", "L4", "PV"): 5,
        ("V4", "L4", "E"): 4, ("V4", "L4", "PV"): 4,
    })
    assert Counter((r["area"], r["layer"], r["cell_type"]) for r in rows) == expected

    pos = _positions(model)
    for i, row in enumerate(rows):
        assert row["neuron_id"] == i
        assert emitter.labels[i] == row["cell_type"]
        assert emitter.layer_labels[i] == row["layer"]
        assert float(emitter.sign[i]) == CELL_TYPE_SIGN[row["cell_type"]]
        if row["area"] == "V4":
            assert pos[i, 0] > 15.0, "V4 neurons sit 20 mm out"
        else:
            assert pos[i, 0] < 5.0, "V1 neurons sit at the origin"
