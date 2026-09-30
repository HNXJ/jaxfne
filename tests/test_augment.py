"""Packet 1 augmentation: clone, cardinality scaling (N), geometry (G).

Tiny-first suite (N0 = 10, scaling <= 100x steps to 100 and 1000). One test
per behaviour; fixtures mirror tests/test_neuronal_tensor.py.
"""
import math

import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.augment import (
    PACKET2_AXES,
    AugmentationSpec,
    GeometryTransform,
    PoseEdit,
    RangeEdit,
    ScaleN,
    ThetaC,
    ThetaX,
    W0,
    H0,
    augment,
    clone_tensor,
)


def _tiny_tensor(n=10):
    layer = nt.Layer(
        name="L4",
        n_neurons=n,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=2.0, H=1.0),
    )
    area = nt.Area(name="V1", layers=[layer], inter_connections=[conn])
    return nt.NeuronalTensor(areas=[area], name="tiny_aug")


def _build(tensor, seed=0):
    return nt.construct_neuronal_tensor(tensor, seed=seed, duration_ms=5.0, dt_ms=0.5)


def _strip_geometry(payload):
    """Return payload with pose + per-layer geometry removed (isolation check)."""
    import copy

    payload = copy.deepcopy(payload)
    for area in payload["areas"]:
        area.pop("pose", None)
        for layer in area.get("layers", []):
            layer.pop("geometry", None)
    return payload


def test_determinism_same_spec_twice_equal():
    tensor = _tiny_tensor()
    spec = AugmentationSpec(transforms=[ScaleN(factor=10)])
    out_a, rec_a = augment(tensor, spec)
    out_b, rec_b = augment(tensor, spec)
    assert out_a.to_dict() == out_b.to_dict()
    assert rec_a == rec_b
    assert rec_a.spec_digest == rec_b.spec_digest


def test_clone_identity_and_bitexact_simulation():
    tensor = _tiny_tensor()
    out, record = clone_tensor(tensor)
    assert out.to_dict() == tensor.to_dict()
    assert out is not tensor
    assert out.areas[0] is not tensor.areas[0]
    assert record.changes == ()
    assert record.realized_order == ()
    base = _build(tensor, seed=0)
    clone = _build(out, seed=0)
    sig_base = jtfne.simulate(base, duration_ms=5.0, dt_ms=0.5, seed=0)
    sig_clone = jtfne.simulate(clone, duration_ms=5.0, dt_ms=0.5, seed=0)
    assert bool(jnp.all(sig_clone.spikes == sig_base.spikes))
    assert bool(jnp.all(sig_clone.V_m == sig_base.V_m))


def test_scale_n_10_to_100_to_1000():
    from jaxfne._config import _counts_from_fractions

    base = _tiny_tensor(n=10)
    mid, _ = augment(base, AugmentationSpec(transforms=[ScaleN(factor=10)]))
    assert mid.areas[0].layers[0].n_neurons == 100
    big, _ = augment(mid, AugmentationSpec(transforms=[ScaleN(factor=10)]))
    assert big.areas[0].layers[0].n_neurons == 1000
    model = _build(big, seed=0)
    rows = model.neuron_table()
    assert len(rows) == 1000
    assert model.params["emitter"].n_neurons == 1000
    # Per-type allocation is whatever the existing rules give (even E/PV split).
    expected = _counts_from_fractions(1000, {"E": 0.5, "PV": 0.5})
    got = {"E": 0, "PV": 0}
    for row in rows:
        got[row["cell_type"]] += 1
    assert got == expected


def test_scale_n_edge_weight_follows_sqrt_formula():
    base = _tiny_tensor(n=10)
    scaled, _ = augment(base, AugmentationSpec(transforms=[ScaleN(factor=10)]))
    model = _build(scaled, seed=0)
    rules = model.cfg.metadata["circuit"]["connections"]
    assert len(rules) == 1
    conn = scaled.areas[0].inter_connections[0]
    total_n = 100
    expected = abs(float(conn.plastic.w_mech) * float(conn.static.g_mech["AMPA"])) / math.sqrt(total_n)
    assert rules[0]["weight"] == pytest.approx(expected)
    assert rules[0]["weight"] == pytest.approx(
        nt._connection_edge_weight(conn, total_n)
    )


def test_scale_n_refusals():
    tensor = _tiny_tensor(n=10)
    for bad in (0, -2, float("nan"), float("inf"), "10", True, None):
        with pytest.raises((ValueError, TypeError)):
            augment(tensor, AugmentationSpec(transforms=[ScaleN(factor=bad)]))
    with pytest.raises(ValueError):
        augment(tensor, AugmentationSpec(transforms=[ScaleN(factor=0.15)]))


def test_duplicate_axis_record_refused():
    with pytest.raises(ValueError):
        AugmentationSpec(transforms=[ScaleN(factor=2), ScaleN(factor=3)])
    with pytest.raises(ValueError):
        AugmentationSpec(transforms=[object()])


def test_every_axis_is_now_implemented():
    tensor = _tiny_tensor()
    for rec in (ThetaC(), ThetaX(), H0()):
        out, record = augment(tensor, AugmentationSpec(transforms=[rec]))
        assert out.to_dict() == tensor.to_dict()  # defaults are no-ops
        assert record.realized_order == ()
    assert PACKET2_AXES == frozenset()


def test_stochastic_without_kv_refused():
    with pytest.raises(ValueError, match="K_V"):
        AugmentationSpec(transforms=[ThetaX(g_jitter=0.1)])
    spec = AugmentationSpec(transforms=[ThetaX(g_jitter=0.1)], k_v=7)
    assert spec.k_v == 7
    out, record = augment(_tiny_tensor(), spec)
    assert record.realized_order == ("Theta_X",)


def test_geometry_pose_shift_exact_and_spikes_unchanged():
    tensor = _tiny_tensor()
    delta = (10.0, -5.0, 2.0)
    spec = AugmentationSpec(
        transforms=[GeometryTransform(pose_edits=[PoseEdit(area="V1", translation=delta)])]
    )
    out, record = augment(tensor, spec)
    assert record.realized_order == ("G",)
    assert len(record.changes) == 1
    assert all(entry.origin == "augmented" for entry in record.changes)
    base = _build(tensor, seed=0)
    moved = _build(out, seed=0)
    shift = moved.params["positions"] - base.params["positions"]
    assert bool(jnp.all(jnp.isfinite(shift)))
    assert jnp.allclose(shift, jnp.asarray(delta, dtype=shift.dtype), atol=1e-6).item()
    sig_base = jtfne.simulate(base, duration_ms=5.0, dt_ms=0.5, seed=0)
    sig_moved = jtfne.simulate(moved, duration_ms=5.0, dt_ms=0.5, seed=0)
    assert bool(jnp.all(sig_moved.spikes == sig_base.spikes))


def test_geometry_range_bounds_and_isolation():
    tensor = _tiny_tensor()
    spec = AugmentationSpec(
        transforms=[
            GeometryTransform(
                range_edits=[RangeEdit(area="V1", layer="L4", x_range=(0.25, 0.5))]
            )
        ]
    )
    out, record = augment(tensor, spec)
    assert out.areas[0].layers[0].geometry.x_range == (0.25, 0.5)
    assert len(record.changes) == 1
    # Non-geometry fields are untouched (axis isolation).
    assert _strip_geometry(out.to_dict()) == _strip_geometry(tensor.to_dict())
    # Constructed x positions sit inside the declared column-relative band.
    model = _build(out, seed=0)
    radius = 0.25
    lo, hi = -radius + 0.25 * 2 * radius, -radius + 0.5 * 2 * radius
    xs = model.params["positions"][:, 0]
    assert bool(jnp.all(xs >= lo - 1e-6)) and bool(jnp.all(xs <= hi + 1e-6))


def test_geometry_refusals():
    tensor = _tiny_tensor()
    with pytest.raises(ValueError, match="unknown area"):
        augment(
            tensor,
            AugmentationSpec(
                transforms=[GeometryTransform(pose_edits=[PoseEdit(area="NX")])]
            ),
        )
    with pytest.raises(ValueError, match="unknown layer"):
        augment(
            tensor,
            AugmentationSpec(
                transforms=[GeometryTransform(range_edits=[RangeEdit(area="V1", layer="NX")])]
            ),
        )
    for bad_range in ((0.5, 1.5), (-0.1, 0.5), (0.7, 0.3)):
        with pytest.raises(ValueError):
            augment(
                tensor,
                AugmentationSpec(
                    transforms=[
                        GeometryTransform(
                            range_edits=[RangeEdit(area="V1", layer="L4", x_range=bad_range)]
                        )
                    ]
                ),
            )


def test_canonical_order_g_before_n_realizes_n_then_g():
    tensor = _tiny_tensor(n=10)
    spec = AugmentationSpec(
        transforms=[
            GeometryTransform(pose_edits=[PoseEdit(area="V1", translation=(1.0, 0.0, 0.0))]),
            ScaleN(factor=10),
        ]
    )
    out, record = augment(tensor, spec)
    assert record.realized_order == ("N", "G")
    assert out.areas[0].layers[0].n_neurons == 100
    assert out.areas[0].pose.translation == (1.0, 0.0, 0.0)


def test_input_tensor_unchanged():
    tensor = _tiny_tensor()
    before = tensor.to_dict()
    augment(
        tensor,
        AugmentationSpec(
            transforms=[
                ScaleN(factor=10),
                GeometryTransform(pose_edits=[PoseEdit(area="V1", translation=(1.0, 0.0, 0.0))]),
            ]
        ),
    )
    assert tensor.to_dict() == before


def _fraction_tensor(n=10):
    layer = nt.Layer(
        name="L4",
        n_neurons=n,
        neuron_types=[
            nt.NeuronType.make("E", fraction=0.8),
            nt.NeuronType.make("PV", fraction=0.1),
            nt.NeuronType.make("SST", fraction=0.07),
            nt.NeuronType.make("VIP", fraction=0.03),
        ],
    )
    return nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[layer])], name="frac_aug")


def _two_area_tensor(n=10):
    def _layer(name):
        return nt.Layer(
            name=name,
            n_neurons=n,
            neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
        )

    a0 = nt.Area(name="A0", layers=[_layer("L4")])
    a1 = nt.Area(name="A1", layers=[_layer("L4")])
    return nt.NeuronalTensor(areas=[a0, a1], name="two_area_aug")


def _two_area_two_layer_tensor(n=10):
    def _layer(name):
        return nt.Layer(
            name=name,
            n_neurons=n,
            neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
        )

    def _conn(layer):
        return nt.InterConnection(
            source_layer=layer, source_neuron_type="E",
            target_layer=layer, target_neuron_type="PV",
            mechanism="AMPA",
            static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=2.0, H=1.0),
        )

    a0 = nt.Area(
        name="A0",
        layers=[_layer("L2/3"), _layer("L5")],
        inter_connections=[_conn("L2/3"), _conn("L5")],
    )
    a1 = nt.Area(
        name="A1",
        layers=[_layer("L2/3"), _layer("L5")],
        inter_connections=[_conn("L2/3")],
    )
    cross = nt.AreaConnection(
        source_area="A0", source_layer="L2/3", source_neuron_type="E",
        target_area="A1", target_layer="L5", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=1.5, H=1.0),
    )
    return nt.NeuronalTensor(areas=[a0, a1], area_connections=[cross], name="complete_aug")


def test_scale_n_zeroed_type_refused_but_clone_and_unit_scale_pass():
    from jaxfne._config import _counts_from_fractions

    fracs = {"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}
    assert _counts_from_fractions(10, fracs)["VIP"] == 0, "fixture must zero VIP at N0=10"
    assert all(v > 0 for v in _counts_from_fractions(100, fracs).values()), (
        "fixture must realize every type at n=100"
    )
    base = _fraction_tensor(n=10)
    # Clone applies no scaling, so it must not refuse even though the bridge
    # allocation realizes 0 VIP neurons at this size.
    out, record = clone_tensor(base)
    assert out.to_dict() == base.to_dict()
    assert record.changes == ()
    # ScaleN(1) is a no-op: accepted, nothing realized, no scaling note.
    out1, rec1 = augment(base, AugmentationSpec(transforms=[ScaleN(factor=1)]))
    assert out1.to_dict() == base.to_dict()
    assert rec1.realized_order == ()
    assert rec1.notes == ()
    # x10 keeps every declared type realized: accepted.
    out10, rec10 = augment(base, AugmentationSpec(transforms=[ScaleN(factor=10)]))
    assert out10.areas[0].layers[0].n_neurons == 100
    assert rec10.realized_order == ("N",)
    # Down-scale 100 -> 10 zeroes VIP: refused, naming area, layer and type.
    big = _fraction_tensor(n=100)
    with pytest.raises(ValueError, match="cell type 'VIP'") as exc:
        augment(big, AugmentationSpec(transforms=[ScaleN(factor=0.1)]))
    assert "'L4'" in str(exc.value) and "'V1'" in str(exc.value)


def test_uncomputable_base_digest_refused(monkeypatch):
    # Root `jtfne.augment` is the exported function (AUG-3), which shadows
    # the submodule attribute, so reach the module via sys.modules.
    import sys

    aug = sys.modules["jaxfne.augment"]

    monkeypatch.setattr(aug, "_tensor_identity_digest", lambda tensor: "")
    with pytest.raises(ValueError, match="digest"):
        augment(_tiny_tensor(), AugmentationSpec(transforms=[ScaleN(factor=10)]))
    with pytest.raises(ValueError, match="digest"):
        clone_tensor(_tiny_tensor())


def test_bare_subrecord_refused_naming_expected_type():
    with pytest.raises(ValueError, match="GeometryTransform"):
        AugmentationSpec(
            transforms=[PoseEdit(area="V1", translation=(1.0, 0.0, 0.0))]
        )
    with pytest.raises(ValueError, match="GeometryTransform"):
        AugmentationSpec(
            transforms=[RangeEdit(area="V1", layer="L4", x_range=(0.0, 0.5))]
        )


def test_augment_rejects_wrong_record_type_naming_expected_type():
    import types

    tensor = _tiny_tensor()
    spec_n = AugmentationSpec(transforms=[ScaleN(factor=10)])
    object.__setattr__(spec_n, "transforms", (types.SimpleNamespace(axis="N"),))
    with pytest.raises(ValueError, match="expects a ScaleN record"):
        augment(tensor, spec_n)
    spec_g = AugmentationSpec(
        transforms=[GeometryTransform(pose_edits=[PoseEdit(area="V1")])]
    )
    object.__setattr__(spec_g, "transforms", (PoseEdit(area="V1"),))
    with pytest.raises(ValueError, match="expects a GeometryTransform record"):
        augment(tensor, spec_g)


def test_noop_transforms_realize_nothing():
    tensor = _tiny_tensor()
    out, record = augment(tensor, AugmentationSpec(transforms=[ScaleN(factor=1)]))
    assert out.to_dict() == tensor.to_dict()
    assert record.realized_order == ()
    assert record.changes == ()
    assert record.notes == ()
    out, record = augment(tensor, AugmentationSpec(transforms=[GeometryTransform()]))
    assert out.to_dict() == tensor.to_dict()
    assert record.realized_order == ()
    assert record.changes == ()
    # Edits equal to current values are no-ops too.
    out, record = augment(
        tensor,
        AugmentationSpec(
            transforms=[
                GeometryTransform(
                    pose_edits=[PoseEdit(area="V1", translation=(0.0, 0.0, 0.0))],
                    range_edits=[RangeEdit(area="V1", layer="L4", x_range=(0.0, 1.0))],
                )
            ]
        ),
    )
    assert out.to_dict() == tensor.to_dict()
    assert record.realized_order == ()
    assert record.changes == ()
    # The w/sqrt(N) note is written only when N actually changed.
    _, rec10 = augment(tensor, AugmentationSpec(transforms=[ScaleN(factor=10)]))
    assert len(rec10.notes) == 1 and "sqrt(N)" in rec10.notes[0]


def test_scale_n_bad_factors_refused_at_construction():
    for bad in (0, -2, 0.0, float("nan"), float("inf"), "10", True, None):
        with pytest.raises(ValueError):
            ScaleN(factor=bad)
    assert ScaleN(factor=10).factor == 10.0
    assert isinstance(ScaleN(factor=10).factor, float)
    assert AugmentationSpec(transforms=[ScaleN(factor=10)]).digest() == (
        AugmentationSpec(transforms=[ScaleN(factor=10.0)]).digest()
    )


def test_duplicate_pose_edit_refused():
    with pytest.raises(ValueError, match="two PoseEdits"):
        GeometryTransform(
            pose_edits=[
                PoseEdit(area="V1", translation=(1.0, 0.0, 0.0)),
                PoseEdit(area="V1", rotation_deg=90.0),
            ]
        )


def test_duplicate_range_edit_refused():
    with pytest.raises(ValueError, match="two RangeEdits"):
        GeometryTransform(
            range_edits=[
                RangeEdit(area="V1", layer="L4", x_range=(0.0, 0.5)),
                RangeEdit(area="V1", layer="L4", y_range=(0.0, 0.5)),
            ]
        )
    # The same layer name in different areas is not ambiguous.
    merged = GeometryTransform(
        range_edits=[
            RangeEdit(area="A0", layer="L4", x_range=(0.0, 0.5)),
            RangeEdit(area="A1", layer="L4", x_range=(0.0, 0.5)),
        ]
    )
    assert len(merged.range_edits) == 2


def test_pose_edit_applies_scale_then_rotation_then_translation():
    tensor = _tiny_tensor()
    tensor.areas[0].pose.translation = (4.0, 0.0, 0.0)
    spec = AugmentationSpec(
        transforms=[
            GeometryTransform(
                pose_edits=[
                    PoseEdit(
                        area="V1",
                        translation_scale=0.5,
                        rotation_deg=90.0,
                        translation=(1.0, 2.0, 3.0),
                    )
                ]
            )
        ]
    )
    out, record = augment(tensor, spec)
    # Scale first ((4,0,0) * 0.5 = (2,0,0)), then translation adds: (3,2,3).
    # Scale-last would give ((4,0,0) + (1,2,3)) * 0.5 = (2.5,1.0,1.5).
    assert out.areas[0].pose.translation == (3.0, 2.0, 3.0)
    assert out.areas[0].pose.rotation_deg == pytest.approx(90.0)
    assert [entry.address for entry in record.changes] == [
        "areas.V1.pose.translation",
        "areas.V1.pose.rotation_deg",
        "areas.V1.pose.translation",
    ]


def test_rotation_about_area_column_frame_and_absolute_translation():
    from jaxfne._construct_population import AREA_X_SPACING_MM

    tensor = _two_area_tensor(n=10)
    spec = AugmentationSpec(
        transforms=[
            GeometryTransform(
                pose_edits=[
                    PoseEdit(area="A1", rotation_deg=90.0, translation=(5.0, 6.0, 7.0))
                ]
            )
        ]
    )
    out, record = augment(tensor, spec)
    assert record.realized_order == ("G",)
    base = _build(tensor, seed=0)
    moved = _build(out, seed=0)
    rows = moved.neuron_table()
    base_pos = base.params["positions"]
    moved_pos = moved.params["positions"]
    origin = AREA_X_SPACING_MM  # edited area A1 sits at column index 1
    theta = math.radians(90.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a1 = [i for i, row in enumerate(rows) if row["area"] == "A1"]
    a0 = [i for i, row in enumerate(rows) if row["area"] == "A0"]
    assert len(a1) == 10 and len(a0) == 10
    for i in a1:
        local = (
            float(base_pos[i, 0]) - origin,
            float(base_pos[i, 1]),
            float(base_pos[i, 2]),
        )
        expected = (
            local[0] * cos_t - local[1] * sin_t + 5.0,
            local[0] * sin_t + local[1] * cos_t + 6.0,
            local[2] + 7.0,
        )
        got = (float(moved_pos[i, 0]), float(moved_pos[i, 1]), float(moved_pos[i, 2]))
        assert got == pytest.approx(expected, abs=1e-5)
    # The unedited area is untouched.
    for i in a0:
        assert tuple(float(v) for v in moved_pos[i]) == pytest.approx(
            tuple(float(v) for v in base_pos[i]), abs=1e-9
        )


def test_translation_scale_multiplies_stored_translation():
    tensor = _two_area_tensor(n=10)
    tensor.areas[1].pose.translation = (4.0, -2.0, 6.0)
    spec = AugmentationSpec(
        transforms=[
            GeometryTransform(
                pose_edits=[PoseEdit(area="A1", translation_scale=0.5)]
            )
        ]
    )
    out, record = augment(tensor, spec)
    assert out.areas[1].pose.translation == (2.0, -1.0, 3.0)
    assert record.realized_order == ("G",)
    base = _build(tensor, seed=0)
    moved = _build(out, seed=0)
    rows = moved.neuron_table()
    a1 = [i for i, row in enumerate(rows) if row["area"] == "A1"]
    assert len(a1) == 10
    shift = moved.params["positions"][jnp.array(a1)] - base.params["positions"][jnp.array(a1)]
    assert bool(jnp.all(jnp.isfinite(shift)))
    assert jnp.allclose(
        shift, jnp.asarray((-2.0, 1.0, -3.0), dtype=shift.dtype), atol=1e-6
    ).item()


def _leaf_paths(payload, prefix=()):
    """Flatten nested dict/list/tuple payload to {path_tuple: leaf_value}."""
    if isinstance(payload, dict):
        out = {}
        for key, value in payload.items():
            out.update(_leaf_paths(value, prefix + (key,)))
        return out
    if isinstance(payload, (list, tuple)):
        out = {}
        for idx, value in enumerate(payload):
            out.update(_leaf_paths(value, prefix + (idx,)))
        return out
    return {prefix: payload}


def _record_prefix(entry_address, before):
    """Resolve a ProvenanceEntry address to its to_dict() leaf-path prefix."""
    parts = entry_address.split(".")
    assert parts[0] == "areas"
    areas = before["areas"]
    ai = next(i for i, a in enumerate(areas) if a["name"] == parts[1])
    if parts[2] == "pose":
        return ("areas", ai, "pose", parts[3])
    assert parts[2] == "layers"
    layers = areas[ai]["layers"]
    li = next(i for i, layer in enumerate(layers) if layer["name"] == parts[3])
    if parts[4] == "n_neurons":
        return ("areas", ai, "layers", li, "n_neurons")
    assert parts[4] == "geometry"
    return ("areas", ai, "layers", li, "geometry", parts[5])


def test_completeness_changed_leaves_equal_record_addresses():
    tensor = _two_area_two_layer_tensor(n=10)
    n_spec = AugmentationSpec(transforms=[ScaleN(factor=10)])
    g_spec = AugmentationSpec(
        transforms=[
            GeometryTransform(
                pose_edits=[PoseEdit(area="A1", translation=(1.0, 2.0, 3.0))],
                range_edits=[RangeEdit(area="A0", layer="L5", z_range=(0.2, 0.8))],
            )
        ]
    )
    for spec, axis in ((n_spec, "N"), (g_spec, "G")):
        before = tensor.to_dict()
        out, record = augment(tensor, spec)
        after = out.to_dict()
        flat_before = _leaf_paths(before)
        flat_after = _leaf_paths(after)
        assert set(flat_before) == set(flat_after)
        changed = {p for p in flat_before if flat_before[p] != flat_after[p]}
        assert changed, "spec must change something"
        for entry in record.changes:
            assert entry.axis == axis
            prefix = _record_prefix(entry.address, before)
            assert any(p[: len(prefix)] == prefix for p in changed), (
                f"record address {entry.address} covers no changed leaf"
            )
        covered = {
            p
            for p in changed
            for entry in record.changes
            if p[: len(_record_prefix(entry.address, before))]
            == _record_prefix(entry.address, before)
        }
        assert changed == covered, (
            f"changed leaves outside the record: {sorted(set(changed) - set(covered))}"
        )
        if axis == "N":
            stray = [
                p
                for p in changed
                if any(k in p for k in ("pose", "geometry", "inter_connections", "area_connections"))
            ]
            assert stray == [], f"N-only spec touched non-N leaves: {stray}"


def test_identity_digest_changes_with_augmentation_and_stable_for_clone():
    from jaxfne.neuronal_tensor import _tensor_identity_digest

    tensor = _tiny_tensor()
    clone, _ = clone_tensor(tensor)
    assert _tensor_identity_digest(clone) == _tensor_identity_digest(tensor)
    scaled, _ = augment(tensor, AugmentationSpec(transforms=[ScaleN(factor=10)]))
    assert _tensor_identity_digest(scaled) != _tensor_identity_digest(tensor)
    moved, _ = augment(
        tensor,
        AugmentationSpec(
            transforms=[
                GeometryTransform(
                    pose_edits=[PoseEdit(area="V1", translation=(1.0, 0.0, 0.0))]
                )
            ]
        ),
    )
    assert _tensor_identity_digest(moved) != _tensor_identity_digest(tensor)


def test_spec_digest_stable_across_processes(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    import jaxfne

    root = Path(jaxfne.__file__).resolve().parent.parent
    expected = AugmentationSpec(
        transforms=[
            ScaleN(factor=10),
            GeometryTransform(
                pose_edits=[PoseEdit(area="V1", translation=(1.0, -2.0, 0.5))]
            ),
        ]
    ).digest()
    script = tmp_path / "_temp_aug_digest.py"
    script.write_text(
        "from jaxfne.augment import AugmentationSpec, GeometryTransform, PoseEdit, ScaleN\n"
        "spec = AugmentationSpec(transforms=[\n"
        "    ScaleN(factor=10),\n"
        "    GeometryTransform(pose_edits=[PoseEdit(area='V1', translation=(1.0, -2.0, 0.5))]),\n"
        "])\n"
        "print(spec.digest())\n",
        encoding="utf-8",
    )
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=300,
        )
    finally:
        script.unlink(missing_ok=True)
    assert result.returncode == 0, (
        f"digest script failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == expected
    assert not script.exists()


def _w_mech_of(tensor):
    """Every PlasticParams.w_mech in address order (inter by area, then cross)."""
    values = []
    for area in tensor.areas:
        for conn in area.inter_connections:
            values.append(float(conn.plastic.w_mech))
    for conn in tensor.area_connections:
        values.append(float(conn.plastic.w_mech))
    return values


def test_w0_deterministic_factor_doubles_only_w_mech():
    tensor = _two_area_two_layer_tensor(n=10)
    assert _w_mech_of(tensor) == [2.0, 2.0, 2.0, 1.5]
    before = tensor.to_dict()
    out, record = augment(tensor, AugmentationSpec(transforms=[W0(factor=2.0)]))
    assert record.realized_order == ("W_0",)
    assert len(record.changes) == 4
    assert all(entry.origin == "augmented" for entry in record.changes)
    assert [entry.address for entry in record.changes] == [
        "A0/inter/0.plastic.w_mech",
        "A0/inter/1.plastic.w_mech",
        "A1/inter/0.plastic.w_mech",
        "area_connection/0.plastic.w_mech",
    ]
    assert _w_mech_of(out) == pytest.approx([4.0, 4.0, 4.0, 3.0])
    # to_dict diff touches only w_mech leaves; nothing else moves.
    flat_before = _leaf_paths(before)
    flat_after = _leaf_paths(out.to_dict())
    assert set(flat_before) == set(flat_after)
    changed = {p for p in flat_before if flat_before[p] != flat_after[p]}
    assert changed
    assert all(p[-1] == "w_mech" for p in changed)
    assert tensor.to_dict() == before  # input tensor never mutated


def test_w0_stochastic_seeded_reproducible_and_bounded():
    tensor = _two_area_two_layer_tensor(n=10)
    olds = _w_mech_of(tensor)
    spec_a = AugmentationSpec(transforms=[W0(factor=1.0, jitter=0.2)], k_v=11)
    spec_b = AugmentationSpec(transforms=[W0(factor=1.0, jitter=0.2)], k_v=11)
    out_a, rec_a = augment(tensor, spec_a)
    out_b, _ = augment(tensor, spec_b)
    assert out_a.to_dict() == out_b.to_dict()  # same k_v: bit-identical
    out_c, _ = augment(
        tensor, AugmentationSpec(transforms=[W0(factor=1.0, jitter=0.2)], k_v=12)
    )
    assert out_c.to_dict() != out_a.to_dict()  # different k_v: different output
    assert all(entry.origin == "augment-sampled" for entry in rec_a.changes)
    assert len(rec_a.changes) == 4
    for old, new in zip(olds, _w_mech_of(out_a)):
        assert old * 0.8 <= new <= old * 1.2
    with pytest.raises(ValueError, match="K_V"):
        AugmentationSpec(transforms=[W0(factor=1.0, jitter=0.2)])


def test_w0_targets_select_and_refusals():
    tensor = _two_area_two_layer_tensor(n=10)
    before = tensor.to_dict()
    out, record = augment(
        tensor, AugmentationSpec(transforms=[W0(factor=2.0, targets=("A1/inter/0",))])
    )
    assert [entry.address for entry in record.changes] == ["A1/inter/0.plastic.w_mech"]
    assert record.realized_order == ("W_0",)
    flat_before = _leaf_paths(before)
    flat_after = _leaf_paths(out.to_dict())
    changed = {p for p in flat_before if flat_before[p] != flat_after[p]}
    assert len(changed) == 1
    with pytest.raises(ValueError, match="NX/inter/0"):
        augment(
            tensor, AugmentationSpec(transforms=[W0(factor=2.0, targets=("NX/inter/0",))])
        )
    with pytest.raises(ValueError, match="area_connection/7"):
        augment(
            tensor,
            AugmentationSpec(transforms=[W0(factor=2.0, targets=("area_connection/7",))]),
        )
    with pytest.raises(ValueError, match="duplicate"):
        W0(factor=2.0, targets=("A0/inter/0", "A0/inter/0"))


def test_w0_bad_records_refused():
    for bad in (0, -1, 0.0, float("nan"), float("inf"), True, "2", None):
        with pytest.raises(ValueError):
            W0(factor=bad)
    for bad in (-0.1, -1.0, 1.0, 2.0, float("nan"), float("inf"), True, "0.1", None):
        with pytest.raises(ValueError):
            W0(jitter=bad)
    assert W0().factor == 1.0
    assert W0().jitter == 0.0
    assert W0().targets == ()
    assert W0().stochastic is False
    assert W0(factor=2.0, jitter=0.5, targets=("A0/inter/0",)).stochastic is True


def test_w0_composes_with_n_in_canonical_order():
    tensor = _tiny_tensor(n=10)
    spec = AugmentationSpec(transforms=[W0(factor=2.0), ScaleN(factor=10)])
    out, record = augment(tensor, spec)
    assert record.realized_order == ("N", "W_0")
    assert out.areas[0].layers[0].n_neurons == 100
    assert out.areas[0].inter_connections[0].plastic.w_mech == pytest.approx(4.0)
    assert [entry.axis for entry in record.changes] == ["N", "W_0"]
    assert record.spec_digest == (
        AugmentationSpec(transforms=[ScaleN(factor=10), W0(factor=2.0)]).digest()
    )


def test_w0_noop_realizes_nothing():
    tensor = _tiny_tensor()
    out, record = augment(tensor, AugmentationSpec(transforms=[W0()]))
    assert out.to_dict() == tensor.to_dict()
    assert record.realized_order == ()
    assert record.changes == ()
    # No connections at all: even factor != 1 realizes nothing.
    bare = _two_area_tensor(n=10)
    out, record = augment(bare, AugmentationSpec(transforms=[W0(factor=2.0)]))
    assert out.to_dict() == bare.to_dict()
    assert record.realized_order == ()
    assert record.changes == ()


def test_w0_digest_covers_factor_jitter_targets():
    assert W0(factor=2.0).factor == 2.0
    base = AugmentationSpec(transforms=[W0(factor=2.0)]).digest()
    assert AugmentationSpec(transforms=[W0(factor=3.0)]).digest() != base
    assert AugmentationSpec(transforms=[W0(factor=2.0, jitter=0.1)], k_v=5).digest() != base
    assert (
        AugmentationSpec(transforms=[W0(factor=2.0, targets=("V1/inter/0",))]).digest()
        != base
    )


def test_w0_sampled_origin_does_not_rebind_jdna_vocabulary():
    from jaxfne.jdna import completion

    assert completion.ORIGIN_SAMPLED == "JDNA-sampled"
    assert completion.ORIGIN_SAMPLED in completion.ORIGINS
    assert completion.ORIGIN_AUGMENT_SAMPLED not in completion.ORIGINS


def test_w0_zero_gain_stays_zero_and_negative_refused():
    tensor = _two_area_two_layer_tensor(n=10)
    tensor.area_connections[0].plastic.w_mech = 0.0
    out, record = augment(tensor, AugmentationSpec(transforms=[W0(factor=2.0, jitter=0.1)], k_v=3))
    assert _w_mech_of(out)[-1] == 0.0
    assert len(record.changes) == 3
    tensor.area_connections[0].plastic.w_mech = -1.0
    with pytest.raises(ValueError, match="negative or not finite"):
        augment(tensor, AugmentationSpec(transforms=[W0(factor=2.0)]))


def _delayed_tensor(n=10):
    """Two-area fixture with declared delays and cross probability (Theta_C needs them)."""
    tensor = _two_area_two_layer_tensor(n=n)
    for area in tensor.areas:
        for conn in area.inter_connections:
            conn.delay_ms = 1.0
    cross = tensor.area_connections[0]
    cross.delay_ms = 2.0
    cross.probability = 0.5
    return tensor


def _static_of(tensor):
    """Every (g_mech dict, dT_ms) in address order (inter by area, then cross)."""
    values = []
    for area in tensor.areas:
        for conn in area.inter_connections:
            values.append((dict(conn.static.g_mech), conn.static.dT_ms))
    for conn in tensor.area_connections:
        values.append((dict(conn.static.g_mech), conn.static.dT_ms))
    return values


def _h_of(tensor):
    """Every PlasticParams.H in address order (inter by area, then cross)."""
    values = []
    for area in tensor.areas:
        for conn in area.inter_connections:
            values.append(conn.plastic.H)
    for conn in tensor.area_connections:
        values.append(conn.plastic.H)
    return values


def test_thetax_deterministic_factor_scales_only_static():
    tensor = _two_area_two_layer_tensor(n=10)
    before = tensor.to_dict()
    out, record = augment(
        tensor, AugmentationSpec(transforms=[ThetaX(g_factor=2.0, tau_factor=3.0)])
    )
    assert record.realized_order == ("Theta_X",)
    assert len(record.changes) == 8
    assert all(entry.origin == "augmented" for entry in record.changes)
    assert [entry.address for entry in record.changes] == [
        "A0/inter/0.static.g_mech.AMPA",
        "A0/inter/0.static.dT_ms",
        "A0/inter/1.static.g_mech.AMPA",
        "A0/inter/1.static.dT_ms",
        "A1/inter/0.static.g_mech.AMPA",
        "A1/inter/0.static.dT_ms",
        "area_connection/0.static.g_mech.AMPA",
        "area_connection/0.static.dT_ms",
    ]
    assert _static_of(out) == [({"AMPA": 2.0}, 6.0)] * 4
    # to_dict diff touches only static leaves; nothing else moves.
    flat_before = _leaf_paths(before)
    flat_after = _leaf_paths(out.to_dict())
    assert set(flat_before) == set(flat_after)
    changed = {p for p in flat_before if flat_before[p] != flat_after[p]}
    assert changed
    assert all(p[-1] in ("AMPA", "dT_ms") for p in changed)
    assert tensor.to_dict() == before  # input tensor never mutated


def test_thetac_deterministic_factor_scales_only_delay_and_probability():
    tensor = _delayed_tensor()
    before = tensor.to_dict()
    out, record = augment(
        tensor,
        AugmentationSpec(transforms=[ThetaC(delay_factor=2.0, probability_factor=0.5)]),
    )
    assert record.realized_order == ("Theta_C",)
    assert all(entry.origin == "augmented" for entry in record.changes)
    assert [entry.address for entry in record.changes] == [
        "A0/inter/0.delay_ms",
        "A0/inter/1.delay_ms",
        "A1/inter/0.delay_ms",
        "area_connection/0.delay_ms",
        "area_connection/0.probability",
    ]
    delays = []
    for area in out.areas:
        for conn in area.inter_connections:
            delays.append(conn.delay_ms)
    delays.append(out.area_connections[0].delay_ms)
    assert delays == [2.0, 2.0, 2.0, 4.0]
    assert out.area_connections[0].probability == pytest.approx(0.25)
    flat_before = _leaf_paths(before)
    flat_after = _leaf_paths(out.to_dict())
    assert set(flat_before) == set(flat_after)
    changed = {p for p in flat_before if flat_before[p] != flat_after[p]}
    assert changed
    assert all(p[-1] in ("delay_ms", "probability") for p in changed)
    assert tensor.to_dict() == before  # input tensor never mutated


def test_h0_offset_shifts_even_default_zero_H():
    tensor = _two_area_two_layer_tensor(n=10)
    for area in tensor.areas:
        for conn in area.inter_connections:
            conn.plastic.H = 0.0
    tensor.area_connections[0].plastic.H = 0.0
    assert _h_of(tensor) == [0.0, 0.0, 0.0, 0.0]
    before = tensor.to_dict()
    out, record = augment(tensor, AugmentationSpec(transforms=[H0(offset=0.5)]))
    assert record.realized_order == ("H_0",)
    assert len(record.changes) == 4
    assert all(entry.origin == "augmented" for entry in record.changes)
    assert [entry.address for entry in record.changes] == [
        "A0/inter/0.plastic.H",
        "A0/inter/1.plastic.H",
        "A1/inter/0.plastic.H",
        "area_connection/0.plastic.H",
    ]
    assert _h_of(out) == [0.5, 0.5, 0.5, 0.5]
    flat_before = _leaf_paths(before)
    flat_after = _leaf_paths(out.to_dict())
    assert set(flat_before) == set(flat_after)
    changed = {p for p in flat_before if flat_before[p] != flat_after[p]}
    assert changed
    assert all(p[-1] == "H" for p in changed)
    assert tensor.to_dict() == before  # input tensor never mutated


def test_thetax_stochastic_seeded_reproducible_and_bounded():
    tensor = _two_area_two_layer_tensor(n=10)
    make = lambda k: AugmentationSpec(
        transforms=[ThetaX(g_factor=1.0, g_jitter=0.2, tau_factor=1.0, tau_jitter=0.1)],
        k_v=k,
    )
    out_a, rec_a = augment(tensor, make(11))
    out_b, _ = augment(tensor, make(11))
    assert out_a.to_dict() == out_b.to_dict()  # same k_v: bit-identical
    out_c, _ = augment(tensor, make(12))
    assert out_c.to_dict() != out_a.to_dict()  # different k_v: different output
    assert all(entry.origin == "augment-sampled" for entry in rec_a.changes)
    assert len(rec_a.changes) == 8
    for (old_g, old_tau), (new_g, new_tau) in zip(_static_of(tensor), _static_of(out_a)):
        assert old_g["AMPA"] * 0.8 <= new_g["AMPA"] <= old_g["AMPA"] * 1.2
        assert old_tau * 0.9 <= new_tau <= old_tau * 1.1
    with pytest.raises(ValueError, match="K_V"):
        AugmentationSpec(transforms=[ThetaX(g_jitter=0.2)])
    # Mixed jitters: only the jittered field draws, origins stay per-field.
    mixed, rec_mixed = augment(
        tensor,
        AugmentationSpec(transforms=[ThetaX(g_factor=2.0, g_jitter=0.2)], k_v=5),
    )
    by_address = {}
    for entry in rec_mixed.changes:
        key = entry.address.rsplit(".static.", 1)[0]
        by_address.setdefault(key, []).append(entry.origin)
    assert set(by_address) == {
        "A0/inter/0", "A0/inter/1", "A1/inter/0", "area_connection/0",
    }
    for origins in by_address.values():
        assert origins == ["augment-sampled"]  # g drew; dT factor is 1.0: no entry


def test_thetac_stochastic_seeded_reproducible_and_bounded():
    tensor = _delayed_tensor()
    make = lambda k: AugmentationSpec(
        transforms=[
            ThetaC(
                delay_factor=1.0, delay_jitter=0.2,
                probability_factor=1.0, probability_jitter=0.1,
            )
        ],
        k_v=k,
    )
    out_a, rec_a = augment(tensor, make(11))
    out_b, _ = augment(tensor, make(11))
    assert out_a.to_dict() == out_b.to_dict()  # same k_v: bit-identical
    out_c, _ = augment(tensor, make(12))
    assert out_c.to_dict() != out_a.to_dict()  # different k_v: different output
    assert all(entry.origin == "augment-sampled" for entry in rec_a.changes)
    assert [entry.address for entry in rec_a.changes] == [
        "A0/inter/0.delay_ms",
        "A0/inter/1.delay_ms",
        "A1/inter/0.delay_ms",
        "area_connection/0.delay_ms",
        "area_connection/0.probability",
    ]
    delays_a = []
    for area in out_a.areas:
        for conn in area.inter_connections:
            delays_a.append(conn.delay_ms)
    delays_a.append(out_a.area_connections[0].delay_ms)
    for old, new in zip((1.0, 1.0, 1.0, 2.0), delays_a):
        assert old * 0.8 <= new <= old * 1.2
    prob = out_a.area_connections[0].probability
    assert 0.5 * 0.9 <= prob <= 0.5 * 1.1
    with pytest.raises(ValueError, match="K_V"):
        AugmentationSpec(transforms=[ThetaC(delay_jitter=0.2)])


def test_h0_stochastic_seeded_reproducible_and_bounded():
    tensor = _two_area_two_layer_tensor(n=10)
    tensor.areas[0].inter_connections[0].plastic.H = 0.0
    tensor.areas[0].inter_connections[1].plastic.H = -0.2
    olds = _h_of(tensor)
    make = lambda k: AugmentationSpec(
        transforms=[H0(offset=0.5, jitter=0.1)], k_v=k
    )
    out_a, rec_a = augment(tensor, make(11))
    out_b, _ = augment(tensor, make(11))
    assert out_a.to_dict() == out_b.to_dict()  # same k_v: bit-identical
    out_c, _ = augment(tensor, make(12))
    assert out_c.to_dict() != out_a.to_dict()  # different k_v: different output
    assert all(entry.origin == "augment-sampled" for entry in rec_a.changes)
    assert len(rec_a.changes) == 4
    for old, new in zip(olds, _h_of(out_a)):
        assert abs(new - old - 0.5) <= 0.1
    with pytest.raises(ValueError, match="K_V"):
        AugmentationSpec(transforms=[H0(offset=0.5, jitter=0.1)])


def test_none_fields_untouched_with_no_entry():
    tensor = _two_area_two_layer_tensor(n=10)
    tensor.areas[0].inter_connections[0].delay_ms = 1.0  # only declared delay
    tensor.areas[0].inter_connections[1].static.dT_ms = None  # undeclared tau
    tensor.areas[1].inter_connections[0].plastic.H = None  # undeclared H
    before = tensor.to_dict()
    out, record = augment(
        tensor,
        AugmentationSpec(
            transforms=[
                ThetaC(delay_factor=2.0, probability_factor=0.5),
                ThetaX(g_factor=2.0, tau_factor=2.0),
                H0(offset=1.0),
            ]
        ),
    )
    assert record.realized_order == ("Theta_C", "Theta_X", "H_0")
    thetac = [e.address for e in record.changes if e.axis == "Theta_C"]
    assert thetac == ["A0/inter/0.delay_ms"]  # probability is None everywhere
    thetax = [e.address for e in record.changes if e.axis == "Theta_X"]
    assert "A0/inter/1.static.dT_ms" not in thetax  # dT None: no entry
    assert "A0/inter/1.static.g_mech.AMPA" in thetax  # g still scales there
    h0 = [e.address for e in record.changes if e.axis == "H_0"]
    assert "A1/inter/0.plastic.H" not in h0  # H None: no entry
    assert len(h0) == 3
    assert out.areas[0].inter_connections[1].static.dT_ms is None
    assert out.areas[1].inter_connections[0].plastic.H is None
    assert out.area_connections[0].probability is None
    assert out.area_connections[0].delay_ms is None
    assert tensor.to_dict() == before  # input tensor never mutated


def test_thetac_probability_above_1_refused():
    tensor = _delayed_tensor()  # cross probability is 0.5
    with pytest.raises(ValueError) as exc:
        augment(tensor, AugmentationSpec(transforms=[ThetaC(probability_factor=3.0)]))
    assert "above 1" in str(exc.value)
    assert "area_connection/0.probability" in str(exc.value)
    assert "1.5" in str(exc.value)


def test_thetax_dt_positive_and_negative_stored_refused():
    tensor = _two_area_two_layer_tensor(n=10)
    tensor.areas[0].inter_connections[0].static.dT_ms = -1.0
    with pytest.raises(ValueError, match="negative or not finite") as exc:
        augment(tensor, AugmentationSpec(transforms=[ThetaX(tau_factor=2.0)]))
    assert "A0/inter/0.static.dT_ms" in str(exc.value)
    tensor.areas[0].inter_connections[0].static.dT_ms = 0.0
    with pytest.raises(ValueError, match="> 0") as exc:
        augment(tensor, AugmentationSpec(transforms=[ThetaX(tau_factor=2.0)]))
    assert "A0/inter/0.static.dT_ms" in str(exc.value)
    tensor = _two_area_two_layer_tensor(n=10)
    tensor.areas[0].inter_connections[0].static.g_mech["AMPA"] = -1.0
    with pytest.raises(ValueError, match="negative or not finite") as exc:
        augment(tensor, AugmentationSpec(transforms=[ThetaX(g_factor=2.0)]))
    assert "A0/inter/0.static.g_mech.AMPA" in str(exc.value)


def test_thetac_negative_stored_delay_refused():
    tensor = _delayed_tensor()
    tensor.areas[0].inter_connections[0].delay_ms = -1.0
    with pytest.raises(ValueError, match="negative or not finite") as exc:
        augment(tensor, AugmentationSpec(transforms=[ThetaC(delay_factor=2.0)]))
    assert "A0/inter/0.delay_ms" in str(exc.value)


def test_h0_negative_stored_accepted_and_shifted_nonfinite_refused():
    tensor = _two_area_two_layer_tensor(n=10)
    tensor.areas[0].inter_connections[0].plastic.H = -0.2
    out, record = augment(tensor, AugmentationSpec(transforms=[H0(offset=0.5)]))
    assert out.areas[0].inter_connections[0].plastic.H == pytest.approx(0.3)
    assert record.realized_order == ("H_0",)
    tensor.areas[0].inter_connections[0].plastic.H = float("nan")
    with pytest.raises(ValueError, match="not finite") as exc:
        augment(tensor, AugmentationSpec(transforms=[H0(offset=0.5)]))
    assert "'A0/inter/0'" in str(exc.value) and "stored H" in str(exc.value)
    tensor.areas[0].inter_connections[0].plastic.H = float("inf")
    with pytest.raises(ValueError, match="not finite"):
        augment(tensor, AugmentationSpec(transforms=[H0(offset=0.5)]))


def test_full_composition_canonical_order_and_pairwise_equivalence():
    tensor = _delayed_tensor()
    scale = ScaleN(factor=10)
    geom = GeometryTransform(
        pose_edits=[PoseEdit(area="A1", translation=(1.0, 2.0, 3.0))],
        range_edits=[RangeEdit(area="A0", layer="L5", z_range=(0.2, 0.8))],
    )
    thetac = ThetaC(delay_factor=2.0, probability_factor=0.5)
    thetax = ThetaX(g_factor=2.0, tau_factor=2.0)
    w0 = W0(factor=2.0)
    h0 = H0(offset=0.5)
    spec = AugmentationSpec(transforms=[h0, w0, thetax, thetac, geom, scale])
    out, record = augment(tensor, spec)
    assert record.realized_order == ("N", "G", "Theta_C", "Theta_X", "W_0", "H_0")
    assert record.spec_digest == (
        AugmentationSpec(transforms=[scale, geom, thetac, thetax, w0, h0]).digest()
    )
    # Applying the same records one by one in canonical order gives the same tensor.
    step = tensor
    for rec in (scale, geom, thetac, thetax, w0, h0):
        step, _ = augment(step, AugmentationSpec(transforms=[rec]))
    assert out.to_dict() == step.to_dict()
    # N does not touch w: W_0-then-N sequential w values equal the composition's.
    w_first, _ = augment(tensor, AugmentationSpec(transforms=[w0]))
    w_then_n, _ = augment(w_first, AugmentationSpec(transforms=[scale]))
    assert _w_mech_of(w_then_n) == pytest.approx(_w_mech_of(out))


def test_thetax_thetac_h0_bad_records_refused():
    bad_factors = (0, -1, 0.0, float("nan"), float("inf"), True, "2", None)
    for bad in bad_factors:
        with pytest.raises(ValueError):
            ThetaX(g_factor=bad)
        with pytest.raises(ValueError):
            ThetaX(tau_factor=bad)
        with pytest.raises(ValueError):
            ThetaC(delay_factor=bad)
        with pytest.raises(ValueError):
            ThetaC(probability_factor=bad)
    for bad in (float("nan"), float("inf"), True, "0.5", None):
        with pytest.raises(ValueError):
            H0(offset=bad)
    assert H0(offset=-0.5).offset == -0.5  # signed shift is valid
    assert H0().offset == 0.0
    bad_jitters = (-0.1, -1.0, 1.0, 2.0, float("nan"), float("inf"), True, "0.1", None)
    for bad in bad_jitters:
        with pytest.raises(ValueError):
            ThetaX(g_jitter=bad)
        with pytest.raises(ValueError):
            ThetaX(tau_jitter=bad)
        with pytest.raises(ValueError):
            ThetaC(delay_jitter=bad)
        with pytest.raises(ValueError):
            ThetaC(probability_jitter=bad)
    for bad in (-0.1, -1.0, float("nan"), float("inf"), True, "0.1", None):
        with pytest.raises(ValueError):
            H0(jitter=bad)
    assert H0(jitter=2.0).jitter == 2.0  # absolute half-width: no upper bound
    assert ThetaX().stochastic is False
    assert ThetaC().stochastic is False
    assert H0().stochastic is False
    assert H0(offset=0.5).stochastic is False  # deterministic shift draws nothing
    assert ThetaX(g_jitter=0.5).stochastic is True
    assert ThetaX(tau_jitter=0.5).stochastic is True
    assert ThetaC(delay_jitter=0.5).stochastic is True
    assert ThetaC(probability_jitter=0.5).stochastic is True
    assert H0(jitter=0.5).stochastic is True
    with pytest.raises(ValueError, match="duplicate"):
        ThetaX(targets=("A0/inter/0", "A0/inter/0"))
    with pytest.raises(ValueError, match="duplicate"):
        ThetaC(targets=("A0/inter/0", "A0/inter/0"))
    with pytest.raises(ValueError, match="duplicate"):
        H0(targets=("A0/inter/0", "A0/inter/0"))
    tensor = _two_area_two_layer_tensor(n=10)
    with pytest.raises(ValueError, match="NX/inter/0"):
        augment(tensor, AugmentationSpec(transforms=[ThetaX(g_factor=2.0, targets=("NX/inter/0",))]))
    with pytest.raises(ValueError, match="area_connection/7"):
        augment(
            tensor,
            AugmentationSpec(transforms=[ThetaC(delay_factor=2.0, targets=("area_connection/7",))]),
        )
    with pytest.raises(ValueError, match="NX/inter/0"):
        augment(tensor, AugmentationSpec(transforms=[H0(offset=1.0, targets=("NX/inter/0",))]))
    # Digests cover every new field.
    assert AugmentationSpec(transforms=[ThetaX(g_factor=3.0)]).digest() != (
        AugmentationSpec(transforms=[ThetaX(g_factor=2.0)]).digest()
    )
    assert AugmentationSpec(transforms=[ThetaC(probability_factor=0.7)]).digest() != (
        AugmentationSpec(transforms=[ThetaC()]).digest()
    )
    assert AugmentationSpec(transforms=[H0(offset=1.0)]).digest() != (
        AugmentationSpec(transforms=[H0()]).digest()
    )


def test_per_axis_rng_streams_are_independent():
    tensor = _two_area_two_layer_tensor(n=10)
    spec = AugmentationSpec(
        transforms=[W0(factor=1.0, jitter=0.2), H0(offset=0.0, jitter=0.2)], k_v=11
    )
    out, record = augment(tensor, spec)
    assert record.realized_order == ("W_0", "H_0")
    u_w = [entry.after / entry.before for entry in record.changes if entry.axis == "W_0"]
    u_h = [entry.after - entry.before for entry in record.changes if entry.axis == "H_0"]
    assert len(u_w) == 4 and len(u_h) == 4
    assert all(0.8 <= u <= 1.2 for u in u_w)
    assert all(abs(u) <= 0.2 for u in u_h)
    # A shared default_rng(k_v) would give u_h == u_w - 1.0 exactly.
    assert u_h != pytest.approx([u - 1.0 for u in u_w])


def _reversed_area_tensor(n=10):
    """Two areas defined as A1, A0: table order differs from sorted order."""

    def _layer():
        return nt.Layer(
            name="L4",
            n_neurons=n,
            neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
        )

    def _conn():
        return nt.InterConnection(
            source_layer="L4", source_neuron_type="E",
            target_layer="L4", target_neuron_type="PV",
            mechanism="AMPA",
            static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
            plastic=nt.PlasticParams(w_mech=2.0, H=1.0),
        )

    a1 = nt.Area(name="A1", layers=[_layer()], inter_connections=[_conn()])
    a0 = nt.Area(name="A0", layers=[_layer()], inter_connections=[_conn()])
    return nt.NeuronalTensor(areas=[a1, a0], name="reversed")


def test_target_order_canonicalised_for_output_and_digest():
    tensor = _reversed_area_tensor()  # definition order A1, A0: not alphabetical
    spec_a = AugmentationSpec(
        transforms=[
            W0(factor=2.0, jitter=0.1, targets=("A0/inter/0", "A1/inter/0"))
        ],
        k_v=3,
    )
    spec_b = AugmentationSpec(
        transforms=[
            W0(factor=2.0, jitter=0.1, targets=("A1/inter/0", "A0/inter/0"))
        ],
        k_v=3,
    )
    assert spec_a.transforms[0].targets == ("A0/inter/0", "A1/inter/0")
    assert spec_b.transforms[0].targets == ("A0/inter/0", "A1/inter/0")
    assert spec_a.digest() == spec_b.digest()
    out_a, rec_a = augment(tensor, spec_a)
    out_b, rec_b = augment(tensor, spec_b)
    assert out_a.to_dict() == out_b.to_dict()
    assert rec_a == rec_b
    # Table (address) order governs sampling, not sorted order.
    assert [entry.address for entry in rec_a.changes] == [
        "A1/inter/0.plastic.w_mech",
        "A0/inter/0.plastic.w_mech",
    ]


def test_thetax_scales_only_connection_mechanism_key():
    tensor = _two_area_two_layer_tensor(n=10)
    conn = tensor.areas[0].inter_connections[0]
    assert conn.mechanism == "AMPA"
    conn.static.g_mech["GABA_A"] = 0.5  # extra key the connection never uses
    before = tensor.to_dict()
    out, record = augment(tensor, AugmentationSpec(transforms=[ThetaX(g_factor=2.0)]))
    g_entries = [entry for entry in record.changes if "g_mech" in entry.address]
    assert [entry.address for entry in g_entries] == [
        "A0/inter/0.static.g_mech.AMPA",
        "A0/inter/1.static.g_mech.AMPA",
        "A1/inter/0.static.g_mech.AMPA",
        "area_connection/0.static.g_mech.AMPA",
    ]
    assert out.areas[0].inter_connections[0].static.g_mech == {"AMPA": 2.0, "GABA_A": 0.5}
    assert tensor.to_dict() == before  # input tensor never mutated


def test_thetax_undeclared_mechanism_conductance_refused():
    tensor = _two_area_two_layer_tensor(n=10)
    tensor.areas[0].inter_connections[0].static.g_mech = {}
    with pytest.raises(ValueError, match="undeclared conductance") as exc:
        augment(tensor, AugmentationSpec(transforms=[ThetaX(g_factor=2.0)]))
    assert "A0/inter/0" in str(exc.value)
    # Tau-only scaling never touches g: no refusal, no g entries.
    out, record = augment(tensor, AugmentationSpec(transforms=[ThetaX(tau_factor=2.0)]))
    assert record.realized_order == ("Theta_X",)
    assert all("g_mech" not in entry.address for entry in record.changes)
    assert out.areas[0].inter_connections[0].static.g_mech == {}


def _rule_weights(model):
    return [float(rule["weight"]) for rule in model.cfg.metadata["circuit"]["connections"]]


def test_w0_consumed_by_construct_edge_weights():
    tensor = _two_area_two_layer_tensor(n=10)
    out, _ = augment(tensor, AugmentationSpec(transforms=[W0(factor=2.0)]))
    base_weights = _rule_weights(_build(tensor, seed=0))
    aug_weights = _rule_weights(_build(out, seed=0))
    assert len(base_weights) == 4
    for base, aug in zip(base_weights, aug_weights):
        assert aug == pytest.approx(2.0 * base)


def test_thetax_g_consumed_by_construct_edge_weights():
    tensor = _two_area_two_layer_tensor(n=10)
    out, _ = augment(tensor, AugmentationSpec(transforms=[ThetaX(g_factor=2.0)]))
    base_weights = _rule_weights(_build(tensor, seed=0))
    aug_weights = _rule_weights(_build(out, seed=0))
    assert len(base_weights) == 4
    for base, aug in zip(base_weights, aug_weights):
        assert aug == pytest.approx(2.0 * base)


def test_thetax_tau_consumed_by_construct_mechanism():
    tensor = _two_area_two_layer_tensor(n=10)
    out, _ = augment(tensor, AugmentationSpec(transforms=[ThetaX(tau_factor=2.0)]))

    def _taus(model):
        return [
            (mech["name"], float(mech["params"]["tau_ms"]))
            for mech in model.cfg.metadata["circuit"]["mechanisms"]
        ]

    assert _taus(_build(tensor, seed=0)) == [("AMPA__dt2__0", 2.0)]
    assert _taus(_build(out, seed=0)) == [("AMPA__dt4__0", 4.0)]


def test_thetac_delay_consumed_by_construct_delay_steps():
    from jaxfne.emitters import resolve_edge_delay_steps

    tensor = _delayed_tensor()  # delays 1.0, 1.0, 1.0 and cross 2.0
    out, _ = augment(tensor, AugmentationSpec(transforms=[ThetaC(delay_factor=2.0)]))
    base = _build(tensor, seed=0)
    aug = _build(out, seed=0)

    def _rule_delays(model):
        return sorted(float(rule["delay_ms"]) for rule in model.cfg.metadata["circuit"]["connections"])

    def _edge_steps(model):
        return sorted({int(v) for v in jnp.asarray(resolve_edge_delay_steps(model.params["edge_list"]))})

    assert _rule_delays(base) == [1.0, 1.0, 1.0, 2.0]
    assert _rule_delays(aug) == [2.0, 2.0, 2.0, 4.0]
    assert _edge_steps(base) == [2, 4]  # round(delay / 0.5)
    assert _edge_steps(aug) == [4, 8]


def _cross_probability_tensor(prob):
    def _layer():
        return nt.Layer(
            name="L4",
            n_neurons=10,
            neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
        )

    cross = nt.AreaConnection(
        source_area="A0", source_layer="L4", source_neuron_type="E",
        target_area="A1", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=1.5, H=1.0),
        probability=prob,
    )
    return nt.NeuronalTensor(
        areas=[nt.Area(name="A0", layers=[_layer()]), nt.Area(name="A1", layers=[_layer()])],
        area_connections=[cross],
        name="cross_prob",
    )


def test_thetac_probability_consumed_by_construct_edge_count():
    tensor = _cross_probability_tensor(prob=1.0)
    out, record = augment(
        tensor, AugmentationSpec(transforms=[ThetaC(probability_factor=0.1)])
    )
    assert record.realized_order == ("Theta_C",)
    base = _build(tensor, seed=0)
    aug = _build(out, seed=0)
    base_rules = base.cfg.metadata["circuit"]["connections"]
    aug_rules = aug.cfg.metadata["circuit"]["connections"]
    assert len(base_rules) == 1 == len(aug_rules)
    assert base_rules[0]["probability"] == pytest.approx(1.0)
    assert aug_rules[0]["probability"] == pytest.approx(0.1)
    assert aug_rules[0]["compiled_n_edges"] < base_rules[0]["compiled_n_edges"]


def test_h0_consumed_by_construct_initial_H():
    import numpy as np

    tensor = _two_area_two_layer_tensor(n=10)  # H = 1.0 everywhere
    out, _ = augment(tensor, AugmentationSpec(transforms=[H0(offset=0.5)]))
    base_h = np.asarray(_build(tensor, seed=0).params["hdp_initial_H"])
    aug_h = np.asarray(_build(out, seed=0).params["hdp_initial_H"])
    assert float(base_h.min()) == pytest.approx(1.0)
    assert float(base_h.max()) == pytest.approx(1.0)
    assert float(aug_h.max()) == pytest.approx(1.5)  # touched target groups shift
    assert float(aug_h.mean()) > float(base_h.mean())
