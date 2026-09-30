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


def test_packet2_records_raise_not_implemented():
    tensor = _tiny_tensor()
    for rec in (ThetaC(), ThetaX(), W0(), H0()):
        spec = AugmentationSpec(transforms=[rec])
        with pytest.raises(NotImplementedError, match="packet 2"):
            augment(tensor, spec)


def test_stochastic_without_kv_refused():
    with pytest.raises(ValueError, match="K_V"):
        AugmentationSpec(transforms=[ThetaX(stochastic=True)])
    spec = AugmentationSpec(transforms=[ThetaX(stochastic=True)], k_v=7)
    assert spec.k_v == 7
    with pytest.raises(NotImplementedError, match="packet 2"):
        augment(_tiny_tensor(), spec)


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
