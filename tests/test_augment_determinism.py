"""Augmentation determinism and provenance (S7, C7).

The public augmentation records that draw from a K_V stream are ThetaC, ThetaX, W0
and H0 (jitter fields). Their seed is ``AugmentationSpec.k_v``. For each one:
the same seed gives identical output, two seeds give different output, and the
record names the transform and binds its parameters.

ScaleN, GeometryTransform (PoseEdit, RangeEdit) take no seed: they are deterministic.
"""
import dataclasses

import pytest

from jaxfne import neuronal_tensor as nt
from jaxfne.augment import H0, AugmentationSpec, ThetaC, ThetaX, W0, augment
from jaxfne.jdna.completion import ORIGIN_AUGMENT_SAMPLED

K_A = 11
K_B = 12
K_PLAIN = 1234  # distinctive seed for the plain-data check


def _fixture():
    """One area, one layer of 10 neurons, two connections with delays, weights and H."""
    layer = nt.Layer(
        name="L4", n_neurons=10,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )
    ee = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=2.0, H=0.5), delay_ms=1.0,
    )
    ep = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA", static=nt.StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=1.5, H=-0.2), delay_ms=1.0,
    )
    area = nt.Area(name="V1", layers=[layer], inter_connections=[ee, ep])
    return nt.NeuronalTensor(areas=[area], name="augment_determinism")


# (record, same record with a different jitter, axis, jitter values the record must bind)
CASES = [
    pytest.param(
        ThetaC(delay_jitter=0.2), ThetaC(delay_jitter=0.25), "Theta_C", (0.2,),
        id="ThetaC",
    ),
    pytest.param(
        ThetaX(g_jitter=0.2, tau_jitter=0.1), ThetaX(g_jitter=0.25, tau_jitter=0.1),
        "Theta_X", (0.2, 0.1), id="ThetaX",
    ),
    pytest.param(
        W0(jitter=0.2), W0(jitter=0.25), "W_0", (0.2,), id="W0",
    ),
    pytest.param(
        H0(jitter=0.5), H0(jitter=0.6), "H_0", (0.5,), id="H0",
    ),
]


@pytest.mark.parametrize("record, _alt, axis, _jitters", CASES)
def test_same_seed_identical_and_different_seed_differs(record, _alt, axis, _jitters):
    tensor = _fixture()
    out_a, rec_a = augment(tensor, AugmentationSpec(transforms=[record], k_v=K_A))
    out_b, rec_b = augment(tensor, AugmentationSpec(transforms=[record], k_v=K_A))
    out_c, rec_c = augment(tensor, AugmentationSpec(transforms=[record], k_v=K_B))
    assert len(rec_a.changes) > 0, "fixture must draw at least one value under K_V"
    assert out_a.to_dict() == out_b.to_dict()
    assert rec_a == rec_b
    assert out_c.to_dict() != out_a.to_dict(), "two seeds must give different output"
    assert rec_c.changes != rec_a.changes


@pytest.mark.parametrize("record, _alt, axis, _jitters", CASES)
def test_record_names_transform_and_binds_its_parameters(record, _alt, axis, _jitters):
    tensor = _fixture()
    spec = AugmentationSpec(transforms=[record], k_v=K_A)
    _, rec = augment(tensor, spec)
    assert len(rec.changes) > 0, "fixture must draw at least one value under K_V"
    assert rec.realized_order == (axis,)
    assert {entry.axis for entry in rec.changes} == {axis}
    assert {entry.origin for entry in rec.changes} == {ORIGIN_AUGMENT_SAMPLED}
    assert rec.spec_digest == spec.digest()
    # The digest binds the seed and the jitter: changing either changes it.
    assert AugmentationSpec(transforms=[record], k_v=K_B).digest() != spec.digest()
    assert AugmentationSpec(transforms=[_alt], k_v=K_A).digest() != spec.digest()


def _plain_leaves(obj):
    """Every scalar leaf of a nested dataclass / tuple / list / dict structure."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            yield from _plain_leaves(getattr(obj, field.name))
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _plain_leaves(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _plain_leaves(item)
    else:
        yield obj


_PLAIN_DATA_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "AugmentationRecord holds only realized values, origins and spec_digest; "
        "K_V and the jitter are bound inside the digest hash, not readable as plain "
        "data from the record (observed: the record's leaves do not contain K_V=1234)"
    ),
)


@pytest.mark.parametrize(
    "record, _alt, axis, jitters",
    [pytest.param(*case.values, id=case.id, marks=_PLAIN_DATA_XFAIL) for case in CASES],
)
def test_record_exposes_seed_and_jitter_as_plain_data(record, _alt, axis, jitters):
    tensor = _fixture()
    spec = AugmentationSpec(transforms=[record], k_v=K_PLAIN)
    _, rec = augment(tensor, spec)
    assert len(rec.changes) > 0, "fixture must draw at least one value under K_V"
    leaves = list(_plain_leaves(rec))
    assert K_PLAIN in leaves, "the record must carry K_V as plain data"
    for jitter in jitters:
        assert jitter in leaves, f"the record must carry jitter {jitter} as plain data"
