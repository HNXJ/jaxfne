"""Augmentation determinism and provenance (S7, C7).

The public augmentation records that draw from a K_V stream are ThetaC, ThetaX, W0
and H0 (jitter fields). Their seed is ``AugmentationSpec.k_v``. For each one:
the same seed gives identical output, two seeds give different output, and the
record names the transform and binds its parameters.

ScaleN, GeometryTransform (PoseEdit, RangeEdit) take no seed: they are deterministic.
"""
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


# (record, same record with a different jitter, axis, expected record.jitters)
CASES = [
    pytest.param(
        ThetaC(delay_jitter=0.2), ThetaC(delay_jitter=0.25), "Theta_C",
        (("ThetaC.delay", 0.2),),
        id="ThetaC",
    ),
    pytest.param(
        ThetaX(g_jitter=0.2, tau_jitter=0.1), ThetaX(g_jitter=0.25, tau_jitter=0.1),
        "Theta_X", (("ThetaX.g", 0.2), ("ThetaX.tau", 0.1)), id="ThetaX",
    ),
    pytest.param(
        W0(jitter=0.2), W0(jitter=0.25), "W_0", (("W0.jitter", 0.2),), id="W0",
    ),
    pytest.param(
        H0(jitter=0.5), H0(jitter=0.6), "H_0", (("H0.jitter", 0.5),), id="H0",
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


@pytest.mark.parametrize("record, _alt, axis, jitters", CASES)
def test_record_exposes_seed_and_jitter_as_plain_data(record, _alt, axis, jitters):
    tensor = _fixture()
    spec = AugmentationSpec(transforms=[record], k_v=K_PLAIN)
    _, rec = augment(tensor, spec)
    assert len(rec.changes) > 0, "fixture must draw at least one value under K_V"
    assert rec.k_v == K_PLAIN
    assert rec.jitters == jitters


def test_record_k_v_field_follows_the_seed_and_the_digest():
    tensor = _fixture()
    record = ThetaC(delay_jitter=0.2)
    assert K_A != K_B, "precondition: the two seeds must differ"
    _, rec_a = augment(tensor, AugmentationSpec(transforms=[record], k_v=K_A))
    _, rec_a_again = augment(tensor, AugmentationSpec(transforms=[record], k_v=K_A))
    _, rec_b = augment(tensor, AugmentationSpec(transforms=[record], k_v=K_B))
    assert len(rec_a.changes) > 0, "fixture must draw at least one value under K_V"
    assert rec_a.k_v == K_A
    assert rec_b.k_v == K_B
    assert rec_a.k_v != rec_b.k_v
    assert rec_a.spec_digest != rec_b.spec_digest
    assert rec_a == rec_a_again
