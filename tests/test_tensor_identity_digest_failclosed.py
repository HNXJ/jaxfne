"""Fail-closed tensor identity digest: a broken to_dict must raise, never ""."""

import pytest

from jaxfne import neuronal_tensor as nt


def _tiny_tensor():
    layer = nt.Layer(
        name="L4",
        n_neurons=10,
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
    return nt.NeuronalTensor(areas=[area], name="tiny_digest")


def test_identity_digest_raises_when_to_dict_fails(monkeypatch):
    tensor = _tiny_tensor()

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(tensor, "to_dict", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        tensor.to_dict()
    with pytest.raises(ValueError, match="tiny_digest"):
        nt._tensor_identity_digest(tensor)
