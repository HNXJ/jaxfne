"""Plasticity boat development model, 100 neurons (dev-100), no built-in tonic drive.

Owner-approved for the development model only (criteria_v1_frozen.md, Model (dev-100)).
The 10-neuron fixture (explore harness build_model) is a separate construction and is unchanged.

Built-in drive is removed by declaring ``drive=0.0`` on both NeuronTypes (declared drive, not a
post-hoc edit of the drive array). Expected layout: E at 0-79, PV at 80-99, 9900 edges.
"""
from __future__ import annotations

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt

N_NEURONS = 100
E_FRACTION = 0.8
PV_FRACTION = 0.2
DRIVE = 0.0
SEED = 0
DT_MS = 1.0
DURATION_MS = 20.0


def build_dev100():
    """The dev-100 network: one area, one layer, E and PV, four inter-connections as the fixture."""

    def conn(src, dst, mech):
        return nt.InterConnection(
            source_layer="L", source_neuron_type=src, target_layer="L",
            target_neuron_type=dst, mechanism=mech,
            static=nt.StaticParams(dT_ms=5.0))

    tensor = nt.NeuronalTensor(
        name="crazy-sailor-dev100",
        areas=[nt.Area(name="A", layers=[nt.Layer(
            name="L", n_neurons=N_NEURONS,
            neuron_types=[nt.NeuronType.make("E", fraction=E_FRACTION, drive=DRIVE),
                          nt.NeuronType.make("PV", fraction=PV_FRACTION, drive=DRIVE)])],
            inter_connections=[
                conn("E", "E", "AMPA"), conn("E", "PV", "AMPA"),
                conn("PV", "E", "GABA_A"), conn("PV", "PV", "GABA_A")])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=SEED, dt_ms=DT_MS, emitter="lif", duration_ms=DURATION_MS))
