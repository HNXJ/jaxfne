"""Tensor-first workflow: build a NeuronalTensor (not a Configuration), save
it to JSON, reload it, construct + simulate it.

This is the canonical entry point for the Areas/Layers/NeuronTypes/Connections
data model (see docs/api/neuronal_tensor.md). `Configuration` is only touched
internally by the bridge (`construct_neuronal_tensor`) -- the user-facing code
below never builds one directly.
"""
import jaxfne as jtfne
from jaxfne import (
    NeuronalTensor, Area, AreaConnection, Layer, NeuronType,
    InterConnection, StaticParams, PlasticParams, Pose3D,
    save_neuronal_tensor, load_neuronal_tensor, construct_neuronal_tensor,
)

# -- Define cell types -------------------------------------------------------
E = NeuronType.make("E")    # relative_size=2.0
PV = NeuronType.make("PV")  # relative_size=1.0

# -- Define one cortical area with within-area (InterConnection) wiring -----
L4 = Layer(name="L4", n_neurons=40, neuron_types=[E, PV])
v1_connections = [
    InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
        plastic=PlasticParams(w_mech=1.5, H=1.0),
    ),
    InterConnection(
        source_layer="L4", source_neuron_type="PV",
        target_layer="L4", target_neuron_type="E",
        mechanism="GABA_A",
        static=StaticParams(g_mech={"GABA_A": 1.0}, reversal_potentials_mV={"GABA_A": -80.0}, dT_ms=5.0),
        plastic=PlasticParams(w_mech=2.0, H=1.0),
    ),
]
v1 = Area(name="V1", layers=[L4], inter_connections=v1_connections, pose=Pose3D("xy"))

# -- A second area, wired to V1 via a between-area AreaConnection -----------
mt_layer = Layer(name="L4", n_neurons=40, neuron_types=[E])
mt = Area(name="MT", layers=[mt_layer], pose=Pose3D("xy", translation=(500.0, 0.0, 0.0)))
feedforward = AreaConnection(
    source_area="V1", source_layer="L4", source_neuron_type="E",
    target_area="MT", target_layer="L4", target_neuron_type="E",
    static=StaticParams(dT_ms=3.0),
    plastic=PlasticParams(w_mech=1.0, H=1.0),
)

tensor = NeuronalTensor(areas=[v1, mt], area_connections=[feedforward], name="v1_mt_tensor_first")

# -- Configs are data, not code: save/reload the declared circuit as JSON ---
path = save_neuronal_tensor(tensor, "/tmp/v1_mt_tensor_first.json")
tensor = load_neuronal_tensor(path)
print(f"Saved + reloaded NeuronalTensor JSON at: {path}")

# -- Construct (bridges to Configuration internally) + simulate -------------
model = construct_neuronal_tensor(tensor, seed=0, duration_ms=200.0, dt_ms=0.5)
signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)

print(f"n_neurons: {model.params['emitter'].n_neurons}")
print(f"recurrent_backend: {model.cfg.metadata.get('recurrent_backend')}")
print(f"spikes finite: {bool(signals.spikes.sum() >= 0)}")
print(f"mean rate (Hz): {float(signals.spikes.mean()) * 1000.0 / 0.5:.2f}")
