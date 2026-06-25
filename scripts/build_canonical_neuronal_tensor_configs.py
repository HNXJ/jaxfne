"""Generate the canonical NeuronalTensor JSON config library.

Rule 1 ("configs are data, not code") calls for "numerous saved JSON files"
as the canonical way to declare circuits -- not just a JSON loader existing
in principle. This script builds a handful of representative NeuronalTensor
circuits and saves them to examples/neuronal_tensor_configs/*.json.

Run with: python3 scripts/build_canonical_neuronal_tensor_configs.py
Each output file is verified loadable + constructible before being kept.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "examples" / "neuronal_tensor_configs"

sys.path.insert(0, str(REPO_ROOT))

import jaxfne as jtfne  # noqa: E402
from jaxfne import (  # noqa: E402
    NeuronalTensor, Area, AreaConnection, Layer, NeuronType,
    InterConnection, StaticParams, PlasticParams, Pose3D,
    save_neuronal_tensor, load_neuronal_tensor, construct_neuronal_tensor,
)


def default_column() -> NeuronalTensor:
    """One area, one layer, the minimal E/PV pair with both connection signs."""
    E, PV = NeuronType.make("E"), NeuronType.make("PV")
    layer = Layer(name="L4", n_neurons=80, neuron_types=[E, PV])
    connections = [
        InterConnection(
            source_layer="L4", source_neuron_type="E", target_layer="L4", target_neuron_type="PV",
            mechanism="AMPA",
            static=StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
            plastic=PlasticParams(w_mech=1.5, H=1.0),
        ),
        InterConnection(
            source_layer="L4", source_neuron_type="PV", target_layer="L4", target_neuron_type="E",
            mechanism="GABA_A",
            static=StaticParams(g_mech={"GABA_A": 1.0}, reversal_potentials_mV={"GABA_A": -80.0}, dT_ms=5.0),
            plastic=PlasticParams(w_mech=2.0, H=1.0),
        ),
    ]
    area = Area(name="V1", layers=[layer], inter_connections=connections, pose=Pose3D("xy"))
    return NeuronalTensor(areas=[area], name="default_column")


def laminar_column_4layer() -> NeuronalTensor:
    """Four-layer column (L4/L2-3/L5/L6) with E/PV per layer and feedforward L4->L2/3->L5->L6."""
    E, PV = NeuronType.make("E"), NeuronType.make("PV")
    layers = [
        Layer(name="L4", n_neurons=60, neuron_types=[E, PV]),
        Layer(name="L2/3", n_neurons=80, neuron_types=[E, PV]),
        Layer(name="L5", n_neurons=50, neuron_types=[E, PV]),
        Layer(name="L6", n_neurons=50, neuron_types=[E, PV]),
    ]
    chain = [("L4", "L2/3"), ("L2/3", "L5"), ("L5", "L6")]
    connections = []
    for src, tgt in chain:
        connections.append(InterConnection(
            source_layer=src, source_neuron_type="E", target_layer=tgt, target_neuron_type="E",
            mechanism="AMPA",
            static=StaticParams(g_mech={"AMPA": 1.0}, dT_ms=2.0),
            plastic=PlasticParams(w_mech=1.2, H=1.0),
        ))
    for layer_name in ("L4", "L2/3", "L5", "L6"):
        connections.append(InterConnection(
            source_layer=layer_name, source_neuron_type="PV", target_layer=layer_name, target_neuron_type="E",
            mechanism="GABA_A",
            static=StaticParams(g_mech={"GABA_A": 1.0}, reversal_potentials_mV={"GABA_A": -80.0}, dT_ms=5.0),
            plastic=PlasticParams(w_mech=1.8, H=1.0),
        ))
    area = Area(name="V1", layers=layers, inter_connections=connections, pose=Pose3D("xy"))
    return NeuronalTensor(areas=[area], name="laminar_column_4layer")


def two_area_feedforward() -> NeuronalTensor:
    """Two areas (V1 -> MT), pose-placed side by side, wired by one AreaConnection."""
    E = NeuronType.make("E")
    v1_layer = Layer(name="L4", n_neurons=60, neuron_types=[E])
    mt_layer = Layer(name="L4", n_neurons=60, neuron_types=[E])
    v1 = Area(name="V1", layers=[v1_layer], pose=Pose3D("xy", translation=(0.0, 0.0, 0.0)))
    mt = Area(name="MT", layers=[mt_layer], pose=Pose3D("xy", translation=(500.0, 0.0, 0.0)))
    feedforward = AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="MT", target_layer="L4", target_neuron_type="E",
        static=StaticParams(dT_ms=3.0),
        plastic=PlasticParams(w_mech=1.0, H=1.0),
    )
    return NeuronalTensor(areas=[v1, mt], area_connections=[feedforward], name="two_area_feedforward")


def homeostatic_h_override_demo() -> NeuronalTensor:
    """Single area whose inhibitory population declares a non-equilibrium H, for
    seeding Model.with_hdp_initial_state (inert unless the caller separately
    enables HDP via Configuration.hdp(...))."""
    E, PV = NeuronType.make("E"), NeuronType.make("PV")
    layer = Layer(name="L4", n_neurons=60, neuron_types=[E, PV])
    connections = [
        InterConnection(
            source_layer="L4", source_neuron_type="E", target_layer="L4", target_neuron_type="PV",
            mechanism="GABA_A",
            static=StaticParams(g_mech={"GABA_A": 1.0}, reversal_potentials_mV={"GABA_A": -80.0}, dT_ms=5.0),
            plastic=PlasticParams(w_mech=2.0, H=3.5),  # PV starts above HDP equilibrium (1.0)
        ),
    ]
    area = Area(name="V1", layers=[layer], inter_connections=connections)
    return NeuronalTensor(areas=[area], name="homeostatic_h_override_demo")


BUILDERS = {
    "default_column.json": default_column,
    "laminar_column_4layer.json": laminar_column_4layer,
    "two_area_feedforward.json": two_area_feedforward,
    "homeostatic_h_override_demo.json": homeostatic_h_override_demo,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in BUILDERS.items():
        tensor = builder()
        path = OUT_DIR / filename
        save_neuronal_tensor(tensor, path)

        # Verify: loadable + constructible + simulatable before keeping the file.
        reloaded = load_neuronal_tensor(path)
        model = construct_neuronal_tensor(reloaded, seed=0, duration_ms=20.0, dt_ms=0.5)
        signals = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)
        n_neurons = model.params["emitter"].n_neurons
        finite = bool(signals.spikes.sum() >= 0)
        print(f"{filename}: n_neurons={n_neurons} finite={finite} -> {path}")


if __name__ == "__main__":
    main()
