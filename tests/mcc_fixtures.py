"""Small, explicit Minimal Complete Circuit fixtures.

These builders intentionally stay test-local.  They exercise the existing
Configuration -> construct -> simulate path without introducing a second
scientific or simulation engine.
"""

from __future__ import annotations

from typing import Any

import jaxfne as jtfne


MCC_COVERAGE_MAP: dict[str, dict[str, bool]] = {
    "CircuitSpec": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
    "NeuronalTensor": {"MCC-1": True, "MCC-2": False, "MCC-3": False},
    "Model": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
    "emitter": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
    "source": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
    "field": {"MCC-1": True, "MCC-2": False, "MCC-3": True},
    "probe": {"MCC-1": True, "MCC-2": False, "MCC-3": False},
    "Signals": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
    "HDP": {"MCC-1": False, "MCC-2": True, "MCC-3": False},
    "continuation": {"MCC-1": False, "MCC-2": True, "MCC-3": False},
    "PRNG": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
    "objective": {"MCC-1": True, "MCC-2": False, "MCC-3": True},
    "optimizer": {"MCC-1": False, "MCC-2": False, "MCC-3": True},
    "manifest": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
    "validation": {"MCC-1": True, "MCC-2": True, "MCC-3": True},
}


def mcc_configuration() -> Any:
    """Return the canonical ten-neuron two-layer E/PV circuit.

    The small configuration path is deliberately used instead of the tensor
    path: its live edge list can be inspected directly and proves the
    four-class, 90-edge topology without relying on tensor connectivity
    normalization.  The graph is complete off-diagonal at this tiny size, but
    execution uses the explicit sparse edge-list representation.
    """

    return (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=20.0, dt_ms=0.5, dtype="float32")
        .population(
            10,
            neurons={"E": 0.5, "I": 0.5},
            layers=["L2/3", "L4"],
            name="V1",
        )
        .cell_types({"E": 0.5, "PV": 0.5})
        .geometry(layer_thickness={"L2/3": 0.5, "L4": 0.5})
        .set_emitter("izhikevich", "cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="mcc_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def mcc_model() -> Any:
    """Construct one canonical MCC model."""

    return jtfne.construct(mcc_configuration())


def mcc_tensor() -> Any:
    """Return the tensor realization of the canonical MCC-1 graph."""
    neuron_types = [
        jtfne.NeuronType.make("E", fraction=0.5),
        jtfne.NeuronType.make("PV", fraction=0.5),
    ]
    layers = [
        jtfne.Layer(name="L2/3", n_neurons=5, neuron_types=neuron_types),
        jtfne.Layer(name="L4", n_neurons=5, neuron_types=neuron_types),
    ]
    connections = [
        jtfne.InterConnection(
            source_layer=source_layer,
            source_neuron_type=source_type,
            target_layer=target_layer,
            target_neuron_type=target_type,
            mechanism="AMPA" if source_type == "E" else "GABA_A",
        )
        for source_layer in ("L2/3", "L4")
        for target_layer in ("L2/3", "L4")
        for source_type in ("E", "PV")
        for target_type in ("E", "PV")
    ]
    return jtfne.NeuronalTensor(
        areas=[
            jtfne.Area(
                name="V1",
                layers=layers,
                inter_connections=connections,
            )
        ]
    )


def mcc_tensor_model() -> Any:
    """Construct the tensor realization of MCC-1."""
    return jtfne.construct(
        mcc_tensor(),
        jtfne.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=0.5),
    )


def edge_list_runtime(*, enable_hdp: bool = False, synaptic_kernel: str = "exponential") -> Any:
    """Return the explicit runtime used by the MCC edge-list fixtures."""

    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        synaptic_kernel=synaptic_kernel,
        enable_hdp=enable_hdp,
        hdp_params={},
    )


def hdp_runtime(*, noise_scale: float = 0.0, **overrides: Any) -> Any:
    """Return a short, active scalar-H HDP runtime for MCC-2."""

    params = {
        "K_HDP": 0.01,
        "K_ctrl": 0.15,
        "K_w_ctrl": 0.001,
        "tau_0_ms": 20.0,
        "alpha": 0.01,
        "barrier_c": 0.01,
        "barrier_d": 0.01,
        "noise_scale": noise_scale,
    }
    params.update(overrides)
    return jtfne.RuntimeConfig(
        dtype="float32",
        recurrent_backend="edge_list",
        synaptic_kernel="exponential",
        enable_hdp=True,
        hdp_params=params,
    )


def mcc_stimulus() -> Any:
    """Return a deterministic targeted event schedule for MCC-1."""

    return jtfne.stimulus_schedule(
        (
            {
                "label": "targeted_probe_pulse",
                "onset_ms": 2.0,
                "duration_ms": 4.0,
                "amplitude": 2.0,
                "target_indices": (0, 5),
            },
        ),
        n_neurons=10,
    )


def objective_dict(objective: Any) -> dict[str, Any]:
    """Convert an Objective to the manifest's existing JSON-shaped contract."""

    return {
        "name": objective.name,
        "losses": objective.losses,
        "regularizers": objective.regularizers,
        "gates": objective.gates,
    }
