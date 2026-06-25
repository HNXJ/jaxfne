"""The "neuronal tensor" standard model — canonical network representation.

Every jaxfne circuit reduces to one shape, regardless of how many areas,
layers, or cell types it has:

    NeuronalTensor = [Areas, AreaConnections]
    Area           = [Layers x NeuronTypes, InterConnections]
    Layer          = always 3D geometry (collapse an axis to 0.0 for 2D/1D)
    NeuronType     = E (relative size 2.0 default), PV/SST/VIP (1.0 default)
    InterConnection (within an area)  = [source(Layer,NeuronType), target(Layer,NeuronType), mechanism]
        mechanism is required, no default (e.g. "AMPA", "GABA")
    AreaConnection  (between areas)   = [source(Area,Layer,NeuronType), target(Area,Layer,NeuronType), mechanism]
        mechanism defaults to "monotonic_cable_synapse"

Static parameters (never plastic/trainable): gMech (conductances), reversal
potentials, dT. Plastic parameters: wMech (per-connection gain matrix), H
(homeostatic H-factor — passive ionic charge income per neuron).

Every numeric field carries a ``value_tag`` of ``"calibrated"``,
``"calibrated_proxy"``, or ``"relative"`` (default ``"relative"`` — opt in to
a stronger tag only with evidence backing it).

Configs are saved/loaded as JSON via :func:`save_neuronal_tensor` /
:func:`load_neuronal_tensor` (built on :mod:`jaxfne.io`'s ``save_json``/
``load_json``) — this module never hardcodes a config as a Python literal;
build a :class:`NeuronalTensor` once, then keep variants as JSON files.

This is additive: the existing :class:`jaxfne.core.Configuration` /
:func:`jaxfne.builders.laminar_cortex_config` path is untouched.
:func:`neuronal_tensor_to_configuration` bridges a NeuronalTensor into that
existing ``construct``/``simulate`` pipeline (with a documented fidelity gap:
Configuration only supports one global cell-type fraction map, so per-area/
per-layer cell-type heterogeneity is flattened at construct time).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional, Sequence

from .io import save_json, load_json
from .core import Configuration

ValueTag = Literal["calibrated", "calibrated_proxy", "relative"]

DEFAULT_RELATIVE_SIZE = {"E": 2.0}
DEFAULT_OTHER_RELATIVE_SIZE = 1.0
DEFAULT_AREA_CONNECTION_MECHANISM = "monotonic_cable_synapse"


def default_relative_size(neuron_type: str) -> float:
    """E defaults to 2.0; every other neuron type (PV/SST/VIP/...) defaults to 1.0."""
    return DEFAULT_RELATIVE_SIZE.get(neuron_type, DEFAULT_OTHER_RELATIVE_SIZE)


@dataclass
class Geometry3D:
    """Always 3D. Collapse an axis to a 2D/1D layer by fixing it at 0.0."""
    distribution: str = "uniform_random"
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    z_range: tuple[float, float] = (0.0, 1.0)
    value_tag: ValueTag = "relative"


@dataclass
class NeuronType:
    name: str
    relative_size: float = 1.0
    value_tag: ValueTag = "relative"

    @classmethod
    def make(cls, name: str, relative_size: Optional[float] = None,
             value_tag: ValueTag = "relative") -> "NeuronType":
        return cls(name=name, relative_size=relative_size if relative_size is not None
                    else default_relative_size(name), value_tag=value_tag)


@dataclass
class Layer:
    name: str
    neuron_types: Sequence[NeuronType] = field(default_factory=tuple)
    geometry: Geometry3D = field(default_factory=Geometry3D)
    n_neurons: int = 0


@dataclass
class StaticParams:
    """Never plastic/trainable/gradientable: conductances, reversal potentials, dT."""
    g_mech: dict = field(default_factory=dict)          # mechanism name -> conductance
    reversal_potentials_mV: dict = field(default_factory=dict)  # mechanism name -> E_rev
    dT_ms: float = 0.1
    value_tag: ValueTag = "relative"


@dataclass
class PlasticParams:
    """Trainable/gradientable: per-connection gain (wMech) and homeostatic H-factor."""
    w_mech: float = 1.0   # connection gain; scale up to a matrix per-synapse if needed
    H: float = 0.0         # homeostatic H-factor: passive ionic charge income / relative accumulated charge
    value_tag: ValueTag = "relative"


@dataclass
class InterConnection:
    """Within-area connection: [source(Layer,NeuronType), target(Layer,NeuronType), mechanism]."""
    source_layer: str
    source_neuron_type: str
    target_layer: str
    target_neuron_type: str
    mechanism: str  # required, no default (e.g. "AMPA", "GABA")
    static: StaticParams = field(default_factory=StaticParams)
    plastic: PlasticParams = field(default_factory=PlasticParams)


@dataclass
class Area:
    name: str
    layers: Sequence[Layer] = field(default_factory=tuple)
    inter_connections: Sequence[InterConnection] = field(default_factory=tuple)


@dataclass
class AreaConnection:
    """Between-area connection: [source(Area,Layer,NeuronType), target(Area,Layer,NeuronType), mechanism]."""
    source_area: str
    source_layer: str
    source_neuron_type: str
    target_area: str
    target_layer: str
    target_neuron_type: str
    mechanism: str = DEFAULT_AREA_CONNECTION_MECHANISM
    static: StaticParams = field(default_factory=StaticParams)
    plastic: PlasticParams = field(default_factory=PlasticParams)


@dataclass
class NeuronalTensor:
    """The canonical network representation: [Areas, AreaConnections]."""
    areas: Sequence[Area] = field(default_factory=tuple)
    area_connections: Sequence[AreaConnection] = field(default_factory=tuple)
    name: str = "untitled"

    def to_dict(self) -> dict:
        return asdict(self)


def save_neuronal_tensor(tensor: NeuronalTensor, path: str | Path) -> str:
    """Save a NeuronalTensor as a JSON config file. Configs are data, never code."""
    path = Path(path)
    save_json(tensor.to_dict(), path)
    return str(path)


def load_neuronal_tensor(path: str | Path) -> NeuronalTensor:
    """Load a NeuronalTensor from a JSON config file."""
    raw = load_json(path)
    areas = [
        Area(
            name=a["name"],
            layers=[
                Layer(
                    name=l["name"],
                    neuron_types=[NeuronType(**nt) for nt in l.get("neuron_types", [])],
                    geometry=Geometry3D(**l.get("geometry", {})),
                    n_neurons=l.get("n_neurons", 0),
                )
                for l in a.get("layers", [])
            ],
            inter_connections=[
                InterConnection(
                    source_layer=ic["source_layer"], source_neuron_type=ic["source_neuron_type"],
                    target_layer=ic["target_layer"], target_neuron_type=ic["target_neuron_type"],
                    mechanism=ic["mechanism"],
                    static=StaticParams(**ic.get("static", {})),
                    plastic=PlasticParams(**ic.get("plastic", {})),
                )
                for ic in a.get("inter_connections", [])
            ],
        )
        for a in raw.get("areas", [])
    ]
    area_connections = [
        AreaConnection(
            source_area=ac["source_area"], source_layer=ac["source_layer"],
            source_neuron_type=ac["source_neuron_type"], target_area=ac["target_area"],
            target_layer=ac["target_layer"], target_neuron_type=ac["target_neuron_type"],
            mechanism=ac.get("mechanism", DEFAULT_AREA_CONNECTION_MECHANISM),
            static=StaticParams(**ac.get("static", {})),
            plastic=PlasticParams(**ac.get("plastic", {})),
        )
        for ac in raw.get("area_connections", [])
    ]
    return NeuronalTensor(areas=areas, area_connections=area_connections, name=raw.get("name", "untitled"))


def neuronal_tensor_to_configuration(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
) -> Configuration:
    """Bridge a :class:`NeuronalTensor` into the existing construct/simulate pipeline.

    Known fidelity gap: :class:`jaxfne.core.Configuration` only supports a
    single global cell-type fraction map (set by :meth:`Configuration.cell_types`)
    shared across every declared column, whereas a NeuronalTensor allows distinct
    per-layer ``NeuronType`` membership per area. This bridge aggregates the
    declared neuron types across all layers/areas (weighted by ``Layer.n_neurons``,
    split evenly across the types declared on that layer) into one global
    fraction map. Per-area/per-layer cell-type heterogeneity is therefore
    flattened at construct time — this is a Configuration-grammar limitation,
    not a NeuronalTensor one. Static/plastic synaptic parameters
    (``g_mech``/reversal potentials/``w_mech``/``H``) on ``InterConnection``/
    ``AreaConnection`` are not yet wired to ``Configuration`` and are dropped.

    Returns
    -------
    Configuration
        A configuration buildable via :func:`jaxfne.construct`.
    """
    if not tensor.areas:
        raise ValueError("NeuronalTensor must declare at least one area to bridge")

    type_weight: dict[str, float] = {}
    for area in tensor.areas:
        for layer in area.layers:
            type_names = [nt.name for nt in layer.neuron_types] or ["E"]
            share = float(layer.n_neurons) / len(type_names)
            for name in type_names:
                type_weight[name] = type_weight.get(name, 0.0) + share
    total_weight = sum(type_weight.values()) or 1.0
    cell_type_fractions = {name: weight / total_weight for name, weight in type_weight.items()}

    cfg = Configuration().runtime(seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
    for area in tensor.areas:
        layer_names = [layer.name for layer in area.layers] or ["single"]
        area_n = sum(layer.n_neurons for layer in area.layers) or 1
        cfg = cfg.column(area.name, layers=layer_names, n=area_n)
    cfg = cfg.cell_types(cell_type_fractions)

    if emitter == "izhikevich":
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
    elif emitter == "lif":
        cfg = cfg.set_emitter("lif")
    elif emitter == "glif":
        cfg = cfg.set_emitter("glif")
    else:
        raise ValueError(f"Unknown emitter: {emitter}. Choose from: izhikevich, lif, glif")

    cfg = cfg.probes(["spikes", "V_m"], n_contacts=16)
    cfg = cfg.field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    return cfg
