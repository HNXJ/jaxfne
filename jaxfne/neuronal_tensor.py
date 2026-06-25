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
existing ``construct``/``simulate`` pipeline.

Multi-area placement: each :class:`Area` carries a :class:`Pose3D` (plane +
rotation + translation) controlling where its layer stack sits in global 3D
space — e.g. orthogonal to xy/xz/yz, tilted 45 degrees, offset anywhere.
:func:`merge_neuronal_tensors` is the "unifier": it takes several existing
NeuronalTensor configs and concatenates their areas/area_connections into one
NeuronalTensor (renaming on name collision). :func:`construct_neuronal_tensor`
bridges + constructs + then overwrites the model's positions with the
pose-correct global placement (jaxfne's own construct() only offsets columns
along x with no rotation, so this step happens post-construct).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Literal, Optional, Sequence

import jax
import jax.numpy as jnp

from .io import save_json, load_json
from .core import Configuration, Model, construct

ValueTag = Literal["calibrated", "calibrated_proxy", "relative"]
Plane = Literal["xy", "xz", "yz"]

DEFAULT_RELATIVE_SIZE = {"E": 2.0}
DEFAULT_OTHER_RELATIVE_SIZE = 1.0
DEFAULT_AREA_CONNECTION_MECHANISM = "monotonic_cable_synapse"

#: Maps a Pose3D.plane to (global_axis_for_local_x, global_axis_for_local_y,
#: global_axis_for_local_depth). "xy" is the canonical default (depth=z, the
#: same orientation jaxfne's own column sampler already uses).
_PLANE_AXIS_MAP: dict[str, tuple[int, int, int]] = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (2, 0, 1),
}


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
class Pose3D:
    """Where an Area's layer stack sits in global 3D space.

    ``plane`` selects which two global axes hold the layer's local (x, y)
    spread and which global axis holds its local depth (z): "xy" (default,
    depth along global z) / "xz" (depth along global y) / "yz" (depth along
    global x) — i.e. "orthogonal to xy/xz/yz". ``rotation_deg`` then twists
    the in-plane (x, y) spread by that many degrees around the depth axis
    (e.g. 45.0 for a 45-degree tilt). ``translation`` shifts the whole area.
    """
    plane: Plane = "xy"
    rotation_deg: float = 0.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    value_tag: ValueTag = "relative"


@dataclass
class Area:
    name: str
    layers: Sequence[Layer] = field(default_factory=tuple)
    inter_connections: Sequence[InterConnection] = field(default_factory=tuple)
    pose: Pose3D = field(default_factory=Pose3D)


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


def _load_pose(area_raw: dict) -> "Pose3D":
    pose_raw = dict(area_raw.get("pose", {}))
    if "translation" in pose_raw:
        pose_raw["translation"] = tuple(pose_raw["translation"])
    return Pose3D(**pose_raw)


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
            pose=_load_pose(a),
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


def merge_neuronal_tensors(
    tensors: Sequence[NeuronalTensor],
    poses: Optional[Sequence[Pose3D]] = None,
    *,
    name: str = "merged",
) -> NeuronalTensor:
    """The "unifier": concatenate several NeuronalTensors' areas into one.

    From the simulator's perspective N areas across M input tensors become one
    flat list of areas (effectively one big layer stack), each individually
    placed via its ``Pose3D``. Area name collisions across inputs are resolved
    by suffixing (``"V1"`` -> ``"V1_1"``); any ``AreaConnection`` inside a given
    input tensor has its ``source_area``/``target_area`` renamed to match.

    Parameters
    ----------
    tensors : Sequence[NeuronalTensor]
        The configs to merge, in order.
    poses : Sequence[Pose3D], optional
        If given, one pose per area in flattened encounter order (area 0 of
        tensor 0, area 1 of tensor 0, ..., area 0 of tensor 1, ...) — overrides
        that area's existing ``pose``. Must have exactly as many entries as
        the total area count across all tensors. If omitted, every area keeps
        its own declared ``pose`` (default: stacked at the same origin).

    Returns
    -------
    NeuronalTensor
        One merged NeuronalTensor; ``area_connections`` from every input are
        preserved (with area names rewritten where collisions were resolved).
        Cross-tensor ``AreaConnection``s are NOT inferred — only declare those
        explicitly on the result if you want areas from different inputs wired
        together.
    """
    total_areas = sum(len(t.areas) for t in tensors)
    if poses is not None and len(poses) != total_areas:
        raise ValueError(f"poses must have one entry per area; got {len(poses)} for {total_areas} areas")

    merged_areas: list[Area] = []
    merged_connections: list[AreaConnection] = []
    seen_names: set[str] = set()
    pose_idx = 0
    for tensor in tensors:
        rename_map: dict[str, str] = {}
        for area in tensor.areas:
            new_name = area.name
            suffix = 1
            while new_name in seen_names:
                new_name = f"{area.name}_{suffix}"
                suffix += 1
            seen_names.add(new_name)
            rename_map[area.name] = new_name
            pose = poses[pose_idx] if poses is not None else area.pose
            pose_idx += 1
            merged_areas.append(replace(area, name=new_name, pose=pose))
        for ac in tensor.area_connections:
            merged_connections.append(replace(
                ac,
                source_area=rename_map.get(ac.source_area, ac.source_area),
                target_area=rename_map.get(ac.target_area, ac.target_area),
            ))
    return NeuronalTensor(areas=merged_areas, area_connections=merged_connections, name=name)


def _sample_local_positions(geometry: Geometry3D, n: int, key: "jax.Array") -> "jax.Array":
    """Sample ``n`` local (x, y, z) points inside a layer's declared Geometry3D ranges."""
    if geometry.distribution != "uniform_random":
        raise NotImplementedError(
            f"Geometry3D.distribution={geometry.distribution!r} is not implemented yet; "
            "only 'uniform_random' is supported."
        )
    if n <= 0:
        return jnp.zeros((0, 3))
    x_key, y_key, z_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, (n,), minval=geometry.x_range[0], maxval=geometry.x_range[1])
    y = jax.random.uniform(y_key, (n,), minval=geometry.y_range[0], maxval=geometry.y_range[1])
    z = jax.random.uniform(z_key, (n,), minval=geometry.z_range[0], maxval=geometry.z_range[1])
    return jnp.stack([x, y, z], axis=1)


def _apply_pose(local_xyz: "jax.Array", pose: Pose3D) -> "jax.Array":
    """Map local (x, y, depth) points into global space per a Pose3D."""
    ax_x, ax_y, ax_depth = _PLANE_AXIS_MAP[pose.plane]
    global_xyz = jnp.zeros_like(local_xyz)
    global_xyz = global_xyz.at[:, ax_x].set(local_xyz[:, 0])
    global_xyz = global_xyz.at[:, ax_y].set(local_xyz[:, 1])
    global_xyz = global_xyz.at[:, ax_depth].set(local_xyz[:, 2])

    theta = jnp.deg2rad(pose.rotation_deg)
    a = global_xyz[:, ax_x]
    b = global_xyz[:, ax_y]
    a_rot = a * jnp.cos(theta) - b * jnp.sin(theta)
    b_rot = a * jnp.sin(theta) + b * jnp.cos(theta)
    global_xyz = global_xyz.at[:, ax_x].set(a_rot).at[:, ax_y].set(b_rot)

    return global_xyz + jnp.asarray(pose.translation, dtype=global_xyz.dtype)


def construct_neuronal_tensor(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
) -> Model:
    """Bridge + construct + apply each Area's Pose3D placement, in one call.

    :func:`jaxfne.construct` only offsets columns along x with no rotation, so
    this samples each layer's local positions from its own declared
    ``Geometry3D`` and re-derives the global placement from each area's
    ``Pose3D`` (plane + rotation + translation) afterward, overwriting
    ``model.params["positions"]`` (and the matching ``x``/``y``/``z`` entries
    in ``model.static["neuron_metadata"]``) so field/LFP/EEG/MEG proxy
    readouts — which read positions from there — see the real layout.

    Each connection's ``PlasticParams.H`` is aggregated (mean, across every
    connection whose target layer/cell_type touches a given neuron;
    untouched neurons default to the HDP equilibrium ``1.0``) into a
    per-neuron initial homeostatic state, applied via
    :meth:`Model.with_hdp_initial_state`. This is stored but inert unless
    the caller separately enables HDP (``cfg.hdp(...)`` before
    :func:`jaxfne.construct`, or via a post-hoc ``RuntimeConfig`` override at
    :func:`jaxfne.simulate` time) — matching ``RuntimeConfig.enable_hdp``'s
    default of ``False``.

    Still does not wire ``AreaConnection``/``InterConnection`` into recurrent
    dynamics (see :func:`neuronal_tensor_to_configuration`'s docstring).
    """
    cfg = neuronal_tensor_to_configuration(
        tensor, seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, emitter=emitter,
    )
    model = construct(cfg)
    rows = model.neuron_table()

    h_sum: dict[tuple[str, str, str], float] = {}
    h_count: dict[tuple[str, str, str], int] = {}

    def _accumulate_h(conn: "InterConnection | AreaConnection", area_name: str) -> None:
        key = (area_name, conn.target_layer, conn.target_neuron_type)
        h_sum[key] = h_sum.get(key, 0.0) + float(conn.plastic.H)
        h_count[key] = h_count.get(key, 0) + 1

    for area in tensor.areas:
        for ic in area.inter_connections:
            _accumulate_h(ic, area.name)
    for ac in tensor.area_connections:
        _accumulate_h(ac, ac.target_area)

    if h_count:
        h_mean = {key: h_sum[key] / h_count[key] for key in h_count}
        H0 = jnp.asarray(
            [h_mean.get((row["area"], row["layer"], row["cell_type"]), 1.0) for row in rows],
            dtype=jnp.float32,
        )
        model = model.with_hdp_initial_state(H0=H0)

    layer_by_key = {(a.name, l.name): l for a in tensor.areas for l in a.layers}
    pose_by_area = {a.name: a.pose for a in tensor.areas}

    base_key = jax.random.PRNGKey(seed)
    position_chunks: list["jax.Array"] = []
    group_index = 0
    i = 0
    n_rows = len(rows)
    while i < n_rows:
        area_name, layer_name = rows[i]["area"], rows[i]["layer"]
        j = i
        while j < n_rows and rows[j]["area"] == area_name and rows[j]["layer"] == layer_name:
            j += 1
        count = j - i
        layer_obj = layer_by_key.get((area_name, layer_name), Layer(name=layer_name))
        pose = pose_by_area.get(area_name, Pose3D())
        sub_key = jax.random.fold_in(base_key, group_index)
        local = _sample_local_positions(layer_obj.geometry, count, sub_key)
        position_chunks.append(_apply_pose(local, pose))
        group_index += 1
        i = j

    positions = jnp.concatenate(position_chunks, axis=0) if position_chunks else jnp.zeros((0, 3))
    updated_rows = [
        dict(row, x=float(positions[idx, 0]), y=float(positions[idx, 1]), z=float(positions[idx, 2]))
        for idx, row in enumerate(rows)
    ]

    new_static = dict(model.static)
    new_static["neuron_metadata"] = updated_rows
    if "geometry" in new_static:
        new_static["geometry"] = dict(new_static["geometry"], neuron_rows=updated_rows)
    new_params = dict(model.params)
    new_params["positions"] = positions
    return replace(model, params=new_params, static=new_static)


def _connection_edge_weight(conn: "InterConnection | AreaConnection", total_n: int) -> float:
    """Native edge magnitude for one InterConnection/AreaConnection.

    ``w_mech`` (plastic gain) is scaled by ``g_mech`` (static conductance for
    this connection's mechanism, default 1.0 if undeclared) and normalized by
    ``1/sqrt(total_n)``, matching the native-weight convention used elsewhere
    in ``core.py`` (e.g. ``_suite2_apply_connectivity``'s ``base_gain * rnd /
    sqrt(n)``) so bridged edges sit on the same magnitude scale as the rest
    of the network instead of dominating or vanishing relative to it.
    """
    g_scale = conn.static.g_mech.get(conn.mechanism, 1.0)
    return abs(float(conn.plastic.w_mech) * float(g_scale)) / math.sqrt(max(total_n, 1))


def _wire_connection(
    cfg: Configuration,
    conn: "InterConnection | AreaConnection",
    *,
    rule_name: str,
    source_area: str,
    target_area: str,
    total_n: int,
    declared_mechanisms: dict[tuple[str, float], str],
) -> Configuration:
    """Declare one real edge rule (mechanism + connection) for an Inter/AreaConnection.

    Uses the generic, selector-based :meth:`Configuration.connections` +
    :meth:`Configuration.mechanisms` hooks (source/target by area/layer/
    cell_type) rather than the same-area-masked dense ``W`` in
    ``_suite2_apply_connectivity`` — selectors carry no same-area restriction,
    so this is what actually couples two different areas (or two specific
    layer x cell-type populations within one area) into the simulated
    dynamics. ``mechanism`` -> ``tau_ms`` comes from ``static.dT_ms``;
    mechanisms are deduplicated by (name, dT_ms) so repeated connections
    sharing both don't raise on ``Configuration``'s duplicate-name guard.
    """
    mech_key = (conn.mechanism, float(conn.static.dT_ms))
    mech_name = declared_mechanisms.get(mech_key)
    if mech_name is None:
        mech_name = f"{conn.mechanism}__dt{conn.static.dT_ms:g}__{len(declared_mechanisms)}"
        mech_params: dict[str, object] = {"tau_ms": float(conn.static.dT_ms)}
        reversal_mV = conn.static.reversal_potentials_mV.get(conn.mechanism)
        if reversal_mV is not None:
            # Declared metadata only: jaxfne's compiled edges are
            # current-based (Izhikevich-style), with no conductance/
            # reversal-potential term anywhere in the dynamics — this value
            # is surfaced into cfg.metadata["circuit"]["mechanisms"] for
            # inspection/provenance, never consumed by simulate().
            mech_params["reversal_mV"] = float(reversal_mV)
        cfg = cfg.mechanisms(name=mech_name, kind=conn.mechanism, params=mech_params)
        declared_mechanisms[mech_key] = mech_name

    sign = "excitatory" if conn.source_neuron_type == "E" else "inhibitory"
    cfg = cfg.connections(
        name=rule_name,
        source={"area": source_area, "layer": conn.source_layer, "cell_type": conn.source_neuron_type},
        target={"area": target_area, "layer": conn.target_layer, "cell_type": conn.target_neuron_type},
        probability=1.0,
        weight=_connection_edge_weight(conn, total_n),
        sign=sign,
        mechanism=mech_name,
    )
    return cfg


def neuronal_tensor_to_configuration(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
) -> Configuration:
    """Bridge a :class:`NeuronalTensor` into the existing construct/simulate pipeline.

    Per-area/per-layer cell-type fractions ARE preserved, via
    :meth:`Configuration.area_layer_cell_types` — each layer's declared
    ``NeuronType`` list is split into even fractions (the tensor stores
    membership, not population fractions, so an even split is the most
    neutral reading) and recorded per area/layer, not flattened globally.

    Cross-area / within-area wiring: every ``InterConnection`` (within an
    area) and ``AreaConnection`` (between areas) is compiled into a REAL edge
    rule via :meth:`Configuration.connections` + :meth:`Configuration.mechanisms`
    (selector-based: ``area``/``layer``/``cell_type``), which carries no
    same-area restriction — this is what actually couples two areas (jaxfne's
    own default recurrent ``W`` in ``_suite2_apply_connectivity`` is
    same-area-masked and would leave bridged areas dynamically isolated).
    ``mechanism`` resolves to a real per-edge ``tau_ms`` (from
    ``static.dT_ms``); edge magnitude is ``w_mech * g_mech / sqrt(total_n)``
    (see :func:`_connection_edge_weight`); sign follows the source neuron
    type (E -> excitatory, else inhibitory). Each rule connects with
    ``probability=1.0`` (full bipartite between the selected populations) —
    the tensor model declares connection membership, not a separate density
    parameter, so full density between the declared layer x cell-type pair
    is the most neutral reading, mirroring the even-split reading used for
    cell-type fractions above.

    Known fidelity gaps still open (not yet wired to ``Configuration``):

    - ``Layer.geometry`` (per-layer ``distribution``/``x_range``/``y_range``/
      ``z_range``) is dropped by THIS function; positions instead come from
      jaxfne's default uniform-random column radius/height. Use
      :func:`construct_neuronal_tensor` instead of calling this bridge alone
      if you need pose-correct/declared-geometry 3D placement — it bridges
      via this function then overwrites positions from each ``Layer.geometry``
      + ``Area.pose``.
    - ``StaticParams.reversal_potentials_mV`` is surfaced into each declared
      mechanism's ``params["reversal_mV"]`` (visible in
      ``cfg.metadata["circuit"]["mechanisms"]``) but still has no numeric
      consumer in jaxfne's dynamics — the compiled edges are native
      current-based (Izhikevich-style), not a conductance/reversal-potential
      scheme. Declared, inspectable metadata only; does not affect
      ``simulate()`` output.
    - ``PlasticParams.H`` (homeostatic charge-balance factor) seeds the HDP
      controller's initial per-neuron state, but ONLY when the caller
      separately enables HDP via ``Configuration.hdp(...)`` — see
      :func:`construct_neuronal_tensor`, which aggregates per-neuron ``H``
      from every connection touching that neuron and applies it via
      ``Model.with_hdp_initial_state``. Stored but inert when HDP is
      disabled (jaxfne's default).

    Returns
    -------
    Configuration
        A configuration buildable via :func:`jaxfne.construct`.
    """
    if not tensor.areas:
        raise ValueError("NeuronalTensor must declare at least one area to bridge")

    cfg = Configuration().runtime(seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
    fallback_weight: dict[str, float] = {}
    area_n_by_name: dict[str, int] = {}
    for area in tensor.areas:
        layer_names = [layer.name for layer in area.layers] or ["single"]
        area_n = sum(layer.n_neurons for layer in area.layers) or 1
        area_n_by_name[area.name] = area_n
        cfg = cfg.column(area.name, layers=layer_names, n=area_n)

        layer_cell_types: dict[str, dict[str, float]] = {}
        for layer in area.layers:
            type_names = [nt.name for nt in layer.neuron_types] or ["E"]
            frac = 1.0 / len(type_names)
            layer_cell_types[layer.name] = {name: frac for name in type_names}
            for name in type_names:
                fallback_weight[name] = fallback_weight.get(name, 0.0) + float(layer.n_neurons) * frac
        if layer_cell_types:
            cfg = cfg.area_layer_cell_types(area.name, layer_cell_types)

    total_weight = sum(fallback_weight.values()) or 1.0
    cfg = cfg.cell_types({name: weight / total_weight for name, weight in fallback_weight.items()})

    total_n = sum(area_n_by_name.values()) or 1
    declared_mechanisms: dict[tuple[str, float], str] = {}
    for area in tensor.areas:
        for idx, ic in enumerate(area.inter_connections):
            cfg = _wire_connection(
                cfg, ic,
                rule_name=f"interconn_{area.name}_{idx}",
                source_area=area.name, target_area=area.name,
                total_n=total_n, declared_mechanisms=declared_mechanisms,
            )
    for idx, ac in enumerate(tensor.area_connections):
        cfg = _wire_connection(
            cfg, ac,
            rule_name=f"areaconn_{idx}",
            source_area=ac.source_area, target_area=ac.target_area,
            total_n=total_n, declared_mechanisms=declared_mechanisms,
        )

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
