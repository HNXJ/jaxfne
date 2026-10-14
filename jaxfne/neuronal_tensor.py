"""The "neuronal tensor" standard model — canonical network representation.

Every jaxfne circuit reduces to one shape, regardless of how many areas,
layers, or cell types it has:

    NeuronalTensor = [Areas, AreaConnections]
    Area           = [Layers x NeuronTypes, InterConnections]
    connectivity   = omitted (UNSPECIFIED) or declared (EXPLICIT)
    Layer          = always 3D geometry (collapse an axis to 0.0 for 2D/1D)
    NeuronType     = E (relative size 5.0 default), PV (1.0), SST/VIP (1.5)
                      -- see emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE, the
                      actual single source of truth this module imports below
    InterConnection (within an area)  = [source(Layer,NeuronType), target(Layer,NeuronType), mechanism]
        mechanism is required, no default (e.g. "AMPA", "GABA_A")
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

import hashlib
import math
import warnings
import weakref
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import jax
import jax.numpy as jnp

from .io import save_json, load_json
from .core import Configuration, Model, construct
from ._config import check_n_contacts
from .emitters import DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE
from .emitter_registry import check_emitter_params, is_registered_emitter
from .presets import RECEPTOR_KINETICS

ValueTag = Literal["calibrated", "calibrated_proxy", "relative"]
ConnectivityMode = Literal["unspecified", "explicit"]
Plane = Literal["xy", "xz", "yz"]

# Single source of truth for NeuronType relative sizes — same table as HDP
# tau_i = tau_0_ms * size_i**3 scaling (emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE).
DEFAULT_RELATIVE_SIZE = DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE
DEFAULT_OTHER_RELATIVE_SIZE = 1.0
DEFAULT_AREA_CONNECTION_MECHANISM = "monotonic_cable_synapse"

#: P-023 declaration vocabulary for NeuronalTensor connection mechanisms.
#: Excitatory receptors may only be driven by E sources; inhibitory receptors
#: only by non-E sources. Any other name is refused as unknown, except the
#: AreaConnection default above, whose sign follows the source cell type.
EXCITATORY_MECHANISMS = frozenset({"AMPA", "NMDA"})
INHIBITORY_MECHANISMS = frozenset({"GABA_A", "GABA_B"})

#: Emitter families the NeuronalTensor bridge routes to a constructor today.
#: Any other registered name is declaration-valid (see
#: :mod:`jaxfne.emitter_registry`) but refused at bridge time as unroutable.
ROUTABLE_TENSOR_EMITTER_FAMILIES = frozenset({"izhikevich", "lif", "glif", "adex"})

#: Length units accepted for geometry-derived delays
#: (:attr:`InterConnection.length_unit` / :attr:`AreaConnection.length_unit`).
LENGTH_UNITS = frozenset({"um", "mm", "m"})

#: Millimetres per length unit: geometry positions are mm, so a pairwise
#: distance in mm converts by this factor before dividing by a velocity in
#: m/s (``delay_ms = distance_mm * mm_per_unit / velocity``; the default
#: ``"mm"`` reads as ``distance_mm / velocity_m_per_s``).
_MM_PER_LENGTH_UNIT = {"um": 1e-3, "mm": 1.0, "m": 1e3}

#: P-023 TFNE carve-out registry: ids of NeuronalTensor objects minted by
#: :func:`jaxfne.tfne.to_neuronal_tensor`, held weakly alongside the
#: :func:`_tensor_identity_digest` taken at mint time. The carve-out in
#: :func:`_wire_connection` consults this — never ``provenance``, which is a
#: free constructor kwarg any caller can set — so a hand-built tensor carrying
#: ``provenance={"tfne_digest": ...}`` is still refused. A saved-and-reloaded
#: tensor loses the status: :func:`save_neuronal_tensor` already strips
#: provenance, and loading constructs a new (unregistered) object, as do
#: copies, merges and any other reconstruction. An in-place mutation also
#: forfeits the status (content identity: the live digest no longer matches
#: the minted one, so the tensor faces the normal P-023 rules); reverting
#: the mutation restores it. Fail closed throughout.
_TFNE_MINTED: dict[int, tuple["weakref.ReferenceType[NeuronalTensor]", str]] = {}


def _register_tfne_minted(tensor: "NeuronalTensor") -> "NeuronalTensor":
    """Record a tensor minted by the TFNE compiler (called from ``to_neuronal_tensor`` only)."""
    key = id(tensor)
    # _tensor_identity_digest is defined below; resolved at call time (same module).
    _TFNE_MINTED[key] = (
        weakref.ref(tensor, lambda _ref, _key=key: _TFNE_MINTED.pop(_key, None)),
        _tensor_identity_digest(tensor),
    )
    return tensor


def _is_tfne_minted(tensor: "NeuronalTensor") -> bool:
    """Whether this exact object was minted by the TFNE compiler, unmutated, and still alive."""
    entry = _TFNE_MINTED.get(id(tensor))
    if entry is None:
        return False
    ref, stored = entry
    if ref() is not tensor:
        return False
    if not stored:
        return False
    current = _tensor_identity_digest(tensor)
    return bool(current) and current == stored

#: Maps a Pose3D.plane to (global_axis_for_local_x, global_axis_for_local_y,
#: global_axis_for_local_depth). "xy" is the canonical default (depth=z, the
#: same orientation jaxfne's own column sampler already uses).
_PLANE_AXIS_MAP: dict[str, tuple[int, int, int]] = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (2, 0, 1),
}
_UNSPECIFIED_CONNECTIONS = object()


def default_relative_size(neuron_type: str) -> float:
    """Default relative soma size by cell type; matches HDP size-scaling table."""
    return float(DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE.get(neuron_type, DEFAULT_OTHER_RELATIVE_SIZE))


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
    fraction: Optional[float] = None
    value_tag: ValueTag = "relative"
    # Per-type emitter declaration (None = undeclared). When every
    # NeuronType in the tensor declares, the declared family routes
    # construction (uniform family required); when none declares, the
    # runtime emitter applies as before. Partial declaration is refused.
    # Names and ``emitter_params`` keys are validated against
    # :mod:`jaxfne.emitter_registry`; unconsumed keys are refused.
    emitter: Optional[str] = None
    emitter_params: Optional[dict] = None

    @classmethod
    def make(
        cls,
        name: str,
        relative_size: Optional[float] = None,
        fraction: Optional[float] = None,
        value_tag: ValueTag = "relative",
        emitter: Optional[str] = None,
        emitter_params: Optional[dict] = None,
    ) -> "NeuronType":
        return cls(
            name=name,
            relative_size=relative_size
            if relative_size is not None
            else default_relative_size(name),
            fraction=fraction,
            value_tag=value_tag,
            emitter=emitter,
            emitter_params=dict(emitter_params) if emitter_params is not None else None,
        )


@dataclass
class Layer:
    name: str
    neuron_types: Sequence[NeuronType] = field(default_factory=tuple)
    geometry: Geometry3D = field(default_factory=Geometry3D)
    n_neurons: int = 0


@dataclass
class StaticParams:
    """Never plastic/trainable/gradientable: conductances, reversal potentials, dT.

    ``dT_ms`` (synaptic time constant in ms) is REQUIRED (P-023): ``None``
    (the default) is refused at wiring time rather than defaulted.
    """

    g_mech: dict = field(default_factory=dict)  # mechanism name -> conductance
    reversal_potentials_mV: dict = field(default_factory=dict)  # mechanism name -> E_rev
    dT_ms: Optional[float] = None
    value_tag: ValueTag = "relative"


@dataclass
class PlasticParams:
    """Trainable/gradientable: per-connection gain (wMech) and H-factor.

    ``H`` is a legacy kernel-specific label (see
    ``docs/doctrine/rbs_rbd_hdp.md`` §1 carve-out ruling): it names the
    seeding role of this scalar inside the RBD/HDP kernels it feeds, not the
    definition of RBS. RBS/H is not intrinsically homeostatic.
    """

    w_mech: float = 1.0  # connection gain; scale up to a matrix per-synapse if needed
    H: float = (
        0.0  # legacy "homeostatic H-factor" seeding label (kernel-specific; see docstring above)
    )
    value_tag: ValueTag = "relative"


def _check_conduction_velocity(
    velocity: "float | None", length_unit: str, *, label: str
) -> "float | None":
    """Validate geometry-derived delay fields shared by both connection types.

    ``length_unit`` names the unit pairwise distances convert through (one of
    ``{"um", "mm", "m"}``, default ``"mm"``); ``velocity`` (m/s, None =
    undeclared) must be finite and > 0 when given.
    """
    if length_unit not in LENGTH_UNITS:
        raise ValueError(
            f"{label} length_unit must be one of {sorted(LENGTH_UNITS)}; "
            f"got {length_unit!r}"
        )
    if velocity is None:
        return None
    try:
        v = float(velocity)
    except (TypeError, ValueError):
        raise ValueError(f"{label} conduction_velocity_m_per_s must be a number; got {velocity!r}")
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(
            f"{label} conduction_velocity_m_per_s must be finite and > 0; got {velocity!r}"
        )
    return v


def _refuse_delay_and_velocity(delay_ms: "float | None", velocity: "float | None", *, label: str) -> None:
    """Refuse a connection declaring both a constant delay and a velocity.

    ``delay_ms`` is one constant per rule; ``conduction_velocity_m_per_s``
    derives one delay per edge from geometry. Both on one connection is
    ambiguous, so it is refused rather than prioritized.
    """
    if delay_ms is not None and velocity is not None:
        raise ValueError(
            f"{label} declares both delay_ms={delay_ms!r} and "
            f"conduction_velocity_m_per_s={velocity!r}; declare one or neither"
        )


@dataclass
class InterConnection:
    """Within-area connection: [source(Layer,NeuronType), target(Layer,NeuronType), mechanism]."""

    source_layer: str
    source_neuron_type: str
    target_layer: str
    target_neuron_type: str
    mechanism: str  # required, no default (e.g. "AMPA", "GABA_A")
    static: StaticParams = field(default_factory=StaticParams)
    plastic: PlasticParams = field(default_factory=PlasticParams)
    # Configured axonal delay in ms (None = undeclared = current behaviour).
    # Carried for inspection and forwarded into the compiled connection rule
    # by :func:`_wire_connection`; realized to steps at construction
    # (0.5.2 decision 0b), where dt is known.
    delay_ms: "float | None" = None
    # Conduction velocity in m/s (None = undeclared). When set, each compiled
    # edge gets its own delay from constructed geometry
    # (``delay_ms = distance_mm * mm_per_unit / velocity``); mutually
    # exclusive with ``delay_ms``. ``length_unit`` names the distance unit.
    conduction_velocity_m_per_s: "float | None" = None
    length_unit: str = "mm"

    def __post_init__(self) -> None:
        if self.delay_ms is not None:
            try:
                ms = float(self.delay_ms)
            except (TypeError, ValueError):
                raise ValueError(
                    f"InterConnection delay_ms must be a number in ms; got {self.delay_ms!r}"
                )
            import math as _math

            if not _math.isfinite(ms) or ms < 0.0:
                raise ValueError(
                    f"InterConnection delay_ms must be finite and >= 0; got {self.delay_ms!r}"
                )
            object.__setattr__(self, "delay_ms", ms)
        object.__setattr__(
            self,
            "conduction_velocity_m_per_s",
            _check_conduction_velocity(
                self.conduction_velocity_m_per_s, self.length_unit,
                label="InterConnection",
            ),
        )
        _refuse_delay_and_velocity(
            self.delay_ms, self.conduction_velocity_m_per_s, label="InterConnection"
        )


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
    inter_connections: Sequence[InterConnection] = field(
        default_factory=lambda: _UNSPECIFIED_CONNECTIONS,
        repr=False,
    )
    pose: Pose3D = field(default_factory=Pose3D)
    connectivity_mode: ConnectivityMode | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "inter_connections" and "_connectivity_initialized" in self.__dict__:
            object.__setattr__(self, "_connectivity_specified", True)
            object.__setattr__(self, "connectivity_mode", "explicit")
        elif name == "connectivity_mode" and "_connectivity_initialized" in self.__dict__:
            if value not in {"unspecified", "explicit"}:
                raise ValueError("Area.connectivity_mode must be 'unspecified' or 'explicit'")
            if value == "unspecified" and self._connectivity_specified:
                raise ValueError(
                    "Area with a provided inter_connections sequence cannot be "
                    "marked as unspecified"
                )
        object.__setattr__(self, name, value)

    def __post_init__(self):
        specified = self.inter_connections is not _UNSPECIFIED_CONNECTIONS
        connections = () if not specified else self.inter_connections
        mode = self.connectivity_mode
        if mode is None:
            mode = "explicit" if specified else "unspecified"
        if mode not in {"unspecified", "explicit"}:
            raise ValueError("Area.connectivity_mode must be 'unspecified' or 'explicit'")
        if mode == "unspecified" and specified:
            raise ValueError(
                "Area with a provided inter_connections sequence cannot be marked as unspecified"
            )
        object.__setattr__(self, "inter_connections", connections)
        object.__setattr__(self, "connectivity_mode", mode)
        object.__setattr__(self, "_connectivity_specified", specified)
        if not isinstance(self.name, str):
            raise TypeError(f"Area.name must be a str, got {type(self.name).__name__}")
        if not all(isinstance(layer, Layer) for layer in self.layers):
            raise TypeError(
                "Area.layers must contain only Layer instances, got "
                f"{[type(layer).__name__ for layer in self.layers]}"
            )
        if not all(isinstance(c, InterConnection) for c in self.inter_connections):
            raise TypeError(
                "Area.inter_connections must contain only InterConnection instances, got "
                f"{[type(c).__name__ for c in self.inter_connections]}"
            )
        object.__setattr__(self, "_connectivity_initialized", True)


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
    # Configured axonal delay in ms (None = undeclared = current behaviour).
    # Same inspection role as InterConnection.delay_ms: forwarded into the
    # compiled rule by :func:`_wire_connection`, realized to steps at
    # construction (0.5.2 decision 0b). Cross-area composition semantics
    # stay 0.5.4 work; the field itself is chain completeness.
    delay_ms: "float | None" = None
    # Conduction velocity in m/s (None = undeclared; mutually exclusive with
    # ``delay_ms``); ``length_unit`` names the distance unit. Same
    # per-edge geometry semantics as InterConnection.
    conduction_velocity_m_per_s: "float | None" = None
    length_unit: str = "mm"
    # Per-pair connection probability in (0, 1] (None = undeclared = 1.0,
    # the all-to-all behaviour before 0.5.5). JDNA's exponential-distance
    # rule derives it per area pair.
    probability: "float | None" = None

    def __post_init__(self) -> None:
        import math as _math

        if self.delay_ms is not None:
            try:
                ms = float(self.delay_ms)
            except (TypeError, ValueError):
                raise ValueError(
                    f"AreaConnection delay_ms must be a number in ms; got {self.delay_ms!r}"
                )

            if not _math.isfinite(ms) or ms < 0.0:
                raise ValueError(
                    f"AreaConnection delay_ms must be finite and >= 0; got {self.delay_ms!r}"
                )
            object.__setattr__(self, "delay_ms", ms)
        object.__setattr__(
            self,
            "conduction_velocity_m_per_s",
            _check_conduction_velocity(
                self.conduction_velocity_m_per_s, self.length_unit,
                label="AreaConnection",
            ),
        )
        _refuse_delay_and_velocity(
            self.delay_ms, self.conduction_velocity_m_per_s, label="AreaConnection"
        )
        if self.probability is not None:
            try:
                p = float(self.probability)
            except (TypeError, ValueError):
                raise ValueError(
                    f"AreaConnection probability must be a number; got {self.probability!r}"
                )
            if not (_math.isfinite(p) and 0.0 < p <= 1.0):
                raise ValueError(
                    f"AreaConnection probability must be in (0, 1]; got {self.probability!r}"
                )
            object.__setattr__(self, "probability", p)


@dataclass
class NeuronalTensor:
    """The canonical network representation: [Areas, AreaConnections].

    Connectivity is inferred from omission versus a provided connection
    sequence. Omitted connectivity uses the configured default topology;
    provided connectivity, including an empty sequence, is authoritative.

    Deliberately unvalidated against arbitrary Python objects beyond a basic
    type check on ``areas``/``area_connections`` — this is a build-time spec,
    not a runtime pytree, so keep validation cheap and loud rather than
    exhaustive. The check below exists specifically to fail fast on the
    documented landmine of passing a ``jaxfne.core.Configuration`` (or any
    other non-``Area`` object) positionally into ``areas`` — previously a
    silent type confusion with no runtime error until something much later
    and more confusing broke.
    """

    areas: Sequence[Area] = field(default_factory=tuple)
    area_connections: Sequence[AreaConnection] = field(
        default_factory=lambda: _UNSPECIFIED_CONNECTIONS,
        repr=False,
    )
    name: str = "untitled"
    connectivity_mode: ConnectivityMode | None = None
    provenance: Optional[dict] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "area_connections" and "_connectivity_initialized" in self.__dict__:
            object.__setattr__(self, "_connectivity_specified", True)
            object.__setattr__(self, "connectivity_mode", "explicit")
            object.__setattr__(self, "_connectivity_mode_declared", True)
        elif name == "connectivity_mode" and "_connectivity_initialized" in self.__dict__:
            if value not in {"unspecified", "explicit"}:
                raise ValueError(
                    "NeuronalTensor.connectivity_mode must be 'unspecified' or 'explicit'"
                )
            if value == "unspecified" and (
                self._connectivity_specified
                or any(a.connectivity_mode == "explicit" for a in self.areas)
            ):
                raise ValueError(
                    "NeuronalTensor with declared connectivity cannot be marked as unspecified"
                )
            object.__setattr__(self, "_connectivity_mode_declared", value == "explicit")
        object.__setattr__(self, name, value)

    def __post_init__(self):
        if isinstance(self.areas, Configuration):
            raise TypeError(
                "NeuronalTensor.areas received a jaxfne.core.Configuration — "
                "there is no Configuration -> NeuronalTensor converter; build a "
                "NeuronalTensor from Area/AreaConnection instances directly, or "
                "use the Configuration/laminar_cortex_config fluent path instead "
                "if that's what you meant to construct."
            )
        if not all(isinstance(a, Area) for a in self.areas):
            raise TypeError(
                "NeuronalTensor.areas must contain only Area instances, got "
                f"{[type(a).__name__ for a in self.areas]}"
            )
        specified = self.area_connections is not _UNSPECIFIED_CONNECTIONS
        connections = () if not specified else self.area_connections
        mode = self.connectivity_mode
        mode_was_provided = mode is not None
        if mode is None:
            mode = (
                "explicit"
                if specified or any(a.connectivity_mode == "explicit" for a in self.areas)
                else "unspecified"
            )
        if mode not in {"unspecified", "explicit"}:
            raise ValueError("NeuronalTensor.connectivity_mode must be 'unspecified' or 'explicit'")
        if mode == "unspecified" and (
            specified or any(a.connectivity_mode == "explicit" for a in self.areas)
        ):
            raise ValueError(
                "NeuronalTensor with declared connectivity cannot be marked as unspecified"
            )
        object.__setattr__(self, "area_connections", connections)
        object.__setattr__(self, "connectivity_mode", mode)
        object.__setattr__(self, "_connectivity_specified", specified)
        object.__setattr__(
            self,
            "_connectivity_mode_declared",
            mode == "explicit" and (mode_was_provided or specified),
        )
        if not all(isinstance(c, AreaConnection) for c in self.area_connections):
            raise TypeError(
                "NeuronalTensor.area_connections must contain only AreaConnection "
                f"instances, got {[type(c).__name__ for c in self.area_connections]}"
            )
        object.__setattr__(self, "_connectivity_initialized", True)
        if not isinstance(self.name, str):
            raise TypeError(f"NeuronalTensor.name must be a str, got {type(self.name).__name__}")

    def to_dict(self) -> dict:
        return asdict(self)


def make_minimal_ei_tensor(
    n: int = 8,
    e_fraction: float = 0.75,
    *,
    layer_name: str = "L1",
    area_name: str = "minimal",
    h: float = 1.0,
) -> NeuronalTensor:
    """One flat Layer of n neurons split E/PV by e_fraction, all 4 pairwise
    E/PV InterConnections (AMPA from E, GABA_A from PV), plastic.H=h on every
    connection -- the minimal all-pairwise E/I circuit shape shared by both
    canonical HDP sanity tests (scripts/major_sanity_test.py,
    scripts/snt_pipeline.py). Time constants are declared per mechanism
    (P-023) from ``presets.RECEPTOR_KINETICS``."""
    e_type = NeuronType.make("E", fraction=e_fraction)
    i_type = NeuronType.make("PV", fraction=1.0 - e_fraction)
    layer = Layer(name=layer_name, n_neurons=n, neuron_types=[e_type, i_type])
    connections = [
        InterConnection(
            source_layer=layer_name,
            source_neuron_type=src,
            target_layer=layer_name,
            target_neuron_type=tgt,
            mechanism=("AMPA" if src == "E" else "GABA_A"),
            static=StaticParams(
                dT_ms=float(
                    RECEPTOR_KINETICS["AMPA" if src == "E" else "GABA_A"]["tau_ms"]
                )
            ),
            plastic=PlasticParams(H=h),
        )
        for src in ("E", "PV")
        for tgt in ("E", "PV")
    ]
    return NeuronalTensor(
        areas=[Area(name=area_name, layers=[layer], inter_connections=connections)]
    )


# Versioned JSON schema for saved NeuronalTensor configs. Bump on any
# breaking change to the on-disk shape (field rename/removal, not additive
# fields); load_neuronal_tensor accepts files with no "schema_version" key
# (pre-versioning saves) as implicitly "neuronal_tensor_v1".
NEURONAL_TENSOR_SCHEMA_VERSION = "neuronal_tensor_v1"


def configs_dir() -> Path:
    """Return the path to the package's canonical NeuronalTensor JSON library
    (``jaxfne/configs/``), shipped as package data. See
    :func:`load_canonical_neuronal_tensor` to load one by name."""
    return Path(__file__).resolve().parent / "configs"


def list_canonical_neuronal_tensors() -> list[str]:
    """Return the names (without ``.json``) of every canonical config in
    :func:`configs_dir`."""
    return sorted(p.stem for p in configs_dir().glob("*.json"))


def load_canonical_neuronal_tensor(name: str) -> NeuronalTensor:
    """Load a canonical NeuronalTensor by name from :func:`configs_dir`.

    ``name`` may be given with or without the ``.json`` suffix, e.g.
    ``load_canonical_neuronal_tensor("default-column")``.
    """
    stem = name[:-5] if name.endswith(".json") else name
    path = configs_dir() / f"{stem}.json"
    if not path.exists():
        available = ", ".join(list_canonical_neuronal_tensors())
        raise FileNotFoundError(
            f"No canonical config named {stem!r} in {configs_dir()}. Available: {available}"
        )
    return load_neuronal_tensor(path)


def save_neuronal_tensor(tensor: NeuronalTensor, path: str | Path) -> str:
    """Save a NeuronalTensor as a JSON config file. Configs are data, never code."""
    path = Path(path)
    payload = dict(tensor.to_dict())
    payload.pop("provenance", None)
    payload["schema_version"] = NEURONAL_TENSOR_SCHEMA_VERSION
    save_json(payload, path)
    return str(path)


def _load_pose(area_raw: dict) -> "Pose3D":
    pose_raw = dict(area_raw.get("pose", {}))
    if "translation" in pose_raw:
        pose_raw["translation"] = tuple(pose_raw["translation"])
    return Pose3D(**pose_raw)


def load(path: str | Path, *, strict: bool = False) -> NeuronalTensor:
    """Canonical loader: load a NeuronalTensor from its JSON config file.

    This is the recommended entry point for the tensor-first workflow
    (``tensor = jtfne.load(path)``). Raises ``ValueError`` if ``path`` does
    not look like a NeuronalTensor JSON config (no top-level ``"areas"``
    key). The legacy ``.jcfg.json``/``JaxFNEConfig`` format (and its
    ``load_config``/``validate_config`` loaders) was removed 2026-06-30 --
    it lived only in tests, never as a real asset outside this package; use
    this function for the NeuronalTensor-schema path, or the
    :class:`Configuration` fluent builder for the Configuration-schema path.
    With ``strict=True``, an unrecognized future ``schema_version`` is refused
    instead of loading with a warning.
    """
    raw = load_json(path)
    if "areas" not in raw:
        raise ValueError(
            f"{path} does not look like a NeuronalTensor JSON config (no top-level 'areas' key)."
        )
    return _load_neuronal_tensor_impl(path, raw, strict=strict)


def load_neuronal_tensor(path: str | Path) -> NeuronalTensor:
    """Compatibility wrapper. Prefer :func:`load`.

    Deprecated for 0.4.14 removal; see ``jaxfne.public_surface.COMPATIBILITY_DEPRECATIONS``.
    """
    return load(path)


def _load_neuronal_tensor_impl(
    path: str | Path, raw: dict, *, strict: bool = False
) -> NeuronalTensor:
    """Validates ``schema_version`` if present (missing = legacy pre-versioning
    save, treated as ``neuronal_tensor_v1`` implicitly). An unrecognized
    *future* version is accepted with a warning, not an error -- this loader
    targets forward-readability of additive fields, not strict lockstep.
    """
    schema_version = raw.get("schema_version", NEURONAL_TENSOR_SCHEMA_VERSION)
    if schema_version != NEURONAL_TENSOR_SCHEMA_VERSION and strict:
        raise ValueError(
            f"NeuronalTensor JSON at {path} declares schema_version="
            f"{schema_version!r}, expected {NEURONAL_TENSOR_SCHEMA_VERSION!r}; "
            "strict=True refuses forward-reads with possibly dropped fields."
        )
    if schema_version != NEURONAL_TENSOR_SCHEMA_VERSION:
        warnings.warn(
            f"NeuronalTensor JSON at {path} declares schema_version="
            f"{schema_version!r}, expected {NEURONAL_TENSOR_SCHEMA_VERSION!r}. "
            "Loading anyway; fields may be silently dropped if the schema "
            "diverged.",
            stacklevel=2,
        )
    legacy_mode = raw.get("connectivity_mode")
    if legacy_mode is None:
        has_declared_connections = bool(raw.get("area_connections"))
        has_declared_connections = has_declared_connections or any(
            bool(area_raw.get("inter_connections")) for area_raw in raw.get("areas", [])
        )
        legacy_mode = "explicit" if has_declared_connections else "unspecified"
    raw_area_connections = raw.get("area_connections", [])
    area_connections_value = (
        _UNSPECIFIED_CONNECTIONS
        if legacy_mode == "unspecified" and not raw_area_connections
        else [
            AreaConnection(
                source_area=ac["source_area"],
                source_layer=ac["source_layer"],
                source_neuron_type=ac["source_neuron_type"],
                target_area=ac["target_area"],
                target_layer=ac["target_layer"],
                target_neuron_type=ac["target_neuron_type"],
                mechanism=ac.get("mechanism", DEFAULT_AREA_CONNECTION_MECHANISM),
                static=StaticParams(**ac.get("static", {})),
                plastic=PlasticParams(**ac.get("plastic", {})),
                # Saved by to_dict/asdict; dropped on load before 0.5.5.
                delay_ms=ac.get("delay_ms"),
                probability=ac.get("probability"),
                conduction_velocity_m_per_s=ac.get("conduction_velocity_m_per_s"),
                length_unit=ac.get("length_unit", "mm"),
            )
            for ac in raw_area_connections
        ]
    )
    areas = [
        Area(
            name=a["name"],
            layers=[
                Layer(
                    name=layer["name"],
                    neuron_types=[NeuronType(**nt) for nt in layer.get("neuron_types", [])],
                    geometry=Geometry3D(**layer.get("geometry", {})),
                    n_neurons=layer.get("n_neurons", 0),
                )
                for layer in a.get("layers", [])
            ],
            inter_connections=(
                _UNSPECIFIED_CONNECTIONS
                if (not a.get("inter_connections", []) and a.get("connectivity_mode") != "explicit")
                else [
                    InterConnection(
                        source_layer=ic["source_layer"],
                        source_neuron_type=ic["source_neuron_type"],
                        target_layer=ic["target_layer"],
                        target_neuron_type=ic["target_neuron_type"],
                        mechanism=ic["mechanism"],
                        static=StaticParams(**ic.get("static", {})),
                        plastic=PlasticParams(**ic.get("plastic", {})),
                        delay_ms=ic.get("delay_ms"),  # dropped on load before 0.5.5
                        conduction_velocity_m_per_s=ic.get("conduction_velocity_m_per_s"),
                        length_unit=ic.get("length_unit", "mm"),
                    )
                    for ic in a.get("inter_connections", [])
                ]
            ),
            pose=_load_pose(a),
            connectivity_mode=(
                a.get(
                    "connectivity_mode",
                    "explicit" if a.get("inter_connections", []) else legacy_mode,
                )
            ),
        )
        for a in raw.get("areas", [])
    ]
    return NeuronalTensor(
        areas=areas,
        area_connections=area_connections_value,
        name=raw.get("name", "untitled"),
        connectivity_mode=legacy_mode,
    )


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
        raise ValueError(
            f"poses must have one entry per area; got {len(poses)} for {total_areas} areas"
        )

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
            if area.connectivity_mode == "unspecified":
                merged_areas.append(
                    replace(
                        area,
                        name=new_name,
                        pose=pose,
                        inter_connections=_UNSPECIFIED_CONNECTIONS,
                    )
                )
            else:
                merged_areas.append(replace(area, name=new_name, pose=pose))
        for ac in tensor.area_connections:
            merged_connections.append(
                replace(
                    ac,
                    source_area=rename_map.get(ac.source_area, ac.source_area),
                    target_area=rename_map.get(ac.target_area, ac.target_area),
                )
            )
    merged_mode: ConnectivityMode = (
        "explicit" if any(t.connectivity_mode == "explicit" for t in tensors) else "unspecified"
    )
    merged_area_connections = (
        _UNSPECIFIED_CONNECTIONS
        if merged_mode == "unspecified" and not merged_connections
        else merged_connections
    )
    return NeuronalTensor(
        areas=merged_areas,
        area_connections=merged_area_connections,
        name=name,
        connectivity_mode=merged_mode,
    )


def _tensor_geometry_domains(
    tensor: NeuronalTensor,
    metadata: dict[str, Any],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Per-(area, layer) declared geometry domains for construction.

    ``Geometry3D`` ranges are column-relative (owner ruling 2026-09-30): x and
    y are fractions of the column's width, z a fraction of the column's depth
    (a JDNA ``depth_band`` of ``[0.1, 0.35]`` is 10-35 % of the column). The
    construction stage reads ``metadata["tfne_geometry"]["domains"]`` as
    fractions of each layer's block; x and y blocks span the whole column, so
    they pass unchanged, and z is converted from the column frame into the
    layer's ``layer_fractions`` block (the result may leave ``[0, 1]`` when
    the band extends past the block). A full ``(0.0, 1.0)`` axis is unconstrained: it is omitted, so
    construction without a sub-range declaration takes the historical code
    path unchanged. A degenerate point range (e.g. ``(0.0, 0.0)``) is kept:
    it collapses that axis to one block fraction, preserving the documented
    2D/1D collapse (at the block position, not at absolute 0.0).

    Only ``distribution="uniform_random"`` is supported -- anything else is
    refused here, where the declaration would otherwise be silently sampled
    as uniform downstream.
    """
    from ._construct_population import _layer_ranges_for

    domains: dict[str, dict[str, dict[str, list[float]]]] = {}
    for area in tensor.areas:
        z_blocks = _layer_ranges_for([layer.name for layer in area.layers], metadata)
        for layer in area.layers:
            geometry = layer.geometry
            if geometry.distribution != "uniform_random":
                raise NotImplementedError(
                    f"Geometry3D.distribution={geometry.distribution!r} is not implemented yet; "
                    "only 'uniform_random' is supported."
                )
            block: dict[str, list[float]] = {}
            for axis, attr in (("x", "x_range"), ("y", "y_range"), ("z", "z_range")):
                lo, hi = (float(v) for v in getattr(geometry, attr))
                if (lo, hi) == (0.0, 1.0):
                    continue
                if axis == "z":
                    z0, z1 = z_blocks[layer.name]
                    if z1 <= z0:
                        raise ValueError(
                            f"{area.name}/{layer.name}: layer block has zero depth "
                            f"({z0}, {z1}); a column-relative z_range cannot be placed in it"
                        )
                    lo, hi = (lo - z0) / (z1 - z0), (hi - z0) / (z1 - z0)
                block[axis] = [lo, hi]
            if block:
                domains.setdefault(area.name, {})[layer.name] = block
    return domains


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


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Execution-only configuration for the tensor-first workflow
    (``jtfne.construct(tensor, runtime)``). Contains no biological structure
    -- areas, layers, populations, geometry, mechanisms, and plastic
    parameters all belong on :class:`NeuronalTensor`, never here.

    Ownership (2026-08-22 W4 scoping): this is the canonical runtime type of
    the *tensor workflow*; it carries no execution-policy semantics beyond the
    wired fields below and is bridged by ``construct`` into
    :class:`jaxfne.RuntimeConfig`, the single owner of execution-policy
    semantics. The two are path-scoped views of one resolved runtime.

    **Wired** (actually consumed by :func:`jaxfne.construct` /
    :func:`jaxfne.simulate` today): ``duration_ms``, ``dt_ms``, ``seed``,
    ``dtype``, ``emitter``, ``device`` (mapped to ``RuntimeConfig.backend``),
    ``jit``, ``n_contacts`` (laminar contacts of the field readout, an int
    >= 2). ``vmap`` must stay ``False``: a no-argument ``simulate`` runs one
    trial, and ``simulate_batch`` reads ``vmap`` from its ``Simulation``.

    ``duration_ms``/``dt_ms``/``seed`` are inherited by a *no-argument*
    ``jtfne.simulate(model)`` (and by ``construct`` itself). Passing an
    explicit ``runtime=`` to ``simulate`` opts into runtime control: the
    ``Simulation``'s own ``duration_ms``/``dt_ms``/``seed`` (defaults
    ``1000.0``/``0.05``/``0``) then apply unless you also pass them to
    ``simulate``.

    **Reserved**: ``solver``, ``probes``, ``outputs``, ``optimizer`` name
    TFNE-grammar stages that this path does not wire; any value other than
    ``None`` is refused. Probes are declared with ``Configuration.probes``,
    recorded outputs with ``Simulation``, and optimization with
    ``Model.tune``.
    """

    duration_ms: float = 1000.0
    dt_ms: float = 0.1
    seed: int = 0
    dtype: str = "float32"
    emitter: str = "izhikevich"
    device: str = "auto"
    jit: "bool | str" = False
    vmap: "bool | str" = False
    solver: "str | None" = None
    probes: "Sequence[str] | None" = None
    n_contacts: int = 16
    outputs: "dict | None" = None
    optimizer: "Any | None" = None
    # AdEx build-call knobs (defaults "exp"/"default", recorded into
    # static["adex"] on the adex path only). Any other value is refused
    # there; non-defaults on a non-adex tensor are refused as unconsumed.
    adex_decay: str = "exp"
    adex_step_order: str = "default"

    def __post_init__(self) -> None:
        for name in ("solver", "probes", "outputs", "optimizer"):
            if getattr(self, name) is not None:
                raise ValueError(
                    f"RuntimeConfiguration.{name} is reserved and not wired on the tensor "
                    f"path; got {getattr(self, name)!r}. Leave it None."
                )
        check_n_contacts(self.n_contacts, "RuntimeConfiguration")
        if self.vmap is not False:
            raise ValueError(
                "RuntimeConfiguration.vmap is not consumed: simulate(model) runs one trial, and "
                "simulate_batch takes its runtime from the Simulation. Use "
                "Simulation(runtime=RuntimeConfig(vmap=...)) with simulate_batch."
            )


def construct_neuronal_tensor(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
    adex_decay: str = "exp",
    adex_step_order: str = "default",
) -> Model:
    """Compatibility wrapper around :func:`jaxfne.construct`.

    Prefer ``jtfne.construct(tensor, RuntimeConfiguration(seed=..., ...))``
    for new code -- this kwarg-style call remains supported unchanged.

    Deprecated for 0.4.14 removal; see ``jaxfne.public_surface.COMPATIBILITY_DEPRECATIONS``.
    """
    runtime = RuntimeConfiguration(
        seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, emitter=emitter,
        adex_decay=adex_decay, adex_step_order=adex_step_order,
    )
    return construct(tensor, runtime)


def _construct_neuronal_tensor_impl(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
    dtype: str = "float32",
    backend: "str | None" = None,
    jit: "bool | str | None" = None,
    vmap: "bool | str | None" = None,
    n_contacts: int = 16,
    adex_decay: str = "exp",
    adex_step_order: str = "default",
) -> Model:
    """Bridge + construct + apply each Area's Pose3D placement, in one call.

    :func:`jaxfne.construct` only offsets columns along x with no rotation, so
    this applies each area's ``Pose3D`` (plane + rotation + translation) to
    the constructed positions afterward, overwriting
    ``model.params["positions"]`` (and the matching ``x``/``y``/``z`` entries
    in ``model.static["neuron_metadata"]``) so field/LFP/EEG/MEG proxy
    readouts — which read positions from there — see the real layout.

    Each connection's ``PlasticParams.H`` is aggregated (mean, across every
    connection whose target layer/cell_type touches a given neuron;
    untouched neurons default to the HDP equilibrium ``1.0``) into a
    per-neuron initial RBS (``h_state`` API), applied via
    :meth:`Model.with_hdp_initial_state`. This is stored but inert unless
    the caller separately enables HDP (``cfg.hdp(...)`` before
    :func:`jaxfne.construct`, or via a post-hoc ``RuntimeConfig`` override at
    :func:`jaxfne.simulate` time) — matching ``RuntimeConfig.enable_hdp``'s
    default of ``False``.

    Positions already carry each ``Layer.geometry``: the bridge wrote it
    into the ``Configuration`` (``tfne_geometry`` domains), so
    :func:`jaxfne.construct` sampled it. This then applies only each
    ``Area``'s ``Pose3D`` placement (plane + rotation + translation) onto
    those constructed positions — no re-sampling, one source of truth for
    geometry. A default pose is the identity, bit-exact.

    Connectivity mode is preserved from the tensor: omitted connectivity uses
    the configuration default topology, while explicit connectivity (including
    an explicit empty declaration) compiles only the declared graph.
    """
    family, overrides = _resolve_tensor_emitter_family(tensor, emitter)
    if family == "adex":
        if n_contacts != 16:
            raise ValueError(
                f"n_contacts={n_contacts!r} is not consumed on the adex tensor route "
                "(no laminar probe readout there); leave the default"
            )
        return _construct_adex_tensor_impl(
            tensor,
            seed=seed,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            dtype=dtype,
            adex_decay=adex_decay,
            adex_step_order=adex_step_order,
            param_overrides=overrides,
        )
    if adex_decay != "exp" or adex_step_order != "default":
        raise ValueError(
            f"adex_decay={adex_decay!r}, adex_step_order={adex_step_order!r} are not "
            f"consumed for emitter {family!r} on the current-based route; leave "
            "the defaults"
        )
    cfg = neuronal_tensor_to_configuration(
        tensor,
        seed=seed,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        emitter=emitter,
        dtype=dtype,
        backend=backend,
        jit=jit,
        vmap=vmap,
        n_contacts=n_contacts,
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

    posed, updated_rows = _pose_positions_and_rows(model.params["positions"], rows, tensor)
    model = _replace_model_positions(model, posed, updated_rows)
    model = _apply_velocity_delays_izhi(model, tensor, updated_rows)
    return model


def _adex_harvest_tensor(tensor: NeuronalTensor) -> "tuple[NeuronalTensor, dict[str, str]]":
    """Harvest-only copy of an adex tensor for the current-based population sampler.

    The harvest needs counts and positions per (area, layer, type-fraction)
    only; the tensor's connections are meaningless there (their P-023 sign
    vocabulary does not apply to adex receptors), so they are stripped, and
    cell types the Izhikevich sampler does not name (e.g. ``"SOM"``) are
    aliased to unused Izhikevich names 1:1, preserving every count and
    fraction — and so every RNG draw. Returns the copy plus the
    ``{alias: original}`` map to restore the harvested rows. Cell types
    outside the AdEx vocabulary are refused here (mirroring
    :func:`jaxfne.emitters_adex.make_adex_params`), before the harvest.
    """
    from .emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS
    from .emitters_adex import ADEX_CELL_TYPE_DEFAULTS

    used: list[str] = []
    for area in tensor.areas:
        for layer in area.layers:
            for nt in layer.neuron_types:
                if nt.name not in used:
                    used.append(nt.name)
    unknown_adex = [name for name in used if name not in ADEX_CELL_TYPE_DEFAULTS]
    if unknown_adex:
        raise ValueError(
            f"unknown AdEx cell type {unknown_adex[0]!r}; "
            f"expected one of {sorted(ADEX_CELL_TYPE_DEFAULTS)}"
        )
    known_izhi = set(IZHIKEVICH_CELL_TYPE_DEFAULTS)
    free = [name for name in IZHIKEVICH_CELL_TYPE_DEFAULTS if name not in used]
    alias: dict[str, str] = {}
    for name in used:
        if name not in known_izhi:
            alias[name] = free.pop(0)
    restore = {v: k for k, v in alias.items()}
    harvest_areas = [
        Area(
            name=area.name,
            layers=[
                Layer(
                    name=layer.name,
                    n_neurons=layer.n_neurons,
                    neuron_types=[
                        NeuronType(
                            name=alias.get(nt.name, nt.name),
                            relative_size=nt.relative_size,
                            fraction=nt.fraction,
                            value_tag=nt.value_tag,
                        )
                        for nt in layer.neuron_types
                    ],
                    geometry=layer.geometry,
                )
                for layer in area.layers
            ],
            inter_connections=[],
            pose=area.pose,
        )
        for area in tensor.areas
    ]
    return (
        NeuronalTensor(name=tensor.name, areas=harvest_areas, area_connections=[]),
        restore,
    )


def _construct_adex_tensor_impl(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    dtype: str = "float32",
    adex_decay: str = "exp",
    adex_step_order: str = "default",
    param_overrides: "dict[str, dict[str, float]] | None" = None,
) -> Model:
    """Build a conductance (``adex``) model from a tensor resolving to ``"adex"``.

    Population size and positions are harvested from the current-based bridge
    (with emitter resolution skipped, so no family switch fires), posed with
    the shared helper, then bridged into the adex configuration and built by
    :func:`jaxfne._construct_adex.construct_adex_model` with the harvested
    rows, posed positions and per-type param overrides. Geometry-derived
    delays overwrite each velocity edge afterward.
    """
    from ._construct_adex import construct_adex_model
    from ._construct_population import _neuron_population_from_config

    harvest_tensor, restore = _adex_harvest_tensor(tensor)
    harvest_cfg = neuronal_tensor_to_configuration(
        harvest_tensor,
        seed=seed,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        emitter="izhikevich",
        dtype=dtype,
        _skip_emitter_resolution=True,
    )
    _, harvested_positions, geometry_meta, _ = _neuron_population_from_config(
        harvest_cfg, dtype=dtype
    )
    rows = geometry_meta["neuron_rows"]
    if restore:
        rows = [
            dict(row, cell_type=restore.get(str(row["cell_type"]), str(row["cell_type"])))
            for row in rows
        ]
    posed, updated_rows = _pose_positions_and_rows(harvested_positions, rows, tensor)
    adex_cfg = neuronal_tensor_to_adex_configuration(
        tensor,
        seed=seed,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        dtype=dtype,
        adex_decay=adex_decay,
        adex_step_order=adex_step_order,
        positions=posed,
        rows=updated_rows,
    )
    model = construct_adex_model(
        adex_cfg,
        neuron_rows=updated_rows,
        positions=posed,
        param_overrides=param_overrides,
    )
    return _apply_velocity_delays_adex(model, tensor, updated_rows)

def _pose_positions_and_rows(
    positions: "jax.Array", rows: Sequence[dict], tensor: NeuronalTensor
) -> "tuple[jax.Array, list[dict]]":
    """Apply each Area's Pose3D placement onto constructed positions.

    Owner ruling 2026-09-30: one frame, the Configuration's (mm, each column
    centred at x = k * AREA_X_SPACING_MM). A non-default Pose3D is applied
    about the area's own column origin (that offset removed first), so its
    translation is absolute; a default pose keeps construct's layout.
    Returns ``(positions, updated_rows)`` with the matching ``x``/``y``/``z``
    entries refreshed, so field/LFP/EEG/MEG proxy readouts — which read
    positions from there — see the real layout. Shared by the current-based
    and adex tensor paths.
    """
    from ._construct_population import AREA_X_SPACING_MM

    pose_by_area = {a.name: a.pose for a in tensor.areas}
    area_index = {a.name: k for k, a in enumerate(tensor.areas)}
    position_chunks: list["jax.Array"] = []
    i = 0
    n_rows = len(rows)
    while i < n_rows:
        area_name = rows[i]["area"]
        j = i
        while j < n_rows and rows[j]["area"] == area_name:
            j += 1
        pose = pose_by_area.get(area_name, Pose3D())
        chunk = positions[i:j]
        if pose != Pose3D():
            origin = jnp.asarray([area_index[area_name] * AREA_X_SPACING_MM, 0.0, 0.0], dtype=chunk.dtype)
            chunk = _apply_pose(chunk - origin, pose)
        position_chunks.append(chunk)
        i = j
    posed = jnp.concatenate(position_chunks, axis=0) if position_chunks else jnp.zeros((0, 3))
    updated_rows = [
        dict(
            row, x=float(posed[idx, 0]), y=float(posed[idx, 1]), z=float(posed[idx, 2])
        )
        for idx, row in enumerate(rows)
    ]
    return posed, updated_rows


def _replace_model_positions(model: Model, positions: "jax.Array", updated_rows: list[dict]) -> Model:
    """Overwrite a constructed model's positions and matching row metadata."""
    new_static = dict(model.static)
    new_static["neuron_metadata"] = updated_rows
    if "geometry" in new_static:
        new_static["geometry"] = dict(new_static["geometry"], neuron_rows=updated_rows)
    new_params = dict(model.params)
    new_params["positions"] = positions
    return replace(model, params=new_params, static=new_static)


def _apply_velocity_delays_izhi(model: Model, tensor: NeuronalTensor, rows: Sequence[dict]) -> Model:
    """Overwrite per-edge ``delay_steps`` for velocity connections (current route).

    The rule-level compiler only fans out one constant delay per rule, so
    geometry-derived delays are applied post-construction, per edge, from
    the posed positions. Connections without a velocity keep their
    rule-compiled delays. Rules compiled without a delay use compact
    (``uniform``/``uniform_zero``) storage with no per-edge array, so the
    array is materialized from the uniform value first.
    """
    import numpy as np

    edges = model.params.get("edge_list")
    if edges is None:
        return model
    dt_ms = float(model.cfg.metadata.get("dt_ms", 0.1))
    from .connectivity import delay_steps_from_ms

    positions = np.asarray(model.params["positions"], dtype=np.float64)
    velocity = _velocity_edge_delays(
        tensor, rows, positions, np.asarray(edges.pre), np.asarray(edges.post)
    )
    if not velocity:
        return model
    n_edges = int(np.asarray(edges.pre).reshape(-1).shape[0])
    if edges.delay_storage == "per_edge":
        steps = np.asarray(edges.delay_steps, dtype=np.int64)
    else:
        steps = np.full(n_edges, int(edges.uniform_delay_steps), dtype=np.int64)
    for _, _, delays, _ in velocity:
        for edge_idx, ms in delays.items():
            steps[edge_idx] = delay_steps_from_ms(ms, dt_ms)
    import dataclasses as _dc

    return replace(
        model,
        params=dict(
            model.params,
            edge_list=_dc.replace(
                edges,
                delay_steps=jnp.asarray(steps, dtype=jnp.int32),
                delay_storage="per_edge",
            ),
        ),
    )


def _apply_velocity_delays_adex(model: Model, tensor: NeuronalTensor, rows: Sequence[dict]) -> Model:
    """Overwrite per-edge ``delay_ms`` for velocity connections (adex route).

    Same post-construction per-edge overwrite as
    :func:`_apply_velocity_delays_izhi`, into ``static["adex"]["delay_ms"]``;
    the kernel realizes steps per edge at simulate time.
    """
    import numpy as np

    st = model.static.get("adex")
    if st is None:
        return model
    positions = np.asarray(model.params["positions"], dtype=np.float64)
    velocity = _velocity_edge_delays(
        tensor, rows, positions, np.asarray(st["pre"]), np.asarray(st["post"])
    )
    if not velocity:
        return model
    delay_ms = np.asarray(st["delay_ms"], dtype=np.float64).reshape(-1).copy()
    for _, _, delays, _ in velocity:
        for edge_idx, ms in delays.items():
            delay_ms[edge_idx] = ms
    new_static = dict(model.static)
    new_static["adex"] = dict(st, delay_ms=delay_ms)
    return replace(model, static=new_static)


def _connection_edge_weight(conn: "InterConnection | AreaConnection", total_n: int) -> float:
    """Native edge magnitude for one InterConnection/AreaConnection.

    ``w_mech`` (plastic gain) is scaled by ``g_mech`` (static conductance for
    this connection's mechanism, default 1.0 if undeclared) and normalized by
    ``1/sqrt(total_n)``, matching the native-weight convention used elsewhere
    in ``core.py`` (e.g. ``_apply_connectivity``'s ``base_gain * rnd /
    sqrt(n)``) so bridged edges sit on the same magnitude scale as the rest
    of the network instead of dominating or vanishing relative to it.
    """
    g_scale = conn.static.g_mech.get(conn.mechanism, 1.0)
    return abs(float(conn.plastic.w_mech) * float(g_scale)) / math.sqrt(max(total_n, 1))


def _check_connection_mechanism_sign(
    conn: "InterConnection | AreaConnection", label: str
) -> str:
    """P-023 sign vocabulary: resolve a connection's sign from its mechanism name.

    Excitatory ``{AMPA, NMDA}`` requires an E source; inhibitory ``{GABA_A,
    GABA_B}`` requires a non-E source; any other name is refused as unknown,
    except ``monotonic_cable_synapse`` (the AreaConnection default), whose
    sign follows the source cell type as before. Returns the resolved sign.
    """
    mech = conn.mechanism
    if mech in EXCITATORY_MECHANISMS:
        sign = "excitatory"
    elif mech in INHIBITORY_MECHANISMS:
        sign = "inhibitory"
    elif mech == DEFAULT_AREA_CONNECTION_MECHANISM:
        return "excitatory" if conn.source_neuron_type == "E" else "inhibitory"
    else:
        raise ValueError(
            f"P-023: {label} uses unknown mechanism {mech!r}; known mechanisms "
            "are {AMPA, NMDA} (excitatory) and {GABA_A, GABA_B} (inhibitory)"
        )
    src = conn.source_neuron_type
    if sign == "inhibitory" and src == "E":
        raise ValueError(
            f"P-023: {label}: E source with inhibitory mechanism {mech!r}; "
            "use an excitatory mechanism (AMPA, NMDA) for E sources"
        )
    if sign == "excitatory" and src != "E":
        raise ValueError(
            f"P-023: {label}: non-E source {src!r} with excitatory mechanism "
            f"{mech!r}; use an inhibitory mechanism (GABA_A, GABA_B) for non-E sources"
        )
    return sign


def _resolve_tensor_emitter_family(
    tensor: NeuronalTensor, runtime_emitter: str
) -> "tuple[str, dict[str, dict[str, float]]]":
    """Resolve which emitter family builds ``tensor`` and its per-type params.

    Every :class:`NeuronType` declares ``emitter`` plus optional
    ``emitter_params`` (validated against :mod:`jaxfne.emitter_registry`;
    unknown names and unconsumed keys are refused). When no type declares,
    the runtime emitter applies as before. When any type declares, every
    type must declare, the family must be uniform, and the runtime emitter
    must agree; a registered-but-unroutable family is refused at bridge
    time. Returns ``(family, overrides)`` where ``overrides`` maps a neuron
    type name to its declared params (empty unless the family consumes
    per-type params); conflicting params for one type name across layers
    are refused.
    """
    declared: list[tuple[str, str, "dict | None"]] = []
    undeclared: list[str] = []
    for area in tensor.areas:
        for layer in area.layers:
            for nt in layer.neuron_types:
                key = f"{area.name}.{layer.name}.{nt.name}"
                if nt.emitter is None:
                    undeclared.append(key)
                else:
                    params = check_emitter_params(nt.emitter, nt.emitter_params)
                    declared.append((key, nt.emitter, params))
    if not declared:
        family = runtime_emitter
        if not is_registered_emitter(family):
            raise ValueError(
                f"Unknown emitter: {family}. Choose from: "
                f"{sorted(ROUTABLE_TENSOR_EMITTER_FAMILIES)}"
            )
        if family not in ROUTABLE_TENSOR_EMITTER_FAMILIES:
            raise ValueError(
                f"emitter {family!r} is registered but not routed by the "
                "NeuronalTensor bridge; routable today: "
                f"{sorted(ROUTABLE_TENSOR_EMITTER_FAMILIES)}"
            )
        return family, {}
    if undeclared:
        raise ValueError(
            "partial per-NeuronType emitter declaration is refused: these "
            f"types declare no emitter while others do: {undeclared}; declare "
            "an emitter on every NeuronType or on none"
        )
    families = {emitter for _, emitter, _ in declared}
    if len(families) > 1:
        by_family: dict[str, list[str]] = {}
        for key, emitter, _ in declared:
            by_family.setdefault(emitter, []).append(key)
        detail = "; ".join(f"{fam}: {keys}" for fam, keys in sorted(by_family.items()))
        raise ValueError(
            "heterogeneous per-NeuronType emitter declaration is refused "
            f"(one family per tensor): {detail}"
        )
    family = next(iter(families))
    if family not in ROUTABLE_TENSOR_EMITTER_FAMILIES:
        raise ValueError(
            f"emitter {family!r} is registered but not routed by the "
            "NeuronalTensor bridge; routable today: "
            f"{sorted(ROUTABLE_TENSOR_EMITTER_FAMILIES)}"
        )
    if runtime_emitter != family:
        raise ValueError(
            f"tensor declares emitter {family!r} on every NeuronType but the "
            f"runtime asks for {runtime_emitter!r}; pass "
            f"RuntimeConfiguration(emitter={family!r}) or leave every "
            "NeuronType undeclared"
        )
    overrides: dict[str, dict[str, float]] = {}
    for key, _, params in declared:
        typename = key.rsplit(".", 1)[-1]
        if params:
            if typename in overrides and overrides[typename] != params:
                raise ValueError(
                    f"NeuronType {typename!r} declares conflicting emitter_params "
                    f"across layers ({overrides[typename]!r} vs {params!r}); "
                    "one param set per type name"
                )
            overrides.setdefault(typename, params)
    return family, overrides


def _checked_tau_ms(conn: "InterConnection | AreaConnection", label: str) -> float:
    """Shared P-023 time-constant check: ``static.dT_ms`` required, real, finite, > 0.

    ``None`` is refused, never defaulted, and so is any other
    non-finite-non-positive value: 0, negative, NaN, inf, bools and strings
    (even numeric ones — no silent coercion), matching the genome-side
    checks; only a real number with ``float(tau)`` finite and ``> 0`` wires.
    """
    if conn.static.dT_ms is None:
        raise ValueError(
            f"P-023: {label} declares no time constant (StaticParams.dT_ms "
            "is None); time constants are required, never defaulted"
        )
    raw_tau = conn.static.dT_ms
    if isinstance(raw_tau, bool) or isinstance(raw_tau, str):
        raise ValueError(
            f"P-023: {label} declares inadmissible time constant "
            f"dT_ms={raw_tau!r}; time constants must be a finite number > 0 in ms"
        )
    try:
        tau_ms = float(raw_tau)
    except (TypeError, ValueError):
        raise ValueError(
            f"P-023: {label} declares non-numeric time constant "
            f"dT_ms={raw_tau!r}; time constants must be a finite number > 0 in ms"
        )
    if not (math.isfinite(tau_ms) and tau_ms > 0.0):
        raise ValueError(
            f"P-023: {label} declares inadmissible time constant "
            f"dT_ms={raw_tau!r}; time constants must be finite and > 0 in ms"
        )
    return tau_ms


def _connection_label(
    conn: "InterConnection | AreaConnection", source_area: str, target_area: str
) -> str:
    return (
        f"{type(conn).__name__} {source_area}.{conn.source_layer}."
        f"{conn.source_neuron_type}->{target_area}.{conn.target_layer}."
        f"{conn.target_neuron_type} ({conn.mechanism!r})"
    )


def _wire_connection(
    cfg: Configuration,
    conn: "InterConnection | AreaConnection",
    *,
    rule_name: str,
    source_area: str,
    target_area: str,
    total_n: int,
    declared_mechanisms: dict[tuple[str, float], str],
    _tfne_resolved: bool = False,
) -> Configuration:
    """Declare one real edge rule (mechanism + connection) for an Inter/AreaConnection.

    Uses the generic, selector-based :meth:`Configuration.connections` +
    :meth:`Configuration.mechanisms` hooks (source/target by area/layer/
    cell_type) rather than the same-area-masked dense ``W`` in
    ``_apply_connectivity`` — selectors carry no same-area restriction,
    so this is what actually couples two different areas (or two specific
    layer x cell-type populations within one area) into the simulated
    dynamics.     ``mechanism`` -> ``tau_ms`` comes from ``static.dT_ms``
    (P-023: required — ``None`` is refused, never defaulted, and so is any
    other non-finite-non-positive value: 0, negative, NaN, inf, bools and
    strings (even numeric ones — no silent coercion), matching the genome-side
    checks; only a real number with ``float(tau)`` finite and ``> 0`` wires);
    mechanisms are deduplicated by (name, dT_ms) so repeated connections
    sharing both don't raise on ``Configuration``'s duplicate-name guard.
    A declared ``InterConnection.delay_ms`` is forwarded into the rule;
    ``None`` leaves the rule without a delay (current behaviour).
    Sign follows the P-023 mechanism vocabulary
    (:func:`_check_connection_mechanism_sign`), except for TFNE-derived
    tensors (``_tfne_resolved``: TFNE carries its own fail-closed mechanism
    vocabulary plus weight-polarity sign, so the declaration vocabulary is
    not re-applied there; owner ruling 2026-09-30, pinned by the TFNE
    custom-mechanism, direct-coupling and polarity tests).
    """
    label = _connection_label(conn, source_area, target_area)
    tau_ms = _checked_tau_ms(conn, label)
    mech_key = (conn.mechanism, tau_ms)
    mech_name = declared_mechanisms.get(mech_key)
    if mech_name is None:
        mech_name = f"{conn.mechanism}__dt{tau_ms:g}__{len(declared_mechanisms)}"
        mech_params: dict[str, object] = {"tau_ms": tau_ms}
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

    if _tfne_resolved:
        sign = "excitatory" if conn.source_neuron_type == "E" else "inhibitory"
    else:
        sign = _check_connection_mechanism_sign(conn, label)
    wire_kw: dict[str, object] = {}
    conn_delay = getattr(conn, "delay_ms", None)
    if conn_delay is not None:
        # Both InterConnection and AreaConnection carry an optional
        # configured delay_ms for inspection; validated again here.
        wire_kw["delay_ms"] = float(conn_delay)
    cfg = cfg.connections(
        name=rule_name,
        source={
            "area": source_area,
            "layer": conn.source_layer,
            "cell_type": conn.source_neuron_type,
        },
        target={
            "area": target_area,
            "layer": conn.target_layer,
            "cell_type": conn.target_neuron_type,
        },
        probability=(1.0 if getattr(conn, "probability", None) is None
                     else float(conn.probability)),
        weight=_connection_edge_weight(conn, total_n),
        sign=sign,
        mechanism=mech_name,
        **wire_kw,  # type: ignore[arg-type]
    )
    return cfg


def _wire_adex_connection(
    cfg: Configuration,
    conn: "InterConnection | AreaConnection",
    *,
    rule_name: str,
    source_area: str,
    target_area: str,
    total_n: int,
    declared_mechanisms: dict[tuple[str, float, float], str],
    velocity_delay_ms: "float | None" = None,
) -> Configuration:
    """Declare one adex-route edge rule for an Inter/AreaConnection.

    Same selectors as :func:`_wire_connection`, but the rule is written for
    the conductance route (:func:`jaxfne._construct_adex.construct_adex_model`):
    the mechanism is declared with ``kind="receptor"`` (tau from the shared
    P-023 check, reversal mandatory — refused when missing, since the kernel
    has no default), no ``sign`` is set (conductance semantics: polarity
    rides on the weight, positive from E sources, negative otherwise), and
    the weight is in nS. ``delay_ms`` is the declared constant, or
    ``velocity_delay_ms`` (the median over the rule's selected pairs when the
    connection carries ``conduction_velocity_m_per_s``); one of the two is
    required, since the adex compiler refuses a rule without a delay. The
    P-023 mechanism-name vocabulary is not applied here: adex receptors are
    user-named (``"ampa"``, ``"gaba"``, ...), resolved against the declared
    set by name.
    """
    label = _connection_label(conn, source_area, target_area)
    tau_ms = _checked_tau_ms(conn, label)
    reversal = conn.static.reversal_potentials_mV.get(conn.mechanism)
    if reversal is None:
        raise ValueError(
            f"P-023: {label} declares no reversal potential for mechanism "
            f"{conn.mechanism!r}; the adex route needs "
            "static.reversal_potentials_mV[mechanism] on every connection"
        )
    try:
        reversal_mV = float(reversal)
    except (TypeError, ValueError):
        raise ValueError(
            f"P-023: {label} declares non-numeric reversal potential "
            f"{reversal!r} for mechanism {conn.mechanism!r}"
        )
    if not math.isfinite(reversal_mV):
        raise ValueError(
            f"P-023: {label} declares non-finite reversal potential "
            f"{reversal!r} for mechanism {conn.mechanism!r}"
        )
    mech_key = (conn.mechanism, tau_ms, reversal_mV)
    mech_name = declared_mechanisms.get(mech_key)
    if mech_name is None:
        mech_name = f"{conn.mechanism}__dt{tau_ms:g}__rev{reversal_mV:g}__{len(declared_mechanisms)}"
        cfg = cfg.mechanisms(
            name=mech_name,
            kind="receptor",
            params={"tau_ms": tau_ms, "reversal_mV": reversal_mV},
        )
        declared_mechanisms[mech_key] = mech_name
    polarity = 1.0 if conn.source_neuron_type == "E" else -1.0
    weight_nS = polarity * _connection_edge_weight(conn, total_n)
    if conn.delay_ms is not None:
        rule_delay_ms = float(conn.delay_ms)
    elif velocity_delay_ms is not None:
        rule_delay_ms = float(velocity_delay_ms)
    else:
        raise ValueError(
            f"{label} declares neither delay_ms nor "
            "conduction_velocity_m_per_s; the adex route needs one delay per rule"
        )
    return cfg.connections(
        name=rule_name,
        source={
            "area": source_area,
            "layer": conn.source_layer,
            "cell_type": conn.source_neuron_type,
        },
        target={
            "area": target_area,
            "layer": conn.target_layer,
            "cell_type": conn.target_neuron_type,
        },
        probability=(1.0 if getattr(conn, "probability", None) is None
                     else float(conn.probability)),
        weight=weight_nS,
        mechanism=mech_name,
        delay_ms=rule_delay_ms,
    )


def _velocity_edge_delays(
    tensor: NeuronalTensor,
    rows: Sequence[dict],
    positions: Any,
    edge_pre: Any,
    edge_post: Any,
) -> "list[tuple[str, Any, dict[int, float], float]]":
    """Per-edge geometry delays for every velocity connection in ``tensor``.

    ``rows``/``positions`` are aligned (``rows[i]`` sits at ``positions[i]``)
    with ``neuron_id``/``area``/``layer``/``cell_type`` keys; ``edge_pre`` /
    ``edge_post`` hold compiled edge endpoints as neuron ids. For each
    connection carrying ``conduction_velocity_m_per_s``, every compiled edge
    whose endpoints match its (area, layer, type) selectors gets
    ``delay_ms = distance_mm * mm_per_unit / velocity`` from the constructed
    positions. An edge selected by more than one velocity connection is
    refused (overlapping velocity rules); a velocity connection selecting no
    compiled edge is refused (its representative delay would be undefined).
    Returns ``[(label, conn, {edge_index: delay_ms}, median_ms)]``; the median
    feeds the adex rule's representative delay, the per-edge map the
    post-construction overwrite on both families.
    """
    import numpy as np

    pos_index = {int(r["neuron_id"]): i for i, r in enumerate(rows)}
    n = len(rows)
    areas = np.array([str(r["area"]) for r in rows], dtype=object)
    layers = np.array([str(r["layer"]) for r in rows], dtype=object)
    ctypes = np.array([str(r["cell_type"]) for r in rows], dtype=object)
    xyz = np.asarray(positions, dtype=np.float64).reshape(n, 3)
    pre_ids = np.asarray(edge_pre, dtype=np.int64).reshape(-1)
    post_ids = np.asarray(edge_post, dtype=np.int64).reshape(-1)
    missing = [int(i) for i in list(pre_ids) + list(post_ids) if int(i) not in pos_index]
    if missing:
        raise ValueError(
            "geometry-delay matcher sees edge endpoints outside the neuron table: "
            f"{sorted(set(missing))[:8]}"
        )
    pre_idx = np.array([pos_index[int(i)] for i in pre_ids], dtype=np.int64)
    post_idx = np.array([pos_index[int(i)] for i in post_ids], dtype=np.int64)

    def _iter_conns():
        for area in tensor.areas:
            for ic in area.inter_connections:
                yield _connection_label(ic, area.name, area.name), ic, area.name, area.name
        for ac in tensor.area_connections:
            yield (
                _connection_label(ac, ac.source_area, ac.target_area),
                ac,
                ac.source_area,
                ac.target_area,
            )

    out: "list[tuple[str, Any, dict[int, float], float]]" = []
    claimed = np.zeros(pre_ids.shape[0], dtype=bool)
    claimed_label = np.array([""] * pre_ids.shape[0], dtype=object)
    for label, conn, source_area, target_area in _iter_conns():
        velocity = conn.conduction_velocity_m_per_s
        if velocity is None:
            continue
        match = (
            (areas[pre_idx] == source_area)
            & (layers[pre_idx] == conn.source_layer)
            & (ctypes[pre_idx] == conn.source_neuron_type)
            & (areas[post_idx] == target_area)
            & (layers[post_idx] == conn.target_layer)
            & (ctypes[post_idx] == conn.target_neuron_type)
        )
        idx = np.nonzero(match)[0]
        if idx.size == 0:
            raise ValueError(
                f"{label} declares conduction_velocity_m_per_s="
                f"{float(velocity)!r} but selects no compiled edge; a velocity "
                "connection needs at least one realized pair"
            )
        if claimed[idx].any():
            first = int(idx[np.nonzero(claimed[idx])[0][0]])
            raise ValueError(
                f"{label} with conduction_velocity_m_per_s selects edge {first}, "
                f"already selected by {claimed_label[first]!r}; overlapping "
                "velocity rules are refused"
            )
        dist_mm = np.linalg.norm(xyz[pre_idx[idx]] - xyz[post_idx[idx]], axis=1)
        ms = dist_mm * _MM_PER_LENGTH_UNIT[conn.length_unit] / float(velocity)
        delays = {int(e): float(x) for e, x in zip(idx.tolist(), ms.tolist())}
        claimed[idx] = True
        claimed_label[idx] = label
        out.append((label, conn, delays, float(np.median(ms))))
    return out


def _tensor_identity_digest(tensor: NeuronalTensor) -> str:
    """Deterministic structural digest of a NeuronalTensor.

    Includes areas, layers, neuron types, geometry, pose, and connections —
    the persisted content of the tensor (the same surface as
    ``save_neuronal_tensor``, which intentionally strips in-memory
    ``provenance``). Runtime knobs (seed/duration/dt/emitter) are excluded —
    they belong to ``RuntimeConfiguration``, not the tensor. Provenance
    (genome rules hash, schema version, development seed, phenotype hash) is
    deliberately excluded so receipt identity is stable across the documented
    save/load roundtrip; provenance identity is carried separately on the
    in-memory tensor and in the JDNA genome persistence owner.

    Written into config metadata so receipt identity distinguishes tensors
    that the Configuration bridge would otherwise collapse together.
    """
    import json as _json

    try:
        payload = dict(tensor.to_dict())
    except Exception:  # defensive: never block construction on hashing
        return ""
    payload.pop("provenance", None)
    blob = _json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda o: o if isinstance(o, (str, int, float, bool)) or o is None else repr(o),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def neuronal_tensor_to_configuration(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
    dtype: str = "float32",
    backend: "str | None" = None,
    jit: "bool | str | None" = None,
    vmap: "bool | str | None" = None,
    n_contacts: int = 16,
    _skip_emitter_resolution: bool = False,
) -> Configuration:
    """Bridge a :class:`NeuronalTensor` into the existing construct/simulate pipeline.

    Per-area/per-layer cell-type fractions ARE preserved, via
    :meth:`Configuration.area_layer_cell_types` — each layer's declared
    ``NeuronType`` list is split into even fractions (the tensor stores
    membership, not population fractions, so an even split is the most
    neutral reading) and recorded per area/layer, not flattened globally.

    Per-area per-layer NEURON COUNT is also preserved (fixed 2026-07-04),
    via :meth:`Configuration.population` (records
    ``metadata["area_layer_count_frac"][area]``, resolved by
    ``core._area_layer_count_frac`` ahead of jaxfne's hardcoded default
    layer-thickness split). Each ``Layer.n_neurons`` is honored exactly —
    previously this function called the plain :meth:`Configuration.column`
    (total count only), so per-layer sizes were silently dropped in favor
    of that default split regardless of what a tensor declared.

    Connectivity semantics: when the tensor omits connectivity, the existing
    configuration default topology is retained. When connectivity is explicit,
    including an explicit empty declaration, the compiled graph contains only
    the tensor-declared edges. Every ``InterConnection`` (within an
    area) and ``AreaConnection`` (between areas) is compiled into a REAL edge
    rule via :meth:`Configuration.connections` + :meth:`Configuration.mechanisms`
    (selector-based: ``area``/``layer``/``cell_type``), which carries no
    same-area restriction — this is what actually couples two areas (jaxfne's
    own default recurrent ``W`` in ``_apply_connectivity`` is
    same-area-masked and would leave bridged areas dynamically isolated).
    ``mechanism`` resolves to a real per-edge ``tau_ms`` (from
    ``static.dT_ms``, P-023-required); edge magnitude is ``w_mech * g_mech / sqrt(total_n)``
    (see :func:`_connection_edge_weight`); sign follows the P-023 mechanism
    vocabulary (excitatory {AMPA, NMDA} from E sources, inhibitory {GABA_A,
    GABA_B} from non-E sources; ``monotonic_cable_synapse`` exempt, sign from
    the source type). TFNE-derived tensors skip the declaration vocabulary:
    TFNE resolves kinetics fail-closed in its own compiler and carries sign
    on weight polarity. Genuine TFNE derivation is recognized only through
    the module-private minted registry (populated by
    ``jaxfne.tfne.to_neuronal_tensor`` for the exact objects it constructs);
    a caller-supplied ``provenance`` dict — including one carrying
    ``tfne_digest`` — never qualifies, and save/load, copies and merges lose
    the status.
    ``probability=1.0`` (full bipartite between the selected populations) —
    the tensor model declares connection membership, not a separate density
    parameter, so full density between the declared layer x cell-type pair
    is the most neutral reading, mirroring the even-split reading used for
    cell-type fractions above.

    Per-layer GEOMETRY is also preserved (fixed P-022), via the same
    fractional-domain channel the TFNE ``to_configuration`` path writes
    (``metadata["tfne_geometry"]["domains"]``, relative,
    ``value_tag="relative"``): each ``Layer``'s ``Geometry3D`` ranges are
    column-relative (z a fraction of the column depth, x and y of its width;
    owner ruling 2026-09-30) and become per-(area, layer) domains in the
    construction stage's block frame. A full ``(0.0, 1.0)`` axis is
    unconstrained and omitted, so default-geometry tensors take the
    historical path unchanged. Only ``distribution="uniform_random"`` is
    supported (anything else is refused); a degenerate point range keeps
    the documented 2D/1D collapse at that column fraction.

    Known fidelity gaps still open (not yet wired to ``Configuration``):
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

    if _skip_emitter_resolution:
        family = emitter
    else:
        family, _ = _resolve_tensor_emitter_family(tensor, emitter)
        if family == "adex":
            raise ValueError(
                "tensor resolves to emitter 'adex'; the conductance route is "
                "built by neuronal_tensor_to_adex_configuration, not this "
                "current-based bridge"
            )

    connectivity_mode: ConnectivityMode
    if getattr(tensor, "_connectivity_mode_declared", False):
        connectivity_mode = tensor.connectivity_mode
    else:
        connectivity_mode = (
            "explicit"
            if any(area.connectivity_mode == "explicit" for area in tensor.areas)
            else "unspecified"
        )
    _runtime_kw: dict[str, Any] = {
        "seed": seed,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "dtype": dtype,
    }
    if backend is not None:
        _runtime_kw["backend"] = backend
    if jit is not None:
        _runtime_kw["jit"] = jit
    if vmap is not None:
        _runtime_kw["vmap"] = vmap
    cfg = Configuration().runtime(**_runtime_kw)
    # Tensor identity: a deterministic structural digest (areas, layers,
    # neuron types, geometry, connections, provenance) written into config
    # metadata so run receipts/config hashes are distinct for tensors with
    # scientifically different structure/geometry — the Configuration bridge
    # alone can collapse distinct tensors to equal metadata.
    cfg = cfg.update_metadata(tensor_identity=_tensor_identity_digest(tensor))
    cfg = cfg.update_metadata(connectivity_mode=connectivity_mode)
    fallback_weight: dict[str, float] = {}
    area_n_by_name: dict[str, int] = {}
    for area in tensor.areas:
        layer_names = [layer.name for layer in area.layers] or ["single"]
        area_n = sum(layer.n_neurons for layer in area.layers) or 1
        area_n_by_name[area.name] = area_n
        if area.layers:
            # Fixed 2026-07-04: previously called cfg.column(...) directly,
            # which only records the total area size -- per-layer neuron
            # counts were silently dropped in favor of jaxfne's hardcoded
            # default layer-thickness split (core._SUITE2_LAYER_FRACTIONS).
            # cfg.population(...) is a strict superset of .column() (it calls
            # .column() internally with the same args) that ALSO records each
            # layer's declared share of the area total under
            # metadata["area_layer_count_frac"][area.name], which
            # core._area_layer_count_frac resolves ahead of the thickness
            # default -- so a Layer's declared n_neurons is now honored.
            cfg = cfg.population(
                area_n,
                neurons={layer.name: (layer.n_neurons or 1) for layer in area.layers},
                name=area.name,
                layers=layer_names,
            )
        else:
            cfg = cfg.column(area.name, layers=layer_names, n=area_n)

        layer_cell_types: dict[str, dict[str, float]] = {}
        for layer in area.layers:
            neuron_types = list(layer.neuron_types) or [NeuronType.make("E")]
            type_names = [nt.name for nt in neuron_types]
            if all(nt.fraction is not None for nt in neuron_types):
                # Every NeuronType in this layer declares its own population
                # fraction -- use those (normalized) instead of the even
                # split, so a layer's declared E:I composition (e.g. the
                # canonical column's deep-E/superficial-I gradient) survives
                # the bridge instead of being flattened to 1/len(types).
                raw = {nt.name: float(nt.fraction) for nt in neuron_types}
                total = sum(raw.values()) or 1.0
                fracs = {name: v / total for name, v in raw.items()}
            else:
                even = 1.0 / len(type_names)
                fracs = {name: even for name in type_names}
            layer_cell_types[layer.name] = fracs
            for name, frac in fracs.items():
                fallback_weight[name] = (
                    fallback_weight.get(name, 0.0) + float(layer.n_neurons) * frac
                )
        if layer_cell_types:
            cfg = cfg.area_layer_cell_types(area.name, layer_cell_types)

    total_weight = sum(fallback_weight.values()) or 1.0
    cfg = cfg.cell_types({name: weight / total_weight for name, weight in fallback_weight.items()})

    geometry_domains = _tensor_geometry_domains(tensor, dict(cfg.metadata))
    if geometry_domains:
        # `declared` keeps the column-relative ranges as written; `domains`
        # are the layer-block fractions construction samples (z may leave [0, 1]).
        declared = {
            area.name: {
                layer.name: {
                    "x": list(layer.geometry.x_range),
                    "y": list(layer.geometry.y_range),
                    "z": list(layer.geometry.z_range),
                }
                for layer in area.layers
                if layer.name in geometry_domains.get(area.name, {})
            }
            for area in tensor.areas
            if area.name in geometry_domains
        }
        cfg = cfg.update_metadata(
            tfne_geometry={
                "value_tag": "relative",
                "frame": "column",
                "declared": declared,
                "domains": geometry_domains,
            }
        )

    total_n = sum(area_n_by_name.values()) or 1
    declared_mechanisms: dict[tuple[str, float], str] = {}
    # P-023 carve-out: only tensors minted by jaxfne.tfne.to_neuronal_tensor
    # (module-private weak registry) skip the declaration vocabulary.
    # Provenance is deliberately NOT consulted: it is a free constructor
    # kwarg, so a forged {"tfne_digest": ...} must not qualify.
    tfne_resolved = _is_tfne_minted(tensor)
    for area in tensor.areas:
        for idx, ic in enumerate(area.inter_connections):
            cfg = _wire_connection(
                cfg,
                ic,
                rule_name=f"interconn_{area.name}_{idx}",
                source_area=area.name,
                target_area=area.name,
                total_n=total_n,
                declared_mechanisms=declared_mechanisms,
                _tfne_resolved=tfne_resolved,
            )
    for idx, ac in enumerate(tensor.area_connections):
        cfg = _wire_connection(
            cfg,
            ac,
            rule_name=f"areaconn_{idx}",
            source_area=ac.source_area,
            target_area=ac.target_area,
            total_n=total_n,
            declared_mechanisms=declared_mechanisms,
            _tfne_resolved=tfne_resolved,
        )

    if family == "izhikevich":
        cfg = cfg.set_emitter("izhikevich", "cortical_eig")
    elif family == "lif":
        cfg = cfg.set_emitter("lif")
    elif family == "glif":
        cfg = cfg.set_emitter("glif")
    else:  # unreachable: the resolver admits routable families only, minus adex above
        raise ValueError(f"Unknown emitter: {family}. Choose from: izhikevich, lif, glif")

    cfg = cfg.probes(["spikes", "V_m"], n_contacts=n_contacts)
    cfg = cfg.field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    return cfg


def neuronal_tensor_to_adex_configuration(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    dtype: str = "float32",
    adex_decay: str = "exp",
    adex_step_order: str = "default",
    positions: Any = None,
    rows: "Sequence[dict] | None" = None,
) -> Configuration:
    """Bridge a :class:`NeuronalTensor` into the conductance (``adex``) route.

    The tensor must resolve to emitter ``"adex"`` (every ``NeuronType``
    declares ``emitter="adex"``, or none declares and the caller routes here
    explicitly). Population size and per-type counts come from the harvested
    ``rows`` (``_neuron_population_from_config`` on the current-based
    bridge, passed in by :func:`jaxfne.construct`, or harvested here when
    omitted); each connection becomes one receptor-rule via
    :func:`_wire_adex_connection` (tau from ``static.dT_ms``, reversal
    mandatory, no sign, weight polarity from the source type, delay constant
    or velocity-median over the rule's selected pairs, computed from
    ``positions``). The returned config carries only fields the adex route
    consumes (no probes, fields, geometry domains, or tensor-identity
    metadata — each is refused there), so :func:`jaxfne.construct` routes it
    to ``construct_adex_model`` unchanged.
    """
    from .emitters_adex import _DECAYS as _ADEX_DECAYS
    from .emitters_adex import _STEP_ORDERS as _ADEX_STEP_ORDERS

    if not tensor.areas:
        raise ValueError("NeuronalTensor must declare at least one area to bridge")
    family, _ = _resolve_tensor_emitter_family(tensor, "adex")
    if family != "adex":  # unreachable: agreement forces "adex" here
        raise ValueError(f"tensor resolves to emitter {family!r}, not 'adex'")
    if adex_decay not in _ADEX_DECAYS:
        raise ValueError(
            f"adex_decay must be one of {list(_ADEX_DECAYS)}; got {adex_decay!r}"
        )
    if adex_step_order not in _ADEX_STEP_ORDERS:
        raise ValueError(
            f"adex_step_order must be one of {list(_ADEX_STEP_ORDERS)}; got {adex_step_order!r}"
        )
    if (positions is None) != (rows is None):
        raise ValueError(
            "positions and rows come as a harvested pair; pass both or neither"
        )
    if rows is None:
        harvest_cfg = neuronal_tensor_to_configuration(
            tensor,
            seed=seed,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            emitter="izhikevich",
            dtype=dtype,
            _skip_emitter_resolution=True,
        )
        from ._construct_population import _neuron_population_from_config

        _, positions, geometry_meta, _ = _neuron_population_from_config(
            harvest_cfg, dtype=dtype
        )
        rows = geometry_meta["neuron_rows"]
    rows = list(rows)
    if not rows:
        raise ValueError("adex tensor bridge harvested no neurons")
    total_n = len(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["cell_type"])] = counts.get(str(row["cell_type"]), 0) + 1
    fractions = {name: count / total_n for name, count in counts.items()}

    cfg = Configuration().runtime(
        seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, dtype=dtype
    )
    cfg = cfg.network(name=tensor.name, n=total_n, cell_types=fractions)
    cfg = cfg.emitter(family="adex", adex_decay=adex_decay, adex_step_order=adex_step_order)

    velocity_median: dict[int, float] = {}
    velocity_conns = [
        (conn, source_area, target_area)
        for area in tensor.areas
        for conn in area.inter_connections
        for source_area, target_area in [(area.name, area.name)]
    ] + [
        (ac, ac.source_area, ac.target_area) for ac in tensor.area_connections
    ]
    velocity_conns = [(c, s, t) for c, s, t in velocity_conns
                      if c.conduction_velocity_m_per_s is not None]
    if velocity_conns:
        import numpy as _np

        xyz_all = _np.asarray(positions, dtype=_np.float64).reshape(total_n, 3)
        for conn, source_area, target_area in velocity_conns:
            pre_sel = [
                i for i, r in enumerate(rows)
                if str(r["area"]) == source_area
                and str(r["layer"]) == conn.source_layer
                and str(r["cell_type"]) == conn.source_neuron_type
            ]
            post_sel = [
                i for i, r in enumerate(rows)
                if str(r["area"]) == target_area
                and str(r["layer"]) == conn.target_layer
                and str(r["cell_type"]) == conn.target_neuron_type
            ]
            # Selected pairs mirror the compiler (self pairs dropped: the
            # tensor bridge never opts into allow_self_connections).
            dists = [
                float(_np.linalg.norm(xyz_all[a] - xyz_all[b]))
                for a in pre_sel for b in post_sel
                if rows[a]["neuron_id"] != rows[b]["neuron_id"]
            ]
            if not dists:
                raise ValueError(
                    f"{_connection_label(conn, source_area, target_area)} declares "
                    "conduction_velocity_m_per_s="
                    f"{float(conn.conduction_velocity_m_per_s)!r} but selects no "
                    "compiled pair; a velocity connection needs at least one pair"
                )
            dists.sort()
            median_mm = dists[len(dists) // 2] if len(dists) % 2 else (
                0.5 * (dists[len(dists) // 2 - 1] + dists[len(dists) // 2])
            )
            velocity_median[id(conn)] = (
                median_mm * _MM_PER_LENGTH_UNIT[conn.length_unit]
                / float(conn.conduction_velocity_m_per_s)
            )

    declared_mechanisms: dict[tuple[str, float, float], str] = {}
    for area in tensor.areas:
        for idx, ic in enumerate(area.inter_connections):
            cfg = _wire_adex_connection(
                cfg,
                ic,
                rule_name=f"interconn_{area.name}_{idx}",
                source_area=area.name,
                target_area=area.name,
                total_n=total_n,
                declared_mechanisms=declared_mechanisms,
                velocity_delay_ms=velocity_median.get(id(ic)),
            )
    for idx, ac in enumerate(tensor.area_connections):
        cfg = _wire_adex_connection(
            cfg,
            ac,
            rule_name=f"areaconn_{idx}",
            source_area=ac.source_area,
            target_area=ac.target_area,
            total_n=total_n,
            declared_mechanisms=declared_mechanisms,
            velocity_delay_ms=velocity_median.get(id(ac)),
        )
    return cfg
