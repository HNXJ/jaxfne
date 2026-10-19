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
import numbers
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
from .emitters import DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE, _resolved_edge_weight
from .emitter_registry import check_emitter_params, is_registered_emitter
from .pcl_topology import validate_footprint, validate_local_inhibition, validate_weight_sharing
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
#: :mod:`jaxfne.emitter_registry`) but refused at bridge time as unroutable:
#: ``lif``/``glif`` have no kernel in ``construct``/``simulate``, so a tensor
#: resolving to them is refused here rather than failing late inside
#: construct with an explicitly-declared-but-unbuilt emitter. Declared
#: dynamics always equal executed dynamics ((h, y) = f(x, h')).
ROUTABLE_TENSOR_EMITTER_FAMILIES = frozenset({"izhikevich", "lif", "adex", "rate"})

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
    return _tensor_identity_digest(tensor) == stored

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
    # Per-triple baseline drive (None = undeclared = the emitter default for
    # this type). A declared value overrides the drive of every neuron built
    # from this NeuronType — i.e. one (area, layer, type) triple — and is
    # applied at construction after ``drive(baseline_drive_by_cell_type=)``
    # and before ``cell_params()``. Must be a finite real number; NaN, inf,
    # bools and strings are refused, never coerced. On the conductance
    # (adex) route a declared value reads as constant external current (one
    # drive unit = one pA on every neuron of the triple); undeclared
    # triples carry none.
    drive: Optional[float] = None

    def __post_init__(self) -> None:
        if self.emitter is None and self.emitter_params is not None:
            raise ValueError(
                f"NeuronType {self.name!r} declares emitter_params="
                f"{self.emitter_params!r} with emitter=None; emitter_params "
                "are validated against the declared emitter and ignored "
                "without one — declare an emitter or drop emitter_params"
            )
        object.__setattr__(
            self, "drive",
            _check_triple_drive(self.drive, label=f"NeuronType {self.name!r}"),
        )

    @classmethod
    def make(
        cls,
        name: str,
        relative_size: Optional[float] = None,
        fraction: Optional[float] = None,
        value_tag: ValueTag = "relative",
        emitter: Optional[str] = None,
        emitter_params: Optional[dict] = None,
        drive: Optional[float] = None,
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
            drive=drive,
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
    ``docs/theory/doctrine/rbs_rbd_hdp.md`` §1 carve-out ruling): it names the
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

    ``length_unit`` names the unit pairwise distances convert through; tensor
    positions are mm, so only ``"mm"`` is accepted (any other value in
    ``{"um", "mm", "m"}`` is refused rather than silently rescaled, and
    anything outside the set is refused as unknown); ``velocity`` (m/s, None =
    undeclared) must be finite and > 0 when given.
    """
    if length_unit not in LENGTH_UNITS:
        raise ValueError(
            f"{label} length_unit must be one of {sorted(LENGTH_UNITS)}; "
            f"got {length_unit!r}"
        )
    if length_unit != "mm":
        raise ValueError(
            f"{label} length_unit must be 'mm' (tensor positions are mm); "
            f"got {length_unit!r} — relabelling the geometry unit is refused, "
            "declare distances in mm"
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


def _check_connection_spatial_sigma(
    spatial_sigma: "float | None", *, label: str
) -> "float | None":
    """Validate the local sampling width shared by both connection types.

    None = undeclared = the compiler default; a declared value must be a
    finite real number > 0 in mm (bools and strings refused, never coerced).
    """
    if spatial_sigma is None:
        return None
    if isinstance(spatial_sigma, bool) or isinstance(spatial_sigma, str):
        raise ValueError(
            f"{label} spatial_sigma must be a finite number > 0 in mm; "
            f"got {spatial_sigma!r}"
        )
    try:
        sigma = float(spatial_sigma)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} spatial_sigma must be a finite number > 0 in mm; "
            f"got {spatial_sigma!r}"
        )
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(
            f"{label} spatial_sigma must be finite and > 0 in mm; got {spatial_sigma!r}"
        )
    return sigma


def _check_connection_allow_self(allow_self: "bool | None", *, label: str) -> "bool | None":
    """Validate the per-rule self-edge opt-in shared by both connection types.

    None = undeclared = self-edges dropped; a declared value must be a bool
    (fail-closed: only an explicit True keeps ``i == i`` pairs).
    """
    if allow_self is None:
        return None
    if not isinstance(allow_self, bool):
        raise ValueError(
            f"{label} allow_self must be a bool; got {allow_self!r}"
        )
    return allow_self


#: Sampling modes the compiler honours per connection rule (None = undeclared
#: = the ``density`` default).
_CONNECTION_SAMPLINGS = ("density", "bernoulli", "capped_bernoulli", "fixed_indegree")


def _check_connection_sampling(sampling: "str | None", *, label: str) -> "str | None":
    """Validate the compiler sampling mode shared by both connection types.

    None = undeclared = the compiler default (density); a declared value
    must name a mode the compiler honours (anything else, including
    non-strings, refused, never coerced).
    """
    if sampling is None:
        return None
    if not isinstance(sampling, str) or sampling not in _CONNECTION_SAMPLINGS:
        raise ValueError(
            f"{label} sampling must be one of {list(_CONNECTION_SAMPLINGS)}; "
            f"got {sampling!r}"
        )
    return sampling


def _check_connection_max_in_degree(max_in_degree: "int | None", *, label: str) -> "int | None":
    """Validate the localized-sampling fan-in cap shared by both connection types.

    None = undeclared = the compiler's density path; a declared value must
    be a positive integer (bools, non-integral numbers and values < 1 are
    refused, never coerced). When set, the rule compiles through the
    localized (distance-weighted, ``spatial_sigma``) sampler instead of the
    ``probability`` density path, so the two are mutually exclusive.
    """
    if max_in_degree is None:
        return None
    if isinstance(max_in_degree, bool) or not isinstance(max_in_degree, numbers.Integral):
        raise ValueError(
            f"{label} max_in_degree must be a positive integer; "
            f"got {max_in_degree!r}"
        )
    degree = int(max_in_degree)
    if degree < 1:
        raise ValueError(
            f"{label} max_in_degree must be a positive integer; "
            f"got {max_in_degree!r}"
        )
    return degree


def _check_connection_in_degree(in_degree: "int | None", *, label: str) -> "int | None":
    """Validate the exact-fixed-in-degree fan-in count shared by both connection types.

    None = undeclared = the ``probability`` density path; a declared value
    must be a positive integer (bools, non-integral numbers and values < 1
    are refused, never coerced). When set, the rule compiles through the
    exact fixed-indegree sampler with this per-post-neuron count instead of
    the density path, so it is mutually exclusive with a declared
    ``probability``, ``max_in_degree``, ``footprint`` and any sampling
    other than ``"fixed_indegree"``.
    """
    if in_degree is None:
        return None
    if isinstance(in_degree, bool) or not isinstance(in_degree, numbers.Integral):
        raise ValueError(
            f"{label} in_degree must be a positive integer; "
            f"got {in_degree!r}"
        )
    degree = int(in_degree)
    if degree < 1:
        raise ValueError(
            f"{label} in_degree must be a positive integer; "
            f"got {in_degree!r}"
        )
    return degree


def _check_triple_drive(drive: "float | None", *, label: str) -> "float | None":
    """Validate a per-(area, layer, type) drive declaration.

    None = undeclared = the emitter default for the type; a declared value
    must be a finite real number (bools, NaN, inf and strings refused, never
    coerced), mirroring ``Configuration.drive``.
    """
    if drive is None:
        return None
    if isinstance(drive, bool) or not isinstance(drive, numbers.Real):
        raise ValueError(
            f"{label} drive must be a finite real number; got {drive!r}"
        )
    value = float(drive)
    if not math.isfinite(value):
        raise ValueError(
            f"{label} drive must be finite; got {drive!r}"
        )
    return value


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
    # Local sampling width in mm (None = undeclared = the compiler default).
    # Forwarded into the compiled connection rule, where the localized
    # (``max_in_degree``) sampler reads it; probability rules carry it as
    # declared metadata. Must be finite and > 0 when set. No consumer exists
    # on the conductance (adex) route, so a declared value there is refused.
    spatial_sigma: "float | None" = None
    # Localized-sampling fan-in cap (None = undeclared = the ``probability``
    # density path). When set, the rule compiles through the localized
    # distance-weighted (``spatial_sigma``) sampler with this per-post-neuron
    # cap instead of the density path; mutually exclusive with a declared
    # ``probability`` (refused jointly). Must be a positive integer when set.
    # No consumer exists on the conductance (adex) route, so a declared
    # value there is refused.
    max_in_degree: "int | None" = None
    # Exact fixed in-degree per post cell (None = undeclared = the
    # ``probability`` density path). When set, the rule compiles through
    # the exact fixed-indegree sampler (``sampling`` ``"fixed_indegree"``)
    # with this per-post-neuron count instead of the density path;
    # mutually exclusive with a declared ``probability``, ``max_in_degree``,
    # ``footprint`` and any other sampling (refused jointly). Must be a
    # positive integer when set. No consumer exists on the conductance
    # (adex) route, so a declared value there is refused.
    in_degree: "int | None" = None
    # Per-rule self-edge opt-in (None = undeclared = self-edges dropped, the
    # historical behaviour). ``True`` keeps ``i == i`` pairs on every
    # sampling path of :func:`jaxfne.connectivity.compile_connection_rules`;
    # ``False`` states the default explicitly. Must be a bool when set.
    allow_self: "bool | None" = None
    # Per-rule compiler sampling mode (None = undeclared = the ``density``
    # default). Forwarded into the compiled connection rule by
    # :func:`_wire_connection` and :func:`_wire_adex_connection`; must name a
    # mode the compiler honours when set.
    sampling: "str | None" = None
    # Retinotopic footprint rule (None = undeclared = the density/localized
    # paths). When set, the rule compiles deterministic (pre, post) pairs
    # from posed positions (:func:`jaxfne.pcl_topology.footprint_pairs`):
    # ``{"size": (sx, sy)}`` is required, ``{"offset": (ox, oy)}`` defaults
    # to zero. Mutually exclusive with ``max_in_degree`` and with a
    # non-default ``sampling`` (refused); forwarded into the compiled rule.
    footprint: "dict | None" = None
    # Weight-sharing mode (None = undeclared = per-edge weights). ``"patch"``
    # shares one plastic row per receiving-depth slot (the footprint rule's
    # co-located post slots); needs ``footprint`` (refused without it).
    weight_sharing: "str | None" = None
    # Same-step local inhibition (None = undeclared = no intra-step fixed
    # point). When set, ``{"rounds": R, "order": ...}`` declares the ruled B3
    # form: R winners resolve per retinotopic position inside the step in the
    # declared depth visit order; delay 0 means "resolved inside the step".
    local_inhibition: "dict | None" = None

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
        object.__setattr__(
            self, "spatial_sigma",
            _check_connection_spatial_sigma(self.spatial_sigma, label="InterConnection"),
        )
        object.__setattr__(
            self, "max_in_degree",
            _check_connection_max_in_degree(self.max_in_degree, label="InterConnection"),
        )
        object.__setattr__(
            self, "in_degree",
            _check_connection_in_degree(self.in_degree, label="InterConnection"),
        )
        object.__setattr__(
            self, "allow_self",
            _check_connection_allow_self(self.allow_self, label="InterConnection"),
        )
        object.__setattr__(
            self, "sampling",
            _check_connection_sampling(self.sampling, label="InterConnection"),
        )
        object.__setattr__(
            self, "footprint",
            validate_footprint(self.footprint, label="InterConnection"),
        )
        object.__setattr__(
            self, "weight_sharing",
            validate_weight_sharing(self.weight_sharing, label="InterConnection"),
        )
        object.__setattr__(
            self, "local_inhibition",
            validate_local_inhibition(self.local_inhibition, label="InterConnection"),
        )
        if self.footprint is not None and self.max_in_degree is not None:
            raise ValueError(
                "InterConnection declares both footprint="
                f"{self.footprint!r} and max_in_degree={self.max_in_degree!r}; "
                "the footprint rule selects pairs deterministically — "
                "declare one or neither"
            )
        if self.footprint is not None and self.in_degree is not None:
            raise ValueError(
                "InterConnection declares both footprint="
                f"{self.footprint!r} and in_degree={self.in_degree!r}; "
                "the footprint rule selects pairs deterministically — "
                "declare one or neither"
            )
        if self.in_degree is not None and self.max_in_degree is not None:
            raise ValueError(
                "InterConnection declares both in_degree="
                f"{self.in_degree!r} and max_in_degree={self.max_in_degree!r}; "
                "exact fixed in-degree and localized (distance-weighted) "
                "sampling are mutually exclusive sampling modes — "
                "declare one or neither"
            )
        if self.sampling == "fixed_indegree" and self.in_degree is None:
            raise ValueError(
                "InterConnection declares sampling='fixed_indegree' without "
                "in_degree; the fixed-indegree sampler draws exactly in_degree "
                "presynaptic partners per post cell — declare both"
            )
        if self.in_degree is not None and self.sampling not in (None, "fixed_indegree"):
            raise ValueError(
                "InterConnection declares in_degree="
                f"{self.in_degree!r} with sampling={self.sampling!r}; the "
                "fixed-indegree sampler runs under sampling='fixed_indegree' "
                "or the default — declare one or neither"
            )
        if self.footprint is not None and self.sampling is not None:
            raise ValueError(
                "InterConnection declares both footprint="
                f"{self.footprint!r} and sampling={self.sampling!r}; the "
                "footprint rule selects pairs deterministically — leave "
                "sampling at its default"
            )
        if self.footprint is None and self.weight_sharing is not None:
            raise ValueError(
                "InterConnection declares weight_sharing="
                f"{self.weight_sharing!r} without footprint; sharing groups "
                "are the footprint rule's co-located post slots — declare "
                "both or neither"
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
    # Placement of this area's layer stack in global 3D space. None =
    # undeclared = the historical stacked layout (converted to a default
    # ``Pose3D()`` at construction, which applies as the identity). Whether
    # the pose was declared is recorded separately (``_pose_specified``):
    # rigid motion holds only when every area's pose is explicit, so a
    # tensor mixing declared and undeclared poses across areas is refused
    # at bridge time, while an all-undeclared tensor keeps the historical
    # layout byte-identically.
    pose: "Pose3D | None" = field(default=None)
    connectivity_mode: ConnectivityMode | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "pose":
            # None = undeclared (converted to Pose3D() in __post_init__);
            # any Pose3D value, even a default-valued one, is a declaration.
            object.__setattr__(self, "_pose_specified", value is not None)
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
        if self.pose is None:
            object.__setattr__(self, "pose", Pose3D())
        if "_pose_specified" not in self.__dict__:
            # Pose assigned without the __setattr__ hook (e.g. unpickled or
            # otherwise restored state): fall back to value comparison.
            object.__setattr__(self, "_pose_specified", self.pose != Pose3D())
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
    # Local sampling width in mm (None = undeclared = the compiler default).
    # Same forwarding semantics as InterConnection.spatial_sigma; no consumer
    # on the conductance (adex) route.
    spatial_sigma: "float | None" = None
    # Localized-sampling fan-in cap (None = undeclared = the ``probability``
    # density path). Same forwarding semantics as
    # InterConnection.max_in_degree; mutually exclusive with a declared
    # ``probability`` (refused jointly). No consumer exists on the
    # conductance (adex) route.
    max_in_degree: "int | None" = None
    # Exact fixed in-degree per post cell (None = undeclared = the
    # ``probability`` density path). Same forwarding semantics as
    # InterConnection.in_degree; mutually exclusive with a declared
    # ``probability``, ``max_in_degree``, ``footprint`` and any other
    # sampling (refused jointly). No consumer exists on the conductance
    # (adex) route.
    in_degree: "int | None" = None
    # Per-rule self-edge opt-in (None = undeclared = self-edges dropped).
    # Same forwarding semantics as InterConnection.allow_self.
    allow_self: "bool | None" = None
    # Per-rule compiler sampling mode (None = undeclared = the ``density``
    # default). Same forwarding semantics as InterConnection.sampling.
    sampling: "str | None" = None
    # Retinotopic footprint rule (None = undeclared). Same semantics as
    # InterConnection.footprint; mutually exclusive with ``probability`` and
    # ``max_in_degree`` and with a non-default ``sampling`` (refused).
    footprint: "dict | None" = None
    # Weight-sharing mode (None = undeclared = per-edge weights). Same
    # semantics as InterConnection.weight_sharing; needs ``footprint``.
    weight_sharing: "str | None" = None
    # Same-step local inhibition (None = undeclared). Same semantics as
    # InterConnection.local_inhibition: the ruled B3 intra-step fixed point.
    local_inhibition: "dict | None" = None

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
        object.__setattr__(
            self, "spatial_sigma",
            _check_connection_spatial_sigma(self.spatial_sigma, label="AreaConnection"),
        )
        object.__setattr__(
            self, "max_in_degree",
            _check_connection_max_in_degree(self.max_in_degree, label="AreaConnection"),
        )
        if self.max_in_degree is not None and self.probability is not None:
            raise ValueError(
                "AreaConnection declares both max_in_degree="
                f"{self.max_in_degree!r} and probability={self.probability!r}; "
                "localized sampling and the density path are mutually exclusive "
                "sampling modes — declare one or neither"
            )
        object.__setattr__(
            self, "in_degree",
            _check_connection_in_degree(self.in_degree, label="AreaConnection"),
        )
        if self.in_degree is not None and self.probability is not None:
            raise ValueError(
                "AreaConnection declares both in_degree="
                f"{self.in_degree!r} and probability={self.probability!r}; "
                "the fixed-indegree sampler draws an exact per-post-cell "
                "count, not a density fraction — declare one or neither"
            )
        object.__setattr__(
            self, "allow_self",
            _check_connection_allow_self(self.allow_self, label="AreaConnection"),
        )
        object.__setattr__(
            self, "sampling",
            _check_connection_sampling(self.sampling, label="AreaConnection"),
        )
        object.__setattr__(
            self, "footprint",
            validate_footprint(self.footprint, label="AreaConnection"),
        )
        object.__setattr__(
            self, "weight_sharing",
            validate_weight_sharing(self.weight_sharing, label="AreaConnection"),
        )
        object.__setattr__(
            self, "local_inhibition",
            validate_local_inhibition(self.local_inhibition, label="AreaConnection"),
        )
        if self.footprint is not None and self.probability is not None:
            raise ValueError(
                "AreaConnection declares both footprint="
                f"{self.footprint!r} and probability={self.probability!r}; "
                "the footprint rule selects pairs deterministically — "
                "declare one or neither"
            )
        if self.footprint is not None and self.max_in_degree is not None:
            raise ValueError(
                "AreaConnection declares both footprint="
                f"{self.footprint!r} and max_in_degree={self.max_in_degree!r}; "
                "the footprint rule selects pairs deterministically — "
                "declare one or neither"
            )
        if self.footprint is not None and self.in_degree is not None:
            raise ValueError(
                "AreaConnection declares both footprint="
                f"{self.footprint!r} and in_degree={self.in_degree!r}; "
                "the footprint rule selects pairs deterministically — "
                "declare one or neither"
            )
        if self.in_degree is not None and self.max_in_degree is not None:
            raise ValueError(
                "AreaConnection declares both in_degree="
                f"{self.in_degree!r} and max_in_degree={self.max_in_degree!r}; "
                "exact fixed in-degree and localized (distance-weighted) "
                "sampling are mutually exclusive sampling modes — "
                "declare one or neither"
            )
        if self.sampling == "fixed_indegree" and self.in_degree is None:
            raise ValueError(
                "AreaConnection declares sampling='fixed_indegree' without "
                "in_degree; the fixed-indegree sampler draws exactly in_degree "
                "presynaptic partners per post cell — declare both"
            )
        if self.in_degree is not None and self.sampling not in (None, "fixed_indegree"):
            raise ValueError(
                "AreaConnection declares in_degree="
                f"{self.in_degree!r} with sampling={self.sampling!r}; the "
                "fixed-indegree sampler runs under sampling='fixed_indegree' "
                "or the default — declare one or neither"
            )
        if self.footprint is not None and self.sampling is not None:
            raise ValueError(
                "AreaConnection declares both footprint="
                f"{self.footprint!r} and sampling={self.sampling!r}; the "
                "footprint rule selects pairs deterministically — leave "
                "sampling at its default"
            )
        if self.footprint is None and self.weight_sharing is not None:
            raise ValueError(
                "AreaConnection declares weight_sharing="
                f"{self.weight_sharing!r} without footprint; sharing groups "
                "are the footprint rule's co-located post slots — declare "
                "both or neither"
            )


#: W0 origin vocabulary for :class:`W0Provenance`: ``"declared"``
#: (initial weights from the declared ``PlasticParams.w_mech`` gains) or
#: ``"trained"`` (per-edge W0 bound by :func:`with_trained_weights` to one
#: run's final weights).
W0_ORIGINS = frozenset({"declared", "trained"})

#: Float dtypes a trained weight vector may carry (exact roundtrip through
#: the JSON persistence: float32 values are exact as float64 and back).
_TRAINED_W_DTYPES = ("float32", "float64")


def _checked_edge_index_tuple(values: Any, name: str) -> tuple:
    """Normalize one TrainedW0 edge-index vector to a tuple of ints.

    Bools, floats, strings and missing values are refused, never coerced.
    """
    try:
        items = tuple(values)
    except TypeError:
        raise ValueError(
            f"TrainedW0 {name} must be a sequence of edge indices; got {values!r}"
        ) from None
    out = []
    for v in items:
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise ValueError(f"TrainedW0 {name} must hold only ints; got {v!r}")
        out.append(int(v))
    return tuple(out)


def _checked_edge_weight_tuple(values: Any) -> tuple:
    """Normalize one TrainedW0 weight vector to a tuple of finite floats.

    Bools, complex values, strings, NaN and inf are refused, never coerced.
    """
    try:
        items = tuple(values)
    except TypeError:
        raise ValueError(
            f"TrainedW0 w must be a sequence of numbers; got {values!r}"
        ) from None
    out = []
    for v in items:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"TrainedW0 w must hold only real numbers; got {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"TrainedW0 w must be all finite; got {v!r}")
        out.append(f)
    return tuple(out)


@dataclass(frozen=True)
class W0Provenance:
    """Provenance of a tensor's initial edge weights (frozen, T-25).

    ``origin="declared"`` (the default) means W0 comes from the declared
    ``PlasticParams.w_mech`` gains and carries no further fields.
    ``origin="trained"`` means per-edge W0 was bound by
    :func:`with_trained_weights` to one run's final weights, and
    ``parent_digest`` is the :func:`_tensor_identity_digest` of the tensor
    that run started from. ``run_manifest_sha``/``n_steps``/``code_sha``
    are caller-supplied run receipts, all optional.
    """

    origin: str = "declared"
    parent_digest: Optional[str] = None
    run_manifest_sha: Optional[str] = None
    n_steps: Optional[int] = None
    code_sha: Optional[str] = None

    def __post_init__(self) -> None:
        if self.origin not in W0_ORIGINS:
            raise ValueError(
                f"W0Provenance origin must be one of {sorted(W0_ORIGINS)}; "
                f"got {self.origin!r}"
            )
        if self.origin == "trained":
            if not isinstance(self.parent_digest, str) or not self.parent_digest:
                raise ValueError(
                    "W0Provenance origin='trained' requires parent_digest "
                    "(the identity digest of the tensor the weights trained from)"
                )
        else:
            for key in ("parent_digest", "run_manifest_sha", "n_steps", "code_sha"):
                if getattr(self, key) is not None:
                    raise ValueError(
                        "W0Provenance origin='declared' carries no "
                        f"{key}; got {getattr(self, key)!r}"
                    )
        if self.n_steps is not None:
            if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int):
                raise ValueError(
                    f"W0Provenance n_steps must be an int >= 0; got {self.n_steps!r}"
                )
            if self.n_steps < 0:
                raise ValueError(
                    f"W0Provenance n_steps must be an int >= 0; got {self.n_steps!r}"
                )
        for key in ("run_manifest_sha", "code_sha"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"W0Provenance {key} must be a non-empty str; got {value!r}")


@dataclass(frozen=True)
class TrainedW0:
    """Per-edge trained initial weights bound to one compiled edge order.

    ``pre``/``post``/``w`` are parallel tuples of length n_edges > 0;
    ``dtype`` records the trained vector's float dtype (``"float32"`` or
    ``"float64"``): realization casts back to the model dtype exactly, and
    refuses when the cast would change any value. Edge identity (same
    edges, same order) and native-sign agreement are verified both when the
    weights bind (:func:`with_trained_weights`) and when they realize
    (:func:`_construct_neuronal_tensor_impl`); stale or foreign weights are
    refused at both gates, never silently rebound.
    """

    pre: tuple = ()
    post: tuple = ()
    w: tuple = ()
    dtype: str = "float32"

    def __post_init__(self) -> None:
        pre = _checked_edge_index_tuple(self.pre, "pre")
        post = _checked_edge_index_tuple(self.post, "post")
        w = _checked_edge_weight_tuple(self.w)
        if self.dtype not in _TRAINED_W_DTYPES:
            raise ValueError(
                f"TrainedW0 dtype must be one of {_TRAINED_W_DTYPES}; got {self.dtype!r}"
            )
        if not (len(pre) == len(post) == len(w) > 0):
            raise ValueError(
                "TrainedW0 pre/post/w must be non-empty and the same length; "
                f"got {(len(pre), len(post), len(w))}"
            )
        object.__setattr__(self, "pre", pre)
        object.__setattr__(self, "post", post)
        object.__setattr__(self, "w", w)


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
    # T-25 frozen W0 provenance plus per-edge trained weights. Both are
    # default-omitted from to_dict, so declared tensors persist and digest
    # byte-identically to earlier releases (loading fills the defaults).
    w0_provenance: W0Provenance = field(default_factory=W0Provenance)
    trained_w0: Optional[TrainedW0] = None

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
        if not isinstance(self.w0_provenance, W0Provenance):
            raise TypeError(
                "NeuronalTensor.w0_provenance must be a W0Provenance, got "
                f"{type(self.w0_provenance).__name__}"
            )
        if self.trained_w0 is not None and not isinstance(self.trained_w0, TrainedW0):
            raise TypeError(
                "NeuronalTensor.trained_w0 must be a TrainedW0 or None, got "
                f"{type(self.trained_w0).__name__}"
            )
        if (self.trained_w0 is None) != (self.w0_provenance.origin == "declared"):
            raise ValueError(
                "NeuronalTensor trained_w0 and w0_provenance.origin disagree: "
                "trained weights require origin='trained', and declared "
                "tensors carry no trained weights"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Optional fields added in 0.5.6 are written only when set, so saved
        # files and identity/phenotype digests of tensors that do not use them
        # stay byte-identical to earlier releases (loading fills the defaults).
        for area in d.get("areas", []):
            for layer in area.get("layers", []):
                for nt_d in layer.get("neuron_types", []):
                    _drop_defaults(nt_d, _NEURON_TYPE_DEFAULTS_056)
            for conn in area.get("inter_connections", []):
                _drop_defaults(conn, _CONNECTION_DEFAULTS_056)
        for conn in d.get("area_connections", []):
            _drop_defaults(conn, _CONNECTION_DEFAULTS_056)
        # T-25: same idiom for the trained-W0 surface. A default-declared
        # provenance compares equal to its plain-data default and is dropped;
        # an absent trained vector is dropped (a present one persists as data).
        if d.get("w0_provenance") == _DECLARED_W0_PROVENANCE_DICT:
            del d["w0_provenance"]
        if d.get("trained_w0") is None:
            del d["trained_w0"]
        return d


#: Plain-data default of :class:`W0Provenance` (``asdict`` form): a declared
#: provenance equal to this is omitted from :meth:`NeuronalTensor.to_dict`.
_DECLARED_W0_PROVENANCE_DICT = {
    "origin": "declared",
    "parent_digest": None,
    "run_manifest_sha": None,
    "n_steps": None,
    "code_sha": None,
}


_NEURON_TYPE_DEFAULTS_056 = {"emitter": None, "emitter_params": None, "drive": None}
_CONNECTION_DEFAULTS_056 = {
    "conduction_velocity_m_per_s": None,
    "length_unit": "mm",
    "spatial_sigma": None,
    "max_in_degree": None,
    "in_degree": None,
    "allow_self": None,
    "sampling": None,
    "footprint": None,
    "weight_sharing": None,
    "local_inhibition": None,
}


def _drop_defaults(d: dict, defaults: dict) -> None:
    for key, default in defaults.items():
        if key in d and d[key] == default and type(d[key]) is type(default):
            del d[key]


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
    """Save a NeuronalTensor as a JSON config file. Configs are data, never code.

    T-25: in-memory ``provenance`` is stripped as before, but a bound trained
    W0 (``trained_w0`` plus ``w0_provenance``) persists deliberately — it is
    plain per-edge data, and :func:`load` reconstructs it exactly.
    """
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
                spatial_sigma=ac.get("spatial_sigma"),
                max_in_degree=ac.get("max_in_degree"),
                in_degree=ac.get("in_degree"),
                allow_self=ac.get("allow_self"),
                sampling=ac.get("sampling"),
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
                        spatial_sigma=ic.get("spatial_sigma"),
                        max_in_degree=ic.get("max_in_degree"),
                        in_degree=ic.get("in_degree"),
                        allow_self=ic.get("allow_self"),
                        sampling=ic.get("sampling"),
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
        # T-25: trained W0 persists as plain data (absent = declared
        # defaults); field validation refuses half-bound files.
        w0_provenance=(
            W0Provenance(**raw["w0_provenance"])
            if raw.get("w0_provenance")
            else W0Provenance()
        ),
        trained_w0=(
            TrainedW0(**raw["trained_w0"]) if raw.get("trained_w0") else None
        ),
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
    # T-25: a bound trained W0 is indexed by one tensor's compiled edges —
    # merging would silently rebind or drop it, so it is refused.
    for tensor in tensors:
        if tensor.trained_w0 is not None:
            raise ValueError(
                f"merge_neuronal_tensors refuses tensor {tensor.name!r} carrying "
                "trained W0 (origin='trained'): merged edges are a new circuit, "
                "so bound per-edge weights cannot transfer — bind trained "
                "weights on the merged tensor instead"
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
            # dataclasses.replace re-passes pose explicitly, which would mark
            # every merged area pose-specified; inherit the declaration flag
            # instead (an explicit poses= entry is a declaration).
            pose_specified = True if poses is not None else bool(
                getattr(area, "_pose_specified", area.pose != Pose3D())
            )
            if area.connectivity_mode == "unspecified":
                merged = replace(
                    area,
                    name=new_name,
                    pose=pose,
                    inter_connections=_UNSPECIFIED_CONNECTIONS,
                )
            else:
                merged = replace(area, name=new_name, pose=pose)
            object.__setattr__(merged, "_pose_specified", pose_specified)
            merged_areas.append(merged)
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
    :func:`jaxfne.construct` sampled it — and the bridge also wrote each
    ``Area``'s ``Pose3D`` (``tensor_pose`` metadata, only when some pose is
    non-default), so localized (``max_in_degree``) edge sampling reads the
    posed layout. This then applies the same placement (plane + rotation +
    translation) onto those constructed positions — no re-sampling, one
    source of truth for geometry. A default pose is the identity,
    bit-exact.

    Connectivity mode is preserved from the tensor: omitted connectivity uses
    the configuration default topology, while explicit connectivity (including
    an explicit empty declaration) compiles only the declared graph.

    A bound trained W0 (T-25) realizes only on the izhikevich route: lif/adex
    tensors carrying one are refused, and the izhikevich route verifies the
    stored edges/signs before applying (see :func:`_apply_trained_w0`).
    """
    family, overrides = _resolve_tensor_emitter_family(tensor, emitter)
    if tensor.trained_w0 is not None and family != "izhikevich":
        raise ValueError(
            f"tensor {tensor.name!r} carries trained W0 but resolves to emitter "
            f"family {family!r}; trained W0 is realized only on the izhikevich "
            "route (lif/adex/rate have no trained-W0 consumer) — declare izhikevich "
            "or drop the trained weights"
        )
    if family == "lif":
        from ._construct_lif import construct_lif_tensor_model

        if n_contacts != 16:
            raise ValueError(
                f"n_contacts={n_contacts!r} is not consumed on the lif tensor route "
                "(no laminar probe readout there); leave the default"
            )
        return construct_lif_tensor_model(
            tensor,
            seed=seed,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            dtype=dtype,
            backend=backend,
            jit=jit,
            vmap=vmap,
            n_contacts=n_contacts,
            param_overrides=overrides,
        )
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
    if family == "rate":
        from ._construct_rate import construct_rate_tensor_model

        if n_contacts != 16:
            raise ValueError(
                f"n_contacts={n_contacts!r} is not consumed on the rate tensor route "
                "(no laminar probe readout there); leave the default"
            )
        return construct_rate_tensor_model(
            tensor,
            seed=seed,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            dtype=dtype,
            backend=backend,
            jit=jit,
            vmap=vmap,
            n_contacts=n_contacts,
            param_overrides=overrides,
        )
    if adex_decay != "exp" or adex_step_order != "default":
        raise ValueError(
            f"adex_decay={adex_decay!r}, adex_step_order={adex_step_order!r} are not "
            f"consumed for emitter {family!r} on the current-based route; leave "
            "the defaults"
        )
    # T-25: the Configuration bridge carries declared structure only (a bound
    # trained W0 is refused there), so bridge a declared copy and stamp the
    # full identity — including the trained vector — into the receipt metadata
    # afterwards; _apply_trained_w0 then realizes the trained vector exactly.
    bridge_tensor = tensor
    if tensor.trained_w0 is not None:
        bridge_tensor = replace(tensor, trained_w0=None, w0_provenance=W0Provenance())
    cfg = neuronal_tensor_to_configuration(
        bridge_tensor,
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
    if bridge_tensor is not tensor:
        cfg = cfg.update_metadata(tensor_identity=_tensor_identity_digest(tensor))
    model = construct(cfg)
    model = _apply_trained_w0(model, tensor)
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


def _as_1d_edge_vector(value: Any, *, name: str) -> "object":
    """Coerce one candidate edge vector to a 1-D numpy array (T-25).

    Shape only — dtype and value checks belong to the caller so each refusal
    names the invariant it guards. Empty and non-1-D inputs are refused.
    """
    import numpy as _np

    try:
        arr = _np.asarray(value)
    except Exception as exc:
        raise ValueError(
            f"with_trained_weights: {name} is not array-like ({exc})"
        ) from exc
    if arr.ndim != 1:
        raise ValueError(
            f"with_trained_weights: {name} must be a 1-D per-edge vector; "
            f"got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(
            f"with_trained_weights: {name} is empty; need at least one edge"
        )
    return arr


def _native_edge_signs(edge_list: Any, emitter: Any) -> "object":
    """Executed per-edge weights of a compiled edge list, as float64 numpy.

    Resolves through :func:`jaxfne.emitters._resolved_edge_weight` — the same
    function the kernels consume — so compact ``weight_storage`` modes (whose
    ``weight`` field is a size-0 placeholder) resolve exactly, never by guess.
    """
    import numpy as _np

    return _np.asarray(
        _resolved_edge_weight(edge_list, jnp.float32, emitter), dtype=_np.float64
    ).reshape(-1)


def with_trained_weights(
    tensor: NeuronalTensor,
    source: Any,
    *,
    pre: Any = None,
    post: Any = None,
    seed: int = 0,
    run_manifest_sha: "str | None" = None,
    n_steps: "int | None" = None,
    code_sha: "str | None" = None,
) -> NeuronalTensor:
    """Bind one run's final per-edge weights to a copy of ``tensor`` (T-25).

    Returns A' — a new :class:`NeuronalTensor` carrying the trained vector as
    :class:`TrainedW0` plus :class:`W0Provenance` (``origin="trained"``,
    ``parent_digest`` = identity digest of ``tensor``). Constructing A' on the
    izhikevich route realizes the trained weights exactly
    (:func:`_construct_neuronal_tensor_impl` verifies stored vs compiled
    ``(pre, post)`` and native signs again before applying). ``tensor`` is
    never mutated.

    ``source`` forms (first match wins):

    - a :class:`jaxfne.core.Model` with HDP diagnostics (primary): the final
      weights come from ``model.last_hdp_diagnostics()["w_final"]`` and the
      reference edges from ``model.params["edge_list"]``; ownership is the
      ``tensor_identity`` config-metadata cross-check (a model built from any
      other tensor is refused). Explicit ``pre``/``post`` alongside a Model
      is refused as ambiguous.
    - a ``(pre, post, w)`` triple, a ``{"pre", "post", "w"}`` dict, a
      :class:`jaxfne._pipeline.ContinuationState` (``dynamic.w``), or a bare
      weight vector — each with ``pre``/``post`` made explicit (kwargs for a
      state or bare vector; inside the triple/dict otherwise). Reference
      edges are a deterministic reconstruction of ``tensor`` at ``seed`` (the
      construct seed the circuit was built with, not the simulate seed);
      compiled ``(pre, post)`` are seed-deterministic, so a wrong seed is
      refused, never rebound.
    - anything else is refused: a bare :class:`Signals` (HDP summaries only,
      never per-edge weights — pass the trained Model, or re-run with
      ``return_state=True`` and pass the state with explicit ``pre``/``post``),
      a checkpoint leaf list (unlabeled — rebind through the trained Model),
      a model with no HDP diagnostics (re-run with ``enable_hdp=True``).

    Every form checks shape (1-D, same length, non-empty), dtype (integer
    ``pre``/``post``, float32/float64 ``w`` — never coerced), finiteness,
    edge identity (same edges, same order) and native-sign agreement (HDP
    updates are sign-preserving magnitude updates, so a flipped sign means
    foreign weights). Optional ``run_manifest_sha``/``n_steps``/``code_sha``
    are recorded on the provenance.
    """
    import numpy as _np

    if not isinstance(tensor, NeuronalTensor):
        raise TypeError(
            "with_trained_weights: tensor must be a NeuronalTensor, got "
            f"{type(tensor).__name__}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"with_trained_weights: seed must be an int; got {seed!r}")

    if isinstance(source, Model):
        if pre is not None or post is not None:
            raise ValueError(
                "with_trained_weights: a Model carries its own edges — pass "
                "pre/post only with weight-vector sources (triple, dict, "
                "state, or bare array), never alongside a Model"
            )
        # Same family gate as the weight-vector forms below: lif/adex tensors
        # are refused at bind, not at construct.
        try:
            family, _ = _resolve_tensor_emitter_family(tensor, "izhikevich")
        except ValueError as exc:
            raise ValueError(
                "with_trained_weights: trained W0 is realized only on the "
                f"izhikevich route ({exc})"
            ) from exc
        if family != "izhikevich":
            raise ValueError(
                "with_trained_weights: trained W0 is realized only on the "
                f"izhikevich route; tensor resolves to {family!r}"
            )
        edge_list = source.params.get("edge_list")
        if edge_list is None:
            raise ValueError(
                "with_trained_weights: model has no params['edge_list'] (the "
                "adex route stores edges in static['adex']); trained W0 binds "
                "only to edge-list models"
            )
        diag = source.last_hdp_diagnostics()
        if diag is None:
            raise ValueError(
                "with_trained_weights: model has no HDP diagnostics — re-run "
                "it with RuntimeConfig(enable_hdp=True, hdp_params={...}) and "
                "pass the trained model"
            )
        w_final = diag.get("w_final")
        if w_final is None:
            raise ValueError(
                "with_trained_weights: HDP diagnostics carry no w_final — "
                "re-run with enable_hdp=True and pass the trained model"
            )
        owner = (getattr(getattr(source, "cfg", None), "metadata", None) or {}).get(
            "tensor_identity"
        )
        if owner != _tensor_identity_digest(tensor):
            raise ValueError(
                "with_trained_weights: model was not built from this tensor "
                "(tensor_identity mismatch) — trained weights bind only to "
                "the circuit that produced them"
            )
        emitter = source.params.get("emitter")
        if emitter is None:
            raise ValueError(
                "with_trained_weights: model has no emitter — cannot resolve "
                "native edge signs"
            )
        ref_pre = _np.asarray(edge_list.pre).reshape(-1)
        ref_post = _np.asarray(edge_list.post).reshape(-1)
        native = _native_edge_signs(edge_list, emitter)
        pre, post = ref_pre, ref_post
    else:
        from ._pipeline import ContinuationState as _ContinuationState
        from ._signals import Signals as _Signals

        if isinstance(source, _Signals):
            raise ValueError(
                "with_trained_weights: bare Signals carry HDP summaries only "
                "(metadata['hdp']), never per-edge weights — pass the trained "
                "Model, or re-run with return_state=True and pass the state "
                "with explicit pre/post"
            )
        if isinstance(source, list):
            raise ValueError(
                "with_trained_weights: unlabeled leaf list (e.g. "
                "checkpoint_state/restore_state leaves carry no edge labels) — "
                "rebind through the trained Model or pass an explicit "
                "pre/post/w triple"
            )
        kw_edges = pre is not None or post is not None
        if isinstance(source, _ContinuationState):
            if pre is None or post is None:
                raise ValueError(
                    "with_trained_weights: a ContinuationState carries w but "
                    "no edges — pass explicit pre= and post= from the trained "
                    "model"
                )
            w_final = source.dynamic.w
        elif isinstance(source, tuple):
            if kw_edges:
                raise ValueError(
                    "with_trained_weights: pass edges once — inside the "
                    "(pre, post, w) triple or as pre=/post= kwargs, never both"
                )
            if len(source) == 3 and not isinstance(source[0], _Signals):
                pre, post, w_final = source
            elif len(source) == 2 and isinstance(source[0], _Signals):
                raise ValueError(
                    "with_trained_weights: a (Signals, state) pair carries no "
                    "edges on its own — pass the state with explicit pre= and "
                    "post= from the trained model"
                )
            else:
                raise ValueError(
                    "with_trained_weights: tuple source must be a "
                    f"(pre, post, w) triple; got length {len(source)}"
                )
        elif isinstance(source, dict):
            if kw_edges:
                raise ValueError(
                    "with_trained_weights: pass edges once — inside the dict "
                    "or as pre=/post= kwargs, never both"
                )
            missing = {"pre", "post", "w"} - set(source)
            if missing:
                raise ValueError(
                    "with_trained_weights: dict source needs keys pre/post/w; "
                    f"missing {sorted(missing)}"
                )
            pre, post, w_final = source["pre"], source["post"], source["w"]
        else:
            if pre is None or post is None:
                raise ValueError(
                    "with_trained_weights: a bare weight vector needs explicit "
                    "pre= and post= from the trained model"
                )
            w_final = source
        try:
            family, _ = _resolve_tensor_emitter_family(tensor, "izhikevich")
        except ValueError as exc:
            raise ValueError(
                "with_trained_weights: trained W0 is realized only on the "
                f"izhikevich route ({exc})"
            ) from exc
        if family != "izhikevich":
            raise ValueError(
                "with_trained_weights: trained W0 is realized only on the "
                f"izhikevich route; tensor resolves to {family!r}"
            )
        ref_model = _construct_neuronal_tensor_impl(
            tensor, seed=seed, duration_ms=10.0, dt_ms=0.5
        )
        ref_list = ref_model.params.get("edge_list")
        if ref_list is None:
            raise ValueError(
                "with_trained_weights: reference construction of tensor "
                f"{tensor.name!r} produced no params['edge_list']"
            )
        ref_emitter = ref_model.params.get("emitter")
        if ref_emitter is None:
            raise ValueError(
                "with_trained_weights: reference construction of tensor "
                f"{tensor.name!r} produced no emitter"
            )
        ref_pre = _np.asarray(ref_list.pre).reshape(-1)
        ref_post = _np.asarray(ref_list.post).reshape(-1)
        native = _native_edge_signs(ref_list, ref_emitter)

    pre_a = _as_1d_edge_vector(pre, name="pre")
    post_a = _as_1d_edge_vector(post, name="post")
    w_a = _as_1d_edge_vector(w_final, name="w")
    if not (pre_a.shape == post_a.shape == w_a.shape):
        raise ValueError(
            "with_trained_weights: pre/post/w must have the same length; got "
            f"{(pre_a.shape, post_a.shape, w_a.shape)}"
        )
    for label, arr in (("pre", pre_a), ("post", post_a)):
        if not _np.issubdtype(arr.dtype, _np.integer):
            raise ValueError(
                f"with_trained_weights: {label} must be an integer array; got "
                f"dtype {arr.dtype} (no silent coercion — pass int edges)"
            )
    if not _np.issubdtype(w_a.dtype, _np.floating) or w_a.dtype.itemsize not in (4, 8):
        raise ValueError(
            "with_trained_weights: w must be float32 or float64; got dtype "
            f"{w_a.dtype}"
        )
    if not bool(_np.all(_np.isfinite(w_a))):
        raise ValueError("with_trained_weights: w must be all finite; got NaN or inf")
    w_dtype = "float32" if w_a.dtype.itemsize == 4 else "float64"
    if pre_a.shape != ref_pre.shape or not bool(_np.array_equal(pre_a, ref_pre)):
        raise ValueError(
            "with_trained_weights: candidate pre does not match the edges "
            f"compiled from tensor {tensor.name!r} at seed={seed} (same edges, "
            "same order required)"
        )
    if post_a.shape != ref_post.shape or not bool(_np.array_equal(post_a, ref_post)):
        raise ValueError(
            "with_trained_weights: candidate post does not match the edges "
            f"compiled from tensor {tensor.name!r} at seed={seed} (same edges, "
            "same order required)"
        )
    w_f = _np.asarray(w_a, dtype=_np.float64)
    if native.shape != w_f.shape:
        raise ValueError(
            "with_trained_weights: reference edges "
            f"({native.shape[0]}) and candidate weights "
            f"({w_f.shape[0]}) disagree in length"
        )
    # A weight clamped to exactly 0 keeps no sign to flip; any nonzero weight must match.
    flipped = int(_np.sum((w_f != 0) & (_np.sign(w_f) != _np.sign(native))))
    if flipped:
        raise ValueError(
            f"with_trained_weights: {flipped}/{w_f.shape[0]} trained weights "
            "flip the native edge sign — HDP updates are sign-preserving "
            "magnitude updates, so these are foreign weights; refusing"
        )
    parent = _tensor_identity_digest(tensor)
    trained = TrainedW0(
        pre=tuple(int(v) for v in pre_a.tolist()),
        post=tuple(int(v) for v in post_a.tolist()),
        w=tuple(float(v) for v in w_f.tolist()),
        dtype=w_dtype,
    )
    provenance = W0Provenance(
        origin="trained",
        parent_digest=parent,
        run_manifest_sha=run_manifest_sha,
        n_steps=n_steps,
        code_sha=code_sha,
    )
    return replace(tensor, trained_w0=trained, w0_provenance=provenance)


def _apply_trained_w0(model: Model, tensor: NeuronalTensor) -> Model:
    """Realize a bound trained W0 on a freshly constructed model (T-25).

    No-op when ``tensor`` carries no trained weights. Otherwise verifies the
    stored ``(pre, post)`` against the compiled edges exactly (same edges,
    same order — stale or foreign weights are refused) and native-sign
    agreement, then swaps in a ``per_edge`` :class:`EdgeList` carrying the
    trained vector — the same replace idiom as
    :func:`jaxfne.emitters.edge_list_per_edge_zeros`, so compact
    ``weight_storage`` modes (whose ``weight`` is a size-0 placeholder)
    execute the trained vector exactly rather than being refused.
    """
    import numpy as _np

    stored = tensor.trained_w0
    if stored is None:
        return model
    edge_list = model.params.get("edge_list")
    if edge_list is None:
        raise ValueError(
            f"tensor {tensor.name!r} carries trained W0 but the constructed "
            "model has no params['edge_list'] — trained W0 cannot realize here"
        )
    pre_s = _np.asarray(stored.pre, dtype=_np.int64)
    post_s = _np.asarray(stored.post, dtype=_np.int64)
    pre_c = _np.asarray(edge_list.pre).reshape(-1)
    post_c = _np.asarray(edge_list.post).reshape(-1)
    if (
        pre_s.shape != pre_c.shape
        or not bool(_np.array_equal(pre_s, pre_c))
        or post_s.shape != post_c.shape
        or not bool(_np.array_equal(post_s, post_c))
    ):
        raise ValueError(
            f"tensor {tensor.name!r} carries trained W0 for {pre_s.shape[0]} "
            f"edges but this construction compiled {pre_c.shape[0]} — rebuild "
            "with the seed/topology the weights trained on (stale or foreign "
            "weights are refused, never rebound)"
        )
    w_s = _np.asarray(stored.w, dtype=_np.dtype(stored.dtype))
    emitter = model.params.get("emitter")
    if emitter is None:
        raise ValueError(
            f"tensor {tensor.name!r} carries trained W0 but the constructed "
            "model has no emitter — cannot resolve native edge signs"
        )
    native = _native_edge_signs(edge_list, emitter)
    flipped = int(_np.sum((w_s != 0) & (_np.sign(w_s) != _np.sign(native))))
    if flipped:
        raise ValueError(
            f"tensor {tensor.name!r}: {flipped}/{w_s.shape[0]} stored weights "
            "flip the compiled native edge sign — refusing"
        )
    jdtype = getattr(getattr(emitter, "v0", None), "dtype", jnp.float32)
    # The stored vector keeps its bound dtype: refuse when the cast to the
    # model dtype changes any value (float64 detail would be silently rounded
    # off on a float32 model), else apply exactly.
    model_np_dtype = _np.dtype(jdtype)
    w_cast = w_s.astype(model_np_dtype)
    changed = int(_np.sum(w_cast.astype(_np.float64) != w_s.astype(_np.float64)))
    if changed:
        raise ValueError(
            f"tensor {tensor.name!r} stores trained W0 in {stored.dtype} but "
            f"{changed}/{w_s.shape[0]} values change when cast to the model "
            f"dtype {model_np_dtype.name} — refusing (float64 detail would be "
            "lost on a float32 model); rebind at the model dtype"
        )
    new_edges = replace(
        edge_list,
        weight=jnp.asarray(w_cast, dtype=jdtype),
        weight_storage="per_edge",
        weight_magnitude=None,
        mechanism_weight_magnitude_table=None,
    )
    return replace(model, params={**model.params, "edge_list": new_edges})


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
    param_overrides: "dict[tuple[str, str, str], dict[str, float]] | None" = None,
) -> Model:
    """Build a conductance (``adex``) model from a tensor resolving to ``"adex"``.

    Population size and positions are harvested from the current-based bridge
    (with emitter resolution skipped, so no family switch fires), posed with
    the shared helper, then bridged into the adex configuration and built by
    :func:`jaxfne._construct_adex.construct_adex_model` with the harvested
    rows, posed positions and per-triple param overrides. Geometry-derived
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
    """Resolve which emitter family builds ``tensor`` and its per-triple params.

    Every :class:`NeuronType` declares ``emitter`` plus optional
    ``emitter_params`` (validated against :mod:`jaxfne.emitter_registry`;
    unknown names and unconsumed keys are refused). When no type declares,
    the runtime emitter applies as before. When any type declares, every
    type must declare, the family must be uniform, and the runtime emitter
    must agree; a registered-but-unroutable family is refused at bridge
    time. Returns ``(family, overrides)`` where ``overrides`` maps an
    (area, layer, type) triple to its declared params (empty unless the
    family consumes per-type params): params apply per triple, matching
    drive semantics, so two triples sharing a type name may declare
    different values without conflict.
    """
    declared: list[tuple[str, str, str, str, "dict | None"]] = []
    undeclared: list[str] = []
    for area in tensor.areas:
        for layer in area.layers:
            for nt in layer.neuron_types:
                key = f"{area.name}.{layer.name}.{nt.name}"
                if nt.emitter is None:
                    undeclared.append(key)
                else:
                    params = check_emitter_params(nt.emitter, nt.emitter_params)
                    declared.append((area.name, layer.name, nt.name, nt.emitter, params))
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
    families = {emitter for _, _, _, emitter, _ in declared}
    if len(families) > 1:
        by_family: dict[str, list[str]] = {}
        for area, layer, typename, emitter, _ in declared:
            by_family.setdefault(emitter, []).append(f"{area}.{layer}.{typename}")
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
    overrides: dict[tuple[str, str, str], dict[str, float]] = {}
    for area, layer, typename, _, params in declared:
        if params:
            overrides[(area, layer, typename)] = params
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


def _refuse_impossible_allow_self(
    conn: "InterConnection | AreaConnection",
    source_area: str,
    target_area: str,
    *,
    label: str,
) -> None:
    """Refuse ``allow_self=True`` where no self pair can exist.

    A self edge needs one neuron selected as both pre and post, so the
    source and target triples must agree in area, layer and type. Opting
    into self-edges across areas, layers or types wires nothing and hides
    a misspecified selector, so it is refused rather than compiled empty.
    """
    if conn.allow_self is True and (
        source_area != target_area
        or conn.source_layer != conn.target_layer
        or conn.source_neuron_type != conn.target_neuron_type
    ):
        raise ValueError(
            f"{label} declares allow_self=True but no self pair can exist "
            f"({source_area}.{conn.source_layer}.{conn.source_neuron_type}"
            f"->{target_area}.{conn.target_layer}.{conn.target_neuron_type}): "
            "self-edges need the same area, layer and type on both sides"
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
    A declared ``max_in_degree`` routes the rule through the localized
    (distance-weighted, ``spatial_sigma``) sampler instead of the
    ``probability`` density path (the two are mutually exclusive, so the
    rule carries ``probability=None`` then); a declared ``in_degree``
    routes the rule through the exact fixed-indegree sampler instead (also
    with ``probability=None``; mutually exclusive with ``max_in_degree``
    and ``footprint``); a declared ``spatial_sigma``
    without ``max_in_degree`` has no consumer on the density path, so it
    is refused here (``spatial_sigma needs max_in_degree``). A declared
    ``footprint`` compiles deterministic pairs from the posed positions
    instead of sampling (``probability=None`` on the rule; ``probability``,
    ``max_in_degree`` and non-default ``sampling`` alongside it are refused
    at declaration); a declared ``weight_sharing`` rides the rule for the
    HDP rule to consume. Sampling reads
    the posed neuron positions — each area's ``Pose3D`` is applied to a
    wiring copy before edge compilation, and the same pose maps the stored
    positions afterwards — so moving an area changes the realized edges
    while a rigid motion of the whole tensor preserves them.
    Sign follows the P-023 mechanism vocabulary
    (:func:`_check_connection_mechanism_sign`), except for TFNE-derived
    tensors (``_tfne_resolved``: TFNE carries its own fail-closed mechanism
    vocabulary plus weight-polarity sign, so the declaration vocabulary is
    not re-applied there; owner ruling 2026-09-30, pinned by the TFNE
    custom-mechanism, direct-coupling and polarity tests).
    """
    label = _connection_label(conn, source_area, target_area)
    _refuse_impossible_allow_self(conn, source_area, target_area, label=label)
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
    conn_sigma = getattr(conn, "spatial_sigma", None)
    conn_degree = getattr(conn, "max_in_degree", None)
    if conn_sigma is not None and conn_degree is None:
        raise ValueError(
            f"{label} declares spatial_sigma={conn_sigma!r} without "
            "max_in_degree; spatial_sigma needs max_in_degree (localized "
            "sampling) — declare both or neither"
        )
    if conn_sigma is not None:
        # Validated at declaration (finite, > 0); read by the localized
        # (``max_in_degree``) sampler.
        wire_kw["spatial_sigma"] = float(conn_sigma)
    if conn_degree is not None:
        # Localized sampling replaces the density path (Configuration
        # refuses the combination), so the rule carries probability=None.
        wire_kw["max_in_degree"] = int(conn_degree)
        wire_probability = None
    conn_in_degree = getattr(conn, "in_degree", None)
    if conn_in_degree is not None:
        # Exact fixed in-degree replaces the density path (Configuration
        # refuses the combination), so the rule carries probability=None.
        wire_kw["in_degree"] = int(conn_in_degree)
        wire_probability = None
    if conn_degree is not None and conn_in_degree is not None:
        # Unreachable through the declaring constructors (both refuse the
        # combination), but the compiler must never see both samplers.
        raise ValueError(
            f"{label} declares both max_in_degree={conn_degree!r} and "
            f"in_degree={conn_in_degree!r}; exact fixed in-degree and "
            "localized (distance-weighted) sampling are mutually exclusive "
            "sampling modes — declare one or neither"
        )
    if conn_degree is None and conn_in_degree is None:
        wire_probability = (
            1.0 if getattr(conn, "probability", None) is None
            else float(conn.probability)
        )
    conn_sampling = getattr(conn, "sampling", None)
    if conn_sampling is not None:
        # Validated at declaration; None leaves the rule without the key
        # (the compiler default).
        wire_kw["sampling"] = conn_sampling
    conn_footprint = getattr(conn, "footprint", None)
    if conn_footprint is not None:
        # Validated and normalized at declaration ({"size", "offset"}); the
        # footprint rule selects pairs deterministically from posed
        # positions, so the rule carries no density probability (the
        # compiler refuses the combination).
        wire_kw["footprint"] = {
            "size": list(conn_footprint["size"]),
            "offset": list(conn_footprint["offset"]),
        }
        wire_probability = None
    conn_sharing = getattr(conn, "weight_sharing", None)
    if conn_sharing is not None:
        # Validated at declaration; needs footprint (refused without it at
        # declaration and again at compile time).
        wire_kw["weight_sharing"] = str(conn_sharing)
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
        probability=wire_probability,
        weight=_connection_edge_weight(conn, total_n),
        sign=sign,
        mechanism=mech_name,
        **wire_kw,  # type: ignore[arg-type]
    )
    if getattr(conn, "allow_self", None) is not None:
        # Per-rule self-edge opt-in (None = undeclared = self-edges dropped).
        # Configuration.connections has no allow_self_connections kwarg, so
        # the opt-in is recorded through the connectivity helper, which
        # copies the named rule entry; the compiler reads the per-rule key
        # on every sampling path.
        from .connectivity import with_rule_allow_self_connections

        cfg = with_rule_allow_self_connections(
            cfg, rule_name, bool(getattr(conn, "allow_self"))
        )
    if getattr(conn, "local_inhibition", None) is not None:
        # Same idiom: Configuration.connections has no local_inhibition
        # kwarg, so the declared intra-step fixed point (R and the round
        # order, validated and normalized at declaration) is recorded on
        # the named rule entry; the compiler echoes it onto the
        # connection-table row so it rides the manifest.
        from .connectivity import with_rule_local_inhibition

        spec = getattr(conn, "local_inhibition")
        cfg = with_rule_local_inhibition(cfg, rule_name, spec)
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
    the weight is in nS (a declaration realizing to 0 nS is refused: a
    declared edge that carries no conductance has no effect). ``delay_ms`` is the declared constant, or
    ``velocity_delay_ms`` (the median over the rule's selected pairs when the
    connection carries ``conduction_velocity_m_per_s``); one of the two is
    required, since the adex compiler refuses a rule without a delay. The
    P-023 mechanism-name vocabulary is not applied here: adex receptors are
    user-named (``"ampa"``, ``"gaba"``, ...), resolved against the declared
    set by name. A declared ``sampling`` is forwarded into the rule (None
    leaves the rule without the key, the compiler default).
    """
    label = _connection_label(conn, source_area, target_area)
    _refuse_impossible_allow_self(conn, source_area, target_area, label=label)
    if getattr(conn, "spatial_sigma", None) is not None:
        raise ValueError(
            f"{label} declares spatial_sigma={getattr(conn, 'spatial_sigma')!r}; "
            "the conductance (adex) route has no localized-sampling consumer "
            "(rule max_in_degree/spatial_sigma are refused there: only None "
            "is admitted) — leave spatial_sigma undeclared there"
        )
    if getattr(conn, "max_in_degree", None) is not None:
        raise ValueError(
            f"{label} declares max_in_degree={getattr(conn, 'max_in_degree')!r}; "
            "the conductance (adex) route has no localized-sampling consumer "
            "(rule max_in_degree/spatial_sigma are refused there: only None "
            "is admitted) — leave max_in_degree undeclared there"
        )
    if getattr(conn, "in_degree", None) is not None:
        raise ValueError(
            f"{label} declares in_degree={getattr(conn, 'in_degree')!r}; "
            "the conductance (adex) route has no fixed-indegree consumer "
            "(exact per-post-cell counts are refused there: only None "
            "is admitted) — leave in_degree undeclared there"
        )
    if getattr(conn, "weight_sharing", None) is not None:
        raise ValueError(
            f"{label} declares weight_sharing={getattr(conn, 'weight_sharing')!r}; "
            "the conductance (adex) route has no shared-weight consumer "
            "(only None is admitted) — leave weight_sharing undeclared there"
        )
    if getattr(conn, "local_inhibition", None) is not None:
        raise ValueError(
            f"{label} declares local_inhibition={getattr(conn, 'local_inhibition')!r}; "
            "the conductance (adex) route has no intra-step fixed-point consumer "
            "(only None is admitted) — leave local_inhibition undeclared there"
        )
    if getattr(conn, "footprint", None) is not None:
        raise ValueError(
            f"{label} declares footprint={getattr(conn, 'footprint')!r}; "
            "the conductance (adex) route has no footprint-rule consumer "
            "(deterministic retinotopic pairs are refused there: only None "
            "is admitted) — leave footprint undeclared there"
        )
    if conn.allow_self is True and conn.conduction_velocity_m_per_s is not None:
        raise ValueError(
            f"{label} declares allow_self=True with "
            "conduction_velocity_m_per_s="
            f"{float(conn.conduction_velocity_m_per_s)!r}; a zero-distance "
            "self edge realizes to 0 delay steps and fails at simulate — "
            "the combination is refused on the conductance (adex) route "
            "(true zero delay is handled separately); drop allow_self or "
            "the velocity there"
        )
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
    if weight_nS == 0.0:
        raise ValueError(
            f"{label} realizes to an edge weight of 0 nS on the conductance "
            f"(adex) route (plastic.w_mech={conn.plastic.w_mech!r} scaled by "
            f"static.g_mech[{conn.mechanism!r}]="
            f"{conn.static.g_mech.get(conn.mechanism, 1.0)!r}); a declared "
            "edge that carries no conductance is refused — declare a nonzero "
            "gain or drop the connection"
        )
    if conn.delay_ms is not None:
        rule_delay_ms = float(conn.delay_ms)
    elif velocity_delay_ms is not None:
        rule_delay_ms = float(velocity_delay_ms)
    else:
        raise ValueError(
            f"{label} declares neither delay_ms nor "
            "conduction_velocity_m_per_s; the adex route needs one delay per rule"
        )
    conn_sampling = getattr(conn, "sampling", None)
    wire_kw: dict[str, object] = {}
    if conn_sampling is not None:
        # Validated at declaration; None leaves the rule without the key
        # (the compiler default).
        wire_kw["sampling"] = conn_sampling
    return_cfg = cfg.connections(
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
        **wire_kw,  # type: ignore[arg-type]
    )
    if getattr(conn, "allow_self", None) is not None:
        # Same per-rule opt-in as the current-based route; the shared
        # compiler reads it on every sampling path, and ADEX_CONSUMED admits
        # exactly this rule key (spatial_sigma stays refused above).
        from .connectivity import with_rule_allow_self_connections

        return_cfg = with_rule_allow_self_connections(
            return_cfg, rule_name, bool(getattr(conn, "allow_self"))
        )
    return return_cfg


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

    Includes areas, layers, neuron types, geometry, pose, connections, and —
    when bound — the trained W0 vector and its provenance: they are the
    circuit's initial condition, so a trained tensor digests distinctly from
    its parent. Untrained tensors omit those fields in ``to_dict`` and digest
    byte-identically to earlier releases. The digest covers the persisted
    content of the tensor (the same surface as ``save_neuronal_tensor``,
    which intentionally strips in-memory ``provenance``). Runtime knobs (seed/duration/dt/emitter) are excluded —
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
    except Exception as exc:  # fail closed: never return a colliding "" identity
        raise ValueError(
            f"cannot compute identity digest for tensor "
            f"{getattr(tensor, 'name', type(tensor).__name__)!r}: {exc}"
        ) from exc
    payload.pop("provenance", None)
    # T-25: a declared W0 provenance compares equal to its plain-data default
    # and is already omitted by to_dict, so untrained tensors digest
    # byte-identically to earlier releases; a bound trained W0 and its
    # provenance are part of the circuit's initial condition, so A' digests
    # distinctly from its parent A, and a second generation A'' records A'
    # as parent (lineage holds the full chain).
    blob = _json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda o: o if isinstance(o, (str, int, float, bool)) or o is None else repr(o),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _refuse_mixed_pose_specification(tensor: NeuronalTensor) -> None:
    """Refuse a tensor mixing declared and undeclared area poses.

    A default ``Pose3D`` on area k >= 1 sits at k * spacing, while an
    explicit near-zero pose sits near the origin, so rigid motion holds
    only when every area's pose is explicit. An all-undeclared tensor keeps
    the historical layout byte-identically; a mixed one is refused rather
    than built on an ambiguous frame. Explicitness is the declaration, not
    the value: even a default-valued ``Pose3D()`` counts as declared.
    """
    declared = [a.name for a in tensor.areas if a._pose_specified]
    undeclared = [a.name for a in tensor.areas if not a._pose_specified]
    if declared and undeclared:
        raise ValueError(
            "tensor mixes declared poses "
            f"({', '.join(declared)}) with undeclared poses "
            f"({', '.join(undeclared)}); rigid motion holds only when every "
            "area's pose is explicit — declare a pose on every area or on none"
        )


def _tensor_pose_spec(tensor: NeuronalTensor) -> "dict[str, object] | None":
    """Plain-data pose record for the construction wiring, or None when idle.

    Returns None when every area's pose is the default ``Pose3D()`` — then
    no metadata is written and the historical path runs byte-identically.
    Otherwise returns ``{"order", "origin_x", "poses"}`` with each area's
    column origin (``k * AREA_X_SPACING_MM``) and pose fields, so
    :func:`jaxfne.construct` can pose a wiring copy of the sampled
    positions before edge compilation.
    """
    from ._construct_population import AREA_X_SPACING_MM

    if all(a.pose == Pose3D() for a in tensor.areas):
        return None
    return {
        "order": [a.name for a in tensor.areas],
        "origin_x": {
            a.name: float(k * AREA_X_SPACING_MM) for k, a in enumerate(tensor.areas)
        },
        "poses": {
            a.name: {
                "plane": a.pose.plane,
                "rotation_deg": float(a.pose.rotation_deg),
                "translation": [float(v) for v in a.pose.translation],
            }
            for a in tensor.areas
        },
    }


def _posed_wiring_positions(
    positions: "jax.Array", area_labels: Sequence[str], pose_spec: dict
) -> "jax.Array":
    """Pose a wiring copy of constructed positions (one geometry, pre-compile).

    Applies the same rule as :func:`_pose_positions_and_rows`
    element-for-element (the same :func:`_apply_pose` call on
    ``chunk - origin`` per area) but reads the plain-data ``pose_spec``
    with per-neuron ``area_labels``, so :func:`jaxfne.construct` can sample
    localized edges from the posed layout while the stored positions are
    posed by the same rule afterwards. ``positions`` itself is never
    mutated; default-valued poses pass their area through untouched.
    """
    import numpy as _np

    order = [str(name) for name in pose_spec["order"]]
    origins = dict(pose_spec["origin_x"])
    poses = dict(pose_spec["poses"])
    out = _np.asarray(positions).copy()
    for area_name in order:
        spec = poses[area_name]
        pose = Pose3D(
            plane=spec["plane"],
            rotation_deg=float(spec["rotation_deg"]),
            translation=tuple(float(v) for v in spec["translation"]),
        )
        if pose == Pose3D():
            continue
        idx = [i for i, name in enumerate(area_labels) if name == area_name]
        if not idx:
            continue
        chunk = jnp.asarray(out[idx], dtype=positions.dtype) - jnp.asarray(
            [float(origins[area_name]), 0.0, 0.0], dtype=positions.dtype
        )
        out[idx] = _np.asarray(_apply_pose(chunk, pose))
    return jnp.asarray(out, dtype=positions.dtype)


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
    # T-25: a bound trained W0 is per-edge data this bridge cannot carry —
    # bridging it would silently run declared W0. Refuse with the route that
    # verifies and realizes the trained vector exactly.
    if tensor.trained_w0 is not None:
        raise ValueError(
            f"tensor {tensor.name!r} carries trained W0 (origin='trained'): "
            "neuronal_tensor_to_configuration builds the declared circuit only "
            "and would silently run declared W0 — build it with "
            "construct(tensor) (the RuntimeConfiguration path), which verifies "
            "the stored edges/signs and realizes the trained vector exactly"
        )
    _refuse_mixed_pose_specification(tensor)
    if not _skip_emitter_resolution:
        # The current-based (izhikevich) bridge is the only non-skip caller:
        # the lif bridge and the adex harvest pass _skip_emitter_resolution
        # (lif executes both keys; the harvest tensor carries no connections).
        for _area in tensor.areas:
            for _conn in list(_area.inter_connections):
                _lbl = _connection_label(_conn, _area.name, _area.name)
                if getattr(_conn, "weight_sharing", None) is not None:
                    raise ValueError(
                        f"{_lbl} declares weight_sharing={getattr(_conn, 'weight_sharing')!r}; "
                        "the current-based (izhikevich) route has no shared-weight consumer "
                        "(only None is admitted) — weight sharing executes on the lif "
                        "route only; leave weight_sharing undeclared there"
                    )
                if getattr(_conn, "local_inhibition", None) is not None:
                    raise ValueError(
                        f"{_lbl} declares local_inhibition={getattr(_conn, 'local_inhibition')!r}; "
                        "the current-based (izhikevich) route has no intra-step fixed-point "
                        "consumer (only None is admitted) — local inhibition executes on "
                        "the lif route only; leave local_inhibition undeclared there"
                    )
        for _conn in list(tensor.area_connections):
            _lbl = _connection_label(_conn, _conn.source_area, _conn.target_area)
            if getattr(_conn, "weight_sharing", None) is not None:
                raise ValueError(
                    f"{_lbl} declares weight_sharing={getattr(_conn, 'weight_sharing')!r}; "
                    "the current-based (izhikevich) route has no shared-weight consumer "
                    "(only None is admitted) — weight sharing executes on the lif "
                    "route only; leave weight_sharing undeclared there"
                )
            if getattr(_conn, "local_inhibition", None) is not None:
                raise ValueError(
                    f"{_lbl} declares local_inhibition={getattr(_conn, 'local_inhibition')!r}; "
                    "the current-based (izhikevich) route has no intra-step fixed-point "
                    "consumer (only None is admitted) — local inhibition executes on "
                    "the lif route only; leave local_inhibition undeclared there"
                )

    if _skip_emitter_resolution:
        family = emitter
    else:
        family, _ = _resolve_tensor_emitter_family(tensor, emitter)
        if family == "lif":
            raise ValueError(
                "tensor resolves to emitter 'lif'; the PCL LIF route is "
                "built by construct_neuronal_tensor, not this "
                "current-based bridge"
            )
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
    pose_spec = _tensor_pose_spec(tensor)
    if pose_spec is not None:
        # Posed wiring: construct() poses a copy of the sampled positions
        # before edge compilation, so localized (max_in_degree) sampling
        # sees the posed layout. All-default poses write nothing, so the
        # historical path stays byte-identical.
        cfg = cfg.update_metadata(tensor_pose=pose_spec)
    fallback_weight: dict[str, float] = {}
    area_n_by_name: dict[str, int] = {}
    for area in tensor.areas:
        for layer in area.layers:
            if layer.n_neurons <= 0:
                raise ValueError(
                    f"Area {area.name!r} Layer {layer.name!r}: n_neurons must be "
                    f"positive, got {layer.n_neurons!r}; a zero-neuron layer is "
                    "refused, never promoted to one"
                )
        layer_names = [layer.name for layer in area.layers] or ["single"]
        # `or 1` covers only the layer-less Area (the historical single-neuron
        # default column); declared layers are all positive above.
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
                neurons={layer.name: layer.n_neurons for layer in area.layers},
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

    # Per-(area, layer, type) drive declarations (NeuronType.drive): recorded
    # as data for the population builder, which applies them after
    # drive(baseline_drive_by_cell_type=) and before cell_params(). None
    # anywhere (the common case) writes nothing, so undeclared tensors take
    # the historical path byte-identically.
    triple_drives: list[dict[str, object]] = []
    for area in tensor.areas:
        for layer in area.layers:
            for nt in layer.neuron_types:
                if nt.drive is not None:
                    triple_drives.append(
                        {
                            "area": area.name,
                            "layer": layer.name,
                            "cell_type": nt.name,
                            "drive": float(nt.drive),
                        }
                    )
    if triple_drives:
        cfg = cfg.update_metadata(tensor_triple_drive=triple_drives)

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
    else:  # unreachable: the resolver admits routable families only, minus
        # lif/adex refused above (a Configuration route exists for neither)
        raise ValueError(
            f"Unknown emitter: {family}. Choose from: {sorted(ROUTABLE_TENSOR_EMITTER_FAMILIES)}"
        )

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
    metadata — each is refused there); declared per-triple drives ride
    along as ``tensor_triple_drive`` metadata for
    :func:`jaxfne._construct_adex.construct_adex_model`, so
    :func:`jaxfne.construct` routes it to ``construct_adex_model`` unchanged.
    """
    from .emitters_adex import _DECAYS as _ADEX_DECAYS
    from .emitters_adex import _STEP_ORDERS as _ADEX_STEP_ORDERS

    if not tensor.areas:
        raise ValueError("NeuronalTensor must declare at least one area to bridge")
    # T-25: same as neuronal_tensor_to_configuration — this bridge cannot carry
    # a bound trained W0 and would silently run declared W0.
    if tensor.trained_w0 is not None:
        raise ValueError(
            f"tensor {tensor.name!r} carries trained W0 (origin='trained'): "
            "neuronal_tensor_to_adex_configuration builds the declared circuit only "
            "and would silently run declared W0; trained W0 is realized on the "
            "izhikevich route through construct(tensor)"
        )
    _refuse_mixed_pose_specification(tensor)
    if tensor.connectivity_mode == "unspecified":
        raise ValueError(
            "tensor connectivity_mode is 'unspecified' (no declared "
            "connections); the conductance (adex) route wires only declared "
            "connections — one receptor rule per connection, each needing a "
            "delay — so it would silently build 0 edges, while the "
            "current-based route builds the default topology there. Declare "
            "inter_connections/area_connections (or connectivity_mode="
            "'explicit') for the adex route"
        )
    family, _ = _resolve_tensor_emitter_family(tensor, "adex")
    if family != "adex":  # unreachable: agreement forces "adex" here
        raise ValueError(f"tensor resolves to emitter {family!r}, not 'adex'")
    # Per-(area, layer, type) drive declarations (NeuronType.drive): recorded
    # as data, one entry per declaring triple, for construct_adex_model,
    # which realizes each entry as constant external current on the
    # triple's neurons. None anywhere writes nothing, so undeclared tensors
    # take the historical path unchanged.
    triple_drives: list[dict[str, object]] = []
    for area in tensor.areas:
        for layer in area.layers:
            for ntype in layer.neuron_types:
                if ntype.drive is not None:
                    triple_drives.append(
                        {
                            "area": area.name,
                            "layer": layer.name,
                            "cell_type": ntype.name,
                            "drive": float(ntype.drive),
                        }
                    )
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
        # Standalone harvest, identical to the construct path
        # (:func:`_construct_adex_tensor_impl`): the harvest strips the
        # tensor's connections and aliases non-Izhikevich cell types, so the
        # population sampler draws the same counts — then labels are restored.
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
        from ._construct_population import _neuron_population_from_config

        _, positions, geometry_meta, _ = _neuron_population_from_config(
            harvest_cfg, dtype=dtype
        )
        rows = geometry_meta["neuron_rows"]
        if restore:
            rows = [
                dict(
                    row,
                    cell_type=restore.get(str(row["cell_type"]), str(row["cell_type"])),
                )
                for row in rows
            ]
        # Pose as the construct path does, so velocity delays read A's geometry.
        positions, rows = _pose_positions_and_rows(
            positions, rows, tensor
        )
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
    if triple_drives:
        cfg = cfg.update_metadata(tensor_triple_drive=triple_drives)

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
            # Selected pairs mirror the compiler's distinct-pair sampling
            # (self pairs dropped from the representative delay even when a
            # rule opts into allow_self_connections: the median describes
            # inter-neuron distance, never a zero self-distance).
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
