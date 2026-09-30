"""Controlled model augmentation: exact clone, cardinality (N), geometry (G).

One canonical TFNE/JDNA specification A is developed by JDNA into a base
:class:`NeuronalTensor`; this module applies typed, independently selectable
transforms to that tensor *before* ``construct``::

    base spec -> augmentation spec -> JDNA completion -> NeuronalTensor
        -> augment -> NeuronalTensor -> construct -> Model -> simulate

Transforms never mutate a ``Model``, never add a simulation path, and never
extend the TFNE grammar. Packet 1 implements the skeleton plus the ``N``
and ``G`` primitives; packet AUG-2a implements the ``W_0`` primitive and
packet AUG-2b the ``Theta_C`` / ``Theta_X`` / ``H_0`` primitives, all four
per-connection axes following the same design (shared addresses, targets,
sampling under ``K_V``, provenance origins).

Owner rulings (2026-09-30): transforms apply in the fixed canonical order
``N -> G -> Theta_C -> Theta_X -> W_0 -> H_0`` whatever order the caller
lists them in (R1); they act on a ``NeuronalTensor`` after JDNA ``develop``
and before ``construct`` (R2); scaling keeps the existing ``w/sqrt(N)``
edge-weight rule (R3); a stochastic transform without its own explicit
``K_V`` seed is refused (R4).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional, Sequence

import numpy as np

from ._config import _check_cell_type_fractions, _counts_from_fractions
from .jdna.completion import ORIGIN_AUGMENT_SAMPLED, ORIGIN_AUGMENTED
from .neuronal_tensor import (
    NeuronalTensor,
    _tensor_identity_digest,
)

#: Fixed canonical application order (R1): N -> G -> Theta_C -> Theta_X -> W_0 -> H_0.
CANONICAL_ORDER: tuple[str, ...] = ("N", "G", "Theta_C", "Theta_X", "W_0", "H_0")

#: Packet-2 axes, all implemented since AUG-2b; the name stays (empty) so
#: imports that reference it do not break.
PACKET2_AXES = frozenset()

_KNOWN_AXES = frozenset(CANONICAL_ORDER)

#: Scaling keeps this rule; the record states it (R3).
_SCALING_NOTE = (
    "cardinality scaling keeps the existing w/sqrt(N) edge-weight rule "
    "(neuronal_tensor._connection_edge_weight)"
)


def _check_scale_factor(factor: Any) -> float:
    if isinstance(factor, bool) or not isinstance(factor, numbers.Real):
        raise ValueError(f"ScaleN factor must be a positive number; got {factor!r}")
    value = float(factor)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"ScaleN factor must be finite and > 0; got {factor!r}")
    return value


@dataclass(frozen=True)
class ScaleN:
    """Cardinality scaling: multiply every ``Layer.n_neurons`` by ``factor``."""

    factor: float

    axis: ClassVar[str] = "N"
    stochastic: ClassVar[bool] = False

    def __post_init__(self) -> None:
        # Validate and normalize at construction so ScaleN(10) and
        # ScaleN(10.0) carry the same stored value and digest; bools,
        # non-reals, non-finite and non-positive factors are refused here.
        object.__setattr__(self, "factor", _check_scale_factor(self.factor))


@dataclass(frozen=True)
class PoseEdit:
    """One area-pose edit: additive translation/rotation deltas, uniform
    translation scale. ``translation`` adds ``(dx, dy, dz)`` to the area's
    ``Pose3D.translation``; ``rotation_deg`` adds degrees about the depth
    axis; ``translation_scale`` multiplies the translation first.

    Within one edit the operations apply in this fixed order: scale, then
    rotation, then translation — i.e. ``(t * translation_scale) +
    translation`` for the stored translation, with the rotation delta added
    to the stored ``rotation_deg`` in between (rotation and translation
    commute on the stored pose; both are additive on independent fields,
    so the observable order is scale-before-translation)."""

    area: str
    translation: Optional[tuple[float, float, float]] = None
    rotation_deg: Optional[float] = None
    translation_scale: Optional[float] = None

    axis: ClassVar[str] = "G"

    def __post_init__(self) -> None:
        if not isinstance(self.area, str) or not self.area:
            raise ValueError(f"PoseEdit requires a non-empty area name; got {self.area!r}")
        if self.translation is not None:
            object.__setattr__(self, "translation", _check_vector3(self.translation, "PoseEdit.translation"))
        if self.rotation_deg is not None:
            object.__setattr__(self, "rotation_deg", _check_finite(self.rotation_deg, "PoseEdit.rotation_deg"))
        if self.translation_scale is not None:
            object.__setattr__(
                self, "translation_scale", _check_finite(self.translation_scale, "PoseEdit.translation_scale")
            )


@dataclass(frozen=True)
class RangeEdit:
    """One layer-geometry edit: absolute replacement of column-relative
    ranges (each within ``[0, 1]`` with ``lo <= hi``)."""

    area: str
    layer: str
    x_range: Optional[tuple[float, float]] = None
    y_range: Optional[tuple[float, float]] = None
    z_range: Optional[tuple[float, float]] = None

    axis: ClassVar[str] = "G"

    def __post_init__(self) -> None:
        if not isinstance(self.area, str) or not self.area:
            raise ValueError(f"RangeEdit requires a non-empty area name; got {self.area!r}")
        if not isinstance(self.layer, str) or not self.layer:
            raise ValueError(f"RangeEdit requires a non-empty layer name; got {self.layer!r}")
        for label in ("x_range", "y_range", "z_range"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, _check_unit_range(value, f"RangeEdit.{label}"))


@dataclass(frozen=True)
class GeometryTransform:
    """Geometry (G) record: pose-level affine edits plus range-level edits."""

    pose_edits: Sequence[PoseEdit] = ()
    range_edits: Sequence[RangeEdit] = ()

    axis: ClassVar[str] = "G"
    stochastic: ClassVar[bool] = False

    def __post_init__(self) -> None:
        poses = tuple(self.pose_edits)
        ranges = tuple(self.range_edits)
        if any(not isinstance(edit, PoseEdit) for edit in poses):
            raise ValueError("GeometryTransform.pose_edits must all be PoseEdit records")
        if any(not isinstance(edit, RangeEdit) for edit in ranges):
            raise ValueError("GeometryTransform.range_edits must all be RangeEdit records")
        seen_areas = set()
        for edit in poses:
            if edit.area in seen_areas:
                raise ValueError(
                    f"ambiguous GeometryTransform: two PoseEdits address area {edit.area!r}; "
                    "merging them would be order-dependent, so this is refused — "
                    "combine them into one PoseEdit"
                )
            seen_areas.add(edit.area)
        seen_layers = set()
        for edit in ranges:
            key = (edit.area, edit.layer)
            if key in seen_layers:
                raise ValueError(
                    f"ambiguous GeometryTransform: two RangeEdits address layer {edit.layer!r} "
                    f"in area {edit.area!r}; merging them would be order-dependent, "
                    "so this is refused — combine them into one RangeEdit"
                )
            seen_layers.add(key)
        object.__setattr__(self, "pose_edits", poses)
        object.__setattr__(self, "range_edits", ranges)


@dataclass(frozen=True)
class ThetaC:
    """Connectivity/rule-parameter variation: scale ``delay_ms`` (every
    connection kind) and ``probability`` (``AreaConnection`` only).

    ``delay_factor`` / ``probability_factor`` are deterministic multipliers
    (finite, > 0); ``delay_jitter`` / ``probability_jitter`` are the
    relative half-widths of per-field stochastic variation drawn uniform in
    ``[1 - jitter, 1 + jitter]`` under ``K_V`` (``0 <= jitter < 1``), so
    either scalar can be varied alone. ``targets`` selects connection
    addresses (``()`` means all connections). ``stochastic`` is a property,
    not a field, so the spec digest covers exactly the factors, jitters
    and ``targets``.
    """

    delay_factor: float = 1.0
    delay_jitter: float = 0.0
    probability_factor: float = 1.0
    probability_jitter: float = 0.0
    targets: tuple[str, ...] = ()

    axis: ClassVar[str] = "Theta_C"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "delay_factor", _check_gain_factor(self.delay_factor, "ThetaC delay_factor")
        )
        object.__setattr__(
            self, "delay_jitter", _check_scale_jitter(self.delay_jitter, "ThetaC delay_jitter")
        )
        object.__setattr__(
            self,
            "probability_factor",
            _check_gain_factor(self.probability_factor, "ThetaC probability_factor"),
        )
        object.__setattr__(
            self,
            "probability_jitter",
            _check_scale_jitter(self.probability_jitter, "ThetaC probability_jitter"),
        )
        object.__setattr__(self, "targets", _check_connection_targets(self.targets, "ThetaC"))

    @property
    def stochastic(self) -> bool:
        return self.delay_jitter > 0.0 or self.probability_jitter > 0.0


@dataclass(frozen=True)
class ThetaX:
    """Dynamical-parameter variation: scale every value in
    ``static.g_mech`` (dict mechanism -> conductance) and ``static.dT_ms``.

    ``g_factor`` / ``tau_factor`` are deterministic multipliers (finite,
    > 0); ``g_jitter`` / ``tau_jitter`` are the relative half-widths of
    per-field stochastic variation drawn uniform in ``[1 - jitter,
    1 + jitter]`` under ``K_V`` (``0 <= jitter < 1``), so conductances and
    the time constant can be varied alone. ``targets`` selects connection
    addresses (``()`` means all connections). ``stochastic`` is a property,
    not a field, so the spec digest covers exactly the factors, jitters
    and ``targets``.
    """

    g_factor: float = 1.0
    g_jitter: float = 0.0
    tau_factor: float = 1.0
    tau_jitter: float = 0.0
    targets: tuple[str, ...] = ()

    axis: ClassVar[str] = "Theta_X"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "g_factor", _check_gain_factor(self.g_factor, "ThetaX g_factor")
        )
        object.__setattr__(
            self, "g_jitter", _check_scale_jitter(self.g_jitter, "ThetaX g_jitter")
        )
        object.__setattr__(
            self, "tau_factor", _check_gain_factor(self.tau_factor, "ThetaX tau_factor")
        )
        object.__setattr__(
            self, "tau_jitter", _check_scale_jitter(self.tau_jitter, "ThetaX tau_jitter")
        )
        object.__setattr__(self, "targets", _check_connection_targets(self.targets, "ThetaX"))

    @property
    def stochastic(self) -> bool:
        return self.g_jitter > 0.0 or self.tau_jitter > 0.0


@dataclass(frozen=True)
class W0:
    """Initial-gain variation: multiply ``PlasticParams.w_mech`` per connection.

    ``factor`` is a deterministic multiplier (finite, > 0); ``jitter`` is the
    relative half-width of per-connection stochastic variation drawn uniform
    in ``[1 - jitter, 1 + jitter]`` under ``K_V`` (``0 <= jitter < 1``);
    ``targets`` selects connection addresses (``()`` means all connections).
    ``stochastic`` is a property, not a field, so the spec digest covers
    exactly ``factor``, ``jitter`` and ``targets``.
    """

    factor: float = 1.0
    jitter: float = 0.0
    targets: tuple[str, ...] = ()

    axis: ClassVar[str] = "W_0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", _check_w0_factor(self.factor))
        object.__setattr__(self, "jitter", _check_w0_jitter(self.jitter))
        object.__setattr__(self, "targets", _check_w0_targets(self.targets))

    @property
    def stochastic(self) -> bool:
        return self.jitter > 0.0


@dataclass(frozen=True)
class H0:
    """Initial-hidden-state variation: shift ``plastic.H`` per connection.

    ``H`` is a signed relative hidden state, not a gain, so the shift is
    additive: ``offset`` is a deterministic shift (finite, any sign) and
    ``jitter`` is the absolute half-width of per-connection stochastic
    variation drawn uniform in ``[-jitter, +jitter]`` under ``K_V``
    (finite, >= 0, no upper bound); ``targets`` selects connection
    addresses (``()`` means all connections). ``stochastic`` is a property,
    not a field, so the spec digest covers exactly ``offset``, ``jitter``
    and ``targets``.
    """

    offset: float = 0.0
    jitter: float = 0.0
    targets: tuple[str, ...] = ()

    axis: ClassVar[str] = "H_0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", _check_offset(self.offset, "H0 offset"))
        object.__setattr__(self, "jitter", _check_abs_jitter(self.jitter, "H0 jitter"))
        object.__setattr__(self, "targets", _check_connection_targets(self.targets, "H0"))

    @property
    def stochastic(self) -> bool:
        return self.jitter > 0.0


@dataclass(frozen=True)
class ProvenanceEntry:
    """One changed value: address, axis, before/after, origin ``"augmented"``."""

    address: str
    axis: str
    before: Any
    after: Any
    origin: str = ORIGIN_AUGMENTED


@dataclass(frozen=True)
class AugmentationRecord:
    """What ``augment`` realized: order, per-value provenance, digests."""

    realized_order: tuple[str, ...] = ()
    changes: tuple[ProvenanceEntry, ...] = ()
    spec_digest: str = ""
    base_digest: str = ""
    notes: tuple[str, ...] = ()


def _check_finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number; got {value!r}")
    return float(value)


def _check_vector3(value: Any, where: str) -> tuple[float, float, float]:
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{where} must be a 3-tuple of finite numbers; got {value!r}")
    if len(items) != 3:
        raise ValueError(f"{where} must be a 3-tuple of finite numbers; got {value!r}")
    return (
        _check_finite(items[0], where),
        _check_finite(items[1], where),
        _check_finite(items[2], where),
    )


def _check_unit_range(value: Any, where: str) -> tuple[float, float]:
    try:
        lo_raw, hi_raw = tuple(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be an (lo, hi) pair within [0, 1]; got {value!r}")
    lo, hi = _check_finite(lo_raw, where), _check_finite(hi_raw, where)
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"{where} must satisfy 0 <= lo <= hi <= 1; got {(lo, hi)!r}")
    return (lo, hi)


def _check_gain_factor(value: Any, label: str) -> float:
    """Shared deterministic-multiplier check (finite, > 0) for every axis."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a positive number; got {value!r}")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{label} must be finite and > 0; got {value!r}")
    return checked


def _check_scale_jitter(value: Any, label: str) -> float:
    """Shared stochastic half-width check (``0 <= jitter < 1``) for every axis."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a number in [0, 1); got {value!r}")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked < 1.0:
        raise ValueError(f"{label} must satisfy 0 <= jitter < 1; got {value!r}")
    return checked


def _check_connection_targets(targets: Any, label: str) -> tuple[str, ...]:
    """Shared connection-address selection check for every axis."""
    if isinstance(targets, str):
        items = (targets,)
    else:
        try:
            items = tuple(targets)
        except TypeError:
            raise ValueError(
                f"{label} targets must be a sequence of connection addresses; got {targets!r}"
            )
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{label} targets must be non-empty address strings; got {item!r}"
            )
    seen: set[str] = set()
    for item in items:
        if item in seen:
            raise ValueError(
                f"duplicate {label} target address {item!r}; list each connection once"
            )
        seen.add(item)
    return items


def _check_offset(value: Any, label: str) -> float:
    """Shared additive-shift check (finite, any sign) for signed states."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a finite number; got {value!r}")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{label} must be finite; got {value!r}")
    return checked


def _check_abs_jitter(value: Any, label: str) -> float:
    """Shared absolute half-width check (finite, >= 0, no upper bound)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a number >= 0; got {value!r}")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0.0:
        raise ValueError(f"{label} must be finite and >= 0; got {value!r}")
    return checked


def _check_w0_factor(factor: Any) -> float:
    return _check_gain_factor(factor, "W0 factor")


def _check_w0_jitter(jitter: Any) -> float:
    return _check_scale_jitter(jitter, "W0 jitter")


def _check_w0_targets(targets: Any) -> tuple[str, ...]:
    return _check_connection_targets(targets, "W0")


def _plain(value: Any) -> Any:
    """JSON-stable plain-data view of a transform record (axis included)."""
    if isinstance(value, (PoseEdit, RangeEdit)):
        out = {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
        out["axis"] = value.axis
        return out
    if hasattr(value, "__dataclass_fields__"):
        out = {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
        out["axis"] = value.axis
        return out
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


#: Expected record type per axis: a bare sub-record (e.g. a ``PoseEdit``
#: where a ``GeometryTransform`` is expected) is refused by the validator.
_AXIS_RECORD_TYPES: dict[str, tuple[type, ...]] = {
    "N": (ScaleN,),
    "G": (GeometryTransform,),
    "Theta_C": (ThetaC,),
    "Theta_X": (ThetaX,),
    "W_0": (W0,),
    "H_0": (H0,),
}


@dataclass(frozen=True)
class AugmentationSpec:
    """Typed augmentation spec: at most one record per axis, optional ``K_V``.

    A stochastic transform without ``K_V`` is refused (R4); ``K_V`` is
    independent of the JDNA development seed ``K_D`` and the simulation
    seed ``K_S``.
    """

    transforms: Sequence[Any] = ()
    k_v: Optional[int] = None

    def __post_init__(self) -> None:
        raw = self.transforms
        if raw is None:
            items: tuple[Any, ...] = ()
        elif hasattr(raw, "axis"):
            items = (raw,)
        else:
            items = tuple(raw)
        seen: set[str] = set()
        for record in items:
            axis = getattr(record, "axis", None)
            if axis not in _KNOWN_AXES:
                raise ValueError(
                    f"unknown augmentation record {record!r}; one record per axis "
                    f"in {sorted(_KNOWN_AXES)}"
                )
            expected = _AXIS_RECORD_TYPES[axis]
            if not isinstance(record, expected):
                names = ", ".join(cls.__name__ for cls in expected)
                raise ValueError(
                    f"axis {axis!r} expects a {names} record; "
                    f"got {type(record).__name__} ({record!r})"
                )
            if axis in seen:
                raise ValueError(f"duplicate augmentation record for axis {axis!r}")
            seen.add(axis)
        if self.k_v is not None:
            if isinstance(self.k_v, bool) or not isinstance(self.k_v, numbers.Integral):
                raise ValueError(f"AugmentationSpec k_v must be an int seed; got {self.k_v!r}")
        if self.k_v is None and any(bool(getattr(record, "stochastic", False)) for record in items):
            raise ValueError("a stochastic transform requires its own explicit seed K_V (R4)")
        object.__setattr__(self, "transforms", items)

    def digest(self) -> str:
        """Stable digest of this spec (canonical axis order, sorted keys)."""
        by_axis = {record.axis: _plain(record) for record in self.transforms}
        payload = {
            "k_v": None if self.k_v is None else int(self.k_v),
            "transforms": {axis: by_axis[axis] for axis in CANONICAL_ORDER if axis in by_axis},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _scaled_count(n: int, factor: float) -> int:
    exact = float(n) * factor
    nearest = round(exact)
    if nearest < 1 or abs(exact - nearest) > 1e-9 * max(1.0, abs(exact)):
        raise ValueError(
            f"ScaleN factor {factor!r} maps n_neurons={n} to {exact!r}, "
            "not a positive integer; refusing (no silent rounding)"
        )
    return int(nearest)


def _layer_fractions(layer: Any) -> dict[str, float]:
    """Per-type fractions exactly as the Configuration bridge reads them."""
    neuron_types = list(layer.neuron_types) or []
    if neuron_types and all(nt is not None and nt.fraction is not None for nt in neuron_types):
        raw = {nt.name: float(nt.fraction) for nt in neuron_types}
        total = sum(raw.values()) or 1.0
        return {name: value / total for name, value in raw.items()}
    names = [nt.name for nt in neuron_types] or ["E"]
    even = 1.0 / len(names)
    return {name: even for name in names}


def _apply_scale_n(tensor: NeuronalTensor, record: ScaleN, changes: list[ProvenanceEntry]) -> None:
    factor = _check_scale_factor(record.factor)
    if factor == 1.0:
        return  # no-op: no scaling, no allocation check, no record entry
    for area in tensor.areas:
        for layer in area.layers:
            new_n = _scaled_count(int(layer.n_neurons), factor)
            if new_n == layer.n_neurons:
                continue
            # Reuse the existing allocation checks: whatever the bridge would
            # refuse for these fractions is refused here, not re-implemented.
            fracs = _check_cell_type_fractions(_layer_fractions(layer))
            counts = _counts_from_fractions(new_n, fracs)
            for name, frac in fracs.items():
                if frac > 0.0 and counts.get(name, 0) == 0:
                    raise ValueError(
                        f"ScaleN factor {factor!r} realizes 0 neurons for cell type "
                        f"{name!r} with fraction {frac!r} in layer {layer.name!r} "
                        f"of area {area.name!r} (n_neurons {layer.n_neurons} -> {new_n}); "
                        "refusing (a declared type must not vanish under scaling)"
                    )
            changes.append(
                ProvenanceEntry(
                    address=f"areas.{area.name}.layers.{layer.name}.n_neurons",
                    axis="N",
                    before=int(layer.n_neurons),
                    after=new_n,
                )
            )
            layer.n_neurons = new_n


def _find_area(tensor: NeuronalTensor, area_name: str) -> Any:
    for area in tensor.areas:
        if area.name == area_name:
            return area
    raise ValueError(f"unknown area {area_name!r} in augmentation address")


def _apply_geometry(
    tensor: NeuronalTensor, record: GeometryTransform, changes: list[ProvenanceEntry]
) -> None:
    for edit in record.pose_edits:
        area = _find_area(tensor, edit.area)
        pose = area.pose
        if edit.translation_scale is not None:
            scaled = tuple(v * edit.translation_scale for v in pose.translation)
            if tuple(scaled) != tuple(pose.translation):
                changes.append(
                    ProvenanceEntry(
                        address=f"areas.{area.name}.pose.translation",
                        axis="G",
                        before=tuple(pose.translation),
                        after=tuple(scaled),
                    )
                )
                pose.translation = tuple(scaled)
        if edit.rotation_deg is not None:
            rotated = float(pose.rotation_deg) + float(edit.rotation_deg)
            if rotated != float(pose.rotation_deg):
                changes.append(
                    ProvenanceEntry(
                        address=f"areas.{area.name}.pose.rotation_deg",
                        axis="G",
                        before=float(pose.rotation_deg),
                        after=rotated,
                    )
                )
                pose.rotation_deg = rotated
        if edit.translation is not None:
            shifted = tuple(v + d for v, d in zip(pose.translation, edit.translation))
            if tuple(shifted) != tuple(pose.translation):
                changes.append(
                    ProvenanceEntry(
                        address=f"areas.{area.name}.pose.translation",
                        axis="G",
                        before=tuple(pose.translation),
                        after=tuple(shifted),
                    )
                )
                pose.translation = tuple(shifted)
    for edit in record.range_edits:
        area = _find_area(tensor, edit.area)
        layer = next((cand for cand in area.layers if cand.name == edit.layer), None)
        if layer is None:
            raise ValueError(
                f"unknown layer {edit.layer!r} in area {edit.area!r} in augmentation address"
            )
        for label in ("x_range", "y_range", "z_range"):
            value = getattr(edit, label)
            if value is not None and tuple(value) != tuple(getattr(layer.geometry, label)):
                changes.append(
                    ProvenanceEntry(
                        address=f"areas.{area.name}.layers.{layer.name}.geometry.{label}",
                        axis="G",
                        before=tuple(getattr(layer.geometry, label)),
                        after=tuple(value),
                    )
                )
                setattr(layer.geometry, label, tuple(value))


def _connection_address_table(
    tensor: NeuronalTensor,
) -> dict[str, tuple[str, int, int]]:
    """Map every connection address to ``(kind, area_idx, conn_idx)``.

    ``kind`` is ``"inter"`` for ``areas[area_idx].inter_connections[conn_idx]``
    or ``"cross"`` for ``area_connections[conn_idx]`` (``area_idx`` is ``-1``).
    Enumeration order is the sampling order: inter connections by area order
    then index, then area connections by index. Shared by all four
    per-connection axes (``Theta_C``, ``Theta_X``, ``W_0``, ``H_0``).
    """
    table: dict[str, tuple[str, int, int]] = {}
    for area_idx, area in enumerate(tensor.areas):
        for conn_idx in range(len(area.inter_connections)):
            table[f"{area.name}/inter/{conn_idx}"] = ("inter", area_idx, conn_idx)
    for conn_idx in range(len(tensor.area_connections)):
        table[f"area_connection/{conn_idx}"] = ("cross", -1, conn_idx)
    return table


#: Original name kept so existing references keep working.
_w0_address_table = _connection_address_table


def _resolve_selected(
    table: dict[str, tuple[str, int, int]],
    targets: tuple[str, ...],
    axis_token: str,
    *,
    in_address_order: bool = False,
) -> list[str]:
    """Validate ``targets`` against ``table``; ``()`` means all addresses.

    ``W_0`` keeps target-list order; the newer axes iterate in address
    (sampling) order whatever order the caller lists.
    """
    if targets:
        for target in targets:
            if target not in table:
                raise ValueError(
                    f"unknown {axis_token} connection address {target!r}; "
                    f"known addresses: {sorted(table)}"
                )
        if in_address_order:
            wanted = set(targets)
            return [address for address in table if address in wanted]
        return list(targets)
    return list(table)


def _stochastic_rng(stochastic: bool, k_v: Optional[int], axis_token: str) -> Any:
    """Return ``numpy.random.default_rng(k_v)`` when the record draws, else None.

    A stochastic record without its own explicit ``K_V`` seed is refused (R4).
    """
    if not stochastic:
        return None
    if k_v is None:  # fail closed; the spec validator normally refuses this first
        raise ValueError(
            f"a stochastic {axis_token} transform requires its own explicit seed K_V (R4)"
        )
    return np.random.default_rng(k_v)


def _draw_multiplier(rng: Any, jitter: float) -> float:
    """One uniform draw in ``[1 - jitter, 1 + jitter]``, or 1.0 when deterministic."""
    if rng is None or not jitter > 0.0:
        return 1.0
    return float(rng.uniform(1.0 - jitter, 1.0 + jitter))


def _draw_absolute(rng: Any, jitter: float) -> float:
    """One uniform draw in ``[-jitter, +jitter]``, or 0.0 when deterministic."""
    if rng is None or not jitter > 0.0:
        return 0.0
    return float(rng.uniform(-jitter, jitter))


def _field_origin(jitter: float) -> str:
    """Per-field provenance origin: sampled when the field drew under ``K_V``."""
    return ORIGIN_AUGMENT_SAMPLED if jitter > 0.0 else ORIGIN_AUGMENTED


def _get_connection(
    tensor: NeuronalTensor,
    table: dict[str, tuple[str, int, int]],
    address: str,
) -> tuple[Any, str, int, int]:
    """Return ``(conn, kind, area_idx, conn_idx)`` for a validated address."""
    kind, area_idx, conn_idx = table[address]
    if kind == "inter":
        conn = tensor.areas[area_idx].inter_connections[conn_idx]
    else:
        conn = tensor.area_connections[conn_idx]
    return conn, kind, area_idx, conn_idx


def _stage_rebuilt(
    inter_new: dict[int, dict[int, Any]],
    cross_new: dict[int, Any],
    kind: str,
    area_idx: int,
    conn_idx: int,
    rebuilt: Any,
) -> None:
    """Stage one rebuilt connection for :func:`_flush_rebuilt`."""
    if kind == "inter":
        inter_new.setdefault(area_idx, {})[conn_idx] = rebuilt
    else:
        cross_new[conn_idx] = rebuilt


def _flush_rebuilt(
    tensor: NeuronalTensor,
    inter_new: dict[int, dict[int, Any]],
    cross_new: dict[int, Any],
) -> None:
    """Write staged rebuilt connections back, preserving tuple/list containers."""
    for area_idx, indexed in inter_new.items():
        seq = list(tensor.areas[area_idx].inter_connections)
        for conn_idx, rebuilt in indexed.items():
            seq[conn_idx] = rebuilt
        tensor.areas[area_idx].inter_connections = (
            tuple(seq) if isinstance(tensor.areas[area_idx].inter_connections, tuple) else seq
        )
    if cross_new:
        seq = list(tensor.area_connections)
        for conn_idx, rebuilt in cross_new.items():
            seq[conn_idx] = rebuilt
        tensor.area_connections = (
            tuple(seq) if isinstance(tensor.area_connections, tuple) else seq
        )


def _checked_nonnegative(value: Any, address: str, field: str, stored: Any) -> float:
    """Coerce a stored gain-like value, refusing garbage/negative/non-finite.

    Mirrors the ``W_0`` stored-gain check: only finite values >= 0 rescale.
    """
    try:
        checked = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{address}: stored {field} {stored!r} is negative or not finite"
        )
    if not (math.isfinite(checked) and checked >= 0.0):
        raise ValueError(
            f"{address}: stored {field} {stored!r} is negative or not finite"
        )
    return checked


def _apply_w0(
    tensor: NeuronalTensor,
    record: W0,
    k_v: Optional[int],
    changes: list[ProvenanceEntry],
) -> None:
    if record.factor == 1.0 and record.jitter == 0.0:
        return  # no-op: no scaling, no sampling, no record entry
    table = _connection_address_table(tensor)
    selected = _resolve_selected(table, record.targets, "W_0")
    if not selected:
        return  # no connections: nothing to realize
    rng = _stochastic_rng(record.jitter > 0.0, k_v, "W_0")
    origin = _field_origin(record.jitter)
    inter_new: dict[int, dict[int, Any]] = {}
    cross_new: dict[int, Any] = {}
    for address in selected:
        conn, kind, area_idx, conn_idx = _get_connection(tensor, table, address)
        # Sign is preserved by construction (factor > 0, u > 0); a stored
        # gain that is negative or not finite is refused, not rescaled. A
        # zero gain stays zero (the draw is still taken, so sampling order
        # does not depend on which gains are zero).
        old_w = _checked_nonnegative(
            conn.plastic.w_mech, f"W_0 refuses {address!r}", "w_mech", conn.plastic.w_mech
        )
        u = _draw_multiplier(rng, record.jitter)
        new_w = old_w * record.factor * u
        if not (math.isfinite(new_w) and (new_w > 0.0 or old_w == 0.0)):
            raise ValueError(
                f"W_0 refuses {address!r}: rescaled w_mech {new_w!r} "
                "is not positive-finite"
            )
        if new_w == old_w:
            continue  # unchanged value: no provenance entry, like N and G
        # Rebuild inward-out so frozen dataclasses keep working; the input
        # tensor is untouched because augment deep-copies before dispatch.
        rebuilt = replace(conn, plastic=replace(conn.plastic, w_mech=new_w))
        _stage_rebuilt(inter_new, cross_new, kind, area_idx, conn_idx, rebuilt)
        changes.append(
            ProvenanceEntry(
                address=f"{address}.plastic.w_mech",
                axis="W_0",
                before=old_w,
                after=new_w,
                origin=origin,
            )
        )
    _flush_rebuilt(tensor, inter_new, cross_new)


def _apply_theta_x(
    tensor: NeuronalTensor,
    record: ThetaX,
    k_v: Optional[int],
    changes: list[ProvenanceEntry],
) -> None:
    if (
        record.g_factor == 1.0
        and record.g_jitter == 0.0
        and record.tau_factor == 1.0
        and record.tau_jitter == 0.0
    ):
        return  # no-op: no scaling, no sampling, no record entry
    table = _connection_address_table(tensor)
    selected = _resolve_selected(table, record.targets, "Theta_X", in_address_order=True)
    if not selected:
        return  # no connections: nothing to realize
    rng = _stochastic_rng(
        record.g_jitter > 0.0 or record.tau_jitter > 0.0, k_v, "Theta_X"
    )
    inter_new: dict[int, dict[int, Any]] = {}
    cross_new: dict[int, Any] = {}
    for address in selected:
        conn, kind, area_idx, conn_idx = _get_connection(tensor, table, address)
        # Within a connection, g_mech keys (sorted) sample before dT_ms.
        new_g: Optional[dict] = None
        for mech in sorted(conn.static.g_mech):
            raw = conn.static.g_mech[mech]
            old = _checked_nonnegative(
                raw,
                f"Theta_X refuses {address}.static.g_mech.{mech}",
                "g_mech",
                raw,
            )
            u = _draw_multiplier(rng, record.g_jitter)
            new = old * record.g_factor * u
            if not (math.isfinite(new) and (new > 0.0 or old == 0.0)):
                raise ValueError(
                    f"Theta_X refuses {address}.static.g_mech.{mech}: "
                    f"rescaled conductance {new!r} is not positive-finite"
                )
            if new == old:
                continue  # unchanged value: no provenance entry, like W_0
            if new_g is None:
                new_g = dict(conn.static.g_mech)
            new_g[mech] = new
            changes.append(
                ProvenanceEntry(
                    address=f"{address}.static.g_mech.{mech}",
                    axis="Theta_X",
                    before=old,
                    after=new,
                    origin=_field_origin(record.g_jitter),
                )
            )
        raw_tau = conn.static.dT_ms
        new_tau: Optional[float] = None
        if raw_tau is not None:  # None = undeclared: left untouched, no entry
            old_tau = _checked_nonnegative(
                raw_tau, f"Theta_X refuses {address}.static.dT_ms", "dT_ms", raw_tau
            )
            u = _draw_multiplier(rng, record.tau_jitter)
            candidate = old_tau * record.tau_factor * u
            if not (math.isfinite(candidate) and candidate > 0.0):
                raise ValueError(
                    f"Theta_X refuses {address}.static.dT_ms: "
                    f"rescaled dT_ms {candidate!r} is not finite and > 0"
                )
            if candidate != old_tau:
                new_tau = candidate
                changes.append(
                    ProvenanceEntry(
                        address=f"{address}.static.dT_ms",
                        axis="Theta_X",
                        before=old_tau,
                        after=candidate,
                        origin=_field_origin(record.tau_jitter),
                    )
                )
        if new_g is None and new_tau is None:
            continue  # unchanged connection: no rebuild, no entry
        rebuilt_static = replace(
            conn.static,
            g_mech=(new_g if new_g is not None else conn.static.g_mech),
            dT_ms=(new_tau if new_tau is not None else conn.static.dT_ms),
        )
        _stage_rebuilt(
            inter_new, cross_new, kind, area_idx, conn_idx, replace(conn, static=rebuilt_static)
        )
    _flush_rebuilt(tensor, inter_new, cross_new)


def _apply_theta_c(
    tensor: NeuronalTensor,
    record: ThetaC,
    k_v: Optional[int],
    changes: list[ProvenanceEntry],
) -> None:
    if (
        record.delay_factor == 1.0
        and record.delay_jitter == 0.0
        and record.probability_factor == 1.0
        and record.probability_jitter == 0.0
    ):
        return  # no-op: no scaling, no sampling, no record entry
    table = _connection_address_table(tensor)
    selected = _resolve_selected(table, record.targets, "Theta_C", in_address_order=True)
    if not selected:
        return  # no connections: nothing to realize
    rng = _stochastic_rng(
        record.delay_jitter > 0.0 or record.probability_jitter > 0.0, k_v, "Theta_C"
    )
    inter_new: dict[int, dict[int, Any]] = {}
    cross_new: dict[int, Any] = {}
    for address in selected:
        conn, kind, area_idx, conn_idx = _get_connection(tensor, table, address)
        # Within a connection, delay_ms samples before probability.
        raw_delay = conn.delay_ms
        new_delay: Optional[float] = None
        delay_touched = False
        if raw_delay is not None:  # None = undeclared: left untouched, no entry
            old_delay = _checked_nonnegative(
                raw_delay, f"Theta_C refuses {address}.delay_ms", "delay_ms", raw_delay
            )
            u = _draw_multiplier(rng, record.delay_jitter)
            candidate = old_delay * record.delay_factor * u
            if not (math.isfinite(candidate) and candidate >= 0.0):
                raise ValueError(
                    f"Theta_C refuses {address}.delay_ms: "
                    f"rescaled delay_ms {candidate!r} is not finite and >= 0"
                )
            if candidate != old_delay:
                new_delay = candidate
                delay_touched = True
                changes.append(
                    ProvenanceEntry(
                        address=f"{address}.delay_ms",
                        axis="Theta_C",
                        before=old_delay,
                        after=candidate,
                        origin=_field_origin(record.delay_jitter),
                    )
                )
        # Only AreaConnection carries probability; other kinds have no such
        # attribute, so getattr defaulting to None skips them with no entry.
        raw_prob = getattr(conn, "probability", None)
        new_prob: Optional[float] = None
        prob_touched = False
        if raw_prob is not None:  # None = undeclared: left untouched, no entry
            try:
                old_prob = float(raw_prob)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Theta_C refuses {address}.probability: "
                    f"stored probability {raw_prob!r} is not in (0, 1]"
                )
            if not (math.isfinite(old_prob) and 0.0 < old_prob <= 1.0):
                raise ValueError(
                    f"Theta_C refuses {address}.probability: "
                    f"stored probability {raw_prob!r} is not in (0, 1]"
                )
            u = _draw_multiplier(rng, record.probability_jitter)
            candidate = old_prob * record.probability_factor * u
            if candidate > 1.0:
                raise ValueError(
                    f"Theta_C refuses {address}.probability: "
                    f"rescaled probability {candidate!r} is above 1 (never clipped)"
                )
            if not (math.isfinite(candidate) and candidate > 0.0):
                raise ValueError(
                    f"Theta_C refuses {address}.probability: "
                    f"rescaled probability {candidate!r} is not finite and > 0"
                )
            if candidate != old_prob:
                new_prob = candidate
                prob_touched = True
                changes.append(
                    ProvenanceEntry(
                        address=f"{address}.probability",
                        axis="Theta_C",
                        before=old_prob,
                        after=candidate,
                        origin=_field_origin(record.probability_jitter),
                    )
                )
        if not (delay_touched or prob_touched):
            continue  # unchanged connection: no rebuild, no entry
        # Rebuild inward-out; the input tensor is untouched because augment
        # deep-copies before dispatch.
        kwargs: dict[str, Any] = {}
        if delay_touched:
            kwargs["delay_ms"] = new_delay
        if prob_touched:
            kwargs["probability"] = new_prob
        _stage_rebuilt(
            inter_new, cross_new, kind, area_idx, conn_idx, replace(conn, **kwargs)
        )
    _flush_rebuilt(tensor, inter_new, cross_new)


def _apply_h0(
    tensor: NeuronalTensor,
    record: H0,
    k_v: Optional[int],
    changes: list[ProvenanceEntry],
) -> None:
    if record.offset == 0.0 and record.jitter == 0.0:
        return  # no-op: no shift, no sampling, no record entry
    table = _connection_address_table(tensor)
    selected = _resolve_selected(table, record.targets, "H_0", in_address_order=True)
    if not selected:
        return  # no connections: nothing to realize
    rng = _stochastic_rng(record.jitter > 0.0, k_v, "H_0")
    origin = _field_origin(record.jitter)
    inter_new: dict[int, dict[int, Any]] = {}
    cross_new: dict[int, Any] = {}
    for address in selected:
        conn, kind, area_idx, conn_idx = _get_connection(tensor, table, address)
        raw = conn.plastic.H
        if raw is None:  # None = undeclared: left untouched, no entry
            continue
        # H is a signed relative state: any finite stored value (negative
        # included) shifts; only a non-finite stored value is refused.
        try:
            old = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"H_0 refuses {address!r}: stored H {raw!r} is not finite"
            )
        if not math.isfinite(old):
            raise ValueError(
                f"H_0 refuses {address!r}: stored H {raw!r} is not finite"
            )
        u = _draw_absolute(rng, record.jitter)
        new = old + record.offset + u
        if not math.isfinite(new):
            raise ValueError(
                f"H_0 refuses {address!r}: rescaled H {new!r} "
                "is not finite"
            )
        if new == old:
            continue  # unchanged value: no provenance entry, like W_0
        # Rebuild inward-out so frozen dataclasses keep working; the input
        # tensor is untouched because augment deep-copies before dispatch.
        rebuilt = replace(conn, plastic=replace(conn.plastic, H=new))
        _stage_rebuilt(inter_new, cross_new, kind, area_idx, conn_idx, rebuilt)
        changes.append(
            ProvenanceEntry(
                address=f"{address}.plastic.H",
                axis="H_0",
                before=old,
                after=new,
                origin=origin,
            )
        )
    _flush_rebuilt(tensor, inter_new, cross_new)


def augment(
    tensor: NeuronalTensor, spec: AugmentationSpec
) -> tuple[NeuronalTensor, AugmentationRecord]:
    """Apply ``spec`` to ``tensor`` in canonical order; return ``(new, record)``.

    The input tensor is never touched: the result is a deep copy with only
    the targeted values rewritten. A transform that changes no value is a
    no-op: it contributes no ``realized_order`` entry (and a no-op ``ScaleN``
    writes no scaling note). Transforms apply in canonical order
    ``N -> G -> Theta_C -> Theta_X -> W_0 -> H_0`` whatever order the caller
    lists them in.
    """
    if not isinstance(tensor, NeuronalTensor):
        raise TypeError(f"augment requires a NeuronalTensor; got {type(tensor).__name__}")
    if not isinstance(spec, AugmentationSpec):
        raise TypeError(f"augment requires an AugmentationSpec; got {type(spec).__name__}")
    base_digest = _tensor_identity_digest(tensor)
    if not base_digest:
        raise ValueError(
            "augment requires a computable tensor identity digest for provenance; "
            "the base tensor's digest could not be computed (refusing, no silent '')"
        )
    out = copy.deepcopy(tensor)
    by_axis = {record.axis: record for record in spec.transforms}
    realized: list[str] = []
    changes: list[ProvenanceEntry] = []
    notes: list[str] = []
    for axis in CANONICAL_ORDER:
        record = by_axis.get(axis)
        if record is None:
            continue
        n_before = len(changes)
        if axis == "N":
            if not isinstance(record, ScaleN):
                raise ValueError(
                    f"augmentation axis 'N' expects a ScaleN record; "
                    f"got {type(record).__name__} ({record!r})"
                )
            _apply_scale_n(out, record, changes)
            if len(changes) > n_before:
                realized.append(axis)
                notes.append(_SCALING_NOTE)
        elif axis == "G":
            if not isinstance(record, GeometryTransform):
                raise ValueError(
                    "augmentation axis 'G' expects a GeometryTransform record; "
                    f"got {type(record).__name__} ({record!r})"
                )
            _apply_geometry(out, record, changes)
            if len(changes) > n_before:
                realized.append(axis)
        elif axis == "Theta_C":
            if not isinstance(record, ThetaC):
                raise ValueError(
                    "augmentation axis 'Theta_C' expects a ThetaC record; "
                    f"got {type(record).__name__} ({record!r})"
                )
            _apply_theta_c(out, record, spec.k_v, changes)
            if len(changes) > n_before:
                realized.append(axis)
        elif axis == "Theta_X":
            if not isinstance(record, ThetaX):
                raise ValueError(
                    "augmentation axis 'Theta_X' expects a ThetaX record; "
                    f"got {type(record).__name__} ({record!r})"
                )
            _apply_theta_x(out, record, spec.k_v, changes)
            if len(changes) > n_before:
                realized.append(axis)
        elif axis == "W_0":
            if not isinstance(record, W0):
                raise ValueError(
                    "augmentation axis 'W_0' expects a W0 record; "
                    f"got {type(record).__name__} ({record!r})"
                )
            _apply_w0(out, record, spec.k_v, changes)
            if len(changes) > n_before:
                realized.append(axis)
        elif axis == "H_0":
            if not isinstance(record, H0):
                raise ValueError(
                    "augmentation axis 'H_0' expects a H0 record; "
                    f"got {type(record).__name__} ({record!r})"
                )
            _apply_h0(out, record, spec.k_v, changes)
            if len(changes) > n_before:
                realized.append(axis)
        else:  # fail closed: no silent skip of a known axis
            raise ValueError(f"augmentation axis {axis!r} has no packet-1 implementation")
    return out, AugmentationRecord(
        realized_order=tuple(realized),
        changes=tuple(changes),
        spec_digest=spec.digest(),
        base_digest=base_digest,
        notes=tuple(notes),
    )


def clone_tensor(tensor: NeuronalTensor) -> tuple[NeuronalTensor, AugmentationRecord]:
    """Exact clone: structurally equal, distinct object, zero changed values."""
    return augment(tensor, AugmentationSpec())


__all__ = [
    "CANONICAL_ORDER",
    "PACKET2_AXES",
    "AugmentationSpec",
    "AugmentationRecord",
    "ProvenanceEntry",
    "ScaleN",
    "GeometryTransform",
    "PoseEdit",
    "RangeEdit",
    "ThetaC",
    "ThetaX",
    "W0",
    "H0",
    "augment",
    "clone_tensor",
]
