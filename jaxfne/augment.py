"""Controlled model augmentation: exact clone, cardinality (N), geometry (G).

One canonical TFNE/JDNA specification A is developed by JDNA into a base
:class:`NeuronalTensor`; this module applies typed, independently selectable
transforms to that tensor *before* ``construct``::

    base spec -> augmentation spec -> JDNA completion -> NeuronalTensor
        -> augment -> NeuronalTensor -> construct -> Model -> simulate

Transforms never mutate a ``Model``, never add a simulation path, and never
extend the TFNE grammar. Packet 1 implements the skeleton plus the ``N``
and ``G`` primitives; the ``Theta_C`` / ``Theta_X`` / ``W_0`` / ``H_0``
records exist as typed placeholders only and are refused with
``NotImplementedError`` naming packet 2.

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
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Sequence

from ._config import _check_cell_type_fractions, _counts_from_fractions
from .jdna.completion import ORIGIN_AUGMENTED
from .neuronal_tensor import (
    NeuronalTensor,
    _tensor_identity_digest,
)

#: Fixed canonical application order (R1): N -> G -> Theta_C -> Theta_X -> W_0 -> H_0.
CANONICAL_ORDER: tuple[str, ...] = ("N", "G", "Theta_C", "Theta_X", "W_0", "H_0")

#: Packet-2 axes: typed records exist, application is not implemented.
PACKET2_AXES = frozenset({"Theta_C", "Theta_X", "W_0", "H_0"})

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
    """Packet-2 placeholder: connectivity/rule-parameter variation."""

    stochastic: bool = False

    axis: ClassVar[str] = "Theta_C"


@dataclass(frozen=True)
class ThetaX:
    """Packet-2 placeholder: dynamical-parameter variation."""

    stochastic: bool = False

    axis: ClassVar[str] = "Theta_X"


@dataclass(frozen=True)
class W0:
    """Packet-2 placeholder: initial plastic/mutable-parameter variation."""

    stochastic: bool = False

    axis: ClassVar[str] = "W_0"


@dataclass(frozen=True)
class H0:
    """Packet-2 placeholder: initial relative-hidden-state variation."""

    stochastic: bool = False

    axis: ClassVar[str] = "H_0"


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


def augment(
    tensor: NeuronalTensor, spec: AugmentationSpec
) -> tuple[NeuronalTensor, AugmentationRecord]:
    """Apply ``spec`` to ``tensor`` in canonical order; return ``(new, record)``.

    The input tensor is never touched: the result is a deep copy with only
    the targeted values rewritten. A transform that changes no value is a
    no-op: it contributes no ``realized_order`` entry (and a no-op ``ScaleN``
    writes no scaling note). Records on packet-2 axes raise
    ``NotImplementedError`` naming packet 2.
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
        if axis in PACKET2_AXES:
            raise NotImplementedError(
                f"augmentation axis {axis!r} is packet 2 (parameter variation); "
                "packet 1 implements clone, N and G only"
            )
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
