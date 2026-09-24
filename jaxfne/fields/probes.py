"""Multimodal probe operators and leadfield transforms for jaxfne.

This module implements simulated electrophysiological probe operators (EEG, MEG, EMM proxies)
under linear_solver boundaries. All signals are processed as proxies,
and physical amplitude claims remain uncalibrated (amplitude_claim_allowed=False).
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _replace
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp


# Item 3 (0.5.2 ENGINE): single source representation Q.
#
# ``CanonicalSource`` is the one source representation consumed by every probe
# in this module and in ``fields/proxy.py``. It carries a ``[T, N]`` relative
# array plus its representation/mode/provenance so the S -> F -> P chain can
# pass Q (not bare arrays) end to end. Every probe also keeps accepting a bare
# array, which is consumed unchanged (bit-identical outputs); only an explicit
# ``CanonicalSource`` adds provenance keys to the report.
CANONICAL_SOURCE_REPRESENTATION = "relative"


@dataclass(frozen=True)
class CanonicalSource:
    """Single source representation Q: ``[T, N]`` relative source array.

    Q is relative-only: any ``representation`` other than ``"relative"``
    is refused at construction (physical units arrive only through the
    item-4 calibration transform, never by relabeling Q).
    """

    data: jax.Array
    representation: str = CANONICAL_SOURCE_REPRESENTATION
    source_mode: str = "caller_supplied"
    provenance: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.representation != CANONICAL_SOURCE_REPRESENTATION:
            raise ValueError(
                "CanonicalSource Q is relative-only "
                f"(representation={self.representation!r}); physical units require "
                "an explicit calibration transform (0.5.2 item 4), not a relabeled Q."
            )
        data = jnp.asarray(self.data)
        if data.ndim != 2:
            raise ValueError(f"CanonicalSource Q must be 2D [T, N]; got shape {data.shape}")
        if not isinstance(data, jax.core.Tracer):
            if not bool(jnp.all(jnp.isfinite(data))):
                raise ValueError("CanonicalSource Q must be finite.")
        object.__setattr__(self, "data", data)


def canonical_source(
    data: jax.Array,
    *,
    source_mode: str = "caller_supplied",
    provenance: Optional[dict[str, Any]] = None,
) -> CanonicalSource:
    """Build the single source representation Q from a ``[T, N]`` array."""
    return CanonicalSource(
        data=jnp.asarray(data),
        representation=CANONICAL_SOURCE_REPRESENTATION,
        source_mode=str(source_mode),
        provenance=dict(provenance) if provenance is not None else None,
    )


def _electrode_report_fragment(
    position: Any = None,
    reference: Any = None,
    filter_spec: Any = None,
) -> dict[str, Any]:
    """Declared probe/electrode semantics for the report (0.5.2 item 5).

    ``position`` (electrode/contact positions), ``reference`` (reference
    scheme) and ``filter_spec`` (filter declaration) are recorded as declared
    or explicitly ``"undeclared"``. Declaration only: proxy probes apply no
    reference arithmetic and no filter; undeclared stays undeclared rather
    than invented.
    """

    def _show(v: Any) -> str:
        return "undeclared" if v is None else str(v)

    return {
        "position": _show(position),
        "reference": _show(reference),
        "filter": _show(filter_spec),
    }


def _unwrap_probe_input(x: jax.Array | CanonicalSource) -> tuple[jax.Array, dict[str, Any]]:
    """Consume ``CanonicalSource | array`` through the one Q point.

    Returns ``(array, report_fragment)``: the array unchanged (bit-identical
    probe outputs either way) and additive report keys only when Q was given.
    """
    if isinstance(x, CanonicalSource):
        if x.representation != CANONICAL_SOURCE_REPRESENTATION:
            raise ValueError(
                "Probe input Q must carry representation "
                f"{CANONICAL_SOURCE_REPRESENTATION!r}; got {x.representation!r}."
            )
        return x.data, {
            "source_identity": "canonical_source_Q",
            "source_representation": x.representation,
            "source_mode": x.source_mode,
        }
    return jnp.asarray(x), {}


# Item 4 (0.5.2 ENGINE): epistemic levels + refusal gate.
#
# Levels: RELATIVE_PROXY != REDUCED_PHYSICAL != CALIBRATED. Every Phi/Y output
# starts RELATIVE_PROXY. A proxy becomes CALIBRATED (or REDUCED_PHYSICAL) only
# through apply_calibration() with an explicit CalibrationTransform declaring
# units + conductivity + distance; every other relabel path is refused.
EPISTEMIC_RELATIVE_PROXY = "RELATIVE_PROXY"
EPISTEMIC_REDUCED_PHYSICAL = "REDUCED_PHYSICAL"
EPISTEMIC_CALIBRATED = "CALIBRATED"
EPISTEMIC_LEVELS = (
    EPISTEMIC_RELATIVE_PROXY,
    EPISTEMIC_REDUCED_PHYSICAL,
    EPISTEMIC_CALIBRATED,
)


class EpistemicRefusal(ValueError):
    """A proxy-to-calibrated relabel was attempted without authority."""


def _declared_number(value: Any, *, name: str, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return  # non-numeric declarations are opaque tags; presence is the gate
    fv = float(value)
    if not (fv == fv) or abs(fv) == float("inf"):
        raise EpistemicRefusal(f"CalibrationTransform {name} must be finite; got {value!r}.")
    if positive and not fv > 0:
        raise EpistemicRefusal(f"CalibrationTransform {name} must be positive; got {value!r}.")
    if not positive and not fv >= 0:
        raise EpistemicRefusal(f"CalibrationTransform {name} must be non-negative; got {value!r}.")


@dataclass(frozen=True)
class CalibrationTransform:
    """Explicit calibration transform: the only authority that relabels a proxy.

    Declares ``units`` (physical unit string, e.g. ``"V"``), ``conductivity``
    (S/m value or declared-model tag), and ``distance`` (distance law/value
    or declared-geometry tag). All three are required; numerics must be
    finite (conductivity positive, distance non-negative). ``target_level``
    is ``CALIBRATED`` (default) or ``REDUCED_PHYSICAL``.
    """

    units: str
    conductivity: Union[str, float]
    distance: Union[str, float]
    target_level: str = EPISTEMIC_CALIBRATED
    method: str = "explicit_boundary_transform"
    declared_by: str = ""

    def __post_init__(self) -> None:
        if self.target_level not in (EPISTEMIC_CALIBRATED, EPISTEMIC_REDUCED_PHYSICAL):
            raise EpistemicRefusal(
                "CalibrationTransform target_level must be CALIBRATED or "
                f"REDUCED_PHYSICAL; got {self.target_level!r}."
            )
        if not isinstance(self.units, str) or not self.units.strip():
            raise EpistemicRefusal("CalibrationTransform requires declared units.")
        for name, value, positive in (
            ("conductivity", self.conductivity, True),
            ("distance", self.distance, False),
        ):
            if value is None or (isinstance(value, str) and not value.strip()):
                raise EpistemicRefusal(f"CalibrationTransform requires declared {name}.")
            _declared_number(value, name=name, positive=positive)


# Private seal: only apply_calibration() in this module mints an
# AppliedCalibration carrying _SEAL. Anything else (hand-built record,
# forged dict, direct construction) fails the __post_init__ gate below.
_SEAL: Any = object()


@dataclass(frozen=True)
class AppliedCalibration:
    """Sealed record that a proxy output was calibrated via a transform."""

    transform: CalibrationTransform
    level: str
    applied_to: str = ""
    _seal: Any = None

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise EpistemicRefusal(
                "AppliedCalibration cannot be constructed directly: relabel a proxy "
                "only via apply_calibration() with an explicit CalibrationTransform."
            )
        if not isinstance(self.transform, CalibrationTransform):
            raise EpistemicRefusal("AppliedCalibration requires a CalibrationTransform.")
        if self.level != self.transform.target_level:
            raise EpistemicRefusal(
                f"AppliedCalibration level {self.level!r} != transform target "
                f"{self.transform.target_level!r}."
            )


def _applied_calibration_to_dict(cal: "AppliedCalibration") -> dict[str, Any]:
    t = cal.transform
    return {
        "level": cal.level,
        "applied_to": cal.applied_to,
        "transform": {
            "units": t.units,
            "conductivity": t.conductivity,
            "distance": t.distance,
            "target_level": t.target_level,
            "method": t.method,
            "declared_by": t.declared_by,
        },
    }


def _check_epistemic_fields(owner: str, level: str, calibration: Any) -> None:
    if level not in EPISTEMIC_LEVELS:
        raise EpistemicRefusal(
            f"{owner} epistemic_level must be one of {list(EPISTEMIC_LEVELS)}; got {level!r}."
        )
    if level == EPISTEMIC_RELATIVE_PROXY:
        if calibration is not None:
            raise EpistemicRefusal(
                f"{owner} at RELATIVE_PROXY must not carry a calibration record."
            )
        return
    if not (
        isinstance(calibration, AppliedCalibration)
        and calibration._seal is _SEAL
        and calibration.level == level
    ):
        raise EpistemicRefusal(
            f"{owner} may become {level} only via apply_calibration() with an "
            "explicit CalibrationTransform declaring units+conductivity+distance."
        )


def apply_calibration(obj: Any, transform: CalibrationTransform) -> Any:
    """Relabel a RELATIVE_PROXY output to the transform's target level.

    The only path from proxy to CALIBRATED/REDUCED_PHYSICAL. Refuses (raises
    :class:`EpistemicRefusal`): a non-transform authority, an incomplete
    transform, an object carrying no epistemic level, or an object that is
    already calibrated (no silent re-calibration).
    """
    if not isinstance(transform, CalibrationTransform):
        raise EpistemicRefusal(
            "Proxy relabel refused: authority must be an explicit "
            f"CalibrationTransform, got {type(transform).__name__}."
        )
    level = getattr(obj, "epistemic_level", None)
    if level is None:
        raise EpistemicRefusal(
            f"Proxy relabel refused: {type(obj).__name__} carries no epistemic level."
        )
    if level != EPISTEMIC_RELATIVE_PROXY:
        raise EpistemicRefusal(
            f"Proxy relabel refused: already {level}; re-calibration is not silent."
        )
    sealed = AppliedCalibration(
        transform=transform,
        level=transform.target_level,
        applied_to=type(obj).__name__,
        _seal=_SEAL,
    )
    try:
        return _replace(obj, epistemic_level=transform.target_level, calibration=sealed)
    except TypeError as exc:
        raise EpistemicRefusal(
            f"Proxy relabel refused: cannot seal {type(obj).__name__}: {exc}"
        ) from exc


def sample_phi_at_probe_depths(
    phi_e: jax.Array | CanonicalSource,
    field_contact_depths: jax.Array,
    probe_contact_depths: jax.Array,
) -> jax.Array:
    """Interpolate laminar ``phi_e`` along contact depth to probe locations.

    ``phi_e`` is ``(T, C_field)``; ``field_contact_depths`` is ``(C_field,)``;
    ``probe_contact_depths`` is ``(P,)``. Returns ``(T, P)``.
    """
    phi_e, _ = _unwrap_probe_input(phi_e)
    field_z = jnp.asarray(field_contact_depths, dtype=phi_e.dtype)
    probe_z = jnp.asarray(probe_contact_depths, dtype=phi_e.dtype)
    if phi_e.ndim != 2:
        raise ValueError(f"phi_e must be 2D (T, C); got shape {phi_e.shape}")
    if field_z.shape[0] != phi_e.shape[1]:
        raise ValueError(
            f"field_contact_depths length {field_z.shape[0]} != phi_e channels {phi_e.shape[1]}"
        )

    def _interp_row(row: jax.Array) -> jax.Array:
        return jnp.interp(probe_z, field_z, row)

    return jax.vmap(_interp_row)(phi_e)


@dataclass(frozen=True)
class ProbeReadout:
    """Container for probe operator output and metadata report.

    A probe operator produces data (array or dict) plus a JSON-safe report
    declaring operator status, units, calibration, truth gates, and assumptions.

    ``epistemic_level`` (0.5.2 item 4) is structural, not a report string:
    every readout starts ``RELATIVE_PROXY`` and becomes ``REDUCED_PHYSICAL``
    or ``CALIBRATED`` only through :func:`apply_calibration` with an explicit
    :class:`CalibrationTransform`. Direct construction at a non-proxy level
    is refused in ``__post_init__``.
    """

    name: str
    kind: str
    data: Any
    report: dict[str, Any]
    epistemic_level: str = "RELATIVE_PROXY"
    calibration: Optional["AppliedCalibration"] = None

    def __post_init__(self) -> None:
        _check_epistemic_fields(type(self).__name__, self.epistemic_level, self.calibration)

    def to_dict(self) -> dict:
        """Return JSON-safe representation of readout and report."""
        from ..io import json_safe

        return {
            "name": self.name,
            "kind": self.kind,
            "data_shape": str(getattr(self.data, "shape", None)),
            "epistemic_level": self.epistemic_level,
            "calibration": json_safe(
                _applied_calibration_to_dict(self.calibration)
                if self.calibration is not None
                else None
            ),
            "report": json_safe(self.report),
        }


def _make_probe_report(
    kind: str,
    method: str,
    operator_status: str = "simulated_proxy",
    operator_type: str = "direct_readout",
    data_shape: tuple | str = None,
    units_or_status: str = "proxy_units",
    calibration_status: str = "uncalibrated_proxy",
    input_representation: str = "relative",
    representation: str = "relative",
    validation_status: str = "computational",
    calibration_transform: str = "explicit_boundary_transform",
    field_solver_status: str = "linear_solver",
    field_claim_level: str = "proxy_readout",
    source_calibration_status: str = "uncalibrated_izhikevich_native_current",
    source_projection_mode: str = "proxy_no_field_solve",
    source_decomposition: str = "proxy_reduced_emitter",
    assumptions: list[str] = None,
    extra_fields: dict = None,
) -> dict:
    """Build a JSON-safe probe operator report."""
    if assumptions is None:
        assumptions = []

    report = {
        "name": kind,
        "kind": kind,
        "operator_status": operator_status,
        "operator_type": operator_type,
        "method": method,
        "data_shape": str(data_shape) if data_shape is not None else "unknown",
        "units_or_status": units_or_status,
        "calibration_status": calibration_status,
        "input_representation": input_representation,
        "representation": representation,
        "validation_status": validation_status,
        "calibration_transform": calibration_transform,
        "source_calibration_status": source_calibration_status,
        "source_projection_mode": source_projection_mode,
        "source_decomposition": source_decomposition,
        "field_solver_status": field_solver_status,
        "field_claim_level": field_claim_level,
        "physical_amplitude_calibrated": False,
        "assumptions": assumptions,
    }

    if extra_fields:
        report.update(extra_fields)

    return report


def create_probe(
    kind: str,
    data: jax.Array | CanonicalSource,
    *,
    method: str,
    units_or_status: str = "proxy_units",
    assumptions: list[str] | None = None,
    extra_fields: dict | None = None,
    **report_kwargs,
) -> ProbeReadout:
    """Generic probe factory — creates a ProbeReadout with a standardized report.

    Replaces repetitive per-kind probe wrappers with a single entry point.
    """
    data, _src = _unwrap_probe_input(data)
    if _src:
        extra_fields = {**_src, **(extra_fields or {})}
    report = _make_probe_report(
        kind=kind,
        method=method,
        data_shape=data.shape,
        units_or_status=units_or_status,
        assumptions=assumptions or [],
        extra_fields=extra_fields,
        **report_kwargs,
    )
    return ProbeReadout(name=kind, kind=kind, data=data, report=report)


def spk_probe(
    spikes: jax.Array | CanonicalSource,
    *,
    position: Any = None,
    reference: Any = None,
    filter_spec: Any = None,
) -> ProbeReadout:
    """SPK probe operator: expose spike events or spike matrix."""
    spikes, _src = _unwrap_probe_input(spikes)
    return create_probe(
        "spk",
        spikes,
        method="threshold_or_emitter_spike_array",
        units_or_status="binary_spike_indicator",
        input_representation="relative_spike_events",
        assumptions=["spike_array_from_emitter_or_threshold", "binary_or_threshold_values"],
        extra_fields={**_src, **_electrode_report_fragment(position, reference, filter_spec)},
    )


def vm_probe(
    voltage: jax.Array | CanonicalSource,
    *,
    position: Any = None,
    reference: Any = None,
    filter_spec: Any = None,
) -> ProbeReadout:
    """Vm probe operator: expose membrane voltage or native reduced-emitter state."""
    voltage, _src = _unwrap_probe_input(voltage)
    return create_probe(
        "vm",
        voltage,
        method="emitter_state_voltage_trace",
        units_or_status="mV_or_native_model_voltage",
        input_representation="relative_vm_state",
        assumptions=[
            "voltage_from_emitter_native_state",
            "not_physical_membrane_voltage_unless_calibrated",
        ],
        extra_fields={**_src, **_electrode_report_fragment(position, reference, filter_spec)},
    )


def source_probe(
    source: jax.Array | CanonicalSource,
    *,
    position: Any = None,
    reference: Any = None,
    filter_spec: Any = None,
) -> ProbeReadout:
    """Source probe operator: expose current/source proxy."""
    source, _src = _unwrap_probe_input(source)
    return create_probe(
        "source",
        source,
        method="declared_source_projection_or_proxy",
        units_or_status="native_current_units_or_proxy",
        input_representation="canonical_relative_source",
        source_decomposition="proxy_reduced_emitter",
        assumptions=[
            "source_from_emitter_native_state",
            "not_physical_membrane_current_unless_calibrated",
        ],
        extra_fields={**_src, **_electrode_report_fragment(position, reference, filter_spec)},
    )


def lfp_proxy_probe(
    phi_e: jax.Array | CanonicalSource,
    contact_depths: jax.Array = None,
    field_contact_depths: jax.Array = None,
    *,
    position: Any = None,
    reference: Any = None,
    filter_spec: Any = None,
    allow_synthesized_field_contacts: bool = False,
) -> ProbeReadout:
    """LFP-proxy probe operator: sample extracellular potential-like state.

    Route note (P8): this CONSTRUCTS a probe readout from ``phi_e`` —
    distinct from declared field access (which raises when probes were not
    requested) and from visualization-only proxies (which never enter
    ``Signals.field``). The three routes are not interchangeable.

    No invented contacts on scientific paths (0.5.2 item 5, extends Rc P4):
    when ``contact_depths`` is given without ``field_contact_depths``, the
    call is refused unless ``allow_synthesized_field_contacts=True``
    explicitly opts into the labeled constructed fallback (synthesized
    ``linspace(0, 1)`` contacts recorded in the report, never silent).
    """
    phi_e, _src = _unwrap_probe_input(phi_e)
    extra: dict[str, Any] = {
        **_src,
        **_electrode_report_fragment(position, reference, filter_spec),
    }
    if position is None and contact_depths is not None:
        extra["position"] = str(jnp.asarray(contact_depths))
    method = "point_or_finite_contact_phi_proxy"
    data = phi_e
    assumptions = [
        "laminar_proxy_field_no_pde",
        "contact_sample_from_phi_e_proxy",
        "not_empirically_calibrated",
    ]
    if contact_depths is not None:
        contact_depths = jnp.asarray(contact_depths)
        if field_contact_depths is None:
            if not allow_synthesized_field_contacts:
                raise ValueError(
                    "lfp_proxy_probe refuses to invent field contacts: "
                    "contact_depths was given without field_contact_depths. "
                    "Declare field_contact_depths, or pass "
                    "allow_synthesized_field_contacts=True to opt into the "
                    "labeled constructed fallback (synthesized linspace(0, 1))."
                )
            n = int(phi_e.shape[-1])
            field_contact_depths = jnp.linspace(0.0, 1.0, n, dtype=phi_e.dtype)
            extra["synthesized_field_contacts"] = True
            assumptions = assumptions + [
                "field_contacts_synthesized_explicit_opt_in_not_declared",
            ]
        data = sample_phi_at_probe_depths(phi_e, field_contact_depths, contact_depths)
        method = "depth_interpolation_on_phi_e_proxy"
        extra["contact_depths_or_layers"] = str(contact_depths)
        extra["field_contact_depths"] = str(field_contact_depths)

    report = _make_probe_report(
        kind="lfp_proxy",
        method=method,
        input_representation="relative_phi_e_proxy",
        data_shape=data.shape,
        units_or_status="proxy_voltage_units_or_V_if_calibrated",
        assumptions=assumptions,
        extra_fields=extra,
    )
    return ProbeReadout(name="lfp_proxy", kind="lfp_proxy", data=data, report=report)


def csd_proxy_probe(
    csd: jax.Array | CanonicalSource,
    csd_sign_convention: str = "positive_equals_extracellular_source",
) -> ProbeReadout:
    """CSD-proxy probe operator: estimate source-profile-like CSD-proxy readout."""
    csd, _src = _unwrap_probe_input(csd)
    report = _make_probe_report(
        kind="csd_proxy",
        method="divergence_proxy_or_second_derivative_laminar",
        operator_type="spatial_derivative",
        input_representation="relative_phi_e_proxy",
        data_shape=csd.shape,
        units_or_status="proxy_A_m^-3_or_proxy_units",
        assumptions=[
            "laminar_proxy_field_no_pde",
            "second_derivative_or_divergence_proxy",
            "not_empirically_calibrated",
        ],
        extra_fields={
            **_src,
            "CSD_sign_convention": csd_sign_convention,
            "amplitude_semantics": "relative",
            "physical_claim": "proxy_readout",
            "observation": {
                "execution_form": "fused",
                "operator_chain": {
                    "source": {
                        "identity": "canonical_relative_source",
                        "representation": "relative",
                    },
                    "field": {
                        "identity": "caller_supplied_or_projected_phi_e_proxy",
                        "note": "wrapper_does_not_recompute_F",
                    },
                    "probe": {
                        "identity": "laminar_second_derivative",
                        "stencil": "negative_second_difference_edge_padded",
                    },
                },
                "amplitude_semantics": "relative",
                "validation_status": "computational",
                "physical_claim": "proxy_readout",
                "output_identity": "csd_proxy",
            },
        },
    )
    return ProbeReadout(name="csd_proxy", kind="csd_proxy", data=csd, report=report)


def eeg_proxy_probe(
    eeg: jax.Array | CanonicalSource,
    leadfield_status: str = "toy_or_declared_proxy",
    n_sensors: int = None,
) -> ProbeReadout:
    """EEG-proxy probe operator: simulated scalp-channel EEG-proxy readout."""
    eeg, _src = _unwrap_probe_input(eeg)
    extra = {
        **_src,
        "leadfield_status": leadfield_status,
        "sensor_geometry_status": "simulated_minimal",
        "amplitude_semantics": "relative",
        "physical_claim": "proxy_readout",
        "observation": {
            "execution_form": "fused",
            "operator_chain": {
                "source": {
                    "identity": "canonical_relative_source",
                    "representation": "relative",
                },
                "field": {
                    "identity": "compiled_into_leadfield",
                    "note": "P_circ_F_not_separately_materialized",
                },
                "probe": {
                    "identity": "linear_leadfield",
                    "leadfield_status": leadfield_status,
                    "n_sensors": int(eeg.shape[-1]) if n_sensors is None else int(n_sensors),
                },
            },
            "amplitude_semantics": "relative",
            "validation_status": "computational",
            "physical_claim": "proxy_readout",
            "output_identity": "eeg_proxy",
        },
    }

    report = _make_probe_report(
        kind="eeg_proxy",
        method="linear_leadfield_proxy",
        operator_type="linear_projection",
        input_representation="canonical_relative_source",
        data_shape=eeg.shape,
        units_or_status="arbitrary_proxy_units",
        assumptions=[
            "simulated_eeg_proxy_readout",
            "toy_or_declared_leadfield",
            "not_validated_against_real_eeg",
            "not_empirically_calibrated",
        ],
        extra_fields=extra,
    )
    return ProbeReadout(name="eeg_proxy", kind="eeg_proxy", data=eeg, report=report)


def meg_proxy_probe(
    meg: jax.Array | CanonicalSource,
    leadfield_status: str = "toy_or_declared_proxy",
    orientation_convention: str = "declared",
    n_sensors: int = None,
) -> ProbeReadout:
    """MEG-proxy probe operator: wrap a relative linear map of scalar source ``Q``.

    Does not construct oriented current or a physical MEG forward operator.
    ``orientation_convention`` is retained as a compatibility argument and is
    not interpreted as a current-orientation claim.
    """
    meg, _src = _unwrap_probe_input(meg)
    _ = orientation_convention  # compatibility argument; not a current-orientation claim
    extra = {
        **_src,
        "leadfield_status": leadfield_status,
        "sensor_geometry_status": "simulated_minimal",
        "orientation_convention": "none",
        "amplitude_semantics": "relative",
        "physical_claim": "proxy_readout",
        "observation": {
            "execution_form": "fused",
            "operator_chain": {
                "source": {
                    "identity": "canonical_relative_source",
                    "representation": "relative",
                    "vector_current": False,
                },
                "field": {
                    "identity": "not_a_physical_forward_operator",
                    "note": "relative_linear_map_on_scalar_Q",
                },
                "probe": {
                    "identity": "relative_linear_map",
                    "orientation_claim": "none",
                    "leadfield_status": leadfield_status,
                    "n_sensors": int(meg.shape[-1]) if n_sensors is None else int(n_sensors),
                },
            },
            "amplitude_semantics": "relative",
            "validation_status": "computational",
            "physical_claim": "proxy_readout",
            "output_identity": "meg_relative_proxy",
        },
    }

    report = _make_probe_report(
        kind="meg_proxy",
        method="relative_linear_map_proxy",
        operator_type="linear_projection",
        input_representation="canonical_relative_source",
        data_shape=meg.shape,
        units_or_status="arbitrary_proxy_units",
        assumptions=[
            "simulated_meg_proxy_readout",
            "toy_or_declared_leadfield",
            "scalar_Q_relative_linear_map",
            "no_current_orientation_claim",
            "not_validated_against_real_meg",
            "not_empirically_calibrated",
        ],
        extra_fields=extra,
    )
    return ProbeReadout(name="meg_proxy", kind="meg_proxy", data=meg, report=report)


def emm_proxy_probe(
    emm: jax.Array | CanonicalSource,
    method: str = "normalized_activity_field_source_cost_proxy",
) -> ProbeReadout:
    """EMM-proxy probe operator: electromagnetic metabolism estimate proxy."""
    emm, _src = _unwrap_probe_input(emm)
    report = _make_probe_report(
        kind="emm_proxy",
        method=method,
        input_representation="relative_activity_source_field",
        data_shape=emm.shape,
        units_or_status="normalized_proxy_units",
        calibration_status="uncalibrated_proxy",
        assumptions=[
            "emm_proxy_normalized_activity_cost",
            "not_biological_metabolism",
            "relative_within_run_comparison_only",
            "proxy_electrophysiological_field_activity_cost",
        ],
        extra_fields=_src or None,
    )
    return ProbeReadout(name="emm_proxy", kind="emm_proxy", data=emm, report=report)


def _leadfield_proxy_transform(
    source: jax.Array | CanonicalSource, leadfield: jax.Array, *, param_name: str
) -> jax.Array:
    """Shared linear leadfield projection behind ``eeg_proxy_transform``/``meg_proxy_transform``.

    Both public functions are ``source @ leadfield.T`` with identical validation,
    differing only in the source array's conventional parameter name (``source``
    for EEG, ``source_oriented`` for MEG) -- ``param_name`` reproduces that name
    in error messages so each public entry point's errors read exactly as before
    (jaxfne-harden rule 10; merged 2026-07-21, both public names kept as thin
    backward-compatible wrappers, zero call-site changes required).
    """
    source, _ = _unwrap_probe_input(source)
    leadfield = jnp.asarray(leadfield)

    if source.ndim != 2:
        raise ValueError(f"{param_name} must be 2D [T, K], got shape {source.shape}")
    if leadfield.ndim != 2:
        raise ValueError(f"leadfield must be 2D [C, K], got shape {leadfield.shape}")

    T, K = source.shape
    C, K_lead = leadfield.shape

    if K != K_lead:
        raise ValueError(f"{param_name} and leadfield K dimension mismatch: {K} vs {K_lead}")

    return source @ leadfield.T


def eeg_proxy_transform(
    source: jax.Array | CanonicalSource,
    leadfield: jax.Array,
) -> jax.Array:
    """Compute EEG-proxy readout via linear leadfield projection."""
    return _leadfield_proxy_transform(source, leadfield, param_name="source")


def meg_proxy_transform(
    source_oriented: jax.Array | CanonicalSource,
    leadfield: jax.Array,
) -> jax.Array:
    """Compute MEG-proxy readout via linear leadfield projection."""
    return _leadfield_proxy_transform(source_oriented, leadfield, param_name="source_oriented")


def emm_proxy_transform(
    spike_rate: jax.Array | CanonicalSource,
    source: jax.Array | CanonicalSource,
    field_potential: jax.Array | CanonicalSource,
    lambda_spk: float = 1.0,
    lambda_src: float = 1.0,
    lambda_field: float = 1.0,
) -> jax.Array:
    """Compute EMM-proxy (normalized activity/source/field cost) readout."""
    spike_rate, _ = _unwrap_probe_input(spike_rate)
    source, _ = _unwrap_probe_input(source)
    field_potential, _ = _unwrap_probe_input(field_potential)

    if spike_rate.ndim == 1:
        spike_rate = spike_rate[:, None]
    elif spike_rate.ndim != 2:
        raise ValueError(f"spike_rate must be 1D or 2D, got shape {spike_rate.shape}")

    if source.ndim != 2:
        raise ValueError(f"source must be 2D [T, K], got shape {source.shape}")

    if field_potential.ndim != 2:
        raise ValueError(f"field_potential must be 2D [T, X], got shape {field_potential.shape}")

    T_spk = spike_rate.shape[0]
    T_src = source.shape[0]
    T_field = field_potential.shape[0]

    if not (T_spk == T_src == T_field):
        raise ValueError(
            f"Time dimension mismatch: spike_rate={T_spk}, source={T_src}, field={T_field}"
        )

    term_spk = lambda_spk * spike_rate
    source_l1 = jnp.sum(jnp.abs(source), axis=1, keepdims=True)
    term_src = lambda_src * source_l1
    field_l2_sq = jnp.sum(jnp.square(field_potential), axis=1, keepdims=True)
    term_field = lambda_field * field_l2_sq

    total_cost = term_spk + term_src + term_field
    total_weight = lambda_spk + lambda_src + lambda_field

    if total_weight > 0:
        emm_proxy = total_cost / total_weight
    else:
        emm_proxy = total_cost

    return emm_proxy
