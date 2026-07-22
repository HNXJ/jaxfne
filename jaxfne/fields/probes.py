"""Multimodal probe operators and leadfield transforms for jaxfne.

This module implements simulated electrophysiological probe operators (EEG, MEG, EMM proxies)
under linear_solver boundaries. All signals are processed as proxies,
and physical amplitude claims remain uncalibrated (amplitude_claim_allowed=False).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ProbeReadout:
    """Container for probe operator output and metadata report.

    A probe operator produces data (array or dict) plus a JSON-safe report
    declaring operator status, units, calibration, truth gates, and assumptions.
    """

    name: str
    kind: str
    data: Any
    report: dict[str, Any]

    def to_dict(self) -> dict:
        """Return JSON-safe representation of readout and report."""
        from ..io import json_safe
        return {
            "name": self.name,
            "kind": self.kind,
            "data_shape": str(getattr(self.data, "shape", None)),
            "report": json_safe(self.report),
        }


def _make_probe_report(
    kind: str,
    method: str,
    operator_status: str = "simulated_proxy",
    data_shape: tuple | str = None,
    units_or_status: str = "proxy_units",
    calibration_status: str = "uncalibrated_proxy",
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
        "method": method,
        "data_shape": str(data_shape) if data_shape is not None else "unknown",
        "units_or_status": units_or_status,
        "calibration_status": calibration_status,
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
    data: jax.Array,
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
    data = jnp.asarray(data)
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


def spk_probe(spikes: jax.Array) -> ProbeReadout:
    """SPK probe operator: expose spike events or spike matrix."""
    return create_probe(
        "spk", spikes,
        method="threshold_or_emitter_spike_array",
        units_or_status="binary_spike_indicator",
        assumptions=["spike_array_from_emitter_or_threshold", "binary_or_threshold_values"],
    )


def vm_probe(voltage: jax.Array) -> ProbeReadout:
    """Vm probe operator: expose membrane voltage or native reduced-emitter state."""
    return create_probe(
        "vm", voltage,
        method="emitter_state_voltage_trace",
        units_or_status="mV_or_native_model_voltage",
        assumptions=["voltage_from_emitter_native_state", "not_physical_membrane_voltage_unless_calibrated"],
    )


def source_probe(source: jax.Array) -> ProbeReadout:
    """Source probe operator: expose current/source proxy."""
    return create_probe(
        "source", source,
        method="declared_source_projection_or_proxy",
        units_or_status="native_current_units_or_proxy",
        source_decomposition="proxy_reduced_emitter",
        assumptions=["source_from_emitter_native_state", "not_physical_membrane_current_unless_calibrated"],
    )


def lfp_proxy_probe(
    phi_e: jax.Array,
    contact_depths: jax.Array = None,
) -> ProbeReadout:
    """LFP-proxy probe operator: sample extracellular potential-like state."""
    phi_e = jnp.asarray(phi_e)
    extra = {}
    if contact_depths is not None:
        contact_depths = jnp.asarray(contact_depths)
        extra["contact_depths_or_layers"] = str(contact_depths)

    report = _make_probe_report(
        kind="lfp_proxy",
        method="point_or_finite_contact_phi_proxy",
        data_shape=phi_e.shape,
        units_or_status="proxy_voltage_units_or_V_if_calibrated",
        assumptions=[
            "laminar_proxy_field_no_pde",
            "contact_sample_from_phi_e_proxy",
            "not_empirically_calibrated",
        ],
        extra_fields=extra,
    )
    return ProbeReadout(name="lfp_proxy", kind="lfp_proxy", data=phi_e, report=report)


def csd_proxy_probe(
    csd: jax.Array,
    csd_sign_convention: str = "positive_equals_extracellular_source",
) -> ProbeReadout:
    """CSD-proxy probe operator: estimate source-profile-like CSD-proxy readout."""
    csd = jnp.asarray(csd)
    report = _make_probe_report(
        kind="csd_proxy",
        method="divergence_proxy_or_second_derivative_laminar",
        data_shape=csd.shape,
        units_or_status="proxy_A_m^-3_or_proxy_units",
        assumptions=[
            "laminar_proxy_field_no_pde",
            "second_derivative_or_divergence_proxy",
            "not_empirically_calibrated",
        ],
        extra_fields={
            "CSD_sign_convention": csd_sign_convention,
        },
    )
    return ProbeReadout(name="csd_proxy", kind="csd_proxy", data=csd, report=report)


def eeg_proxy_probe(
    eeg: jax.Array,
    leadfield_status: str = "toy_or_declared_proxy",
    n_sensors: int = None,
) -> ProbeReadout:
    """EEG-proxy probe operator: simulated scalp-channel EEG-proxy readout."""
    eeg = jnp.asarray(eeg)
    extra = {
        "leadfield_status": leadfield_status,
        "sensor_geometry_status": "simulated_minimal",
    }

    report = _make_probe_report(
        kind="eeg_proxy",
        method="linear_leadfield_proxy",
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
    meg: jax.Array,
    leadfield_status: str = "toy_or_declared_proxy",
    orientation_convention: str = "declared",
    n_sensors: int = None,
) -> ProbeReadout:
    """MEG-proxy probe operator: simulated magnetometer MEG-proxy readout."""
    meg = jnp.asarray(meg)
    extra = {
        "leadfield_status": leadfield_status,
        "sensor_geometry_status": "simulated_minimal",
        "orientation_convention": orientation_convention,
    }

    report = _make_probe_report(
        kind="meg_proxy",
        method="linear_current_orientation_proxy",
        data_shape=meg.shape,
        units_or_status="arbitrary_proxy_units",
        assumptions=[
            "simulated_meg_proxy_readout",
            "toy_or_declared_leadfield",
            "current_orientation_proxy",
            "not_validated_against_real_meg",
            "not_empirically_calibrated",
        ],
        extra_fields=extra,
    )
    return ProbeReadout(name="meg_proxy", kind="meg_proxy", data=meg, report=report)


def emm_proxy_probe(
    emm: jax.Array,
    method: str = "normalized_activity_field_source_cost_proxy",
) -> ProbeReadout:
    """EMM-proxy probe operator: electromagnetic metabolism estimate proxy."""
    emm = jnp.asarray(emm)
    report = _make_probe_report(
        kind="emm_proxy",
        method=method,
        data_shape=emm.shape,
        units_or_status="normalized_proxy_units",
        calibration_status="uncalibrated_proxy",
        assumptions=[
            "emm_proxy_normalized_activity_cost",
            "not_biological_metabolism",
            "relative_within_run_comparison_only",
            "proxy_electrophysiological_field_activity_cost",
        ],
    )
    return ProbeReadout(name="emm_proxy", kind="emm_proxy", data=emm, report=report)


def _leadfield_proxy_transform(source: jax.Array, leadfield: jax.Array, *, param_name: str) -> jax.Array:
    """Shared linear leadfield projection behind ``eeg_proxy_transform``/``meg_proxy_transform``.

    Both public functions are ``source @ leadfield.T`` with identical validation,
    differing only in the source array's conventional parameter name (``source``
    for EEG, ``source_oriented`` for MEG) -- ``param_name`` reproduces that name
    in error messages so each public entry point's errors read exactly as before
    (jaxfne-harden rule 10; merged 2026-07-21, both public names kept as thin
    backward-compatible wrappers, zero call-site changes required).
    """
    source = jnp.asarray(source)
    leadfield = jnp.asarray(leadfield)

    if source.ndim != 2:
        raise ValueError(f"{param_name} must be 2D [T, K], got shape {source.shape}")
    if leadfield.ndim != 2:
        raise ValueError(f"leadfield must be 2D [C, K], got shape {leadfield.shape}")

    T, K = source.shape
    C, K_lead = leadfield.shape

    if K != K_lead:
        raise ValueError(
            f"{param_name} and leadfield K dimension mismatch: {K} vs {K_lead}"
        )

    return source @ leadfield.T


def eeg_proxy_transform(
    source: jax.Array,
    leadfield: jax.Array,
) -> jax.Array:
    """Compute EEG-proxy readout via linear leadfield projection."""
    return _leadfield_proxy_transform(source, leadfield, param_name="source")


def meg_proxy_transform(
    source_oriented: jax.Array,
    leadfield: jax.Array,
) -> jax.Array:
    """Compute MEG-proxy readout via linear leadfield projection."""
    return _leadfield_proxy_transform(source_oriented, leadfield, param_name="source_oriented")


def emm_proxy_transform(
    spike_rate: jax.Array,
    source: jax.Array,
    field_potential: jax.Array,
    lambda_spk: float = 1.0,
    lambda_src: float = 1.0,
    lambda_field: float = 1.0,
) -> jax.Array:
    """Compute EMM-proxy (normalized activity/source/field cost) readout."""
    spike_rate = jnp.asarray(spike_rate)
    source = jnp.asarray(source)
    field_potential = jnp.asarray(field_potential)

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
