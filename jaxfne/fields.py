"""Source-to-field and probe utilities for :mod:`jaxfne`.

This module intentionally implements a laminar proxy readout, not a full PDE
solver.  The full TFNE resistive field problem is reserved for a later operator:

    J_e = -sigma_e grad(phi_e)
    div(J_e) = q
    div(-sigma_e grad(phi_e)) = q
    CSD = div(J_e)

Until that operator exists, outputs are named and diagnosed as proxies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FieldOutput:
    """Container for laminar proxy field/readout arrays.

    ``source_proxy``, ``phi_e_proxy``, ``csd_proxy``, and ``lfp_proxy`` are
    proxy readouts.  They are not calibrated physical amplitudes.
    """

    source_proxy: jax.Array
    phi_e_proxy: jax.Array
    csd_proxy: jax.Array
    lfp_proxy: jax.Array
    kernel: jax.Array
    contact_depths: jax.Array
    diagnostics: dict[str, Any]

    @property
    def phi_e(self) -> jax.Array:
        """Compatibility alias for the proxy potential-like readout."""

        return self.phi_e_proxy

    @property
    def csd(self) -> jax.Array:
        """Compatibility alias for the proxy CSD-proxy readout."""

        return self.csd_proxy

    @property
    def lfp(self) -> jax.Array:
        """Compatibility alias for the proxy LFP-proxy readout."""

        return self.lfp_proxy


def _dtype_name(array: jax.Array) -> str:
    return str(array.dtype)


def _finite_bool(array: jax.Array) -> bool:
    return bool(jnp.all(jnp.isfinite(array)))


def _row_normalize(kernel: jax.Array, eps: float = 1e-8) -> jax.Array:
    return kernel / (jnp.sum(kernel, axis=1, keepdims=True) + eps)


def project_laminar_sources(
    sources: jax.Array,
    positions: jax.Array,
    *,
    n_contacts: int = 16,
    width: float = 0.10,
    dtype: str = "float32",
) -> FieldOutput:
    """Project source traces to laminar proxy contacts.

    Parameters
    ----------
    sources:
        Source-proxy traces with shape ``[T, N]``.
    positions:
        Emitter positions with shape ``[N, 3]``.  The third coordinate is
        interpreted as relative laminar depth in ``[0, 1]``.
    n_contacts:
        Number of laminar contacts.
    width:
        Gaussian width in relative laminar-depth units.
    dtype:
        Requested dtype policy.  Float64 is honored only when JAX x64 is enabled.
    """

    if dtype == "float64" and bool(jax.config.read("jax_enable_x64")):
        jdtype = jnp.float64
    else:
        jdtype = jnp.float32

    sources = jnp.asarray(sources, dtype=jdtype)
    positions = jnp.asarray(positions, dtype=jdtype)
    depth = positions[:, 2]
    contacts = jnp.linspace(0.0, 1.0, int(n_contacts), dtype=jdtype)
    width_value = jnp.asarray(width, dtype=jdtype)
    raw_kernel = jnp.exp(-0.5 * ((contacts[:, None] - depth[None, :]) / width_value) ** 2)
    kernel = _row_normalize(raw_kernel)

    source_proxy = sources @ kernel.T
    lfp_proxy = source_proxy
    phi_e_proxy = lfp_proxy

    # Proxy CSD-proxy readout from a laminar second derivative.  This is not a
    # PDE solution and is deliberately diagnosed as laminar_proxy_no_pde.
    dz = contacts[1] - contacts[0] if n_contacts > 1 else jnp.asarray(1.0, dtype=jdtype)
    first_depth_derivative = jnp.gradient(phi_e_proxy, dz, axis=1)
    csd_proxy = -jnp.gradient(first_depth_derivative, dz, axis=1)

    diagnostics = validate_projection_invariants(
        sources=sources,
        positions=positions,
        kernel=kernel,
        source_proxy=source_proxy,
        phi_e_proxy=phi_e_proxy,
        csd_proxy=csd_proxy,
        lfp_proxy=lfp_proxy,
    )
    diagnostics.update(
        {
            "field_solver_status": "laminar_proxy_no_pde",
            "field_solver": "laminar_proxy_no_pde",
            "source_projection_mode": "proxy_no_field_solve",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "source_decomposition": "proxy_reduced_emitter",
            "CSD_sign_convention": "proxy_positive_equals_extracellular_source_like",
            "physical_amplitude_claim_allowed": False,
            "field_claim_level": "proxy_readout_only",
        }
    )
    return FieldOutput(
        source_proxy=source_proxy,
        phi_e_proxy=phi_e_proxy,
        csd_proxy=csd_proxy,
        lfp_proxy=lfp_proxy,
        kernel=kernel,
        contact_depths=contacts,
        diagnostics=diagnostics,
    )


def project_sources_to_laminar_field(
    sources: jax.Array,
    positions: jax.Array,
    n_contacts: int = 16,
    *,
    dtype: str = "float32",
) -> FieldOutput:
    """Backward-compatible wrapper for v0.0.3 public name."""

    return project_laminar_sources(sources, positions, n_contacts=n_contacts, dtype=dtype)


def validate_projection_invariants(
    *,
    sources: jax.Array,
    positions: jax.Array,
    kernel: jax.Array,
    source_proxy: jax.Array,
    phi_e_proxy: jax.Array,
    csd_proxy: jax.Array,
    lfp_proxy: jax.Array,
) -> dict[str, Any]:
    """Return source/probe invariant diagnostics for the laminar proxy layer.

    v0.2.0: Includes field admissibility checks (finiteness, kernel normalization).
    """

    kernel_row_sums = jnp.sum(kernel, axis=1)
    kernel_row_sum_max_abs_error = jnp.max(jnp.abs(kernel_row_sums - 1.0))
    t_steps = int(sources.shape[0])
    n_emitters = int(sources.shape[1])
    n_contacts = int(kernel.shape[0])
    warnings: list[str] = []
    if positions.shape != (n_emitters, 3):
        warnings.append("positions_shape_not_N_by_3")
    if source_proxy.shape != (t_steps, n_contacts):
        warnings.append("source_proxy_shape_mismatch")
    if not _finite_bool(source_proxy):
        warnings.append("non_finite_source_proxy")
    if not _finite_bool(csd_proxy):
        warnings.append("non_finite_csd_proxy")

    return {
        "source_shape": tuple(int(x) for x in sources.shape),
        "positions_shape": tuple(int(x) for x in positions.shape),
        "kernel_shape": tuple(int(x) for x in kernel.shape),
        "source_proxy_shape": tuple(int(x) for x in source_proxy.shape),
        "phi_e_proxy_shape": tuple(int(x) for x in phi_e_proxy.shape),
        "csd_proxy_shape": tuple(int(x) for x in csd_proxy.shape),
        "lfp_proxy_shape": tuple(int(x) for x in lfp_proxy.shape),
        "dtype": _dtype_name(sources),
        "kernel_row_sum_max_abs_error": float(kernel_row_sum_max_abs_error),
        "finite_sources": _finite_bool(sources),
        "finite_positions": _finite_bool(positions),
        "finite_kernel": _finite_bool(kernel),
        "finite_source_proxy": _finite_bool(source_proxy),
        "finite_phi_e_proxy": _finite_bool(phi_e_proxy),
        "finite_csd_proxy": _finite_bool(csd_proxy),
        "finite_lfp_proxy": _finite_bool(lfp_proxy),
        # Compatibility keys for old status code.
        "finite_phi_e": _finite_bool(phi_e_proxy),
        "finite_CSD": _finite_bool(csd_proxy),
        # v0.2.0/v0.2.2: Field admissibility and proxy status
        "field_admissibility": {
            "field_arrays_finite": {
                "phi_e_finite": _finite_bool(phi_e_proxy),
                "csd_finite": _finite_bool(csd_proxy),
                "lfp_finite": _finite_bool(lfp_proxy),
            },
            # Backwards compatibility
            "kernel_normalization_valid": float(kernel_row_sum_max_abs_error) < 1e-6,
            "source_conservation_status": "proxy_not_solved",
            # v0.2.2: Explicit proxy metadata for clarity
            "kernel_row_stochastic_valid": float(kernel_row_sum_max_abs_error) < 1e-6,
            "kernel_normalization_definition": "contact_rows_sum_to_one_proxy",
            "source_current_conservation_status": "not_applicable_proxy_mode",
            "boundary_condition_status": "declared_metadata_only",
            "gauge_status": "declared_metadata_only",
        },
        "warnings": warnings,
    }


def validate_source_field_status(
    field_output: FieldOutput | None = None,
    cfg_metadata: Mapping[str, Any] | None = None,
    *,
    requested_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return truth-preserving status for source-field readouts.

    This is a status report, not a rejection gate.  It enforces the wording that
    laminar readouts are proxies until a real source-conserving PDE solve exists.
    """

    metadata = dict(cfg_metadata or {})
    diagnostics = dict(field_output.diagnostics) if field_output is not None else {}
    warnings = list(diagnostics.get("warnings", []))
    requested_modes = tuple(requested_modes or ())
    proxy_modes_requested = [mode for mode in requested_modes if mode in {"CSD", "LFP", "phi_e"}]
    if proxy_modes_requested:
        warnings.append("proxy_readout_modes_requested:" + ",".join(proxy_modes_requested))
    if "J_e" in requested_modes:
        warnings.append("J_e_not_computed_without_real_field_solver")

    source_projection_mode = metadata.get(
        "source_projection_mode", diagnostics.get("source_projection_mode", "proxy_no_field_solve")
    )
    source_decomposition = metadata.get(
        "source_decomposition", diagnostics.get("source_decomposition", "proxy_reduced_emitter")
    )
    source_calibration_status = metadata.get(
        "source_calibration_status",
        diagnostics.get("source_calibration_status", "uncalibrated_izhikevich_native_current"),
    )
    field_solver_status = metadata.get(
        "field_solver_status", diagnostics.get("field_solver_status", "laminar_proxy_no_pde")
    )

    source_status = str(source_calibration_status).lower()
    field_status = str(field_solver_status).lower()
    projection_status = str(source_projection_mode).lower()
    is_proxy = "proxy" in field_status or "no_pde" in field_status or "proxy" in projection_status
    is_calibrated = "calibrated" in source_status and "uncalibrated" not in source_status
    physical_amplitude_claim_allowed = False
    field_claim_level = "proxy_readout_only"
    if "uncalibrated" in source_status:
        warnings.append("uncalibrated_reduced_emitter_source")
    if is_proxy:
        warnings.append("laminar_proxy_no_pde_not_physical_field_solution")

    finite_flags = {
        "finite_source_proxy": diagnostics.get("finite_source_proxy"),
        "finite_phi_e_proxy": diagnostics.get("finite_phi_e_proxy"),
        "finite_csd_proxy": diagnostics.get("finite_csd_proxy"),
        "finite_lfp_proxy": diagnostics.get("finite_lfp_proxy"),
    }
    return {
        "source_projection_mode": source_projection_mode,
        "source_decomposition": source_decomposition,
        "source_calibration_status": source_calibration_status,
        "boundary_condition": metadata.get("boundary_condition", "mean_zero_neumann"),
        "gauge": metadata.get("gauge", "mean_zero"),
        "csd_sign_convention": metadata.get(
            "csd_sign_convention", "proxy_positive_equals_extracellular_source_like"
        ),
        "field_solver_status": field_solver_status,
        "field_claim_level": field_claim_level,
        "physical_amplitude_claim_allowed": physical_amplitude_claim_allowed,
        "is_proxy": bool(is_proxy),
        "is_calibrated": bool(is_calibrated),
        "validation_status": "proxy_status_report_only",
        "finite_flags": finite_flags,
        "warnings": sorted(set(str(w) for w in warnings)),
    }


# ─── v0.2.1 Multimodal Probe Operators ────────────────────────────────────────


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
        import json
        from .io import json_safe
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
    field_solver_status: str = "laminar_proxy_no_pde",
    field_claim_level: str = "proxy_readout_only",
    source_calibration_status: str = "uncalibrated_izhikevich_native_current",
    source_projection_mode: str = "proxy_no_field_solve",
    source_decomposition: str = "proxy_reduced_emitter",
    assumptions: list[str] = None,
    extra_fields: dict = None,
) -> dict:
    """Build a JSON-safe probe operator report.

    Parameters
    ----------
    kind : str
        Operator kind: spk, vm, source, lfp_proxy, csd_proxy, eeg_proxy, meg_proxy, emm_proxy.
    method : str
        Description of the method (e.g., "threshold_or_emitter_spike_array").
    operator_status : str
        Status: simulated_proxy, physical_forward_model, or calibrated_empirical.
    data_shape : tuple or str
        Output array shape.
    units_or_status : str
        Units or status string.
    calibration_status : str
        Calibration status.
    field_solver_status : str
        Field solver status.
    field_claim_level : str
        Field claim level.
    source_calibration_status : str
        Source calibration status.
    source_projection_mode : str
        Source projection mode.
    source_decomposition : str
        Source decomposition.
    assumptions : list[str]
        List of assumptions.
    extra_fields : dict
        Additional fields to merge into report.

    Returns
    -------
    dict
        JSON-safe report.
    """
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
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "source_calibration_status": source_calibration_status,
        "source_projection_mode": source_projection_mode,
        "source_decomposition": source_decomposition,
        "field_solver_status": field_solver_status,
        "field_claim_level": field_claim_level,
        "physical_amplitude_claim_allowed": False,
        "assumptions": assumptions,
    }

    if extra_fields:
        report.update(extra_fields)

    return report


def spk_probe(spikes: jax.Array) -> ProbeReadout:
    """SPK probe operator: expose spike events or spike matrix.

    Parameters
    ----------
    spikes : jax.Array
        Spike array with shape [T, N] (time × neurons).

    Returns
    -------
    ProbeReadout
        Readout with spike data and JSON-safe report.
    """
    spikes = jnp.asarray(spikes)
    report = _make_probe_report(
        kind="spk",
        method="threshold_or_emitter_spike_array",
        data_shape=spikes.shape,
        units_or_status="binary_spike_indicator",
        assumptions=[
            "spike_array_from_emitter_or_threshold",
            "binary_or_threshold_values",
        ],
    )
    return ProbeReadout(name="spk", kind="spk", data=spikes, report=report)


def vm_probe(voltage: jax.Array) -> ProbeReadout:
    """Vm probe operator: expose membrane voltage or native reduced-emitter state.

    Parameters
    ----------
    voltage : jax.Array
        Voltage/state array with shape [T, N] (time × neurons).

    Returns
    -------
    ProbeReadout
        Readout with voltage data and JSON-safe report.
    """
    voltage = jnp.asarray(voltage)
    report = _make_probe_report(
        kind="vm",
        method="emitter_state_voltage_trace",
        data_shape=voltage.shape,
        units_or_status="mV_or_native_model_voltage",
        assumptions=[
            "voltage_from_emitter_native_state",
            "not_physical_membrane_voltage_unless_calibrated",
        ],
    )
    return ProbeReadout(name="vm", kind="vm", data=voltage, report=report)


def source_probe(source: jax.Array) -> ProbeReadout:
    """Source probe operator: expose current/source proxy.

    Parameters
    ----------
    source : jax.Array
        Source/current proxy with shape [T, N] (time × neurons).

    Returns
    -------
    ProbeReadout
        Readout with source data and JSON-safe report.
    """
    source = jnp.asarray(source)
    report = _make_probe_report(
        kind="source",
        method="declared_source_projection_or_proxy",
        data_shape=source.shape,
        units_or_status="native_current_units_or_proxy",
        source_decomposition="proxy_reduced_emitter",
        assumptions=[
            "source_from_emitter_native_state",
            "not_physical_membrane_current_unless_calibrated",
        ],
    )
    return ProbeReadout(name="source", kind="source", data=source, report=report)


def lfp_proxy_probe(
    phi_e: jax.Array,
    contact_depths: jax.Array = None,
) -> ProbeReadout:
    """LFP-proxy probe operator: sample extracellular potential-like state.

    Parameters
    ----------
    phi_e : jax.Array
        Potential proxy array with shape [T, n_contacts].
    contact_depths : jax.Array, optional
        Contact depths for metadata.

    Returns
    -------
    ProbeReadout
        Readout with LFP-proxy data and JSON-safe report.
    """
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
    """CSD-proxy probe operator: estimate source-profile-like CSD-proxy readout.

    Parameters
    ----------
    csd : jax.Array
        CSD proxy array with shape [T, n_contacts].
    csd_sign_convention : str
        CSD sign convention declaration.

    Returns
    -------
    ProbeReadout
        Readout with CSD-proxy data and JSON-safe report.
    """
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
    """EEG-proxy probe operator: simulated scalp-channel EEG-proxy readout.

    Parameters
    ----------
    eeg : jax.Array
        EEG proxy array with shape [T, n_sensors].
    leadfield_status : str
        Status of the leadfield projection.
    n_sensors : int, optional
        Number of EEG sensors.

    Returns
    -------
    ProbeReadout
        Readout with EEG-proxy data and JSON-safe report.
    """
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
    """MEG-proxy probe operator: simulated magnetometer MEG-proxy readout.

    Parameters
    ----------
    meg : jax.Array
        MEG proxy array with shape [T, n_sensors].
    leadfield_status : str
        Status of the leadfield projection.
    orientation_convention : str
        Current orientation convention.
    n_sensors : int, optional
        Number of MEG sensors.

    Returns
    -------
    ProbeReadout
        Readout with MEG-proxy data and JSON-safe report.
    """
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
    """EMM-proxy probe operator: electromagnetic metabolism estimate proxy.

    Parameters
    ----------
    emm : jax.Array
        EMM-proxy array, typically 1D (time,) or small shape.
    method : str
        Method description.

    Returns
    -------
    ProbeReadout
        Readout with EMM-proxy data and JSON-safe report.
    """
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


def probe_laminar_modes(
    field_output: FieldOutput,
    modes: Sequence[str],
) -> dict[str, Any]:
    """Extract public readout modes from a laminar proxy field output."""

    out: dict[str, Any] = {}
    if "source" in modes or "sources" in modes:
        out["source_proxy"] = field_output.source_proxy
    if "phi_e" in modes:
        out["phi_e_proxy"] = field_output.phi_e_proxy
    if "CSD" in modes:
        out["CSD"] = field_output.csd_proxy
        out["csd_proxy"] = field_output.csd_proxy
    if "LFP" in modes:
        out["LFP"] = field_output.lfp_proxy
        out["lfp_proxy"] = field_output.lfp_proxy
    if "J_e" in modes:
        out["J_e_status"] = "not_computed_without_real_field_solver"
    if any(mode in {"source", "sources", "phi_e", "CSD", "LFP", "J_e"} for mode in modes):
        out["readout_metadata"] = validate_source_field_status(
            field_output, requested_modes=modes
        )
    return out
