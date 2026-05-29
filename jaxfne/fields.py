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


def _test_kernel_row_normalization(kernel: jax.Array, tol: float = 1e-6) -> dict[str, bool | float]:
    """Test that kernel is row-stochastic (rows sum to 1.0).

    Returns diagnostics dict with row normalization validation results.
    """
    row_sums = jnp.sum(kernel, axis=1)
    max_abs_error = float(jnp.max(jnp.abs(row_sums - 1.0)))
    is_valid = max_abs_error < tol

    return {
        "kernel_row_normalization_valid": is_valid,
        "kernel_row_sum_max_abs_error": max_abs_error,
        "kernel_row_sum_tolerance": tol,
        "kernel_row_stochastic_valid": is_valid,
    }


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

    # Build field solution report using hardened contract
    field_solution_report = _make_field_solution_report(
        field_solver_status="laminar_proxy_no_pde",
        solver_name="laminar_proxy",
        boundary_condition="mean_zero_neumann",
        gauge="mean_zero",
        csd_sign_convention="positive_equals_extracellular_source",
        current_density_layout="not_applicable",
        solver_residual_l2_relative=None,
        n_iterations=None,
        converged=None,
        finite_phi_e=_finite_bool(phi_e_proxy),
        finite_J_e=False,  # Not computed in proxy mode
        finite_CSD=_finite_bool(csd_proxy),
        field_claim_level="proxy_readout_only",
        physical_amplitude_claim_allowed=False,
        source_projection_mode="proxy_no_field_solve",
        source_current_conservation_status="not_applicable_proxy_mode",
        source_conservation_tested=False,
        source_conservation_claim_allowed=False,
    )

    # Merge field solution report into diagnostics
    diagnostics.update(field_solution_report)
    diagnostics.update(
        {
            "field_solver": "laminar_proxy_no_pde",
            "source_projection_status": "contact_row_normalized_proxy",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "source_decomposition": "proxy_reduced_emitter",
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
    v0.2.4: Explicit source conservation and projection status diagnostics.
    """

    # v0.2.4: Test kernel row normalization explicitly
    kernel_norm_tests = _test_kernel_row_normalization(kernel, tol=1e-6)
    kernel_row_sum_max_abs_error = kernel_norm_tests["kernel_row_sum_max_abs_error"]

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
        # v0.2.0/v0.2.2/v0.2.4: Field admissibility and proxy status
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
            "source_current_conservation_test": "not_applicable_proxy_mode",
            "boundary_condition_status": "declared_metadata_only",
            "gauge_status": "declared_metadata_only",
            # v0.2.4: Explicit kernel normalization diagnostics
            "kernel_row_normalization_valid": kernel_norm_tests["kernel_row_normalization_valid"],
            "kernel_row_sum_max_abs_error_v024": kernel_norm_tests["kernel_row_sum_max_abs_error"],
            "kernel_row_sum_tolerance_v024": kernel_norm_tests["kernel_row_sum_tolerance"],
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
            "csd_sign_convention", "positive_equals_extracellular_source"
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


def _make_field_solution_report(
    field_solver_status: str = "laminar_proxy_no_pde",
    solver_name: str = "laminar_proxy",
    boundary_condition: str = "declared_metadata_only",
    gauge: str = "declared_metadata_only",
    csd_sign_convention: str = "positive_equals_extracellular_source",
    current_density_layout: str = "not_applicable",
    solver_residual_l2_relative: float | None = None,
    n_iterations: int | None = None,
    converged: bool | None = None,
    finite_phi_e: bool = True,
    finite_J_e: bool = True,
    finite_CSD: bool = True,
    field_claim_level: str = "proxy_readout_only",
    physical_amplitude_claim_allowed: bool = False,
    source_projection_mode: str = "proxy_no_field_solve",
    source_current_conservation_status: str = "not_applicable_proxy_mode",
    source_conservation_tested: bool = False,
    source_conservation_claim_allowed: bool = False,
) -> dict:
    """Build a JSON-safe field solution report.

    Parameters
    ----------
    field_solver_status : str
        Field solver status: laminar_proxy_no_pde, solved_resistive_poisson, solved_bidomain, or not_implemented.
    solver_name : str
        Human-readable solver name.
    boundary_condition : str
        Boundary condition declaration.
    gauge : str
        Gauge convention declaration.
    csd_sign_convention : str
        CSD sign convention: positive_equals_extracellular_source.
    current_density_layout : str
        Current density layout status.
    solver_residual_l2_relative : float or None
        Relative L2 residual of solver (None for proxy).
    n_iterations : int or None
        Number of solver iterations (None for proxy).
    converged : bool or None
        Whether solver converged (None for proxy).
    finite_phi_e : bool
        Whether phi_e values are finite.
    finite_J_e : bool
        Whether J_e values are finite.
    finite_CSD : bool
        Whether CSD values are finite.
    field_claim_level : str
        Field claim level: proxy_readout_only, physical_admissible_candidate, or empirical_candidate.
    physical_amplitude_claim_allowed : bool
        Whether physical amplitude claims are allowed (False for proxy).
    source_projection_mode : str
        Source projection mode declaration.
    source_current_conservation_status : str
        Source conservation status.
    source_conservation_tested : bool
        Whether conservation was tested.
    source_conservation_claim_allowed : bool
        Whether conservation claims are allowed.

    Returns
    -------
    dict
        JSON-safe field solution report with 18+ required fields.
    """
    report = {
        "field_solver_status": field_solver_status,
        "solver_name": solver_name,
        "boundary_condition": boundary_condition,
        "gauge": gauge,
        "csd_sign_convention": csd_sign_convention,
        "current_density_layout": current_density_layout,
        "solver_residual_l2_relative": solver_residual_l2_relative,
        "n_iterations": n_iterations,
        "converged": converged,
        "finite_phi_e": finite_phi_e,
        "finite_J_e": finite_J_e,
        "finite_CSD": finite_CSD,
        "field_claim_level": field_claim_level,
        "physical_amplitude_claim_allowed": physical_amplitude_claim_allowed,
        "source_projection_mode": source_projection_mode,
        "source_current_conservation_status": source_current_conservation_status,
        "source_conservation_tested": source_conservation_tested,
        "source_conservation_claim_allowed": source_conservation_claim_allowed,
    }

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


# Real proxy operator implementations
# ====================================

def eeg_proxy_transform(
    source: jax.Array,
    leadfield: jax.Array,
) -> jax.Array:
    """Compute EEG-proxy readout via linear leadfield projection.

    Mathematical model:
        Y_eeg[c,t] = sum_k L_eeg[c,k] * S[t,k]

    where:
    - Y_eeg[C, T]: EEG-proxy readout (C sensors, T timesteps)
    - L_eeg[C, K]: Leadfield matrix (C sensors, K sources)
    - S[T, K]: Source proxy array (T timesteps, K sources)

    Parameters
    ----------
    source : jax.Array
        Source proxy array with shape [T, K].
    leadfield : jax.Array
        EEG leadfield matrix with shape [C, K].

    Returns
    -------
    jax.Array
        EEG-proxy readout with shape [T, C].

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    source = jnp.asarray(source)
    leadfield = jnp.asarray(leadfield)

    if source.ndim != 2:
        raise ValueError(f"source must be 2D [T, K], got shape {source.shape}")
    if leadfield.ndim != 2:
        raise ValueError(f"leadfield must be 2D [C, K], got shape {leadfield.shape}")

    T, K = source.shape
    C, K_lead = leadfield.shape

    if K != K_lead:
        raise ValueError(
            f"source and leadfield K dimension mismatch: {K} vs {K_lead}"
        )

    # Y_eeg[T, C] = S[T, K] @ L[C, K].T
    y_eeg = source @ leadfield.T  # [T, K] @ [K, C] = [T, C]
    return y_eeg


def meg_proxy_transform(
    source_oriented: jax.Array,
    leadfield: jax.Array,
) -> jax.Array:
    """Compute MEG-proxy readout via linear leadfield projection.

    Mathematical model:
        Y_meg[c,t] = sum_k L_meg[c,k] * J_oriented[t,k]

    where:
    - Y_meg[C, T]: MEG-proxy readout (C sensors, T timesteps)
    - L_meg[C, K]: MEG leadfield matrix (C sensors, K sources)
    - J_oriented[T, K]: Oriented source activity (T timesteps, K sources)

    Parameters
    ----------
    source_oriented : jax.Array
        Oriented source/activity array with shape [T, K].
    leadfield : jax.Array
        MEG leadfield matrix with shape [C, K].

    Returns
    -------
    jax.Array
        MEG-proxy readout with shape [T, C].

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    source_oriented = jnp.asarray(source_oriented)
    leadfield = jnp.asarray(leadfield)

    if source_oriented.ndim != 2:
        raise ValueError(f"source_oriented must be 2D [T, K], got shape {source_oriented.shape}")
    if leadfield.ndim != 2:
        raise ValueError(f"leadfield must be 2D [C, K], got shape {leadfield.shape}")

    T, K = source_oriented.shape
    C, K_lead = leadfield.shape

    if K != K_lead:
        raise ValueError(
            f"source_oriented and leadfield K dimension mismatch: {K} vs {K_lead}"
        )

    # Y_meg[T, C] = J_oriented[T, K] @ L[C, K].T
    y_meg = source_oriented @ leadfield.T  # [T, K] @ [K, C] = [T, C]
    return y_meg


def emm_proxy_transform(
    spike_rate: jax.Array,
    source: jax.Array,
    field_potential: jax.Array,
    lambda_spk: float = 1.0,
    lambda_src: float = 1.0,
    lambda_field: float = 1.0,
) -> jax.Array:
    """Compute EMM-proxy (normalized activity/source/field cost) readout.

    Mathematical model:
        E_proxy[t] = (lambda_spk * R_spk[t] + lambda_src * ||S[t]||_1
                      + lambda_field * ||Phi[t]||_2^2) / normalization

    where:
    - E_proxy[T]: Normalized cost trace (T timesteps)
    - R_spk[T]: Spike rate or activity trace
    - S[T, K]: Source proxy array
    - Phi[T, X]: Field potential proxy array
    - lambda_*: Weighting factors (default 1.0)

    Parameters
    ----------
    spike_rate : jax.Array
        Spike rate or activity array with shape [T] or [T, 1].
    source : jax.Array
        Source proxy array with shape [T, K].
    field_potential : jax.Array
        Field potential proxy array with shape [T, X].
    lambda_spk : float
        Weight for spike-rate term (default 1.0).
    lambda_src : float
        Weight for source L1-norm term (default 1.0).
    lambda_field : float
        Weight for field L2-norm term (default 1.0).

    Returns
    -------
    jax.Array
        EMM-proxy cost trace with shape [T] or [T, 1].

    Raises
    ------
    ValueError
        If array shapes are incompatible.
    """
    spike_rate = jnp.asarray(spike_rate)
    source = jnp.asarray(source)
    field_potential = jnp.asarray(field_potential)

    # Normalize input shapes
    if spike_rate.ndim == 1:
        spike_rate = spike_rate[:, None]  # [T] -> [T, 1]
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

    # Compute weighted cost terms
    # Term 1: Spike rate (already shape [T, 1])
    term_spk = lambda_spk * spike_rate

    # Term 2: Source L1-norm per timestep [T, K] -> [T, 1]
    source_l1 = jnp.sum(jnp.abs(source), axis=1, keepdims=True)
    term_src = lambda_src * source_l1

    # Term 3: Field L2-norm (Frobenius norm) per timestep [T, X] -> [T, 1]
    field_l2_sq = jnp.sum(jnp.square(field_potential), axis=1, keepdims=True)
    term_field = lambda_field * field_l2_sq

    # Total cost [T, 1]
    total_cost = term_spk + term_src + term_field

    # Normalize by sum of weights (for relative comparison)
    total_weight = lambda_spk + lambda_src + lambda_field
    if total_weight > 0:
        emm_proxy = total_cost / total_weight
    else:
        emm_proxy = total_cost

    return emm_proxy


def compute_conservation_proxy_diagnostics(
    *,
    source: "jax.Array | None" = None,
    phi_e: "jax.Array | None" = None,
    csd: "jax.Array | None" = None,
    lfp: "jax.Array | None" = None,
    field_solution: "FieldOutput | None" = None,
    source_calibration_status: str = "uncalibrated_izhikevich_native_current",
    field_solver_status: str = "laminar_proxy_no_pde",
    field_claim_level: str = "proxy_readout_only",
) -> dict[str, Any]:
    """Compute conservation-inspired proxy diagnostics over existing source/field arrays.

    v0.2.27 proxy diagnostics — no Poisson solver, no Maxwell solver, no physical
    amplitude claims.  All values are derived from the existing laminar-proxy arrays
    (source_proxy, phi_e_proxy, csd_proxy, lfp_proxy) already produced by the
    pipeline.  Nothing is fabricated; missing arrays yield ``None``.

    Parameters
    ----------
    source:
        Source proxy array [T, N] or [T, X].  Optional.
    phi_e:
        Potential-like proxy array [T, X].  Optional.
    csd:
        CSD proxy array [T, X].  Optional.
    lfp:
        LFP proxy array [T, X].  Optional.
    field_solution:
        ``FieldOutput`` object.  If provided, arrays are extracted from it unless
        the explicit keyword arguments override.
    source_calibration_status:
        Calibration status string; passed through to output.
    field_solver_status:
        Solver status string; must not claim solved state for proxy runs.
    field_claim_level:
        Claim level; must be ``"proxy_readout_only"`` for v0.2.x.

    Returns
    -------
    dict
        JSON-safe dict with proxy diagnostic scalars and explicit non-implementation
        markers for Poisson, Maxwell, Poynting, and stress-energy machinery.

    Notes
    -----
    Claim boundaries:

    * ``physical_amplitude_claim_allowed``: always ``False``.
    * ``biological_metabolism_claim_allowed``: always ``False``.
    * ``j_dot_e_proxy``: ``None`` — J_e is not computed in proxy mode.
    * ``poynting_flux_proxy``: ``None`` — unavailable in laminar proxy mode.
    * ``poisson_solver_status``: ``"not_implemented"`` — no solver in v0.2.27.
    * ``maxwell_solver_status``: ``"not_implemented"`` — no solver in v0.2.x.
    * ``stress_energy_tensor_status``: ``"not_implemented"`` — no solver in v0.2.x.
    """

    def _safe_float(v: "jax.Array") -> float:
        """Return finite float or raise."""
        f = float(v)
        if not (f == f) or abs(f) == float("inf"):
            raise ValueError(f"non-finite diagnostic value: {f}")
        return f

    def _norm_l1(arr: "jax.Array") -> "float | None":
        try:
            return _safe_float(jnp.mean(jnp.abs(arr)))
        except Exception:
            return None

    def _norm_l2(arr: "jax.Array") -> "float | None":
        try:
            return _safe_float(jnp.sqrt(jnp.mean(arr ** 2)))
        except Exception:
            return None

    def _abs_mean(arr: "jax.Array") -> "float | None":
        try:
            return _safe_float(jnp.mean(jnp.abs(arr)))
        except Exception:
            return None

    # Resolve arrays: explicit kwargs override field_solution
    _src = source
    _phi = phi_e
    _csd = csd
    _lfp = lfp

    if field_solution is not None:
        if _src is None:
            _src = field_solution.source_proxy
        if _phi is None:
            _phi = field_solution.phi_e_proxy
        if _csd is None:
            _csd = field_solution.csd_proxy
        if _lfp is None:
            _lfp = field_solution.lfp_proxy

    # Convert non-None arrays to JAX arrays, reject nonfinite
    def _coerce(arr: Any) -> "jax.Array | None":
        if arr is None:
            return None
        a = jnp.asarray(arr)
        if not _finite_bool(a):
            return None  # nonfinite — diagnostics are not available
        return a

    _src = _coerce(_src)
    _phi = _coerce(_phi)
    _csd = _coerce(_csd)
    _lfp = _coerce(_lfp)

    # ── Source diagnostics ────────────────────────────────────────────────────
    source_norm_l1: "float | None" = _norm_l1(_src) if _src is not None else None
    source_norm_l2: "float | None" = _norm_l2(_src) if _src is not None else None
    source_abs_mean: "float | None" = _abs_mean(_src) if _src is not None else None

    # Source conservation proxy residual: proxy for ∫q dV ≈ 0.
    # Computed as mean(|mean_space(q)(t)|) — measures how far the spatial mean
    # departs from zero at each timestep, then averages over time.
    # This is a proxy only; the true conservation integral requires a solved field.
    source_conservation_proxy_residual: "float | None" = None
    if _src is not None:
        try:
            spatial_mean = jnp.mean(_src, axis=-1)          # [T]
            source_conservation_proxy_residual = _safe_float(jnp.mean(jnp.abs(spatial_mean)))
        except Exception:
            source_conservation_proxy_residual = None

    # ── Potential-field diagnostics ───────────────────────────────────────────
    phi_abs_mean: "float | None" = _abs_mean(_phi) if _phi is not None else None

    # phi_gradient_proxy_norm2: proxy for |∇φ_e|² — derived from existing phi_e_proxy
    # via finite differences along the spatial axis (axis=1).
    # Only computed when phi_e has a spatial axis (ndim >= 2).
    phi_gradient_proxy_norm2: "float | None" = None
    if _phi is not None and _phi.ndim >= 2 and _phi.shape[1] > 1:
        try:
            grad = jnp.gradient(_phi, axis=1)
            phi_gradient_proxy_norm2 = _safe_float(jnp.mean(grad ** 2))
        except Exception:
            phi_gradient_proxy_norm2 = None

    # ── CSD diagnostics ───────────────────────────────────────────────────────
    csd_abs_mean: "float | None" = _abs_mean(_csd) if _csd is not None else None
    csd_norm_l2: "float | None" = _norm_l2(_csd) if _csd is not None else None

    # ── LFP diagnostics ───────────────────────────────────────────────────────
    lfp_abs_mean: "float | None" = _abs_mean(_lfp) if _lfp is not None else None
    lfp_norm_l2: "float | None" = _norm_l2(_lfp) if _lfp is not None else None

    # ── Field-energy-like proxy ───────────────────────────────────────────────
    # Proxy for ∫|∇φ_e|² dV — combines phi gradient norm with volume proxy.
    # This is NOT a physical field energy; no calibrated conductivity exists.
    field_energy_like_proxy: "float | None" = phi_gradient_proxy_norm2  # same quantity

    return {
        "diagnostic_status": "proxy",
        "diagnostic_version": "v0.2.27",
        "claim_level": "computational_scaffold",
        "field_solver_status": str(field_solver_status),
        "field_claim_level": str(field_claim_level),
        "source_calibration_status": str(source_calibration_status),
        "physical_amplitude_claim_allowed": False,
        "biological_metabolism_claim_allowed": False,
        # ── Source diagnostics ────────────────────────────────────────────────
        "source_norm_l1": source_norm_l1,
        "source_norm_l2": source_norm_l2,
        "source_abs_mean": source_abs_mean,
        "source_conservation_proxy_residual": source_conservation_proxy_residual,
        # ── Potential-field diagnostics ───────────────────────────────────────
        "phi_abs_mean": phi_abs_mean,
        "phi_gradient_proxy_norm2": phi_gradient_proxy_norm2,
        # ── CSD diagnostics ───────────────────────────────────────────────────
        "csd_abs_mean": csd_abs_mean,
        "csd_norm_l2": csd_norm_l2,
        # ── LFP diagnostics ───────────────────────────────────────────────────
        "lfp_abs_mean": lfp_abs_mean,
        "lfp_norm_l2": lfp_norm_l2,
        # ── Field-energy-like proxy ───────────────────────────────────────────
        "field_energy_like_proxy": field_energy_like_proxy,
        # ── Explicitly not-implemented gates ─────────────────────────────────
        # J·E is not computed: J_e is not produced in proxy mode.
        "j_dot_e_proxy": None,
        # Poynting flux: future doctrine only; not computed in v0.2.27.
        "poynting_flux_proxy": None,
        # All solver/dynamics machinery remains gated.
        "stress_energy_tensor_status": "not_implemented",
        "poisson_solver_status": "not_implemented",
        "maxwell_solver_status": "not_implemented",
        "notes": [
            "proxy diagnostics only — no physical amplitude claim",
            "no biological metabolism claim",
            "no Poisson solver in v0.2.27",
            "Maxwell/Poynting/stress-energy quantities are unavailable in laminar proxy mode",
            "j_dot_e_proxy is None: J_e not computed in laminar_proxy_no_pde mode",
            "source_conservation_proxy_residual is a spatial-mean proxy, not PDE-enforced",
        ],
    }


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


# =============================================================================
# Patch B: Connectivity and Synapse Operators
# =============================================================================

def make_laminar_connectivity(
    neurons_df: Any,
    positions_m: jax.Array,
    control_params: Mapping[str, float] | None = None,
    seed: int = 0,
    local_decay_m: float = 0.001,
    p_local_e: float = 0.18,
    p_local_i: float = 0.30,
    p_feedforward: float = 0.060,
    p_feedback: float = 0.055,
    w_e_range: tuple[float, float] = (0.012, 0.055),
    w_i_range: tuple[float, float] = (-0.145, -0.055),
    w_ff_range: tuple[float, float] = (0.007, 0.030),
    w_fb_range: tuple[float, float] = (0.006, 0.026),
) -> dict[str, Any]:
    """Construct laminar connectivity matrix with control-dependent gains."""
    import numpy as np

    if control_params is None:
        control_params = {
            "local_exc_gain": 1.0,
            "local_inh_gain": 1.0,
            "feedforward_gain": 1.0,
            "feedback_gain": 1.0,
        }

    n = len(neurons_df)
    area = np.array(neurons_df.get("area", [""]*n))
    layer = np.array(neurons_df.get("layer", [""]*n))
    cell_type = np.array(neurons_df.get("cell_type", [""]*n))

    W_local_exc = np.zeros((n, n), dtype=np.float32)
    W_local_inh = np.zeros((n, n), dtype=np.float32)
    W_ff = np.zeros((n, n), dtype=np.float32)
    W_fb = np.zeros((n, n), dtype=np.float32)

    rng = np.random.default_rng(seed)
    area_order = sorted(set(area[area != ""]))
    area_rank = {a: i for i, a in enumerate(area_order)}

    for pre in range(n):
        for post in range(n):
            if pre == post:
                continue

            same_area = area[pre] == area[post]
            dxy = np.linalg.norm(positions_m[post, :2] - positions_m[pre, :2])
            local_gain = np.exp(-((dxy / local_decay_m) ** 2))

            if same_area:
                if cell_type[pre] == "E" and rng.random() < p_local_e:
                    W_local_exc[post, pre] = rng.uniform(*w_e_range) * local_gain
                elif cell_type[pre] != "E" and rng.random() < p_local_i:
                    W_local_inh[post, pre] = rng.uniform(*w_i_range) * (0.65 + 0.35 * local_gain)
            elif cell_type[pre] == "E":
                delta = area_rank.get(area[post], 0) - area_rank.get(area[pre], 0)
                if delta == 1 and layer[pre] in ("L2", "L3") and layer[post] == "L4" and rng.random() < p_feedforward:
                    W_ff[post, pre] = rng.uniform(*w_ff_range)
                if delta == -1 and layer[pre] in ("L2", "L3", "L6") and layer[post] in ("L5", "L6") and rng.random() < p_feedback:
                    W_fb[post, pre] = rng.uniform(*w_fb_range)

    W = (
        control_params.get("local_exc_gain", 1.0) * W_local_exc
        + control_params.get("local_inh_gain", 1.0) * W_local_inh
        + control_params.get("feedforward_gain", 1.0) * W_ff
        + control_params.get("feedback_gain", 1.0) * W_fb
    )

    E_mask = np.array([cell_type[i] == "E" for i in range(n)])
    I_mask = ~E_mask

    audit = {
        "total_neurons": n,
        "excitatory_neurons": int(np.sum(E_mask)),
        "inhibitory_neurons": int(np.sum(I_mask)),
        "nonzero_edges": int(np.count_nonzero(W)),
        "sum_abs_weight": float(np.sum(np.abs(W))),
        "control_gains": dict(control_params),
    }

    return {
        "W": jnp.asarray(W),
        "E_mask": jnp.asarray(E_mask),
        "I_mask": jnp.asarray(I_mask),
        "audit": audit,
    }


def exponential_synaptic_trace(
    spikes: jax.Array,
    tau_ms: float,
    dt_ms: float,
) -> jax.Array:
    """Compute exponential synaptic trace from spike times."""
    import math

    alpha = math.exp(-dt_ms / tau_ms)
    spikes_arr = jnp.asarray(spikes, dtype=jnp.float32)

    if spikes_arr.ndim == 1:
        state = jnp.zeros(1, dtype=jnp.float32)
        trace = []
        for spike in spikes_arr:
            state = alpha * state + spike
            trace.append(state)
        return jnp.array(trace)
    else:
        T, N = spikes_arr.shape
        state = jnp.zeros(N, dtype=jnp.float32)
        trace = []
        for t in range(T):
            state = alpha * state + spikes_arr[t]
            trace.append(state)
        return jnp.stack(trace)


def synaptic_current(
    spikes: jax.Array,
    W: jax.Array,
    tau_ms: float,
    dt_ms: float,
) -> jax.Array:
    """Compute synaptic current from spikes and connectivity matrix."""
    trace = exponential_synaptic_trace(spikes, tau_ms, dt_ms)
    return jnp.dot(trace, W.T)


# =============================================================================
# Patch D: Multi-Area Source Projector
# =============================================================================

def filtered_spike_source(
    spikes: jax.Array,
    neurons: "Mapping[str, Any]",
    tau_ms: float = 5.0,
    cell_signs: "Mapping[str, int] | None" = None,
) -> jax.Array:
    """Filter spikes through exponential decay per cell type.

    Parameters
    ----------
    spikes : jax.Array
        Spike raster [T, N].
    neurons : Mapping[str, Any]
        Neuron metadata with cell_type key.
    tau_ms : float
        Decay time constant (ms).
    cell_signs : Mapping[str, int], optional
        Sign per cell type (E=+1, I=-1). Default: auto-detect.

    Returns
    -------
    source : jax.Array
        Filtered source [T, N].
    """
    if cell_signs is None:
        cell_signs = {"E": 1, "PV": -1, "SST": -1, "VIP": -1}

    cell_types = neurons.get("cell_type", ["E"] * spikes.shape[1])
    dt_ms = 0.1  # Assume standard timestep; TODO: parametrize
    trace = exponential_synaptic_trace(spikes, tau_ms, dt_ms)

    # Apply signs per cell type
    signs = jnp.array([cell_signs.get(ct, 1) for ct in cell_types], dtype=jnp.float32)
    return trace * signs[None, :]


def synaptic_resonance_source(
    neurons: "Mapping[str, Any]",
    n_steps: int,
    dt_ms: float = 0.1,
    control_params: "Mapping[str, float] | None" = None,
) -> jax.Array:
    """Generate oscillatory source with layer and area specificity.

    Parameters
    ----------
    neurons : Mapping[str, Any]
        Neuron metadata with area, layer, cell_type keys.
    n_steps : int
        Number of timesteps.
    dt_ms : float
        Timestep in ms (default 0.1).
    control_params : Mapping[str, float], optional
        Control parameters including:
        - alpha_beta_gain: amplitude for 10-25 Hz (default 1.0)
        - gamma_gain: amplitude for 70-120 Hz (default 1.0)
        - resonance_scale: global scaling (default 1.0)

    Returns
    -------
    resonance : jax.Array
        Oscillatory source [T, N].
    """
    import numpy as np

    if control_params is None:
        control_params = {
            "alpha_beta_gain": 1.0,
            "gamma_gain": 1.0,
            "resonance_scale": 1.0,
        }

    n = len(neurons.get("area", []))
    if n == 0:
        return jnp.zeros((n_steps, 1), dtype=jnp.float32)

    areas = np.array(neurons.get("area", ["V1"] * n))
    layers = np.array(neurons.get("layer", ["L4"] * n))

    # Time vector
    t = np.arange(n_steps) * dt_ms / 1000.0  # Convert to seconds

    # Default frequencies
    alpha_beta_freq = 15.0  # Hz
    gamma_freq = 90.0  # Hz

    resonance = np.zeros((n_steps, n), dtype=np.float32)

    for i in range(n):
        area = areas[i]
        layer = layers[i]

        # Layer-specific alpha/beta amplitude
        if layer in ("L1", "L2", "L3"):
            alpha_beta_amp = 0.5
        elif layer == "L4":
            alpha_beta_amp = 0.3
        else:  # L5, L6
            alpha_beta_amp = 0.6

        # Gamma stronger in superficial layers
        if layer in ("L1", "L2", "L3"):
            gamma_amp = 0.8
        else:
            gamma_amp = 0.2

        # Generate oscillations
        alpha_beta_sig = alpha_beta_amp * np.sin(2.0 * np.pi * alpha_beta_freq * t)
        gamma_sig = gamma_amp * np.sin(2.0 * np.pi * gamma_freq * t)

        # Combine with control gains
        resonance[:, i] = (
            control_params.get("alpha_beta_gain", 1.0) * alpha_beta_sig
            + control_params.get("gamma_gain", 1.0) * gamma_sig
        ) * control_params.get("resonance_scale", 1.0)

    return jnp.asarray(resonance, dtype=jnp.float32)


def combined_multi_area_source(
    spikes: jax.Array,
    neurons: "Mapping[str, Any]",
    n_steps: int,
    dt_ms: float = 0.1,
    control_params: "Mapping[str, float] | None" = None,
    spike_tau_ms: float = 5.0,
) -> jax.Array:
    """Combine filtered spikes and resonance into single source tensor.

    Parameters
    ----------
    spikes : jax.Array
        Spike raster [T, N].
    neurons : Mapping[str, Any]
        Neuron metadata.
    n_steps : int
        Number of timesteps.
    dt_ms : float
        Timestep in ms.
    control_params : Mapping[str, float], optional
        Control parameters including:
        - spike_source_scale: weighting on spike component (default 1.0)
        - resonance_source_scale: weighting on resonance (default 1.0)
    spike_tau_ms : float
        Decay constant for spike filtering (default 5.0).

    Returns
    -------
    source : jax.Array
        Combined source [T, N].

    Metadata
    --------
    Scope: simulated source tensor for relative spectrolaminar profiling.
    Evidence: finite source arrays.
    Interpretation: emitter activity mapped to laminar readout basis.
    """
    if control_params is None:
        control_params = {
            "spike_source_scale": 1.0,
            "resonance_source_scale": 1.0,
        }

    # Filtered spike component
    spike_src = filtered_spike_source(spikes, neurons, tau_ms=spike_tau_ms)

    # Resonance component
    resonance = synaptic_resonance_source(neurons, n_steps, dt_ms, control_params)

    # Combine
    source = (
        control_params.get("spike_source_scale", 1.0) * spike_src
        + control_params.get("resonance_source_scale", 1.0) * resonance
    )

    return jnp.asarray(source, dtype=jnp.float32)


# =============================================================================
# Patch E: Spectrolaminar Probe and Readout
# =============================================================================

def spectrolaminar_psd(
    signal: "jax.Array",
    dt_ms: float = 0.1,
    freq_min: float = 1.0,
    freq_max: float = 150.0,
    n_freqs: int = 128,
) -> tuple["jax.Array", "jax.Array"]:
    """Compute power spectral density along laminar profile.

    Parameters
    ----------
    signal : jax.Array
        LFP or source signal [T, N_contacts].
    dt_ms : float
        Sampling interval in ms (default 0.1).
    freq_min, freq_max : float
        Frequency bounds in Hz.
    n_freqs : int
        Number of frequency bins (default 128).

    Returns
    -------
    freqs : jax.Array
        Frequency axis [n_freqs].
    psd : jax.Array
        Power spectral density [n_freqs, N_contacts].
    """
    import numpy as np

    freqs = np.linspace(float(freq_min), float(freq_max), int(n_freqs))
    dt_s = float(dt_ms) / 1000.0
    fs = 1.0 / dt_s

    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]

    T, n_contacts = sig.shape
    psd = np.zeros((int(n_freqs), n_contacts), dtype=np.float32)

    for ci in range(n_contacts):
        x = sig[:, ci]
        for fi, freq in enumerate(freqs):
            k = freq / fs
            phase = 2.0 * np.pi * k * np.arange(T)
            psd[fi, ci] = np.abs(np.dot(x, np.exp(-1j * phase))) / max(T, 1)

    return jnp.asarray(freqs, dtype=jnp.float32), jnp.asarray(psd, dtype=jnp.float32)


def spectrolaminar_readout(
    signal: "jax.Array",
    neurons: "Mapping[str, Any]",
    area: str,
    dt_ms: float = 0.1,
    freq_min: float = 1.0,
    freq_max: float = 150.0,
    n_freqs: int = 128,
    n_contacts: "int | None" = None,
) -> dict[str, "Any"]:
    """Generate spectrolaminar readout for a given area.

    Parameters
    ----------
    signal : jax.Array
        Source or LFP signal [T, N].
    neurons : Mapping[str, Any]
        Neuron metadata with area, layer, positions.
    area : str
        Area label (e.g., "V1", "V4", "PFC").
    dt_ms : float
        Sampling interval in ms.
    freq_min, freq_max : float
        Frequency bounds in Hz.
    n_freqs : int
        Number of frequency bins.
    n_contacts : int, optional
        Number of contacts. If None, pool neurons by layer.

    Returns
    -------
    readout : dict[str, Any]
        Spectrolaminar readout with keys:
        - freq_hz: frequency axis
        - pos_from_l4: relative depth per contact
        - relative_power: PSD [n_freqs, n_contacts]
        - alpha_beta: power 10-25 Hz per contact
        - gamma: power 40-150 Hz per contact
        - contact_depths_m: absolute depth per contact
        - n_contacts: number of contacts
        - n_neurons: number of neurons in area
        - area: area label
    """
    import numpy as np

    # Extract neurons in this area
    areas = np.array(neurons.get("area", ["V1"] * signal.shape[1]))
    layers = np.array(neurons.get("layer", ["L4"] * signal.shape[1]))
    layer_indices = np.where(areas == area)[0]

    if len(layer_indices) == 0:
        # Return empty readout for missing area
        return {
            "freq_hz": np.linspace(freq_min, freq_max, n_freqs),
            "pos_from_l4": np.array([]),
            "relative_power": np.zeros((n_freqs, 0), dtype=np.float32),
            "alpha_beta": np.array([]),
            "gamma": np.array([]),
            "contact_depths_m": np.array([]),
            "n_contacts": 0,
            "n_neurons": 0,
            "area": area,
        }

    # Use neurons as contacts if n_contacts not specified
    if n_contacts is None:
        n_contacts = len(layer_indices)
    else:
        n_contacts = min(int(n_contacts), len(layer_indices))

    # Pool neurons by layer
    layer_set = sorted(set(layers[layer_indices]))
    contact_layers = layer_set[:n_contacts] if len(layer_set) > 0 else ["L4"]

    # Create contact signal (average across neurons in each layer)
    contact_signal = np.zeros((signal.shape[0], len(contact_layers)), dtype=np.float32)
    for ci, layer_label in enumerate(contact_layers):
        layer_neurons = layer_indices[np.where(layers[layer_indices] == layer_label)[0]]
        if len(layer_neurons) > 0:
            contact_signal[:, ci] = np.mean(
                np.asarray(signal[:, layer_neurons]), axis=1
            )

    # Compute PSD
    freqs, psd = spectrolaminar_psd(
        contact_signal, dt_ms=dt_ms, freq_min=freq_min, freq_max=freq_max, n_freqs=n_freqs
    )

    # Extract band power
    freqs_arr = np.asarray(freqs)
    alpha_beta_idx = (freqs_arr >= 10.0) & (freqs_arr <= 25.0)
    gamma_idx = (freqs_arr >= 40.0) & (freqs_arr <= 150.0)

    alpha_beta_power = np.sum(psd[alpha_beta_idx, :], axis=0)
    gamma_power = np.sum(psd[gamma_idx, :], axis=0)

    # Normalize
    total_power = np.sum(psd, axis=0) + 1e-10
    alpha_beta_profile = alpha_beta_power / total_power
    gamma_profile = gamma_power / total_power

    # Depth from L4 (assume L1-L6 order)
    layer_order = {"L1": 0, "L2": 1, "L3": 2, "L4": 3, "L5": 4, "L6": 5}
    depth_from_l4 = np.array(
        [layer_order.get(l, 3) - 3 for l in contact_layers], dtype=np.float32
    )

    # Absolute depth (simplified: 0.1 mm per layer)
    contact_depths = np.abs(depth_from_l4) * 0.1

    return {
        "freq_hz": np.asarray(freqs, dtype=np.float32),
        "pos_from_l4": depth_from_l4,
        "relative_power": np.asarray(psd, dtype=np.float32),
        "alpha_beta": np.asarray(alpha_beta_profile, dtype=np.float32),
        "gamma": np.asarray(gamma_profile, dtype=np.float32),
        "contact_depths_m": np.asarray(contact_depths, dtype=np.float32),
        "n_contacts": int(len(contact_layers)),
        "n_neurons": int(len(layer_indices)),
        "area": str(area),
    }


def multi_area_spectrolaminar_readout(
    signal: "jax.Array",
    neurons: "Mapping[str, Any]",
    dt_ms: float = 0.1,
) -> dict[str, dict[str, "Any"]]:
    """Compute spectrolaminar readout for all areas.

    Parameters
    ----------
    signal : jax.Array
        Source or LFP signal [T, N].
    neurons : Mapping[str, Any]
        Neuron metadata with area, layer, positions.
    dt_ms : float
        Sampling interval in ms.

    Returns
    -------
    readouts : dict[str, dict]
        Per-area spectrolaminar readouts.
    """
    import numpy as np

    areas = np.array(neurons.get("area", ["V1"] * signal.shape[1]))
    area_set = sorted(set(areas[areas != ""]))

    readouts = {}
    for area in area_set:
        readouts[area] = spectrolaminar_readout(signal, neurons, area, dt_ms=dt_ms)

    return readouts
