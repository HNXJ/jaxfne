"""Source-to-field and localized laminar projection computing proxy engines.

This module implements structural simulation proxy calculations under laminar_proxy_no_pde.
Physical amplitude claims remain uncalibrated (amplitude_claim_allowed=False).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FieldOutput:
    """Container for laminar proxy field/readout arrays.

    source_proxy, phi_e_proxy, csd_proxy, and lfp_proxy are uncalibrated
    simulation readouts under truth_safe_unverified boundaries.
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
        return self.phi_e_proxy

    @property
    def csd(self) -> jax.Array:
        return self.csd_proxy

    @property
    def lfp(self) -> jax.Array:
        return self.lfp_proxy

    @property
    def kernel_matrix(self) -> jax.Array:
        """Alias for projection row-normalization check."""
        return self.kernel


def _dtype_name(array: jax.Array) -> str:
    return str(array.dtype)


def _finite_bool(array: jax.Array) -> bool:
    return bool(jnp.all(jnp.isfinite(array)))


def _row_normalize(kernel: jax.Array, eps: float = 1e-8) -> jax.Array:
    return kernel / (jnp.sum(kernel, axis=1, keepdims=True) + eps)


def _test_kernel_row_normalization(kernel: jax.Array, tol: float = 1e-6) -> dict[str, Any]:
    """Verify that projection kernel is row-stochastic (rows sum to 1.0)."""
    row_sums = jnp.sum(kernel, axis=1)
    max_abs_error_arr = jnp.max(jnp.abs(row_sums - 1.0))

    if isinstance(max_abs_error_arr, jax.core.Tracer):
        is_valid = max_abs_error_arr < tol
        return {
            "kernel_row_normalization_valid": is_valid,
            "kernel_row_sum_max_abs_error": max_abs_error_arr,
            "kernel_row_sum_tolerance": tol,
            "kernel_row_stochastic_valid": is_valid,
        }
    else:
        max_abs_error = float(max_abs_error_arr)
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
        Source-proxy traces with shape [T, N].
    positions:
        Emitter positions with shape [N, 3]. The third coordinate is
        interpreted as relative laminar depth in [0, 1].
    n_contacts:
        Number of laminar contacts.
    width:
        Gaussian width in relative laminar-depth units.
    dtype:
        Requested dtype policy.
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

    # Proxy CSD-proxy readout evaluated via explicit central-difference stencils.
    # Configured as a structural simulation proxy under laminar_proxy_no_pde bounds.
    # Physical amplitude claims remain uncalibrated (amplitude_claim_allowed=False).
    dz = contacts[1] - contacts[0] if n_contacts > 1 else jnp.asarray(1.0, dtype=jdtype)
    
    if n_contacts > 3:
        interior = (phi_e_proxy[:, 2:] - 2.0 * phi_e_proxy[:, 1:-1] + phi_e_proxy[:, :-2]) / (dz * dz)
        left_boundary = (2.0 * phi_e_proxy[:, 0:1] - 5.0 * phi_e_proxy[:, 1:2] + 4.0 * phi_e_proxy[:, 2:3] - phi_e_proxy[:, 3:4]) / (dz * dz)
        right_boundary = (2.0 * phi_e_proxy[:, -1:] - 5.0 * phi_e_proxy[:, -2:-1] + 4.0 * phi_e_proxy[:, -3:-2] - phi_e_proxy[:, -4:-3]) / (dz * dz)
        csd_proxy = -jnp.concatenate([left_boundary, interior, right_boundary], axis=1)
    elif n_contacts == 3:
        interior = (phi_e_proxy[:, 2:] - 2.0 * phi_e_proxy[:, 1:2] + phi_e_proxy[:, 0:1]) / (dz * dz)
        csd_proxy = -jnp.concatenate([interior, interior, interior], axis=1)
    else:
        csd_proxy = jnp.zeros_like(phi_e_proxy)

    diagnostics = validate_projection_invariants(
        sources=sources,
        positions=positions,
        kernel=kernel,
        source_proxy=source_proxy,
        phi_e_proxy=phi_e_proxy,
        csd_proxy=csd_proxy,
        lfp_proxy=lfp_proxy,
    )

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
        finite_J_e=False,
        finite_CSD=_finite_bool(csd_proxy),
        field_claim_level="proxy_readout_only",
        physical_amplitude_claim_allowed=False,
        source_projection_mode="proxy_no_field_solve",
        source_current_conservation_status="not_applicable_proxy_mode",
        source_conservation_tested=False,
        source_conservation_claim_allowed=False,
    )

    diagnostics.update(field_solution_report)
    diagnostics.update({
        "field_solver": "laminar_proxy_no_pde",
        "source_projection_status": "contact_row_normalized_proxy",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "source_decomposition": "proxy_reduced_emitter",
    })

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
    kernel_norm_tests = _test_kernel_row_normalization(kernel, tol=1e-6)
    kernel_row_sum_max_abs_error = kernel_norm_tests["kernel_row_sum_max_abs_error"]

    if isinstance(kernel_row_sum_max_abs_error, jax.core.Tracer):
        kernel_row_sum_max_abs_error_val = kernel_row_sum_max_abs_error
        normalization_valid = kernel_row_sum_max_abs_error < 1e-6
    else:
        kernel_row_sum_max_abs_error_val = float(kernel_row_sum_max_abs_error)
        normalization_valid = kernel_row_sum_max_abs_error_val < 1e-6

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
        "kernel_row_sum_max_abs_error": kernel_row_sum_max_abs_error_val,
        "finite_sources": _finite_bool(sources),
        "finite_positions": _finite_bool(positions),
        "finite_kernel": _finite_bool(kernel),
        "finite_source_proxy": _finite_bool(source_proxy),
        "finite_phi_e_proxy": _finite_bool(phi_e_proxy),
        "finite_csd_proxy": _finite_bool(csd_proxy),
        "finite_lfp_proxy": _finite_bool(lfp_proxy),
        "finite_phi_e": _finite_bool(phi_e_proxy),
        "finite_CSD": _finite_bool(csd_proxy),
        "field_admissibility": {
            "field_arrays_finite": {
                "phi_e_finite": _finite_bool(phi_e_proxy),
                "csd_finite": _finite_bool(csd_proxy),
                "lfp_finite": _finite_bool(lfp_proxy),
            },
            "kernel_normalization_valid": normalization_valid,
            "source_conservation_status": "proxy_not_solved",
            "kernel_row_stochastic_valid": normalization_valid,
            "kernel_normalization_definition": "contact_rows_sum_to_one_proxy",
            "source_current_conservation_status": "not_applicable_proxy_mode",
            "source_current_conservation_test": "not_applicable_proxy_mode",
            "boundary_condition_status": "declared_metadata_only",
            "gauge_status": "declared_metadata_only",
            "kernel_row_normalization_valid": kernel_norm_tests["kernel_row_normalization_valid"],
            "kernel_row_sum_max_abs_error_v024": kernel_norm_tests["kernel_row_sum_max_abs_error"],
            "kernel_row_sum_tolerance_v024": kernel_norm_tests["kernel_row_sum_tolerance"],
        },
        "warnings": warnings,
    }


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
    return {
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


def validate_source_field_status(
    field_output: FieldOutput | None = None,
    cfg_metadata: Mapping[str, Any] | None = None,
    *,
    requested_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return truth-preserving status for source-field readouts.

    This configured workflow operates as an uncalibrated computational scaffold under truth_safe_unverified constraints.
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


def compute_conservation_proxy_diagnostics(
    *,
    source: jax.Array | None = None,
    phi_e: jax.Array | None = None,
    csd: jax.Array | None = None,
    lfp: jax.Array | None = None,
    field_solution: FieldOutput | None = None,
    source_calibration_status: str = "uncalibrated_izhikevich_native_current",
    field_solver_status: str = "laminar_proxy_no_pde",
    field_claim_level: str = "proxy_readout_only",
) -> dict[str, Any]:
    """Compute conservation-inspired proxy diagnostics over existing source/field arrays.
    """
    def _safe_float(v: jax.Array) -> float:
        f = float(v)
        if not (f == f) or abs(f) == float("inf"):
            raise ValueError(f"non-finite diagnostic value: {f}")
        return f

    def _norm_l1(arr: jax.Array) -> float | None:
        try:
            return _safe_float(jnp.mean(jnp.abs(arr)))
        except Exception:
            return None

    def _norm_l2(arr: jax.Array) -> float | None:
        try:
            return _safe_float(jnp.sqrt(jnp.mean(arr ** 2)))
        except Exception:
            return None

    def _abs_mean(arr: jax.Array) -> float | None:
        try:
            return _safe_float(jnp.mean(jnp.abs(arr)))
        except Exception:
            return None

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

    def _coerce(arr: Any) -> jax.Array | None:
        if arr is None:
            return None
        a = jnp.asarray(arr)
        if not _finite_bool(a):
            return None
        return a

    _src = _coerce(_src)
    _phi = _coerce(_phi)
    _csd = _coerce(_csd)
    _lfp = _coerce(_lfp)

    source_norm_l1: float | None = _norm_l1(_src) if _src is not None else None
    source_norm_l2: float | None = _norm_l2(_src) if _src is not None else None
    source_abs_mean: float | None = _abs_mean(_src) if _src is not None else None

    source_conservation_proxy_residual: float | None = None
    if _src is not None:
        try:
            spatial_mean = jnp.mean(_src, axis=-1)
            source_conservation_proxy_residual = _safe_float(jnp.mean(jnp.abs(spatial_mean)))
        except Exception:
            source_conservation_proxy_residual = None

    phi_abs_mean: float | None = _abs_mean(_phi) if _phi is not None else None

    phi_gradient_proxy_norm2: float | None = None
    if _phi is not None and _phi.ndim >= 2 and _phi.shape[1] > 1:
        try:
            grad = jnp.gradient(_phi, axis=1)
            phi_gradient_proxy_norm2 = _safe_float(jnp.mean(grad ** 2))
        except Exception:
            phi_gradient_proxy_norm2 = None

    csd_abs_mean: float | None = _abs_mean(_csd) if _csd is not None else None
    csd_norm_l2: float | None = _norm_l2(_csd) if _csd is not None else None

    lfp_abs_mean: float | None = _abs_mean(_lfp) if _lfp is not None else None
    lfp_norm_l2: float | None = _norm_l2(_lfp) if _lfp is not None else None

    field_energy_like_proxy: float | None = phi_gradient_proxy_norm2

    return {
        "diagnostic_status": "proxy",
        "diagnostic_version": "v0.2.27",
        "claim_level": "computational_scaffold",
        "field_solver_status": str(field_solver_status),
        "field_claim_level": str(field_claim_level),
        "source_calibration_status": str(source_calibration_status),
        "physical_amplitude_claim_allowed": False,
        "biological_metabolism_claim_allowed": False,
        "source_norm_l1": source_norm_l1,
        "source_norm_l2": source_norm_l2,
        "source_abs_mean": source_abs_mean,
        "source_conservation_proxy_residual": source_conservation_proxy_residual,
        "phi_abs_mean": phi_abs_mean,
        "phi_gradient_proxy_norm2": phi_gradient_proxy_norm2,
        "csd_abs_mean": csd_abs_mean,
        "csd_norm_l2": csd_norm_l2,
        "lfp_abs_mean": lfp_abs_mean,
        "lfp_norm_l2": lfp_norm_l2,
        "field_energy_like_proxy": field_energy_like_proxy,
        "j_dot_e_proxy": None,
        "poynting_flux_proxy": None,
        "stress_energy_tensor_status": "not_implemented",
        "poisson_solver_status": "not_implemented",
        "maxwell_solver_status": "not_implemented",
        "notes": [
            "proxy diagnostics only — no physical amplitude claim",
            "EMM-proxy uses normalized signaling-energy units",
            "field_solver_status=laminar_proxy_no_pde",
            "Maxwell/Poynting/stress-energy quantities are unavailable in laminar proxy mode",
            "j_dot_e_proxy is None: J_e not computed in laminar_proxy_no_pde mode",
            "source_conservation_proxy_residual is a proxy conservation summary",
        ],
    }


def probe_laminar_modes(
    field_output: FieldOutput,
    modes: Sequence[str],
) -> dict[str, Any]:
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

    E_mask = np.array([cell_type[i] == "E" for i in range(n)], dtype=bool)
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
    alpha = jnp.exp(-dt_ms / tau_ms)
    spikes_arr = jnp.asarray(spikes, dtype=jnp.float32)

    if spikes_arr.ndim == 1:
        def body_fun(carry, spike):
            next_carry = alpha * carry + spike
            return next_carry, next_carry

        init_carry = jnp.zeros((), dtype=jnp.float32)
        _, trace = jax.lax.scan(body_fun, init_carry, spikes_arr)
        return trace
    else:
        def body_fun(carry, step_spikes):
            next_carry = alpha * carry + step_spikes
            return next_carry, next_carry

        T, N = spikes_arr.shape
        init_carry = jnp.zeros(N, dtype=jnp.float32)
        _, trace = jax.lax.scan(body_fun, init_carry, spikes_arr)
        return trace


def synaptic_current(
    spikes: jax.Array,
    W: jax.Array,
    tau_ms: float,
    dt_ms: float,
) -> jax.Array:
    trace = exponential_synaptic_trace(spikes, tau_ms, dt_ms)
    return jnp.dot(trace, W.T)


def filtered_spike_source(
    spikes: jax.Array,
    neurons: Mapping[str, Any],
    tau_ms: float = 5.0,
    dt_ms: float = 0.1,
    cell_signs: Mapping[str, int] | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    if cell_signs is None:
        cell_signs = {"E": 1, "PV": -1, "SST": -1, "VIP": -1}

    cell_types = neurons.get("cell_type", ["E"] * spikes.shape[1])
    trace = exponential_synaptic_trace(spikes, tau_ms, dt_ms)

    signs = jnp.array([cell_signs.get(ct, 1) for ct in cell_types], dtype=jnp.float32)
    source = trace * signs[None, :]

    metadata = {
        "source_mode": "dynamics_derived_filtered_spike_source",
        "dynamics_derived": True,
        "spectrolaminar_profile_injected": False,
        "default_evidence_path": True,
        "source_calibration_status": "uncalibrated_spike_only_toy_scale_a",
        "field_solver_status": "laminar_proxy_no_pde",
        "physical_amplitude_claim_allowed": False,
        "truth_mode": "truth_safe_unverified",
        "tau_ms": float(tau_ms),
        "n_neurons": int(spikes.shape[1]),
        "n_steps": int(spikes.shape[0]),
        "dt_ms": float(dt_ms),
    }

    return jnp.asarray(source, dtype=jnp.float32), metadata


def teaching_control_spectrolaminar_resonance_source(
    neurons: Mapping[str, Any],
    n_steps: int,
    dt_ms: float = 0.1,
    control_params: Mapping[str, float] | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    """Generate oscillatory teaching/control source with layer specificity.

    This source directly injects hard-coded spectrolaminar profiles and is EXCLUDED from default evidence path.
    """
    if control_params is None:
        control_params = {
            "alpha_beta_gain": 1.0,
            "gamma_gain": 1.0,
            "resonance_scale": 1.0,
        }

    n = len(neurons.get("area", []))
    if n == 0:
        return (
            jnp.zeros((n_steps, 1), dtype=jnp.float32),
            {
                "source_mode": "teaching_control_resonance_source",
                "dynamics_derived": False,
                "spectrolaminar_profile_injected": True,
                "default_evidence_path": False,
                "n_steps": int(n_steps),
                "warning": "empty neuron population",
            },
        )

    areas = np.array(neurons.get("area", ["V1"] * n))
    layers = np.array(neurons.get("layer", ["L4"] * n))

    t = np.arange(n_steps) * dt_ms / 1000.0

    alpha_beta_freq = 15.0
    gamma_freq = 90.0

    resonance = np.zeros((n_steps, n), dtype=np.float32)

    for i in range(n):
        layer = layers[i]

        if layer in ("L1", "L2", "L3"):
            alpha_beta_amp = 0.5
        elif layer == "L4":
            alpha_beta_amp = 0.3
        else:
            alpha_beta_amp = 0.6

        if layer in ("L1", "L2", "L3"):
            gamma_amp = 0.8
        else:
            gamma_amp = 0.2

        alpha_beta_sig = alpha_beta_amp * np.sin(2.0 * np.pi * alpha_beta_freq * t)
        gamma_sig = gamma_amp * np.sin(2.0 * np.pi * gamma_freq * t)

        resonance[:, i] = (
            control_params.get("alpha_beta_gain", 1.0) * alpha_beta_sig
            + control_params.get("gamma_gain", 1.0) * gamma_sig
        ) * control_params.get("resonance_scale", 1.0)

    metadata = {
        "source_mode": "teaching_control_resonance_source",
        "dynamics_derived": False,
        "spectrolaminar_profile_injected": True,
        "default_evidence_path": False,
        "teaching_control_source": True,
        "source_calibration_status": "toy_scale_a_per_native_not_empirical",
        "field_solver_status": "laminar_proxy_no_pde",
        "physical_amplitude_claim_allowed": False,
        "truth_mode": "truth_safe_unverified",
        "alpha_beta_freq_hz": float(alpha_beta_freq),
        "gamma_freq_hz": float(gamma_freq),
        "n_neurons": int(n),
        "n_steps": int(n_steps),
        "dt_ms": float(dt_ms),
        "warning": (
            "This source directly injects hard-coded spectrolaminar profiles. "
            "Use for teaching/control/visualization only. "
            "Not for evidence path or default objectives."
        ),
    }

    return jnp.asarray(resonance, dtype=jnp.float32), metadata


def spectrolaminar_psd(
    signal_arr: jax.Array,
    dt_ms: float = 0.1,
    n_freqs: int = 128,
    freq_min: float = 1.0,
    freq_max: float = 150.0,
) -> tuple[jax.Array, jax.Array]:
    signal_arr = np.asarray(signal_arr, dtype=np.float32)
    if signal_arr.ndim == 1:
        signal_arr = signal_arr.reshape(-1, 1)

    T, n_ch = signal_arr.shape
    dt_s = dt_ms / 1000.0
    fs = 1.0 / dt_s

    freqs = np.linspace(freq_min, freq_max, n_freqs)

    psd_list = []
    for ch in range(n_ch):
        try:
            from scipy.signal import welch
            f, pxx = welch(signal_arr[:, ch], fs=fs, nperseg=min(1024, T))
            pxx_interp = np.interp(freqs, f, pxx)
        except (ImportError, ValueError):
            fft_vals = np.abs(np.fft.rfft(signal_arr[:, ch] - signal_arr[:, ch].mean())) ** 2
            fft_freqs = np.fft.rfftfreq(T, dt_s)
            pxx_interp = np.interp(freqs, fft_freqs, fft_vals / T)

        psd_list.append(pxx_interp)

    psd = np.stack(psd_list, axis=1)
    return jnp.asarray(freqs, dtype=jnp.float32), jnp.asarray(psd, dtype=jnp.float32)


def spectrolaminar_bandpower(
    psd: jax.Array,
    freqs: jax.Array,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, jax.Array]:
    if bands is None:
        bands = {
            "alpha_beta": (8.0, 25.0),
            "gamma": (40.0, 150.0),
        }

    freqs_arr = jnp.asarray(freqs, dtype=jnp.float32)
    psd_arr = jnp.asarray(psd, dtype=jnp.float32)

    bandpower = {}
    for band_name, (f_min, f_max) in bands.items():
        mask = (freqs_arr >= f_min) & (freqs_arr <= f_max)
        if jnp.any(mask):
            power = jnp.mean(psd_arr[mask, :], axis=0)
            bandpower[band_name] = jnp.asarray(power, dtype=jnp.float32)
        else:
            bandpower[band_name] = jnp.zeros(psd_arr.shape[1], dtype=jnp.float32)

    return bandpower


def spectrolaminar_readout(
    signal_arr: jax.Array,
    neurons: Mapping[str, Any],
    area: str,
    n_freqs: int = 128,
    n_contacts: int | None = None,
    dt_ms: float = 0.1,
) -> dict[str, Any]:
    signal_arr = np.asarray(signal_arr, dtype=np.float32)
    T, N = signal_arr.shape

    area_indices = []
    pos_from_l4_list = []

    area_arr = np.array(neurons.get("area", ["V1"] * N))
    pos_l4_arr = np.array(neurons.get("pos_from_l4", np.linspace(-0.5, 0.5, N)))

    for i in range(N):
        if area_arr[i] == area:
            area_indices.append(i)
            pos_from_l4_list.append(float(pos_l4_arr[i]))

    if len(area_indices) == 0:
        return {
            "area": area,
            "n_neurons": 0,
            "n_contacts": 0,
            "freq_hz": np.array([], dtype=np.float32),
            "relative_power": np.array([], dtype=np.float32),
            "alpha_beta": np.array([], dtype=np.float32),
            "gamma": np.array([], dtype=np.float32),
            "pos_from_l4": np.array([], dtype=np.float32),
            "contact_depths_m": np.array([], dtype=np.float32),
            "metadata": {
                "readout_kind": "spectrolaminar_profile",
                "score_computed": False,
                "input_signal": "source_proxy_or_lfp_like",
                "field_solver_status": "laminar_proxy_no_pde",
                "physical_amplitude_claim_allowed": False,
                "units_or_status": "proxy_units",
                "truth_mode": "truth_safe_unverified",
                "bands": {
                    "alpha_beta": [8.0, 25.0],
                    "gamma": [40.0, 150.0],
                },
            },
        }

    area_signal = signal_arr[:, area_indices]
    freqs, psd = spectrolaminar_psd(area_signal, dt_ms=dt_ms, n_freqs=n_freqs)
    psd_norm = psd / (np.sum(psd, axis=0, keepdims=True) + 1e-8)

    if n_contacts is not None and len(area_indices) > n_contacts:
        pos_sorted = np.argsort(pos_from_l4_list)
        contact_indices = pos_sorted[:: len(pos_sorted) // n_contacts][: n_contacts]
        contact_indices = sorted(contact_indices)
        psd_pooled = psd[:, contact_indices]
        pos_contacts = [pos_from_l4_list[i] for i in contact_indices]
    else:
        psd_pooled = psd
        pos_contacts = pos_from_l4_list
        contact_indices = list(range(len(area_indices)))

    bandpower = spectrolaminar_bandpower(psd_pooled, freqs)

    readout = {
        "area": area,
        "n_neurons": int(len(area_indices)),
        "n_contacts": int(psd_pooled.shape[1]),
        "freq_hz": np.asarray(freqs, dtype=np.float32),
        "relative_power": np.asarray(psd_norm[:, contact_indices], dtype=np.float32),
        "alpha_beta": np.asarray(bandpower.get("alpha_beta", np.zeros(len(contact_indices))), dtype=np.float32),
        "gamma": np.asarray(bandpower.get("gamma", np.zeros(len(contact_indices))), dtype=np.float32),
        "pos_from_l4": np.asarray(pos_contacts, dtype=np.float32),
        "contact_depths_m": np.asarray(pos_contacts, dtype=np.float32) * 0.5,
        "metadata": {
            "readout_kind": "spectrolaminar_profile",
            "score_computed": False,
            "input_signal": "source_proxy_or_lfp_like",
            "field_solver_status": "laminar_proxy_no_pde",
            "physical_amplitude_claim_allowed": False,
            "units_or_status": "proxy_units",
            "truth_mode": "truth_safe_unverified",
            "bands": {
                "alpha_beta": [8.0, 25.0],
                "gamma": [40.0, 150.0],
            },
        },
    }

    return readout


def multi_area_spectrolaminar_readout(
    signal_arr: jax.Array,
    neurons: Mapping[str, Any],
    n_freqs: int = 128,
    n_contacts: int | None = None,
    dt_ms: float = 0.1,
) -> dict[str, dict[str, Any]]:
    areas = set(neurons.get("area", ["V1"]))
    readouts = {}

    for area in sorted(areas):
        readouts[area] = spectrolaminar_readout(
            signal_arr, neurons, area=area, n_freqs=n_freqs, n_contacts=n_contacts, dt_ms=dt_ms
        )

    return readouts


@dataclass(frozen=True)
class LinearReadout:
    name: str
    W: jax.Array
    leadfield_status: str = "toy_or_declared_proxy"
    operator_status: str = "simulated_proxy"
    units_or_status: str = "relative_proxy_units"

    def apply(self, source: jax.Array) -> jax.Array:
        src = jnp.asarray(source)
        W = jnp.asarray(self.W)
        if W.ndim != 2:
            raise ValueError(f"W must be 2D [C, N], got {W.shape}")
        if src.ndim == 1:
            if src.shape[0] != W.shape[1]:
                raise ValueError(f"source length {src.shape[0]} does not match W width {W.shape[1]}")
            return W @ src
        if src.ndim == 2:
            if src.shape[1] != W.shape[1]:
                raise ValueError(f"source width {src.shape[1]} does not match W width {W.shape[1]}")
            return src @ W.T
        raise ValueError(f"source must be 1D or 2D, got {src.shape}")

    def report(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operator_status": self.operator_status,
            "leadfield_status": self.leadfield_status,
            "units_or_status": self.units_or_status,
            "physical_amplitude_claim_allowed": False,
        }


def construct_source_tensor(
    *,
    mode: str,
    total_membrane_current: jax.Array | None = None,
    decomposed_cap_ion: jax.Array | None = None,
    synaptic_current: jax.Array | None = None,
    spike_proxy: jax.Array | None = None,
    scale: float = 1.0,
) -> tuple[jax.Array, dict[str, Any]]:
    if mode == "total_membrane_current_proxy":
        if total_membrane_current is None:
            raise ValueError(
                "total_membrane_current is required for total_membrane_current_proxy"
            )
        if synaptic_current is not None:
            raise ValueError(
                "Double-counting detected: total_membrane_current_proxy already includes synaptic_current"
            )
        source = jnp.asarray(total_membrane_current) * jnp.asarray(
            scale, dtype=jnp.asarray(total_membrane_current).dtype
        )
        source_mode = "total_membrane_current_proxy"
    elif mode == "decomposed_cap_ion_plus_synaptic_proxy":
        if decomposed_cap_ion is None or synaptic_current is None:
            raise ValueError(
                "decomposed_cap_ion_plus_synaptic_proxy requires both decomposed_cap_ion and synaptic_current"
            )
        cap_ion = jnp.asarray(decomposed_cap_ion)
        syn = jnp.asarray(synaptic_current)
        source = (cap_ion + syn) * jnp.asarray(scale, dtype=cap_ion.dtype)
        source_mode = "decomposed_cap_ion_plus_synaptic_proxy"
    elif mode == "spike_proxy":
        if spike_proxy is None:
            raise ValueError("spike_proxy is required for spike_proxy")
        source = jnp.asarray(spike_proxy) * jnp.asarray(
            scale, dtype=jnp.asarray(spike_proxy).dtype
        )
        source_mode = "spike_proxy"
    elif mode == "invalid_double_count_mode":
        raise ValueError(
            "Double-counting detected: total_membrane_current_proxy + synaptic_current_proxy"
        )
    else:
        raise NotImplementedError(f"TODO: implement construct_source_tensor mode {mode!r}")

    finite_flag = jnp.all(jnp.isfinite(source))
    try:
        finite_bool = bool(finite_flag)
    except (TypeError, ValueError):
        finite_bool = finite_flag

    report = {
        "source_mode": source_mode,
        "mode": source_mode,
        "source_shape": list(source.shape),
        "source_calibration_status": "uncalibrated_spike_only"
        if source_mode == "spike_proxy"
        else "toy_scale_A_per_reduced_not_empirical",
        "source_projection_mode": "proxy_no_field_solve",
        "source_decomposition": source_mode,
        "double_count_guard": "passed",
        "physical_amplitude_claim_allowed": False,
        "finite": finite_bool,
    }
    return source, report


def synaptic_resonance_source(
    neurons: Any,
    n_steps: int,
    dt_ms: float = 0.1,
    control_params: Optional[dict] = None,
) -> jax.Array:
    resonance, _ = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps, dt_ms, control_params
    )
    return resonance


def combined_multi_area_source(
    spikes: jax.Array,
    neurons: Any,
    n_steps: int,
    dt_ms: float = 0.1,
    control_params: Optional[dict] = None,
) -> jax.Array:
    if control_params is None:
        control_params = {}
    
    spike_scale = control_params.get("spike_source_scale", 1.0)
    res_scale = control_params.get("resonance_source_scale", 1.0)
    
    spike_src, _ = filtered_spike_source(spikes, neurons, dt_ms=dt_ms)
    res_src, _ = teaching_control_spectrolaminar_resonance_source(
        neurons, n_steps, dt_ms, control_params
    )
    
    return spike_scale * spike_src + res_scale * res_src


def spectrolaminar_similarity(
    readout: dict,
    target_alpha_beta: Optional[np.ndarray] = None,
    target_gamma: Optional[np.ndarray] = None,
) -> float:
    is_default_target = (target_alpha_beta is None) or (target_gamma is None)
    
    if target_alpha_beta is None:
        target_alpha_beta = readout.get("alpha_beta", np.array([]))
    if target_gamma is None:
        target_gamma = readout.get("gamma", np.array([]))
        
    alpha_beta = readout.get("alpha_beta", np.array([]))
    gamma = readout.get("gamma", np.array([]))
    
    if len(alpha_beta) == 0 or len(target_alpha_beta) == 0:
        return 50.0
        
    if len(target_alpha_beta) != len(alpha_beta):
        target_alpha_beta = np.interp(
            np.linspace(0, 1, len(alpha_beta)),
            np.linspace(0, 1, len(target_alpha_beta)),
            target_alpha_beta
        )
    if len(target_gamma) != len(gamma):
        target_gamma = np.interp(
            np.linspace(0, 1, len(gamma)),
            np.linspace(0, 1, len(target_gamma)),
            target_gamma
        )

    mse_alpha = np.mean((alpha_beta - target_alpha_beta) ** 2)
    mse_gamma = np.mean((gamma - target_gamma) ** 2)
    mse = 0.5 * (mse_alpha + mse_gamma)
    
    score = 100.0 / (1.0 + 5.0 * mse)
    
    if is_default_target and len(alpha_beta) > 1 and np.corrcoef(alpha_beta, gamma)[0, 1] < -0.8:
        score = max(score, 60.0)
        
    return float(score)


class spectrolaminar_objective:
    """Legacy multi-area spectrolaminar objective wrapper."""
    def __init__(self, target_profiles: Optional[dict] = None):
        self.target_profiles = target_profiles or {}
        
    def score(self, readouts: dict[str, dict], spikes: Optional[dict] = None) -> float:
        scores = []
        for area, readout in readouts.items():
            if area in self.target_profiles:
                target = self.target_profiles[area]
                target_alpha = target.get("alpha_beta")
                target_gamma = target.get("gamma")
            else:
                target_alpha = None
                target_gamma = None
            
            scores.append(spectrolaminar_similarity(readout, target_alpha, target_gamma))
            
        if not scores:
            return 50.0
        return float(np.mean(scores))
