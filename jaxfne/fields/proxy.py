"""Source-to-field and localized laminar projection computing proxy engines.

Docs: ``docs/api/fields.md`` (https://jaxfne.readthedocs.io/en/latest/api/fields/) —
update that page when this module's public API changes.

This module implements structural simulation proxy calculations under linear_solver.
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

# Canonical source for these field diagnostics/validation helpers is
# diagnostics.py. They are imported (not redefined) here so that
# ``fields.proxy.<name>`` continues to resolve and the public
# ``jtfne.validate_projection_invariants`` has exactly one definition.
from .diagnostics import (
    _dtype_name,
    _finite_bool,
    _test_kernel_row_normalization,
    validate_projection_invariants,
    _make_field_solution_report,
)


@dataclass(frozen=True)
class FieldOutput:
    """Container for laminar proxy field/readout arrays.

    source_proxy, phi_e_proxy, csd_proxy, and lfp_proxy are uncalibrated
    simulation readouts.
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
        """Documented public function `phi_e`."""
        return self.phi_e_proxy

    @property
    def csd(self) -> jax.Array:
        """Documented public function `csd`."""
        return self.csd_proxy

    @property
    def lfp(self) -> jax.Array:
        """Documented public function `lfp`."""
        return self.lfp_proxy

    @property
    def kernel_matrix(self) -> jax.Array:
        """Alias for projection row-normalization check."""
        return self.kernel


def _row_normalize(kernel: jax.Array, eps: float = 1e-8) -> jax.Array:
    return kernel / (jnp.sum(kernel, axis=1, keepdims=True) + eps)


def _proxy_truth_gate_fragment(
    *,
    field_solver_status: str = "linear_solver",
    physical_amplitude_calibrated: bool = False,
    source_calibration_status: str | None = None,
    claim_level: str | None = None,
) -> dict[str, Any]:
    """Common truth-gate key/value fragment repeated across this module's
    report/metadata dicts. Merge into a caller's dict via
    ``**_proxy_truth_gate_fragment(...)`` -- each caller keeps its own unique
    fields; this only centralizes the constant trio/quartet so a future
    truth-gate-default change can't silently drift between call sites.
    """
    fragment: dict[str, Any] = {
        "field_solver_status": field_solver_status,
        "physical_amplitude_calibrated": physical_amplitude_calibrated,
    }
    if source_calibration_status is not None:
        fragment["source_calibration_status"] = source_calibration_status
    if claim_level is not None:
        fragment["claim_level"] = claim_level
    return fragment


def _spectrolaminar_readout_metadata() -> dict[str, Any]:
    """Metadata dict for :func:`spectrolaminar_readout`, identical across its
    empty-area and populated-area branches -- factored to a single definition.
    """
    return {
        "readout_kind": "spectrolaminar_profile",
        "score_computed": False,
        "input_signal": "source_proxy_or_lfp_proxy",
        **_proxy_truth_gate_fragment(),
        "units_or_status": "proxy_units",
        "bands": {
            "alpha_beta": [8.0, 25.0],
            "gamma": [40.0, 150.0],
        },
    }


def project_laminar_sources(
    sources: jax.Array,
    positions: jax.Array,
    *,
    n_contacts: int = 16,
    width: float = 0.10,
    mode: str = "density_preserving",
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
    mode:
        Projection mode: 'density_preserving' (default) or 'row_normalize'.
        ``'row_normalize'`` erases source density and flattens depth
        structure (each contact's weights always sum to 1 regardless of how
        attenuated the raw Gaussian is, so e.g. a contact outside the
        modeled population reads as loud as one in the densest layer) — it
        is kept only for explicit opt-in / backward compatibility, matching
        the stance already documented for ``JaxleyBridge.simulate_laminar_field``
        (``docs/api/bridges.md``).
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
    
    if mode == "row_normalize":
        kernel = _row_normalize(raw_kernel)
    elif mode == "density_preserving":
        kernel = raw_kernel
    else:
        raise ValueError(f"Invalid projection mode: {mode!r}. Must be 'row_normalize' or 'density_preserving'.")

    source_proxy = sources @ kernel.T
    lfp_proxy = source_proxy
    phi_e_proxy = lfp_proxy

    # CSD proxy: spatial second-derivative tensor (see csd_tensor)
    dz = contacts[1] - contacts[0] if n_contacts > 1 else jnp.asarray(1.0, dtype=jdtype)
    csd_proxy = csd_tensor(phi_e_proxy, dz)

    diagnostics = validate_projection_invariants(
        sources=sources,
        positions=positions,
        kernel=kernel,
        source_proxy=source_proxy,
        phi_e_proxy=phi_e_proxy,
        csd_proxy=csd_proxy,
        lfp_proxy=lfp_proxy,
        mode=mode,
    )

    field_solution_report = _make_field_solution_report(
        field_solver_status="linear_solver",
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
        field_claim_level="proxy_readout",
        physical_amplitude_calibrated=False,
        source_projection_mode="proxy_no_field_solve",
        source_current_conservation_status="not_applicable_proxy_mode",
        source_conservation_tested=False,
        source_conservation_claim_allowed=False,
    )

    diagnostics.update(field_solution_report)
    diagnostics.update({
        "field_solver": "linear_solver",
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
    mode: str = "density_preserving",
    dtype: str = "float32",
) -> FieldOutput:
    """Project sources onto ``n_contacts`` laminar contacts (proxy field).

    Convenience wrapper over :func:`project_laminar_sources`: maps ``sources`` at
    ``positions`` to a :class:`FieldOutput` via the Gaussian-leadfield laminar
    proxy. This is not a PDE/volume-conductor solve
    (``field_solver_status = "linear_solver"``); outputs are relative-unit
    proxies.
    """
    return project_laminar_sources(sources, positions, n_contacts=n_contacts, mode=mode, dtype=dtype)



def validate_source_field_status(
    field_output: FieldOutput | None = None,
    cfg_metadata: Mapping[str, Any] | None = None,
    *,
    requested_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return truth-preserving status for source-field readouts.

    This configured workflow operates as an uncalibrated computational scaffold.
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
        "field_solver_status", diagnostics.get("field_solver_status", "linear_solver")
    )

    source_status = str(source_calibration_status).lower()
    field_status = str(field_solver_status).lower()
    projection_status = str(source_projection_mode).lower()
    is_proxy = "proxy" in field_status or "no_pde" in field_status or "proxy" in projection_status
    is_calibrated = "calibrated" in source_status and "uncalibrated" not in source_status
    physical_amplitude_calibrated = False
    field_claim_level = "proxy_readout"
    if "uncalibrated" in source_status:
        warnings.append("uncalibrated_reduced_emitter_source")
    if is_proxy:
        warnings.append("linear_solver_not_physical_field_solution")

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
        "physical_amplitude_calibrated": physical_amplitude_calibrated,
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
    field_solver_status: str = "linear_solver",
    field_claim_level: str = "proxy_readout",
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
        "physical_amplitude_calibrated": False,
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
            "field_solver_status=linear_solver",
            "Maxwell/Poynting/stress-energy quantities are unavailable in laminar proxy mode",
            "j_dot_e_proxy is None: J_e not computed in linear_solver mode",
            "source_conservation_proxy_residual is a proxy conservation summary",
        ],
    }


def probe_laminar_modes(
    field_output: FieldOutput,
    modes: Sequence[str] = ("source", "phi_e", "CSD", "LFP"),
) -> dict[str, Any]:
    """Extract requested proxy readouts from a :class:`FieldOutput`.

    ``modes`` selects among the available laminar proxies (``"source"``,
    ``"phi_e"``, ``"CSD"``, ``"LFP"``, ``"J_e"``); returns a dict mapping each to
    its ``*_proxy`` array. Default extracts the full proxy set. All returned
    arrays are proxy readouts (``field_solver_status = "linear_solver"``), not
    physical signals.
    """
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
    """Documented public function `make_laminar_connectivity`."""
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

    pos_np = np.asarray(positions_m)
    dxy = np.linalg.norm(pos_np[:, None, :2] - pos_np[None, :, :2], axis=-1)
    local_gain = np.exp(-((dxy / local_decay_m) ** 2))

    same_area = area[:, None] == area[None, :]
    no_self = ~np.eye(n, dtype=bool)

    # 1. Local Exc
    draw_p_local_e = rng.random((n, n))
    draw_w_e = rng.uniform(*w_e_range, size=(n, n))
    mask_local_e = same_area & (cell_type[None, :] == "E") & no_self & (draw_p_local_e < p_local_e)
    W_local_exc = np.where(mask_local_e, draw_w_e * local_gain, 0.0).astype(np.float32)

    # 2. Local Inh
    draw_p_local_i = rng.random((n, n))
    draw_w_i = rng.uniform(*w_i_range, size=(n, n))
    mask_local_i = same_area & (cell_type[None, :] != "E") & no_self & (draw_p_local_i < p_local_i)
    W_local_inh = np.where(mask_local_i, draw_w_i * (0.65 + 0.35 * local_gain), 0.0).astype(np.float32)

    # 3. Feedforward and Feedback (delta area rank)
    ranks = np.array([area_rank.get(a, 0) for a in area])
    delta = ranks[:, None] - ranks[None, :]

    # FF: delta == 1 and layer[pre] in ("L2", "L3") and layer[post] == "L4"
    pre_ff_layer = np.isin(layer, ["L2", "L3"])
    post_ff_layer = layer == "L4"
    mask_ff_geom = (~same_area) & no_self & (delta == 1) & (cell_type[None, :] == "E") & pre_ff_layer[None, :] & post_ff_layer[:, None]
    draw_p_ff = rng.random((n, n))
    draw_w_ff = rng.uniform(*w_ff_range, size=(n, n))
    mask_ff = mask_ff_geom & (draw_p_ff < p_feedforward)
    W_ff = np.where(mask_ff, draw_w_ff, 0.0).astype(np.float32)

    # FB: delta == -1 and layer[pre] in ("L2", "L3", "L6") and layer[post] in ("L5", "L6")
    pre_fb_layer = np.isin(layer, ["L2", "L3", "L6"])
    post_fb_layer = np.isin(layer, ["L5", "L6"])
    mask_fb_geom = (~same_area) & no_self & (delta == -1) & (cell_type[None, :] == "E") & pre_fb_layer[None, :] & post_fb_layer[:, None]
    draw_p_fb = rng.random((n, n))
    draw_w_fb = rng.uniform(*w_fb_range, size=(n, n))
    mask_fb = mask_fb_geom & (draw_p_fb < p_feedback)
    W_fb = np.where(mask_fb, draw_w_fb, 0.0).astype(np.float32)

    W = (
        control_params.get("local_exc_gain", 1.0) * W_local_exc
        + control_params.get("local_inh_gain", 1.0) * W_local_inh
        + control_params.get("feedforward_gain", 1.0) * W_ff
        + control_params.get("feedback_gain", 1.0) * W_fb
    )

    E_mask = cell_type == "E"
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
    """Documented public function `exponential_synaptic_trace`."""
    alpha = jnp.exp(-dt_ms / tau_ms)
    spikes_arr = jnp.asarray(spikes, dtype=jnp.float32)

    if spikes_arr.ndim == 1:
        def body_fun(carry, spike):
            """Documented public function `body_fun`."""
            next_carry = alpha * carry + spike
            return next_carry, next_carry

        init_carry = jnp.zeros((), dtype=jnp.float32)
        _, trace = jax.lax.scan(body_fun, init_carry, spikes_arr)
        return trace
    else:
        def body_fun(carry, step_spikes):
            """Documented public function `body_fun`."""
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
    """Documented public function `synaptic_current`."""
    trace = exponential_synaptic_trace(spikes, tau_ms, dt_ms)
    return jnp.dot(trace, W.T)


def filtered_spike_source(
    spikes: jax.Array,
    neurons: Mapping[str, Any],
    tau_ms: float = 5.0,
    dt_ms: float = 0.1,
    cell_signs: Mapping[str, int] | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    """Documented public function `filtered_spike_source`."""
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
        **_proxy_truth_gate_fragment(source_calibration_status="uncalibrated_spike_only_toy_scale_a"),
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

    in_l123 = np.isin(layers, ["L1", "L2", "L3"])
    in_l4 = layers == "L4"

    alpha_beta_amp = np.where(in_l123, 0.5, np.where(in_l4, 0.3, 0.6))
    gamma_amp = np.where(in_l123, 0.8, 0.2)

    sin_alpha_beta = np.sin(2.0 * np.pi * alpha_beta_freq * t)
    sin_gamma = np.sin(2.0 * np.pi * gamma_freq * t)

    alpha_beta_sig = sin_alpha_beta[:, None] * alpha_beta_amp[None, :]
    gamma_sig = sin_gamma[:, None] * gamma_amp[None, :]

    resonance = (
        control_params.get("alpha_beta_gain", 1.0) * alpha_beta_sig
        + control_params.get("gamma_gain", 1.0) * gamma_sig
    ) * control_params.get("resonance_scale", 1.0)
    resonance = resonance.astype(np.float32)

    metadata = {
        "source_mode": "teaching_control_resonance_source",
        "dynamics_derived": False,
        "spectrolaminar_profile_injected": True,
        "default_evidence_path": False,
        "teaching_control_source": True,
        **_proxy_truth_gate_fragment(source_calibration_status="toy_scale_a_per_native_not_empirical"),
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
    """Documented public function `spectrolaminar_psd`."""
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
    """Documented public function `spectrolaminar_bandpower`."""
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
    """Documented public function `spectrolaminar_readout`."""
    signal_arr = np.asarray(signal_arr, dtype=np.float32)
    T, N = signal_arr.shape

    area_indices = []
    pos_from_l4_list = []

    area_arr = np.array(neurons.get("area", ["V1"] * N))
    pos_l4_arr = np.array(neurons.get("pos_from_l4", np.linspace(-0.5, 0.5, N)))

    mask = area_arr == area
    area_indices = np.where(mask)[0].tolist()
    pos_from_l4_list = pos_l4_arr[mask].astype(float).tolist()

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
            "metadata": _spectrolaminar_readout_metadata(),
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
        "metadata": _spectrolaminar_readout_metadata(),
    }

    return readout


def multi_area_spectrolaminar_readout(
    signal_arr: jax.Array,
    neurons: Mapping[str, Any],
    n_freqs: int = 128,
    n_contacts: int | None = None,
    dt_ms: float = 0.1,
) -> dict[str, dict[str, Any]]:
    """Documented public function `multi_area_spectrolaminar_readout`."""
    areas = set(neurons.get("area", ["V1"]))
    readouts = {}

    for area in sorted(areas):
        readouts[area] = spectrolaminar_readout(
            signal_arr, neurons, area=area, n_freqs=n_freqs, n_contacts=n_contacts, dt_ms=dt_ms
        )

    return readouts


@dataclass(frozen=True)
class LinearReadout:
    """Linear (superposition-respecting) proxy readout operator ``y = source @ Wᵀ``.

    ``name`` labels the operator; ``W`` is the readout/leadfield matrix
    (``[n_contacts, n_sources]``). The status fields document the truth gates and
    are not escalated by the code: ``leadfield_status`` defaults to
    ``"toy_or_declared_proxy"`` (no calibrated leadfield), ``operator_status`` to
    ``"simulated_proxy"``, and ``units_or_status`` to ``"relative_proxy_units"``
    (relative units, not calibrated physical amplitudes).
    """

    name: str
    W: jax.Array
    leadfield_status: str = "toy_or_declared_proxy"
    operator_status: str = "simulated_proxy"
    units_or_status: str = "relative_proxy_units"

    def apply(self, source: jax.Array) -> jax.Array:
        """Documented public function `apply`."""
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
        """Documented public function `report`."""
        return {
            "name": self.name,
            "operator_status": self.operator_status,
            "leadfield_status": self.leadfield_status,
            "units_or_status": self.units_or_status,
            "physical_amplitude_calibrated": False,
        }


def csd_tensor(phi_e_proxy: jax.Array, dz: jax.Array | float) -> jax.Array:
    """Spatial second-derivative CSD tensor (readout family, depth-axis stage).

    ``csd_proxy[c] = -(phi[c+1] - 2*phi[c] + phi[c-1]) / dz**2`` along the
    contact axis, edge-padded at the boundaries. Standard pipeline shape:

    ``source -> project_laminar_sources (-> phi_e_proxy) -> csd_tensor -> csd_proxy``

    Factored out of :func:`project_laminar_sources` (which still calls this
    function internally) so CSD can be recomputed standalone from any
    ``[T, n_contacts]`` potential-proxy array without re-running the full
    projection. Unlike :func:`cable_filter_sources`, this is a purely spatial
    (depth-axis) operator, not a frequency-domain one -- CSD is the 2nd
    spatial derivative of whatever LFP-proxy it is given, nothing more.

    Parameters:
        phi_e_proxy: extracellular-potential proxy, shape ``[T, n_contacts]``.
        dz: contact spacing in the same relative-depth units as the contact
            axis (scalar).

    Returns: CSD proxy, shape ``[T, n_contacts]``. Returns zeros when
    ``n_contacts < 3`` (the stencil is undefined with fewer than 3 points).
    """
    phi_e_proxy = jnp.asarray(phi_e_proxy)
    n_contacts = phi_e_proxy.shape[-1]
    if n_contacts < 3:
        return jnp.zeros_like(phi_e_proxy)
    dz = jnp.asarray(dz, dtype=phi_e_proxy.dtype)
    padded = jnp.pad(phi_e_proxy, ((0, 0), (1, 1)), mode="edge")
    return -(padded[:, 2:] - 2.0 * padded[:, 1:-1] + padded[:, :-2]) / (dz * dz)


def cable_filter_tau(
    cell_type: Sequence[str] | np.ndarray,
    depth_z: jax.Array,
    *,
    tau_e_superficial_ms: float = 1.0,
    tau_e_deep_ms: float = 5.0,
    tau_pv_ms: float = 0.5,
    tau_sst_ms: float = 2.0,
    tau_vip_ms: float = 2.0,
) -> jax.Array:
    """Per-neuron cable time constant tau(cell_type, depth) in seconds.

    Builds the ``tau_s`` input expected by :func:`cable_filter_sources`. ``E``
    cells get a depth-graded tau, linearly interpolated from
    ``tau_e_superficial_ms`` at ``depth_z=0`` to ``tau_e_deep_ms`` at
    ``depth_z=1`` (long apical dendrites on deep pyramidal cells => longer
    cable time constant => lower low-pass cutoff). ``PV``/``SST``/``VIP``
    each get a fixed tau (fast-spiking ``PV`` shortest, so gamma passes at
    every depth). Any other/unrecognized cell type falls back to the
    ``SST``/``VIP`` tau. Defaults are the validated ``order2_same_tau`` sweep
    point (see :func:`cable_filter_sources`).

    Parameters
    ----------
    cell_type:
        Per-neuron cell-type labels, length [N] (e.g. from
        ``model.neuron_table()``).
    depth_z:
        Per-neuron normalized laminar depth in [0, 1], length [N]
        (0=superficial, 1=deep).
    """
    cell_type_arr = np.asarray(cell_type)
    depth_z_arr = np.asarray(depth_z, dtype=np.float64)
    if cell_type_arr.shape[0] != depth_z_arr.shape[0]:
        raise ValueError(
            f"cell_type length {cell_type_arr.shape[0]} does not match depth_z length {depth_z_arr.shape[0]}"
        )

    tau_ms = np.full_like(depth_z_arr, tau_sst_ms)
    is_e = cell_type_arr == "E"
    tau_ms[is_e] = tau_e_superficial_ms + (tau_e_deep_ms - tau_e_superficial_ms) * depth_z_arr[is_e]
    tau_ms[cell_type_arr == "PV"] = tau_pv_ms
    tau_ms[cell_type_arr == "SST"] = tau_sst_ms
    tau_ms[cell_type_arr == "VIP"] = tau_vip_ms

    return jnp.asarray(tau_ms / 1000.0, dtype=jnp.float32)


def cable_filter_sources(
    sources: jax.Array,
    tau_s: jax.Array,
    dt_ms: float,
    *,
    order: int = 2,
) -> jax.Array:
    """Apply a depth/cell-type-dependent passive-cable low-pass TENSOR to sources.

    Standard cable-filter stage of the source-generation pipeline::

        emitter -> (source_scale gain tensor) -> source
                -> cable_filter_sources (this tensor)
                -> readout (project_laminar_sources / eeg_proxy_transform / meg_proxy_transform)

    Computes, per neuron, a cascaded single-pole RC low-pass transfer
    function ``H[f, n] = 1 / (1 + 2j*pi*f*tau_s[n]) ** order`` and applies it
    along the time axis via FFT. This is a phenomenological proxy for passive
    dendritic cable filtering, motivated by depth/cell-type-dependent
    membrane time constants (deep pyramidal apical dendrites => long tau =>
    relatively preserved low-frequency power; fast-spiking interneurons =>
    short tau => high-frequency power passes everywhere) — it is not a
    calibrated cable-equation solve: ``field_solver_status`` stays
    ``"linear_solver"`` and ``physical_amplitude_calibrated`` stays ``False``.

    Validated on a 100-neuron canonical V1 column (10 trials x 6000 ms,
    ``tau_s`` from :func:`cable_filter_tau` defaults, ``order=2``): alpha/beta
    deep:superficial power ratio 1.30 (clearly deep-dominant, unchanged
    direction from the unfiltered baseline) and gamma deep:superficial power
    ratio 0.66 (flips to superficial-dominant, a genuine absolute
    band-selective effect — absent from the unfiltered baseline, which has no
    frequency-dependent operator and shows the same flat ~1.7x deep gain in
    every band). Theta is left effectively unaffected (2.13 vs the unfiltered
    baseline). ``order=1`` produces the same qualitative direction but a much
    weaker gamma flip (0.93, barely below parity) — ``order=2`` is the
    validated default for a clean split.

    Parameters
    ----------
    sources:
        Source-proxy traces with shape [T, N].
    tau_s:
        Per-neuron cable time constant in seconds, shape [N]. Build with
        :func:`cable_filter_tau`.
    dt_ms:
        Simulation timestep in milliseconds.
    order:
        Number of cascaded single-pole sections (default 2; validated —
        see above).
    """
    sources = jnp.asarray(sources, dtype=jnp.float32)
    tau_s = jnp.asarray(tau_s, dtype=jnp.float32)
    if sources.ndim != 2:
        raise ValueError(f"sources must be 2D [T, N], got shape {sources.shape}")
    if tau_s.shape[0] != sources.shape[1]:
        raise ValueError(
            f"tau_s length {tau_s.shape[0]} does not match sources width {sources.shape[1]}"
        )

    T = sources.shape[0]
    dt_s = dt_ms / 1000.0
    freqs = jnp.fft.rfftfreq(T, d=dt_s)
    H = 1.0 / (1.0 + 1j * 2.0 * jnp.pi * freqs[:, None] * tau_s[None, :])
    H = H ** order
    src_fft = jnp.fft.rfft(sources, axis=0)
    filtered = jnp.fft.irfft(src_fft * H, n=T, axis=0)
    return jnp.asarray(filtered, dtype=jnp.float32)


def cable_filter_report(tau_s: jax.Array, order: int = 2) -> dict[str, Any]:
    """JSON-safe truth-gate report for a :func:`cable_filter_sources` call."""
    tau_s_np = np.asarray(tau_s)
    cutoff_hz = 1.0 / (2.0 * np.pi * tau_s_np)
    finite = bool(np.all(np.isfinite(tau_s_np)))
    return {
        "operator": "cable_filter_tensor",
        "operator_status": "simulated_proxy",
        "mechanism": "passive_dendritic_cable_low_pass_phenomenological",
        "order": int(order),
        "tau_s_mean": float(np.mean(tau_s_np)) if finite else None,
        "tau_s_min": float(np.min(tau_s_np)) if finite else None,
        "tau_s_max": float(np.max(tau_s_np)) if finite else None,
        "cutoff_hz_mean": float(np.mean(cutoff_hz)) if finite else None,
        **_proxy_truth_gate_fragment(
            source_calibration_status="uncalibrated_izhikevich_native_current",
            claim_level="computational_scaffold",
        ),
        "finite_tau": finite,
    }


def construct_source_tensor(
    *,
    mode: str = "total_membrane_current_proxy",
    total_membrane_current: jax.Array | None = None,
    decomposed_cap_ion: jax.Array | None = None,
    synaptic_current: jax.Array | None = None,
    spike_proxy: jax.Array | None = None,
    scale: float = 1.0,
) -> tuple[jax.Array, dict[str, Any]]:
    """Assemble a source tensor for laminar projection from the named ``mode``.

    Selects one source basis via ``mode`` (default
    ``"total_membrane_current_proxy"``; also ``"decomposed_cap_ion_plus_synaptic_proxy"``
    or ``"spike_proxy"``) and applies ``scale``. The array required by the chosen
    mode must be supplied. Returns the source array plus a JSON-safe metadata dict
    recording the mode and proxy status. The result is a proxy source basis, not a
    physical current density.
    """
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
        raise ValueError(
            f"Unsupported construct_source_tensor mode {mode!r}. "
            "Valid modes: 'total_membrane_current_proxy', "
            "'decomposed_cap_ion_plus_synaptic_proxy', 'spike_proxy'."
        )

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
        "physical_amplitude_calibrated": False,
        "finite": finite_bool,
    }
    return source, report


def synaptic_resonance_source(
    neurons: Any,
    n_steps: int,
    dt_ms: float = 0.1,
    control_params: Optional[dict] = None,
) -> jax.Array:
    """Documented public function `synaptic_resonance_source`."""
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
    """Documented public function `combined_multi_area_source`."""
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
    """Documented public function `spectrolaminar_similarity`."""
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


class LegacyMultiAreaSpectrolaminarObjective:
    """Legacy multi-area spectrolaminar objective wrapper.

    Distinct from the active :func:`jaxfne.objectives.spectrolaminar_objective`
    function (the maintained scoring path, called by
    :func:`jaxfne.objectives.spectrolaminar_objective_factory`). This class
    predates that function and is retained for backward compatibility under
    its original name :data:`spectrolaminar_objective` (this module); prefer
    importing it as ``LegacyMultiAreaSpectrolaminarObjective`` in new code to
    avoid the name collision with ``jaxfne.objectives.spectrolaminar_objective``.
    """
    def __init__(self, target_profiles: Optional[dict] = None):
        self.target_profiles = target_profiles or {}

    def score(self, readouts: dict[str, dict], spikes: Optional[dict] = None) -> float:
        """Documented public function `score`."""
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


# Backward-compatible alias: this is the original public name. Existing
# imports of `spectrolaminar_objective` from this module/`jaxfne.fields`
# continue to work unchanged; new code should prefer the disambiguated name
# above to avoid colliding with `jaxfne.objectives.spectrolaminar_objective`.
spectrolaminar_objective = LegacyMultiAreaSpectrolaminarObjective
