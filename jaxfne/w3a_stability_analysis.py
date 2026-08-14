"""Protocol W3a — activity-enabled closed-loop stability (analysis only).

Preserves frozen silent-rest W3 result at ``953d03b``. Scans tonic operating drive
``I_tonic`` with fixed nominal HDP parameters and characterizes activity-dependent
``b_HW``, loop gain, and stability margins.

Gates (ordered):
  W3a-FP — self-consistent active fixed point search
  W3a-PO — periodic-orbit return-map / monodromy if FP with syn*>0 unavailable

See ``artifacts/protocol_w/w3a_stability/w3a_stability_spec.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.w3_stability_analysis import (
    W3NominalParameters,
    analytical_reduced_jacobian_continuous,
    derive_b_hw_symbolic_path,
    izhikevich_dv_du,
    pack_state,
    rbd_f1_h_next,
    recurrent_gain,
    spectral_summary,
    unpack_state,
    weight_from_omega_scalar,
)
from jaxfne.w1a_omega_plasticity import W1aConfig, euler_step_omega

jax.config.update("jax_enable_x64", True)
_JDTYPE = jnp.float64

W3_SILENT_REST_SHA = "953d03b60640ed48636f73524f03a6fce6fe92dc"

STATE_LABELS = [
    "v_A",
    "u_A",
    "v_B",
    "u_B",
    "syn_AB",
    "syn_BA",
    "H_A",
    "H_B",
    "omega_AB",
    "omega_BA",
]


@dataclass(frozen=True)
class W3aScanConfig:
    """Preregistered W3a operating-drive scan (I_tonic is the independent variable)."""

    i_tonic_grid: tuple[float, ...] = tuple(np.linspace(0.0, 60.0, 31).tolist())
    burn_in_steps: int = 400
    sample_steps: int = 600
    period_search_max: int = 80
    period_match_tol: float = 1e-4
    fp_newton_iters: int = 60
    fp_tol: float = 1e-10
    near_critical_margin: float = 0.01


def with_tonic_drive(p: W3NominalParameters, i_tonic: float) -> W3NominalParameters:
    return replace(p, drive_a=float(i_tonic), drive_b=float(i_tonic))


def w3a_step_spiking(z: jax.Array, p: W3NominalParameters) -> jax.Array:
    """One deterministic W3 step with Izhikevich spike reset (delay=0, analysis map)."""
    v, u, syn, h, omega = unpack_state(z)
    w_ab = weight_from_omega_scalar(omega[0], p.w0)
    w_ba = weight_from_omega_scalar(omega[1], p.w0)
    edge_w = jnp.asarray([w_ab, w_ba], dtype=_JDTYPE)

    i_edge = edge_w * syn
    i_rec = jnp.asarray([i_edge[1], i_edge[0]], dtype=_JDTYPE)

    h_next = rbd_f1_h_next(
        h,
        i_rec,
        dt=p.dt_ms,
        tau_h=p.tau_h_ms,
        kappa_h=p.kappa_h,
        i_ref=p.i_ref,
    )

    g_h = recurrent_gain(h, beta_h=p.beta_h)
    drive = jnp.asarray([p.drive_a, p.drive_b], dtype=_JDTYPE)
    i_native = drive + g_h * i_rec

    dv, du = izhikevich_dv_du(v, u, i_native, a=p.a, b=p.b)
    v_next = v + p.dt_ms * dv
    u_next = u + p.dt_ms * du

    spike = (v_next >= p.spike_threshold).astype(_JDTYPE)
    v_reset = jnp.where(spike > 0.5, jnp.asarray(p.c, dtype=_JDTYPE), v_next)
    u_reset = jnp.where(spike > 0.5, u_next + p.d, u_next)

    decay = jnp.exp(-p.dt_ms / p.syn_tau_ms)
    presyn_ab = spike[0]
    presyn_ba = spike[1]
    syn_ab_next = syn[0] * decay + presyn_ab
    syn_ba_next = syn[1] * decay + presyn_ba
    syn_next = jnp.asarray([syn_ab_next, syn_ba_next], dtype=_JDTYPE)

    dh_ab = h[0] - h[1]
    cfg_w = W1aConfig(
        tau_w=p.tau_w_ms,
        kappa_w=p.kappa_w,
        lambda_w=p.lambda_w,
        w0=p.w0,
        dt=p.dt_ms,
    )
    omega_next = jnp.asarray(
        [
            euler_step_omega(omega[0], dh_ab, cfg_w),
            euler_step_omega(omega[1], -dh_ab, cfg_w),
        ],
        dtype=_JDTYPE,
    )
    return pack_state(v_reset, u_reset, syn_next, h_next, omega_next)


def simulate_trajectory(
    z0: jax.Array,
    p: W3NominalParameters,
    n_steps: int,
) -> jax.Array:
    def body(z: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        z_next = w3a_step_spiking(z, p)
        return z_next, z_next

    z_final, trace = jax.lax.scan(body, z0, None, length=int(n_steps))
    return trace


def initial_state_from_silent_rest(p: W3NominalParameters) -> jax.Array:
    from jaxfne.w3_stability_analysis import izhikevich_silent_fixed_point

    try:
        v_s, u_s = izhikevich_silent_fixed_point(
            a=p.a, b=p.b, i_native=float(p.drive_a)
        )
    except ValueError:
        v_s, u_s = float(p.v_rest_init), float(p.b * p.v_rest_init)
    return pack_state(
        jnp.asarray([v_s, v_s], dtype=_JDTYPE),
        jnp.asarray([u_s, u_s], dtype=_JDTYPE),
        jnp.zeros(2, dtype=_JDTYPE),
        jnp.ones(2, dtype=_JDTYPE),
        jnp.zeros(2, dtype=_JDTYPE),
    )


def detect_period(
    trace: np.ndarray,
    *,
    max_period: int,
    tol: float,
) -> dict[str, Any]:
    """Detect minimal period T of tail trajectory."""
    n = trace.shape[0]
    if n < 2 * max_period:
        return {"found": False, "reason": "trace_too_short"}
    tail = trace[-max_period:]
    ref = tail[-1]
    for period in range(1, max_period + 1):
        if period >= n:
            break
        candidate = trace[-period]
        if float(np.max(np.abs(candidate - ref))) < tol:
            # verify full period consistency
            ok = True
            for k in range(1, min(period, 5)):
                if float(np.max(np.abs(trace[-period - k] - trace[-k]))) > 10 * tol:
                    ok = False
                    break
            if ok:
                return {"found": True, "period": int(period), "reference_state": ref.tolist()}
    return {"found": False, "reason": "no_period_within_max"}


def monodromy_matrix(
    z0: jax.Array,
    p: W3NominalParameters,
    period: int,
) -> np.ndarray:
    """Monodromy M = d P_T / d z for return map over one period."""

    def return_map(z: jax.Array) -> jax.Array:
        z_cur = z
        for _ in range(int(period)):
            z_cur = w3a_step_spiking(z_cur, p)
        return z_cur

    return np.asarray(jax.jacfwd(return_map)(z0), dtype=np.float64)


def floquet_summary(m: np.ndarray, *, near_critical: float) -> dict[str, Any]:
    eigvals = np.linalg.eigvals(m)
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    rho = float(np.max(np.abs(eigvals)))
    # neutral Floquet multiplier ~1 along periodic orbit; report subleading
    sorted_mod = np.sort(np.abs(eigvals))[::-1]
    rho_sub = float(sorted_mod[1]) if sorted_mod.size > 1 else rho
    margin = float(1.0 - rho_sub)
    return {
        "eigenvalues": [
            {"real": float(np.real(ev)), "imag": float(np.imag(ev))} for ev in eigvals
        ],
        "spectral_radius_all": rho,
        "spectral_radius_subleading": rho_sub,
        "floquet_margin_subleading": margin,
        "stable_subleading": rho_sub < 1.0,
        "near_critical": margin < near_critical,
    }


def loop_gain_hdp(
    p: W3NominalParameters,
    *,
    syn_ab: float,
    syn_ba: float,
    w_star: float,
) -> dict[str, float]:
    """Activity-dependent scalar loop gain L_HDP ~ (dH/dI)(dI/domega)(domega/dH)."""
    d_h_d_i = p.kappa_h / (p.i_ref * p.tau_h_ms)
    # antisymmetric channel: dI_antisym/domega ~ s* W*
    s_antisym = 0.5 * abs(syn_ab - syn_ba) + 0.5 * (syn_ab + syn_ba)  # use mean syn magnitude
    d_i_d_omega = float(w_star * max(syn_ab, syn_ba, s_antisym))
    d_omega_d_h = 2.0 * p.kappa_w / p.tau_w_ms
    l_hdp = d_h_d_i * d_i_d_omega * d_omega_d_h
    return {
        "dH_dI": d_h_d_i,
        "dI_domega": d_i_d_omega,
        "domega_dH": d_omega_d_h,
        "L_HDP": l_hdp,
    }


def reduced_stability_from_b_hw(
    p: W3NominalParameters,
    b_hw: float,
) -> dict[str, Any]:
    j_red = analytical_reduced_jacobian_continuous(p, b_hw)
    spec = spectral_summary(j_red, dt=None)
    ineq = float(p.tau_h_ms * p.lambda_w - 2.0 * p.kappa_w * b_hw * p.tau_w_ms)
    det = float(np.linalg.det(j_red))
    return {
        "J_red": j_red.tolist(),
        "b_HW": b_hw,
        "det_J_red": det,
        "det_positive": det > 0.0,
        "stability_inequality": ineq,
        "stability_inequality_pass": ineq > 0.0,
        "gain_bound_form": "a_H * lambda_W * tau_H > 2 * kappa_W * b_HW * tau_W",
        "spectral": spec,
    }


def search_active_fixed_point(
    p: W3NominalParameters,
    *,
    z0: jax.Array,
    cfg: W3aScanConfig,
) -> dict[str, Any]:
    """W3a-FP: Newton on spiking step map; report syn*>0 if found."""

    def residual(z: jax.Array) -> jax.Array:
        return w3a_step_spiking(z, p) - z

    z = z0
    hist = []
    for _ in range(cfg.fp_newton_iters):
        res = residual(z)
        norm = float(jnp.linalg.norm(res))
        hist.append(norm)
        if norm < cfg.fp_tol:
            break
        j = jax.jacfwd(w3a_step_spiking)(z, p)
        dz = jnp.linalg.solve(j - jnp.eye(z.shape[0], dtype=_JDTYPE), -res)
        z = z + dz

    z_host = np.asarray(z, dtype=np.float64)
    res_norm = float(jnp.linalg.norm(residual(z)))
    syn_ab, syn_ba = float(z_host[4]), float(z_host[5])
    active_syn = (syn_ab > 1e-8) or (syn_ba > 1e-8)
    b_path = derive_b_hw_symbolic_path(p, z) if res_norm < 1e-6 else None
    b_hw = float(b_path["b_HW_from_autodiff"]) if b_path else None
    return {
        "converged": res_norm < cfg.fp_tol,
        "residual_norm": res_norm,
        "z_star": z_host.tolist(),
        "syn_AB": syn_ab,
        "syn_BA": syn_ba,
        "active_synaptic_state": active_syn,
        "b_HW": b_hw,
        "iteration_norms": hist,
    }


def analyze_operating_point(
    i_tonic: float,
    p_base: W3NominalParameters,
    cfg: W3aScanConfig,
) -> dict[str, Any]:
    p = with_tonic_drive(p_base, i_tonic)
    z0 = initial_state_from_silent_rest(p)
    trace = np.asarray(
        simulate_trajectory(z0, p, cfg.burn_in_steps + cfg.sample_steps),
        dtype=np.float64,
    )
    tail = trace[cfg.burn_in_steps :]
    period_info = detect_period(
        tail,
        max_period=cfg.period_search_max,
        tol=cfg.period_match_tol,
    )

    fp = search_active_fixed_point(p, z0=z0, cfg=cfg)

    mean_syn_ab = float(np.mean(tail[:, 4]))
    mean_syn_ba = float(np.mean(tail[:, 5]))
    mean_spike_proxy = float(np.mean((tail[:, 0] <= p.c + 0.5).astype(float)))

    z_on_orbit = jnp.asarray(tail[-1], dtype=_JDTYPE)
    b_path = derive_b_hw_symbolic_path(p, z_on_orbit)
    b_hw = float(b_path["b_HW_from_autodiff"])
    w_star = float(p.w0)
    l_gain = loop_gain_hdp(p, syn_ab=mean_syn_ab, syn_ba=mean_syn_ba, w_star=w_star)
    red = reduced_stability_from_b_hw(p, b_hw)

    po_block: dict[str, Any] = {"activated": False}
    syn_active = (mean_syn_ab > 1e-3) or (mean_syn_ba > 1e-3)
    if period_info.get("found") and syn_active:
        period = int(period_info["period"])
        z_ref = jnp.asarray(tail[-period], dtype=_JDTYPE)
        m = monodromy_matrix(z_ref, p, period)
        po_block = {
            "activated": True,
            "period_steps": period,
            "monodromy": floquet_summary(m, near_critical=cfg.near_critical_margin),
        }

    j_step = np.asarray(jax.jacfwd(w3a_step_spiking)(z_on_orbit, p), dtype=np.float64)
    step_spec = spectral_summary(j_step, dt=p.dt_ms)

    return {
        "I_tonic": float(i_tonic),
        "mean_syn_AB": mean_syn_ab,
        "mean_syn_BA": mean_syn_ba,
        "syn_active": syn_active,
        "b_HW": b_hw,
        "b_HW_derivation": b_path,
        "loop_gain": l_gain,
        "reduced_stability": red,
        "local_step_jacobian": step_spec,
        "w3a_fp": fp,
        "w3a_po": po_block,
        "period_detection": period_info,
    }


def run_w3a_stability_scan(
    p: W3NominalParameters | None = None,
    cfg: W3aScanConfig | None = None,
) -> dict[str, Any]:
    p = p or W3NominalParameters()
    cfg = cfg or W3aScanConfig()
    scan = [analyze_operating_point(float(i), p, cfg) for i in cfg.i_tonic_grid]

    fp_active = [s for s in scan if s["w3a_fp"]["converged"] and s["w3a_fp"]["active_synaptic_state"]]
    po_active = [s for s in scan if s["w3a_po"].get("activated") and s["syn_active"]]

    b_hw_curve = [(s["I_tonic"], s["b_HW"]) for s in scan]
    margin_curve = [
        (
            s["I_tonic"],
            s["w3a_po"]["monodromy"]["floquet_margin_subleading"]
            if s["w3a_po"].get("activated")
            else s["local_step_jacobian"].get("discrete_spectral_margin"),
        )
        for s in scan
    ]

    # critical boundary: first active-syn point where reduced inequality fails OR Floquet unstable
    critical_ineq = None
    critical_floquet = None
    prev_margin = None
    prev_i = None
    for s in scan:
        if s["syn_active"] and not s["reduced_stability"]["stability_inequality_pass"]:
            critical_ineq = {
                "I_tonic_estimate": s["I_tonic"],
                "criterion": "a_H*lambda_W*tau_H <= 2*kappa_W*b_HW*tau_W",
                "b_HW": s["b_HW"],
            }
            break
    for s in scan:
        if not s["w3a_po"].get("activated"):
            continue
        m = s["w3a_po"]["monodromy"]["floquet_margin_subleading"]
        if prev_margin is not None and prev_margin > 0 and m < 0:
            critical_floquet = {
                "I_tonic_bracket": (prev_i, s["I_tonic"]),
                "margin_before": prev_margin,
                "margin_after": m,
            }
            break
        prev_margin = m
        prev_i = s["I_tonic"]

    return {
        "schema": "protocol_w_w3a_stability_receipt.v1",
        "status": "FROZEN_ANALYSIS",
        "analysis_only": True,
        "w3_kernel_implementation_authorized": False,
        "preserved_silent_rest_result_sha": W3_SILENT_REST_SHA,
        "theoretical_result_silent_rest": (
            "At silent equilibrium syn*=0, hence dI_rec/domega=0 and b_HW=0: "
            "synaptic HDP feedback is linearly dormant at rest."
        ),
        "scientific_question": (
            "Is the HDP closed loop locally stable around an active recurrent operating state?"
        ),
        "nominal_hdp_parameters": "frozen from W3NominalParameters; only I_tonic varies",
        "scan_config": {
            "i_tonic_grid": list(cfg.i_tonic_grid),
            "burn_in_steps": cfg.burn_in_steps,
            "sample_steps": cfg.sample_steps,
            "period_search_max": cfg.period_search_max,
            "near_critical_margin": cfg.near_critical_margin,
        },
        "gates": {
            "W3a_FP": {
                "description": "search for genuine active fixed point with syn*>0",
                "active_fp_found": len(fp_active) > 0,
                "count": len(fp_active),
            },
            "W3a_PO": {
                "description": "periodic-orbit return map / monodromy if FP unavailable",
                "activated_count": len(po_active),
                "primary_gate_when_fp_unavailable": True,
            },
        },
        "scan_results": scan,
        "summary_curves": {
            "I_tonic_vs_b_HW": b_hw_curve,
            "I_tonic_vs_stability_margin": margin_curve,
        },
        "critical_boundary_estimate": {
            "reduced_inequality": critical_ineq,
            "floquet_monodromy": critical_floquet,
        },
        "interpretation": {
            "activity_dependent_loop_gain": (
                "L_HDP grows with synaptic activity because dI/domega ~ s* W*."
            ),
            "near_critical_policy": (
                f"Margins below {cfg.near_critical_margin} are near-critical, not robustly stable."
            ),
        },
    }


def export_w3a_stability_receipt(
    p: W3NominalParameters | None = None,
    cfg: W3aScanConfig | None = None,
) -> dict[str, Any]:
    return run_w3a_stability_scan(p, cfg)
