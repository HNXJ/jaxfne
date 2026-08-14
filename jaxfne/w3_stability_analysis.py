"""Protocol W3 — closed-loop local stability analysis (analysis only, no W3 kernel).

Derives equilibrium verification (gate zero), reduced antisymmetric Jacobian,
and full discrete-time step-map Jacobian faithful to the implemented RBD +
Izhikevich + synapse + W0 exp(omega) coupling grammar.

See ``artifacts/protocol_w/w3_stability/w3_stability_receipt.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import logm

from jaxfne.w1a_omega_plasticity import W1aConfig, euler_step_omega

jax.config.update("jax_enable_x64", True)
_JDTYPE = jnp.float64

IDX_A = 0
IDX_B = 1


@dataclass(frozen=True)
class W3NominalParameters:
    """Preregistered nominal W3 closed-loop parameter set (not tuned post-hoc)."""

    dt_ms: float = 1.0
    tau_h_ms: float = 80.0
    tau_w_ms: float = 100.0
    kappa_w: float = 1.0
    lambda_w: float = 0.1
    kappa_h: float = 0.05
    beta_h: float = 0.0
    i_ref: float = 1.0
    w0: float = 6.0
    syn_tau_ms: float = 3.0
    rbd_family: str = "f1"
    # Izhikevich (matches W1b/W2 two-node circuit)
    v_rest_init: float = -65.0
    u_rest_init: float = 0.0
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    spike_threshold: float = 30.0
    drive_a: float = 0.0
    drive_b: float = 0.0
    w1a: W1aConfig = field(default_factory=lambda: W1aConfig())

    def memory_timescale_ms(self) -> float:
        return self.tau_w_ms / self.lambda_w

    def timescale_hierarchy_ok(self) -> bool:
        return self.memory_timescale_ms() > self.tau_h_ms


def izhikevich_dv_du(
    v: jax.Array,
    u: jax.Array,
    i_native: jax.Array,
    *,
    a: float,
    b: float,
) -> tuple[jax.Array, jax.Array]:
    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + i_native
    du = a * (b * v - u)
    return dv, du


def izhikevich_silent_fixed_point(
    *,
    a: float = 0.02,
    b: float = 0.2,
    i_native: float = 0.0,
) -> tuple[float, float]:
    """Continuous-time silent fixed point (dv=du=0) for constant drive."""
    # u = b*v from du=0; quadratic from dv=0
    coeff = 0.04
    linear = 5.0 - b
    const = 140.0 + float(i_native)
    disc = linear * linear - 4.0 * coeff * const
    if disc < 0:
        raise ValueError("no real silent fixed point for given drive")
    sqrt_disc = float(np.sqrt(disc))
    v1 = (-linear + sqrt_disc) / (2.0 * coeff)
    v2 = (-linear - sqrt_disc) / (2.0 * coeff)
    # lower-v branch is the stable silent attractor for standard Izhikevich params
    v_star = min(v1, v2)
    u_star = b * v_star
    return float(v_star), float(u_star)


def rbd_f1_h_next(
    h: jax.Array,
    i_rec: jax.Array,
    *,
    dt: float,
    tau_h: float,
    kappa_h: float,
    i_ref: float,
) -> jax.Array:
    r = 1.0 - h
    d_h = (r + kappa_h * (i_rec / i_ref)) / tau_h
    return h + dt * d_h


def recurrent_gain(h: jax.Array, *, beta_h: float) -> jax.Array:
    if beta_h == 0.0:
        return jnp.ones_like(h)
    return 1.0 + beta_h * (h - 1.0)


def weight_from_omega_scalar(omega: jax.Array, w0: float) -> jax.Array:
    return jnp.asarray(w0, dtype=_JDTYPE) * jnp.exp(omega)


def pack_state(
    v: jax.Array,
    u: jax.Array,
    syn: jax.Array,
    h: jax.Array,
    omega: jax.Array,
) -> jax.Array:
    return jnp.concatenate([v, u, syn, h, omega], axis=0)


def unpack_state(z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    v = z[0:2]
    u = z[2:4]
    syn = z[4:6]
    h = z[6:8]
    omega = z[8:10]
    return v, u, syn, h, omega


def w3_step_subthreshold(
    z: jax.Array,
    p: W3NominalParameters,
) -> jax.Array:
    """One deterministic W3 step (subthreshold branch, no spike reset)."""
    v, u, syn, h, omega = unpack_state(z)
    w_ab = weight_from_omega_scalar(omega[0], p.w0)
    w_ba = weight_from_omega_scalar(omega[1], p.w0)
    edge_w = jnp.asarray([w_ab, w_ba], dtype=_JDTYPE)

    i_edge = edge_w * syn
    # post indices: edge0 A->B, edge1 B->A
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

    decay = jnp.exp(-p.dt_ms / p.syn_tau_ms)
    # subthreshold: presynaptic spikes = 0
    syn_next = syn * decay

    dh_ab = h[0] - h[1]
    omega_ab_next = euler_step_omega(
        omega[0],
        dh_ab,
        W1aConfig(
            tau_w=p.tau_w_ms,
            kappa_w=p.kappa_w,
            lambda_w=p.lambda_w,
            w0=p.w0,
            dt=p.dt_ms,
        ),
    )
    omega_ba_next = euler_step_omega(
        omega[1],
        -dh_ab,
        W1aConfig(
            tau_w=p.tau_w_ms,
            kappa_w=p.kappa_w,
            lambda_w=p.lambda_w,
            w0=p.w0,
            dt=p.dt_ms,
        ),
    )
    omega_next = jnp.asarray([omega_ab_next, omega_ba_next], dtype=_JDTYPE)

    return pack_state(v_next, u_next, syn_next, h_next, omega_next)


def equilibrium_residual(z: jax.Array, p: W3NominalParameters) -> jax.Array:
    return w3_step_subthreshold(z, p) - z


def find_equilibrium(
    p: W3NominalParameters,
    *,
    z0: jax.Array | None = None,
    max_iters: int = 80,
    tol: float = 1e-12,
) -> tuple[jax.Array, dict[str, Any]]:
    if z0 is None:
        v_s, u_s = izhikevich_silent_fixed_point(a=p.a, b=p.b, i_native=0.0)
        z0 = pack_state(
            jnp.asarray([v_s, v_s], dtype=_JDTYPE),
            jnp.asarray([u_s, u_s], dtype=_JDTYPE),
            jnp.zeros(2, dtype=_JDTYPE),
            jnp.ones(2, dtype=_JDTYPE),
            jnp.zeros(2, dtype=_JDTYPE),
        )
    z = z0
    hist = []
    for _ in range(max_iters):
        res = equilibrium_residual(z, p)
        norm = float(jnp.linalg.norm(res))
        hist.append(norm)
        if norm < tol:
            break
        j = jax.jacfwd(w3_step_subthreshold)(z, p)
        dz = jnp.linalg.solve(j - jnp.eye(z.shape[0], dtype=_JDTYPE), -res)
        z = z + dz
    final_res = equilibrium_residual(z, p)
    info = {
        "residual_norm": float(jnp.linalg.norm(final_res)),
        "residual_max_abs": float(jnp.max(jnp.abs(final_res))),
        "iteration_norms": hist,
        "converged": float(jnp.linalg.norm(final_res)) < tol,
    }
    return z, info


def jacobian_step(z: jax.Array, p: W3NominalParameters) -> jax.Array:
    return jax.jacfwd(w3_step_subthreshold)(z, p)


def jacobian_finite_difference(
    z: jax.Array,
    p: W3NominalParameters,
    *,
    eps: float = 1e-7,
) -> np.ndarray:
    n = z.shape[0]
    j = np.zeros((n, n), dtype=np.float64)
    base = w3_step_subthreshold(z, p)
    for i in range(n):
        dz = jnp.zeros_like(z).at[i].set(eps)
        j[:, i] = np.asarray((w3_step_subthreshold(z + dz, p) - base) / eps, dtype=np.float64)
    return j


def spectral_summary(j: np.ndarray, *, dt: float | None = None) -> dict[str, Any]:
    eigvals = np.linalg.eigvals(j)
    idx = np.argsort(-np.real(eigvals))
    eigvals = eigvals[idx]
    rho = float(np.max(np.abs(eigvals)))
    max_re = float(np.max(np.real(eigvals)))
    out: dict[str, Any] = {
        "eigenvalues": [
            {"real": float(np.real(ev)), "imag": float(np.imag(ev))} for ev in eigvals
        ],
        "spectral_radius": rho,
        "max_real_eigenvalue": max_re,
        "stable": max_re < 0.0,
        "discrete_stable": rho < 1.0,
    }
    if dt is not None:
        rates = (np.real(eigvals) - 1.0) / dt
        out["continuous_rate_from_step_eigenvalues"] = {
            "max_real": float(np.max(rates)),
            "min_real": float(np.min(rates)),
            "stable": bool(np.max(rates) < 0.0),
        }
        out["discrete_spectral_margin"] = float(1.0 - rho)
    return out


def analytical_reduced_jacobian_continuous(p: W3NominalParameters, b_hw: float) -> np.ndarray:
    """Analytic antisymmetric reduced Jacobian at F1 equilibrium."""
    a_h = 1.0
    return np.asarray(
        [
            [-a_h / p.tau_h_ms, b_hw / p.tau_h_ms],
            [2.0 * p.kappa_w / p.tau_w_ms, -p.lambda_w / p.tau_w_ms],
        ],
        dtype=np.float64,
    )


def antisymmetric_reduction_blocks(
    j_full: np.ndarray,
    *,
    state_labels: list[str],
) -> dict[str, Any]:
    """Extract antisymmetric (delta, Omega) block from full state ordering."""
    idx = {name: i for i, name in enumerate(state_labels)}
    # delta = h_A - h_B; Omega = omega_AB - omega_BA
    # linearization: delta_h = [1, -1] @ h, Omega = [1, -1] @ omega
    t_h = np.zeros(j_full.shape[0])
    t_h[idx["H_A"]] = 1.0
    t_h[idx["H_B"]] = -1.0
    t_o = np.zeros(j_full.shape[0])
    t_o[idx["omega_AB"]] = 1.0
    t_o[idx["omega_BA"]] = -1.0

    modes = np.vstack([t_h, t_o])
    j_red = modes @ j_full @ modes.T
    return {
        "projection_matrix": modes.tolist(),
        "J_red_step": j_red.tolist(),
        "spectral": spectral_summary(j_red, dt=None),
        "trace": float(np.trace(j_red)),
        "det": float(np.linalg.det(j_red)),
    }


def continuous_reduced_from_step(
    j_step: np.ndarray,
    *,
    dt: float,
    state_labels: list[str],
) -> dict[str, Any]:
    """Map discrete local Jacobian to continuous generator estimate J_c = log(J_step)/dt."""
    idx = {name: i for i, name in enumerate(state_labels)}
    t_h = np.zeros(j_step.shape[0])
    t_h[idx["H_A"]] = 1.0
    t_h[idx["H_B"]] = -1.0
    t_o = np.zeros(j_step.shape[0])
    t_o[idx["omega_AB"]] = 1.0
    t_o[idx["omega_BA"]] = -1.0
    modes = np.vstack([t_h, t_o])
    j_red_step = modes @ j_step @ modes.T
    j_cont = np.real(logm(j_red_step)) / dt
    a_h = -float(j_cont[0, 0])
    b_hw = float(j_cont[0, 1])
    kappa_block = float(j_cont[1, 0])
    lam_block = -float(j_cont[1, 1])
    return {
        "J_red_continuous_estimate": j_cont.tolist(),
        "a_H": a_h,
        "b_HW": b_hw,
        "kappa_eff": kappa_block,
        "lambda_eff": lam_block,
        "stability_inequality_aH_lambda_vs_2kappa_bHW": a_h * lam_block
        - 2.0 * kappa_block * b_hw,
        "spectral": spectral_summary(j_cont),
    }


def derive_b_hw_symbolic_path(p: W3NominalParameters, z_star: jax.Array) -> dict[str, Any]:
    """Derive b_HW from omega -> W -> I_rec -> H path at equilibrium via autodiff."""
    v, u, syn, h, omega = unpack_state(z_star)

    def delta_h_dot(omega_ab: jax.Array, omega_ba: jax.Array) -> jax.Array:
        w_ab = weight_from_omega_scalar(omega_ab, p.w0)
        w_ba = weight_from_omega_scalar(omega_ba, p.w0)
        i_edge = jnp.asarray([w_ab, w_ba]) * syn
        i_rec = jnp.asarray([i_edge[1], i_edge[0]])
        h_next = rbd_f1_h_next(
            h,
            i_rec,
            dt=p.dt_ms,
            tau_h=p.tau_h_ms,
            kappa_h=p.kappa_h,
            i_ref=p.i_ref,
        )
        delta_next = h_next[0] - h_next[1]
        delta = h[0] - h[1]
        return (delta_next - delta) / p.dt_ms

    grad_ab = jax.grad(lambda oa: delta_h_dot(oa, omega[1]))(omega[0])
    grad_ba = jax.grad(lambda ob: delta_h_dot(omega[0], ob))(omega[1])
    # antisymmetric Omega = omega_ab - omega_ba
    b_hw = float(grad_ab - grad_ba)
    return {
        "b_HW_from_autodiff": b_hw,
        "grad_delta_hdot_wrt_omega_ab": float(grad_ab),
        "grad_delta_hdot_wrt_omega_ba": float(grad_ba),
        "syn_ab_at_equilibrium": float(syn[0]),
        "syn_ba_at_equilibrium": float(syn[1]),
    }


def gate_zero_doctrinal_point(p: W3NominalParameters) -> dict[str, Any]:
    """Test whether doctrinal reference (v0,u0,H=1,omega=0,syn=0) is a fixed point."""
    z_doc = pack_state(
        jnp.asarray([p.v_rest_init, p.v_rest_init], dtype=_JDTYPE),
        jnp.asarray([p.u_rest_init, p.u_rest_init], dtype=_JDTYPE),
        jnp.zeros(2, dtype=_JDTYPE),
        jnp.ones(2, dtype=_JDTYPE),
        jnp.zeros(2, dtype=_JDTYPE),
    )
    res = equilibrium_residual(z_doc, p)
    return {
        "z_doctrinal": np.asarray(z_doc).tolist(),
        "residual_norm": float(jnp.linalg.norm(res)),
        "residual_max_abs": float(jnp.max(jnp.abs(res))),
        "is_fixed_point": float(jnp.linalg.norm(res)) < 1e-9,
    }


def run_w3_stability_analysis(
    p: W3NominalParameters | None = None,
) -> dict[str, Any]:
    p = p or W3NominalParameters()
    state_labels = [
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
    gate0_doc = gate_zero_doctrinal_point(p)
    v_s, u_s = izhikevich_silent_fixed_point(a=p.a, b=p.b, i_native=0.0)
    z_star, eq_info = find_equilibrium(p)
    z_host = np.asarray(z_star, dtype=np.float64)
    j_step_ad = np.asarray(jacobian_step(z_star, p), dtype=np.float64)
    j_step_fd = jacobian_finite_difference(z_star, p)
    fd_diff = float(np.max(np.abs(j_step_ad - j_step_fd)))
    spec_step = spectral_summary(j_step_ad, dt=p.dt_ms)
    red_step = antisymmetric_reduction_blocks(j_step_ad, state_labels=state_labels)
    red_cont = continuous_reduced_from_step(j_step_ad, dt=p.dt_ms, state_labels=state_labels)
    b_path = derive_b_hw_symbolic_path(p, z_star)

    b_hw = b_path["b_HW_from_autodiff"]
    j_red_analytic = analytical_reduced_jacobian_continuous(p, b_hw)
    red_analytic_spec = spectral_summary(j_red_analytic, dt=None)
    stability_ineq = float(p.tau_h_ms * p.lambda_w - 2.0 * p.kappa_w * b_hw * p.tau_w_ms)

    unstable_mode = None
    if not spec_step["discrete_stable"]:
        ev = spec_step["eigenvalues"][0]
        eigvecs = np.linalg.eig(j_step_ad)[1]
        unstable_mode = {
            "leading_eigenvalue": ev,
            "state_vector": eigvecs[:, 0].tolist(),
            "state_labels": state_labels,
        }

    gate_discrete_pass = bool(spec_step["discrete_stable"])
    gate_continuous_from_step = bool(
        spec_step.get("continuous_rate_from_step_eigenvalues", {}).get("stable", False)
    )

    return {
        "schema": "protocol_w_w3_stability_receipt.v1",
        "status": "FROZEN_ANALYSIS",
        "analysis_only": True,
        "w3_implementation_authorized": False,
        "nominal_parameters": asdict(p),
        "timescale_hierarchy": {
            "tau_mem_W_ms": p.memory_timescale_ms(),
            "tau_H_ms": p.tau_h_ms,
            "ordering_satisfied": p.timescale_hierarchy_ok(),
        },
        "state_ordering": state_labels,
        "gate_zero_doctrinal_reference": gate0_doc,
        "gate_zero_silent_izhikevich_fixed_point": {
            "v_star": v_s,
            "u_star": u_s,
            "note": "continuous dv=du=0 at I_native=0; used to seed numerical fixed-point search",
        },
        "equilibrium": {
            "z_star": z_host.tolist(),
            "convergence": eq_info,
            "H_A": float(z_host[6]),
            "H_B": float(z_host[7]),
            "omega_AB": float(z_host[8]),
            "omega_BA": float(z_host[9]),
            "syn_AB": float(z_host[4]),
            "syn_BA": float(z_host[5]),
        },
        "b_HW_derivation": b_path,
        "reduced_antisymmetric": {
            "definition": {
                "delta": "H_A - H_B",
                "Omega": "omega_AB - omega_BA",
            },
            "plastic_linearization_exact": "tau_W * Omega_dot = 2*kappa_W*delta - lambda_W*Omega",
            "rbd_linearization_exact_at_F1": "tau_H * delta_dot = -delta + (kappa_H/i_ref) * Delta_I_rec_antisym",
            "J_red_continuous_analytic": {
                "matrix": j_red_analytic.tolist(),
                "a_H": 1.0,
                "b_HW": b_hw,
                "spectral": red_analytic_spec,
                "stability_inequality_aH_tauH_lambdaW_vs_2kappaW_bHW_tauW": stability_ineq,
                "stability_inequality_pass": stability_ineq > 0.0,
            },
            "from_full_step_projection": red_step,
            "continuous_logm_estimate": red_cont,
            "closed_loop_linearly_active_at_equilibrium": abs(b_hw) > 1e-12,
        },
        "full_step_jacobian": {
            "J_step": j_step_ad.tolist(),
            "autodiff_vs_finite_difference_max_abs": fd_diff,
            "spectral": spec_step,
            "first_unstable_mode": unstable_mode,
        },
        "stability_gate": {
            "gate_zero_doctrinal_pass": bool(gate0_doc["is_fixed_point"]),
            "gate_zero_verified_equilibrium_pass": bool(eq_info["converged"]),
            "discrete_criterion": "rho(J_step) < 1",
            "discrete_pass": gate_discrete_pass,
            "discrete_spectral_margin": spec_step.get("discrete_spectral_margin"),
            "continuous_from_step_criterion": "max Re((lambda_step - 1)/dt) < 0",
            "continuous_from_step_pass": gate_continuous_from_step,
            "reduced_analytic_continuous_pass": bool(red_analytic_spec["stable"]),
            "reduced_inequality_form": "a_H * lambda_W * tau_H > 2 * kappa_W * b_HW * tau_W",
            "reduced_inequality_value": stability_ineq,
            "reduced_inequality_pass": stability_ineq > 0.0,
            "nominal_overall_discrete_gate_pass": gate_discrete_pass,
            "w3_kernel_implementation_authorized": False,
            "blocking_notes": [
                "Doctrinal (v,u)=(-65,0) is not a fixed point of the implemented subthreshold map.",
                "At verified silent equilibrium (syn=0), b_HW=0 exactly: H-omega closed loop is linearly inactive at rest.",
                "Transient/post-perturbation stability is not certified by this local rest analysis alone.",
            ],
        },
    }


def export_w3_stability_receipt(
    p: W3NominalParameters | None = None,
) -> dict[str, Any]:
    return run_w3_stability_analysis(p)
