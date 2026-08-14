"""Protocol W3b — parameter-domain map for active robust HDP (analysis only).

Branches from frozen W3a at ``08bd4a2``. Maps
(I_tonic, kappa_H, kappa_W, lambda_W, tau_H, tau_W) to dynamical regimes using
corrected Floquet margins (see ``w3a_margin_audit.json``).

Does not tune parameters post-hoc; may return D_useful = empty.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxfne.w3_stability_analysis import W3NominalParameters
from jaxfne.w3a_stability_analysis import (
    W3aScanConfig,
    detect_period,
    floquet_summary,
    initial_state_from_silent_rest,
    monodromy_matrix,
    simulate_trajectory,
    with_tonic_drive,
)

W3A_FREEZE_SHA = "08bd4a2b2ebc89373697cd4be1da85551e7c5952"

# Preregistered full W3b lattice (frozen in w3b_parameter_domain_spec.json).
FROZEN_LATTICE_KAPPA_H = (0.02, 0.05, 0.1)
FROZEN_LATTICE_KAPPA_W = (0.5, 1.0, 2.0)
FROZEN_LATTICE_LAMBDA_W = (0.05, 0.1, 0.2)
FROZEN_LATTICE_TAU_H_MS = (60.0, 80.0, 120.0)
FROZEN_LATTICE_TAU_W_MS = (80.0, 100.0, 150.0)
FROZEN_LATTICE_I_TONIC = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)


class Regime(str, Enum):
    DORMANT = "D"
    STABLE = "S"
    CRITICAL = "C"
    UNSTABLE = "U"
    UNCLASSIFIED = "X"  # active but no validated periodic orbit for Floquet


@dataclass(frozen=True)
class W3bFrozenGates:
    """Preregistered W3b quantitative gates (fixed before domain scan)."""

    active_syn_threshold: float = 1e-3
    robust_margin: float = 0.02
    neutral_multiplier_epsilon: float = 0.05
    min_period_steps: int = 2
    l_hdp_min: float = 1e-6
    r_tau_min: float = 1.0


@dataclass(frozen=True)
class W3bParameterPoint:
    """One point in the W3b parameter domain (HDP architecture knobs only)."""

    kappa_h: float
    kappa_w: float
    lambda_w: float
    tau_h_ms: float
    tau_w_ms: float

    def r_tau(self) -> float:
        return (self.tau_w_ms / self.lambda_w) / self.tau_h_ms

    def to_nominal(self, base: W3NominalParameters | None = None) -> W3NominalParameters:
        b = base or W3NominalParameters()
        return replace(
            b,
            kappa_h=self.kappa_h,
            kappa_w=self.kappa_w,
            lambda_w=self.lambda_w,
            tau_h_ms=self.tau_h_ms,
            tau_w_ms=self.tau_w_ms,
        )


@dataclass(frozen=True)
class W3bDomainScanConfig:
    """Preregistered W3b scan lattice."""

    i_tonic_grid: tuple[float, ...] = FROZEN_LATTICE_I_TONIC
    parameter_points: tuple[W3bParameterPoint, ...] = ()
    w3a_sim: W3aScanConfig = W3aScanConfig(
        i_tonic_grid=(0.0,),
        burn_in_steps=400,
        sample_steps=600,
        period_search_max=80,
    )

    @staticmethod
    def frozen_full_lattice() -> W3bDomainScanConfig:
        pts = tuple(
            W3bParameterPoint(kh, kw, lw, th, tw)
            for kh, kw, lw, th, tw in product(
                FROZEN_LATTICE_KAPPA_H,
                FROZEN_LATTICE_KAPPA_W,
                FROZEN_LATTICE_LAMBDA_W,
                FROZEN_LATTICE_TAU_H_MS,
                FROZEN_LATTICE_TAU_W_MS,
            )
        )
        return W3bDomainScanConfig(parameter_points=pts)


def floquet_margin_nonneutral(
    m: np.ndarray,
    *,
    epsilon_neutral: float,
) -> dict[str, Any]:
    """m_F = 1 - rho_nonneutral(M) excluding multipliers near unit circle."""
    eigvals = np.linalg.eigvals(m)
    mods = np.abs(eigvals)
    nonneutral = mods[np.abs(mods - 1.0) > epsilon_neutral]
    if nonneutral.size == 0:
        return {
            "rho_nonneutral": 1.0,
            "m_F": 0.0,
            "all_multipliers_neutral": True,
        }
    rho_nn = float(np.max(nonneutral))
    return {
        "rho_nonneutral": rho_nn,
        "m_F": float(1.0 - rho_nn),
        "all_multipliers_neutral": False,
        "eigenvalue_moduli": [float(x) for x in np.sort(mods)[::-1]],
    }


def validate_period_for_floquet(
    trace: np.ndarray,
    period_info: dict[str, Any],
    *,
    gates: W3bFrozenGates,
) -> dict[str, Any]:
    if not period_info.get("found"):
        return {"valid": False, "reason": "no_period_detected"}
    period = int(period_info["period"])
    if period < gates.min_period_steps:
        return {"valid": False, "reason": "period_one_rejected", "period": period}
    tail = trace[-period:]
    spike_proxy = (tail[:, 0] <= -64.5) | (tail[:, 1] <= -64.5)
    if not bool(np.any(spike_proxy)):
        return {"valid": False, "reason": "no_spike_in_period", "period": period}
    return {"valid": True, "period": period}


def gamma_hdp(p: W3bParameterPoint, b_hw: float, *, a_h: float = 1.0) -> float:
    """|2*kappa_W*b_HW / (a_H*lambda_W)| per reduced antisymmetric convention."""
    denom = a_h * p.lambda_w
    if denom == 0:
        return float("inf")
    return abs(2.0 * p.kappa_w * b_hw / denom)


def classify_regime(
    *,
    mean_syn: float,
    l_hdp: float,
    r_tau: float,
    m_f: float | None,
    floquet_available: bool,
    gates: W3bFrozenGates,
) -> Regime:
    if mean_syn <= gates.active_syn_threshold:
        return Regime.DORMANT
    if r_tau <= gates.r_tau_min:
        return Regime.UNCLASSIFIED
    if not floquet_available or m_f is None:
        return Regime.UNCLASSIFIED
    if l_hdp <= gates.l_hdp_min:
        return Regime.DORMANT
    if m_f <= 0.0:
        return Regime.UNSTABLE
    if m_f <= gates.robust_margin:
        return Regime.CRITICAL
    return Regime.STABLE


def analyze_w3b_point(
    param: W3bParameterPoint,
    i_tonic: float,
    *,
    gates: W3bFrozenGates | None = None,
    base: W3NominalParameters | None = None,
    sim_cfg: W3aScanConfig | None = None,
) -> dict[str, Any]:
    gates = gates or W3bFrozenGates()
    sim_cfg = sim_cfg or W3bDomainScanConfig().w3a_sim
    p = with_tonic_drive(param.to_nominal(base), i_tonic)

    z0 = initial_state_from_silent_rest(p)
    trace = np.asarray(
        simulate_trajectory(z0, p, sim_cfg.burn_in_steps + sim_cfg.sample_steps),
        dtype=np.float64,
    )
    tail = trace[sim_cfg.burn_in_steps :]
    period_info = detect_period(
        tail,
        max_period=sim_cfg.period_search_max,
        tol=sim_cfg.period_match_tol,
    )
    period_val = validate_period_for_floquet(tail, period_info, gates=gates)

    mean_syn = float(max(np.mean(tail[:, 4]), np.mean(tail[:, 5])))
    from jaxfne.w3a_stability_analysis import derive_b_hw_symbolic_path, loop_gain_hdp

    z_end = jnp.asarray(tail[-1])
    b_hw = float(derive_b_hw_symbolic_path(p, z_end)["b_HW_from_autodiff"])
    l_gain = loop_gain_hdp(p, syn_ab=float(np.mean(tail[:, 4])), syn_ba=float(np.mean(tail[:, 5])), w_star=p.w0)
    l_hdp = float(l_gain["L_HDP"])
    g_hdp = gamma_hdp(param, b_hw)

    m_f = None
    floquet_block: dict[str, Any] = {"available": False}
    if period_val.get("valid"):
        period = int(period_val["period"])
        z_ref = jnp.asarray(tail[-period])
        m = monodromy_matrix(z_ref, p, period)
        floquet_block = {
            "available": True,
            "period": period,
            "legacy_subleading": floquet_summary(m, near_critical=gates.robust_margin),
            "nonneutral": floquet_margin_nonneutral(
                m, epsilon_neutral=gates.neutral_multiplier_epsilon
            ),
        }
        m_f = float(floquet_block["nonneutral"]["m_F"])

    regime = classify_regime(
        mean_syn=mean_syn,
        l_hdp=l_hdp,
        r_tau=param.r_tau(),
        m_f=m_f,
        floquet_available=bool(floquet_block.get("available")),
        gates=gates,
    )

    orbit_status = "dormant"
    period_t: int | None = None
    rho_nn: float | None = None
    if mean_syn > gates.active_syn_threshold:
        if period_val.get("valid"):
            orbit_status = "validated_periodic_orbit"
            period_t = int(period_val["period"])
            rho_nn = float(floquet_block["nonneutral"]["rho_nonneutral"])
        else:
            orbit_status = str(period_val.get("reason", "stability_unresolved"))

    return {
        "kappa_H": param.kappa_h,
        "kappa_W": param.kappa_w,
        "lambda_W": param.lambda_w,
        "tau_H": param.tau_h_ms,
        "tau_W": param.tau_w_ms,
        "I_tonic": float(i_tonic),
        "mean_syn": mean_syn,
        "L_HDP": l_hdp,
        "r_tau": param.r_tau(),
        "Gamma_HDP": g_hdp,
        "b_HW": b_hw,
        "orbit_status": orbit_status,
        "T": period_t,
        "rho_nonneutral": rho_nn,
        "m_F": m_f,
        "regime": regime.value,
        "in_D_useful": regime == Regime.STABLE,
        "parameter_point": {
            "kappa_h": param.kappa_h,
            "kappa_w": param.kappa_w,
            "lambda_w": param.lambda_w,
            "tau_h_ms": param.tau_h_ms,
            "tau_w_ms": param.tau_w_ms,
            "r_tau": param.r_tau(),
        },
        "I_tonic": float(i_tonic),
        "Gamma_HDP": g_hdp,
        "b_HW": b_hw,
        "L_HDP": l_hdp,
        "mean_syn": mean_syn,
        "m_F": m_f,
        "period_validation": period_val,
        "floquet": floquet_block,
        "regime": regime.value,
        "in_D_useful": regime == Regime.STABLE,
    }


def selection_rule_max_margin(
    candidates: list[dict[str, Any]],
    *,
    gates: W3bFrozenGates | None = None,
) -> dict[str, Any] | None:
    """Fixed rule: among S-regime points, pick max m_F subject to gates (pre-memory)."""
    gates = gates or W3bFrozenGates()
    pool = [c for c in candidates if c.get("regime") == Regime.STABLE.value]
    if not pool:
        return None
    pool = [c for c in pool if c["L_HDP"] > gates.l_hdp_min and c["parameter_point"]["r_tau"] > gates.r_tau_min]
    if not pool:
        return None
    return max(pool, key=lambda c: float(c.get("m_F") or -1.0))


def _count_active_unresolved(
    results: list[dict[str, Any]],
    *,
    gates: W3bFrozenGates,
) -> int:
    return sum(
        1
        for r in results
        if r["regime"] == Regime.UNCLASSIFIED.value
        and r["mean_syn"] > gates.active_syn_threshold
    )


def _interpretation_branch(n_s: int, n_x: int) -> dict[str, Any]:
    if n_s > 0:
        branch = "N_S_gt_0"
        text = "Nonempty robust active domain; apply preregistered selection rule; consider W3."
        next_steps = ["freeze_selected_operating_point", "authorize_W3_perturb_relax_probe"]
    elif n_x == 0:
        branch = "N_S_eq_0_and_N_X_eq_0"
        text = (
            "Genuine negative parameter-domain result over tested lattice; "
            "close minimal HDP law and version new F_W."
        )
        next_steps = ["version_new_plasticity_law"]
    else:
        branch = "N_S_eq_0_and_N_X_gt_0"
        text = (
            "No demonstrated robust active domain; active regimes stability-unresolved "
            "(X != U). Next step is orbit characterization (W3c), not law redesign."
        )
        next_steps = ["W3c_orbit_characterization"]
    return {
        "branch": branch,
        "interpretation": text,
        "next_steps": next_steps,
    }


def run_w3b_domain_scan(
    cfg: W3bDomainScanConfig | None = None,
    gates: W3bFrozenGates | None = None,
) -> dict[str, Any]:
    cfg = cfg or W3bDomainScanConfig.frozen_full_lattice()
    gates = gates or W3bFrozenGates()
    results: list[dict[str, Any]] = []
    n_total = len(cfg.parameter_points) * len(cfg.i_tonic_grid)
    for i_pt, param in enumerate(cfg.parameter_points):
        for i_tonic in cfg.i_tonic_grid:
            results.append(analyze_w3b_point(param, float(i_tonic), gates=gates, sim_cfg=cfg.w3a_sim))
            if len(results) % 100 == 0:
                print(f"W3b scan progress: {len(results)}/{n_total}", flush=True)

    useful = [r for r in results if r["in_D_useful"]]
    selected = selection_rule_max_margin(results, gates=gates)
    n_s = len(useful)
    n_x = _count_active_unresolved(results, gates=gates)

    regime_counts = {k.value: 0 for k in Regime}
    for r in results:
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1

    branch = _interpretation_branch(n_s, n_x)

    return {
        "schema": "protocol_w_w3b_domain_receipt.v1",
        "status": "FROZEN_ANALYSIS",
        "write_once": True,
        "analysis_only": True,
        "w3_kernel_implementation_authorized": False,
        "parent_w3a_sha": W3A_FREEZE_SHA,
        "margin_audit": "artifacts/protocol_w/w3a_stability/w3a_margin_audit.json",
        "specification": "artifacts/protocol_w/w3b_parameter_domain/w3b_parameter_domain_spec.json",
        "scientific_question": (
            "Exists (kappa_H, kappa_W, lambda_W, tau_H, tau_W, I_tonic) in preregistered lattice "
            "satisfying useful-domain gates?"
        ),
        "frozen_lattice": {
            "kappa_H": list(FROZEN_LATTICE_KAPPA_H),
            "kappa_W": list(FROZEN_LATTICE_KAPPA_W),
            "lambda_W": list(FROZEN_LATTICE_LAMBDA_W),
            "tau_H_ms": list(FROZEN_LATTICE_TAU_H_MS),
            "tau_W_ms": list(FROZEN_LATTICE_TAU_W_MS),
            "I_tonic": list(cfg.i_tonic_grid),
            "total_points": n_total,
        },
        "frozen_gates": {
            "active_syn_threshold": gates.active_syn_threshold,
            "robust_margin_m_F": gates.robust_margin,
            "L_HDP_min": gates.l_hdp_min,
            "r_tau_min": gates.r_tau_min,
            "neutral_multiplier_epsilon": gates.neutral_multiplier_epsilon,
            "min_period_steps": gates.min_period_steps,
        },
        "regime_labels": {
            "D": "dormant/vanishing feedback",
            "S": "robustly stable active HDP (m_F > 0.02)",
            "C": "near-critical (0 < m_F <= 0.02)",
            "U": "unstable (m_F <= 0); negative evidence",
            "X": "active but stability-unresolved (X != U)",
        },
        "aggregate_quantities": {
            "N_S": n_s,
            "N_X": n_x,
            "D_useful_count": n_s,
            "D_useful_empty": n_s == 0,
        },
        "regime_counts": regime_counts,
        "regime_distribution": regime_counts,
        "interpretation": branch,
        "scan_results": results,
        "selection_rule": "max m_F among S-regime points with L_HDP>L_min and r_tau>1 (fixed before memory experiment)",
        "selected_operating_point": selected,
    }


def export_w3b_domain_receipt(
    cfg: W3bDomainScanConfig | None = None,
    gates: W3bFrozenGates | None = None,
) -> dict[str, Any]:
    return run_w3b_domain_scan(cfg, gates)
