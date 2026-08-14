"""Protocol W3b — parameter-domain map for active robust HDP (analysis only).

Branches from frozen W3a at ``08bd4a2``. Maps
(I_tonic, kappa_H, kappa_W, lambda_W, tau_H, tau_W) to dynamical regimes using
corrected Floquet margins (see ``w3a_margin_audit.json``).

Does not tune parameters post-hoc; may return D_useful = empty.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
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
    """Preregistered coarse W3b scan lattice (analysis budget, not tuned)."""

    i_tonic_grid: tuple[float, ...] = tuple(np.linspace(0.0, 40.0, 9).tolist())
    parameter_points: tuple[W3bParameterPoint, ...] = (
        W3bParameterPoint(0.05, 1.0, 0.1, 80.0, 100.0),  # W3a nominal
    )
    w3a_sim: W3aScanConfig = W3aScanConfig(
        i_tonic_grid=(0.0,),  # overridden per call
        burn_in_steps=400,
        sample_steps=600,
        period_search_max=80,
    )


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

    return {
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


def run_w3b_domain_scan(
    cfg: W3bDomainScanConfig | None = None,
    gates: W3bFrozenGates | None = None,
) -> dict[str, Any]:
    cfg = cfg or W3bDomainScanConfig()
    gates = gates or W3bFrozenGates()
    results: list[dict[str, Any]] = []
    for param in cfg.parameter_points:
        for i_tonic in cfg.i_tonic_grid:
            results.append(analyze_w3b_point(param, float(i_tonic), gates=gates, sim_cfg=cfg.w3a_sim))

    useful = [r for r in results if r["in_D_useful"]]
    selected = selection_rule_max_margin(useful, gates=gates)

    regime_counts = {k.value: 0 for k in Regime}
    for r in results:
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1

    return {
        "schema": "protocol_w_w3b_domain_scan.v1",
        "status": "SPECIFICATION_OPEN",
        "analysis_only": True,
        "w3_kernel_implementation_authorized": False,
        "parent_w3a_sha": W3A_FREEZE_SHA,
        "margin_audit": "artifacts/protocol_w/w3a_stability/w3a_margin_audit.json",
        "scientific_question": "What parameter domain permits simultaneously active, bounded, nontrivial, robust HDP?",
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
            "U": "unstable (m_F <= 0)",
            "X": "unclassified (no validated Floquet or timescale fail)",
        },
        "D_useful_definition": "active AND robust_stable AND nontrivial AND r_tau>1",
        "D_useful_empty": len(useful) == 0,
        "D_useful_count": len(useful),
        "regime_counts": regime_counts,
        "scan_results": results,
        "selection_rule": "max m_F among S-regime points with L_HDP>L_min and r_tau>1 (fixed before memory experiment)",
        "selected_operating_point": selected,
        "interpretation_if_empty": (
            "If D_useful is empty, nominal linear HDP law may be insufficient; "
            "change plasticity law in a new protocol rather than patch parameters."
        ),
    }
