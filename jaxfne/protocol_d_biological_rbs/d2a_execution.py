"""D2a autonomous H_K F1 relaxation — execution and gate receipts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import (
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery,
)
from jaxfne.io import json_safe
from jaxfne.protocol_d_biological_rbs.d1_execution import _build_isolated_circuit, _drive_schedule
from jaxfne.protocol_d_biological_rbs.d1_protocol import load_d1_spec
from jaxfne.protocol_d_biological_rbs.d2a_protocol import (
    D2A_EXECUTION_RECEIPT_PATH,
    d2a_h_k0_values,
    load_d2a_spec,
    validate_d2a_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def h_k_f1_analytic(h0: float, t_ms: np.ndarray, *, tau_k_ms: float) -> np.ndarray:
    """Continuous F1 solution ``H_K(t) = 1 + (H0-1) exp(-t/tau_K)``."""
    return 1.0 + (float(h0) - 1.0) * np.exp(-np.asarray(t_ms, dtype=np.float64) / float(tau_k_ms))


def h_k_f1_euler_trace(h0: float, n_steps: int, *, dt_ms: float, tau_k_ms: float) -> np.ndarray:
    """Independent Euler recurrence matching the frozen float32 discrete contract."""
    H = np.float32(h0)
    dt = np.float32(dt_ms)
    tau = np.float32(tau_k_ms)
    one = np.float32(1.0)
    out = np.empty((n_steps,), dtype=np.float32)
    for _ in range(n_steps):
        H = H + dt * (one - H) / tau
        out[_] = H
    return out.astype(np.float64)


def _run_dynamic(
    spec: dict[str, Any],
    d1: dict[str, Any],
    *,
    h_k0: float,
    seed: int,
) -> dict[str, Any]:
    sim = spec["simulation_policy"]
    params, edges = _build_isolated_circuit(d1)
    n_steps = int(sim["n_steps"])
    dt_ms = float(sim["dt_ms"])
    tau_k = float(sim["tau_k_ms"])
    sched = jnp.asarray(_drive_schedule({**d1, "simulation_policy": {**d1["simulation_policy"], **sim}}))
    n = int(d1["simulation_policy"]["n_neurons"])
    h_k = jnp.full((n,), float(h_k0), dtype=jnp.float32)
    key = jax.random.PRNGKey(int(seed))
    v, sp, src, st = simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery(
        params,
        edges,
        n_steps,
        dt_ms,
        key,
        h_k0=h_k,
        tau_k_ms=tau_k,
        dtype=str(sim["dtype"]),
        drive_schedule=sched,
        noise_scale=float(sim["noise_scale"]),
    )
    H_trace = np.asarray(st["H_K_trace"], dtype=np.float64)[..., 0]
    t_ms = (np.arange(n_steps, dtype=np.float64) + 1.0) * dt_ms
    return {
        "H_K0": float(h_k0),
        "seed": int(seed),
        "V_m": np.asarray(v, dtype=np.float64),
        "u": np.asarray(st["u_trace"], dtype=np.float64),
        "spikes": np.asarray(sp, dtype=np.float64),
        "H_K_trace": H_trace,
        "H_K_final": float(H_trace[-1]),
        "t_ms": t_ms,
        "euler_reference": h_k_f1_euler_trace(h_k0, n_steps, dt_ms=dt_ms, tau_k_ms=tau_k),
        "analytic_at_steps": h_k_f1_analytic(h_k0, t_ms, tau_k_ms=tau_k),
    }


def _run_classical(d1: dict[str, Any], sim: dict[str, Any], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    params, edges = _build_isolated_circuit(d1)
    merged = {**d1, "simulation_policy": {**d1["simulation_policy"], **sim}}
    sched = jnp.asarray(_drive_schedule(merged))
    key = jax.random.PRNGKey(int(seed))
    v, sp, _, _ = simulate_edge_recurrent_izhikevich(
        params,
        edges,
        int(sim["n_steps"]),
        float(sim["dt_ms"]),
        key,
        dtype=str(sim["dtype"]),
        drive_schedule=sched,
        noise_scale=float(sim["noise_scale"]),
    )
    return np.asarray(v, dtype=np.float64), np.asarray(sp, dtype=np.float64)


def run_d2a_autonomous_relaxation(
    spec: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    spec = spec or load_d2a_spec()
    validate_d2a_spec(spec)
    d1 = load_d1_spec()
    sim = spec["simulation_policy"]
    seeds = [int(s) for s in sim["seeds"]]
    h0_values = d2a_h_k0_values(spec)
    tau_k = float(sim["tau_k_ms"])
    dt_ms = float(sim["dt_ms"])
    h_tol = float(spec["convergence_gate"]["h_k_final_tol"])
    tail_frac = float(spec["convergence_gate"]["v_diff_tail_fraction"])

    cells: list[dict[str, Any]] = []
    gates: dict[str, Any] = {
        "G1_discrete_relaxation": {"passed": True, "details": []},
        "G2_analytic_consistency": {"passed": True, "details": []},
        "G3_baseline_invariance": {"passed": True, "details": []},
        "G4_positivity": {"passed": True, "details": []},
        "G5_deterministic_repeatability": {"passed": True, "details": []},
        "G6_classical_convergence": {"passed": True, "details": []},
    }

    for seed in seeds:
        v_class, sp_class = _run_classical(d1, sim, seed=seed)
        for h0 in h0_values:
            row = _run_dynamic(spec, d1, h_k0=h0, seed=seed)
            euler_err = float(np.max(np.abs(row["H_K_trace"] - row["euler_reference"])))
            analytic_err = float(np.max(np.abs(row["H_K_trace"] - row["analytic_at_steps"])))
            gates["G1_discrete_relaxation"]["details"].append(
                {"seed": seed, "H_K0": h0, "max_abs_euler_diff": euler_err}
            )
            if euler_err > 1e-6:
                gates["G1_discrete_relaxation"]["passed"] = False
            gates["G2_analytic_consistency"]["details"].append(
                {"seed": seed, "H_K0": h0, "max_abs_analytic_diff": analytic_err}
            )
            if analytic_err > 0.05:
                gates["G2_analytic_consistency"]["passed"] = False
            if np.any(row["H_K_trace"] <= 0):
                gates["G4_positivity"]["passed"] = False

            if h0 == 1.0:
                v_match = float(np.max(np.abs(row["V_m"] - v_class))) == 0.0
                sp_match = float(np.max(np.abs(row["spikes"] - sp_class))) == 0.0
                h_const = float(np.max(np.abs(row["H_K_trace"] - 1.0)))
                gates["G3_baseline_invariance"]["details"].append(
                    {
                        "seed": seed,
                        "V_m_bit_exact": v_match,
                        "spikes_bit_exact": sp_match,
                        "H_K_max_deviation": h_const,
                    }
                )
                if not (v_match and sp_match and h_const == 0.0):
                    gates["G3_baseline_invariance"]["passed"] = False
            else:
                v_diff = np.abs(row["V_m"] - v_class)
                peak = float(np.max(v_diff))
                tail_n = max(1, int(len(v_diff) * tail_frac))
                tail_mean = float(np.mean(v_diff[-tail_n:]))
                h_final_err = abs(row["H_K_final"] - 1.0)
                gates["G6_classical_convergence"]["details"].append(
                    {
                        "seed": seed,
                        "H_K0": h0,
                        "H_K_final": row["H_K_final"],
                        "H_K_final_err": h_final_err,
                        "v_diff_peak": peak,
                        "v_diff_tail_mean": tail_mean,
                    }
                )
                if h_final_err > h_tol:
                    gates["G6_classical_convergence"]["passed"] = False
                if tail_mean >= peak:
                    gates["G6_classical_convergence"]["passed"] = False

            repeat = _run_dynamic(spec, d1, h_k0=h0, seed=seed)
            rep_err = float(np.max(np.abs(repeat["V_m"] - row["V_m"])))
            gates["G5_deterministic_repeatability"]["details"].append(
                {"seed": seed, "H_K0": h0, "max_abs_repeat_diff": rep_err}
            )
            if rep_err != 0.0:
                gates["G5_deterministic_repeatability"]["passed"] = False

            cells.append(
                json_safe(
                    {
                        "seed": seed,
                        "H_K0": h0,
                        "H_K_final": row["H_K_final"],
                        "max_euler_diff": euler_err,
                        "max_analytic_diff": analytic_err,
                        "n_spikes": int(np.sum(row["spikes"] > 0.5)),
                        "H_K_monotone_to_one": bool(
                            (h0 < 1.0 and row["H_K_final"] > h0)
                            or (h0 > 1.0 and row["H_K_final"] < h0)
                            or h0 == 1.0
                        ),
                    }
                )
            )

    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_d_biological_rbs.d2a_execution_receipt.v1",
        "checkpoint": "D2a",
        "status": "FROZEN",
        "write_once": True,
        "package_head": package_head or _git_head(),
        "spec_path": "artifacts/protocol_d_biological_rbs/d2a_autonomous_h_k_relaxation_spec.json",
        "tau_k_ms": tau_k,
        "dynamics": "tau_K * dH_K/dt = 1 - H_K; kappa_K = 0",
        "coupling": "b_eff = H_K(t) * b (D1 map)",
        "n_cells": len(cells),
        "cells": cells,
        "gates": gates,
        "d2b_status": "not_implemented",
        "interpretation_deferred_to": "D2b",
    }
    for gname, gval in gates.items():
        if not gval["passed"]:
            raise RuntimeError(f"D2a gate failed: {gname}")
    return json_safe(receipt)


def write_d2a_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_d2a_autonomous_relaxation(package_head=package_head)
    D2A_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_d2a_execution_receipt() -> dict[str, Any]:
    return json.loads(D2A_EXECUTION_RECEIPT_PATH.read_text())
