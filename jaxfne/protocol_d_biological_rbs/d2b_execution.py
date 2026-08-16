"""D2b two-coordinate activity-coupled RBS — execution and gate receipts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import (
    _advance_h_a_trace,
    _advance_h_k_activity_coupled,
    simulate_edge_recurrent_izhikevich_activity_h_k_rbd,
    simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery,
)
from jaxfne.io import json_safe
from jaxfne.protocol_d_biological_rbs.d1_execution import _build_isolated_circuit, _drive_schedule
from jaxfne.protocol_d_biological_rbs.d1_protocol import load_d1_spec
from jaxfne.protocol_d_biological_rbs.d2a_protocol import load_d2a_spec
from jaxfne.protocol_d_biological_rbs.d2b_protocol import (
    D2B_EXECUTION_RECEIPT_PATH,
    load_d2b_spec,
    validate_d2b_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def h_a_analytic(a0: float, t_ms: np.ndarray, *, tau_a_ms: float) -> np.ndarray:
    """Continuous solution ``H_A(t) = H_A(0) exp(-t/tau_A)`` with ``S=0``."""
    return float(a0) * np.exp(-np.asarray(t_ms, dtype=np.float64) / float(tau_a_ms))


def h_k_deviation_analytic_two_timescale(
    h0: float,
    a0: float,
    t_ms: np.ndarray,
    *,
    tau_a_ms: float,
    tau_k_ms: float,
    kappa_ak: float,
) -> np.ndarray:
    """``h_K = H_K - 1`` closed form for ``tau_A != tau_K`` post-stimulus contract."""
    t = np.asarray(t_ms, dtype=np.float64)
    tau_a = float(tau_a_ms)
    tau_k = float(tau_k_ms)
    kappa = float(kappa_ak)
    a0f = float(a0)
    h0f = float(h0)
    exp_a = np.exp(-t / tau_a)
    exp_k = np.exp(-t / tau_k)
    coeff = kappa * a0f * tau_a / (tau_a - tau_k)
    return h0f * exp_k + coeff * (exp_a - exp_k)


def h_k_analytic_two_timescale(
    h0: float,
    a0: float,
    t_ms: np.ndarray,
    *,
    tau_a_ms: float,
    tau_k_ms: float,
    kappa_ak: float,
) -> np.ndarray:
    return 1.0 + h_k_deviation_analytic_two_timescale(
        h0, a0, t_ms, tau_a_ms=tau_a_ms, tau_k_ms=tau_k_ms, kappa_ak=kappa_ak
    )


def rbs_post_stimulus_euler_traces(
    h_a0: float,
    h_k0: float,
    n_steps: int,
    *,
    dt_ms: float,
    tau_a_ms: float,
    tau_k_ms: float,
    kappa_ak: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent float32 Euler recurrence for ``S=0`` (implementation authority)."""
    H_A = np.float32(h_a0)
    H_K = np.float32(h_k0)
    dt = np.float32(dt_ms)
    tau_a = np.float32(tau_a_ms)
    tau_k = np.float32(tau_k_ms)
    kappa = np.float32(kappa_ak)
    zero = np.float32(0.0)
    one = np.float32(1.0)
    ha_out = np.empty((n_steps,), dtype=np.float32)
    hk_out = np.empty((n_steps,), dtype=np.float32)
    for i in range(n_steps):
        H_A_old = H_A
        H_A = H_A + dt * (-H_A + zero) / tau_a
        H_K = H_K + dt * ((one - H_K) + kappa * H_A_old) / tau_k
        ha_out[i] = H_A
        hk_out[i] = H_K
    return ha_out.astype(np.float64), hk_out.astype(np.float64)


def _sim_policy(d2a: dict[str, Any]) -> dict[str, Any]:
    sim = d2a["simulation_policy"]
    d2b = load_d2b_spec()
    ts = d2b["dynamics"]["timescales"]
    return {
        "duration_ms": float(sim["duration_ms"]),
        "dt_ms": float(sim["dt_ms"]),
        "n_steps": int(sim["n_steps"]),
        "tau_k_ms": float(ts["tau_K_ms"]),
        "tau_a_ms": float(ts["tau_A_ms"]),
        "kappa_ak": float(d2b["dynamics"]["coupling_constant"]["value"]),
        "noise_scale": float(sim["noise_scale"]),
        "dtype": str(sim["dtype"]),
        "seeds": [int(s) for s in sim["seeds"]],
    }


def _run_d2b(
    d1: dict[str, Any],
    sim: dict[str, Any],
    *,
    seed: int,
    kappa_ak: float | None = None,
    n_neurons: int | None = None,
) -> dict[str, Any]:
    merged_d1 = d1 if n_neurons is None else {**d1, "simulation_policy": {**d1["simulation_policy"], "n_neurons": n_neurons}}
    params, edges = _build_isolated_circuit(merged_d1)
    n = int(merged_d1["simulation_policy"]["n_neurons"])
    n_steps = int(sim["n_steps"])
    dt_ms = float(sim["dt_ms"])
    tau_a = float(sim["tau_a_ms"])
    tau_k = float(sim["tau_k_ms"])
    kappa = float(sim["kappa_ak"] if kappa_ak is None else kappa_ak)
    sched = jnp.asarray(
        _drive_schedule({**merged_d1, "simulation_policy": {**merged_d1["simulation_policy"], **sim}})
    )
    h_a = jnp.zeros((n,), dtype=jnp.float32)
    h_k = jnp.ones((n,), dtype=jnp.float32)
    key = jax.random.PRNGKey(int(seed))
    w0 = np.asarray(params.W, dtype=np.float64)
    v, sp, src, st = simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
        params,
        edges,
        n_steps,
        dt_ms,
        key,
        h_a0=h_a,
        h_k0=h_k,
        tau_a_ms=tau_a,
        tau_k_ms=tau_k,
        kappa_ak=kappa,
        dtype=str(sim["dtype"]),
        drive_schedule=sched,
        noise_scale=float(sim["noise_scale"]),
    )
    H_A = np.asarray(st["H_A_trace"], dtype=np.float64)
    H_K = np.asarray(st["H_K_trace"], dtype=np.float64)
    S = np.asarray(st["S_trace"], dtype=np.float64)
    t_ms = (np.arange(n_steps, dtype=np.float64) + 1.0) * dt_ms
    return {
        "seed": int(seed),
        "kappa_ak": kappa,
        "V_m": np.asarray(v, dtype=np.float64),
        "u": np.asarray(st["u_trace"], dtype=np.float64),
        "spikes": np.asarray(sp, dtype=np.float64),
        "H_A_trace": H_A,
        "H_K_trace": H_K,
        "H_trace": np.asarray(st["H_trace"], dtype=np.float64),
        "S_trace": S,
        "t_ms": t_ms,
        "w_initial": np.asarray(st["w_initial"], dtype=np.float64),
        "w_final": np.asarray(st["w_final"], dtype=np.float64),
        "W_initial": w0,
        "W_final": np.asarray(params.W, dtype=np.float64),
        "H_A_final": float(H_A[-1, 0]),
        "H_K_final": float(H_K[-1, 0]),
    }


def _drive_off_index(d1: dict[str, Any], sim: dict[str, Any]) -> int:
    ev = d1["simulation_policy"]["drive"]
    dt = float(sim["dt_ms"])
    onset = int(round(float(ev["onset_ms"]) / dt))
    dur = int(round(float(ev["duration_ms"]) / dt))
    return min(int(sim["n_steps"]), onset + dur)


def run_d2b_activity_h_k_coupling(
    spec: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    """Execute frozen D2b gates; raise RuntimeError on any failure."""
    spec = spec or load_d2b_spec()
    validate_d2b_spec(spec)
    d1 = load_d1_spec()
    d2a = load_d2a_spec()
    sim = _sim_policy(d2a)
    seeds = sim["seeds"]
    tau_a = float(sim["tau_a_ms"])
    tau_k = float(sim["tau_k_ms"])
    kappa = float(sim["kappa_ak"])
    dt_ms = float(sim["dt_ms"])
    drive_end = _drive_off_index(d1, sim)
    recovery_tol = float(d2a["convergence_gate"]["h_k_final_tol"])

    gates: dict[str, Any] = {
        "G1_activity_writing": {"passed": True, "details": []},
        "G2_causal_transfer": {"passed": True, "details": []},
        "G3_reference_recovery": {"passed": True, "details": []},
        "G4_d2a_reduction": {"passed": True, "details": []},
        "G5_admissibility": {"passed": True, "details": []},
        "G6_analytic_discrete_recovery": {"passed": True, "details": []},
        "G7_no_plasticity": {"passed": True, "details": []},
        "D_node_local_diagnostic": {"passed": True, "details": []},
    }
    cells: list[dict[str, Any]] = []

    for seed in seeds:
        row = _run_d2b(d1, sim, seed=seed)
        S0 = row["S_trace"][:, 0]
        HA0 = row["H_A_trace"][:, 0]
        HK0 = row["H_K_trace"][:, 0]

        spike_idx = np.where(S0 > 0.5)[0]
        if spike_idx.size == 0:
            gates["G1_activity_writing"]["passed"] = False
            gates["G1_activity_writing"]["details"].append({"seed": seed, "error": "no spikes"})
        else:
            ha_after_spike = HA0[spike_idx]
            g1_ok = bool(np.all(ha_after_spike > 0))
            gates["G1_activity_writing"]["details"].append(
                {"seed": seed, "n_spike_steps": int(spike_idx.size), "min_H_A_on_spike": float(np.min(ha_after_spike))}
            )
            if not g1_ok:
                gates["G1_activity_writing"]["passed"] = False

        ha_pos_idx = np.where(HA0[:-1] > 1e-8)[0]
        if ha_pos_idx.size > 0:
            hk_after_ha = HK0[ha_pos_idx + 1]
            g2_ok = bool(np.all(hk_after_ha > 1.0))
            gates["G2_causal_transfer"]["details"].append(
                {
                    "seed": seed,
                    "min_H_K_when_prior_H_A_positive": float(np.min(hk_after_ha)),
                    "first_prior_H_A_positive_step": int(ha_pos_idx[0]),
                    "ordering": "H_K^{n+1} uses H_A^n (lagged check)",
                }
            )
            if not g2_ok:
                gates["G2_causal_transfer"]["passed"] = False

        ha_tail_err = float(np.max(np.abs(HA0[-500:])))
        hk_tail_err = float(np.max(np.abs(HK0[-500:] - 1.0)))
        gates["G3_reference_recovery"]["details"].append(
            {
                "seed": seed,
                "H_A_final": row["H_A_final"],
                "H_K_final": row["H_K_final"],
                "tail_max_abs_H_A": ha_tail_err,
                "tail_max_abs_H_K_minus_1": hk_tail_err,
            }
        )
        if ha_tail_err > recovery_tol or hk_tail_err > recovery_tol:
            gates["G3_reference_recovery"]["passed"] = False

        d2b_null = _run_d2b(d1, sim, seed=seed, kappa_ak=0.0)
        d2a_row = _run_d2a_match(d1, sim, seed=seed)
        hk_diff = float(np.max(np.abs(d2b_null["H_K_trace"][:, 0] - d2a_row["H_K_trace"][:, 0])))
        gates["G4_d2a_reduction"]["details"].append({"seed": seed, "max_abs_H_K_diff": hk_diff})
        if hk_diff > 1e-6:
            gates["G4_d2a_reduction"]["passed"] = False

        min_hk = float(np.min(HK0))
        gates["G5_admissibility"]["details"].append({"seed": seed, "min_H_K": min_hk})
        if min_hk <= 0:
            gates["G5_admissibility"]["passed"] = False

        ha0_post = float(HA0[drive_end - 1]) if drive_end > 0 else float(HA0[0])
        hk0_post = float(HK0[drive_end - 1]) if drive_end > 0 else float(HK0[0])
        n_post = int(sim["n_steps"]) - drive_end
        if n_post > 0:
            t_post = (np.arange(n_post, dtype=np.float64) + 1.0) * dt_ms
            ha_sim = HA0[drive_end:]
            hk_sim = HK0[drive_end:]
            ha_ref, hk_ref = rbs_post_stimulus_euler_traces(
                ha0_post,
                hk0_post,
                n_post,
                dt_ms=dt_ms,
                tau_a_ms=tau_a,
                tau_k_ms=tau_k,
                kappa_ak=kappa,
            )
            ha_an = h_a_analytic(ha0_post, t_post, tau_a_ms=tau_a)
            hk_an = h_k_analytic_two_timescale(
                hk0_post - 1.0,
                ha0_post,
                t_post,
                tau_a_ms=tau_a,
                tau_k_ms=tau_k,
                kappa_ak=kappa,
            )
            euler_ha_err = float(np.max(np.abs(ha_sim - ha_ref)))
            euler_hk_err = float(np.max(np.abs(hk_sim - hk_ref)))
            analytic_hk_err = float(np.max(np.abs(hk_sim - hk_an)))
            gates["G6_analytic_discrete_recovery"]["details"].append(
                {
                    "seed": seed,
                    "post_stimulus_origin_step": drive_end,
                    "H_A0_post": ha0_post,
                    "H_K0_post": hk0_post,
                    "max_abs_euler_H_A_diff": euler_ha_err,
                    "max_abs_euler_H_K_diff": euler_hk_err,
                    "max_abs_analytic_H_K_diff": analytic_hk_err,
                }
            )
            if euler_ha_err > 1e-6 or euler_hk_err > 1e-6:
                gates["G6_analytic_discrete_recovery"]["passed"] = False
            if analytic_hk_err > 0.05:
                gates["G6_analytic_discrete_recovery"]["passed"] = False

        edge_w = row["w_final"]
        edge_w0 = row["w_initial"]
        edge_err = 0.0 if edge_w.size == 0 else float(np.max(np.abs(edge_w - edge_w0)))
        w_err = float(np.max(np.abs(row["W_final"] - row["W_initial"])))
        total_err = max(edge_err, w_err)
        gates["G7_no_plasticity"]["details"].append(
            {"seed": seed, "max_abs_W_matrix_diff": w_err, "max_abs_edge_weight_diff": edge_err}
        )
        if total_err != 0.0:
            gates["G7_no_plasticity"]["passed"] = False

        local = _run_d2b(d1, sim, seed=seed, n_neurons=2)
        ha_n0 = local["H_A_trace"][:, 0]
        ha_n1 = local["H_A_trace"][:, 1]
        s_n1 = local["S_trace"][:, 1]
        n1_spike_steps = int(np.sum(s_n1 > 0.5))
        n1_ha_max = float(np.max(ha_n1))
        n0_ha_max = float(np.max(ha_n0))
        local_ok = n1_ha_max == 0.0 or n1_spike_steps > 0
        gates["D_node_local_diagnostic"]["details"].append(
            {
                "seed": seed,
                "neuron0_max_H_A": n0_ha_max,
                "neuron1_max_H_A": n1_ha_max,
                "neuron1_spike_steps": n1_spike_steps,
                "node_local_ok": local_ok,
            }
        )
        if not local_ok:
            gates["D_node_local_diagnostic"]["passed"] = False

        cells.append(
            json_safe(
                {
                    "seed": seed,
                    "n_spikes": int(np.sum(row["spikes"] > 0.5)),
                    "H_A_final": row["H_A_final"],
                    "H_K_final": row["H_K_final"],
                    "min_H_K": min_hk,
                }
            )
        )

    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_d_biological_rbs.d2b_execution_receipt.v1",
        "checkpoint": "D2b",
        "status": "FROZEN",
        "write_once": True,
        "implementation_authorized": True,
        "package_head": package_head or _git_head(),
        "spec_path": "artifacts/protocol_d_biological_rbs/d2b_activity_h_k_coupling_spec.json",
        "activity_input_S": "binary spike indicator S_n in {0,1} per Euler step (timestep-independent)",
        "causal_ordering": "H_K^{n+1} uses H_A^n not H_A^{n+1}",
        "dynamics": "tau_A dH_A/dt = -H_A + S; tau_K dH_K/dt = (1-H_K) + kappa_AK H_A",
        "coupling": "b_eff = H_K(t) * b (D1 map)",
        "tau_A_ms": tau_a,
        "tau_K_ms": tau_k,
        "kappa_AK": kappa,
        "n_cells": len(cells),
        "cells": cells,
        "gates": gates,
        "d3_status": "not_authorized",
        "interpretation_deferred_to": "D3",
        "scientific_contract": "activity -> H_A -> H_K -> b_eff -> X (state-space only; no adaptation phenotype claim)",
    }
    for gname, gval in gates.items():
        if not gval["passed"]:
            raise RuntimeError(f"D2b gate failed: {gname}")
    return json_safe(receipt)


def _run_d2a_match(d1: dict[str, Any], sim: dict[str, Any], *, seed: int) -> dict[str, Any]:
    params, edges = _build_isolated_circuit(d1)
    n_steps = int(sim["n_steps"])
    dt_ms = float(sim["dt_ms"])
    sched = jnp.asarray(_drive_schedule({**d1, "simulation_policy": {**d1["simulation_policy"], **sim}}))
    n = int(d1["simulation_policy"]["n_neurons"])
    h_k = jnp.ones((n,), dtype=jnp.float32)
    key = jax.random.PRNGKey(int(seed))
    _, _, _, st = simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery(
        params,
        edges,
        n_steps,
        dt_ms,
        key,
        h_k0=h_k,
        tau_k_ms=float(sim["tau_k_ms"]),
        dtype=str(sim["dtype"]),
        drive_schedule=sched,
        noise_scale=float(sim["noise_scale"]),
    )
    return {"H_K_trace": np.asarray(st["H_K_trace"], dtype=np.float64)}


def write_d2b_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_d2b_activity_h_k_coupling(package_head=package_head)
    D2B_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_d2b_execution_receipt() -> dict[str, Any]:
    return json.loads(D2B_EXECUTION_RECEIPT_PATH.read_text())
