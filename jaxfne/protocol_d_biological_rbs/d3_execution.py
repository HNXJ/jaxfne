"""D3 adaptation/recovery phenotype — schedule, metrics, execution, classification."""

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
    simulate_edge_recurrent_izhikevich_activity_h_k_rbd,
    simulate_edge_recurrent_izhikevich_static_h_k_recovery,
)
from jaxfne.io import json_safe
from jaxfne.protocol_d_biological_rbs.d1_execution import _build_isolated_circuit
from jaxfne.protocol_d_biological_rbs.d1_protocol import load_d1_spec
from jaxfne.protocol_d_biological_rbs.d2b_protocol import load_d2b_spec
from jaxfne.protocol_d_biological_rbs.d3_protocol import (
    D3_EXECUTION_RECEIPT_PATH,
    load_d3_spec,
    validate_d3_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _ms_to_step(t_ms: float, *, dt_ms: float) -> int:
    return int(round(float(t_ms) / float(dt_ms)))


def build_d3_pulse_onsets_ms(spec: dict[str, Any]) -> list[float]:
    pt = spec["pulse_train"]
    first = float(spec["timing"]["first_pulse_onset_ms"])
    m = int(pt["n_pulses_m"])
    isi = float(pt["isi_ms"])
    return [first + k * isi for k in range(m)]


def build_d3_rechallenge_onset_ms(spec: dict[str, Any], *, T_recovery_ms: float) -> float:
    return float(spec["timing"]["train_block_end_ms"]) + float(T_recovery_ms)


def build_d3_response_window_overlap_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    """Document frozen overlapping post-onset windows (80 ms window, 60 ms ISI)."""
    pt = spec["pulse_train"]
    window_ms = float(spec["response_metrics"]["primary"]["response_window_ms"])
    isi_ms = float(pt["isi_ms"])
    m = int(pt["n_pulses_m"])
    onsets = build_d3_pulse_onsets_ms(spec)
    overlaps: list[dict[str, Any]] = []
    for j in range(m - 1):
        overlap_ms = max(0.0, window_ms - (onsets[j + 1] - onsets[j]))
        overlaps.append(
            {
                "pulse_index_1_based": j + 1,
                "next_pulse_index_1_based": j + 2,
                "overlap_ms": float(overlap_ms),
                "note": (
                    f"R_{j + 1} window includes first {overlap_ms:g} ms after pulse {j + 2} onset"
                    if overlap_ms > 0
                    else "no overlap"
                ),
            }
        )
    return {
        "response_window_ms": window_ms,
        "isi_ms": isi_ms,
        "isi_definition": pt["isi_definition"],
        "train_pulse_overlaps": overlaps,
        "semantics": (
            "R_j counts spikes in [pulse_j_onset, pulse_j_onset + window); "
            "not an isolated single-pulse response when overlap_ms > 0"
        ),
    }


def build_d3_drive_schedule(
    spec: dict[str, Any],
    *,
    T_recovery_ms: float,
) -> tuple[np.ndarray, list[float], float]:
    """Return (schedule, train_onsets_ms, rechallenge_onset_ms)."""
    pt = spec["pulse_train"]
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    n_steps = int(sim["n_steps"])
    amp = float(pt["amplitude"])
    dur_steps = _ms_to_step(float(pt["duration_ms"]), dt_ms=dt_ms)

    sched = np.zeros((n_steps, 1), dtype=np.float32)
    train_onsets_ms = build_d3_pulse_onsets_ms(spec)
    for onset_ms in train_onsets_ms:
        o = _ms_to_step(onset_ms, dt_ms=dt_ms)
        e = min(n_steps, o + dur_steps)
        if o < n_steps:
            sched[o:e, 0] = amp

    rech_ms = build_d3_rechallenge_onset_ms(spec, T_recovery_ms=T_recovery_ms)
    o = _ms_to_step(rech_ms, dt_ms=dt_ms)
    e = min(n_steps, o + dur_steps)
    if o < n_steps:
        sched[o:e, 0] = amp
    return sched, train_onsets_ms, rech_ms


def _window_slice(onset_ms: float, *, window_ms: float, dt_ms: float, n_steps: int) -> slice:
    start = _ms_to_step(onset_ms, dt_ms=dt_ms)
    end = min(n_steps, start + _ms_to_step(window_ms, dt_ms=dt_ms))
    return slice(start, end)


def compute_response_metrics(
    voltages: np.ndarray,
    spikes: np.ndarray,
    *,
    onset_ms: float,
    window_ms: float,
    dt_ms: float,
) -> dict[str, Any]:
    n_steps = spikes.shape[0]
    sl = _window_slice(onset_ms, window_ms=window_ms, dt_ms=dt_ms, n_steps=n_steps)
    sp_win = spikes[sl, 0] > 0.5
    R = int(np.sum(sp_win))
    v_win = voltages[sl, 0]
    v_int = float(np.sum(v_win) * dt_ms) if v_win.size else 0.0
    spike_idx = np.where(sp_win)[0]
    if spike_idx.size == 0:
        t_first = None
    else:
        t_first = float((sl.start + int(spike_idx[0]) + 1) * dt_ms - onset_ms)
    return {
        "R": R,
        "V_int": v_int,
        "t_first_spike_ms": t_first,
        "window_start_ms": float(sl.start * dt_ms),
        "window_end_ms": float(sl.stop * dt_ms),
    }


def compute_adaptation_indices(
    R_train: list[int],
    R_rechallenge: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    early_idx = [i - 1 for i in spec["adaptation_index"]["R_early"]["pulse_indices_1_based"]]
    late_idx = [i - 1 for i in spec["adaptation_index"]["R_late"]["pulse_indices_1_based"]]
    R_early = float(np.mean([R_train[i] for i in early_idx]))
    R_late = float(np.mean([R_train[i] for i in late_idx]))
    A_adapt: float | None
    if R_early > 0:
        A_adapt = float(1.0 - R_late / R_early)
    else:
        A_adapt = None
    denom = R_early - R_late
    if abs(denom) < 1e-12:
        R_recovery = None
    else:
        R_recovery = float((float(R_rechallenge) - R_late) / denom)
    return {
        "R_early": R_early,
        "R_late": R_late,
        "A_adapt": A_adapt,
        "R_recovery": R_recovery,
        "R_rechallenge": int(R_rechallenge),
    }


def _mean_hidden_over_windows(
    trace: np.ndarray,
    onsets_ms: list[float],
    *,
    window_ms: float,
    dt_ms: float,
) -> float:
    vals: list[float] = []
    n_steps = trace.shape[0]
    for onset_ms in onsets_ms:
        sl = _window_slice(onset_ms, window_ms=window_ms, dt_ms=dt_ms, n_steps=n_steps)
        vals.append(float(np.mean(trace[sl])))
    return float(np.mean(vals)) if vals else 0.0


def compute_d_arm_mechanism_summaries(
    H_A_trace: np.ndarray,
    H_K_trace: np.ndarray,
    spec: dict[str, Any],
    *,
    train_onsets_ms: list[float],
    rechallenge_onset_ms: float,
    T_recovery_ms: float,
) -> dict[str, Any]:
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    window_ms = float(spec["response_metrics"]["primary"]["response_window_ms"])
    th = spec["frozen_thresholds"]
    baseline_ms = float(spec["timing"]["baseline_duration_ms"])
    n_steps = H_A_trace.shape[0]

    baseline_sl = slice(0, _ms_to_step(baseline_ms, dt_ms=dt_ms))
    H_A_baseline = float(np.mean(H_A_trace[baseline_sl]))
    H_K_baseline = float(np.mean(H_K_trace[baseline_sl]))

    early_onsets = [
        train_onsets_ms[i - 1]
        for i in spec["adaptation_index"]["R_early"]["pulse_indices_1_based"]
    ]
    late_onsets = [
        train_onsets_ms[i - 1]
        for i in spec["adaptation_index"]["R_late"]["pulse_indices_1_based"]
    ]
    H_A_early = _mean_hidden_over_windows(H_A_trace, early_onsets, window_ms=window_ms, dt_ms=dt_ms)
    H_K_early = _mean_hidden_over_windows(H_K_trace, early_onsets, window_ms=window_ms, dt_ms=dt_ms)
    H_A_late = _mean_hidden_over_windows(H_A_trace, late_onsets, window_ms=window_ms, dt_ms=dt_ms)
    H_K_late = _mean_hidden_over_windows(H_K_trace, late_onsets, window_ms=window_ms, dt_ms=dt_ms)

    rech_step = _ms_to_step(rechallenge_onset_ms, dt_ms=dt_ms)
    rech_step = min(max(rech_step, 0), n_steps - 1)
    H_A_at_rechallenge = float(H_A_trace[rech_step])
    H_K_at_rechallenge = float(H_K_trace[rech_step])

    rec_start = _ms_to_step(float(spec["timing"]["train_block_end_ms"]), dt_ms=dt_ms)
    rec_end = rech_step
    if rec_end > rec_start:
        H_A_recovery_mean = float(np.mean(H_A_trace[rec_start:rec_end]))
        H_K_recovery_mean = float(np.mean(H_K_trace[rec_start:rec_end]))
    else:
        H_A_recovery_mean = H_A_at_rechallenge
        H_K_recovery_mean = H_K_at_rechallenge

    M1 = H_A_late > H_A_baseline
    M2 = H_K_late > 1.0 + float(th["theta_H"])
    M3 = abs(H_A_at_rechallenge) <= float(th["recovery_state_tol_H_A"])
    M4 = abs(H_K_at_rechallenge - 1.0) <= float(th["recovery_state_tol_H_K_minus_1"])

    return {
        "H_A_baseline_mean": H_A_baseline,
        "H_K_baseline_mean": H_K_baseline,
        "H_A_early_mean": H_A_early,
        "H_K_early_mean": H_K_early,
        "H_A_late_mean": H_A_late,
        "H_K_late_mean": H_K_late,
        "H_A_recovery_interval_mean": H_A_recovery_mean,
        "H_K_recovery_interval_mean": H_K_recovery_mean,
        "H_A_at_rechallenge_onset": H_A_at_rechallenge,
        "H_K_at_rechallenge_onset": H_K_at_rechallenge,
        "abs_H_K_minus_1_at_rechallenge": float(abs(H_K_at_rechallenge - 1.0)),
        "T_recovery_ms": float(T_recovery_ms),
        "M1_pass": bool(M1),
        "M2_pass": bool(M2),
        "M3_pass": bool(M3),
        "M4_pass": bool(M4),
        "mechanism_ok": bool(M1 and M2),
    }


def classify_d3_cell(cell: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Prospective three-way classification from frozen thresholds."""
    th = spec["frozen_thresholds"]
    min_re = float(th["min_mean_R_early"])
    theta_a = float(th["theta_A"])
    arm = cell["null_arm"]
    R_early = float(cell["R_early"])
    A = cell.get("A_adapt")
    facilitation = A is not None and float(A) < 0.0

    if R_early < min_re or A is None:
        return {
            "classification": "UNRESOLVED",
            "facilitation": facilitation,
            "reason": "sparse_early_response",
        }

    if arm == "D":
        mech = cell.get("mechanism", {})
        mechanism_ok = bool(mech.get("mechanism_ok", False))
        if float(A) > theta_a and mechanism_ok:
            return {
                "classification": "ADAPTATION",
                "facilitation": False,
                "mechanism_ok": True,
            }
        return {
            "classification": "NO_ADAPTATION",
            "facilitation": facilitation,
            "mechanism_ok": mechanism_ok,
        }

    if float(A) > theta_a:
        return {"classification": "ADAPTATION", "facilitation": False}
    return {"classification": "NO_ADAPTATION", "facilitation": facilitation}


def _run_single_cell(
    spec: dict[str, Any],
    d1: dict[str, Any],
    d2b: dict[str, Any],
    *,
    seed: int,
    null_arm: str,
    recovery_level: dict[str, Any],
) -> dict[str, Any]:
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    n_steps = int(sim["n_steps"])
    window_ms = float(spec["response_metrics"]["primary"]["response_window_ms"])
    T_recovery = float(recovery_level["T_recovery_ms"])

    merged = {**d1, "simulation_policy": {**d1["simulation_policy"], **sim}}
    params, edges = _build_isolated_circuit(merged)
    sched_np, train_onsets_ms, rech_ms = build_d3_drive_schedule(spec, T_recovery_ms=T_recovery)
    sched = jnp.asarray(sched_np)
    key = jax.random.PRNGKey(int(seed))

    ts = d2b["dynamics"]["timescales"]
    tau_a = float(ts["tau_A_ms"])
    tau_k = float(ts["tau_K_ms"])
    kappa_full = float(d2b["dynamics"]["coupling_constant"]["value"])
    h_a0 = jnp.asarray([float(sim["initial_RBS"]["H_A0"])], dtype=jnp.float32)
    h_k0 = jnp.asarray([float(sim["initial_RBS"]["H_K0"])], dtype=jnp.float32)

    if null_arm == "N0":
        v, sp, _, _ = simulate_edge_recurrent_izhikevich(
            params, edges, n_steps, dt_ms, key,
            dtype=str(sim["dtype"]), drive_schedule=sched, noise_scale=float(sim["noise_scale"]),
        )
        st: dict[str, Any] = {}
    elif null_arm == "N1":
        v, sp, _, st = simulate_edge_recurrent_izhikevich_static_h_k_recovery(
            params, edges, n_steps, dt_ms, key,
            h_k=h_k0, dtype=str(sim["dtype"]), drive_schedule=sched,
            noise_scale=float(sim["noise_scale"]),
        )
    elif null_arm == "N2":
        v, sp, _, st = simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
            params, edges, n_steps, dt_ms, key,
            h_a0=h_a0, h_k0=h_k0, tau_a_ms=tau_a, tau_k_ms=tau_k, kappa_ak=0.0,
            dtype=str(sim["dtype"]), drive_schedule=sched, noise_scale=float(sim["noise_scale"]),
        )
    elif null_arm == "D":
        v, sp, _, st = simulate_edge_recurrent_izhikevich_activity_h_k_rbd(
            params, edges, n_steps, dt_ms, key,
            h_a0=h_a0, h_k0=h_k0, tau_a_ms=tau_a, tau_k_ms=tau_k, kappa_ak=kappa_full,
            dtype=str(sim["dtype"]), drive_schedule=sched, noise_scale=float(sim["noise_scale"]),
        )
    else:
        raise ValueError(f"unknown null_arm {null_arm!r}")

    v_np = np.asarray(v, dtype=np.float64)
    sp_np = np.asarray(sp, dtype=np.float64)

    R_train: list[int] = []
    train_metrics: list[dict[str, Any]] = []
    for onset_ms in train_onsets_ms:
        m = compute_response_metrics(v_np, sp_np, onset_ms=onset_ms, window_ms=window_ms, dt_ms=dt_ms)
        R_train.append(int(m["R"]))
        train_metrics.append(m)

    rech_m = compute_response_metrics(v_np, sp_np, onset_ms=rech_ms, window_ms=window_ms, dt_ms=dt_ms)
    indices = compute_adaptation_indices(R_train, int(rech_m["R"]), spec)

    cell: dict[str, Any] = {
        "seed": int(seed),
        "null_arm": null_arm,
        "recovery_interval_id": recovery_level["id"],
        "T_recovery_ms": T_recovery,
        "R_train": R_train,
        "train_pulse_onsets_ms": train_onsets_ms,
        "rechallenge_onset_ms": float(rech_ms),
        "train_metrics": train_metrics,
        "rechallenge_metrics": rech_m,
        **indices,
    }

    if null_arm == "D":
        H_A = np.asarray(st["H_A_trace"], dtype=np.float64)[:, 0]
        H_K = np.asarray(st["H_K_trace"], dtype=np.float64)[:, 0]
        cell["H_A_trace"] = H_A.tolist()
        cell["H_K_trace"] = H_K.tolist()
        cell["mechanism"] = compute_d_arm_mechanism_summaries(
            H_A, H_K, spec,
            train_onsets_ms=train_onsets_ms,
            rechallenge_onset_ms=rech_ms,
            T_recovery_ms=T_recovery,
        )

    cell.update(classify_d3_cell(cell, spec))
    return json_safe(cell)


def run_d3_adaptation_recovery(
    spec: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    spec = spec or load_d3_spec()
    validate_d3_spec(spec)
    d1 = load_d1_spec()
    d2b = load_d2b_spec()

    cells: list[dict[str, Any]] = []
    for seed in spec["simulation_policy"]["seeds"]:
        for arm in spec["null_hierarchy"]["arms"]:
            for level in spec["recovery_intervals"]["levels"]:
                cells.append(
                    _run_single_cell(
                        spec, d1, d2b,
                        seed=int(seed),
                        null_arm=str(arm["id"]),
                        recovery_level=level,
                    )
                )

    if len(cells) != int(spec["simulation_policy"]["cell_grid"]["n_cells"]):
        raise RuntimeError(f"expected {spec['simulation_policy']['cell_grid']['n_cells']} cells, got {len(cells)}")

    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_d_biological_rbs.d3_execution_receipt.v1",
        "checkpoint": "D3",
        "status": "FROZEN",
        "write_once": True,
        "implementation_authorized": True,
        "package_head": package_head or _git_head(),
        "spec_path": "artifacts/protocol_d_biological_rbs/d3_adaptation_recovery_phenotype_spec.json",
        "n_cells": len(cells),
        "response_window_semantics": build_d3_response_window_overlap_metadata(spec),
        "cells": cells,
    }
    return json_safe(receipt)


def write_d3_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_d3_adaptation_recovery(package_head=package_head)
    D3_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_d3_execution_receipt() -> dict[str, Any]:
    return json.loads(D3_EXECUTION_RECEIPT_PATH.read_text())
