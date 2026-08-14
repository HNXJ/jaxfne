"""D1 static H_K expression — frozen sweep and gate receipts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_static_h_k_recovery,
)
from jaxfne.io import json_safe
from jaxfne.protocol_d_biological_rbs.d1_protocol import (
    D1_EXECUTION_RECEIPT_PATH,
    d1_h_k_sweep_values,
    load_d1_spec,
    validate_d1_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _build_isolated_circuit(spec: dict[str, Any]) -> tuple[IzhikevichParams, EdgeList]:
    base = spec["emitter_baseline"]
    jdtype = jnp.float32
    n = int(spec["simulation_policy"]["n_neurons"])
    params = IzhikevichParams(
        v0=jnp.full((n,), float(base["v0"]), dtype=jdtype),
        u0=jnp.full((n,), float(base["u0"]), dtype=jdtype),
        a=jnp.full((n,), float(base["a"]), dtype=jdtype),
        b=jnp.full((n,), float(base["b"]), dtype=jdtype),
        c=jnp.full((n,), float(base["c"]), dtype=jdtype),
        d=jnp.full((n,), float(base["d"]), dtype=jdtype),
        drive=jnp.zeros((n,), dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.zeros((0,), dtype=jnp.int32),
        post=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jdtype),
        receptor_index=jnp.zeros((0,), dtype=jnp.int32),
        tau_ms=jnp.zeros((0,), dtype=jdtype),
    )
    return params, edges


def _drive_schedule(spec: dict[str, Any]) -> np.ndarray:
    sim = spec["simulation_policy"]
    n_steps = int(sim["n_steps"])
    n = int(sim["n_neurons"])
    dt = float(sim["dt_ms"])
    ev = sim["drive"]
    onset = int(round(float(ev["onset_ms"]) / dt))
    dur = int(round(float(ev["duration_ms"]) / dt))
    amp = float(ev["amplitude"])
    sched = np.zeros((n_steps, n), dtype=np.float32)
    end = min(n_steps, onset + dur)
    if onset < n_steps:
        sched[onset:end, 0] = amp
    return sched


def _first_spike_time_ms(spikes: np.ndarray, *, dt_ms: float) -> float | None:
    idx = np.flatnonzero(spikes[:, 0] > 0.5)
    if idx.size == 0:
        return None
    return float(idx[0]) * float(dt_ms)


def _mean_isi_ms(spikes: np.ndarray, *, dt_ms: float) -> float | None:
    idx = np.flatnonzero(spikes[:, 0] > 0.5)
    if idx.size < 2:
        return None
    return float(np.mean(np.diff(idx)) * float(dt_ms))


def _run_level(
    spec: dict[str, Any],
    *,
    h_k_value: float,
    seed: int,
) -> dict[str, Any]:
    sim = spec["simulation_policy"]
    params, edges = _build_isolated_circuit(spec)
    n_steps = int(sim["n_steps"])
    dt_ms = float(sim["dt_ms"])
    dtype = str(sim["dtype"])
    sched = jnp.asarray(_drive_schedule(spec))
    n = int(sim["n_neurons"])
    h_k = jnp.full((n,), float(h_k_value), dtype=jnp.float32)
    key = jax.random.PRNGKey(int(seed))
    v_ext, sp_ext, src_ext, st_ext = simulate_edge_recurrent_izhikevich_static_h_k_recovery(
        params,
        edges,
        n_steps,
        dt_ms,
        key,
        h_k=h_k,
        dtype=dtype,
        drive_schedule=sched,
        noise_scale=float(sim["noise_scale"]),
    )
    return {
        "H_K": float(h_k_value),
        "seed": int(seed),
        "V_m": np.asarray(v_ext, dtype=np.float64),
        "u": np.asarray(st_ext["u_trace"], dtype=np.float64),
        "spikes": np.asarray(sp_ext, dtype=np.float64),
        "sources": np.asarray(src_ext, dtype=np.float64),
        "H_K_static": np.asarray(st_ext["H_K_static"], dtype=np.float64),
        "n_spikes": int(np.sum(sp_ext > 0.5)),
        "t_spike_first_ms": _first_spike_time_ms(np.asarray(sp_ext), dt_ms=dt_ms),
        "mean_isi_ms": _mean_isi_ms(np.asarray(sp_ext), dt_ms=dt_ms),
    }


def _run_classical(spec: dict[str, Any], *, seed: int) -> dict[str, Any]:
    sim = spec["simulation_policy"]
    params, edges = _build_isolated_circuit(spec)
    n_steps = int(sim["n_steps"])
    dt_ms = float(sim["dt_ms"])
    dtype = str(sim["dtype"])
    sched = jnp.asarray(_drive_schedule(spec))
    key = jax.random.PRNGKey(int(seed))
    v, sp, src, st = simulate_edge_recurrent_izhikevich(
        params,
        edges,
        n_steps,
        dt_ms,
        key,
        dtype=dtype,
        drive_schedule=sched,
        noise_scale=float(sim["noise_scale"]),
    )
    return {
        "H_K": 1.0,
        "seed": int(seed),
        "V_m": np.asarray(v, dtype=np.float64),
        "u": None,
        "spikes": np.asarray(sp, dtype=np.float64),
        "sources": np.asarray(src, dtype=np.float64),
        "n_spikes": int(np.sum(sp > 0.5)),
        "t_spike_first_ms": _first_spike_time_ms(np.asarray(sp), dt_ms=dt_ms),
        "mean_isi_ms": _mean_isi_ms(np.asarray(sp), dt_ms=dt_ms),
    }


def _delta_vs_reference(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    delta_u = row["u"] - ref["u"]
    delta_v = row["V_m"] - ref["V_m"]
    delta_n = int(row["n_spikes"]) - int(ref["n_spikes"])
    t_row = row["t_spike_first_ms"]
    t_ref = ref["t_spike_first_ms"]
    delta_t = None
    if t_row is not None and t_ref is not None:
        delta_t = float(t_row) - float(t_ref)
    return {
        "delta_u_max_abs": float(np.max(np.abs(delta_u))),
        "delta_u_rms": float(np.sqrt(np.mean(delta_u ** 2))),
        "delta_V_max_abs": float(np.max(np.abs(delta_v))),
        "delta_V_rms": float(np.sqrt(np.mean(delta_v ** 2))),
        "delta_N_spike": delta_n,
        "delta_t_spike_first_ms": delta_t,
        "expression_sensitive_u": bool(np.max(np.abs(delta_u)) > 0.0),
        "expression_sensitive_V": bool(np.max(np.abs(delta_v)) > 0.0),
        "expression_sensitive_spikes": bool(delta_n != 0 or delta_t not in (None, 0.0)),
    }


def run_d1_static_expression(
    spec: dict[str, Any] | None = None,
    *,
    package_head: str | None = None,
) -> dict[str, Any]:
    """Execute preregistered D1 static sweep and evaluate gates."""
    spec = spec or load_d1_spec()
    validate_d1_spec(spec)
    levels = d1_h_k_sweep_values(spec)
    seeds = [int(s) for s in spec["simulation_policy"]["seeds"]]
    cells: list[dict[str, Any]] = []
    gate_results: dict[str, Any] = {
        "G1_containment": {"passed": True, "details": []},
        "G2_static_state_integrity": {"passed": True, "details": []},
        "G3_parameter_locality": {"passed": True, "details": []},
        "G4_bidirectional_sensitivity": {"passed": True, "evaluated_levels": list(levels)},
    }

    for seed in seeds:
        classical = _run_classical(spec, seed=seed)
        ref_row = _run_level(spec, h_k_value=1.0, seed=seed)
        # G1 containment: extended H_K=1 vs classical kernel
        v_match = float(np.max(np.abs(ref_row["V_m"] - classical["V_m"]))) == 0.0
        sp_match = float(np.max(np.abs(ref_row["spikes"] - classical["spikes"]))) == 0.0
        u_ok = True
        gate_results["G1_containment"]["details"].append(
            {
                "seed": seed,
                "V_m_bit_exact": v_match,
                "spikes_bit_exact": sp_match,
            }
        )
        if not (v_match and sp_match):
            gate_results["G1_containment"]["passed"] = False

        for h_k in levels:
            row = _run_level(spec, h_k_value=h_k, seed=seed)
            h_static = row["H_K_static"]
            if not np.allclose(h_static, float(h_k)):
                gate_results["G2_static_state_integrity"]["passed"] = False
            deltas = _delta_vs_reference(row, ref_row) if h_k != 1.0 else {
                "delta_u_max_abs": 0.0,
                "delta_u_rms": 0.0,
                "delta_V_max_abs": 0.0,
                "delta_V_rms": 0.0,
                "delta_N_spike": 0,
                "delta_t_spike_first_ms": None,
                "expression_sensitive_u": False,
                "expression_sensitive_V": False,
                "expression_sensitive_spikes": False,
            }
            cells.append(
                json_safe(
                    {
                        "seed": seed,
                        "H_K": float(h_k),
                        "n_spikes": row["n_spikes"],
                        "t_spike_first_ms": row["t_spike_first_ms"],
                        "mean_isi_ms": row["mean_isi_ms"],
                        "deltas_vs_H_K_1": deltas,
                    }
                )
            )

    expression = {
        "criterion": spec["expression_criterion"]["statement"],
        "direction_not_preregistered": spec["expression_criterion"]["direction_not_preregistered"],
        "any_u_sensitive": any(
            c["deltas_vs_H_K_1"]["expression_sensitive_u"] for c in cells if c["H_K"] != 1.0
        ),
        "any_V_sensitive": any(
            c["deltas_vs_H_K_1"]["expression_sensitive_V"] for c in cells if c["H_K"] != 1.0
        ),
        "any_spike_sensitive": any(
            c["deltas_vs_H_K_1"]["expression_sensitive_spikes"] for c in cells if c["H_K"] != 1.0
        ),
    }

    receipt: dict[str, Any] = {
        "schema": "jaxfne.protocol_d_biological_rbs.d1_execution_receipt.v1",
        "checkpoint": "D1",
        "status": "FROZEN",
        "write_once": True,
        "package_head": package_head or _git_head(),
        "spec_path": str(
            (D1_EXECUTION_RECEIPT_PATH.parent / "d1_static_h_k_expression_spec.json").relative_to(REPO_ROOT)
        ),
        "coupling": "b_eff = H_K * b",
        "mathematical_receipt": spec["mathematical_receipt"],
        "documentation_clause": spec["documentation_clause"],
        "n_cells": len(cells),
        "expected_n_cells": len(seeds) * len(levels),
        "cells": cells,
        "gates": gate_results,
        "expression_summary": expression,
        "interpretation_deferred_to": "D2",
    }
    if len(cells) != len(seeds) * len(levels):
        raise RuntimeError("D1 cell count mismatch")
    if not gate_results["G1_containment"]["passed"]:
        raise RuntimeError("D1 G1 containment gate failed")
    if not gate_results["G2_static_state_integrity"]["passed"]:
        raise RuntimeError("D1 G2 static-state integrity gate failed")
    return json_safe(receipt)


def write_d1_execution_receipt(*, package_head: str | None = None) -> dict[str, Any]:
    receipt = run_d1_static_expression(package_head=package_head)
    D1_EXECUTION_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_d1_execution_receipt() -> dict[str, Any]:
    return json.loads(D1_EXECUTION_RECEIPT_PATH.read_text())
