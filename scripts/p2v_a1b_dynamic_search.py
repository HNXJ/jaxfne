"""A-1b dynamic positive-control search - frozen-lattice execution.

Executes the FROZEN spec (artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_spec.json):
15 (v_c, K) points x 3 seeds = 45 cells, using the embedded delay vectors
verbatim. No adaptive extension. Writes the FROZEN receipt once; refuses to
overwrite an existing receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import jaxfne as jtfne
from jaxfne.h4_matrix import build_ring_params_edges_km
from jaxfne.protocol_c.c3_execution import (
    _build_c3_model,
    _ordered_arc_positions_mm,
    _ring_positions_3d,
    _stimulus_schedule,
    replay_c3_cell,
)
from jaxfne.protocol_c.c3_protocol import load_c3_spec
from jaxfne.protocol_c.estimator import estimate_traveling_wave
from jaxfne.protocol_c.protocol import load_protocol_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
A1B_DIR = REPO_ROOT / "artifacts" / "protocol_c" / "p2v_a1b_dynamic_search"
SPEC_PATH = A1B_DIR / "p2v_a1b_spec.json"
RECEIPT_PATH = A1B_DIR / "p2v_a1b_receipt.json"

ANCHOR_VC = 0.131
ANCHOR_K = 1


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def estimator_module_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD:jaxfne/protocol_c/estimator.py"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def cell_is_invalid(vm: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(vm).all():
        return True, "non-finite_V_m"
    if float(np.abs(vm).max()) > 150.0:
        return True, "vm_blowup_gt_150mv"
    return False, ""


def simulate_cell(
    c3_spec: dict,
    vc: float,
    k: int,
    delays: list[int],
    seed: int,
) -> dict:
    n = int(c3_spec["frozen_topology_and_weights"]["n_neurons"])
    sim_pol = c3_spec["simulation_policy"]
    dt_ms = float(sim_pol["dt_ms"])
    params, edges = build_ring_params_edges_km(
        n,
        k,
        arc_delay_steps=np.asarray(delays, dtype=np.int32),
        weight=float(c3_spec["frozen_topology_and_weights"]["edge_weight"]),
        tau_ms=float(c3_spec["frozen_topology_and_weights"]["syn_tau_ms"]),
    )
    positions_3d = _ring_positions_3d(n, radius_mm=float(c3_spec["frozen_spatial_coordinates"]["ring_radius_mm"]))
    model = _build_c3_model(c3_spec, edges=edges, emitter=params, positions_3d=positions_3d, seed=seed)
    runtime = jtfne.RuntimeConfig(
        dtype=str(sim_pol["dtype"]),
        recurrent_backend="edge_list",
        enable_hdp=False,
        enable_homeostasis=False,
        hdp_params={"noise_scale": float(sim_pol["noise_scale"])},
    )
    sim = jtfne.Simulation(
        duration_ms=float(sim_pol["duration_ms"]),
        dt_ms=dt_ms,
        seed=int(seed),
        runtime=runtime,
        record_sources=True,
        record_fields=bool(sim_pol["record_fields"]),
    )
    signals = model.simulate(sim, paradigm=_stimulus_schedule(c3_spec))
    vm = np.asarray(signals.V_m, dtype=np.float64)
    positions = _ordered_arc_positions_mm(n, radius_mm=float(c3_spec["frozen_spatial_coordinates"]["ring_radius_mm"]))
    est = estimate_traveling_wave(vm, positions, dt_ms=dt_ms, spec=load_protocol_spec())
    invalid, reason = cell_is_invalid(vm)
    above = vm > -20.0
    spike_counts = np.sum(above[1:] & ~above[:-1], axis=0)
    duration_s = float(sim_pol["duration_ms"]) / 1000.0
    row = {
        "point_id": f"vc{vc:.3f}_k{k}",
        "velocity_mm_per_ms": vc,
        "k_neighbors": k,
        "delay_steps_per_skip": [int(d) for d in delays],
        "seed": int(seed),
        "invalid": invalid,
        "invalid_reason": reason,
        "classification": est.classification,
        "quality_reasons": est.quality_reasons,
        "estimator": est.to_dict(),
        "vm_max_abs": float(np.abs(vm).max()),
        "vm_sha256": hashlib.sha256(np.ascontiguousarray(vm).tobytes()).hexdigest(),
        "activity_summary": {
            "n_neurons_with_spikes": int(np.count_nonzero(spike_counts)),
            "mean_spike_rate_hz": float(spike_counts.mean() / duration_s),
            "max_spike_rate_hz": float(spike_counts.max() / duration_s),
        },
    }
    return row


def anchor_identity_check(c3_spec: dict, signatures: dict[str, list[int]]) -> dict:
    rows = []
    ok = True
    for seed in (1001, 1002, 1003):
        max_diff = _anchor_trace_maxdiff(c3_spec, signatures, seed)
        if max_diff != 0.0:
            ok = False
        rows.append({"seed": int(seed), "vm_max_abs_diff": max_diff, "bitwise": max_diff == 0.0})
    return {"anchor_point": f"vc{ANCHOR_VC:.3f}_k{ANCHOR_K}", "pass": ok, "cells": rows}


def _anchor_trace_maxdiff(c3_spec: dict, signatures: dict[str, list[int]], seed: int) -> float:
    rep = replay_c3_cell("ordered_uniform", seed, spec=c3_spec)
    n = int(c3_spec["frozen_topology_and_weights"]["n_neurons"])
    sim_pol = c3_spec["simulation_policy"]
    dt_ms = float(sim_pol["dt_ms"])
    delays = signatures[f"vc{ANCHOR_VC:.3f}_k{ANCHOR_K}"]
    params, edges = build_ring_params_edges_km(
        n,
        ANCHOR_K,
        arc_delay_steps=np.asarray(delays, dtype=np.int32),
        weight=float(c3_spec["frozen_topology_and_weights"]["edge_weight"]),
        tau_ms=float(c3_spec["frozen_topology_and_weights"]["syn_tau_ms"]),
    )
    positions_3d = _ring_positions_3d(n, radius_mm=float(c3_spec["frozen_spatial_coordinates"]["ring_radius_mm"]))
    model = _build_c3_model(c3_spec, edges=edges, emitter=params, positions_3d=positions_3d, seed=seed)
    runtime = jtfne.RuntimeConfig(
        dtype=str(sim_pol["dtype"]),
        recurrent_backend="edge_list",
        enable_hdp=False,
        enable_homeostasis=False,
        hdp_params={"noise_scale": float(sim_pol["noise_scale"])},
    )
    sim = jtfne.Simulation(
        duration_ms=float(sim_pol["duration_ms"]),
        dt_ms=dt_ms,
        seed=int(seed),
        runtime=runtime,
        record_sources=True,
        record_fields=bool(sim_pol["record_fields"]),
    )
    signals = model.simulate(sim, paradigm=_stimulus_schedule(c3_spec))
    vm_anchor = np.asarray(signals.V_m, dtype=np.float64)
    return float(np.abs(vm_anchor - rep["V_m"]).max())


def run_search() -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    c3_spec = load_c3_spec()
    signatures = spec["delay_family"]["signatures_per_point"]
    points = []
    for pid, delays in signatures.items():
        vc = float(pid.split("_k")[0][2:])
        k = int(pid.split("_k")[1])
        cells = [simulate_cell(c3_spec, vc, k, delays, seed) for seed in (1001, 1002, 1003)]
        points.append({"point_id": pid, "velocity_mm_per_ms": vc, "k_neighbors": k, "delay_steps_per_skip": delays, "cells": cells})
    point_outcomes = {}
    for p in points:
        invalid = sum(1 for c in p["cells"] if c["invalid"])
        n_tw = sum(1 for c in p["cells"] if not c["invalid"] and c["classification"] == "TRAVELING_WAVE")
        if invalid > 0:
            outcome = "UNRESOLVED"
        elif n_tw >= 2:
            outcome = "POSITIVE"
        elif n_tw == 1:
            outcome = "MARGINAL"
        else:
            outcome = "NEGATIVE"
        point_outcomes[p["point_id"]] = {"outcome": outcome, "n_traveling_wave_cells": n_tw, "n_invalid_cells": invalid}
    n_pos = sum(1 for v in point_outcomes.values() if v["outcome"] == "POSITIVE")
    n_unres = sum(1 for v in point_outcomes.values() if v["outcome"] == "UNRESOLVED")
    if n_pos >= 1:
        domain_outcome = "POSITIVE_DOMAIN_FOUND"
    elif n_unres >= 1:
        domain_outcome = "UNRESOLVED"
    else:
        domain_outcome = "NO_POSITIVE_DOMAIN_IN_TESTED_RANGE"
    anchor = anchor_identity_check(c3_spec, signatures)
    receipt = {
        "schema": "jaxfne.protocol_c.p2v_a1b_receipt.v1",
        "protocol_id": "protocol_c_p2v_a1b",
        "phase": "post-freeze reviewer-motivated validation",
        "checkpoint": "A-1b",
        "status": "FROZEN",
        "write_once": True,
        "package_head": git_head(),
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "estimator_module_sha": estimator_module_sha(),
        "n_points": len(points),
        "n_cells": sum(len(p["cells"]) for p in points),
        "point_outcomes": point_outcomes,
        "domain_outcome": domain_outcome,
        "anchor_identity": anchor,
        "no_adaptive_extension_observed": True,
        "points": points,
    }
    if receipt["n_cells"] != spec["design_matrix"]["n_cells"]:
        raise RuntimeError(f"expected {spec['design_matrix']['n_cells']} cells, got {receipt['n_cells']}")
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {RECEIPT_PATH}")
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"A-1b done | points: {len(points)} | domain: {domain_outcome} | anchor_pass: {anchor['pass']}")
    for pid, v in point_outcomes.items():
        print(f"  {pid}: {v['outcome']} (TW {v['n_traveling_wave_cells']}, invalid {v['n_invalid_cells']})")
    return receipt


if __name__ == "__main__":
    run_search()