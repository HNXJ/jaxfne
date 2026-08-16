"""A-1b pre-freeze evidence: exact anchor identity + delay-signature enumeration.

Frozen facts, computed before any A-1b outcome is observed:
  1. Exact relation between frozen ``ordered_uniform`` delay steps, ring arc
     length, dt, and the proposed anchor v_c = 0.131 mm/ms.
  2. Full enumeration of (v_c, K) -> delay signature over the 15-point
     lattice, with the unique-signature count and quantization plateaus.
  3. Construction-level anchor identity: K=1 anchor edges/emitter bitwise vs
     the frozen ``build_ring_params_edges`` path, plus full-cell V_m trace
     bitwise identity (seeds 1001-1003) against the frozen ordered_uniform
     condition.

Output: ``artifacts/protocol_c/p2v_a1b_dynamic_search/p2v_a1b_prefreeze_evidence.json``
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import jaxfne as jtfne
from jaxfne.h4_matrix import build_ring_params_edges, build_ring_params_edges_km
from jaxfne.protocol_c.c3_execution import (
    _build_c3_model,
    _ring_positions_3d,
    _stimulus_schedule,
    replay_c3_cell,
)
from jaxfne.protocol_c.c3_protocol import load_c3_spec
from jaxfne.protocol_c.estimator import estimate_traveling_wave
from jaxfne.protocol_c.protocol import load_protocol_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
A1B_DIR = REPO_ROOT / "artifacts" / "protocol_c" / "p2v_a1b_dynamic_search"
EVIDENCE_PATH = A1B_DIR / "p2v_a1b_prefreeze_evidence.json"

ANCHOR_VC = 0.131
V_C_LATTICE = [0.033, 0.065, 0.131, 0.262, 0.524]
K_LATTICE = [1, 2, 4]
ANCHOR_SEEDS = [1001, 1002, 1003]


@dataclass
class EnumResult:
    signature: list[int]
    ratio64: float
    ratio32: float
    ceil64: int
    ceil32: int


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def estimator_module_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD:jaxfne/protocol_c/estimator.py"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def arc_length_mm(n: int, radius_mm: float) -> float:
    return 2.0 * math.pi * radius_mm / float(n)


def delay_signature(n: int, radius_mm: float, dt_ms: float, v_c: float, k: int) -> EnumResult:
    a = arc_length_mm(n, radius_mm)
    a32 = np.float32(a)
    delays64, delays32 = [], []
    for m in range(1, k + 1):
        r64 = m * a / (v_c * dt_ms)
        r32 = float(np.float32(np.float32(m) * a32 / (np.float32(np.float32(v_c) * np.float32(dt_ms)))))
        delays64.append(int(math.ceil(r64)))
        delays32.append(int(math.ceil(r32)))
        if delays64[-1] != delays32[-1]:
            raise AssertionError(f"float32/float64 divergence at m={m}, v_c={v_c}: {delays64[-1]} vs {delays32[-1]}")
    return EnumResult(signature=delays64, ratio64=r64, ratio32=r32, ceil64=delays64[-1], ceil32=delays32[-1])


def edges_equal(a, b) -> tuple[bool, dict[str, str]]:
    diffs = {}
    for field_name in ("pre", "post", "weight", "receptor_index", "tau_ms", "delay_steps"):
        va, vb = getattr(a, field_name), getattr(b, field_name)
        same = bool((va == vb).all())
        if same:
            diffs[field_name] = "identical"
        else:
            dv = va.astype("float64") - vb.astype("float64")
            diffs[field_name] = f"DIFF max={float(np.abs(dv).max())}"
    ok = all(v == "identical" for v in diffs.values())
    return ok, diffs


def params_equal(a, b) -> tuple[bool, dict[str, str]]:
    diffs = {}
    for field_name in ("v0", "u0", "a", "b", "c", "d", "drive", "sign", "W", "source_scale"):
        va, vb = getattr(a, field_name), getattr(b, field_name)
        same = bool((va == vb).all())
        diffs[field_name] = "identical" if same else "DIFF"
    ok = all(v == "identical" for v in diffs.values())
    return ok, diffs


def build_anchor_cell(vc: float, k: int, seed: int, spec: dict, signatures: dict[tuple[float, int], list[int]]):
    delay_vec = np.asarray(signatures[(vc, k)], dtype=np.int32)
    n = int(spec["frozen_topology_and_weights"]["n_neurons"])
    dt_ms = float(spec["simulation_policy"]["dt_ms"])
    params, edges = build_ring_params_edges_km(
        n,
        k,
        arc_delay_steps=delay_vec,
        weight=float(spec["frozen_topology_and_weights"]["edge_weight"]),
        tau_ms=float(spec["frozen_topology_and_weights"]["syn_tau_ms"]),
    )
    positions_3d = _ring_positions_3d(n, radius_mm=float(spec["frozen_spatial_coordinates"]["ring_radius_mm"]))
    model = _build_c3_model(spec, edges=edges, emitter=params, positions_3d=positions_3d, seed=seed)
    sim_pol = spec["simulation_policy"]
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
    signals = model.simulate(sim, paradigm=_stimulus_schedule(spec))
    return np.asarray(signals.V_m, dtype=np.float64), edges, params


def main() -> dict:
    spec = load_c3_spec()
    topo = spec["frozen_topology_and_weights"]
    n = int(topo["n_neurons"])
    radius = float(spec["frozen_spatial_coordinates"]["ring_radius_mm"])
    dt_ms = float(spec["simulation_policy"]["dt_ms"])
    a = arc_length_mm(n, radius)
    d_frozen = int(spec["delay_construction"]["uniform_delay_steps"])

    # --- 1. exact anchor relation ---
    ratio64 = a / (ANCHOR_VC * dt_ms)
    ceil_res = int(math.ceil(ratio64))
    exact_identity = ceil_res == d_frozen
    lo = a / (d_frozen * dt_ms)  # v_c at which ratio == d_frozen exactly (ceil -> d_frozen)
    hi = a / ((d_frozen - 1) * dt_ms)  # open upper bound: ratio -> d_frozen - 1
    margin = d_frozen - ratio64

    # --- 2. signature enumeration over the 15 lattice points ---
    enum: dict[str, dict] = {}
    signature_map: dict[tuple[float, int], list[int]] = {}
    for vc in V_C_LATTICE:
        for k in K_LATTICE:
            r = delay_signature(n, radius, dt_ms, vc, k)
            enum[f"vc{vc:.3f}_k{k}"] = {
                "velocity_mm_per_ms": vc,
                "k_neighbors": k,
                "delay_steps_per_skip": r.signature,
                "last_skip_ratio": r.ratio64,
            }
            signature_map[(vc, k)] = r.signature
    unique_sigs = {tuple(v) for v in signature_map.values()}
    # quantization plateaus: (delay value, [(vc, skip m) ...]) for cross-point aliases
    plateaus: dict[int, list[tuple[float, int]]] = {}
    for vc in V_C_LATTICE:
        for m in range(1, 4):
            d_val = signature_map[(vc, 4)][m - 1]
            plateaus.setdefault(int(d_val), []).append((float(vc), int(m)))

    # --- 3. tensor identity at the anchor point (vc=0.131, K=1) ---
    frozen_delay = np.full((n,), d_frozen, dtype=np.int32)
    params_f, edges_f = build_ring_params_edges(
        n,
        delay_steps=frozen_delay,
        weight=float(topo["edge_weight"]),
        tau_ms=float(topo["syn_tau_ms"]),
    )
    params_a, edges_a = build_ring_params_edges_km(
        n,
        1,
        arc_delay_steps=signature_map[(ANCHOR_VC, 1)],
        weight=float(topo["edge_weight"]),
        tau_ms=float(topo["syn_tau_ms"]),
    )
    edges_ok, edges_diffs = edges_equal(edges_a, edges_f)
    params_ok, params_diffs = params_equal(params_a, params_f)

    # --- 4. full-cell trace identity vs frozen ordered_uniform (seeds 1001-1003) ---
    trace_rows = []
    traces_identical = True
    for seed in ANCHOR_SEEDS:
        rep = replay_c3_cell("ordered_uniform", seed, spec=spec)
        vm_replay = rep["V_m"]
        vm_anchor, edges_a2, params_a2 = build_anchor_cell(ANCHOR_VC, 1, seed, spec, signature_map)
        max_diff = float(np.abs(vm_anchor - vm_replay).max())
        sha_replay = hashlib.sha256(np.ascontiguousarray(vm_replay).tobytes()).hexdigest()
        sha_anchor = hashlib.sha256(np.ascontiguousarray(vm_anchor).tobytes()).hexdigest()
        est_anchor = estimate_traveling_wave(
            vm_anchor, rep["positions"], dt_ms=dt_ms, spec=load_protocol_spec()
        )
        est_equal = bool(rep["estimator"].to_dict() == est_anchor.to_dict())
        if max_diff != 0.0:
            traces_identical = False
        trace_rows.append(
            {
                "seed": int(seed),
                "vm_max_abs_diff": max_diff,
                "vm_sha256_replay": sha_replay,
                "vm_sha256_anchor": sha_anchor,
                "estimator_equal_replay_vs_anchor": est_equal,
            }
        )

    evidence = {
        "schema": "jaxfne.protocol_c.p2v_a1b_prefreeze_evidence.v1",
        "phase": "post-freeze reviewer-motivated validation",
        "checkpoint": "A-1b (pre-freeze)",
        "package_head": git_head(),
        "estimator_module_sha": estimator_module_sha(),
        "frozen_parents": {
            "c3_spec": "artifacts/protocol_c/c3_neural_experiment_spec.json",
            "c3_receipt": "artifacts/protocol_c/c3_execution_receipt.json",
            "a1a_receipt": "artifacts/protocol_c/p2v_a1a_synthetic_control/p2v_a1a_receipt.json",
        },
        "geometry_and_units": {
            "n_neurons": n,
            "ring_radius_mm": radius,
            "arc_length_mm": a,
            "dt_ms_for_delay": dt_ms,
            "directed_edge_convention": "skip m: neuron i -> neuron (i+m) mod n",
            "delay_equation": "delay_steps(m) = ceil(m * arc_length_mm / (v_c_mm_per_ms * dt_ms))",
            "d_frozen_ordered_uniform": d_frozen,
        },
        "anchor_relation": {
            "proposed": {"v_c_mm_per_ms": ANCHOR_VC, "k": 1},
            "exact_ratio_arc_over_vc_dt": ratio64,
            "ceil_result": ceil_res,
            "exact_identity_holds": exact_identity,
            "velocity_interval_for_d_frozen": {
                "lower_inclusive": lo,
                "upper_open": hi,
                "rationale": "ceil(a/(v_c*dt)) == d_frozen iff ratio in (d_frozen-1, d_frozen], i.e. v_c in [a/(d_frozen*dt), a/((d_frozen-1)*dt))",
            },
            "margin_to_next_integer": margin,
            "note": "0.131 mm/ms lies inside the exact interval reproducing d_frozen=4; identity holds by ceiling construction and is additionally verified at tensor and trace level below",
        },
        "delay_signatures": {
            "formula": "delay_steps(m) = ceil(m * a / (v_c * dt)), verified identical in float32 and float64 arithmetic",
            "lattice_points": enum,
            "n_nominal_points": len(enum),
            "unique_signatures": len(unique_sigs),
            "unique_count_note": "each of the 15 (v_c, K) points maps to a distinct delay vector; no collapsing required",
            "quantization_plateaus": {
                "delay_value": [
                    {"delay_steps": int(d), "arcs_aliased": [(float(v), m) for v, m in xs]}
                    for d, xs in sorted(plateaus.items())
                ],
                "reading": "identical integer delays across different (v_c, skip) pairs are quantization-equivalent arcs; they do not collapse the lattice because each point keeps its own full per-skip vector",
            },
        },
        "anchor_tensor_identity": {
            "edges": edges_diffs,
            "edges_bitwise_identical": edges_ok,
            "emitter": params_diffs,
            "emitter_bitwise_identical": params_ok,
        },
        "anchor_trace_identity": {
            "seeds": [int(s) for s in ANCHOR_SEEDS],
            "cells": trace_rows,
            "all_traces_bitwise_identical": traces_identical,
            "interpretation": "bitwise V_m equality vs the frozen ordered_uniform replay implies the entire construction (topology, edge ordering, delays, weights, initialization) is identical, not merely classification-equivalent",
        },
    }
    A1B_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["anchor_relation"], indent=2))
    print(f"unique_signatures: {len(unique_sigs)} / {len(enum)}")
    print(f"edges_bitwise_identical: {edges_ok} | emitter_bitwise_identical: {params_ok}")
    print(f"all_traces_bitwise_identical: {traces_identical}")
    return evidence


if __name__ == "__main__":
    main()