#!/usr/bin/env python3
"""W7: classify every dense N×N allocation/materialization site (construction + simulation).

Classification only — no representation changes. Reconciles with W10/W11 rather than
competing with them:

  - Static sites: every source location that can materialize (n,n) or (N,N) arrays.
  - Measured sites: N×N arrays observed at representative configurations via w10 walk.

Memory buckets (aligned with W10):
  M_persistent, M_construction_peak (M_peak_rss_delta), M_simulation_state, M_recording

Usage:
    python scripts/audit_w7_dense_nxn_sites.py
    python scripts/audit_w7_dense_nxn_sites.py --json artifacts/audit/w7_dense_nxn_inventory.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_PERF = ROOT / "scripts" / "perf"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

from w10_allocation_map import (  # noqa: E402
    RSSSampler,
    build_config,
    describe,
    total_bytes,
    walk_arrays,
)

OUT_DEFAULT = ROOT / "artifacts" / "audit" / "w7_dense_nxn_inventory.json"

# Static registry: construction + simulation paths only (excludes tutorial/protocol/analysis-only).
# roadmap_class mirrors ROADMAP §W7; role is the user-requested AUTHORITATIVE|... taxonomy.
STATIC_SITES: list[dict[str, Any]] = [
    {
        "id": "construct_population.dense_W",
        "file": "jaxfne/_construct_population.py",
        "line": 400,
        "array": "params.emitter.W",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "construct() dense within-area connectivity (p_connect unset, 1.0, or masked sparse on dense path)",
        "lifetime": "M_persistent",
        "backend": "dense",
        "required_by": "simulate(dense recurrent_backend); inspect; checkpoint when dense",
        "role": "AUTHORITATIVE",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "yes_when_dense_backend_or_dual_storage",
        "exact_reduction_candidate": "W10 U_k=0 on edge_list backend (COMPATIBILITY_ONLY dual storage)",
    },
    {
        "id": "construct_population.dense_rnd",
        "file": "jaxfne/_construct_population.py",
        "line": 392,
        "array": "rnd (construction transient)",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "dense path weight draw before W assembly",
        "lifetime": "M_construction_peak",
        "backend": "dense",
        "required_by": "W assembly only",
        "role": "TEMPORARY",
        "roadmap_class": "AVOIDABLE",
        "bounded_degree_still_allocates": "no_when_sparse_direct_or_edge_list_only",
        "exact_reduction_candidate": "sparse-direct path at n>=5000",
    },
    {
        "id": "construct_population.dense_eye",
        "file": "jaxfne/_construct_population.py",
        "line": 399,
        "array": "eye",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "dense path self-connection mask",
        "lifetime": "M_construction_peak",
        "backend": "dense",
        "required_by": "W assembly only",
        "role": "TEMPORARY",
        "roadmap_class": "AVOIDABLE",
        "bounded_degree_still_allocates": "no_when_sparse_direct",
        "exact_reduction_candidate": "fused into sparse edge generation",
    },
    {
        "id": "construct_population.dense_bernoulli_mask",
        "file": "jaxfne/_construct_population.py",
        "line": 446,
        "array": "mask",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "0<p_connect<1 on dense path (n < sparse-direct threshold or special specs)",
        "lifetime": "M_construction_peak",
        "backend": "dense",
        "required_by": "masked dense W",
        "role": "TEMPORARY",
        "roadmap_class": "AVOIDABLE",
        "bounded_degree_still_allocates": "yes_on_dense_masked_path",
        "exact_reduction_candidate": "sparse-direct edge list (n>=5000, plain within-area)",
    },
    {
        "id": "construct_population.sparse_direct_placeholder_W",
        "file": "jaxfne/_construct_population.py",
        "line": 388,
        "array": "params.emitter.W",
        "shape": "(0, 0)",
        "dtype": "float32|float64",
        "created_when": "sparse-direct path (p_connect<1, n>=5000, no TCM/interarea)",
        "lifetime": "M_persistent",
        "backend": "edge_list",
        "required_by": "placeholder only; dynamics use edge_list",
        "role": "DERIVED",
        "roadmap_class": "COMPATIBILITY_ONLY",
        "bounded_degree_still_allocates": "no",
        "exact_reduction_candidate": "already reduced",
    },
    {
        "id": "construct_connectivity.interarea_Wadd",
        "file": "jaxfne/_construct_connectivity.py",
        "line": 76,
        "array": "Wadd",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "suite2_interarea / inter-column additive cross-area weights",
        "lifetime": "M_persistent",
        "backend": "dense",
        "required_by": "merged ensemble dense additive connectivity",
        "role": "AUTHORITATIVE",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "yes_for_interarea_specs",
        "exact_reduction_candidate": "no; masks require full (n,n) plane",
    },
    {
        "id": "construct_connectivity.interarea_bern_wval",
        "file": "jaxfne/_construct_connectivity.py",
        "lines": [95, 96],
        "array": "bern, wval",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "interarea pair_mask sampling during Wadd build",
        "lifetime": "M_construction_peak",
        "backend": "dense",
        "required_by": "Wadd assembly",
        "role": "TEMPORARY",
        "roadmap_class": "AVOIDABLE",
        "bounded_degree_still_allocates": "yes_when_interarea_active",
        "exact_reduction_candidate": "pair-local sparse accumulation (future)",
    },
    {
        "id": "emitters.default_eig_W",
        "file": "jaxfne/emitters.py",
        "line": 266,
        "array": "weights -> emitter.W",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "cortical_eig preset when build_dense_connectivity=True",
        "lifetime": "M_persistent",
        "backend": "dense",
        "required_by": "default izhikevich eig connectivity template",
        "role": "AUTHORITATIVE",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "yes_when_preset_used_without_edge_list",
        "exact_reduction_candidate": "edge_list replacement when rules materialize",
    },
    {
        "id": "emitters.placeholder_W",
        "file": "jaxfne/emitters.py",
        "line": 378,
        "array": "W",
        "shape": "(0, 0)",
        "dtype": "float32|float64",
        "created_when": "build_dense_connectivity=False (edge-list-first construction)",
        "lifetime": "M_persistent",
        "backend": "edge_list",
        "required_by": "compatibility placeholder",
        "role": "DERIVED",
        "roadmap_class": "COMPATIBILITY_ONLY",
        "bounded_degree_still_allocates": "no",
        "exact_reduction_candidate": "already reduced",
    },
    {
        "id": "construct_core.homeostatic_G0",
        "file": "jaxfne/_construct_core.py",
        "line": 939,
        "array": "G0",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "HDP/homeostatic EI params attached at construct",
        "lifetime": "M_persistent",
        "backend": "dense",
        "required_by": "homeostatic_ei simulate path",
        "role": "AUTHORITATIVE",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "yes_when_homeostasis_enabled",
        "exact_reduction_candidate": "no; dense G is model definition for this emitter",
    },
    {
        "id": "emitters_homeostatic_ei.G0",
        "file": "jaxfne/emitters_homeostatic_ei.py",
        "line": 442,
        "array": "HomeostaticEIParams.G0",
        "shape": "(n, n)",
        "dtype": "float32|float64",
        "created_when": "homeostatic_ei canonical params factory",
        "lifetime": "M_persistent",
        "backend": "dense",
        "required_by": "simulate_homeostatic_ei scan carry",
        "role": "AUTHORITATIVE",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "yes",
        "exact_reduction_candidate": "no; documented O(N^2) design",
    },
    {
        "id": "emitters_homeostatic_ei.G_history",
        "file": "jaxfne/emitters_homeostatic_ei.py",
        "line": 516,
        "array": "G_history",
        "shape": "(n_steps, n, n)",
        "dtype": "float32|float64",
        "created_when": "simulate_homeostatic_ei when recording full G trajectory",
        "lifetime": "M_recording",
        "backend": "dense",
        "required_by": "returned diagnostics/history only",
        "role": "DERIVED",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "yes",
        "exact_reduction_candidate": "streaming/recording throttle (W17.2); not removal of G state",
    },
    {
        "id": "plasticity.stdp_W",
        "file": "jaxfne/plasticity.py",
        "line": 23,
        "array": "STDPState.W",
        "shape": "(n_neurons, n_neurons)",
        "dtype": "float32|float64",
        "created_when": "STDP state initialization / updates",
        "lifetime": "M_simulation_state",
        "backend": "dense",
        "required_by": "run_stdp_stream low-level API (not model.params)",
        "role": "AUTHORITATIVE",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "yes_for_stdp_path",
        "exact_reduction_candidate": "no without changing STDP semantics",
    },
    {
        "id": "model_tune.matrix_mask",
        "file": "jaxfne/_model_tune.py",
        "line": 911,
        "array": "mask",
        "shape": "(n, n)",
        "dtype": "bool",
        "created_when": "matrix parameter tuning scaffold",
        "lifetime": "M_construction_peak",
        "backend": "any",
        "required_by": "tune() matrix AGSDR extraction",
        "role": "TEMPORARY",
        "roadmap_class": "ANALYSIS_ONLY",
        "bounded_degree_still_allocates": "only_during_tune",
        "exact_reduction_candidate": "sparse mask from edge_list (future)",
    },
    {
        "id": "geometry.random_W",
        "file": "jaxfne/geometry.py",
        "line": 41,
        "array": "W, W_signed",
        "shape": "(n_neurons, n_neurons)",
        "dtype": "float32|float64",
        "created_when": "geometry-driven weight initialization helpers",
        "lifetime": "M_construction_peak",
        "backend": "dense",
        "required_by": "geometry workflows feeding construct",
        "role": "DERIVED",
        "roadmap_class": "INTRINSIC",
        "bounded_degree_still_allocates": "depends_on_workflow",
        "exact_reduction_candidate": "no",
    },
    {
        "id": "sanity_runtime.zeros_W",
        "file": "jaxfne/sanity_runtime.py",
        "line": 83,
        "array": "W",
        "shape": "(n_neurons, n_neurons)",
        "dtype": "float64",
        "created_when": "sanity / smoke runtime checks",
        "lifetime": "M_construction_peak",
        "backend": "dense",
        "required_by": "sanity_runtime only",
        "role": "VALIDATION_ONLY",
        "roadmap_class": "ANALYSIS_ONLY",
        "bounded_degree_still_allocates": "n/a",
        "exact_reduction_candidate": "n/a",
    },
]

# Sites explicitly out of W7 construction/simulation scope (documented, not UNKNOWN).
OUT_OF_SCOPE: list[dict[str, str]] = [
    {"file": "jaxfne/tutorial_utils.py", "reason": "TUTORIAL_ONLY"},
    {"file": "jaxfne/fields/proxy.py", "reason": "TUTORIAL_ONLY proxy field demos"},
    {"file": "jaxfne/h4_matrix.py", "reason": "protocol harness"},
    {"file": "jaxfne/protocol_d_biological_rbs/d1_execution.py", "reason": "protocol harness"},
    {"file": "jaxfne/h3_decodability.py", "reason": "ANALYSIS_ONLY (n_features², not neuron N)"},
    {"file": "jaxfne/w3_stability_analysis.py", "reason": "ANALYSIS_ONLY"},
    {"file": "jaxfne/validation.py", "reason": "VALIDATION_ONLY shape checks, no allocation"},
]

MEASUREMENT_CELLS = [
    {"label": "dense_all_to_all_N1000", "n": 1000, "p_connect": None, "max_in_degree": None},
    {"label": "bounded_degree_N1000_K100", "n": 1000, "p_connect": 0.0, "max_in_degree": 100},
    {"label": "dense_masked_N1000_p01", "n": 1000, "p_connect": 0.1, "max_in_degree": None},
    {"label": "sparse_direct_N6000_p01", "n": 6000, "p_connect": 0.1, "max_in_degree": None},
]


def _bytes_n(n: int, dtype: str = "float32") -> int:
    bpe = 8 if dtype == "float64" else 4
    return n * n * bpe


def _is_nxn(shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and shape[0] == shape[1] and shape[0] > 0


def _measure_cell(cell: dict[str, Any], duration_ms: float, dt_ms: float, seed: int) -> dict[str, Any]:
    import jaxfne as jtfne

    cfg = build_config(cell["n"], cell["p_connect"], cell["max_in_degree"], duration_ms, dt_ms, seed)
    with RSSSampler() as rss:
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)

    m_persistent = total_bytes(model)
    m_recording = total_bytes(signals)
    m_peak = rss.peak - rss.baseline

    nxn_arrays: list[dict[str, Any]] = []
    for path, array in walk_arrays(model):
        rec = describe(path, array)
        if not _is_nxn(rec["shape"]):
            continue
        n = rec["shape"][0]
        rec.update(
            {
                "ARRAY": rec["name"],
                "SHAPE": rec["shape"],
                "DTYPE": rec["dtype"],
                "BYTES_N": rec["bytes"],
                "BYTES_formula_at_n": _bytes_n(n, "float64" if "64" in rec["dtype"] else "float32"),
                "CREATED_WHEN": "observed post-construct in model graph",
                "LIFETIME": "M_persistent",
                "BACKEND": signals.metadata.get("recurrent_backend"),
                "REQUIRED_BY": "see W10 U_k probes (w10_allocation_map.py)",
                "role": "AUTHORITATIVE",
                "R_k": rec.get("R_k"),
            }
        )
        nxn_arrays.append(rec)

    # Recording-phase N×N (e.g. homeostatic G_history not in default izhikevich cell)
    for path, array in walk_arrays(signals):
        rec = describe(path, array)
        if not _is_nxn(rec["shape"]):
            continue
        rec.update(
            {
                "ARRAY": rec["name"],
                "SHAPE": rec["shape"],
                "DTYPE": rec["dtype"],
                "BYTES_N": rec["bytes"],
                "LIFETIME": "M_recording",
                "BACKEND": signals.metadata.get("recurrent_backend"),
                "role": "DERIVED",
                "R_k": rec.get("R_k"),
            }
        )
        nxn_arrays.append(rec)

    edge_list = model.params.get("edge_list")
    n_edges = int(edge_list.n_edges) if edge_list is not None else 0
    k_max_realized = None
    if edge_list is not None and n_edges > 0:
        import numpy as np

        post = np.asarray(edge_list.post)
        if post.size:
            _, counts = np.unique(post, return_counts=True)
            k_max_realized = int(counts.max())

    return {
        "cell": cell["label"],
        "config": cell,
        "realized": {
            "n_neurons": cell["n"],
            "n_edges": n_edges,
            "k_max_in_realized": k_max_realized,
            "recurrent_backend": signals.metadata.get("recurrent_backend"),
        },
        "memory_bytes": {
            "M_persistent": m_persistent,
            "M_recording": m_recording,
            "M_construction_peak_rss_delta": m_peak,
            "M_simulation_state_note": "JAX/XLA carry arrays not fully visible via walk_arrays; see W10 M_temporary",
        },
        "nxn_arrays_materialized": nxn_arrays,
        "nxn_count": len(nxn_arrays),
        "nxn_bytes_total": sum(a["BYTES_N"] for a in nxn_arrays),
    }


def _enrich_static(site: dict[str, Any], n: int = 1000) -> dict[str, Any]:
    out = dict(site)
    dtype = site.get("dtype", "float32")
    base = "float64" if "64" in dtype else "float32"
    out["BYTES_at_N"] = _bytes_n(n, base) if "(n, n)" in site.get("shape", "") else 0
    shape = site.get("shape", "")
    m = re.search(r"\(n_steps,\s*n,\s*n\)", shape)
    if m:
        out["BYTES_at_N_note"] = "n_steps * n^2 * dtype; recording bucket"
    return out


def build_report() -> dict[str, Any]:
    static = [_enrich_static(s) for s in STATIC_SITES]
    unknown_static = [s for s in static if s.get("roadmap_class") == "UNKNOWN"]
    measurements = [_measure_cell(c, duration_ms=20.0, dt_ms=0.5, seed=1) for c in MEASUREMENT_CELLS]

    return {
        "audit": "w7_dense_nxn_inventory",
        "schema": "jaxfne.w7.dense_nxn.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reconciles_with": ["scripts/perf/w10_allocation_map.py", "scripts/perf/w11_scaling_atlas.py"],
        "memory_buckets": {
            "M_persistent": "arrays reachable from constructed model after construct()",
            "M_construction_peak": "RSS delta peak during construct+simulate (W10 M_peak_rss_delta)",
            "M_simulation_state": "time-step carry not always RSS-visible; STDP W, scan carries",
            "M_recording": "arrays returned by simulate() / history buffers",
        },
        "R_k_note": "R_k = materialized_elements / distinct_values; high R_k does not imply removability (W10 consumer probes required)",
        "static_sites": static,
        "static_site_count": len(static),
        "out_of_scope": OUT_OF_SCOPE,
        "measurements": measurements,
        "unexplained_sites": unknown_static,
        "unexplained_count": len(unknown_static),
        "w7_verdict": "CLOSED" if len(unknown_static) == 0 else "BLOCKED",
        "presentation_note": "129 presentation broad-exception handlers unchanged (W3); W7 does not revisit them.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, default=str(OUT_DEFAULT))
    args = ap.parse_args(argv)

    report = build_report()
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "path": str(out.relative_to(ROOT)),
        "static_sites": report["static_site_count"],
        "unexplained": report["unexplained_count"],
        "w7_verdict": report["w7_verdict"],
        "cells": [m["cell"] for m in report["measurements"]],
    }))
    return 0 if report["unexplained_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
