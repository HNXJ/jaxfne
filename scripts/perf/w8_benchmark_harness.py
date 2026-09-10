#!/usr/bin/env python3
"""W8: benchmark harness baseline (measurement only, no optimization).

Uses the W7/W10/W11 allocation taxonomy. Anchor profiles match W7 measurement
cells; an additional N sweep {1, 10, 100, 1000} records dense all-to-all scaling.

Memory buckets (aligned with W7):
  M_persistent          — arrays reachable from constructed model
  M_construction_peak   — RSS delta peak during construct
  M_recording           — arrays returned by simulate()
  M_simulation_state    — derived upper bound: max(0, M_peak_run - M_recording)
                          (JAX/XLA carry not fully visible via walk_arrays)

High R_k is recorded for class-shared arrays but is NOT treated as removability proof.

Usage:
    python scripts/perf/w8_benchmark_harness.py
    python scripts/perf/w8_benchmark_harness.py --json artifacts/audit/w8_baseline.json
    python scripts/perf/w8_benchmark_harness.py --cell dense_all_to_all_N1000
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_PERF = ROOT / "scripts" / "perf"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_PERF) not in sys.path:
    sys.path.insert(0, str(_PERF))

from w10_allocation_map import RSSSampler, build_config, describe, walk_arrays  # noqa: E402
from w11_scaling_atlas import _component_of, _realized_degree, _split_bytes  # noqa: E402

OUT_DEFAULT = ROOT / "artifacts" / "audit" / "w8_baseline.json"

# W7 anchor profiles (same configs as w7_dense_nxn_inventory measurements).
W7_PROFILES: list[dict[str, Any]] = [
    {
        "profile": "dense_all_to_all",
        "w7_cell": "dense_all_to_all_N1000",
        "n": 1000,
        "p_connect": None,
        "max_in_degree": None,
        "backend_requested": "dense",
    },
    {
        "profile": "bounded_degree_dual_storage",
        "w7_cell": "bounded_degree_N1000_K100",
        "n": 1000,
        "p_connect": 0.0,
        "max_in_degree": 100,
        "backend_requested": "edge_list",
        "note": "emitter.W may remain dense (COMPATIBILITY_ONLY); dynamics use edge_list",
    },
    {
        "profile": "dense_masked",
        "w7_cell": "dense_masked_N1000_p01",
        "n": 1000,
        "p_connect": 0.1,
        "max_in_degree": None,
        "backend_requested": "dense",
    },
    {
        "profile": "sparse_direct",
        "w7_cell": "sparse_direct_N6000_p01",
        "n": 6000,
        "p_connect": 0.1,
        "max_in_degree": None,
        "backend_requested": "edge_list",
    },
]

STANDARD_NS = (1, 10, 100, 1000)

# Fields that may drift between runs on the same commit (RSS sampling, JIT timing).
VOLATILE_FIELDS = (
    "generated_utc",
    "t_construct_s",
    "t_first_run_s",
    "t_second_run_s",
    "t_compile_s",
    "t_step_s",
    "M_construction_peak_bytes",
    "M_peak_run_bytes",
    "M_simulation_state_derived_bytes",
)


def _git_field(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _provenance(dtype: str = "float32") -> dict[str, Any]:
    import jax

    try:
        import jaxlib
        jaxlib_version = jaxlib.__version__
    except Exception:
        jaxlib_version = "unknown"
    devices = jax.devices()
    return {
        "git_head": _git_field(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_field(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib_version,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in devices[:8]],
        "dtype": dtype,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "recording_mode": "probes=['spikes'] via build_config (W10/W11 canonical cell)",
    }


def _is_nxn(shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and shape[0] == shape[1] and shape[0] > 0


def _nxn_summary(model, signals) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, array in walk_arrays(model):
        rec = describe(path, array)
        if not _is_nxn(tuple(rec["shape"])):
            continue
        rows.append({
            "array": rec["name"],
            "shape": rec["shape"],
            "dtype": rec["dtype"],
            "bytes": rec["bytes"],
            "R_k": rec.get("R_k"),
            "lifetime": "M_persistent",
        })
    for path, array in walk_arrays(signals):
        rec = describe(path, array)
        if not _is_nxn(tuple(rec["shape"])):
            continue
        rows.append({
            "array": rec["name"],
            "shape": rec["shape"],
            "dtype": rec["dtype"],
            "bytes": rec["bytes"],
            "R_k": rec.get("R_k"),
            "lifetime": "M_recording",
        })
    return rows


def run_profile(
    profile: dict[str, Any],
    duration_ms: float,
    dt_ms: float,
    seed: int,
) -> dict[str, Any]:
    import jaxfne as jtfne

    n = int(profile["n"])
    p_connect = profile.get("p_connect")
    max_in = profile.get("max_in_degree")
    backend = profile.get("backend_requested", "dense")

    cfg = build_config(n, p_connect, max_in, duration_ms, dt_ms, seed)
    cfg = cfg.runtime(recurrent_backend=backend)

    rec: dict[str, Any] = {
        "profile": profile.get("profile"),
        "w7_cell": profile.get("w7_cell"),
        "config": {
            "n": n,
            "p_connect": p_connect,
            "max_in_degree": max_in,
            "backend_requested": backend,
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "seed": seed,
        },
    }
    if profile.get("note"):
        rec["note"] = profile["note"]
    if profile.get("sweep"):
        rec["sweep"] = profile["sweep"]

    with RSSSampler() as rss_construct:
        t0 = time.perf_counter()
        model = jtfne.construct(cfg)
        rec["t_construct_s"] = time.perf_counter() - t0
    rec["M_construction_peak_bytes"] = int(rss_construct.peak - rss_construct.baseline)

    edge_list = model.params.get("edge_list")
    if edge_list is not None:
        e, k_in = _realized_degree(edge_list)
        rec["E"] = e
        rec["K_max_realized"] = k_in
    else:
        rec["E"] = 0
        rec["K_max_realized"] = None

    split, unclassified = _split_bytes(model)
    rec["M_persistent_components"] = split
    rec["M_other_paths"] = unclassified
    rec["M_persistent_bytes"] = int(sum(split.values()))

    with RSSSampler() as rss_run:
        t0 = time.perf_counter()
        signals = jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)
        rec["t_first_run_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=0)
        rec["t_second_run_s"] = time.perf_counter() - t0
    rec["M_peak_run_bytes"] = int(rss_run.peak - rss_run.baseline)

    n_steps = max(1, int(round(duration_ms / dt_ms)))
    rec["t_compile_s"] = max(0.0, rec["t_first_run_s"] - rec["t_second_run_s"])
    rec["t_step_s"] = rec["t_second_run_s"] / n_steps

    meta = getattr(signals, "metadata", {}) or {}
    rec["backend_realized"] = meta.get("recurrent_backend", "unreported")
    rec["M_recording_bytes"] = int(
        sum(int(np.asarray(a).nbytes) for _, a in walk_arrays(signals, path="signals"))
    )
    rec["M_simulation_state_derived_bytes"] = max(
        0, rec["M_peak_run_bytes"] - rec["M_recording_bytes"]
    )
    rec["M_simulation_state_note"] = (
        "derived RSS upper bound; JAX/XLA carry not fully visible via walk_arrays"
    )
    rec["nxn_arrays"] = _nxn_summary(model, signals)
    rec["nxn_bytes_persistent"] = sum(
        r["bytes"] for r in rec["nxn_arrays"] if r["lifetime"] == "M_persistent"
    )
    rec["status"] = "ok"
    return rec


def _standard_sweep_cells(duration_ms: float, dt_ms: float, seed: int) -> list[dict[str, Any]]:
    cells = []
    for n in STANDARD_NS:
        cells.append({
            "profile": f"dense_all_to_all_N{n}",
            "w7_cell": None,
            "n": n,
            "p_connect": None,
            "max_in_degree": None,
            "backend_requested": "dense",
            "sweep": "standard_N",
        })
    return cells


def build_report(duration_ms: float, dt_ms: float, seed: int, include_sweep: bool) -> dict[str, Any]:
    profiles = list(W7_PROFILES)
    if include_sweep:
        profiles = profiles + _standard_sweep_cells(duration_ms, dt_ms, seed)

    results = []
    for prof in profiles:
        results.append(run_profile(prof, duration_ms, dt_ms, seed))

    return {
        "audit": "w8_benchmark_baseline",
        "schema": "jaxfne.w8.baseline.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": _provenance(dtype="float32"),
        "volatile_fields": list(VOLATILE_FIELDS),
        "package_version_note": "baseline for current dev SHA; not a universal performance claim",
        "taxonomy": {
            "M_persistent": "constructed model arrays (W10 walk_arrays on model)",
            "M_construction_peak": "RSS delta during construct",
            "M_recording": "simulate() return arrays",
            "M_simulation_state": "derived from run RSS minus recording; upper bound only",
            "R_k": "materialized/distinct; high R_k does not imply removability",
        },
        "reconciles_with": [
            "artifacts/audit/w7_dense_nxn_inventory.json",
            "scripts/perf/w10_allocation_map.py",
            "scripts/perf/w11_scaling_atlas.py",
        ],
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "seed": seed,
        "cells": results,
        "w8_verdict": "BASELINE_RECORDED",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, default=str(OUT_DEFAULT))
    ap.add_argument("--duration-ms", type=float, default=20.0)
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-sweep", action="store_true", help="W7 anchors only")
    ap.add_argument("--cell", type=str, default=None, help="run one profile by w7_cell or profile name")
    args = ap.parse_args(argv)

    if args.cell:
        match = None
        for p in W7_PROFILES + _standard_sweep_cells(args.duration_ms, args.dt_ms, args.seed):
            if args.cell in (p.get("w7_cell"), p.get("profile")):
                match = p
                break
        if match is None:
            print(json.dumps({"error": f"unknown cell {args.cell!r}"}))
            return 1
        print(json.dumps(run_profile(match, args.duration_ms, args.dt_ms, args.seed)))
        return 0

    report = build_report(args.duration_ms, args.dt_ms, args.seed, not args.no_sweep)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "path": str(out.relative_to(ROOT)),
        "cells": len(report["cells"]),
        "w8_verdict": report["w8_verdict"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
