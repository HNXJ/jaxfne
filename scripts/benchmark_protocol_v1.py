#!/usr/bin/env python3
"""Benchmark protocol v1 (P-06): synchronized, deterministic wall-time measurement.

Measures existing simulation paths (standard EIG/Izhikevich and HDP-enabled)
without modifying jaxfne. Timing uses ``time.perf_counter`` around a
simulation call that is forced to finish with ``jax.block_until_ready``, so
reported seconds include device execution (not just driver-enqueue time).

Deliberate scope (this script is measurement infrastructure, not analysis):
- No speedup/throughput/improvement claims are printed or inferred.
- The HDP arm uses the same workload as the standard arm with HDP enabled;
  it does not assert stability or superiority.
- Fixed-seed runs are deterministic per backend; repeated runs may differ in
  wall time and the report makes no cross-run equality claim beyond shape and
  finiteness.

Usage:
    PYTHONPATH=. JAX_PLATFORMS=cpu python scripts/benchmark_protocol_v1.py \
        --backend cpu --mode both --warmup 1 --runs 1 \
        --record-fields off --record-weight-trace off --json-out out.json

    # GPU availability check on a GPU-less machine -> structured SKIP, exit 0:
    PYTHONPATH=. python scripts/benchmark_protocol_v1.py --backend gpu --json-out out_gpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from typing import Any, Optional

import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne import hdp_network

SCHEMA_VERSION = "benchmark_protocol_v1.schema.json"
CLAIM_LEVEL = "local_environment_receipt_only"

MODE_STANDARD = "standard"
MODE_HDP = "hdp"
MODES = (MODE_STANDARD, MODE_HDP)

# Workload presets: the default is deliberately short (CPU-comfortable HDP
# arm ~5s/run, verified by probe); "large" is the T=1000ms/N=1000 workload
# the handout asked to verify before defaulting, kept as explicit opt-in.
PRESETS = {
    "default": {"n_neurons": 400, "duration_ms": 200.0, "dt_ms": 0.05},
    "large": {"n_neurons": 1000, "duration_ms": 1000.0, "dt_ms": 0.05},
}

# HDP parameter defaults come from the module's own verified preset
# (hdp_network.DEFAULT_HDP). The benchmark forces record_weight_trace=False
# by default (stacking w_trace (T,E) is the documented allocation risk).
HDP_PARAMS = dict(hdp_network.DEFAULT_HDP)
HDP_PARAMS["record_weight_trace"] = False

_DTYPE = "float32"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Requested execution backend (default: auto = resolver-selected).",
    )
    p.add_argument("--runs", type=int, default=3, help="measured runs per mode (default 3).")
    p.add_argument(
        "--warmup", type=int, default=1, help="warmup runs per mode, excluded (default 1)."
    )
    p.add_argument("--seed", type=int, default=0, help="explicit JAX PRNG seed (default 0).")
    p.add_argument(
        "--mode",
        choices=("standard", "hdp", "both"),
        default="both",
        help="which simulation path to measure (default both).",
    )
    p.add_argument(
        "--record-fields",
        choices=("on", "off"),
        default="off",
        help="record the proxy field output (default off; minimal load).",
    )
    p.add_argument(
        "--record-weight-trace",
        choices=("on", "off"),
        default="off",
        help="record the HDP w_trace (default off).",
    )
    p.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="default",
        help="workload preset (default default; 'large' is explicit opt-in).",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="write the JSON report to this path (optional).",
    )
    return p.parse_args(argv)


class GpuUnavailable(Exception):
    """Raised when the GPU backend was requested but no GPU devices exist."""


def resolve_backend(requested: str) -> tuple[str, str]:
    """Return (resolved_platform, note). Never silently falls back: GPU
    requested without GPU devices raises GpuUnavailable so the caller can
    emit a structured SKIP result."""
    if requested == "gpu":
        try:
            devices = jax.devices("gpu")
        except RuntimeError:  # platform absent entirely
            raise GpuUnavailable()
        if not devices:
            raise GpuUnavailable()
        return "gpu", f"found {len(devices)} device(s)"
    if requested == "cpu":
        return "cpu", "forced (JAX_PLATFORMS honored by simulation scope)"
    try:
        default = jax.devices()
    except Exception:  # pragma: no cover - exotic platforms
        return "auto", "no device list available"
    if not default:
        return "auto", "no devices reported"
    return "auto", f"jax default platform: {default[0].platform}"


def build_model(n_neurons: int) -> jtfne.Model:
    """Canonical benchmark network: cortical column, EIG emitter, proxy field."""
    cfg = (
        jtfne.configuration()
        .network(name="bench_protocol_v1", kind="cortical_column", n=n_neurons)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _build_sim(
    *,
    mode: str,
    seed: int,
    backend: str,
    record_fields: bool,
    record_weight_trace: bool,
    duration_ms: float,
    dt_ms: float,
) -> jtfne.Simulation:
    if mode == MODE_HDP:
        hdp_params = dict(HDP_PARAMS)
        hdp_params["record_weight_trace"] = record_weight_trace
        rt = jtfne.RuntimeConfig(
            backend=backend, dtype=_DTYPE, enable_hdp=True, hdp_params=hdp_params
        )
    else:
        rt = jtfne.RuntimeConfig(backend=backend, dtype=_DTYPE)
    return jtfne.simulation(
        duration_ms=duration_ms, dt_ms=dt_ms, seed=seed,
        record_fields=record_fields, runtime=rt,
    )


def _run_once(model: jtfne.Model, sim: jtfne.Simulation, *, sync: bool) -> tuple[float, Any]:
    """One simulation. sync=True (measured) performs jax.block_until_ready on
    every result array; sync=False (warmup) skips the sync."""
    t0 = time.perf_counter()
    sig = jtfne.simulate(model, sim=sim)
    if sync:
        leaves: list[Any] = [sig.V_m, sig.spikes]
        if sig.sources is not None:
            leaves.append(sig.sources)
        if sig.field is not None:
            leaves.append(sig.field)
        jax.tree.map(jax.block_until_ready, leaves)
    elapsed = time.perf_counter() - t0
    return elapsed, sig


def _shape_summary(sig: Any) -> dict[str, Any]:
    summary = {
        "V_m": list(sig.V_m.shape),
        "spikes": list(sig.spikes.shape),
        "sources": list(sig.sources.shape) if sig.sources is not None else None,
        "record_field": sig.field is not None,
    }
    if sig.field is not None:
        def _leaf_shape(x: Any) -> list[int]:
            try:
                return list(x.shape)
            except AttributeError:  # pragma: no cover - non-array leaf
                return None
        summary["field_leaves"] = jax.tree.map(_leaf_shape, sig.field)
    return summary


def _finite_summary(sig: Any) -> tuple[dict[str, bool], Optional[str]]:
    """Return (per-array finiteness, first failure description or None)."""
    checks: list[tuple[str, Any]] = [("V_m", sig.V_m), ("spikes", sig.spikes)]
    if sig.sources is not None:
        checks.append(("sources", sig.sources))
    out: dict[str, bool] = {}
    for name, arr in checks:
        out[name] = bool(jnp.all(jnp.isfinite(arr)).item())
    failure = next((name for name, ok in out.items() if not ok), None)
    return out, failure


def measure_mode(
    *,
    mode: str,
    model: jtfne.Model,
    n_neurons: int,
    duration_ms: float,
    dt_ms: float,
    seed: int,
    backend: str,
    warmup: int,
    runs: int,
    record_fields: bool,
    record_weight_trace: bool,
) -> dict[str, Any]:
    """Return per-mode measurement data. Warmup runs are excluded. Non-finite
    measured output fails loudly (RuntimeError), per host protocol."""
    sim = _build_sim(
        mode=mode,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=seed,
        backend=backend,
        record_fields=record_fields,
        record_weight_trace=record_weight_trace,
    )
    for _ in range(max(warmup, 0)):
        _run_once(model, sim, sync=False)

    measured: list[float] = []
    shape: Optional[dict[str, Any]] = None
    finite: Optional[dict[str, bool]] = None
    for _ in range(max(runs, 0)):
        elapsed, sig = _run_once(model, sim, sync=True)
        measured.append(elapsed)
        shape = _shape_summary(sig)
        finite, failure = _finite_summary(sig)
        if failure is not None:
            raise RuntimeError(f"{mode}: non-finite {failure} in measured run")

    out: dict[str, Any] = {
        "mode": mode,
        "warmup_runs": warmup,
        "measured_runs": len(measured),
        "timings_seconds": measured,
        "shape_summary": shape,
        "finite_summary": finite,
    }
    if measured:
        out["median_seconds"] = statistics.median(measured)
        out["min_seconds"] = min(measured)
        out["max_seconds"] = max(measured)
    else:
        out["median_seconds"] = None
        out["min_seconds"] = None
        out["max_seconds"] = None
    return out


def gpu_skip_report() -> dict[str, Any]:
    """Structured SKIP result; no CPU fallback per protocol."""
    kind = ""
    try:
        devices = jax.devices()
        kind = f"default: {devices[0].platform}" if devices else "none"
    except Exception:  # pragma: no cover - exotic platforms
        kind = "unreported"
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_level": CLAIM_LEVEL,
        "status": "skipped",
        "skip_reason": "gpu-unavailable: no GPU devices; no CPU fallback per protocol",
        "backend_requested": "gpu",
        "backend_resolved": None,
        "device_note": kind,
        "jax_version": getattr(jax, "__version__", "unknown"),
        "warmup_runs": 0,
        "measured_runs": 0,
        "modes": {},
    }


def _device_summary(backend: str) -> dict[str, Any]:
    try:
        devices = jax.devices(backend if backend != "cpu" else "cpu")
    except Exception:  # pragma: no cover
        devices = []
    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "devices": [f"{d.platform}:{d.id}" for d in devices],
    }


def _resolve_jit(n_neurons: int, n_steps: int) -> bool:
    """Mirror the runtime's own auto rule without constructing per-sim state."""
    return jtfne.RuntimeConfig(dtype=_DTYPE).resolve_jit(n_steps, n_neurons)


def build_report(
    *,
    args: argparse.Namespace,
    backend_resolved: str,
    backend_note: str,
    mode_results: dict[str, Any],
    n_steps: int,
) -> dict[str, Any]:
    preset = PRESETS[args.preset]
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_level": CLAIM_LEVEL,
        "status": "ok",
        "backend_requested": args.backend,
        "backend_resolved": backend_resolved,
        "backend_note": backend_note,
        "device": _device_summary(backend_resolved),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "jaxlib_version": getattr(sys.modules.get("jaxlib", None), "__version__", "unknown"),
        "dtype": _DTYPE,
        "jit_policy_resolved": _resolve_jit(preset["n_neurons"], n_steps),
        "workload": {
            "preset": args.preset,
            "n_neurons": preset["n_neurons"],
            "duration_ms": preset["duration_ms"],
            "dt_ms": preset["dt_ms"],
            "n_steps": n_steps,
            "seed": args.seed,
        },
        "flags": {
            "runs": args.runs,
            "warmup": args.warmup,
            "mode": args.mode,
            "record_fields": args.record_fields,
            "record_weight_trace": args.record_weight_trace,
        },
        "modes": mode_results,
    }


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}s" if v is not None else "n/a"


def _write_report(report: dict[str, Any], path: Optional[str]) -> None:
    if path is None:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"report -> {path}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    try:
        backend_resolved, backend_note = resolve_backend(args.backend)
    except GpuUnavailable:
        report = gpu_skip_report()
        _write_report(report, args.json_out)
        print(f"SKIP: GPU backend requested but unavailable; no CPU fallback (exit 0).")
        return 0

    preset = PRESETS[args.preset]
    n_neurons = preset["n_neurons"]
    duration_ms = preset["duration_ms"]
    dt_ms = preset["dt_ms"]
    n_steps = int(round(duration_ms / dt_ms))

    model = build_model(n_neurons)
    record_fields = args.record_fields == "on"
    record_weight_trace = args.record_weight_trace == "on"

    modes = MODES if args.mode == "both" else (args.mode,)
    mode_results: dict[str, Any] = {}
    for mode in modes:
        mode_results[mode] = measure_mode(
            mode=mode,
            model=model,
            n_neurons=n_neurons,
            duration_ms=duration_ms,
            dt_ms=dt_ms,
            seed=args.seed,
            backend=backend_resolved,
            warmup=args.warmup,
            runs=args.runs,
            record_fields=record_fields,
            record_weight_trace=record_weight_trace,
        )

    report = build_report(
        args=args,
        backend_resolved=backend_resolved,
        backend_note=backend_note,
        mode_results=mode_results,
        n_steps=n_steps,
    )
    _write_report(report, args.json_out)

    meta = report["workload"]
    print(
        f"[benchmark_protocol_v1] backend={backend_resolved} "
        f"N={meta['n_neurons']} T={meta['duration_ms']}ms dt={meta['dt_ms']}ms "
        f"jit_policy={report['jit_policy_resolved']}"
    )
    for mode, data in mode_results.items():
        print(
            f"  {mode:9s} runs={data['measured_runs']} "
            f"median={_fmt(data.get('median_seconds'))} "
            f"min={_fmt(data.get('min_seconds'))} max={_fmt(data.get('max_seconds'))}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())