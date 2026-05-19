#!/usr/bin/env python3
"""
Benchmark script for scan-backed recurrent backends.

Measures wall time for dense and edge-list simulations at various scales.
CPU-safe; JAX warmup included.

Usage:
    PYTHONPATH=. python scripts/benchmark_scan_backends.py
"""

import time
import json
import sys

import jaxfne as jtfne


def benchmark_backend(backend_name, n, duration_ms, dt_ms, jit=True):
    """
    Run a single benchmark and return wall time + metadata.

    Args:
        backend_name: "dense" or "edge_list"
        n: number of neurons
        duration_ms: simulation duration in ms
        dt_ms: timestep in ms
        jit: whether to use JIT compilation

    Returns:
        dict with backend, n, duration_ms, dt_ms, n_steps, elapsed_seconds, shapes
    """

    # Configure model
    cfg = (
        jtfne.configuration()
        .network(name="bench", kind="cortical_column", n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )

    model = jtfne.construct(cfg)

    # Simulation parameters
    n_steps = int(duration_ms / dt_ms)
    rt = jtfne.runtime(recurrent_backend=backend_name, seed=0)
    sim = jtfne.simulation(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        seed=0,
        runtime=rt
    )

    # Warmup run (discarded)
    try:
        _ = model.simulate(sim)
    except Exception as e:
        return {
            "backend": backend_name,
            "n": n,
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "n_steps": n_steps,
            "jit": jit,
            "elapsed_seconds": None,
            "error": str(e),
        }

    # Timed run
    start = time.perf_counter()
    signals = model.simulate(sim)
    elapsed = time.perf_counter() - start

    # Extract shapes
    spikes_shape = tuple(signals.spikes.shape) if hasattr(signals.spikes, 'shape') else None
    voltage_shape = tuple(signals.V_m.shape) if hasattr(signals.V_m, 'shape') else None

    # Extract metadata
    used_backend = signals.metadata.get("recurrent_backend", "unknown")

    return {
        "backend": backend_name,
        "used_backend": used_backend,
        "n": n,
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "n_steps": n_steps,
        "jit": jit,
        "elapsed_seconds": round(elapsed, 4),
        "spikes_shape": spikes_shape,
        "voltage_shape": voltage_shape,
    }


def main():
    print("=== jaxfne scan-backend performance benchmark ===\n")

    benchmarks = [
        ("dense", 50, 100.0, 0.1, True),
        ("dense", 100, 1000.0, 0.1, True),
        ("edge_list", 100, 1000.0, 0.1, True),
    ]

    results = []
    for backend, n, duration, dt, jit_flag in benchmarks:
        print(f"Running: backend={backend}, n={n}, duration={duration} ms, dt={dt} ms, jit={jit_flag}...")
        result = benchmark_backend(backend, n, duration, dt, jit=jit_flag)
        results.append(result)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  elapsed: {result['elapsed_seconds']} s")
            print(f"  shapes: spikes={result['spikes_shape']}, voltage={result['voltage_shape']}")
        print()

    # Print summary table
    print("=== Summary ===\n")
    print(f"{'Backend':<12} {'N':<6} {'Duration':<12} {'Steps':<8} {'Time (s)':<10} {'Metadata':<20}")
    print("-" * 80)
    for r in results:
        if "error" not in r:
            print(
                f"{r['backend']:<12} {r['n']:<6} {r['duration_ms']:<12} {r['n_steps']:<8} "
                f"{r['elapsed_seconds']:<10} {r.get('used_backend', '?'):<20}"
            )
        else:
            print(f"{r['backend']:<12} {r['n']:<6} ERROR: {r['error']}")

    print("\n=== Metadata ===\n")
    print(json.dumps(results, indent=2))

    # Exit code: success if all benchmarks completed without error
    has_error = any("error" in r for r in results)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
