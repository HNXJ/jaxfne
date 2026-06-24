#!/usr/bin/env python
"""Construction/simulation scaling evidence: N=100 / 1000 / 10000.

Measures wall-clock time per pipeline phase and peak resident memory at
three network sizes, with the *same* duration_ms/dt_ms held fixed across
all three so the only varying factor is N. This is receipt evidence for
whether the dense (O(N^2)) recurrent-backend path's measured cost is
consistent with quadratic growth -- it does not optimize anything; it
just measures what is actually there.

Each case runs in its own subprocess so peak_rss_mb is isolated per case
(resource.getrusage's ru_maxrss is a running process-wide maximum, so an
in-process loop over cases would report a monotonically non-decreasing,
case-blended number instead of a real per-N delta).

Claim status: local_environment_receipt_only. Single-machine, single-run
timings; not a multi-trial statistical benchmark and not a universal
performance claim (see jaxfne/AGENTS.md claim-language rules).

Usage:
    PYTHONPATH=. python benchmarks/scaling_benchmark.py
"""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_CASE_WORKER_SRC = """
import json, platform, resource, time
import jax
import jaxfne as jtfne

n_neurons = {n_neurons}
duration_ms = {duration_ms}
dt_ms = {dt_ms}

timings = {{}}

t0 = time.perf_counter()
cfg = (
    jtfne.configuration()
    .network(
        name="scaling_net", kind="cortical_column", n=n_neurons,
        layers=["L1", "L2/3", "L4", "L5", "L6"],
        cell_types={{"E": 0.80, "PV": 0.10, "SST": 0.07, "VIP": 0.03}},
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
    .probe(name="laminar_probe", modes=["spikes", "V_m"])
)
timings["configuration_setup_ms"] = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
model = jtfne.construct(cfg)
timings["construct_ms"] = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=0)
signals = model.simulate(sim)
timings["simulate_ms"] = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
_ = model.probe(signals, modes=["spikes", "V_m"])
timings["probe_ms"] = (time.perf_counter() - t0) * 1000.0

raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_rss_mb = raw_rss / (1024.0 * 1024.0) if platform.system() == "Darwin" else raw_rss / 1024.0

print(json.dumps({{
    "n_neurons": n_neurons,
    "duration_ms": duration_ms,
    "dt_ms": dt_ms,
    "timings": timings,
    "total_ms": sum(timings.values()),
    "peak_rss_mb": peak_rss_mb,
}}))
"""


def _hardware_info() -> dict[str, Any]:
    import jax

    devices = jax.devices()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": str(jax.__version__),
        "device_kind": str(devices[0].device_kind) if devices else "none",
        "device_count": len(devices),
    }


def run_case_isolated(name: str, n_neurons: int, duration_ms: float = 200.0, dt_ms: float = 0.1) -> dict[str, Any]:
    """Run one scaling case in a fresh subprocess so peak RSS is per-case, not cumulative."""
    src = _CASE_WORKER_SRC.format(n_neurons=n_neurons, duration_ms=duration_ms, dt_ms=dt_ms)
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    result["case_name"] = name
    return result


def main() -> None:
    cases = [
        ("n100", 100),
        ("n1000", 1000),
        ("n10000", 10000),
    ]

    results = []
    for name, n in cases:
        print(f"Running {name} (N={n}) in isolated subprocess...", flush=True)
        r = run_case_isolated(name, n)
        results.append(r)
        print(f"  construct={r['timings']['construct_ms']:.0f}ms  "
              f"simulate={r['timings']['simulate_ms']:.0f}ms  "
              f"peak_rss={r['peak_rss_mb']:.0f}MB", flush=True)

    # Empirical growth ratios between consecutive N (10x each step).
    growth = []
    for prev, cur in zip(results[:-1], results[1:]):
        growth.append({
            "from": prev["case_name"],
            "to": cur["case_name"],
            "n_ratio": cur["n_neurons"] / prev["n_neurons"],
            "construct_ratio": cur["timings"]["construct_ms"] / max(prev["timings"]["construct_ms"], 1e-6),
            "simulate_ratio": cur["timings"]["simulate_ms"] / max(prev["timings"]["simulate_ms"], 1e-6),
            "rss_ratio": cur["peak_rss_mb"] / max(prev["peak_rss_mb"], 1e-6),
        })

    report = {
        "benchmark_series": "scaling_evidence_n100_1000_10000",
        "claim_level": "local_environment_receipt_only",
        "hardware_info": _hardware_info(),
        "results": results,
        "growth_ratios": growth,
        "notes": (
            "Single-run, single-machine wall-clock timings, one isolated subprocess "
            "per case so peak_rss_mb is a real per-N measurement (not a cumulative "
            "process-wide maximum). recurrent_backend is dense by default, so cost is "
            "expected to grow faster than linear in N. A ratio near 100x per 10x step "
            "in N is consistent with O(N^2) dense-weight-matrix behavior; a ratio near "
            "10x indicates near-linear/sparse-equivalent scaling. This script reports "
            "the measured ratio, it does not assume one."
        ),
    }

    output_dir = Path("outputs/benchmarks_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "scaling_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(f"\nGrowth ratios (per 10x step in N):")
    for g in growth:
        print(f"  {g['from']} -> {g['to']}: construct x{g['construct_ratio']:.1f}, "
              f"simulate x{g['simulate_ratio']:.1f}, rss x{g['rss_ratio']:.1f}")

    print(f"\nReport written to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
