#!/usr/bin/env python
"""Performance benchmark script for jaxfne v0.2.30+.

Measures wall-clock time across construction, simulation, probing, and manifest
generation phases. Outputs JSON-safe report with hardware metadata and claim gates.

Claim status: local_environment_receipt_only, no universal performance claims.
truth_mode: truth_safe_unverified
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp

import jaxfne as jtfne


@dataclass(frozen=True)
class BenchmarkCase:
    """Benchmark configuration."""
    name: str
    n_neurons: int
    duration_ms: float
    dt_ms: float = 0.1


@dataclass(frozen=True)
class TimingBlock:
    """Wall-clock timing for one phase."""
    phase: str
    elapsed_ms: float


@dataclass
class BenchmarkResult:
    """Complete benchmark report with metadata."""
    case_name: str
    timings: list[dict[str, Any]]
    total_ms: float
    hardware_info: dict[str, Any]
    jaxfne_version: str
    claim_level: str
    truth_mode: str
    local_environment_receipt_only: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dict."""
        return {
            "case_name": self.case_name,
            "timings": self.timings,
            "total_ms": float(self.total_ms),
            "hardware_info": self.hardware_info,
            "jaxfne_version": str(self.jaxfne_version),
            "claim_level": str(self.claim_level),
            "truth_mode": str(self.truth_mode),
            "local_environment_receipt_only": bool(self.local_environment_receipt_only),
            "notes": str(self.notes),
        }


def get_hardware_info() -> dict[str, Any]:
    """Collect hardware and runtime metadata."""
    devices = jax.devices()
    device_info = []
    for dev in devices:
        device_info.append({
            "device_kind": str(dev.device_kind),
            "device_id": int(dev.id),
        })

    import platform
    import numpy
    try:
        import jaxlib
        jaxlib_version = str(jaxlib.__version__)
    except (AttributeError, ImportError):
        jaxlib_version = "unknown"

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "jax_version": str(jax.__version__),
        "jaxlib_version": jaxlib_version,
        "numpy_version": str(numpy.__version__),
        "devices": device_info,
        "device_count": len(devices),
        "default_device": str(devices[0]) if devices else "none",
    }


def run_benchmark_case(case: BenchmarkCase) -> BenchmarkResult:
    """Run one benchmark case with timing instrumentation."""
    timings: list[dict[str, Any]] = []
    total_start = time.time()

    # Phase 1: Configuration setup
    t0 = time.time()
    cfg = jtfne.configuration()
    cfg = cfg.network(
        name="benchmark_net",
        kind="cortical_column",
        n=case.n_neurons,
        layers=["L1", "L2/3", "L4", "L5", "L6"],
        cell_types={"E": 0.80, "PV": 0.10, "SST": 0.07, "VIP": 0.03},
    )
    cfg = cfg.emitter(family="izhikevich", preset="cortical_eig")
    cfg = cfg.field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy", gauge="mean_zero")
    cfg = cfg.probe(name="laminar_probe", modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"])
    cfg = cfg.update_metadata(
        truth_mode="truth_safe_unverified",
        claim_level="computational_scaffold",
    )
    timings.append({"phase": "configuration_setup", "elapsed_ms": float((time.time() - t0) * 1000.0)})

    # Phase 2: Model construction
    t0 = time.time()
    model = jtfne.construct(cfg)
    timings.append({"phase": "model_construction", "elapsed_ms": float((time.time() - t0) * 1000.0)})

    # Phase 3: Simulation setup
    t0 = time.time()
    sim = jtfne.simulation(
        duration_ms=case.duration_ms,
        dt_ms=case.dt_ms,
        plasticity=0.0,
        seed=0,
    )
    timings.append({"phase": "simulation_setup", "elapsed_ms": float((time.time() - t0) * 1000.0)})

    # Phase 4: Core simulation (emitter, sources)
    t0 = time.time()
    signals = model.simulate(sim)
    timings.append({"phase": "simulate_core", "elapsed_ms": float((time.time() - t0) * 1000.0)})

    # Phase 5: Probing/readout
    t0 = time.time()
    readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    timings.append({"phase": "probe_readout", "elapsed_ms": float((time.time() - t0) * 1000.0)})

    # Phase 6: Objective evaluation
    t0 = time.time()
    report = model.evaluate(signals, objective="smoke")
    timings.append({"phase": "evaluate_objective", "elapsed_ms": float((time.time() - t0) * 1000.0)})

    # Phase 7: Manifest/receipt generation
    t0 = time.time()
    manifest = model.manifest(signals)
    timings.append({"phase": "manifest_generation", "elapsed_ms": float((time.time() - t0) * 1000.0)})

    total_elapsed_ms = (time.time() - total_start) * 1000.0

    return BenchmarkResult(
        case_name=case.name,
        timings=timings,
        total_ms=total_elapsed_ms,
        hardware_info=get_hardware_info(),
        jaxfne_version=jtfne.__version__,
        claim_level="computational_scaffold",
        truth_mode="truth_safe_unverified",
        local_environment_receipt_only=True,
        notes="Benchmark conducted on local CPU. No universal performance claim. Times include Python/JAX overhead and are environment-specific.",
    )


def main():
    """Run all benchmark cases and write JSON report."""
    output_dir = Path("outputs/benchmarks_v030")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define benchmark cases
    cases = [
        BenchmarkCase(name="small_50n_100ms", n_neurons=50, duration_ms=100.0, dt_ms=0.1),
        BenchmarkCase(name="medium_100n_300ms", n_neurons=100, duration_ms=300.0, dt_ms=0.1),
    ]

    results = []
    for case in cases:
        print(f"Running benchmark: {case.name}...", flush=True)
        result = run_benchmark_case(case)
        results.append(result)
        print(f"  {case.name}: {result.total_ms:.1f} ms total", flush=True)

    # Write JSON report
    report = {
        "benchmark_series": "v0.2.30_performance_baseline",
        "results": [r.to_dict() for r in results],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_file = output_dir / "benchmark_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(f"\nBenchmark complete. Report: {output_file}")


if __name__ == "__main__":
    main()
