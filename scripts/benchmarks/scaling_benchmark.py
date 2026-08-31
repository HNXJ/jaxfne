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
    PYTHONPATH=. python scripts/benchmarks/scaling_benchmark.py
"""

from __future__ import annotations

import json
import platform
try:
    import resource  # POSIX; Windows uses GetProcessMemoryInfo in workers
except ImportError:  # pragma: no cover
    resource = None
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_CASE_WORKER_SRC = """
import json, platform, time
try:
    import resource
except ImportError:
    resource = None
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

if platform.system() == "Windows":
    import ctypes
    import ctypes.wintypes as _wt

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", _wt.DWORD), ("PageFaultCount", _wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    _pmc = _PMC()
    _pmc.cb = ctypes.sizeof(_pmc)
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [_wt.HANDLE, ctypes.POINTER(_PMC), _wt.DWORD]
    if psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(_pmc), _pmc.cb):
        peak_rss_mb = _pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
    else:
        peak_rss_mb = None
elif resource is not None and platform.system() == "Darwin":
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
elif resource is not None:
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
else:
    peak_rss_mb = None

print(json.dumps({{
    "n_neurons": n_neurons,
    "duration_ms": duration_ms,
    "dt_ms": dt_ms,
    "timings": timings,
    "total_ms": sum(timings.values()),
    "peak_rss_mb": peak_rss_mb,
}}))
"""


_SPARSE_CASE_WORKER_SRC = """
import json, platform, time
try:
    import resource
except ImportError:
    resource = None
import jax
import jaxfne as jtfne

n_neurons = {n_neurons}
duration_ms = {duration_ms}
dt_ms = {dt_ms}
p_connect = {p_connect}

timings = {{}}

t0 = time.perf_counter()
cfg = (
    jtfne.configuration()
    .network(name="scaling_net", n=n_neurons, cell_types={{"E": 0.80, "PV": 0.10, "SST": 0.07, "VIP": 0.03}})
    .uniform3d()
    .connectivity(p_connect=p_connect)
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

if platform.system() == "Windows":
    import ctypes
    import ctypes.wintypes as _wt

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", _wt.DWORD), ("PageFaultCount", _wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    _pmc = _PMC()
    _pmc.cb = ctypes.sizeof(_pmc)
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [_wt.HANDLE, ctypes.POINTER(_PMC), _wt.DWORD]
    if psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(_pmc), _pmc.cb):
        peak_rss_mb = _pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
    else:
        peak_rss_mb = None
elif resource is not None and platform.system() == "Darwin":
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
elif resource is not None:
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
else:
    peak_rss_mb = None

print(json.dumps({{
    "n_neurons": n_neurons,
    "p_connect": p_connect,
    "duration_ms": duration_ms,
    "dt_ms": dt_ms,
    "timings": timings,
    "total_ms": sum(timings.values()),
    "peak_rss_mb": peak_rss_mb,
}}))
"""


def run_sparse_case_isolated(
    name: str, n_neurons: int, p_connect: float,
    duration_ms: float = 200.0, dt_ms: float = 0.1,
) -> dict[str, Any]:
    """Run one sparse-connectivity (``p_connect<1``) scaling case in a fresh subprocess.

    A dense within-area matrix at N=100,000 would require ~40GB for ``W`` alone
    (float32, ``n*n*4`` bytes) -- infeasible on most machines. The recommended
    lever at that scale (jaxfne-harden rule 3, ``_DENSE_CONNECTIVITY_WARN_N``) is
    a sparse ``p_connect<1`` request, which routes through
    ``_apply_connectivity``'s sparse-direct escape (edge list built directly,
    never materializing the (n,n) matrix) -- but ONLY when the ``Configuration``
    also carries ``columns``/``layer_cell_types``/``uniform_3d`` metadata
    (``_construct_build_network``'s routing condition); this worker calls
    ``.uniform3d()`` for exactly that reason. A ``.network(kind=..., layers=[...])``
    call alone does NOT set that metadata (``layers`` lives in the per-network
    dict, not ``cfg.metadata``) and silently falls through to
    ``make_eig_network``'s always-dense builder instead, making any
    ``connectivity(p_connect=...)`` call inert. Confirmed directly 2026-07-21:
    the pre-``.uniform3d()`` version of this worker ran ~370s and peaked at
    ~50GB RSS at N=100,000, p_connect=0.0005 -- identical cost to the fully
    dense path, because that is what actually ran. Fixed in two places: this
    worker now uses the metadata-routing recipe (11s construct, 1.6s simulate,
    1.15GB peak RSS at the same N/p_connect -- verified), and
    ``_construct_build_network``'s dense fallback (``jaxfne/_construct_core.py``)
    now warns (matching ``_DENSE_CONNECTIVITY_WARN_N``) whenever a large-N
    config falls through to it, so a future recipe mistake is surfaced rather
    than silently OOMing.

    This is a DIFFERENT code path from ``run_case_isolated``'s dense
    all-to-all cases above -- not comparable via growth ratios against them
    (that would understate the dense path's real O(N^2) cost by conflating
    it with the sparse path's near-linear one). ``p_connect`` should be
    chosen to keep mean out-degree bounded (``p_connect = target_degree /
    n_neurons``) -- an unbounded choice like a flat 0.01 at N=100,000 still
    yields ~1e8 edges (mean degree 1000), a large, slow, memory-heavy graph
    despite being "sparse" in the p_connect<1 sense.
    """
    src = _SPARSE_CASE_WORKER_SRC.format(
        n_neurons=n_neurons, duration_ms=duration_ms, dt_ms=dt_ms, p_connect=p_connect,
    )
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

    # Separate sparse-path case at N=100,000 -- the dense path above is not run
    # at this N (a dense (n,n) float32 W would need ~40GB), so this exercises
    # the actually-recommended lever at scale (p_connect<1, bounded mean
    # out-degree) instead of extending the same dense growth-ratio series.
    print("\nRunning n100000_sparse (N=100000, p_connect bounded to mean "
          "degree ~50) in isolated subprocess...", flush=True)
    target_mean_degree = 50.0
    n_sparse = 100_000
    p_connect_sparse = target_mean_degree / n_sparse
    sparse_result = run_sparse_case_isolated(
        "n100000_sparse", n_neurons=n_sparse, p_connect=p_connect_sparse,
        duration_ms=50.0, dt_ms=0.5,
    )
    print(f"  construct={sparse_result['timings']['construct_ms']:.0f}ms  "
          f"simulate={sparse_result['timings']['simulate_ms']:.0f}ms  "
          f"peak_rss={sparse_result['peak_rss_mb']:.0f}MB", flush=True)
    report["sparse_scaling_case"] = {
        "benchmark_series": "sparse_scaling_evidence_n100000_p_connect_bounded_degree",
        "claim_level": "local_environment_receipt_only",
        "result": sparse_result,
        "notes": (
            "Distinct from the dense growth_ratios series above -- not directly "
            "comparable (different code path: sparse-direct edge-list escape, "
            "not the dense (n,n) matrix). p_connect chosen as "
            f"target_mean_degree({target_mean_degree})/n to keep edge count "
            "bounded (an unscaled flat p_connect at this N would itself be a "
            "very large, slow graph despite being 'sparse' in the p_connect<1 "
            "sense). This case's own purpose: confirm the sparse-direct escape "
            "keeps memory/time bounded at N=100,000, where the dense path is "
            "infeasible."
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
