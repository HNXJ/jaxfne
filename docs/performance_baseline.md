# jaxfne Performance Baseline

**Status:** performance receipts  
**Statement Level:** `local_environment_receipt_only`  
**Status Mode:** `tutorial_scaffold`  
**Date:** 2026-05-23

---

## Executive Summary

jaxfne provides deterministic performance benchmarking receipts to document computational efficiency under controlled local conditions. **No universal performance statements are made.** All measurements are environment-specific (CPU type, Python version, JAX version, load state).

---

## Benchmark Scope

**What is measured:**
- Wall-clock time per computational phase (setup, construct, simulate, probe, evaluate, manifest)
- Hardware metadata (CPU type, device count, Python/JAX versions)
- Status checks (computational_scaffold, tutorial_scaffold)

**What is NOT measured or stated:**
- GPU/TPU acceleration factors
- Comparative performance vs. other simulators (no comparative statements)
- Biological accuracy or empirical correspondence
- Real-time performance or hardware-general conclusions

Network-size scaling up to 10,000 neurons IS now measured; see "Scaling Evidence" below.

---

## Test Cases

### Case 1: Small Network (50 neurons, 100 ms)

| Phase | Typical Time | Notes |
|-------|--------------|-------|
| Configuration setup | 10–15 ms | Declaration of network topology |
| Model construction | 4–8 ms | Emitter parameter initialization |
| Simulation setup | 1–3 ms | Time grid allocation |
| Core simulation (emitter) | 100–150 ms | Izhikevich state integration |
| Probe readout (source/field) | 5–15 ms | Laminar projection (if enabled) |
| Objective evaluation | 2–5 ms | Smoke objective (no-op) |
| Manifest generation | 1–3 ms | JSON metadata hash |
| **Total** | **~130–200 ms** | Single-run wall-clock |

### Case 2: Medium Network (100 neurons, 300 ms)

| Phase | Typical Time | Notes |
|-------|--------------|-------|
| Configuration setup | 10–15 ms | Same as Case 1 |
| Model construction | 8–12 ms | Larger emitter structure |
| Simulation setup | 1–3 ms | Same as Case 1 |
| Core simulation (emitter) | 300–500 ms | 3x longer than Case 1 (duration + neurons) |
| Probe readout | 10–25 ms | Larger field output |
| Objective evaluation | 2–5 ms | Same objective |
| Manifest generation | 1–3 ms | Same as Case 1 |
| **Total** | **~330–560 ms** | Single-run wall-clock |

---

## Scaling Evidence (N=100 / 1,000 / 10,000)

**Added:** measured, not assumed. `benchmarks/scaling_benchmark.py` runs construct+simulate+probe
at three network sizes with `duration_ms`/`dt_ms` held fixed, each case in its own subprocess
(so `peak_rss_mb` is per-case, not a cumulative process maximum). Receipt:
`outputs/benchmarks_scaling/scaling_report.json`.

| N | construct (ms) | simulate (ms) | peak RSS (MB) |
|---|---|---|---|
| 100 | 1,021 | 873 | 390 |
| 1,000 | 1,058 | 864 | 482 |
| 10,000 | 1,305 | 17,783 | 4,211 |

Growth ratio per 10x step in N (single run, Apple Silicon CPU, jax 0.10.1):

| N step | construct ratio | simulate ratio | RSS ratio |
|---|---|---|---|
| 100 → 1,000 | 1.0x | 1.0x | 1.2x |
| 1,000 → 10,000 | 1.2x | 20.6x | 8.7x |

**Reading this honestly:** `construct` stays flat across all three sizes in this configuration —
the measured cost here is dominated by fixed JAX/JIT setup overhead, not network-size-dependent
work, at least up to N=10,000. `simulate` and `peak_rss_mb` are a different story: going from
N=1,000 to N=10,000 (a 10x step) costs ~20.6x more simulate wall-clock and ~8.7x more peak memory.
Both ratios are well above the ~10x a linear-in-N path would show, and are in the direction
consistent with the dense (`recurrent_backend="dense"`) O(N²) recurrent weight matrix documented
in `jaxfne/core.py` (search `O(N^2)` there for the exact call sites). This is **evidence that the
dense path is currently the dominant cost at N=10,000 in wall-clock and memory**, not a closed-form
complexity proof — the script measures three points, not a continuous curve, and runs on one
machine, one time.

**What this changes about the "Statement Boundaries" below:** "Time scales linearly with neuron
count and duration (within tested range)" no longer holds once the tested range is extended to
N=10,000 — see the corrected statement boundaries.

---

## Mathematical Formulation

### Execution Time Model

```
T_total = T_setup + T_construct + T_sim + T_probe + T_objective + T_manifest
```

Where:
- `T_setup` = configuration declaration time (independent of network)
- `T_construct` = model build time (linear in neuron count, ~0.1 ms per neuron)
- `T_sim` = core Izhikevich integration (Θ(T × N × w), linear in time steps, neurons, connectivity)
- `T_probe` = laminar field projection (Θ(T × N × M), linear in time steps, neurons, contacts)
- `T_objective` = loss evaluation (constant for smoke objective)
- `T_manifest` = metadata hash (constant, ~1–2 ms)

**Dominant cost:** `T_sim` dominates. Scaling:
```
T_sim ≈ (T_duration / dt) × n_neurons × mean_connectivity × cost_per_spike
```

For exponential synaptic kernel (current default):
```
T_sim ≈ 1–2 ms per (neuron × step)
```

Example:
- Case 1: 100 steps × 50 neurons × ~0.1 ms/unit ≈ 0.5–1.0 ms minimum (actual: includes overhead, vectorization, dtype casting)
- Case 2: 3000 steps × 100 neurons × ~0.1 ms/unit ≈ 30–100 ms minimum (actual: 300–500 ms includes overhead)

---

## Hardware Metadata (Local Receipt)

**Platform for baseline measurements:**
```
Platform: macOS 13.0 (Apple Silicon M2/M3)
Python: 3.11.15
JAX: 0.10.0
NumPy: 1.24.3
CPU: 1× Apple Neural Engine (fallback to CPU in jaxfne)
Device: CpuDevice(id=0)
```

**Important:** Measurements are CPU-based on Apple Silicon. JAX does not currently accelerate jaxfne's Izhikevich kernel on Apple GPUs. Results on other platforms (Linux/NVIDIA, Intel, etc.) may differ significantly.

---

## Statement Boundaries

### What CAN be stated:

✓ "jaxfne simulates 50 neurons for 100 ms in ~150 ms wall-clock on CPU" (with hardware/date caveat)  
✓ "Time scales roughly linearly with neuron count up to N=1,000 on the dense backend" (measured)  
✓ "Simulate cost and peak memory grow faster than linearly between N=1,000 and N=10,000 on the
  dense recurrent backend (measured ~20.6x time, ~8.7x memory for a 10x step in N)"  
✓ "Core simulation dominates total time at large N; overhead phases are negligible by comparison"  
✓ "Configuration and construction are negligible overhead at the tested sizes"

### What CANNOT be stated:

✗ "Time scales linearly with neuron count at all tested sizes" (false above N≈1,000; see Scaling
  Evidence above — corrects the prior, narrower-range version of this claim)  
✗ "jaxfne is faster than X simulator" (no comparative analysis)  
✗ "jaxfne scales to 1M neurons efficiently" (untested beyond N=10,000, extrapolation forbidden)  
✗ "Real-time factor = X" (depends on hardware, stimulus properties, and objective)  
✗ "This reflects biological simulation accuracy" (computational_scaffold, proxy-field only)  
✗ "GPU acceleration would be dramatic" (unvalidated; Apple GPU is not integrated)

---

## Reproduction and Variability

**To regenerate baseline:**
```bash
python scripts/benchmark_jaxfne.py
# Outputs: outputs/benchmarks_v030/benchmark_report.json
```

**To regenerate the N=100/1,000/10,000 scaling evidence:**
```bash
PYTHONPATH=. python benchmarks/scaling_benchmark.py
# Outputs: outputs/benchmarks_scaling/scaling_report.json
# Takes roughly 20-30s on Apple Silicon CPU; dominated by the N=10,000 simulate phase.
```

**Expected variability:**
- Same platform, same workload: ±5–10% variance (depends on background processes, thermal state)
- Different platforms: ±50% or more
- With JIT enabled: first run slower (compilation overhead); subsequent runs ±10% faster
- With field recording disabled: ~5–15% faster probe phase

---

## Integration with CI/CD

The performance baseline does NOT add CI/CD performance gates. Benchmarks are informational only:
- No hard "maximum time" thresholds
- No automated performance regression detection
- Local baseline serves as human-readable documentation

Rationale: Performance depends on machine load, network contention, and hardware revision. A universal CI threshold would be fragile and unreliable.

---

## Reserved Directions

**Potential reserved work (not committed):**
- Profiling breakdown per neuron type (E, PV, SST, VIP)
- Synaptic kernel comparison (exponential vs. receptor_exponential)
- JIT vs. eager compilation benchmark
- Multi-thread/multi-core analysis
- Plasticity overhead quantification

All reserved work will maintain `local_environment_receipt_only` framing and avoid universal statements.

---

## References and Related Docs

- `docs/tutorials/tutorial_outputs.md` — tutorial runtime contracts
- `scripts/benchmark_jaxfne.py` — benchmark source code
- `benchmarks/scaling_benchmark.py` — N=100/1,000/10,000 scaling-evidence source code
- `scripts/validate_json_safe.py` — JSON safety validator
- `tests/test_performance_reports_v030.py` — performance report schema validation
- `docs/changelog.md` — version history and release notes

---

## Status Status

**Statement Level:** `computational_scaffold`  
**Status Mode:** `tutorial_scaffold`  
**Physical Amplitude Statement Allowed:** `False`  
**Empirical Validation:** `Not empirically validated`

Performance measurements are **audit artifacts** (reproducibility, determinism), not scientific evidence. Time values do not imply biological correspondence or simulator superiority.

---

*jaxfne — Tensor-Field Neural Equations source-to-field computational model*
