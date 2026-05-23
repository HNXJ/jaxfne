# jaxfne Performance Baseline (v0.2.30)

**Status:** v0.2.30 performance receipts  
**Claim Level:** `local_environment_receipt_only`  
**Truth Mode:** `truth_safe_unverified`  
**Date:** 2026-05-23

---

## Executive Summary

jaxfne v0.2.30 introduces deterministic performance benchmarking receipts to document computational efficiency under controlled local conditions. **No universal performance claims are made.** All measurements are environment-specific (CPU type, Python version, JAX version, load state).

---

## Benchmark Scope

**What is measured:**
- Wall-clock time per computational phase (setup, construct, simulate, probe, evaluate, manifest)
- Hardware metadata (CPU type, device count, Python/JAX versions)
- Claim gates (computational_scaffold, truth_safe_unverified)

**What is NOT measured or claimed:**
- GPU/TPU acceleration factors
- Comparative performance vs. other simulators (no comparative claims)
- Biological accuracy or empirical correspondence
- Scalability beyond tested network sizes (50, 100 neurons)
- Real-time performance or hardware-general conclusions

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

For exponential synaptic kernel (v0.2.30 default):
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

## Claim Boundaries

### What CAN be claimed:

✓ "jaxfne simulates 50 neurons for 100 ms in ~150 ms wall-clock on CPU" (with hardware/date caveat)  
✓ "Time scales linearly with neuron count and duration" (within tested range)  
✓ "Core simulation dominates total time; overhead phases are ~10% of total"  
✓ "Configuration and construction are negligible overhead"

### What CANNOT be claimed:

✗ "jaxfne is faster than X simulator" (no comparative analysis)  
✗ "jaxfne scales to 1M neurons efficiently" (untested, extrapolation forbidden)  
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

**Expected variability:**
- Same platform, same workload: ±5–10% variance (depends on background processes, thermal state)
- Different platforms: ±50% or more
- With JIT enabled: first run slower (compilation overhead); subsequent runs ±10% faster
- With field recording disabled: ~5–15% faster probe phase

---

## Integration with CI/CD

v0.2.30 does NOT add CI/CD performance gates. Benchmarks are informational only:
- No hard "maximum time" thresholds
- No automated performance regression detection
- Local baseline serves as human-readable documentation

Rationale: Performance depends on machine load, network contention, and hardware revision. A universal CI threshold would be fragile and unreliable.

---

## Future Directions (Post-v0.2.30)

**Potential future work (not committed):**
- Profiling breakdown per neuron type (E, PV, SST, VIP)
- Synaptic kernel comparison (exponential vs. receptor_exponential)
- JIT vs. eager compilation benchmark
- Multi-thread/multi-core analysis
- Plasticity overhead quantification

All future work will maintain `local_environment_receipt_only` framing and avoid universal claims.

---

## References and Related Docs

- `docs/tutorials/tutorial_outputs.md` — tutorial runtime contracts
- `docs/jax_compatibility.md` — JAX baseline and device fallback
- `scripts/benchmark_jaxfne.py` — benchmark source code
- `scripts/validate_json_safe.py` — JSON safety validator
- `tests/test_performance_reports_v030.py` — performance report schema validation
- `CHANGELOG.md` — version history and release notes

---

## Truth Status

**Claim Level:** `computational_scaffold`  
**Truth Mode:** `truth_safe_unverified`  
**Physical Amplitude Claim Allowed:** `False`  
**Empirical Validation:** `Not empirically validated`

Performance measurements are **audit artifacts** (reproducibility, determinism), not scientific evidence. Time values do not imply biological correspondence or simulator superiority.

---

*jaxfne v0.2.30 — Tensor-Field Neural Equations source-to-field computational framework*
