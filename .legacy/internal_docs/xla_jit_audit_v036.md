# jaxfne XLA JIT Engineering Audit (v0.3.6)

This document provides technical design details and validation receipts for JAX compilation (XLA) on top of the jaxfne source-to-field simulation hot paths.

---

## 1. Kernels Tested & JIT Compilation

The core numerical kernels in `jaxfne` execute phenomenological spiking dynamics (Izhikevich models) paired with recurrent projection networks. We verified XLA compilation for:

* **`simulate_eig_izhikevich`:** Pure numerical kernel executing population-level neural update steps.
* **`jtfne.simulate` (Facade-level):** High-level user interface compiling the numerical loop transparently using `RuntimeConfig(jit=True)`.

### Compiler Optimization Mechanics
* **Loop Unrolling / scan:** JAX unrolls or compiles the main time-stepping loop via XLA, yielding major latency drops on long simulation durations.
* **Memory Parity:** All compiled paths run entirely inside device memory, minimizing CPU-GPU trace transfers until readout evaluation.

---

## 2. Static Arguments Design

To prevent excessive recompilation (which triggers significant overhead in interactive environments), the following parameters are treated as static or shape-defining variables:
* **`n_steps`:** Static integer representing the number of integration steps ($N = \text{duration} / dt$).
* **`dt_ms`:** Constant float step size.
* **`n_neurons` / Network Geometry:** Network dimension shapes are statically sized at construct/compilation boundary.

---

## 3. Eager/JIT Equivalence Receipt

A strict test suite (`tests/test_jit_equivalence_v036.py`) has been added to guarantee that the eager execution path and JIT-compiled paths are numerically identical:
* **Seed Determinism:** Validated that identical PRNG keys yield bit-wise identical voltages and spike matrices across eager and compiled modes.
* **Finite Output Check:** Ensured that outputs remain strictly finite (no `NaN` or `inf` values) under high-frequency recurrent drive.
* **Dtype Stability:** Verified that both eager and compiled paths adhere to `float32` precision boundaries on standard CPU operations.

---

## 4. Deferred Parallelization & Scaling Plans

### A. Deferred `vmap` Sweep Helper
* **Current status:** Single parameter sets are vectorized internally across neural population shapes.
* **Future scope:** Vectorize whole parameter grids (`vmap` over configuration arrays) to enable highly-parallel parameter sweeps directly on GPU/TPU accelerators.

### B. Deferred `pmap` / `pjit` Plan
* **Current status:** Not implemented. All workflows assume single-accelerator or single-core local execution.
* **Future scope:** Implement multi-device parallel distribution (`pmap`) for extremely large multi-area network populations (e.g., dual-hemisphere models).

### C. Benchmark Script Plan
* **Current status:** Performance profiles are generated via manual execution.
* **Future scope:** Deliver a robust command-line benchmarking script (`scripts/benchmark_simulation.py`) to systematically record CPU/GPU compile-time and run-time profiles.

---

## 5. Solver Roadmap Status
* **Boundary:** All projection and field solvers remain pure computational **proxies** (LFP-proxy, CSD-proxy, etc.). The mathematical model assumes direct projection of source currents onto sensor geometry templates.
* **Roadmap doctrine:** Implementing physical partial differential equation (PDE) solvers (e.g., full Maxwell/Poisson solvers) remains strictly **design-only / deferred** for future engineering milestones. No physical PDE claims are made by the current codebase.
