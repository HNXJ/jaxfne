# v0.1.2 — Scan-backed Recurrent Paths: Performance Validation & Benchmark Harness

## Scope

v0.1.2 validates and documents the existing scan-backed recurrent computation paths in jaxfne. Both dense and edge-list recurrent backends use `jax.lax.scan` for deterministic, JAX-native time stepping.

This release adds:
- Explicit documentation that dense and edge-list paths use `lax.scan`
- Benchmark harness for CPU-safe performance measurement
- Tests ensuring scan-backed execution is preserved
- Metadata confirmation in signals and manifests

No kernel refactoring is required; the implementations are already scan-backed.

## Execution Invariants

**Dense path (`simulate_eig_izhikevich`):**
- Uses `jax.lax.scan` over time steps
- Supports optional drive schedules (stimulus injection)
- Preserves PRNG determinism via key splitting
- Records voltage, spikes, sources at each step

**Edge-list path (`simulate_edge_recurrent_izhikevich`):**
- Uses `jax.lax.scan` over time steps
- Aggregates synaptic inputs via `jax.ops.segment_sum`
- Preserves PRNG determinism via key splitting
- Records voltage, spikes, sources at each step

**Receptor-exponential path (`simulate_receptor_exponential_izhikevich`):**
- Uses `jax.lax.scan` over time steps
- Looks up receptor decay constants from registry
- Preserves PRNG determinism via key splitting
- Records voltage, spikes, sources at each step

All paths preserve:
- Output shapes: `(n_steps, n_neurons)`
- Truth metadata: `truth_safe_unverified`, `computational_scaffold`, `laminar_proxy_no_pde`
- Status metadata: `source_calibration_status`, `physical_amplitude_claim_allowed = false`
- Metadata reporting: `recurrent_backend`, `synaptic_kernel`

## Performance Benchmark Plan

**Measurements:**
1. Dense, n=50 neurons, duration=100 ms, dt=0.1 ms (1000 steps)
2. Dense, n=100 neurons, duration=1000 ms, dt=0.1 ms (10000 steps)
3. Edge-list, n=100 neurons, duration=1000 ms, dt=0.1 ms (10000 steps) — if affordable on CPU

**Metrics per run:**
- Backend name
- Network size (n_neurons)
- Simulation duration (ms)
- Timestep (ms)
- Total steps
- JIT enabled (yes/no)
- Wall time (seconds)
- Output shapes (spikes, voltage, sources)
- Metadata backend field

**Script:** `scripts/benchmark_scan_backends.py`
- CPU-safe (no GPU assumptions)
- JAX warmup before timing
- Results to stdout only (no file artifacts)
- JSON-structured output

## Metadata Invariants

**Runtime metadata to be confirmed present:**
```python
metadata = {
    "recurrent_backend": "dense" | "edge_list",
    "synaptic_kernel": "exponential" | "receptor_exponential",
    "time_integration_backend": "lax_scan",
    "truth_mode": "truth_safe_unverified",
    "field_solver_status": "laminar_proxy_no_pde",
    "physical_amplitude_claim_allowed": False,
}
```

If `time_integration_backend` is not already present, it will be added during this release.

## Numerical Parity

Dense, edge-list, and receptor-exponential paths are **not expected to be numerically identical** due to different aggregation strategies:

- Dense uses matrix multiplication: `W @ prev_spikes`
- Edge-list uses sparse segment_sum aggregation
- Receptor-exponential uses receptor-indexed state lookup + segment_sum

Validation confirms:
- Shapes match expected
- Deterministic for same seed
- Metadata is consistent across paths
- Truth gates are preserved

**No bitwise parity test is required** because architectural differences (dense vs. sparse) are intentional and produce valid but different trajectories.

## Deferred Work

- Multi-area simulation
- Conductance-based synapses
- NMDA Mg-block gating
- iCSD (current source density via implicit PDE)
- Physical field solver (PDE integration)
- Jaxley bridge activation
- New biological claims

## Validation Commands

```bash
cd /Users/hamednejat/workspace/main/jaxfne
source venv/bin/activate

# Syntax check
python -m compileall jaxfne tests examples scripts

# Full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q

# Benchmark
PYTHONPATH=. python scripts/benchmark_scan_backends.py

# Examples
PYTHONPATH=. python examples/00_minimal_column.py
PYTHONPATH=. python examples/01_source_field_manifest.py
PYTHONPATH=. python examples/02_omission_scaffold.py
PYTHONPATH=. python examples/03_objective_and_tune_smoke.py
PYTHONPATH=. python examples/04_blackbox_tuning_loop.py
PYTHONPATH=. python examples/05_dataset_bridge_manifest.py
PYTHONPATH=. python examples/06_edge_list_recurrent_backend.py

# Release rehearsal
./scripts/release_rehearsal.sh
```

## Scientific Framing

v0.1.2 improves confidence in the deterministic JAX execution paths for CPU-first spectrolaminar proxy workflows. Both dense and edge-list recurrent backends use `jax.lax.scan` for compilation, batching, and reproducibility. This release adds benchmark evidence and metadata confirmation.

All scientific status fields remain unchanged: `truth_safe_unverified`, `computational_scaffold`, `laminar_proxy_no_pde`, no calibrated physical claims.
