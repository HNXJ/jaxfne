# v0.2.0 Theoretical Validation and Source-Field Admissibility

**Status:** v0.2.0 architectural contract document  
**Date:** 2026-05-19  
**Version:** 0.2.0 (in development)  
**Scope:** Theoretical validation, source bookkeeping, field admissibility, computational correctness, minimal spectrolaminar path

---

## Purpose and Scope

v0.2.0 advances jaxfne from a computational scaffold (v0.1.x) toward theoretically validated computational biophysics. The release hardens source bookkeeping, strengthens field admissibility checks, validates backend consistency, and provides a minimal spectrolaminar workflow that is scientifically coherent across emitter, source, field, and manifest layers.

**v0.2.0 is NOT:**
- Empirical validation (no datasets, nulls, ablations, or perturbations)
- Biological mechanism proof (no claim that Izhikevich dynamics explain cortical circuits)
- Whole-brain simulator (laminar proxy remains local and anatomically simplified)
- Physical amplitude calibration (CSD/LFP outputs remain proxy/readout metadata)

**v0.2.0 IS:**
- Theoretical validation of equations, source bookkeeping, field admissibility, and conservation laws
- Explicit source-mode exclusivity and synaptic current accounting
- Deterministic computational correctness under matched seeds
- Strict manifest semantics enabling reproducible science

---

## Architectural Pipeline

```
Emitter → Source → Field → Probe → Objective → Optimizer
  ↓          ↓        ↓       ↓         ↓
 Eqns    Bookkeep  Admiss. Readout  Grammar
```

Each layer has explicit acceptance gates documented here.

---

## Truth Position (Frozen at v0.2.0)

```
truth_mode:                          truth_safe_unverified
claim_level:                         computational_scaffold | theoretical_validation_candidate
source_calibration_status:           uncalibrated_izhikevich_native_current
source_projection_mode:              proxy_no_field_solve
field_solver_status:                 laminar_proxy_no_pde
field_claim_level:                   proxy_readout_only
physical_amplitude_claim_allowed:    false
empirical_validation_status:         not_empirically_validated
mechanism_claim_status:              not_claimed
theoretical_validation_status:       admissibility_gates_pass (if v0.2.0 completion)
```

**Non-escalation Rule:**
If v0.2.0 acceptance gates pass, claim_level may advance to `theoretical_validation_candidate`. No empirical mechanism claim is permitted without:
- Independent datasets (brain recordings, simulations from other tools)
- Null distributions (shuffled data, lesioned models)
- Ablations (removed currents, disabled plasticity)
- Perturbations (injected stimuli, parameter sweeps)
- Peer review and replication

---

## Source Bookkeeping: Declared Modes and Forbidden Patterns

### Allowed Source Modes

Exactly one mode per run. Declared in `RuntimeConfig.source_mode` and `Signals.metadata["source_mode"]`.

#### Mode 1: total_membrane_current
```
q = I_cap + I_ion + I_syn + q_ext + I_spike_proxy
```
- All currents folded into membrane potential derivative.
- No separate decomposed tracking.
- Synaptic current **not** added again as separate source term.
- Use case: Minimal forward-field, minimal tracking overhead.

#### Mode 2: decomposed_cap_ion_syn
```
q = [I_cap, I_ion, I_syn]  (separate tracks)
I_total = I_cap + I_ion + I_syn + q_ext + I_spike_proxy
```
- Capacitive, ionic, and synaptic currents tracked separately in source proxy.
- Synaptic current appears **only** in the I_syn decomposition track.
- **Never** added again to total.
- Use case: Detailed source analysis, layer-specific decomposition.

#### Mode 3: spike_kernel_proxy
```
q = spike_kernel_convolution(spikes) + I_cap + I_ion + q_ext
```
- Spike impulse modeled as feedforward kernel rather than native reset.
- Synaptic current modeled as scaled spike → post kernel (receptor-indexed).
- No double-counting of synaptic state.
- Use case: JAX-native spike representations, efficient batch computation.

#### Mode 4: receptor_state_proxy
```
q = I_cap + I_ion + I_syn_receptor_convolution + q_ext
```
- Receptor-indexed exponential synaptic kernels (AMPA, NMDA, GABA_A, GABA_B).
- Each receptor has its own tau and sign.
- Synaptic current computed as **single** convolution per receptor.
- No double-counting.
- Use case: Multi-receptor realism without dense state tensors.

### Forbidden Pattern: Double Synaptic Counting

```
FORBIDDEN:
  q = chi(I_cap + I_ion + I_syn) + q_syn_separate_term + q_ext
     ↑                                ↑
     synaptic current counted twice
```

This pattern is:
- Detected during validation and raises ValueError.
- Documented in test_source_bookkeeping.py.
- Never permitted, even under custom plasticity.

---

## Source Bookkeeping Metadata Requirements

Every run must export:

```python
signals.metadata["source_bookkeeping"] = {
    "source_mode": "total_membrane_current" | "decomposed_cap_ion_syn" | "spike_kernel_proxy" | "receptor_state_proxy",
    "source_projection_mode": "proxy_no_field_solve",
    "source_decomposition": "proxy_reduced_emitter",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "synaptic_current_counting": "single_proxy_expression_no_extra_synaptic_source",
    "source_mode_exclusive": True,  # Exactly one mode in this run
    "physical_amplitude_claim_allowed": False,
    "double_count_guard": "rejected" | "passed",
    "double_count_evidence": None | {error details}
}

manifest["source_bookkeeping"] = signals.metadata["source_bookkeeping"]
```

---

## Field Admissibility Gates

### Tensor Conductivity (σ)

- **Symmetric Positive Definite (SPD) check:** σ must pass eigenvalue test; all eigenvalues > 0.
- **Passivity check:** Field solver must not amplify energy.
- **Metadata:** Report eigenvalue range and condition number.

### Boundary Conditions

- **Boundary type:** mean_zero_neumann (primary), other types require explicit metadata.
- **Gauge declaration:** mean_zero (primary), other gauges require justification.
- **Metadata:** Every manifest must list boundary and gauge.

### Source Conservation

- **Integrated source check:** Sum of all source currents across domain must close (with tolerance).
- **Mean-zero property:** Gauge-constrained current should average to zero across layers.
- **Residual reporting:** Export source_integral_residual and gauge_residual to manifest.
- **Tolerance:** Relative error < 1e-6 or documented override.

### Field Outputs (phi_e, J_e, CSD, LFP)

- **Finite checks:** All arrays must pass isfinite() or report to metadata.
- **NaN/Inf rejection:** json.dumps(manifest, allow_nan=False) must pass.
- **Metadata:** Report any clipped/clamped values.

### Proxy-Only Metadata

- **field_claim_level:** "proxy_readout_only" unless full PDE validated.
- **field_solver_status:** "laminar_proxy_no_pde" for laminar proxy path.
- **CSD/LFP amplitude:** No physical unit claim without calibration.

---

## Runtime and Dtype Validation

### CPU-First Default

- **Default backend:** CPU (JAX cpu device).
- **x32 default:** float32 computations unless x64 explicitly opted in.
- **GPU fallback:** Allowed; test on CPU.
- **Metadata:** Export `default_backend`, `actual_backend`, `x64_enabled`.

### Runtime Report

Every run exports:

```python
manifest["runtime"] = {
    "jax_version": str(jax.__version__),
    "jaxlib_version": str(jaxlib.__version__),
    "default_backend": "cpu",
    "actual_backend": str(jax.devices()[0].platform),
    "available_devices": [str(d) for d in jax.devices()],
    "dtype": "float32",
    "x64_enabled": False,  # unless jax.config.update('jax_enable_x64', True)
    "seed": int,
    "n_steps": int,
    "recurrent_backend": "dense" | "edge_list",
    "synaptic_kernel": "exponential" | "receptor_exponential",
}
```

---

## Backend Parity: Dense vs Edge-List

### Deterministic Reproducibility

- **Same seed → same trajectory** on same backend.
- Reproducibility tested on small networks (n_neurons ≤ 100, duration ≤ 100 ms).
- Exact match expected for dense backend.

### Numerical Consistency

- **Different backends (dense vs edge-list) may diverge intentionally:**
  - Dense: dense matrix multiplication of weights.
  - Edge-list: sparse segment_sum aggregation.
  - Different numerical precision → different trajectories OK.
- **Tolerance table:** Empirically establish relative tolerance (e.g., max abs spike-time diff < 1 ms).
- **Metadata:** Report which backend was used; include tolerance in manifest.

### Acceptance Criterion

- Dense backend deterministic test: same seed → identical spike times (within float32 eps).
- Edge-list parity test: matched-seed runs diverge < tolerance threshold.
- Both backends JSON-safe.

---

## Objective Grammar and Null-Ready Outputs

### Score Grammar

Objectives declare:

```python
objective = jtfne.objective()
    .loss("name", metric="metric_name", target=value, weight=weight)
    .regularizer("name", metric="metric_name", target=value, weight=weight)
    .gate("name", metric="metric_name", threshold=value, criterion="below|above|equal|in_range")
    .null("name", metric="metric_name", distribution="empirical|gaussian", sample_size=N)
```

### Synchrony Gates

- **Anti-synchrony gate:** flag if spike correlation > threshold (e.g., > 0.7).
- **Burst gate:** flag if interspike intervals < 5 ms (if not realistic for cell type).
- **Metadata:** Include gate evaluation in objective report.

### Null-Ready Outputs

- **S_lam (synchrony metric):** Must include null distribution (spike-time shuffled or phase-randomized control).
- **Never report S_lam without nulls in public workflows.**
- **Metadata:** Include null percentile (p < 0.05 or p-value).

---

## Minimal Spectrolaminar Path: V1–V4–PFC Workflow

### Target Architecture

- **3-area hierarchy:** V1 (64 neurons) → V4 (256 neurons) → PFC (512 neurons).
- **Layers:** Each area has L2/3, L4 (if sensory), L5, L6.
- **Cell types:** E (80%), PV (10%), SST (7%), VIP (3%) in each layer.
- **Total:** ~850 neurons.
- **Connectivity:** Feedforward (L4→L2/3→L5→L6) + feedback (L6→L4, etc.).

### Inputs and Readouts

- **V1 input:** Synthetic visual stimulus (grating, 100 ms duration).
- **Readouts:** Spikes, layer-average source, LFP-like proxy, CSD-like proxy.
- **Outputs:** manifest with runtime, source bookkeeping, field metadata, objective (if used).

### Acceptance Criterion

- Notebook runs in Colab and locally without errors.
- Example runs on installed jaxfne==0.2.0.
- Manifest is JSON-safe and includes all required metadata.
- No custom solver functions; no legacy code.
- No empirical claims; all truth gates consistent.

---

## Acceptance Gates Summary

| Gate | Criterion | Test | Reject If |
|------|-----------|------|-----------|
| Emitter equations | Izhikevich reset, adaptation, native current documented | test_izhikevich_equations | Claims biological calibration |
| Source mode exclusivity | Exactly one mode per run; double-count pattern impossible | test_source_bookkeeping_exclusivity | Multiple modes or double-count passes |
| Source conservation | Integrated source residual < 1e-6 | test_source_conservation | Residual > tolerance or NaN |
| Field admissibility | SPD tensor, passivity, finite phi_e/J_e/CSD | test_field_admissibility_gates | Any field metric is NaN/Inf or non-SPD |
| Backend parity | Dense deterministic + edge-list consistent | test_backend_parity_dense_edge_list | Parity tolerance exceeded or seed divergence |
| Runtime/dtype | CPU-first, x32 default, runtime report present | test_runtime_dtype_validation | CUDA required or x64 hardcoded |
| Manifest strictness | json.dumps(..., allow_nan=False) passes | test_manifest_json_strictness | Any NaN/Inf in manifest |
| Spectrolaminar path | Notebook runs, 3-area V1–V4–PFC, manifest consistent | test_minimal_spectrolaminar_path | Missing metadata, custom solvers, or empirical claims |

---

## Implementation Roadmap

### Phase 1 (Current): Source Bookkeeping + Theoretical Contract
- Add this document
- Enforce source-mode exclusivity in code
- Add focused source bookkeeping tests
- **Gate:** All 4 source modes declared; double-count test rejects forbidden pattern

### Phase 2: Field Admissibility
- Add SPD/passivity checks
- Strengthen conservation and gauge validation
- Add residual reporting
- **Gate:** Field admissibility tests pass; manifest includes field metadata

### Phase 3: Backend Parity
- Implement deterministic reproducibility test for dense backend
- Establish tolerance table for edge-list parity
- Add backend parity tests
- **Gate:** Dense deterministic + edge-list consistent

### Phase 4: Runtime/Dtype Validation
- Add CPU-first and x32-default enforcement
- Runtime report completeness
- **Gate:** Runtime/dtype tests pass; manifest includes runtime metadata

### Phase 5: Objective Grammar and Null Discipline
- Implement null-ready objective grammar
- Add synchrony gates
- Enforce null distribution for S_lam
- **Gate:** Objective tests pass; null discipline enforced

### Phase 6: Minimal Spectrolaminar Path
- Create V1–V4–PFC notebook/example
- Integrate all prior phases
- JSON-safe manifest validation
- **Gate:** Spectrolaminar workflow runs; manifest is strict JSON-safe

### Phase 7: Release Prep
- Version bump to 0.2.0
- CHANGELOG update
- Full test/example/build suite
- **Gate:** 282+ tests pass; 7+ examples pass; install smoke succeeds

---

## Non-Escalation Pledge

This document is an architectural contract for v0.2.0. It makes no empirical or biological claims. Any future work claiming mechanism-level validation, biological calibration, or circuit-function proof must:

1. Establish baseline metrics from real experimental data
2. Validate nulls (lesions, perturbations, shuffled data)
3. Show model ablations
4. Demonstrate replication across conditions
5. Pass peer review

Until then, all outputs remain computational scaffolds under `truth_safe_unverified / computational_scaffold`.

---

## References

- v0.1.0 release: compact public API and scaffold
- v0.1.1 release: source/field truth metadata hardening
- v0.1.2 release: scan-backend performance validation
- DOCTRINE.md: Gamma Labyrinth alignment
- jaxfne/core.py: Signals, Manifest, RunReceipt classes
- jaxfne/fields.py: laminar proxy source/field logic
- jaxfne/validation.py: invariant checking

