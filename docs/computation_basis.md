# Computation Basis: TFNE as a Collapsible Tensor-Field Scaffold

## Purpose

This document describes the computation basis of TFNE (Tensor-Field Neural Emitter): a principled framework for composing neural emitters, source projections, field approximations, and readout operators into reproducible, extensible workflows.

The key principle: **TFNE is collapsible.** Every operator (emitter, source, field, probe) has a well-defined input/output shape contract. When not computed or not claimed, operators can be "collapsed" to identity or proxy equivalents without breaking the pipeline. This enables safe modularity and reduces the barrier to adding new domains (whole-brain, ephaptic, ionic, etc.).

This is a **doctrine document**, not an implementation guide. It complements [Mathematical Glossary Flow](mathematical_glossary_flow.md) (equations) and [Source/Field Equations](source_field_equations.md) (implementation detail).

---

## Core Principle: Collapsible Dimensions

TFNE operates on tensors with the following canonical dimensions:

| Dimension | Symbol | Meaning | Example |
|-----------|--------|---------|---------|
| **Time** | $T$ | Number of timesteps | 1000 |
| **Units** | $N$ | Number of neurons/cells | 100 |
| **Space** | $X$ | Spatial locations (contact points, pixels) | 16 (probe contacts) |
| **Features** | $F$ | Neuron features (compartments, species, etc.) | 1 (soma) or 3 (soma + axon + dendrite) |
| **Readout** | $R$ | Readout channels (modalities, metrics) | 8 (SPK, Vm, source, LFP, CSD, EEG, MEG, EMM) |

### Canonical Shapes

Each stage of the pipeline has a **canonical shape** and can be **collapsed** if not needed:

| Stage | Input | Output | Collapse Rule |
|-------|-------|--------|---------------|
| **Emitter** | $[T, N]$ stimulus | $[T, N]$ state | Optional; always present |
| **Source** | $[T, N]$ state | $[T, X]$ density | Set to None if not used |
| **Field** | $[T, X]$ source | $[T, X]$ potential | Set to None if proxy-only |
| **Probe** | $[T, X]$ field | $[T, R]$ readout | Selective operator choice |

### Collapse Examples

**Example 1: Emitter-only (V_m readout, no field)**
```
[T, N] state → [T, N] V_m voltage → report only
(source, field, CSD/LFP readouts are None or skipped)
```

**Example 2: Emitter → Source → Readout (no PDE solve)**
```
[T, N] state → [T, N] I_native → [T, X] source (spatial projection)
              → [T, X] CSD_proxy → report
(field solve is skipped; CSD computed from source)
```

**Example 3: Full pipeline (future, with solver)**
```
[T, N] state → [T, N] I → [T, X] source → [T, X] potential (solved)
             → [T, X] current → [T, R] readouts (CSD, LFP, EEG, MEG, EMM)
```

---

## No Fake Dimensions Rule

**CRITICAL:** TFNE forbids adding dimensions that are not grounded in the problem.

### Forbidden Patterns

**FORBIDDEN: Artificial batch dimension**
```python
# WRONG:
signals.V_m.shape = [1, T, N]  # Extra artificial batch axis
# This is a "fake" dimension: not part of the problem, just implementation artifact

# RIGHT:
signals.V_m.shape = [T, N]  # Canonical: time × units
# For multi-trial, run separate simulations and stack post-hoc
```

**FORBIDDEN: Implicit stochastic dimension**
```python
# WRONG:
# Model.simulate() with vmap but not declared
# Output shape: [T, N, N_stochastic_samples]  # Hidden dimension

# RIGHT:
# Declare: simulate_batch() with explicit PRNG key splitting
# Output: [N_seeds, T, N]  # Explicit batch dimension
```

**FORBIDDEN: Phantom feature dimension**
```python
# WRONG:
# Izhikevich state includes [T, N, 4] (v, u, w, ...) mixed with spike [T, N]
# Different users interpret last axis differently

# RIGHT:
# Keep state and spike separate:
# Signals.V_m: [T, N]
# Signals.spikes: [T, N]
# Signals.state (if internal): [T, N, 4]  # Explicitly a state dimension
```

### Dimension Naming Convention

When adding a new dimension:

1. **Name it explicitly** in docstrings and Manifest
2. **Document its size and meaning** (not "axis 2" but "spike_threshold_samples")
3. **Test shape stability** — if shape changes unexpectedly, it breaks downstream operators
4. **Collapse or broadcast** — never silently drop or pad dimensions

---

## Basis Changes and Canonical Representations

TFNE supports multiple **basis choices** for the same underlying computation. Each basis is valid if:
1. It preserves the input/output shape contract
2. It is documented in Manifest
3. It is tested for numerical equivalence

### Example: Source Projection Bases

**Basis 1: Spatial contact projection (current default)**
$$q_\alpha(t) = \sum_n w_{n\alpha} \cdot I_n(t)$$
- $w_{n\alpha}$: spatial coupling weight (distance-based)
- $q_\alpha$: source at contact $\alpha$
- Implementation: `jaxfne.fields.project_laminar_sources(..., basis="spatial_contact")`

**Basis 2: Anatomical layer projection (future)**
$$q_\ell(t) = \sum_{n \in \ell} I_n(t)$$
- $q_\ell$: total current in layer $\ell$
- Implementation: `jaxfne.fields.project_laminar_sources(..., basis="layer")`

**Basis 3: Frequency/spectral projection (future)**
$$\hat{q}_f(t) = \mathcal{F}[\sum_n \phi_f(n) \cdot I_n(t)]$$
- $\hat{q}_f$: spectral component at frequency $f$
- Implementation: `jaxfne.fields.project_laminar_sources(..., basis="spectral")`

**Contract:** All bases must have output shape $[T, X]$ for some $X$. Basis choice is documented in Manifest.

---

## Declared-Future Field Regimes

TFNE defines a roadmap of **future field computation regimes**, each with different complexity and assumptions.

### v0.2.24–v0.2.27: Proxy Field (Current)

```
Field solver status: laminar_proxy_no_pde
Regime: No PDE solve
Equation: ∇·(-σ_e ∇φ_e) = q  [DECLARED, NOT SOLVED]
Implementation: CSD ∝ ∇·q from source (proxy)
Conductivity: Proxy (scalar, isotropic, no calibration)
Boundary: Metadata-only (future use)
Gauge: Metadata-only (future use)
Claim: Computational scaffold; no physical conductivity claim
```

**What is computed:**
- Source projection: $q(x,t)$ from emitter state
- Proxy CSD: $\mathrm{CSD}_\mathrm{proxy} = \nabla \cdot q$ (kernel convolution, no solve)
- Proxy LFP: $\mathrm{LFP}_\mathrm{proxy} = \sum_x K_\mathrm{LFP}(x) \cdot q(x)$ (spatial filter)

**What is NOT computed:**
- Field solve: $\phi_e$, $\mathbf{J}_e$ (not solved; only proxy)
- Conductivity validation: no SPD check, no physical unit assignment
- Boundary/gauge enforcement: metadata only

### v0.2.27 (Declared Future): Conservation-Inspired Proxy Diagnostics

> **Approved scope:** conservation-inspired proxy diagnostics only.
> No Poisson solver is introduced in v0.2.27.
> `solved_poisson` remains `implemented=False` and `claim_allowed=False`.
> A Poisson solver remains gated future work and requires separate approval before any implementation begins.

```
Field solver status: poisson_admissibility_diagnostic (planned, not implemented)
Regime: Conservation-inspired proxy diagnostics
Equation: ∇·q ≈ 0 [source conservation proxy check, NOT solved]
Implementation: Proxy diagnostics over existing field/source outputs; no Poisson solver in v0.2.27
Conductivity: Proxy (still scalar, no calibration)
Boundary: Metadata-only (future use)
Gauge: Metadata-only (future use)
Diagnostic: Source conservation proxy (∫∫q dA ≈ 0 checked as proxy, not PDE-enforced)
Claim: Proxy conservation check; no field solve; conductivity still uncalibrated
```

**Motivation:** Validate that source declarations satisfy approximate conservation laws at the proxy level, without claiming physical conductivity or solving a PDE. Useful for debugging source projection and testing model consistency.

**Planned diagnostic outputs (proxy-only, no field solve):**
- Source conservation proxy check: `source_integral_proxy` (scalar, proxy)
- Gradient proxy: `∇·q` approximation over existing source array
- Diagnostics: source_integral_check, proxy_conservation_residual

**What is NOT in v0.2.27:**
- No Poisson solve: $\phi_e$, $\mathbf{J}_e$ are not computed
- No CG/MINRES or iterative solver
- No boundary/gauge enforcement (metadata only)
- No calibrated conductivity
- No physical-amplitude claim

### v0.3.x (Declared Future): Calibrated Physical Field

```
Field solver status: calibrated_physical_conductivity (not in v0.2.x roadmap)
Regime: Solve Poisson with calibrated conductivity
Equation: ∇·(-σ_e(x) ∇φ_e) = q  [SOLVED with SPD σ_e(x)]
Implementation: Fast FFT-based or high-order FEM with validated geometry
Conductivity: Calibrated to empirical tissue (SPD tensor, anisotropic)
Boundary: Dipole/inhomogeneous (empirical)
Gauge: Physical (zero-flux far-field)
Claim: Field is physical under empirical conductivity calibration
```

**When to claim this regime:**
- Tissue conductivity is measured (anisotropic tensor, not scalar proxy)
- Field geometry is validated against real probe data (lead field measured)
- CSD/LFP/EEG outputs match empirical recordings (validation data present)
- Methods section documents calibration source and validation benchmark

---

## Extensibility Doctrine: Adding New Domains

When extending TFNE to a new domain (whole-brain, multi-area, ephaptic coupling, etc.), follow this framework:

### Step 1: Define Input/Output Shapes

```
Problem: Want to simulate [motor cortex] ← [somatosensory area] ← [thalamus]

New emitter: [T, N_motor] ← [T, N_sensory, N_thalamus] via learned connectivity
Input shape: [T, N_sensory + N_thalamus]
Output shape: [T, N_motor]
Collapse rule: Can be isolated to single area if connectivity is sparse

New source: Whole-brain anatomy (motor/sensory/thalamus in 3D space)
Input shape: [T, N_total] (all neurons, pooled)
Output shape: [T, X_3d] (3D voxel grid)
Collapse rule: Can project to single area (subset of X_3d) or whole brain
```

### Step 2: Check Dimension Conservation

```
✓ Does emitter output shape [T, N_motor] match input to source?
✓ Does source output [T, X] have constant X across time?
✓ Does field solver accept [T, X] and preserve shape?
✓ Does probe operator produce [T, R] for declared R?
```

### Step 3: Document Basis and Assumptions

```
Basis: Anatomical connectivity from Allen Mouse Brain Atlas
Assumption: Straight-line distance (not curved axon paths)
Assumption: Conductivity isotropic (future: anisotropic white matter)
Assumption: No ephaptic coupling (future: add extracellular voltage feedback to emitter)
Collapse rule: Can run single area with all-to-all connectivity if whole-brain is too large
```

### Step 4: Declare Claim Boundary

```
truth_mode: truth_safe_unverified
claim_level: computational_scaffold (new domain, not validated)
physical_amplitude_claim_allowed: False (connectivity not empirically calibrated)
source_calibration_status: uncalibrated_multi_area_izhikevich
field_solver_status: laminar_proxy_no_pde (even for 3D, still proxy in v0.2.24)
Validation required: None yet; this is exploratory setup
```

### Step 5: Test Collapse and Reshape

```python
# Test 1: Single area (collapse multi-area)
model_motor_only = configure_motor_area_only()
signals_motor = model_motor_only.simulate(sim)
assert signals_motor.V_m.shape == [T, N_motor]

# Test 2: Multi-area (full)
model_full = configure_three_area_system()
signals_full = model_full.simulate(sim)
assert signals_full.V_m.shape == [T, N_total]

# Test 3: Reshape readout (compress spatial, keep time)
readouts = model_full.compute_readout(signals_full, [
    jtfne.readout_spec("motor_spikes", "spike_rate_hz", probe_subset="motor"),
    jtfne.readout_spec("sensory_spikes", "spike_rate_hz", probe_subset="sensory"),
])
```

---

## Numerical Stability and Determinism

### PRNG Contract

TFNE uses JAX's deterministic PRNG. All simulations with the same seed are reproducible:

```python
sim1 = jtfne.simulation(seed=42, duration_ms=100.0, dt_ms=0.1)
signals1 = model.simulate(sim1)

sim2 = jtfne.simulation(seed=42, duration_ms=100.0, dt_ms=0.1)
signals2 = model.simulate(sim2)

assert jnp.allclose(signals1.V_m, signals2.V_m)  # ✓ Same trajectory
```

### Finiteness Guarantee

All readouts are checked for NaN/Inf before serialization:

```python
manifest = model.manifest(signals, readouts)
json_output = json.dumps(manifest, allow_nan=False)  # Fails if NaN/Inf present
```

### Shape Stability

If an operator's output shape changes unexpectedly, the pipeline breaks. Test this:

```python
# Good: shape is stable across runs
for seed in range(10):
    sim = jtfne.simulation(seed=seed, duration_ms=100.0, dt_ms=0.1)
    signals = model.simulate(sim)
    assert signals.V_m.shape == [T, N], f"Shape changed: {signals.V_m.shape}"
```

---

## Doctrine Summary

| Principle | Rule | Example |
|-----------|------|---------|
| **Canonical shapes** | Every operator has explicit I/O shape [T, X] or similar | Emitter: [T,N]; Source: [T,X]; Field: [T,X] |
| **Collapse safety** | Operators can be None or proxy without breaking contract | Field=None, field_solver=proxy are both valid |
| **No fake dimensions** | Never add dimensions not grounded in the problem | Batch, stochasticity, features must be explicit |
| **Basis choice** | Multiple bases allowed if input/output shapes preserved | Spatial/layer/spectral projection bases equivalent |
| **Declared-future regimes** | Future solver/conductivity modes are declared but not implemented | v0.2.27 diagnostics, v0.3.x physical conductivity are future |
| **Extensibility** | New domains follow: shapes → dimensions → basis → claims → test | Whole-brain extension example above |
| **Determinism** | Same seed → same trajectory (PRNG contract) | `seed=42` reproducible across runs |
| **Finiteness** | All outputs are JSON-safe (no NaN/Inf before serialization) | `json.dumps(manifest, allow_nan=False)` enforced |

---

---

## Implemented in v0.2.26

The following computation-basis contract objects are implemented in jaxfne v0.2.26:

| Object / Function | Location | Purpose |
|-------------------|----------|---------|
| `AxisSpec` | `jaxfne.core` | Typed descriptor for one tensor axis (name, status, size, units) |
| `BasisSpec` | `jaxfne.core` | Typed descriptor for the full computation basis (space, time, field regime, source mode, probe basis) |
| `default_basis_spec()` | `jaxfne.core` | Returns the default BasisSpec matching the current laminar-proxy scaffold |
| `validate_basis_spec()` | `jaxfne.validation` | Validates a BasisSpec or dict against computation-basis contracts |
| `basis_claim_gate()` | `jaxfne.validation` | Evaluates physical-amplitude claim eligibility given basis + runtime status |
| `manifest["basis"]` | `jaxfne.core.Model.manifest()` | Nested basis metadata block in every run manifest |

**Allowed `BasisSpec.field_regime` values and their status:**

| Regime | Status | Implemented | Claim Allowed |
|--------|--------|-------------|---------------|
| `laminar_proxy` | Active (default) | True | False |
| `quasi_static_resistive` | Reserved | False | False |
| `solved_poisson` | Gated future (no solver in v0.2.x) | **False** | **False** |
| `future_admittive` | Declared future (v0.3.x) | **False** | **False** |
| `future_maxwell` | Declared future (v0.3.x) | **False** | **False** |

`solved_poisson`, `future_maxwell`, and `future_admittive` are **not capabilities** — they are named future-doctrine markers only. `implemented=False`, `claim_allowed=False` are structurally enforced and cannot be escalated. A Poisson solver requires separate approval before any implementation begins.

---

## See Also

- [Mathematical Glossary Flow](mathematical_glossary_flow.md) — Core TFNE equations
- [Source/Field Equations](source_field_equations.md) — Source modes, forbidden patterns, field metadata
- [Tensor-Network Ancestry](tensor_network_ancestry.md) — v0.2.29 conceptual context: basis-transform doctrine and historical parallels
- [Probe Operators](probe_operators.md) — Readout operators and their claim boundaries
- [Scope and Limitations](scope_and_limitations.md) — What TFNE claims and does not claim
