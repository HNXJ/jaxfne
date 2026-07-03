# Computation Basis: TFNE as a Collapsible Tensor-Field Scaffold

## Purpose

This document describes the computation basis of TFNE (Tensor-Field Neural Equations): a principled framework for composing neural emitters, source projections, field approximations, and readout operators into reproducible, extensible workflows.

The key principle: **TFNE is collapsible.** Every operator (emitter, source, field, probe) has a well-defined input/output shape contract. When not computed or not stated, operators can be "collapsed" to identity or proxy equivalents without breaking the pipeline. This enables safe modularity and reduces the barrier to adding new domains (whole-brain, ephaptic, ionic, etc.).

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

**Example 3: Full pipeline (reserved, with solver)**
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

**Basis 2: Anatomical layer projection (reserved)**
$$q_\ell(t) = \sum_{n \in \ell} I_n(t)$$
- $q_\ell$: total current in layer $\ell$
- Implementation: `jaxfne.fields.project_laminar_sources(..., basis="layer")`

**Basis 3: Frequency/spectral projection (reserved)**
$$\hat{q}_f(t) = \mathcal{F}[\sum_n \phi_f(n) \cdot I_n(t)]$$
- $\hat{q}_f$: spectral component at frequency $f$
- Implementation: `jaxfne.fields.project_laminar_sources(..., basis="spectral")`

**Contract:** All bases must have output shape $[T, X]$ for some $X$. Basis choice is documented in Manifest.

---

## Proxy field regime

The shipped field regime is the laminar proxy:

```
Field solver status: linear_solver
Regime: proxy readout (no PDE solve)
Equation: CSD_proxy = nabla . q  (kernel convolution)
Conductivity: scalar proxy (uncalibrated)
Boundary / gauge: metadata fields
```

**Computed:**
- Source projection: $q(x,t)$ from emitter state
- Proxy CSD: $\mathrm{CSD}_\mathrm{proxy} = \nabla \cdot q$ (kernel convolution)
- Proxy LFP: $\mathrm{LFP}_\mathrm{proxy} = \sum_x K_\mathrm{LFP}(x) \cdot q(x)$ (spatial filter)

The field potential $\phi_e$ and current density $\mathbf{J}_e$ are represented by
proxy operators rather than a volume-conductor solve, consistent with
`field_solver_status = "linear_solver"` and
`physical_amplitude_calibrated = False`. Reserved field regimes
(conservation diagnostics, elliptic solver, calibrated physical field) are catalogued in
[Limitations and future plans](limitations_and_future_plans.md).

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
Assumption: Conductivity isotropic (reserved: anisotropic white matter)
Assumption: No ephaptic coupling (reserved: add extracellular voltage feedback to emitter)
Collapse rule: Can run single area with all-to-all connectivity if whole-brain is too large
```

### Step 4: Declare Statement Boundary

```
run_status: tutorial_scaffold
model_status: computational_scaffold (new domain, candidate-status)
amplitude_status: False (connectivity not empirically calibrated)
source_calibration_status: uncalibrated_multi_area_izhikevich
field_solver_status: linear_solver (even for 3D, still proxy in v0.2.24)
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
| **Reserved regimes** | Elliptic/electrodynamic solver and conductivity modes are catalogued in [Limitations and future plans](limitations_and_future_plans.md) | reserved |
| **Extensibility** | New domains follow: shapes → dimensions → basis → statements → test | Whole-brain extension example above |
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
| `validate_basis_spec(spec)` | `jaxfne.validation` | Validates a BasisSpec or dict against computation-basis contracts |
| `manifest["basis"]` | `jaxfne.core.Model.manifest()` | Nested basis metadata block in every run manifest (populated by `jaxfne.core._default_basis_dict()`) |

> **Note (re-verified 2026-07-03):** `basis_statement_gate()` does not exist —
> not in `jaxfne.validation`, not anywhere in the `jaxfne` package, and not in
> `jaxfne.__all__`. This table previously claimed it as implemented; that was
> false and has been removed. Physical-amplitude/claim-level eligibility is
> represented directly in report dicts (`claim_level`,
> `physical_amplitude_calibrated`, etc. — see
> [Tensor Operators — Report shapes](api/tensor_operators.md#report-shapes))
> rather than through a dedicated gate function. `AxisSpec`, `BasisSpec`, and
> `default_basis_spec()` all exist in `jaxfne.core` and are re-exported at the
> `jaxfne` top level (`jaxfne.AxisSpec` works directly) as documented.

**Allowed `BasisSpec.field_regime` values and their status:**

| Regime | Status | Implemented | Statement Allowed |
|--------|--------|-------------|---------------|
| `laminar_proxy` | Active (default) | True | False |
| `quasi_static_resistive` | Reserved | False | False |
| `solved_poisson` | Gated reserved (no solver in v0.2.x) | **False** | **False** |
| `reserved_admittive` | Declared reserved (v0.3.x) | **False** | **False** |
| `reserved_maxwell` | Declared reserved (v0.3.x) | **False** | **False** |

`solved_poisson`, `reserved_maxwell`, and `reserved_admittive` are **reserved markers** — they are named reserved-doctrine markers only. `implemented=False`, `status_enabled=False` are structurally enforced and cannot be escalated. An elliptic-equation field solve for this regime requires separate approval before any implementation begins.

---

## See Also

- [Mathematical Glossary Flow](mathematical_glossary_flow.md) — Core TFNE equations
- [Source/Field Equations](source_field_equations.md) — Source modes, forbidden patterns, field metadata
- [Tensor-Network Ancestry](tensor_network_ancestry.md) — v0.2.29 conceptual context: basis-transform doctrine and historical parallels
- [Probe Operators](guides/probe_operators.md) — Readout operators and their statement boundaries
- [Scope and Limitations](limitations_and_future_plans.md) — What TFNE statements and stays scoped to
- [TFNE Operator Doctrine](operator_doctrine.md) — Per-stage contract table built on these tensor shapes
- [Tensor Electromagnetics Scope](tensor_electromagnetics_scope.md) — Reserved field-solver stages referenced by the reserved markers above
