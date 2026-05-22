# Source and Field Equations

## Purpose

This document specifies source bookkeeping modes, field metadata contracts, forbidden patterns, and the mappings between jaxfne core equations and runtime manifest/report fields.

It complements the [Mathematical Glossary Flow](mathematical_glossary_flow.md) by grounding source and field equations in implementation detail: which manifest field controls which equation, what modes are allowed, what combinations are forbidden, and how to interpret claim boundaries in code.

---

## Source Bookkeeping and Calibration

### Definition: Source Density

The source density $q(x,t)$ is the current per unit volume (or area) at position $x$ and time $t$:

$$q(x,t) = P_s[z(t), I(t), \chi(x)]$$

Physically, $q$ is the transmembrane current density that becomes the boundary condition for the field equation.

### Source Modes (Mutually Exclusive)

A jaxfne model declares **exactly one** source mode per simulation run. All others are inactive.

| Mode | Equation | Status | Implementation |
|------|----------|--------|-----------------|
| **total_membrane_current** | $q = I_\mathrm{mem}(z, t)$ | Reserved (future) | Not in v0.2.24–v0.2.27 |
| **decomposed_cap_ion_syn** | $q = I_\mathrm{cap} + I_\mathrm{ion} + I_\mathrm{syn}$ | Reserved (future) | Not in v0.2.24–v0.2.27 |
| **proxy_no_field_solve** | $q = \text{declared proxy} \approx I_\mathrm{native}$ | **Active (current default)** | `jaxfne.fields.project_laminar_sources()` |
| (none declared) | Signals.sources = None | Allowed (no source) | Field=None, no readouts |

**Rule:** Declare source_projection_mode and source_calibration_status in Manifest. If both are None, Signals.sources remains None.

### Current Default: proxy_no_field_solve

In v0.2.24–v0.2.27, the active mode is:

```
source_projection_mode = "proxy_no_field_solve"
source_calibration_status = "uncalibrated_izhikevich_native_current"
```

**What this means:**
- $q$ is computed as: emitter native current (Izhikevich $I_k$) + spike impulse proxy (20× gain)
- $q$ is NOT validated against empirical synaptic or ionic current
- $q$ is spatial proxy: neuron position (in laminar_source_geometry) → spatial contact coupling
- Field computation is **not performed** (field_solver_status = "laminar_proxy_no_pde")
- CSD is **computed from source**, not solved: $\mathrm{CSD} \propto \nabla \cdot q$ (proxy)

**Code example:**
```python
cfg = jtfne.configuration()
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",  # metadata-only
        gauge="mean_zero",  # metadata-only
    )

# In the Manifest:
manifest = model.manifest(signals, readouts)
print(manifest["source_calibration_status"])  # → "uncalibrated_izhikevich_native_current"
print(manifest["source_projection_mode"])  # → "proxy_no_field_solve"
print(manifest["field_solver_status"])  # → "laminar_proxy_no_pde"
```

---

## Forbidden Pattern: Synaptic Double-Counting

**Critical rule: A source declaration must NOT count synaptic current twice.**

### The Pattern (FORBIDDEN)

```
FORBIDDEN:
q(x,t) = χ(x) · (I_cap(t) + I_ion(t) + I_syn(t)) + q_syn_extra(t)
         ↑ Single source (membrane current)         ↑ Extra synaptic term
         
Result: I_syn is counted twice (once in total membrane, once in q_syn_extra)
```

**Why it's forbidden:**
- Current conservation: ∇·q must balance all currents once, not twice
- Gauge invariance breaks: mean-zero gauge cannot hold if synaptic current is duplicated
- Probe readout corruption: CSD, LFP, and EMM metrics become nonphysical

### The Pattern (ALLOWED)

**Option A: Single membrane-current source**
```
ALLOWED:
q(x,t) = χ(x) · (I_cap(t) + I_ion(t) + I_syn(t))
         Single transmembrane source, all components included once
```

**Option B: Decomposed sources (future v0.2.26)**
```
ALLOWED (declared future):
q_cap(x,t) = χ(x) · I_cap(t)
q_ion(x,t) = χ(x) · I_ion(t)
q_syn(x,t) = χ(x) · I_syn(t)
→ Keep separate in Signals, sum only at field boundary
→ Requires explicit source_decomposition contract
```

**Option C: Proxy current (current default v0.2.24)**
```
ALLOWED (active):
q_proxy(x,t) = χ(x) · (I_Iz(t) + I_spike_impulse(t))
              ↑ Native Izhikevich current (NOT decomposed)
              ↑ Spike impulse proxy (20× gain, derived)
→ Single proxy source, no decomposition
→ source_calibration_status: "uncalibrated_izhikevich_native_current"
```

### How to Audit Your Code

Before releasing a model:

```python
# Check manifest:
manifest = model.manifest(signals, readouts)
source_cal = manifest.get("source_calibration_status")
source_mode = manifest.get("source_projection_mode")
source_decomp = manifest.get("source_decomposition", "unknown")

# Assertion:
assert source_cal in [
    "uncalibrated_izhikevich_native_current",
    "uncalibrated_hh_native_current",
    "uncalibrated_jaxley_voltage_proxy",
    None,  # no source declared
], f"Unexpected source_calibration_status: {source_cal}"

# Check: if source_decomposition is "decomposed_cap_ion_syn", 
# ensure Signals.sources has shape [n_time, n_neurons, 3] (cap, ion, syn channels)
if source_decomp == "decomposed_cap_ion_syn":
    assert signals.sources.shape[-1] == 3, "Expected 3 source channels (cap, ion, syn)"
```

---

## Field Metadata and Claim Boundaries

### Field Solver Status

The **field_solver_status** field in Manifest declares whether the field PDE is solved or proxy-only.

| Status | Solver | φ_e | Current | CSD | Claim |
|--------|--------|-----|---------|-----|-------|
| `laminar_proxy_no_pde` | **None** | Proxy | Proxy | **Proxy** | Computational scaffold; no physical conductivity claim |
| `specified_future_solver` | **Reserved** | To be solved | To be solved | **Solved** | Future v0.2.27+; not implemented yet |

**Current default (v0.2.24–v0.2.27):**
```
field_solver_status = "laminar_proxy_no_pde"
```

**What it means:**
- The field equation $\nabla \cdot (-\sigma_e \nabla \phi_e) = q$ is **declared but NOT solved**
- $\phi_e$, $\mathbf{J}_e$, and $\mathrm{CSD}$ are computed from $q$ using laminar-proxy kernels (no PDE solve, no conductivity calibration)
- Boundary conditions and gauge are metadata-only (do not affect computation)
- CSD sign convention is declared: positive = extracellular source (current flowing outward)

### Boundary Conditions and Gauge (Metadata-Only in v0.2.24)

```python
cfg = jtfne.configuration()
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",  ← Metadata field (v0.2.24)
        gauge="mean_zero",              ← Metadata field (v0.2.24)
    )
```

In v0.2.24–v0.2.27, these are stored in Manifest but do **not** affect simulation:

```
boundary_condition: Specifies Neumann (zero-flux) condition (future solver will enforce)
gauge: Specifies mean-zero constraint (future solver will use)
```

**In v0.2.27+ (future)**, when a field solver is added:
- boundary_condition will be enforced during PDE solve
- gauge will be applied to enforce $\int \phi_e \, dx = 0$ (mean-zero potential)

**Code example (current):**
```python
manifest = model.manifest(signals, readouts)
print(manifest["field_solver_status"])  # → "laminar_proxy_no_pde"
print(manifest["boundary_condition"])  # → "mean_zero_neumann"
print(manifest["gauge"])  # → "mean_zero"

# These fields are informational only in v0.2.24.
# They document intended future behavior.
```

### CSD Sign Convention

The current jaxfne convention:

$$\mathrm{CSD}(x,t) = \nabla \cdot \mathbf{J}_e(x,t) = -\nabla \cdot (\sigma_e \nabla \phi_e)$$

**Sign convention in jaxfne:**
- **Positive CSD** = extracellular current diverging (flowing outward)
- **Negative CSD** = extracellular current converging (flowing inward)
- **Interpretation:** Positive CSD suggests membrane **sink** (inward membrane current); negative CSD suggests **source** (outward membrane current)

**Declared in Manifest:**
```python
manifest = model.manifest(signals, readouts)
print(manifest.get("csd_sign_convention"))  # → "positive_equals_extracellular_source"
```

**Validation:** Always verify CSD sign convention when comparing to external data or publications. Different fields/literature use opposite conventions.

---

## Calibration Labels and Constraints

### Source Calibration Status

The **source_calibration_status** field documents the empirical grounding of the source model.

| Status | Meaning | Biological Claim | Allowed? |
|--------|---------|------------------|----------|
| `uncalibrated_izhikevich_native_current` | Izhikevich native current, no empirical validation | None; computational scaffold | ✓ v0.2.24+ default |
| `uncalibrated_hh_native_current` | Hodgkin-Huxley native current, no empirical validation | None; computational scaffold | ✓ Reserved |
| `uncalibrated_jaxley_voltage_proxy` | Voltage trace proxy from external emitter, no empirical validation | None; computational scaffold | ✓ v0.2.22+ bridge |
| `calibrated_*` | Validated against empirical current/field data | Conditional; requires methods section & receipt | ✗ v0.2.24–v0.2.26; future |

**Current constraint:**
```
physical_amplitude_claim_allowed = False
```

This immutable field means:
- No claim that readout values are in physical units (pA, mV, μA/mm³)
- Voltage and current are computational proxies
- CSD and LFP are readout proxies (derived from proxy source + proxy field)
- Biological interpretation requires separate calibration and validation

---

## Mapping Equations to Implementation

### From Emitter Dynamics to Source Projection

**Equation chain:**

$$z(t) \xrightarrow{\text{Emitter}} \text{state} \xrightarrow{\text{Native current}} I(t) \xrightarrow{\text{Source projection}} q(x,t) \xrightarrow{\text{Readout}} \mathrm{CSD}(x,t)$$

**Implementation mapping:**

| Equation | Code Location | Manifest Field | Signals Field |
|----------|----------------|-----------------|----------------|
| $\frac{dz}{dt} = F_\theta(z, u, t)$ | `jaxfne.emitters.simulate_eig_izhikevich()` | `emitter_family`, `emitter_preset` | — |
| $I(t) = I_\mathrm{Iz}(z, \theta)$ | `jaxfne.emitters.simulate_eig_izhikevich()` internal | — | — |
| $q(x,t) = P_s[I(t), \chi(x)]$ | `jaxfne.fields.project_laminar_sources()` | `source_projection_mode`, `source_calibration_status` | `Signals.sources` |
| $\mathrm{CSD} = \nabla \cdot q$ | `jaxfne.fields.project_laminar_sources()` | `field_solver_status` | `Signals.field.csd` |

### Manifest Fields (Complete List)

**Source declaration:**
```python
manifest["source_calibration_status"]    # E.g. "uncalibrated_izhikevich_native_current"
manifest["source_projection_mode"]       # E.g. "proxy_no_field_solve"
manifest["source_decomposition"]         # E.g. "proxy_voltage_trace_not_current" (if applicable)
manifest["source_model"]                 # Struct: {"izhikevich_native_current_plus_spike_impulse_proxy": {...}}
```

**Field declaration:**
```python
manifest["field_solver_status"]          # E.g. "laminar_proxy_no_pde"
manifest["field_claim_level"]            # E.g. "proxy_readout_only"
manifest["boundary_condition"]           # E.g. "mean_zero_neumann"
manifest["gauge"]                        # E.g. "mean_zero"
manifest["conductivity_status"]          # E.g. "proxy" (not "calibrated_physical")
```

**Claim gates (immutable):**
```python
manifest["physical_amplitude_claim_allowed"]  # Always False in v0.2.24
manifest["claim_level"]                       # Always "computational_scaffold" in v0.2.24
manifest["truth_mode"]                        # Always "truth_safe_unverified" in v0.2.24
```

### Readout Report Fields

The `ProbeReport` returned by `compute_readout()` includes per-operator metadata:

```python
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("spikes", "spike_rate_hz"),
    jtfne.readout_spec("source", "source_abs_mean"),
    jtfne.readout_spec("csd", "csd_abs_mean"),
])

# Each ReadoutResult includes:
# - result.name (e.g., "spikes")
# - result.metric (e.g., "spike_rate_hz")
# - result.value (computed value)
# - result.status (e.g., "computed", "placeholder")
# - result.operator_status (if applicable)
```

---

## Minimal Code Example: Tracing Equations

### Example 1: Izhikevich → Source → CSD (Current Default)

```python
import jaxfne as jtfne

# Configuration declares source and field modes
cfg = (
    jtfne.configuration()
    .network(name="V1_proxy", kind="cortical_column", n=100)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    .probe(name="laminar_probe", modes=["spikes", "source", "CSD"])
)

model = jtfne.construct(cfg)

sim = jtfne.simulation(
    duration_ms=100.0,
    dt_ms=0.1,
    record_sources=True,
    record_fields=True,
)

# Simulate
signals = model.simulate(sim)

# Check source declaration
manifest = model.manifest(signals, [])
print(f"Source calibration: {manifest['source_calibration_status']}")
# → "uncalibrated_izhikevich_native_current"

print(f"Field solver: {manifest['field_solver_status']}")
# → "laminar_proxy_no_pde"

print(f"CSD sign convention: {manifest['csd_sign_convention']}")
# → "positive_equals_extracellular_source"

# Signals.sources [T, N] = Izhikevich native current + spike impulse
# Signals.field.csd [T, N] = ∇·q (proxy, not solved)
```

### Example 2: Jaxley Voltage Proxy → Source → LFP

```python
import jaxfne as jtfne

# External voltage trace (e.g., from Jaxley simulation)
voltage_trace = jnp.ones((1000, 100))  # [time, neurons]

# Convert to jaxfne Signals via bridge
from jaxfne.bridges import jaxley_trace_to_signals, JaxleyTraceSpec

spec = JaxleyTraceSpec(
    layout="time_by_unit",
    dt_ms=0.025,
    spike_threshold=0.0,
)

signals = jaxley_trace_to_signals(
    voltage_trace,
    spec=spec,
    source=None,  # Use voltage proxy
)

# Check source declaration (from bridge)
print(f"Source calibration: {signals.metadata.get('source_calibration_status')}")
# → "uncalibrated_jaxley_voltage_proxy"

print(f"Physical amplitude allowed: {signals.metadata.get('physical_amplitude_claim_allowed')}")
# → False

# Signals.sources [T, N] = voltage proxy (no field computation)
# Signals.field = None (no field in bridge; computed downstream if needed)
```

---

## Audit Checklist

Before releasing a model, verify:

- [ ] Source calibration status is declared and one of: uncalibrated_izhikevich_native_current, uncalibrated_hh_native_current, uncalibrated_jaxley_voltage_proxy, or None
- [ ] Source projection mode is declared (if source_calibration_status is not None)
- [ ] Field solver status is declared and is either "laminar_proxy_no_pde" or a future solver name
- [ ] Boundary condition and gauge are documented (metadata-only in v0.2.24)
- [ ] CSD sign convention is documented: positive = extracellular source (current flowing outward)
- [ ] physical_amplitude_claim_allowed is False
- [ ] No forbidden synaptic double-counting pattern in source computation
- [ ] Manifest JSON is NaN/Inf-free and JSON-safe

---

## See Also

- [Mathematical Glossary Flow](mathematical_glossary_flow.md) — Formal equations, term glossaries, bridge terms, claim boundaries
- [Probe Operators](probe_operators.md) — Readout modalities (SPK, Vm, source, LFP, CSD, EEG, MEG, EMM)
- [Output Bundles](output_bundles.md) — Manifest and report schema
- [Scope and Limitations](scope_and_limitations.md) — What jaxfne claims and does not claim
