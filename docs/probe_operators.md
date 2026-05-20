# Probe Operators in jaxfne v0.2.1+

## Overview

The probe layer is a first-class component of the TFNE pipeline:

```
Emitter → Source → Field → Probe → Objective → Optimizer
```

Eight probe operators expose different aspects of neural/field state as named readouts. Each operator returns data plus a JSON-safe report declaring operator status, units, calibration, truth gates, and assumptions.

## Operator Kinds

### SPK

**Purpose:** Expose spike events or spike matrix.

**Minimum output:**
- `kind: spk`
- `method: threshold_or_emitter_spike_array`
- `units_or_status: binary_spike_indicator`
- `operator_status: simulated_proxy`

**Shape examples:**
- Spike matrix: `[T, N]` (time × neurons)
- Event table: rows with timestamp, neuron_id, area, layer, cell_type

**v0.2.1 status:** Simulated spike readout from emitter.

---

### Vm

**Purpose:** Expose membrane voltage or native reduced-emitter state.

**Minimum output:**
- `kind: vm`
- `method: emitter_state_voltage_trace`
- `units_or_status: mV_or_native_model_voltage`
- `operator_status: simulated_proxy`

**Truth note:** If the emitter voltage is not physical membrane voltage, the report states this.

**v0.2.1 status:** Native state voltage from emitter; not physical unless empirically calibrated.

---

### Source

**Purpose:** Expose current/source proxy or calibrated source trace used by the field layer.

**Minimum output:**
- `kind: source`
- `method: declared_source_projection_or_proxy`
- `source_decomposition: proxy_reduced_emitter` (or other declared mode)
- `source_calibration_status: uncalibrated_izhikevich_native_current` (or calibrated status)

**v0.2.1 status:** Proxy source from emitter native state; not calibrated to physical units.

---

### LFP-proxy

**Purpose:** Sample or average extracellular potential-like state at contacts.

**Acceptable v0.2.1 implementation:**
- Point sample from `phi_e` proxy/solution
- Finite-contact average
- Laminar contact average over declared depths/layers

**Minimum report:**
- `kind: lfp_proxy`
- `method: point_or_finite_contact_phi_proxy`
- `units_or_status: proxy_voltage_units_or_V_if_calibrated`
- `field_solver_status: laminar_proxy_no_pde`
- `physical_amplitude_claim_allowed: false`

**Important:** v0.2.1 uses `-proxy` terminology, not `-like`. This declares the operator status: it is a computational proxy, not a validated LFP measure.

**v0.2.1 status:** Laminar proxy readout; not empirically calibrated.

---

### CSD-proxy

**Purpose:** Estimate current-source density-like source profile or second spatial derivative/divergence proxy.

**Acceptable v0.2.1 implementation:**
- `div(J_e)` when `J_e` exists
- Second spatial derivative of laminar potential proxy
- Source-profile proxy with explicit sign convention

**Minimum report:**
- `kind: csd_proxy`
- `method: divergence_proxy_or_second_derivative_laminar`
- `CSD_sign_convention: positive_equals_extracellular_source` (or other declared convention)
- `units_or_status: proxy_A_m^-3_or_proxy_units`

**Required tests:**
- Constant potential gives near-zero CSD-proxy output
- Linear potential gives near-zero second derivative
- Finite output
- Sign convention exported

**v0.2.1 status:** Laminar proxy readout; not empirically calibrated.

---

### EEG-proxy

**Purpose:** Provide a simulated scalp-channel-like readout using a declared toy or proxy lead field.

**Acceptable v0.2.1 implementation:**

```
y_eeg(t, c) = sum_k L_eeg[c, k] * s_k(t)
```

where `s_k(t)` is a declared source/current/potential feature and `L_eeg` is a toy or user-declared projection matrix.

**Minimum report:**
- `kind: eeg_proxy`
- `method: linear_leadfield_proxy`
- `leadfield_status: toy_or_declared_proxy`
- `sensor_geometry_status: simulated_minimal`
- `units_or_status: arbitrary_proxy_units`
- `operator_status: simulated_proxy`
- `physical_amplitude_claim_allowed: false`

**Forbidden in v0.2.1 docs/reports unless backed by real model evidence:**
- `real EEG`
- `validated EEG`
- `sensor-level EEG amplitude`

**v0.2.1 status:** Simulated EEG-proxy readout; not validated against real data.

---

### MEG-proxy

**Purpose:** Provide a simulated magnetometer-like readout using a declared current-orientation or lead-field proxy.

**Acceptable v0.2.1 implementation:**

```
y_meg(t, c) = sum_k L_meg[c, k] * j_oriented_k(t)
```

**Minimum report:**
- `kind: meg_proxy`
- `method: linear_current_orientation_proxy`
- `leadfield_status: toy_or_declared_proxy`
- `sensor_geometry_status: simulated_minimal`
- `orientation_convention: declared`
- `units_or_status: arbitrary_proxy_units`
- `operator_status: simulated_proxy`
- `physical_amplitude_claim_allowed: false`

**Forbidden in v0.2.1 docs/reports unless backed by real model evidence:**
- `real MEG`
- `validated MEG`
- `sensor-level MEG field strength`

**v0.2.1 status:** Simulated MEG-proxy readout; not validated against real data.

---

### EMM-proxy

**Name:**

```
EMM = electromagnetic metabolism estimate proxy
```

**Interpretation in v0.2.1:**

```
normalized electrophysiological activity / electromagnetic energy-like cost proxy
```

**Possible first operator:**

```
EMM(t) = w_spk * normalized_spike_rate(t)
       + w_src * normalized(||source(t)||_1 or ||source(t)||_2^2)
       + w_field * normalized(||grad(phi_e)(t)||_2^2 or ||J_e(t)||_2^2)
       + w_syn * normalized_synaptic_activity_proxy(t)
```

**Minimum report:**
- `kind: emm_proxy`
- `method: normalized_activity_field_source_cost_proxy`
- `units_or_status: normalized_proxy_units`
- `biophysical_calibration_status: uncalibrated_proxy`
- `operator_status: simulated_proxy`
- `physical_amplitude_claim_allowed: false`

**Important:** EMM-proxy is valid for relative within-run comparisons. It is not biological metabolism in v0.2.x.

**v0.2.1 status:** Normalized activity cost proxy; exploratory metric for optimization.

---

## Common Report Contract

Every probe operator returns a JSON-safe report with these fields:

```yaml
name: string (operator name)
kind: string (spk | vm | source | lfp_proxy | csd_proxy | eeg_proxy | meg_proxy | emm_proxy)
operator_status: simulated_proxy | physical_forward_model | calibrated_empirical
method: string
data_shape: string (shape of output data)
units_or_status: string
calibration_status: string
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
source_calibration_status: string
source_projection_mode: string
source_decomposition: string
field_solver_status: string
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
assumptions: list[string]
```

Optional fields (when applicable):

```yaml
leadfield_status: toy_or_declared_proxy | physical_forward_model | not_applicable
sensor_geometry_status: simulated_minimal | declared_geometry | not_applicable
orientation_convention: string_or_null
CSD_sign_convention: string_or_null
contact_depths_or_layers: list_or_null
normalization: string_or_null
```

---

## Truth Gates (Frozen in v0.2.x)

All probe operators preserve these immutable gates:

| Gate | Value |
|------|-------|
| `truth_mode` | `truth_safe_unverified` |
| `claim_level` | `computational_scaffold` |
| `field_claim_level` | `proxy_readout_only` |
| `physical_amplitude_claim_allowed` | `False` |

These gates mean:
- **No biological mechanism validation.** The pipeline is a computational scaffold, not a model of real physiology.
- **No calibrated amplitude claims.** Sources, fields, and readouts are proxy operators unless explicitly calibrated.
- **No whole-brain simulation.** The field solver is a laminar proxy, not a full-brain PDE solution.

---

## Current Status: v0.2.1 Simulated / Proxy

All eight operators in v0.2.1 are simulated or proxy readouts:

| Operator | Status | Notes |
|----------|--------|-------|
| SPK | Simulated spike readout | From emitter threshold or declared spike array |
| Vm | Simulated voltage trace | From emitter native state, not calibrated |
| source | Simulated source proxy | From emitter native state or declared mode |
| LFP-proxy | Laminar proxy readout | Point sample or contact average; no PDE |
| CSD-proxy | Laminar proxy readout | Second derivative or divergence proxy; no PDE |
| EEG-proxy | Simulated linear projection | Toy leadfield; not validated against real EEG |
| MEG-proxy | Simulated linear projection | Toy leadfield; not validated against real MEG |
| EMM-proxy | Normalized cost proxy | Relative metric for optimization; not biological metabolism |

---

## Future Path to Calibrated/Physical Operators

**v0.2.x → v0.3.x:**
- Receptor-level synaptic dynamics (synaptic current vs. native emitter state)
- Empirically calibrated source projection
- Hodgkin-Huxley ion channels with biophysical parameters
- Optional full-PDE field solver (resistive/impedance forward model)

**v0.3.x → v0.4.x+ (longer term):**
- Empirically calibrated LFP/CSD read out with validated sign conventions
- Real MEG/EEG leadfields from head model (not toy projection)
- Whole-brain connectivity integrated with laminar columns
- Plasticity and learning rules with evidence-backed parameters

---

## API Example

```python
import jaxfne
from jaxfne.fields import spk_probe, lfp_proxy_probe, emm_proxy_probe

# Simulate a small network
cfg = jaxfne.configuration().network(n=32).emitter(...).field(...).probe(...)
model = jaxfne.construct(cfg)
signals = model.simulate(jaxfne.simulation(duration_ms=1000.0))

# Apply probe operators
spk_readout = spk_probe(signals.spikes)
lfp_readout = lfp_proxy_probe(signals.field.lfp)
emm_readout = emm_proxy_probe(signals.field.lfp + signals.spikes.mean(axis=1))

# Each readout has JSON-safe report
print(spk_readout.report)
print(lfp_readout.report)
print(emm_readout.report)
```

---

## Terminology Notes

- Use `*-proxy` to denote declared computational operators, not `*-like` informal analogy.
- `LFP-proxy`, `CSD-proxy`, `EEG-proxy`, `MEG-proxy` are the canonical public labels in v0.2.1+.
- Avoid `real EEG`, `real MEG`, `validated EEG`, `validated MEG`, `biological metabolism` unless backed by empirical evidence.
- All operators are "simulated" or "proxy" in v0.2.x, making the computational intent clear.

---

**Document version:** v0.2.1  
**Truth mode:** truth_safe_unverified  
**Last updated:** 2026-05-19
