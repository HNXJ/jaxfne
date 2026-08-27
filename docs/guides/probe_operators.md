# Probe operators

**Read out** spikes, membrane voltage, source currents, and field-proxy traces
(LFP, CSD, EEG, MEG, EMM) from a finished simulation. Eight named probe kinds
map to the multimodal outputs most laminar workflows need.

Each operator returns data plus a JSON-safe report (units, calibration status,
scope declarations, assumptions).

## Pipeline context

Probes sit at the end of the simulation chain:

```
Emitter → Source → Field → Probe → Objective → Optimizer
```

## Operator kinds

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

**Status (verified v0.4.8, development tree):** Simulated spike readout from emitter.

---

### Vm

**Purpose:** Expose membrane voltage or emitter state variable.

**Minimum output:**
- `kind: vm`
- `method: emitter_state_voltage_trace`
- `units_or_status: mV_or_model_state_variable`
- `operator_status: simulated_proxy`

**Status (verified v0.4.8, development tree):** State voltage from emitter; proxy readout.

---

### Source

**Purpose:** Expose current/source proxy or calibrated source trace used by the field layer.

**Minimum output:**
- `kind: source`
- `method: declared_source_projection_or_proxy`
- `source_decomposition: proxy_reduced_emitter` (or other declared mode)
- `source_calibration_status: proxy_izhikevich_current` (or calibrated status)

**Status (verified v0.4.8, development tree):** Proxy source from emitter state; computational readout.

---

### LFP-proxy

**Purpose:** Sample or average extracellular potential state at contacts.

**Acceptable implementation:**
- Point sample from `phi_e` proxy/solution
- Finite-contact average
- Laminar contact average over declared depths/layers

**Minimum report:**
- `kind: lfp_proxy`
- `method: point_or_finite_contact_phi_proxy`
- `units_or_status: proxy_voltage_units_or_V_if_calibrated`
- `field_solver_status: linear_solver`
- `amplitude_status: false`

**Important:** jaxfne uses `-proxy` terminology to declare operator status explicitly. Proxy-scale operators are computational readouts suitable for tutorial and validation workflows.

**Status (verified v0.4.8, development tree):** Laminar proxy-scale readout; calibration metadata available for calibrated workflows.

---

### CSD-proxy

**Purpose:** Estimate current-source density profile or second spatial derivative/divergence proxy.

**Acceptable implementation:**
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

**Status (verified v0.4.8, development tree):** Laminar proxy-scale readout; calibration metadata available for calibrated workflows.

---

### EEG-proxy

**Purpose:** Provide a simulated scalp-channel proxy readout using a declared toy or proxy lead field.

**Acceptable implementation:**

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
- `amplitude_status: false`

**Scope:** EEG readouts are computational proxies with declared toy leadfields. Calibration against empirical reference data is covered in [Limitations and future plans](../limitations_and_future_plans.md).

**Status (verified v0.4.8, development tree):** Simulated EEG-proxy readout; proxy-scale calibration for tutorial workflows.

---

### MEG-proxy

**Purpose:** Provide a relative linear map of scalar source ``Q``. This is not a physical MEG forward operator and does not construct oriented current.

**Acceptable implementation:**

```
y_meg(t, c) = sum_k L_meg[c, k] * Q_k(t)
```

**Minimum report:**
- `kind: meg_proxy`
- `method: relative_linear_map_proxy`
- `leadfield_status: toy_or_declared_proxy`
- `sensor_geometry_status: simulated_minimal`
- `orientation_convention: none`
- `units_or_status: arbitrary_proxy_units`
- `operator_status: simulated_proxy`
- `amplitude_status: false`

**Scope:** MEG-proxy readouts are relative linear maps on scalar ``Q``. They do not claim current orientation or physical MEG semantics. Calibration against empirical reference data is covered in [Limitations and future plans](../limitations_and_future_plans.md).

**Status (verified v0.4.8, development tree):** Simulated MEG-proxy readout; proxy-scale calibration for tutorial workflows.

---

### EMM-proxy

**Name:**

```
EMM = electromagnetic metabolism estimate proxy
```

**Interpretation:**

```
normalized electrophysiological activity / electromagnetic energy cost proxy
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
- `amplitude_status: false`

**Important:** EMM-proxy is valid for relative within-run comparisons. It represents a signaling-energy proxy suitable for optimization workflows in v0.2.x.

**Status (verified v0.4.8, development tree):** Normalized activity cost proxy; exploratory metric for optimization.

---

## Mathematical Forms

This section formalizes the eight operators in operator-form notation. These are computational forms declared as proxy-scale operators suitable for tutorial and model-development workflows.

### SPK (Spike Detection)

$$\mathrm{SPK}_n(t)=\mathbb{1}[V_n(t)\geq \theta_n]$$

Spike indicator for neuron $n$ at time $t$: 1 if membrane voltage exceeds threshold $\theta_n$, 0 otherwise.

### Vm (Membrane Voltage)

$$\mathrm{Vm}_n(t)=V_n(t)$$

Direct readout of state voltage from emitter $n$. Proxy readout.

### Source

$$S_n(t)=f_{\mathrm{source}}(x_n(t),\theta_n)$$

Source/current proxy derived from emitter state $x_n(t)$ and parameters $\theta_n$. Status: proxy-scale units in v0.2.x; calibration path documented in the calibration guide.

### LFP-proxy

$$\phi_{\mathrm{proxy}}(t,c)=\sum_{n=1}^{N}W_{cn}S_n(t), \quad \sum_{n=1}^{N}W_{cn}=1$$

Row-normalized kernel projection: potential at contact $c$ is a weighted sum of sources, with row sums equal to one. Status: laminar proxy via solver-free kernel projection.

### CSD-proxy

$$\mathrm{CSD}_{\mathrm{proxy}}(t,c) \approx \frac{\phi_{\mathrm{proxy}}(t,c+1)-2\phi_{\mathrm{proxy}}(t,c)+\phi_{\mathrm{proxy}}(t,c-1)}{(\Delta z)^2}$$

Second spatial derivative (Laplacian) of laminar potential proxy. Sign convention: `positive_equals_extracellular_source`.

### EEG-proxy

$$Y_{\mathrm{EEG\text{-}proxy}}(t,s) = \sum_{c=1}^{C} L^{\mathrm{EEG}}_{sc} \phi_{\mathrm{proxy}}(t,c)$$

Linear leadfield projection: each scalp electrode $s$ is a weighted combination of laminar potentials. $L^{\mathrm{EEG}}$ is a declared toy/proxy leadfield suitable for tutorial workflows.

### MEG-proxy

$$Y_{\mathrm{MEG\text{-}proxy}}(t,s) = \sum_{n=1}^{N} L^{\mathrm{MEG}}_{sn} o_n S_n(t)$$

Magnetometer readout: each sensor $s$ sums orientation-weighted source projections. $o_n$ is source orientation; $L^{\mathrm{MEG}}$ is a proxy leadfield.

### EMM-proxy (Activity/Cost Proxy)

$$\mathrm{EMM}_{\mathrm{proxy}}(t) = \alpha \|S(t,\cdot)\|_1 + \beta \|\phi_{\mathrm{proxy}}(t,\cdot)\|_1$$

Normalized electrophysiological activity cost proxy combining source and field energy norms. Represents signaling-energy proxy; valid for relative within-run comparisons and optimization workflows.

### Probe Report Sidecar

$$R_k = \{\mathrm{kind}, \mathrm{method}, \mathrm{units\_or\_status}, \mathrm{operator\_status}, \mathrm{physical\_amplitude\_statement\_allowed}, \ldots\}$$

Each operator returns a JSON-safe report $R_k$ declaring operator type, computation method, status (proxy/simulated), and statement constraints.

---

## Current Status: Simulated / Proxy

*Verified as of v0.4.8 development tree — status moved from title parenthetical to sentence.*

All eight operators remain simulated or proxy readouts as of v0.4.8:

| Operator | Status | Notes |
|----------|--------|-------|
| SPK | Simulated spike readout | From emitter threshold or declared spike array |
| Vm | Simulated voltage trace | From emitter state variable |
| source | Simulated source proxy | From emitter state or declared mode |
| LFP-proxy | Laminar proxy readout | Point sample or contact average; solver-free proxy mode |
| CSD-proxy | Laminar proxy readout | Second derivative or divergence proxy; solver-free proxy mode |
| EEG-proxy | Simulated linear projection | Proxy implementation with toy leadfield; calibration metadata exposed |
| MEG-proxy | Simulated linear projection | Proxy implementation with toy leadfield; calibration metadata exposed |
| EMM-proxy | Normalized cost proxy | Relative metric for optimization; signaling-energy proxy |

---

## Future Path

jaxfne preserves proxy operators as stable public readouts while adding clearer validation metadata, calibration specifications, and tutorial coverage.

Planned areas include:
- field/proxy mathematics and admissibility diagnostics
- calibration specification and reporting workflows
- Colab-ready tutorial stack and tutorial smoke tests
- laminar profile templates using literature-derived technical references, including Lichtenfeld et al. (2024) and Mendoza-Halliday et al. (2024). These templates support declared profile construction and tutorial design; they reference literature without stating reproduction of the reference datasets.
- operator status export, package audit, release candidate, and consolidated practical scaffold release

Calibration workflows and advanced tutorials are developed in the docs and examples.

**Beyond the current line:**
- Receptor-level synaptic dynamics (synaptic current modeling)
- Empirically calibrated source projection
- Hodgkin-Huxley ion channels with biophysical parameters
- Full-PDE field solver — see [Limitations and future plans](../limitations_and_future_plans.md)
- Empirically calibrated LFP/CSD readouts with validated sign conventions
- Physical leadfields from anatomically-validated head models
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

- Use `*-proxy` to denote declared computational operators: `lfp_proxy`, `csd_proxy`, `eeg_proxy`, `meg_proxy` are the canonical public labels.
- This terminology explicitly declares computational intent and prevents informal analogy with empirically validated readouts.
- All operators are "simulated" or "proxy" in v0.2.x. Statements of physical equivalence require separate calibration and validation evidence.
- EMM-proxy is valid for relative within-run comparisons; it represents a signaling-energy proxy in v0.2.x.

---

**Document version:** v0.4.8 (development tree)  
**Last updated:** 2026-07-03
