# Single-Neuron Multimodal Proxy Tutorial

**Status:** v0.2.1 computational scaffold tutorial  
**File:** `examples/03_single_neuron_multimodal_probe.py`  
**Runtime:** ~5 seconds (CPU-safe, 100ms simulation)  
**Output:** `outputs/v023_single_neuron_multimodal/`

---

## Overview

This tutorial demonstrates the v0.2.1 **multimodal proxy probe stack** on a minimal single-neuron Izhikevich emitter.

The tutorial:
1. Constructs a single excitatory (E) neuron
2. Simulates 100 milliseconds of activity (CPU-safe)
3. Applies all **eight proxy operators** (v0.2.1 contract)
4. Generates a standardized output bundle with claim-status metadata

**Key learning:** How to call the eight proxy readout operators and interpret their reports as computational validation, not biological evidence.

---

## Running the Tutorial

```bash
cd /Users/hamednejat/workspace/main/jaxfne
python examples/03_single_neuron_multimodal_probe.py
```

Expected runtime: ~5 seconds on CPU.

---

## Output Files

The tutorial generates five files in `outputs/v023_single_neuron_multimodal/`:

| File | Purpose |
|------|---------|
| `manifest.json` | Model configuration, emitter parameters, field setup, and validation metadata |
| `probe_report.json` | All eight operator reports with operator status, data shapes, methods, and claim-status metadata |
| `metrics.json` | Basic signal statistics (spike rate, Vm mean/std) |
| `validation_report.json` | Claim-status metadata verification (all operators frozen as computational scaffold) |
| `asset_hashes.json` | SHA256 hashes for file integrity verification |

All JSON files are **strict** — no NaN or Inf values.

---

## The Eight Proxy Operators

Each operator returns a `ProbeReadout` with:
- `name` — readout identifier
- `kind` — operator type
- `data` — signal array
- `report` — JSON-safe metadata dict

### 1. SPK (Spike Readout)

```
kind:          spk
method:        threshold_or_emitter_spike_array
operator_status: simulated_proxy
units_or_status: binary_spike_indicator
```

Exposes binary spike events (0 = no spike, 1 = spike) at each timestep.

### 2. Vm (Membrane Voltage)

```
kind:          vm
method:        emitter_state_voltage_trace
operator_status: simulated_proxy
units_or_status: mV_or_native_model_voltage_status_declared
```

Exposes the Izhikevich neuron's native voltage variable (`V`). Not calibrated to biological membrane voltage; Izhikevich native units.

### 3. Source (Source Current Proxy)

```
kind:          source
method:        declared_source_projection_or_proxy
operator_status: simulated_proxy
source_calibration_status: uncalibrated_izhikevich_native_current
```

Exposes the proxy source/current used by the field solver. Derived from the emitter state; not empirically calibrated.

### 4. LFP-proxy (Local Field Potential Proxy)

```
kind:          lfp_proxy
method:        point_or_finite_contact_phi_proxy
operator_status: simulated_proxy
field_solver_status: laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
```

Samples the electric potential from the field domain. Uses a laminar proxy (no PDE solution). **Not validated amplitude.**

### 5. CSD-proxy (Current-Source-Density Proxy)

```
kind:          csd_proxy
method:        divergence_proxy_or_second_derivative_laminar
operator_status: simulated_proxy
CSD_sign_convention: positive_equals_extracellular_source
```

Estimates CSD as divergence of current density or second spatial derivative of potential. Includes explicit sign convention documentation.

### 6. EEG-proxy (Scalp Channel Proxy)

```
kind:          eeg_proxy
method:        linear_leadfield_proxy
operator_status: simulated_proxy
leadfield_status: toy_or_declared_proxy
sensor_geometry_status: simulated_minimal
physical_amplitude_claim_allowed: false
```

Simulated scalp-level readout using a toy leadfield (linear projection). No real sensor geometry or validated amplitude.

### 7. MEG-proxy (Magnetometer Proxy)

```
kind:          meg_proxy
method:        linear_current_orientation_proxy
operator_status: simulated_proxy
leadfield_status: toy_or_declared_proxy
orientation_convention: declared
physical_amplitude_claim_allowed: false
```

Simulated magnetometer readout using current-orientation projection. No real sensor array or validated amplitude.

### 8. EMM-proxy (Electromagnetic Metabolism Estimate Proxy)

```
kind:          emm_proxy
method:        normalized_activity_field_source_cost_proxy
operator_status: simulated_proxy
biophysical_calibration_status: uncalibrated_proxy
physical_amplitude_claim_allowed: false
```

Normalized cost proxy combining spike rate, source magnitude, and field energy. **Not biological metabolism.** Valid for relative within-run comparisons only.

---

## Understanding Claim-Status Metadata

All eight operators freeze the same claim-status metadata:

```yaml
claim_level:                      computational_scaffold
field_solver_status:              laminar_proxy_no_pde
field_claim_level:                proxy_readout_only
source_calibration_status:        uncalibrated_izhikevich_native_current
physical_amplitude_claim_allowed: false
```

**What this means:**

- ✓ The operators implement a **computational scaffold** for TFNE workflows.
- ✓ All field solutions are **laminar proxies** (no full PDE solver).
- ✓ All readouts are **proxy-only** — no physical amplitude validation.
- ✓ Sources are **uncalibrated** — native Izhikevich units, not biophysical current.
- ✓ **No biological mechanism claims** or empirical validation.

**In your own work:**

- Use these readouts for **algorithm development, visualization, and teaching**.
- Do NOT cite these readouts as evidence for neural computation or brain mechanisms.
- If calibrating for publication, add separate validation evidence (voltage clamp, patch clamp, etc.).

---

## Running Your Own Single-Neuron Example

To adapt this tutorial for your own configuration:

1. Modify the `configuration()` call:
   ```python
   cfg = (
       jtfne.configuration()
       .network(name="...", kind="...", n=..., cell_types={...})
       .emitter(family="izhikevich", preset="...")
       .field(domain="...", conductivity="...", ...)
       .probe(name="...", modes=[...])
   )
   ```

2. Adjust simulation duration if needed:
   ```python
   sim = jtfne.simulation(duration_ms=200.0, dt_ms=0.1, seed=42)
   ```

3. Call the eight operators with your signals.

4. JSON output is automatically strict (no NaN/Inf).

---

## Next Steps

- See `examples/02_spectrolaminar_oddball_scaffold.py` for laminar cortical circuits with task paradigms.
- See `docs/probe_operators.md` for the full operator contract specification.
- See `docs/releases/v0.2.1.md` for release notes and API examples.

---

## Scientific Truth Status

**This tutorial is NOT:**
- Biologically validated
- Empirically calibrated
- A whole-brain model
- Evidence of neural mechanisms

**This tutorial IS:**
- A reproducible computational scaffold
- A reference for multimodal readout workflows
- Suitable for algorithm development and teaching
- A starting point for calibrated variants (future)

---

**Related files:**
- `examples/03_single_neuron_multimodal_probe.py` — example code
- `jaxfne/fields.py` — probe operator implementations
- `docs/probe_operators.md` — operator contract details
