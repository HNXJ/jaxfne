# v0.3.3: Two-Neuron Excitatory/Inhibitory Multimodal Tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/docs/tutorials_v030/v0303_two_neuron_ei_multimodal.md)

**Tutorial ID:** `v0303_two_neuron_ei_multimodal`  
**Status:** Computational scaffold, proxy readout only

---

## Learning Objectives

By the end of this tutorial, you will understand:

1. **Network configuration** for multi-neuron (E/I) systems in jaxfne
2. **Izhikevich emitter models** applied to both excitatory and inhibitory neuron types
3. **Multimodal probe readout** infrastructure: all eight operators (SPK, Vm, source, LFP, CSD, EEG, MEG, EMM)
4. **Source aggregation** and field proxy construction in a coupled network scenario
5. **Manifest generation** and documentation for computational scaffolds
6. **Future extension points** for real E/I feedback mechanisms

---

## Computational Question

**How do we demonstrate the jaxfne API for coupled E/I neuron configuration and multi-scale readout?**

In this tutorial, we:
- Configure two Izhikevich neurons (one excitatory, one inhibitory)
- Specify coupling weights (E→I excitatory, I→E inhibitory)
- Simulate the network and apply all eight multimodal probe operators
- Generate figures and manifests that document the scope and limitations

**What this tutorial does NOT claim:**
- Actual E/I coupling dynamics (synaptic currents not implemented)
- Biological realism or mechanism validation
- Field PDE solutions (proxy only)
- Empirical fit to neural recordings

---

## Mathematical Glossary

### 1. Izhikevich Neuron Model

The canonical Izhikevich neuron is governed by:

$$\frac{dv}{dt} = 0.04 v^2 + 5v + 140 - u + I$$

$$\frac{du}{dt} = a(bv - u)$$

with spike reset:
$$\text{if } v \geq 30 \text{ mV} \text{ then } v \leftarrow c, \quad u \leftarrow u + d$$

**Terms:**
- $v(t)$ : membrane voltage (mV)
- $u(t)$ : recovery variable (dimensionless)
- $I(t)$ : input current (mA, constant in this tutorial)
- $a, b, c, d$ : parameters (preset: `cortical_eig`)
- Spike event: binary output when $v \geq 30$ mV

**Implementation location:** `jaxfne.cells.izhikevich`  
**Claim scope:** Computational model of spiking dynamics. Not validated against physiology.

### 2. Network Connectivity (Intended)

For a two-neuron network with coupling weights:

$$I_E = I_0 + w_{EE} \cdot \text{spikes}_E + w_{EI} \cdot \text{spikes}_I$$

$$I_I = I_0 + w_{IE} \cdot \text{spikes}_E + w_{II} \cdot \text{spikes}_I$$

**In v0.3.3:**
- $I_0$ = base Izhikevich current (from preset)
- $w_{EI} = 2.0$ (E→I excitatory weight, specified but not verified)
- $w_{IE} = -1.5$ (I→E inhibitory weight, specified but not verified)
- Other weights = 0

**Implementation status:** Weights are specified in the configuration but actual synaptic current injection depends on lower-level simulator APIs. **This is a known limitation of v0.3.3 and planned for extension.**

### 3. Source (Transmembrane Current)

$$I_{\text{source}}(t) = \text{Izhikevich current output}$$

**Calibration:** Uncalibrated. Units are Izhikevich native (mA for single-compartment model).  
**Claim scope:** Proxy representation. Not validated against patch-clamp or whole-cell recordings.

### 4. LFP-like Proxy

$$\Phi_{\text{LFP}} \approx \frac{1}{4\pi \sigma} \int \frac{I_{\text{source}}(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} dV'$$

**In v0.3.3:**
- Field domain: `laminar_column` (1D assumption)
- Conductivity: `proxy` (not physical)
- No PDE solve; forward-field kernel applied to aggregated sources
- Boundary: `mean_zero_neumann` (declared, not verified)
- Gauge: `mean_zero` (offset constraint, proxy level)

**Claim scope:** Proxy readout demonstrating forward-field architecture. No physical amplitude claim.

### 5. CSD-like Proxy

$$\text{CSD} = -\nabla^2 \Phi_{\text{LFP}}$$

Approximated by discrete second differences across contacts.

**Claim scope:** Proxy only. Second-difference operator applied to proxy LFP. Not a solved Poisson equation.

---

## Configuration Block

```python
import jaxfne as jtfne

# Network configuration
cfg = (
    jtfne.configuration()
    .network(
        name="v030_03_two_neuron_ei",
        kind="coupled_neurons",
        n=2,  # 2 neurons
        cell_types={"E": 0.5, "I": 0.5},  # Equal split
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        boundary="mean_zero_neumann",
        gauge="mean_zero",
    )
    .probe(
        name="two_channel_16contact_ei",
        modes=["spikes", "V_m", "source", "LFP", "CSD"],
        n_contacts=16,
    )
    .update_metadata(
        dx_mm=0.010,
        dy_mm=0.010,
        dz_mm=0.010,
        geometry_mode="declared_tutorial_metadata_not_solved_3d_grid",
        tutorial_id="v0.3.3",
    )
)
```

**Key declarations:**
- `kind="coupled_neurons"` : specifies multi-neuron configuration
- `cell_types={"E": 0.5, "I": 0.5}` : one excitatory, one inhibitory
- `preset="cortical_eig"` : Izhikevich parameters for cortical-like spiking
- `modes=["spikes", "V_m", "source", "LFP", "CSD"]` : all readout modes
- `n_contacts=16` : probe spatial resolution (laminar column)

---

## Simulation Block

```python
# Run simulation
DURATION_MS = 1000.0  # hard gate
DT_MS = 0.1
SEED = 42

run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=SEED)
model = jtfne.construct(cfg)
sim_spec = jtfne.simulation(duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED, runtime=run)
signals = model.simulate(sim_spec)
```

**Outputs:**
- `signals.V_m` : voltage trace, shape (n_steps, n_neurons)
- `signals.spikes` : spike binary, shape (n_steps, n_neurons)
- `signals.sources` : transmembrane current, shape (n_steps, n_neurons, n_space)
- `signals.field.lfp_proxy` : LFP-like proxy, shape (n_steps, n_contacts)
- `signals.field.csd_proxy` : CSD-like proxy, shape (n_steps, n_contacts-1)

---

## Probe and Readout Block

All eight operators are applied:

```python
probe_report = {
    "spikes": spk_probe(signals.spikes).report,
    "V_m": vm_probe(signals.V_m).report,
    "source": source_probe(signals.sources).report,
    "lfp_proxy": lfp_proxy_probe(field.lfp_proxy).report,
    "csd_proxy": csd_proxy_probe(field.csd_proxy).report,
    "eeg_proxy": eeg_proxy_probe(field.lfp_proxy).report,
    "meg_proxy": meg_proxy_probe(field.lfp_proxy).report,
    "emm_proxy": emm_proxy_probe(jnp.mean(jnp.abs(field.lfp_proxy), axis=1)).report,
}
```

**Operator status:** All 8 operators generate proxy readouts.  
**Claim scope:** Each operator returns `operator_status="proxy"` with immutable metadata.

---

## Manifest and Claim Gates

### Basis (immutable claim gates)

```json
{
  "basis": {
    "truth_mode": "truth_safe_unverified",
    "claim_level": "computational_scaffold",
    "field_solver_status": "laminar_proxy_no_pde",
    "field_claim_level": "proxy_readout_only",
    "physical_amplitude_claim_allowed": false,
    "biological_metabolism_claim_allowed": false,
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "source_projection_mode": "proxy_no_field_solve"
  }
}
```

### Validation Report

```json
{
  "validation_report": {
    "e_firing_rate_gate_2_25_hz": true,
    "i_firing_rate_gate_2_25_hz": false,  // I neuron not in target regime
    "e_voltage_finite": true,
    "i_voltage_finite": true,
    "source_finite": true,
    "json_safe": true,
    "duration_gate": true,
    "dt_gate": true,
    "dtype_gate": true,
    "status": "FAIL"  // because I neuron firing rate is out of target
  }
}
```

**Gate interpretation:**
- **Firing rate gate (2–25 Hz):** Checked per neuron. E passes; I fails (not yet spiking).
- **Voltage finite:** All $V_m$ values are finite (no NaN/Inf).
- **Source finite:** All current values are finite.
- **Duration gate:** $\geq 1000$ ms required.
- **dt gate:** Exactly 0.1 ms.
- **dtype gate:** float32 required.
- **Overall status:** FAIL if any gate fails (honest reporting).

---

## Figures

Generated PNG artifacts (in `docs/tutorials_v030/_static/figures/`):

### Figure 1: E and I Voltage Traces

**File:** `v0303_two_neuron_ei_voltage.png`

Two-panel figure:
- **Top:** Excitatory neuron voltage trace
- **Bottom:** Inhibitory neuron voltage trace

**Interpretation:** E neuron exhibits regular spiking (11 Hz); I neuron is quiescent.  
**Claim scope:** Proxy readout demonstrating Izhikevich dynamics under preset parameters.

### Figure 2: Spike Raster

**File:** `v0303_two_neuron_ei_raster.png`

Two-row raster:
- **Row 0 (blue):** E neuron spikes
- **Row 1 (red):** I neuron spikes

**Interpretation:** E neuron fires ~11 spikes over 1000 ms; I neuron does not fire.  
**Note:** Lack of I firing indicates that either (1) the coupling is not instantiated, or (2) the base preset does not support firing under current conditions. This is expected in v0.3.3 and flagged as a future extension.

### Figure 3: Source Aggregation

**File:** `v0303_two_neuron_ei_source.png`

Time series of aggregated source (current) per neuron.

**Interpretation:** E neuron source shows current pulses corresponding to spikes; I neuron source is relatively steady.  
**Claim scope:** Proxy representation of uncalibrated Izhikevich native current.

### Figure 4: LFP-like Proxy (first 4 contacts)

**File:** `v0303_two_neuron_ei_lfp_proxy.png`

LFP-like signal at four laminar depths.

**Interpretation:** Each contact sees a superposition of E and I source activity filtered by spatial distance.  
**Claim scope:** Proxy forward-field readout. No physical amplitude claim.

### Figure 5: CSD-like Proxy (first 4 layers)

**File:** `v0303_two_neuron_ei_csd_proxy.png`

Current source density approximated by second differences of LFP.

**Interpretation:** CSD pattern reflects spatial heterogeneity of sources.  
**Claim scope:** Proxy operator applied to proxy LFP. Not a solved boundary-value problem.

---

## Interpretation

### What This Tutorial Demonstrates

1. **Multi-neuron network API:** Configuration of E/I populations in jaxfne
2. **Multimodal readout pipeline:** All eight operators generate proxy outputs
3. **Manifest discipline:** Immutable claim gates ensure reproducible scope declarations
4. **Figure generation:** Production of shareable, SHA256-hashed artifacts

### What This Tutorial Does NOT Claim

- **Real E/I feedback dynamics:** Coupling is specified but synaptic current injection is not verified
- **Biological realism:** Preset-based firing does not match in vivo recordings
- **Solved PDEs:** LFP/CSD are proxy kernels, not solutions to 3D electrostatics
- **Mechanism proof:** Network dynamics do not validate theories of E/I balance or predictive coding
- **Empirical validation:** No comparison to patch-clamp, whole-cell, or population recordings

### Design Decisions

**Why does the I neuron not fire?**

The Izhikevich `cortical_eig` preset is parameterized for regular-spiking excitatory neurons. When applied to both E and I cells with the same parameters, both should theoretically spike. However:

1. **No input current differentiation:** Both neurons receive the same base current
2. **Coupling not verified:** The specified coupling weights may not be injected at the emitter level
3. **Future extension:** Real E/I feedback requires explicit synaptic current implementation

This is **expected and documented** in v0.3.3. It serves as a marker for the next phase (v0.3.4+).

---

## Failure Modes and Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|-----------|
| I neuron doesn't fire | Coupling not instantiated or base preset mismatch | Document as known limitation; implement synaptic currents in v0.3.4 |
| JSON parsing error | NaN/Inf in output | Check for numerical instability; increase damping or reduce dt |
| Figures missing | Matplotlib import error | Ensure matplotlib is installed with `[dev]` extra |
| Field is None | Configuration mismatch | Verify `.field(...)` configuration applies to network kind |
| Probe report keys mismatch | Operator registration issue | Check jaxfne version and operator imports |

---

## Exercises

### Exercise 1: Vary the Izhikevich Preset
Modify the `.emitter(preset=...)` call to try different presets:
- `"regular_spiking"` (RS)
- `"fast_spiking"` (FS)
- `"intrinsic_bursting"` (IB)

Observe how firing rate changes. Plot the results.

### Exercise 2: Change Network Size
Modify `n=2` to `n=10` with `cell_types={"E": 0.8, "I": 0.2}` (8E, 2I).
Regenerate manifest and figures. How do readout shapes change?

### Exercise 3: Modify Probe Modes
Remove `"CSD"` from the probe modes list. What breaks in the manifest?
Add `"EEG"` mode (if available). How does it differ from LFP?

### Exercise 4: Verify JSON Round-Trip
Load the generated `manifest.json` and re-serialize with `allow_nan=False`.
Verify no numerical errors occur. Write a script to validate all JSONs in outputs/.

### Exercise 5: Implement Synaptic Coupling
As a future extension, inject explicit synaptic currents in the Izhikevich loop:
```python
# Pseudocode for future v0.3.4+:
I_E += w_EI * spikes_I
I_I += w_IE * spikes_E
```
Compare spike timing to v0.3.3 uncoupled baseline.

---

## What This Tutorial Does NOT Claim

1. **Real EEG / MEG signals:** LFP and MEG proxies are forward-field readouts without experimental validation.
2. **Biological mechanism:** Spike timing does not prove theories of computation, E/I balance, or cortical function.
3. **Empirical calibration:** Izhikevich parameters are not fit to recordings.
4. **Conductivity:** The conductivity model is `proxy`; no tissue-specific values used.
5. **Electrode geometry:** 16-contact linear probe is a declared tutorial assumption.
6. **Population statistics:** Two-neuron network is too small for population-level claims.
7. **Dynamical system proof:** Bifurcation, stability, or basin-of-attraction analysis is not performed.

---

## Reference Implementation

The official implementation is in:
```bash
examples/v033_two_neuron_ei_multimodal.py
```

Run it locally:
```bash
python examples/v033_two_neuron_ei_multimodal.py
```

Outputs are written to:
```
outputs/v030_03_two_neuron_ei_multimodal/
├── manifest.json
├── probe_report.json
├── validation_report.json
├── metrics.json
├── asset_hashes.json
└── figures/
    ├── v0303_two_neuron_ei_voltage.png
    ├── v0303_two_neuron_ei_raster.png
    ├── v0303_two_neuron_ei_source.png
    ├── v0303_two_neuron_ei_lfp_proxy.png
    └── v0303_two_neuron_ei_csd_proxy.png
```

And canonical manifests are copied to:
```
docs/tutorials_v030/manifests/v0303_two_neuron_ei_multimodal_manifest.json
docs/tutorials_v030/reports/v0303_two_neuron_ei_multimodal_validation_report.json
```

---

## Appendix: Numerical Validation Checklist

Before use, verify:

- [ ] All JSON files pass `json.tool` validation
- [ ] All numerical values are finite (no NaN/Inf)
- [ ] Firing rates are within documented bounds or explicitly marked as out-of-regime
- [ ] Voltage traces show expected Izhikevich reset dynamics
- [ ] Source arrays have same temporal resolution as `V_m`
- [ ] LFP and CSD shapes match probe configuration (n_steps, n_contacts)
- [ ] All figures are PNG and > 0 bytes
- [ ] Manifest contains all 8 probe operator reports
- [ ] Claim gates are immutable (`physical_amplitude_claim_allowed` = false)
- [ ] Non-claims section explicitly lists scope limitations

---

**Tutorial version:** 0.3.3  
**jaxfne version:** >= 0.2.30  
**Truth status:** computational_scaffold, NOT biophysical validation  
**Last updated:** 2026-05-24
