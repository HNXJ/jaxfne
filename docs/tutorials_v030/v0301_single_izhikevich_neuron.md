# v0.3.1: Single Izhikevich Neuron Tutorial

**Tutorial ID:** v0301_single_izhikevich_neuron  
**Version:** v0.3.1  
**jaxfne version:** 0.2.30  
**Date:** 2026-05-23  
**Truth status:** `truth_safe_unverified`  

[Open in Colab](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/notebooks/v030/v0301_single_izhikevich_neuron.ipynb)

---

## Section 1: Learning Objectives

After completing this tutorial, you will:

1. Understand the Izhikevich neuron model equations and parameters
2. Configure and simulate a single Izhikevich neuron using jaxfne
3. Extract and interpret neuronal readouts (voltage, spikes, firing rate)
4. Validate simulation outputs against computational acceptances gates (firing rate, numerical stability, JSON safety)
5. Recognize the difference between simulation and biological validation
6. Work with jaxfne manifest and validation report infrastructure

---

## Section 2: Biological/Computational Question

**Question:** How does the Izhikevich model neuron respond to intrinsic dynamics and recurrent inputs?

**Context:** The Izhikevich model is a reduced two-variable spiking model that captures key biophysical behaviors (spike timing, post-spike dynamics) with low computational cost. This tutorial demonstrates a single neuron in isolation under the `cortical_eig` (cortical eigenmode) preset, which provides typical recurrent and drive currents for a regular-spiking cortical neuron.

**Approach:** We simulate a single neuron for 1 second of biological time, measure its spike output and voltage trajectory, and validate the outputs against a set of hard computational gates. We then interpret the results **as a computational proxy**, explicitly stating what we do NOT claim.

---

## Section 3: Mathematical Glossary

### Izhikevich Voltage Dynamics

**Equation (displayed):**

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I(t)$$

**Term Glossary:**
- $v$ = membrane voltage (mV)
- $u$ = recovery variable (dimensionless, fast inactivation)
- $I(t)$ = input current (pA; native current, not empirically calibrated)
- $0.04v^2 + 5v + 140$ = voltage-dependent dynamics (cubic-linear, phenomenological)

**Worded Equation:**
The voltage changes based on cubic-quadratic intrinsic dynamics (fast), recovery/inactivation feedback (slow), and input current.

**Implementation Location:**
`jaxfne/emitters.py::IzhikevichNeuron.step()`  or configured via `jtfne.emitter(family='izhikevich', preset='cortical_eig')`

**Claim Boundary:**
This is a mathematical model fit to spike timing and shape, not a derivation from first-principles electrophysiology. The current $I(t)$ is a composite of intrinsic drive and recurrent synaptic input, internally configured by the network preset, not validated against measured patch-clamp data.

---

### Izhikevich Recovery Dynamics

**Equation (displayed):**

$$\frac{du}{dt} = a(bv - u)$$

**Term Glossary:**
- $a$ = time scale (s$^{-1}$)
- $b$ = voltage sensitivity of recovery (dimensionless)
- $v$ = voltage (mV)
- $u$ = recovery variable (dimensionless)

**Worded Equation:**
Recovery evolves slowly and is sensitive to voltage; it implements negative feedback and spike-timing aftereffects.

**Implementation Location:**
`jaxfne/emitters.py::IzhikevichNeuron.step()`, using the `a` and `b` parameters from the `cortical_eig` preset.

**Claim Boundary:**
The recovery variable is a phenomenological abstraction, not a biophysically identified current or channel. Different values of $a$ and $b$ yield different spiking phenotypes; we use the preset `cortical_eig` without claiming biological tuning to a specific cell type.

---

### Spike Condition and Reset

**Equation (displayed):**

$$\text{if } v \geq 30 \text{ mV, then } v \leftarrow c, \quad u \leftarrow u + d$$

**Term Glossary:**
- $v$ = voltage (mV)
- $c$ = reset voltage (mV)
- $d$ = recovery reset increment (dimensionless)

**Worded Equation:**
When voltage reaches a threshold (30 mV), it resets to a resting value and recovery is incremented.

**Implementation Location:**
`jaxfne/emitters.py::IzhikevichNeuron.step()`, spike condition implemented as a JAX boolean mask.

**Claim Boundary:**
The threshold (30 mV) and reset values are mathematical hyperparameters, not biophysically measured action potential kinetics. No ionic channels are modeled.

---

## Section 4: Canonical Import

All jaxfne code in this tutorial uses the **required canonical alias**:

```python
import jaxfne as jtfne
```

**Forbidden patterns:**
- ❌ `import jaxfne` (bare, no alias)
- ❌ `from jaxfne import *` (wildcard)
- ❌ `import jaxfne as jtnfe` (typo)
- ❌ `import jaxfne as jtFNE` (wrong case)

---

## Section 5: Configuration and Model Setup

```python
import jaxfne as jtfne

# Configure a single Izhikevich neuron
cfg = (
    jtfne.configuration()
    .network(name="SingleNeuron", kind="single", n=1)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    .probe(name="single_channel_16contact", modes=["spikes", "V_m", "source", "CSD"], n_contacts=16)
)

model = jtfne.construct(cfg)
```

**Configuration details:**
- **Network:** 1 single neuron (no recurrent connectivity within the neuron; recurrent currents come from the network preset)
- **Emitter:** Izhikevich family with `cortical_eig` preset (provides regular-spiking parameters and internal drive/recurrent currents)
- **Field:** Laminar column domain, proxy field solver (no PDE solution), mean-zero Neumann boundary condition
- **Probe:** Multimodal readout with 16 contact sites (laminar depth), extracting spikes, voltage, source current proxy, and CSD proxy

---

## Section 6: Simulation and Data Generation

```python
# Simulation parameters
duration_ms = 1000.0
dt_ms = 0.1
seed = 42

sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
signals = model.simulate(sim)
```

**Simulation details:**
- Duration: 1000 ms (1 second of biological time)
- Time step: 0.1 ms
- Deterministic seed: 42 (ensures reproducibility)
- Outputs:
  - `signals.V_m`: voltage trace [T=10000, N=1]
  - `signals.spikes`: spike raster [T=10000, N=1]
  - `signals.source`: proxy source current [T=10000, X=16 laminar contacts] (if present)

---

## Section 7: Probe and Multimodal Readout

```python
# Extract readouts
spike_count = model.compute_readout(signals, [jtfne.readout_spec("spikes", "spike_count")])
firing_rate_hz = (spike_count / duration_ms) * 1000.0

# Voltage statistics
V_min = signals.V_m.min()
V_max = signals.V_m.max()
V_mean = signals.V_m.mean()
```

**Readouts computed:**
- **Spikes:** Binary spike raster, detected as voltage ≥ 30 mV
- **Voltage:** Membrane potential trace
- **Firing rate:** (# spikes / duration) × 1000 Hz
- **Source current:** Proxy native Izhikevich current (not empirically calibrated)
- **CSD-proxy:** Spatial divergence of source (not solved via Poisson equation)

**Gate validation:**
- Firing rate: **11.0 Hz** (within 2–25 Hz gate) ✓
- All values finite: **True** ✓
- Voltage range: **[-76.0, 25.6] mV** (biologically plausible range, but not validated) ✓

---

## Section 8: Manifest and Claim Gates

The manifest captures immutable claim gates:

```json
{
  "truth_mode": "truth_safe_unverified",
  "claim_level": "computational_scaffold",
  "physical_amplitude_claim_allowed": false,
  "field_solver_status": "laminar_proxy_no_pde",
  "field_claim_level": "proxy_readout_only",
  "firing_rate_hz": 11.0,
  "firing_rate_gate_2_25_hz": true,
  "neuron_model": "izhikevich",
  "preset": "cortical_eig",
  "jaxfne_version": "0.2.30"
}
```

**Claim gates (frozen for v0.3.1):**
- `truth_mode`: truth_safe_unverified
- `claim_level`: computational_scaffold
- `physical_amplitude_claim_allowed`: False
- `field_solver_status`: laminar_proxy_no_pde
- `biological_validation_claim_allowed`: False

These gates are **immutable** and define the scope of the tutorial. No biological mechanism is proven; no physical field is solved; no external real-world validation is claimed.

---

## Section 9: Figures and Artifacts

### Voltage Trace
![v0301_single_neuron_voltage](../../outputs/v031_single_neuron/figures/v0301_single_neuron_voltage.png)

**Figure description:** Membrane voltage over 1 second. The neuron exhibits spontaneous spiking driven by internal dynamics and network currents. No external stimulus is applied; the spike pattern emerges from the configuration parameters.

### Spike Raster
![v0301_single_neuron_raster](../../outputs/v031_single_neuron/figures/v0301_single_neuron_raster.png)

**Figure description:** Spike times across the 1-second simulation. 11 spikes were detected, corresponding to an 11 Hz mean firing rate. The spike pattern is irregular, consistent with the Izhikevich model's built-in adaptation.

---

## Section 10: Interpretation and Analysis

### Firing Rate Evidence

The simulation produced **11 spikes over 1000 ms**, yielding a firing rate of **11.0 Hz**. This value:
- Passes the 2–25 Hz hard acceptance gate ✓
- Falls within the range of cortical pyramidal neurons in vivo (typically 0.1–30 Hz, depending on layer and stimulus)
- Emerges purely from the network preset and intrinsic dynamics (no tuning to a specific experimental dataset)

### Voltage Dynamics

The voltage trace exhibits:
1. **Baseline variability** around –67 mV (resting potential)
2. **Upstroke peaks** reaching ~25 mV (action potentials)
3. **Adaptation** manifest as variable inter-spike intervals (consistent with $u$ variable feedback)

These dynamics are **computational signatures**, not validated against measured whole-cell recordings.

### No Physical Field Solution

The CSD-proxy output is computed without solving the Poisson equation:

$$\nabla^2 \phi = -\frac{\nabla \cdot \mathbf{J}_e}{\sigma}$$

Instead, the laminar_proxy_no_pde mode computes a contact-normalized projection of source current. This is suitable for visualization and order-of-magnitude reasoning, but **not a physical electric potential**.

---

## Section 11: Failure Modes and Edge Cases

### Firing Rate Outside 2–25 Hz Range

If the simulation yielded <2 Hz or >25 Hz, the tutorial would:
1. Flag `firing_rate_gate_2_25_hz: false` in the manifest
2. Either re-tune the network parameters (e.g., increase drive current, modify $a$/$b$) or accept the result as an alternative proof of concept

In this run, the firing rate is well within bounds, so no tuning was needed.

### Voltage Numerical Instability

If any voltage value became NaN or Inf:
1. The simulation would fail JSON serialization (`allow_nan=False`)
2. The validation report would flag `json_safe: false`
3. The tutorial would report `V031_BLOCKED_VALIDATION_FAILURE`

In this run, all voltages are finite.

### Missing Spike Detection

If the spike threshold (30 mV) is never reached, the neuron is "silent" (0 Hz). This could arise from:
- Insufficient drive current
- Parameters that yield a stable fixed point below threshold

For the cortical_eig preset, spiking is expected and observed.

---

## Section 12: Exercises and Extensions

1. **Modify the preset:** Change `preset="cortical_eig"` to `preset="thalamic"` or `preset="chattering"` and observe how firing rate and voltage shape change.

2. **Change network size:** Extend to `n=100` neurons and observe recurrent dynamics (requires a field solver for biophysically accurate CSD; currently uses proxy).

3. **Add a synaptic input:** Introduce external synaptic currents via a drive matrix (requires a more complex network setup; defer to v0.3.x).

4. **Validate against experiment:** Collect whole-cell recordings from a cortical pyramidal cell and fit the Izhikevich parameters to match observed voltage dynamics (exceeds the scope of a computational scaffold; requires v0.3.x calibration phase).

---

## Section 13: Non-Claim Statement (Mandatory)

### What this tutorial IS

✓ An executable, reproducible computational scaffold  
✓ A demonstration of how the Izhikevich model operates in jaxfne  
✓ A teaching tool for understanding single-neuron dynamics  
✓ Suitable for exploration and hypothesis generation  
✓ A foundation for more complex models (multi-compartment, networks, learning)

### What this tutorial IS NOT

❌ A biological validation or calibration of the Izhikevich model  
❌ A proof that the Izhikevich model describes real neurons  
❌ An empirical study; no comparison to experimental whole-cell data  
❌ A claim of physiological realism (other than qualitative spike shape)  
❌ Evidence for any specific neural computation or mechanism  
❌ A complete field simulation (no Poisson PDE solution; proxy only)  
❌ A brain simulator or large-scale network model

### Scientific boundaries (immutable as of v0.2.30)

- **truth_mode:** truth_safe_unverified
- **claim_level:** computational_scaffold
- **physical_amplitude_claim_allowed:** False
- **field_solver_status:** laminar_proxy_no_pde (no Maxwell/Poisson solvers)
- **source_calibration_status:** uncalibrated (teaching proxy, not measured)
- **biological_validation:** Not performed; not claimed

### Invitation

If you:
- Find a bug in the code (please open a GitHub issue)
- Identify misleading language (please request a clarification)
- Have ideas for extensions or follow-up tutorials (v0.3.2+)

Please reach out via GitHub Issues or Discussions.

---

**End of v0.3.1: Single Izhikevich Neuron Tutorial**

---

**Manifest:** [v0301_single_izhikevich_neuron_manifest.json](../../docs/tutorials_v030/manifests/v0301_single_izhikevich_neuron_manifest.json)  
**Validation Report:** [v0301_single_izhikevich_neuron_validation_report.json](../../docs/tutorials_v030/reports/v0301_single_izhikevich_neuron_validation_report.json)

