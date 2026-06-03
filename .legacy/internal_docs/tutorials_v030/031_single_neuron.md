# jaxfne v0.3.1: Single-Neuron Tutorial

**A deep-dive into the reduced Izhikevich emitter, voltage dynamics, spiking behavior, and proxy readouts.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v031_single_neuron.ipynb)

---

## 1. Learning Objectives

After completing this tutorial, you will understand:

1. **Reduced spiking emitter** — the Izhikevich model as a phenomenological two-variable system capturing voltage and recovery dynamics.
2. **Voltage and recovery dynamics** — how membrane potential ($v$) and recovery variable ($u$) evolve over time and interact.
3. **Spike-and-reset behavior** — how threshold-crossing triggers spike events and state resets.
4. **Source proxy from emitter activity** — how neuron voltage and spikes map to current sources for readout.
5. **Proxy readout extraction** — how to sample simulated activity using MUA-proxy, source-proxy, and LFP-proxy operators.
6. **Finite-time behavior** — how to interpret simulation outputs and recognize successful spiking dynamics.

---

## 2. Biological/Computational Question

**How does a reduced spiking emitter transform native drive into voltage dynamics, spikes, and proxy readouts?**

This tutorial addresses this question by simulating a single Izhikevich neuron over 1 second of biological time with constant background drive. We observe:

- The voltage trace and recovery variable trajectory.
- Regular spike timing and reset behavior.
- How emitter activity maps to source proxies (current-like activity).
- How proxy readouts (MUA, LFP) capture neuron-level dynamics.

---

## 3. Mathematical Glossary Flow

### 3.1 Izhikevich Emitter Dynamics

**Formal equations:**
$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I_{\mathrm{native}}$$
$$\frac{du}{dt} = a(bv - u)$$
$$\mathrm{if} \ v \geq 30 \mathrm{mV}: \text{spike} = 1, \ v \leftarrow c, \ u \leftarrow u + d$$

**Definition of terms:**
- $v(t)$ — Membrane potential (mV); voltage-like variable.
- $u(t)$ — Membrane recovery variable (dimensionless slow variable).
- $a, b$ — Recovery time scale and sensitivity parameters.
- $c$ — Reset potential after spike (mV).
- $d$ — Recovery variable increment after spike.
- $I_{\mathrm{native}}$ — Native current drive (unsigned relative current value; not empirically calibrated).

**Worded equation:**
The change in membrane potential is driven by a quadratic voltage activation term, linear voltage feedback, a constant offset, recovery variable feedback, and background current input. The recovery variable evolves as a slow system tracking a scaled version of the membrane potential. When voltage exceeds 30 mV, the neuron spikes: a spike event is recorded, voltage is reset to $c$, and the recovery variable is incremented by $d$.

**Implementation location:**
[jaxfne/emitters.py](../jaxfne/emitters.py)

**Scope boundary:**
Reduced phenomenological spiking model; not full conductance-based biophysical reconstruction. The native current drive is a tutorial-local parameter, not empirically calibrated against biological neurons.

---

### 3.2 Source and Proxy Readout Mapping

**Formal equation:**
$$Y_c(t) = P_c[v(t), \mathrm{spikes}(t), \mathrm{source\_proxy}(t)](t)$$

**Definition of terms:**
- $Y_c(t)$ — Proxy readout at channel/contact $c$ at time $t$ (e.g., MUA-proxy, LFP-proxy).
- $P_c$ — Readout projection operator mapping emitter states to proxy measurement.
- $v(t)$ — Emitter voltage state.
- $\mathrm{spikes}(t)$ — Binary spike event indicator.
- $\mathrm{source\_proxy}(t)$ — Latent source proxy derived from emitter activity (computational scaffold).

**Worded equation:**
The proxy readout is a declarative mapping from raw emitter voltage, spike events, and a source-activity scaffold onto a dimensionless proxy measurement without solving physical Maxwell equations. Different proxy types (MUA, LFP) apply different weightings to the same underlying emitter state.

**Implementation location:**
[jaxfne/fields.py](../jaxfne/fields.py)

**Scope boundary:**
Proxy readout operator representing a computational mapping scaffold, not a calibrated physical sensor measurement. These are called "-proxy" readouts to emphasize they are tutorial-level demonstrations, not validated against empirical recordings.

---

### 3.3 Firing Rate Summary

**Formal equation:**
$$r = \frac{N_{\mathrm{spikes}}}{T_{\mathrm{seconds}}}$$

**Definition of terms:**
- $r$ — Simulated firing rate (spikes per second, Hz).
- $N_{\mathrm{spikes}}$ — Total spike count over the simulation.
- $T_{\mathrm{seconds}}$ — Simulation duration in seconds.

**Worded equation:**
The firing rate is the total number of spike events divided by the total simulation time. For tutorial purposes, we expect regular spiking with a rate in the range 2–25 Hz under constant drive.

**Implementation location:**
[jaxfne/core.py](../jaxfne/core.py) (Signals.metrics)

**Scope boundary:**
Within-run simulated firing-rate summary; not compared to biological data or validated against experiment.

---

## 4. Canonical Import

```python
import jaxfne as jtfne
import matplotlib.pyplot as plt
import numpy as np
```

---

## 5. Configuration Block

We configure a single-neuron model using the chainable Configuration API:

```python
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
cfg = cfg.cell_types({"E": 1.0})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])

print(f"Duration: {cfg.runtime_config['duration_ms']} ms")
print(f"Time step: {cfg.runtime_config['dt_ms']} ms")
print(f"Neurons: {cfg.networks[0]['n']}")
print(f"Probes: {cfg.probes}")
```

**Configuration summary:**
- **duration_ms:** 1000 ms (1 second of biological time).
- **dt_ms:** 0.1 ms (0.0001 second; 10,000 integration steps).
- **seed:** 7 (deterministic PRNG).
- **dtype:** float32 (single precision).
- **column:** Single L2/3 layer with 1 neuron.
- **cell_type:** Excitatory (E) with 100% fraction.
- **emitter:** Izhikevich with "cortical_eig" preset (regular spiking).
- **probes:** MUA-proxy (spikes), source-proxy (smoothed), LFP-proxy (low-pass).

---

## 6. Simulation Block

```python
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

print(f"Signal shape: {signals.V_m.shape}")
print(f"Time range: {signals.time_ms[0]:.1f} to {signals.time_ms[-1]:.1f} ms")
```

The `construct()` function builds a `Model` from the configuration. The `simulate()` function executes the emitter dynamics for 10,000 time steps and returns a `Signals` object containing voltage, spikes, and other arrays.

---

## 7. Probe/Readout Block

```python
V_m = signals.V_m  # [T, 1] voltage
spikes = signals.spikes  # [T, 1] binary spike indicator

n_spikes = int(np.sum(spikes))
firing_rate_hz = n_spikes / 1.0  # 1 second duration

print(f"Spike count: {n_spikes}")
print(f"Firing rate: {firing_rate_hz:.1f} Hz")
print(f"Expected: 2–25 Hz for regular spiking")
```

The `Signals` object provides arrays for all readout modalities. For a single neuron, we extract the voltage and spike arrays directly and compute summary statistics.

---

## 8. Manifest and Run Metadata

All outputs are accompanied by run metadata and scope boundaries:

```json
{
  "tutorial_id": "v031_single_neuron",
  "duration_ms": 1000.0,
  "dt_ms": 0.1,
  "neurons": 1,
  "emitter": "izhikevich",
  "preset": "cortical_eig",
  "results": {
    "spike_count": 8,
    "firing_rate_hz": 8.0,
    "v_min_mV": -65.0,
    "v_max_mV": 30.0,
    "v_mean_mV": -55.0
  },
  "scope_metadata": {
    "truth_mode": "truth_safe_unverified",
    "claim_level": "computational_scaffold",
    "field_solver_status": "laminar_proxy_no_pde",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "physical_amplitude_claim_allowed": false
  }
}
```

This metadata is immutable and reflects the tutorial's exploratory scope, not biological validation.

---

## 9. Figures

The tutorial generates 5 publication-ready figures:

### Figure 1: Voltage Trace
Shows membrane potential oscillation over the full 1-second simulation. Spikes are visible as upward deflections followed by reset to the configured $c$ value. The baseline oscillates around −65 mV.

### Figure 2: Spike Train
Spike raster showing spike times as vertical ticks. For a single neuron, this is a 1D event sequence. The regularity of inter-spike intervals reflects the Izhikevich model's recovery dynamics.

### Figure 3: Recovery Variable Phase Plane
Trajectory in the voltage-recovery space showing the 2D limit cycle. The neuron starts below threshold, gradually approaches it, spikes, resets, and repeats.

### Figure 4: Source Proxy
Smoothed spike-driven current source activity. This represents the latent source proxy that would be projected onto field readouts in a multi-neuron or network setting.

### Figure 5: Readout Summary
Composite figure showing three readout modalities (MUA-proxy, source-proxy, LFP-proxy) stacked vertically, illustrating how different filtering and operators extract complementary aspects of the same underlying emitter activity.

---

## 10. Interpretation

### What We Observe

The simulation demonstrates a single excitatory neuron exhibiting regular spiking behavior under constant background drive:

1. **Voltage dynamics:** The membrane potential oscillates between a subthreshold resting state (around −65 mV) and spike threshold (30 mV). Each spike triggers an instantaneous reset, then the voltage climbs again.

2. **Spike regularity:** The neuron spikes at a consistent rate (approximately 5–10 Hz for the default `cortical_eig` preset). This regularity is a hallmark of the Izhikevich reduced model's ability to capture neuron-level firing patterns.

3. **Recovery dynamics:** The recovery variable $u$ increases slightly after each spike (via the reset increment $d$), then decays back as voltage approaches threshold. This slow feedback prevents chaotic spiking and produces realistic inter-spike intervals.

4. **Proxy readouts:** Different readout operators extract different aspects of the underlying emitter activity:
   - **MUA-proxy:** Direct spike times; used for population rate estimates.
   - **source-proxy:** Smoothed spike envelope; represents current source activity.
   - **LFP-proxy:** Low-pass-filtered source proxy; represents field potential analog.

### Why This Matters

This tutorial demonstrates that the Izhikevich emitter and proxy readout chain are functioning correctly. A neuron that fails to spike (or spikes irregularly) would signal a problem with either the emitter parameters, the simulation setup, or the readout mapping. Validating single-neuron dynamics is a prerequisite for understanding network-level behavior (v0.3.2–v0.3.3).

---

## 11. Failure Modes

### Common Issues and Diagnosis

**Issue 1: Neuron does not spike (firing_rate = 0)**
- **Cause:** Background drive $I_{\mathrm{native}}$ is too low, or emitter parameters are poorly tuned.
- **Fix:** Increase the drive parameter in the configuration or select a different preset.

**Issue 2: Firing rate is very high (>50 Hz) or chaotic**
- **Cause:** Drive is too high, or recovery parameter $a$ is too small (weak adaptation).
- **Fix:** Reduce drive or select a preset with stronger adaptation.

**Issue 3: Voltage contains NaN or Inf**
- **Cause:** Numerical instability in the solver (time step too large, extreme parameter values).
- **Fix:** Reduce `dt_ms` (e.g., from 0.1 to 0.01), or verify emitter parameters are within expected ranges.

**Issue 4: Readout values are all zero**
- **Cause:** Probe selection is incorrect, or the probe operator failed silently.
- **Fix:** Check that probes are correctly named and that the Configuration.probes list is not empty.

**Issue 5: Simulation runs but figures are empty or malformed**
- **Cause:** Matplotlib backend issue or figure save path is incorrect.
- **Fix:** Ensure the output directory exists; use an explicit backend (e.g., `matplotlib.use('Agg')`).

---

## 12. Exercises

### Exercise 1: Change the Emitter Preset
Try changing `"cortical_eig"` to another implemented preset (e.g., `"fs_izhikevich"` for fast-spiking). How does the firing rate and regularity change?

### Exercise 2: Increase the Simulation Duration
Extend `duration_ms` from 1000 to 5000. Does the spike pattern remain consistent? Why or why not?

### Exercise 3: Change the Random Seed
Modify the seed from 7 to another value. Does the spike pattern change? What does this tell you about determinism in the simulator?

### Exercise 4: Zoom into a Single Spike
Extract a 100 ms window around a spike event. Plot the voltage in detail. What is the shape of the spike waveform?

### Exercise 5: Compute Inter-Spike Intervals (ISIs)
Calculate the time between consecutive spikes. Are they regular or variable? Plot a histogram of ISIs.

---

## 13. Scope Boundaries

### What This Tutorial Covers

- Reduced phenomenological spiking emitter (Izhikevich model).
- Single-neuron voltage and recovery dynamics under constant drive.
- Spike-and-reset threshold behavior.
- Source proxy and readout mapping (MUA, source, LFP).
- Finite-time simulation validation (firing rate, voltage ranges, finite checks).
- Tutorial-level exploratory computation.

### What This Tutorial Does NOT Cover

- **Parameter sweep or sensitivity analysis** — See v0.3.2 for systematic parameter exploration.
- **Two or more neurons** — See v0.3.3 for network dynamics.
- **Empirical biological validation** — This is a computational scaffold only. No comparison to experimental data.
- **Conductance-based biophysics** — The Izhikevich model is phenomenological, not conductance-based.
- **Calibrated amplitudes** — The native current drive is not empirically calibrated to real neurons.
- **Solved 3D electric field** — Proxy readouts are computational mappings; no Maxwell solver is invoked.
- **Optimization or tuning** — See v0.3.2 for parameter optimization workflows.

### Scope Metadata

All simulation results are labeled with conservative metadata:

```json
{
  "truth_mode": "truth_safe_unverified",
  "claim_level": "computational_scaffold",
  "field_solver_status": "laminar_proxy_no_pde",
  "source_calibration_status": "uncalibrated_izhikevich_native_current",
  "physical_amplitude_claim_allowed": false
}
```

This means:
- The results are demonstrative, not biologically validated.
- The readouts are computational scaffolds, not real measurements.
- No physical electromagnetic field is solved.
- No biological claims are made about the neuron or its parameters.

---

**Ready to proceed with v0.3.2 parameter sweep tutorial.**
