# jaxfne v0.3.3: Two-Neuron E/I Dynamics Tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/docs/tutorials_v030/033_two_neuron_ei.md)

**How excitatory and inhibitory coupling shapes spike timing, voltage, and neural readouts.**

## Learning Objectives

1. **E/I coupling** — how excitatory drive and inhibitory feedback interact in a minimal two-neuron motif.
2. **Dynamic synaptic currents** — simulating genuine real-time coupling (not post-hoc) via exponential synaptic state in lax.scan.
3. **Voltage and spike dynamics** — observing phase lags, synchrony, and regime shifts in coupled E/PV networks.
4. **Multimodal readouts** — extracting spikes, voltage, source proxy, and field proxies from coupled E/I networks.
5. **Finite-output validation** — ensuring all coupling coefficients and probes produce finite, interpretable outputs.

## Biological/Computational Question

**How do E→I excitatory drive and I→E inhibitory feedback shape the spike timing, voltage trajectory, and proxy readouts of a minimal two-neuron excitatory-inhibitory pair?**

We address this by simulating an Izhikevich E neuron with direct E→I excitatory coupling and reciprocal I→E inhibitory feedback, recording firing rates, voltage dynamics, and all eight proxy readouts.

## Mathematical Glossary Flow

### E/I Coupling Dynamics

**Formal equation:**
$$V_E^{(t)} = V_E^{(t-1)} + \Delta t \cdot (f_E(V_E, u_E) + g_{EI}(t))$$
$$V_I^{(t)} = V_I^{(t-1)} + \Delta t \cdot (f_I(V_I, u_I) + g_{IE}(t))$$

**Definition of terms:**
- $V_E, u_E$ — E neuron membrane voltage and recovery variable (Izhikevich state).
- $V_I, u_I$ — I (PV) neuron voltage and recovery variable.
- $f_E, f_I$ — Izhikevich dynamics for each neuron type.
- $g_{EI}(t)$ — Excitatory synaptic current from E to I (exponential trace, tuned by $\tau_{syn,E}$).
- $g_{IE}(t)$ — Inhibitory synaptic current from I to E (exponential trace, tuned by $\tau_{syn,I}$).

**Worded equation:** Each neuron's dynamics are governed by Izhikevich equations plus dynamic synaptic injection. Synaptic currents are computed in real-time via exponential traces in the lax.scan carry state, not post-hoc.

**Scope boundary:** Coupling is empirically tuned to produce target firing rates (~10–15 Hz); not derived from biological literature.

### Firing-Rate Response in Coupled Networks

**Formal equation:**
$$r_E = \frac{N_{\mathrm{spikes},E}}{T_{\mathrm{seconds}}}, \quad r_I = \frac{N_{\mathrm{spikes},I}}{T_{\mathrm{seconds}}}$$

**Worded equation:** For the E and I neurons separately, count spikes and divide by total simulation time to obtain per-neuron firing rate.

**Scope boundary:** Within-run summary; no comparison to biology or empirical recordings.

## Configuration Block

```python
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("ei_pair", layers=["L2/3"], n=2)
cfg = cfg.cell_types({"E": 0.5, "PV": 0.5})
cfg = cfg.connectivity(excitatory_to_inhibitory=True, inhibitory_to_excitatory=True)
cfg = cfg.set_emitter("izhikevich", "cortical_eig_e_plus_pv")
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy", "CSD-proxy", "EEG-proxy", "MEG-proxy", "EMM-proxy"])
```

Two-neuron network with dynamic E→I excitatory and I→E inhibitory coupling via exponential synaptic traces (tau_syn_e=5ms, tau_syn_i=10ms).

## Interpretation

### What the Coupled E/I Pair Shows

1. **Firing-rate response:** Both E and I neurons settle into target firing rates (2–25 Hz) when coupling is stable.
2. **Phase relationships:** I neuron typically lags E neuron (inhibitory feedback creates delay).
3. **Voltage range:** Both neurons remain finite and within physiological-like bounds (-80 to +30 mV).
4. **Coupling dynamics:** E→I excitation and I→E inhibition demonstrate genuine reciprocal motif behavior.
5. **Finite outputs:** All probe readouts (spikes, voltage, source, LFP, CSD, EEG, MEG, EMM) are finite and nonzero.

This demonstrates minimal but mechanistically interesting E/I dynamics without claiming biological realism.

## Scope Boundaries

### Covers

- Two-neuron excitatory-inhibitory network motif.
- Dynamic synaptic coupling via exponential traces in lax.scan.
- Izhikevich emitter with E and PV presets.
- All eight proxy readouts and their interpretation.
- Finite-output validation gates.

### Does NOT Cover

- Networks with >2 neurons (v0.3.4 network suite).
- Synaptic plasticity or learning.
- Biological calibration or empirical validation.
- Detailed conductance-based models.
- Inhibitory receptors (only single inhibitory class).

All results labeled: `truth_safe_unverified / computational_scaffold / uncalibrated_izhikevich_native_current / proxy_readout_only`

