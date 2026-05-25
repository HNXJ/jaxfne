# jaxfne Suite No. 2: Spectrolaminar Motif

**A compact tutorial demonstrating multi-column cortical column layout declarations, vectorized spontaneous activity simulation, and publication-ready spectrolaminar visualizations.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **Chainable configuration facade** — how to declare model anatomy, runtime dynamics, cell types, connectivity, and readout modalities using an immutable verb-based Configuration API.
2. **Multi-column layouts** — how to declare separate V1 and PFC cortical column populations and define feedforward and feedback connections between them.
3. **Multimodal readouts** — how to sample simulated source dynamics using various proxy sensors, including MUA-proxy, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, and EMM-proxy.
4. **Spectrolaminar power profiling** — how to extract frequency-depth profiles from laminar extracellular potential signals using JAX-native vis tools.
5. **Optimization search** — how to perform CPU-safe parameter tuning against evocation or spectral objectives.

---

## The Big Question

**Can separate sensory and associative cortical columns, connected in a feedforward-feedback loop, generate stable laminar and spectral patterns that map onto distinct proxy-level biophysical sensors?**

This tutorial walks through constructing this multi-column model and inspecting its laminar proxy readouts.

---

## 1. Setup and Chainable Configuration

We define our cortical columns using the package-native compact chainable facade methods:

```python
import jaxfne as jtfne

# 1. Initialize configuration and set runtime parameters
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)

# 2. Add columns with different layers and sizes
cfg = cfg.column("V1", layers=["L2/3", "L4", "L5", "L6"], n=80)
cfg = cfg.column("PFC", layers=["L2/3", "L5", "L6"], n=80)

# 3. Specify cell type fractions, loop connectivity, and emitters
cfg = cfg.cell_types({"E": 0.75, "PV": 0.12, "SST": 0.08, "VIP": 0.05})
cfg = cfg.connectivity(feedforward=("V1", "PFC"), feedback=("PFC", "V1"))
cfg = cfg.set_emitter("izhikevich", "cortical_eig")

# 4. Declare multimodal proxy probes
cfg = cfg.probes(["MUA-proxy", "LFP-proxy", "CSD-proxy", "EEG-proxy", "MEG-proxy", "EMM-proxy"])
```

---

## 2. Model Construction & Geometry

We compile the model, placing the units along a 3D laminar layout:

```python
model = jtfne.construct(cfg)
```

### Figure 01: V1/PFC 3D Layout
![V1/PFC 3D Layout](../../outputs/suite_no_2_spectrolaminar_motif/figures/01_v1_pfc_3d_layout.png)

---

## 3. Vectorized Simulation

We run a spontaneous activity simulation with deterministic parameters:

```python
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

### Figure 02: Baseline Raster
![Baseline Raster](../../outputs/suite_no_2_spectrolaminar_motif/figures/02_baseline_raster.png)

### Figure 03: Population Rates
![Population Rates](../../outputs/suite_no_2_spectrolaminar_motif/figures/03_population_rates.png)

### Figure 04: Voltage Traces
![Voltage Traces](../../outputs/suite_no_2_spectrolaminar_motif/figures/04_voltage_traces.png)

---

## 4. Multimodal Readout Analysis

Sampling the simulated dynamics through our declared proxy readouts allows depth-resolved and macro-scale analysis.

### Figure 05: MUA-Proxy
![MUA-Proxy](../../outputs/suite_no_2_spectrolaminar_motif/figures/05_mua_proxy.png)

### Figure 06: LFP-Proxy
![LFP-Proxy](../../outputs/suite_no_2_spectrolaminar_motif/figures/06_lfp_proxy.png)

### Figure 07: CSD-Proxy
![CSD-Proxy](../../outputs/suite_no_2_spectrolaminar_motif/figures/07_csd_proxy.png)

### Figure 08: EEG/MEG/EMM Proxy Summary
![EEG/MEG/EMM Proxy](../../outputs/suite_no_2_spectrolaminar_motif/figures/08_eeg_meg_emm_proxy.png)

---

## 5. Spectrolaminar Motif Visualization

Calling the standard visual API generates a 3-panel publication-ready figure:

```python
jtfne.vis.spectrolaminar(signals)
```

### Figure 09: Spectrolaminar Heatmap
![Spectrolaminar Heatmap](../../outputs/suite_no_2_spectrolaminar_motif/figures/09_spectrolaminar_heatmap.png)

### Figure 10: Layer-Band Profiles
![Layer-Band Profiles](../../outputs/suite_no_2_spectrolaminar_motif/figures/10_layer_band_profiles.png)

---

## 6. Fine-Tuning Optimization Search

We run a small parameter search to fit connection weights against target spectral dynamics:

### Figure 11: Tuning Loss
![Tuning Loss](../../outputs/suite_no_2_spectrolaminar_motif/figures/11_tuning_loss.png)

### Figure 12: Pre/Post Spectrolaminar Comparison
![Pre/Post Spectrolaminar](../../outputs/suite_no_2_spectrolaminar_motif/figures/12_pre_post_spectrolaminar.png)

### Figure 13: Parameter Trajectory
![Parameter Trajectory](../../outputs/suite_no_2_spectrolaminar_motif/figures/13_parameter_trajectory.png)

---

## Summary of Truth-Safe Readout Status

All generated signals are calculated under `truth_safe_unverified` guidelines and represent uncalibrated laminar proxy readouts (`laminar_proxy_no_pde`). No physical amplitude or biological mechanism claims are introduced.
