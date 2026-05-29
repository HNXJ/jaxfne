# TFNE Alignment with Arkhipov and Allen Institute Biophysical Schemes

This note clarifies how the Tensor-Field Neural Equation (TFNE) framework in `jaxfne` aligns mathematically and conceptually with the biological modeling schemas popularized by Arkhipov et al. (2018), Gouwens et al. (2018), Billeh et al. (2020), and Rimehaug et al. (2023).

---

## 1. Architectural Alignment

The standard workflow in `jaxfne` translates multiscale biophysical and network architectures into an efficient, tensor-parallel computation graph:

```
[Emitter (Local Spiking/HH Dynamics)]
        │
        ▼ (Presynaptic Spikes / Conductances)
[Synapse / Connectivity Layer]
        │
        ▼ (Recurrent Drive & Aggregate Inputs)
[Source Layer (Transmembrane Current Bookkeeping)]
        │
        ▼ (Laminar/Spatial Geometry Weighting)
[Passive Field / Linear Readout]
        │
        ▼ (Extracellular Contact Probes)
[LFP / CSD / EEG / MEG Readout Proxies]
```

Under this view, local nonlinearities (voltage gates, synaptic receptors, adaptation variables) are strictly isolated within the **Emitter** and **Synapse** components, while spatial propagation and electrode readouts are represented as fast, parallelized **Linear Readout** operators.

---

## 2. Core Modeling Anchors

`jaxfne` adopts canonical mathematical formulations from major Allen Institute and computational neuroscience benchmarks:

### 2.1. Local Conductance & Adaptation (Gouwens 2018)
Multicompartment Hodgkin-Huxley-style gate kinetics, calcium shells, and voltage reset adaptations are represented inside the generalized **Emitter** loop.

### 2.2. Network Connectivity & Synaptic Kernels (Arkhipov 2018)
Point-neuron approximations or compartmental models map recurrent connectivity weights through explicit synaptic trace equations. 

Postsynaptic current updates are governed by:
- **Exponential Synapse**:
  $$s_j[t+1] = s_j[t]\exp\left(-\frac{\Delta t}{\tau_j}\right) + z_j[t]$$
- **Alpha Synapse**:
  $$k_j(t) = \frac{t}{\tau_j}\exp\left(1-\frac{t}{\tau_j}\right)$$
- **Double-Exponential Synapse**:
  $$k_j(t) = A_j \left(e^{-t/\tau_{decay,j}} - e^{-t/\tau_{rise,j}}\right)$$

### 2.3. Dual Multiscale Representation (Billeh 2020)
Like the Allen V1 multiscale modeling paradigm, TFNE architectures separate computationally expensive cellular dynamics from low-dimensional network descriptions (e.g., GLIF vs. full biophysical compartment models), allowing users to switch out emitter classes while maintaining a unified linear readout layer.

### 2.4. Extracellular Signal Readouts (Rimehaug 2023)
Transmembrane currents are projected onto extracellular recording probes (e.g., laminar silicon probes) using a linear transfer-resistance proxy model:
$$Y_c(t) = \sum_n W_{cn} S_n(t)$$

---

## 3. Methodological and Scaling Disclaimers

> [!IMPORTANT]
> **Proxy Readouts Only (Uncalibrated)**
> Extracellular potential metrics (LFP-like, CSD-like, EEG-like, MEG-like, and EMM-proxy) produced by `jaxfne` are *numerical proxy readouts* designed for machine learning objective functions, optimization, and system-level comparison.
>
> - **Laminar Proxy (No PDE)**: The calculations do not solve dynamic volume-conductor partial differential equations (PDEs) or complex boundary-element models.
> - **No Uncalibrated Physical Amplitude Claims**: Unless a physical-conductivity solver has been explicitly instantiated and calibrated with local tissue properties, users MUST NOT claim absolute physical amplitude metrics (e.g., microvolts, nanoamperes) in publications. The outputs are represented in uncalibrated *proxy units*.

---

## 4. Canonical Coding Pattern

In notebooks, tutorials, and scripts, the package must always be imported using its canonical alias:

```python
import jaxfne as jtfne

# 1. Instantiate the bridge (e.g. JaxleyBridge)
bridge = jtfne.bridges.JaxleyBridge(
    model=jaxley_model,
    source_mode="transmembrane_current",
    compartment_axis="last"
)

# 2. Extract uncalibrated transmembrane source bookkeeping
sources = bridge.extract_sources(simulation_result)

# 3. Plot proxy signals using the visualizer namespace
fig = jtfne.vis.lfp(sources)
```

---
Footer: Agent: antigravity_front / Model: Gemini 3.5 Flash / Role: TFNE Modeling & Biophysics Alignment / Plane: Control-Execution Boundary / Repo: jaxfne / Date: 2026-05-28
