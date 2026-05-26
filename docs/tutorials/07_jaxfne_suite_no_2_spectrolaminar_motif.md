# jaxfne Suite No. 2: Corticospectrolaminar Motif

**A compact tutorial demonstrating multi-column cortical column layout declarations, vectorized spontaneous activity simulation, and publication-ready spectrolaminar visualizations.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_2_spectrolaminar_motif.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **Chainable configuration facade** — how to declare model anatomy, runtime dynamics, cell types, connectivity, and readout modalities using an immutable verb-based Configuration API.
2. **Multi-column layouts** — how to declare separate V1 and PFC cortical column populations and define feedforward and feedback connections between them.
3. **Multimodal readouts** — how to sample simulated source dynamics using various proxy sensors, including MUA-proxy, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, and EMM-proxy.
4. **Spectrolaminar power profiling** — how to extract frequency-depth profiles from laminar extracellular potential signals using JAX-based vis tools.
5. **Optimization search** — how to perform CPU-safe parameter tuning against evocation or spectral objectives.

---

## Biological/Computational Question

Can separate sensory (V1) and associative (PFC) cortical columns, connected in a feedforward-feedback loop, generate stable laminar and spectral patterns that map onto distinct proxy-level biophysical sensors?

This tutorial walks through constructing this multi-column model and inspecting its laminar proxy readouts.

---

## Mathematical Glossary Flow

Here, we outline the foundational operators and equations defining the emitters, projections, and vis targets:

### 1. Izhikevich Emitter

* **Boundary definition:**
  $$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I_{\text{drive}}$$
  $$\frac{du}{dt} = a(bv - u)$$
* **Definition of terms:**
  * $v$: Membrane potential (mV).
  * $u$: Membrane recovery variable.
  * $a, b$: Time scale and sensitivity parameters of the recovery variable.
  * $I_{\text{drive}}$: Current drive representing default background inputs.
* **Worded equation:**
  The change in membrane potential over time is the sum of a quadratic voltage activation term, linear scale, offset constant, recovery feedback, and internal driving currents. The change in the recovery variable is scaled by the difference between scaled potential and the recovery variable itself.
* **Implementation location:**
  [emitters.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/emitters.py)
* **Boundary:**
  Reduced emitter dynamics (phenomenological spiking model); conductance-based biophysical reconstruction is future work.

---

### 2. Source/Readout Bridge

* **Boundary definition:**
  $$Y_c(t) = P_c[\text{signals}, \text{source\_proxy}, \text{field\_proxy}](t)$$
* **Definition of terms:**
  * $Y_c(t)$: Multimodal proxy readout at contact or sensor channel $c$ at time $t$.
  * $P_c$: Readout projection operator mapping raw source states to proxy readouts.
  * $\text{signals}$: Active cell state arrays (voltage, spikes).
  * $\text{source\_proxy}, \text{field\_proxy}$: Laminar metadata templates.
* **Worded equation:**
  The multimodal proxy readout is the evaluation of a declarative projection operator mapping raw vectorized emitter variables onto a spatial-depth profile of contacts without solving physical Maxwell PDEs.
* **Implementation location:**
  [fields.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/fields.py)
* **Boundary:**
  Proxy readout operator representing a computational mapping scaffold for tutorial workflows; physical calibration is future work.

---

### 3. Spectrolaminar Motif Target

* **Boundary definition:**
  $$\text{relative alpha/beta: deeper-biased profile}$$
  $$\text{relative gamma: superficial-biased profile}$$
  $$\text{crossing: near L4 reference depth}$$
* **Definition of terms:**
  * $\text{relative alpha/beta}$: Power distribution focused in deep layers (L5, L6).
  * $\text{relative gamma}$: Power distribution focused in superficial layers (L2/3).
  * $\text{crossing}$: Point of inversion near layer L4.
* **Worded equation:**
  The spectral-depth distribution exhibits a superficial bias for gamma band power, a deep bias for alpha/beta band power, and a reference inversion near mid-column layers.
* **Implementation location:**
  [vis.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/vis.py)
* **Boundary:**
  Tutorial motif visualization for structural scaffolds; empirical biological validation is future work.

---

## Canonical Import

Every notebook script and library call standardizes to the canonical import:

```python
import jaxfne as jtfne
```

All public APIs are called through the unified `jtfne` namespace.

---

## Configuration Block (Compact Grammar)

We define our cortical columns using the package compact chainable facade methods:

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

## Simulation Block

We run a spontaneous activity simulation with deterministic parameters:

```python
# Construct model
model = jtfne.construct(cfg)

# Simulate vectorized emitter states
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

---

## Probe/Readout Block

Sampling the simulated dynamics through our declared proxy readouts allows depth-resolved and macro-scale analysis.

```python
# Sample LFP-proxy and CSD-proxy
lfp_proxy = signals.field.lfp_proxy
csd_proxy = signals.field.csd_proxy
```

---

## Run Metadata and Scope Fields

| Metadata Key | Value / Status | Description |
|---|---|---|
| `truth_mode` | `truth_safe_unverified` | Designed for exploratory modeling; empirical truth and biological accuracy validation pending. |
| `scope_status` | `computational_scaffold` | The package acts as a programmatic scaffold for exploratory learning; physical simulation is future work. |
| `field_solver_status` | `laminar_proxy_no_pde` | laminar extracellular readouts are computed as weighted proxies; PDE solvers are future work. |
| `geometry_mode` | `declared_metadata_not_solved_3d_pde_grid` | 3D coordinate layout is declarative metadata; PDE-based geometry solving is future work. |
| `physical_amplitude_allowed` | `false` | Readouts use proxy-scale units for exploratory workflows; physical amplitude calibration pending. |
| `connectivity_status` | `declared_metadata_proxy` | Multi-column connectivity is a declarative structural skeleton. |

---

## Figures

The generated assets represent the standard visual bundle:
- **Figure 01** — V1/PFC 3D Layout
- **Figure 02** — Baseline Raster
- **Figure 03** — Population Rates
- **Figure 04** — Voltage Traces
- **Figure 05** — MUA-Proxy
- **Figure 06** — LFP-Proxy
- **Figure 07** — CSD-Proxy
- **Figure 08** — EEG/MEG/EMM Proxy Summary
- **Figure 09** — Spectrolaminar Heatmap
- **Figure 10** — Layer-Band Profiles
- **Figure 11** — Tuning Loss
- **Figure 12** — Pre/Post Spectrolaminar Comparison
- **Figure 13** — Parameter Trajectory

---

## Interpretation

The multi-column spontaneous activity simulation demonstrates how structural feedforward/feedback loops generate localized oscillatory rhythms. By projecting these active patterns through the laminar proxy operator, we recover a spectrolaminar profile where high-frequency (gamma) power peaks in superficial layers (L2/3) and lower-frequency (alpha/beta) power is prominent in deep layers (L5/L6), capturing a classical spectrolaminar motif entirely within a phenomenological computational scaffold.

---

## Failure Modes

1. **Emitter Saturation:** Re-tuning connection weights too high causes network-wide depolarization block (runaway excitation).
2. **Frequency Locking:** Weak feedback weights can decouple V1 and PFC, leading to isolated, single-frequency oscillations that fail to display the bi-band crossing motif.
3. **Contact Misalignment:** Choosing an arbitrary layout geometry metadata (e.g. negative dz step sizes) shifts the crossing point away from Layer 4.

---

## Exercises

1. **Tuning Feedback:** Adjust the PFC feedback weight in `cfg.connectivity` and plot how the deep alpha/beta power changes.
2. **Modified Emitter Presets:** Swap the `cortical_eig` emitter preset for an custom drive array to examine baseline firing rates.
3. **Layer Expansion:** Add a `L1` layer to both V1 and PFC, and examine the MUA-proxy readout at the upper contact boundary.

---

## Scope Boundaries

* **Scope:** The proxy readouts (LFP-proxy, CSD-proxy, etc.) are scope-bounded for exploratory workflows; calibration to physical measurements is future work.
* **Scope:** The generated oscillations are products of a simplified Izhikevich system.
* **Scope:** Field computation via proxy operators; PDE-based field solving is future work.
