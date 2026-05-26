# jaxfne Suite No. 1: Computational Biophysics

**A 4-part interactive course covering neural models, laminar circuits, and hypothesis tuning.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **Source-generating neuron equations** — how Izhikevich-style reduced neuronal models emit voltage, spikes, and source-driving activity.
2. **Vectorized simulation** — how to scale from a single neuron to populations using JAX arrays, connectivity matrices, and batched circuit operations.
3. **Laminar cortical columns** — how to construct a cortical column with specified layers, cell types, connectivity structure, and 3D geometry.
4. **Multimodal proxy readouts** — how to extract MUA-proxy, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy, and spectral features from simulated source dynamics.
5. **Hypothesis tuning** — how to formulate loss functions and optimize circuit parameters against rate, spectral, and laminar-profile targets.

---

## Biological/Computational Question

Can a simplified, phenomenological multi-layer spiking network, when stimulated with local sensory-like drive, produce depth-resolved field potential proxies (LFP-proxy, CSD-proxy) and spectral features (alpha, beta, gamma oscillations) that match predefined population rate and oscillatory targets?

We address this question by walking through a 4-part biophysical tutorial scaling from a single neuron up to a laminar column, followed by a randomized parameter search to optimize and fit circuit dynamics against the targets.

---

## Mathematical Glossary Flow

Here, we outline the foundational operators and equations defining the emitters, projections, and optimization targets:

### 1. Izhikevich Emitter (Part 1)

* **Boundary definition:**
  $$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I_{\text{native}}$$
  $$\frac{du}{dt} = a(bv - u)$$
  with reset when $v \geq 30\text{ mV}$, then $v \leftarrow c, u \leftarrow u + d$.
* **Definition of terms:**
  * $v$: Membrane potential (mV).
  * $u$: Membrane recovery variable.
  * $a, b, c, d$: Time scale, sensitivity, reset value, and recovery reset parameters.
  * $I_{\text{native}}$: Background driving current (unsigned relative value).
* **Worded equation:**
  The change in membrane potential over time is the sum of quadratic voltage activation, linear voltage scaling, offset current, recovery feedback, and background current inputs. The change in the recovery variable is scaled by the difference between scaled membrane potential and the recovery variable itself.
* **Implementation location:**
  [emitters.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/emitters.py)
* **Boundary:**
  Reduced emitter dynamics (phenomenological spiking model), not full conductance-based biophysical reconstruction.

---

### 2. Source/Readout Bridge (Part 3)

* **Boundary definition:**
  $$Y_c(t) = P_c[\text{signals}, \text{source\_proxy}, \text{field\_proxy}](t)$$
* **Definition of terms:**
  * $Y_c(t)$: Multimodal proxy readout (LFP-proxy, CSD-proxy, etc.) at channel $c$ at time $t$.
  * $P_c$: Readout projection operator mapping raw source states to proxy readouts.
  * $\text{signals}$: Active cell state arrays (voltage, spikes).
  * $\text{source\_proxy}, \text{field\_proxy}$: Laminar metadata templates.
* **Worded equation:**
  The multimodal proxy readout is the evaluation of a declarative projection operator mapping raw vectorized emitter variables onto a spatial-depth profile of contacts without solving physical Maxwell PDEs.
* **Implementation location:**
  [fields.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/fields.py)
* **Boundary:**
  Proxy readout operator representing a computational mapping scaffold, not calibrated physical sensor measurement.

---

### 3. Optimization Search (Part 4)

* **Boundary definition:**
  $$\text{loss} = w_{\text{rate}} \cdot \text{rate\_error} + w_{\text{bandpower}} \cdot \text{bandpower\_error} + w_{\text{sync}} \cdot \text{sync\_penalty}$$
* **Definition of terms:**
  * $\text{loss}$: Total objective mismatch.
  * $w_{\text{rate}}, w_{\text{bandpower}}, w_{\text{sync}}$: Predefined importance weights.
  * $\text{rate\_error}, \text{bandpower\_error}, \text{sync\_penalty}$: Multi-objective distance functions.
* **Worded equation:**
  The total network loss is the weighted sum of target rate discrepancies across E, PV, SST, and VIP cell types, the discrepancy between target and simulated alpha/beta and gamma power ratios, and a penalty on excessive network synchronization.
* **Implementation location:**
  [vis.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/vis.py) (visual comparison and metrics calculations)
* **Boundary:**
  Tutorial motif tuning for structural scaffolds, not empirical biological validation.

---

## Canonical Import

Every notebook script and library call standardizes to the canonical import:

```python
import jaxfne as jtfne
```

All public APIs are called through the unified `jtfne` namespace.

---

## Configuration Block (Compact Grammar)

We define our cortical column layout using the package-native compact configuration facade:

```python
import jaxfne as jtfne

# 1. Initialize configuration and set runtime parameters
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=1500.0, dt_ms=0.25)

# 2. Add columns with specified layers and sizes
cfg = cfg.column("V1", layers=["L2/3", "L4", "L5", "L6"], n=100)

# 3. Specify cell type fractions, loop connectivity, and emitters
cfg = cfg.cell_types({"E": 0.50, "PV": 0.25, "SST": 0.15, "VIP": 0.10})
cfg = cfg.set_emitter("izhikevich", "cortical_eig")

# 4. Declare multimodal proxy probes
cfg = cfg.probes(["MUA-proxy", "LFP-proxy", "CSD-proxy", "EEG-proxy", "MEG-proxy", "EMM-proxy"])
```

---

## Simulation Block

We run our spontaneous or evoked activity simulation with deterministic parameters:

```python
# Construct model
model = jtfne.construct(cfg)

# Simulate vectorized emitter states
signals = jtfne.simulate(model, duration_ms=1500.0, dt_ms=0.25, seed=42)
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
| `truth_mode` | `truth_safe_unverified` | No assumptions of empirical truth or biological accuracy are introduced. |
| `scope_status` | `computational_scaffold` | The package acts as a programmatic scaffold, not a physical simulator. |
| `field_solver_status` | `laminar_proxy_no_pde` | laminar extracellular readouts are computed as weighted proxies, not PDEs. |
| `geometry_mode` | `declared_metadata_not_solved_3d_pde_grid` | 3D coordinate layout is declarative metadata only. |
| `physical_amplitude_allowed` | `false` | Readouts do not map to physical volt or ampere units. |
| `connectivity_status` | `declared_metadata_proxy` | Multi-column connectivity is a declarative structural skeleton. |

---

## Figures

The tutorial generates **22 high-quality figures** covering:
- **Figure 01** — Voltage trace (single E neuron, proxy voltage)
- **Figure 02** — Spike raster (events at threshold)
- **Figure 03** — Source proxy (toy-scale current-like emission)
- **Figure 04** — All-to-all connectivity matrix
- **Figure 05** — Sparse connectivity (p=0.1)
- **Figure 06** — Population spike raster (E vs PV)
- **Figure 07** — Population firing rate timeseries
- **Figure 08** — Laminar layout (3D E/IN distribution, L2/3–L6)
- **Figure 09** — Spontaneous raster (−500 to +1000 ms, event at t=0)
- **Figure 10** — Per-type firing rates (25 ms bins, 4 cell types)
- **Figure 11** — LFP-proxy laminar probe (16 contacts)
- **Figure 12** — CSD-proxy heatmap (spatial derivative)
- **Figure 13** — Time-frequency representation (STFT, 0–80 Hz)
- **Figure 14** — Alpha/beta and gamma bandpower
- **Figure 15** — Evoked raster (L4 drive 0–500 ms)
- **Figure 16** — Evoked LFP-proxy laminar response
- **Figure 17** — Evoked vs. spontaneous bandpower (PSD comparison)
- **Figure 18** — Loss curve (15-step optimization, demo scale)
- **Figure 19** — Parameter trajectory (5 tuning parameters)
- **Figure 20** — Pre vs. post raster (before/after tuning)
- **Figure 21** — Pre vs. post TFR (time-frequency comparison)
- **Figure 22** — Pre vs. post bandpower (spectral comparison)

---

## Interpretation

The Suite No. 1 biophysical tutorial demonstrates how simple phenomenological spiking dynamics (quadratic Izhikevich system) can be parallelized and projected through spatial depth metadata. By formulating structured objective functions (mismatch on rates, spectral band ratios, and synchronization), we show that a randomized parameter search can move the network dynamics towards target rates, producing distinct spectral changes and localized LFP/CSD profiles.

---

## Failure Modes

1. **Emitter Saturation:** Re-tuning connection weights too high causes network-wide depolarization block (runaway excitation).
2. **Frequency Locking:** Weak feedback weights can decouple population layers, leading to isolated, single-frequency oscillations that fail to display the bi-band crossing motif.
3. **Contact Misalignment:** Choosing an arbitrary layout geometry metadata (e.g. negative dz step sizes) shifts the crossing point away from Layer 4.

---

## Exercises

### Exercise 1: Cell Type Manipulation
Modify the `cell_type_fractions` in Part 3 to change E/PV/SST/VIP ratios. How do firing rates change?

### Exercise 2: Connectivity Hypothesis
Replace the uniform random connectivity with a **layer-specific** connectivity matrix. Connect:
- L2/3 E → L4 E (feedforward)
- L4 E ← L4 PV (local inhibition)
- L5 E ← L4 E (inter-laminar)

Simulate and compare bandpower.

### Exercise 3: Drive Schedule
In Part 3, modify `drive_schedule` to stimulate **multiple layers** simultaneously or in sequence. How does multi-layer drive change the evoked response?

### Exercise 4: Loss Function Design
Reformulate the Part 4 loss to optimize for a **different target**:
- "High alpha, low gamma"
- "Synchronized E bursting"
- "Stable baseline rate, strong evoked response"

Run 15 steps of optimization and report the best parameters.

### Exercise 5: Sparse vs. Dense
In Part 2, compare a **fully connected** network (W = all-to-all, W_all) with a **sparse network** (p=0.1). Which stabilizes better? Which has higher synchrony?

---

## Scope Boundaries

* **No biophysical calibration:** The proxy readouts (LFP-proxy, CSD-proxy, etc.) represent arbitrary numerical matrices; they are not calibrated to physical microvolts or microamperes.
* **No biological mechanism proof:** The generated oscillations are products of a simplified, phenomenological Izhikevich system and do not prove any specific biological mechanism.
* **No PDE field solve:** No physical Maxwell equations are solved in 3D grid layouts; readouts are projection proxies based on laminar depth metadata.
