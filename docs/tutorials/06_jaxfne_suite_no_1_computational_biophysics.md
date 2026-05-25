# jaxfne Suite No. 1: Computational Biophysics

**A 4-part interactive course covering neural models, laminar circuits, and hypothesis tuning.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **What computational models are** — how Izhikevich neurons emit voltage and spikes
2. **Vectorized simulation** — scaling from single neurons to populations with connectivity matrices
3. **Laminar cortical structure** — how cell types and connectivity create layered circuits
4. **Readout operators** — extracting LFP-proxy, CSD-proxy, spectral features from source dynamics
5. **Hypothesis tuning** — formulating loss functions and running optimization over circuit parameters

---

## The Big Question

**Can a simple computational model, when properly tuned, generate population signals with recognizable structure (layering, spectral content, evoked responses)?**

This tutorial answers "yes" in a controlled, educational setting. It does **not** prove anything about biology — that requires empirical validation.

---

## Course Structure

### Part 1: Computational Models and Biophysics

Start with a **single Izhikevich neuron**:

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I_{\text{native}}$$

$$\frac{du}{dt} = a(bv - u)$$

**with reset when** $v \geq 30 \text{ mV}$ **then** $v \leftarrow c, u \leftarrow u + d$.

Learn:
- How native current (not physical amperes) drives spiking
- How source proxy converts voltage dynamics to a field-like quantity
- Why models need explicit claims about what they represent

**Outputs:**
- Figure 01: Voltage trace (proxy, not validated)
- Figure 02: Spike raster (events at threshold)
- Figure 03: Source proxy (toy-scale current-like quantity)

---

### Part 2: Parallelization and Linear Algebra

Scale to **N neurons** with a **connectivity matrix** $\mathbf{W} \in \mathbb{R}^{N \times N}$:

$$\frac{dv_i}{dt} = 0.04v_i^2 + 5v_i + 140 - u_i + I_{i,\text{native}} + \sum_{j} W_{ij} s_j(t)$$

where $s_j(t)$ is the spike output of neuron $j$.

Learn:
- How $\mathbf{W}$ encodes circuit hypotheses
- How to simulate vectorized populations
- How sparse vs. dense connectivity affects dynamics

**Outputs:**
- Figure 04: All-to-all connectivity matrix
- Figure 05: Sparse connectivity (p=0.1)
- Figure 06: Population raster (E/PV neurons)
- Figure 07: Firing rate timeseries

---

### Part 3: Cortex — Building and Testing Readouts

Build a **laminar cortical column** with **4 cell types** across **4 layers**:

| Layer | E (n=50) | PV (n=25) | SST (n=15) | VIP (n=10) | Total |
|-------|----------|-----------|-----------|-----------|-------|
| L2/3 | ✓ | ✓ | ✓ | ✓ | ~29 |
| L4 | ✓ | ✓ | — | — | ~23 |
| L5 | ✓ | ✓ | — | — | ~31 |
| L6 | ✓ | ✓ | ✓ | ✓ | ~17 |

Apply **readout operators**:
- **Spike raster** — raw spike times
- **Firing rate** — binned population rate
- **LFP-proxy** — laminar field readout (NOT real LFP, proxy only)
- **CSD-proxy** — spatial derivative of LFP-proxy (NOT real CSD)
- **Time-frequency representation** — STFT of mean LFP-proxy
- **Bandpower** — alpha/beta and gamma power timeseries

Learn:
- How depth-based readouts relate to anatomy
- How to interpret LFP-proxy and CSD-proxy (they are NOT validated neural signals)
- How evoked responses emerge from layer-specific drive

**Outputs:**
- Figure 08: Laminar column layout (3D projection)
- Figure 09: Spontaneous raster (100 ms pre + 500 ms post event + 500 ms post-event)
- Figure 10: Spontaneous firing rates (per cell type)
- Figure 11: LFP-proxy laminar probe (16 contacts)
- Figure 12: CSD-proxy heatmap (NOT real CSD)
- Figure 13: Time-frequency representation
- Figure 14: Alpha/beta and gamma bandpower
- Figure 15: Evoked raster (L4 stimulation 0–500 ms)
- Figure 16: Evoked LFP-proxy
- Figure 17: Evoked vs. spontaneous bandpower

---

### Part 4: Fine-Tuning for Hypothesis Tests

Formulate a **tuning objective** and run a **GSDR-style random search** (15 steps):

**Hypothesis:** Can we tune $\{l4\_drive, e\_gain, pv\_gain, sst\_gain, vip\_gain\}$ to bring rates and spectral content closer to targets?

**Loss function:**
$$\text{loss} = w_{\text{rate}} \cdot \text{rate\_error} + w_{\text{bandpower}} \cdot \text{bandpower\_error} + w_{\text{sync}} \cdot \text{sync\_penalty}$$

**Parameter bounds:**
- $l4\_drive \in [2, 12]$ — evoked drive amplitude
- $e\_gain, pv\_gain, sst\_gain, vip\_gain \in [0.5, 2]$ — per-type gain multipliers

Learn:
- How to specify circuit constraints as optimization objectives
- Why finite random search is appropriate for demos (NOT convergence)
- How to track parameter and loss trajectories
- How to compare pre/post tuning behavior

**Outputs:**
- Figure 18: Loss curve (15 steps)
- Figure 19: Parameter trajectory (5 parameters, 15 steps)
- Figure 20: Pre vs. post raster comparison
- Figure 21: Pre vs. post TFR
- Figure 22: Pre vs. post bandpower

---

## Export Artifacts

At the end, three files are saved to your Colab working directory:

1. **`tutorial_metrics.csv`** — Per-neuron firing rates, voltage stats, layer assignment, cell type
2. **`tuning_history.csv`** — Optimization step, loss, parameters, rates, spectral metrics
3. **`tutorial_manifest.json`** — Claim gates, metadata, figure list, best tuning parameters

---

## Mathematical Glossary

| Term | Meaning | Claim Status |
|------|---------|--------------|
| **Izhikevich neuron** | 2D reduced model with polynomial spiking | Tutorial scaffold |
| **Native current** | Toy-scale quantity driving spikes (NOT physical amperes) | Computational proxy |
| **Source proxy** | Emitted source quantity (NOT validated as real current source) | Proxy, uncalibrated |
| **LFP-proxy** | Laminar field readout from sources (NOT real LFP) | Proxy, no field solver |
| **CSD-proxy** | Spatial derivative of LFP-proxy (NOT real CSD) | Proxy, no field solver |
| **Firing rate** | Spikes per unit time | Descriptive, not validated |
| **Bandpower** | Power in frequency bands (alpha/beta, gamma) | Descriptive, not calibrated |

---

## Visualization Gallery

The tutorial generates **22 high-quality figures** covering:

### Single Neurons (Part 1)
- **Figure 01** — Voltage trace (single E neuron, proxy not validated)
- **Figure 02** — Spike raster (events at threshold)
- **Figure 03** — Source proxy (toy-scale current-like emission)

### Population Connectivity (Part 2)
- **Figure 04** — All-to-all connectivity matrix
- **Figure 05** — Sparse connectivity (p=0.1)
- **Figure 06** — Population spike raster (E vs PV)
- **Figure 07** — Population firing rate timeseries

### Laminar Column (Part 3)
- **Figure 08** — Laminar layout (3D E/IN distribution, L2/3–L6)
- **Figure 09** — Spontaneous raster (−500 to +1000 ms, event at t=0)
- **Figure 10** — Per-type firing rates (25 ms bins, 4 cell types)
- **Figure 11** — LFP-proxy laminar probe (16 contacts, NOT real LFP)
- **Figure 12** — CSD-proxy heatmap (spatial derivative, NOT real CSD)
- **Figure 13** — Time-frequency representation (STFT, 0–80 Hz)
- **Figure 14** — Alpha/beta and gamma bandpower
- **Figure 15** — Evoked raster (L4 drive 0–500 ms)
- **Figure 16** — Evoked LFP-proxy laminar response
- **Figure 17** — Evoked vs. spontaneous bandpower (PSD comparison)

### Tuning (Part 4)
- **Figure 18** — Loss curve (15-step optimization, demo scale)
- **Figure 19** — Parameter trajectory (5 tuning parameters)
- **Figure 20** — Pre vs. post raster (before/after tuning)
- **Figure 21** — Pre vs. post TFR (time-frequency comparison)
- **Figure 22** — Pre vs. post bandpower (spectral comparison)

All figures are **PNG + SVG** and saved to the Colab working directory. Open the notebook to see rendered outputs.

---

## Configuration and Execution

**Environment:** Colab (CPU-safe, no GPU required)

**Installation:** The notebook auto-installs jaxfne and dependencies with:

```python
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'jaxfne', 'numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn'], check=True)
```

**Runtime:** ~2–3 minutes on Colab (100 neurons, 1500 ms duration, 15 optimization steps)

**API used (all existing jaxfne):**
- `jaxfne.emitters.izhikevich_eig_params()` — parameter initialization
- `jaxfne.emitters.simulate_eig_izhikevich()` — simulation
- `jaxfne.fields.project_laminar_sources()` — field projection
- `IzhikevichParams` — parameter dataclass
- Standard JAX operations (`jax.random`, `jax.numpy`, `vmap`, etc.)

No new package-level functions. All helper utilities (savefig, compute_metrics, etc.) are tutorial-local.

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

## Failure Modes and Gotchas

### Gotcha 1: Drive Amplitude
If `drive` is too low, neurons don't spike. If too high, they saturate. The demo uses layer-specific and cell-type-specific drives to balance.

### Gotcha 2: Connectivity Sign
The notebook uses **unsigned connectivity** (all positive). To add **inhibition**, you would multiply PV/SST/VIP synaptic weights by **–1**. This is left for Exercise 2.

### Gotcha 3: LFP-proxy vs. Real LFP
LFP-proxy is computed from **source positions** and **synthetic sources**, not from a real neurophysiological recording. It has **no biological calibration**. The `-proxy` suffix is mandatory.

### Gotcha 4: CSD and Field Solver
CSD-proxy is a **spatial derivative** of LFP-proxy, not a field solution. A real field solver would solve Poisson's equation with conductivity tensors. We don't do that here (deferred to advanced courses).

### Gotcha 5: Optimization Convergence
15 steps is **too few** to guarantee convergence. The demo shows the *trajectory* and *relative improvement*, not a converged solution.

---

## What This Tutorial Does NOT Claim

- ❌ **Real LFP:** LFP-proxy is a toy-scale synthetic readout. Real LFP requires empirical recording and calibration.
- ❌ **Real CSD:** CSD-proxy is a spatial derivative of LFP-proxy. Real CSD requires field reconstruction with conductivity models.
- ❌ **Biological calibration:** Izhikevich parameters are not fit to any biological neuron. The native current is toy-scale.
- ❌ **Proof of mechanism:** This tutorial demonstrates *whether a model can generate plausible-looking signals*, not *how biology works*.
- ❌ **EEG or MEG:** Those require scaling to whole-brain networks and source localization (not covered).
- ❌ **Optimization convergence:** 15 steps do not guarantee the optimizer found a good solution.

---

## Interpretation Guide

When you see a figure in this tutorial:

1. **Ask:** What is it showing? (e.g., "Is this a spike raster or a voltage trace?")
2. **Check the subtitle:** Every figure has a claim label (proxy, scaffold, computational)
3. **Look for the disclaimer:** Axis labels explicitly state "NOT physical", "proxy", "toy-scale", etc.
4. **Consider alternatives:** Could a different model, connectivity, or drive produce similar outputs?
5. **Check the manifest:** The `tutorial_manifest.json` file has all metadata (claim gates, figure list, best parameters)

---

## Next Steps

After this tutorial:

1. **Read guides** — [Tensor-field workflows](../tensor_field_workflows.md), [Probe operators](../probe_operators.md)
2. **Use the API** — Extend the notebook with different connectivity patterns, cell types, or readouts
3. **Run other tutorials** — [Single-neuron multimodal](01_single_neuron_multimodal.md), [Two-neuron E/I](02_two_neuron_ei.md)
4. **Explore optimization** — Formulate your own circuit hypothesis and tune it

---

## Manifest and Claim Gates

**Immutable claim gates (frozen in `tutorial_manifest.json`):**

| Field | Value | Reason |
|-------|-------|--------|
| `claim_level` | `computational_scaffold` | Tutorial is educational, not biological proof |
| `source_calibration_status` | `toy_scale_not_empirical` | Native current is proxy, not calibrated to biology |
| `source_projection_mode` | `proxy_no_field_solve` | No Poisson solve, no conductivity model |
| `readout_status` | `LFP-proxy and CSD-proxy readouts` | Readouts are synthetic, not validated |
| `truth_mode` | `truth_safe_unverified` | No scientific truth claims without empirical validation |

**Tutorial metadata:**
- **Version:** `jaxfne-colab-tutorial-v1`
- **n_total:** 100 neurons
- **dt_ms:** 0.25 ms
- **duration_ms:** 1500 ms
- **n_opt_steps:** 15 (demo scale, not converged)
- **Figures:** 22 (PNG + SVG)
- **Export files:** `tutorial_metrics.csv`, `tuning_history.csv`, `tutorial_manifest.json`

---

## Citation

If you use this tutorial in your own work, cite:

```bibtex
@misc{jaxfne_suite_1,
  title={jaxfne Suite No. 1: Computational Biophysics},
  author={HNXJ},
  year={2026},
  note={Interactive Colab tutorial on neural modeling and laminar circuits},
  url={https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb}
}
```

---

**Status:** ✓ Complete | ✓ Claim gates immutable | ✓ All 22 figures present | ✓ Notebooks and CSV exports ready

*jaxfne-colab-tutorial-v1 | truth_safe_unverified | tutorial_exploratory_not_biological_truth*
