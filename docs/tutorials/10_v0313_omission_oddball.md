# Suite No. 3: Sensory Omission & Oddball Detection Paradigm

**A tutorial demonstrating expected sensory stimuli, unexpected deviants, and sensory omission detection under package-native paradigm configuration.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v0313_omission_oddball.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **Sensory Omission Config** — how to declare stimulus schedules where expected sensory events are dynamically omitted.
2. **Oddball (Deviant) Paradigm** — how to configure rare deviant stimulus streams (expected vs unexpected).
3. **Event Window Analysis** — how to segment and aggregate population responses across baseline, stimulus, and post-stimulus windows.
4. **Independent Readouts** — how to inspect multi-area column outputs under distinct stimulus conditions.
5. **Coverage Limits** — why omission responses are diagnosed as uncalibrated proxy readouts.

---

## Biological/Computational Question

**Question:** How does a sensory omission (unexpected silence) modulate population and field readouts in a feedforward-recurrent column compared to standard and deviant tones?

**Context:**
Spike adaptation and recurrent inhibition can cause a transient suppression of responses to repeated standard tones. In contrast, rare unexpected deviant tones or the abrupt omission of an expected tone can trigger distinctive transient rebound or novelty-proxy responses. Projecting these under clear paradigm conditions ensures exact validation of mismatch and omission diagnostics.

---

## Mathematical Glossary Flow

Here, we define the events and conditions for the sensory omission paradigm:

### 1. expected Condition (Standard Tone)

* **Sensory sequence definition:**
  A repeated standard tone (80% probability) presented after a baseline buffer.
  $$\text{sequence} = (\text{pre}, \text{standard}, \text{post})$$
* **Onset and Duration:**
  $$\tau_{\text{onset}} = 200.0\text{ ms}, \quad \tau_{\text{dur}} = 100.0\text{ ms}$$
* **Worded description:**
  The expected standard condition models repeated sensory inputs with standard gain.
* **Implementation location:**
  `jtfne.omission_oddball_paradigm()`

### 2. unexpected Condition (Deviant Tone)

* **Sensory sequence definition:**
  A rare deviant tone (10% probability) replacing the standard stimulus.
  $$\text{sequence} = (\text{pre}, \text{deviant}, \text{post})$$
* **Worded description:**
  The unexpected deviant condition models sensory novelty with deviant gains.
* **Implementation location:**
  `jtfne.omission_oddball_paradigm()`

### 3. omitted Condition (Sensory Omission)

* **Sensory sequence definition:**
  The expected tone is omitted (10% probability), resulting in complete silence.
  $$\text{sequence} = (\text{pre}, \text{silence}, \text{post\_omission})$$
* **Worded description:**
  The omitted condition verifies whether network transients or sensory rebounds occur at the expected stimulus window in the absence of external drive.
* **Implementation location:**
  `jtfne.omission_oddball_paradigm()`

---

## Canonical Import

All notebooks and scripts leverage the canonical packagefacade import:

```python
import jaxfne as jtfne
```

---

## Configuration & Simulation

```python
# Create the omission/oddball paradigm configuration
paradigm = jtfne.omission_oddball_paradigm(
    standard_onset_ms=200.0,
    standard_duration_ms=100.0,
    deviant_duration_ms=100.0,
    pre_stimulus_buffer_ms=200.0,
    post_stimulus_buffer_ms=500.0,
)

# Set up column and register probes
cfg = (jtfne.Configuration()
    .runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
    .column("V1_column", layers=["L2/3", "L4", "L5"], n=60)
    .cell_type_drives({"E": 6.5, "PV": 3.0})
    .set_emitter("izhikevich", "cortical_eig")
    .probes(["spikes", "LFP-proxy", "CSD-proxy"]))

# Construct model and apply paradigm conditions
model = jtfne.construct(cfg)
```

---

## Figures

- **Figure 1: Sensory Event Timeline** — Overview of expected tone, unexpected deviant tone, and omitted tone event windows.
- **Figure 2: Expected raster/rate** — Spiking activity of regular standard conditions.
- **Figure 3: Unexpected deviant raster/rate** — Spiking activity under rare deviant stimulus presentation.
- **Figure 4: Omission raster/rate** — Network activity during expected tone window silence and post-omission window.
- **Figure 5: LFP/CSD-proxy Contrast** — Extracellular-like profile comparison showing sensory vs omission profiles.

---

## Coverage Boundary

This tutorial covers:
- Declarative omission/oddball sensory configuration.
- Stimulus-locked time segmentation and event window metadata.
- Layer-specific standard, deviant, and omission comparisons.

This tutorial does **NOT** cover:
- Physiological active inference validation.
- Real biological metabolism modeling.
- Maxwell/Poisson-based physical field solve equations.
- Calibrated extracellular amplitude mapping.
