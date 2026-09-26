# Sensory Omission & Oddball Detection Paradigm

**Expected stimuli, unexpected deviants, and omission detection under package-native paradigm config.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_v0313_omission_oddball.ipynb)

---

## Learning Objectives

You will learn:

1. **Sensory Omission Config** — declare schedules where expected events get omitted.
2. **Oddball (Deviant) Paradigm** — configure rare deviant streams (expected vs unexpected).
3. **Event Window Analysis** — segment and aggregate responses across baseline, stimulus, post-stimulus windows.
4. **Independent Readouts** — inspect multi-area column outputs under distinct conditions.
5. **Coverage Limits** — omission responses stay uncalibrated proxy readouts.

---

## Biological/Computational Question

**Question:** How does a sensory omission (unexpected silence) modulate population and field readouts in a feedforward-recurrent column compared to standard and deviant tones?

**Context:**
Adaptation and recurrent inhibition transiently suppress responses to repeated standards. Rare deviants or omission of an expected tone can trigger rebound or novelty-proxy responses. Clear paradigm conditions keep mismatch and omission diagnostics exact.

---

## Mathematical Glossary Flow

Events and conditions for the omission paradigm:

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
    .drive(baseline_drive_by_cell_type={"E": 6.5, "PV": 3.0})
    .set_emitter("izhikevich", "cortical_eig")
    .probes(["spikes", "LFP-proxy", "CSD-proxy"]))

# Construct model and apply paradigm conditions
model = jtfne.construct(cfg)
```

---

## Figures

- **Figure 1: Sensory Event Timeline** — expected, deviant, and omitted event windows.
- **Figure 2: Expected raster/rate** — spiking under regular standards.
- **Figure 3: Unexpected deviant raster/rate** — spiking under rare deviant.
- **Figure 4: Omission raster/rate** — activity in expected-tone silence and post-omission window.
- **Figure 5: LFP/CSD-proxy Contrast** — sensory vs omission extracellular-like profiles.

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

## Interactive atlas (dark)

Dark-theme Plotly panels from the page's 60-neuron column under plain drive (1000 ms, dt 0.5 ms, seed 42; the standard/deviant/omission conditions are declared via `jtfne.omission_oddball_paradigm` on top of this circuit): [index](../_static/atlas/omission_60/index.html) · [schema](../_static/atlas/omission_60/schema.html) · [3D](../_static/atlas/omission_60/network_3d.html) · [raster](../_static/atlas/omission_60/raster.html) · [LFP](../_static/atlas/omission_60/lfp.html) · [H](../_static/atlas/omission_60/h_dynamics.html) · [HDP](../_static/atlas/omission_60/hdp.html) · [oscillatory](../_static/atlas/omission_60/oscillatory.html).

Regenerate: `python scripts/generate_doc_page_atlases.py --slug omission_60`.
