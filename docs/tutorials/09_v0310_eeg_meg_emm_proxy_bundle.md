# jaxfne Suite: Multimodal Sensor Projections and EEG/MEG/EMM Proxy Bundle

**A tutorial demonstrating separate projection pathways for scalp potential, magnetic field, and metabolic activity proxies.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **Multimodal Projections** — how a single underlying population simulation drives multiple distinct macroscopic measurement streams.
2. **EEG-Proxy Projection** — how to project laminar potential profiles to simulated scalp sensors.
3. **MEG-Proxy Projection** — how to project oriented current dipoles to simulated magnetometer arrays.
4. **Metabolic EMM-Proxy** — how to summarize activity costs as a separate metabolic energy proxy stream.
5. **Operator Separation** — why separate sensor paths must remain distinct rather than merged into a single generic signal.

---

## Biological/Computational Question

**Question:** How can we obtain separate, distinct macroscopic sensor readouts from a single neural population simulation, and how do their spatial projections differ?

**Context:** 
EEG registers volume-conducted scalp potentials. MEG registers magnetic fields induced by intracellular current dipoles. EMM-proxy summarizes metabolic activity cost. Projecting them via separate operator pathways ensures distinct signal profiles that represent complementary views of population dynamics.

---

## Mathematical Glossary Flow

Here, we outline the foundational equations defining the distinct sensor paths:

### 1. EEG-Proxy Sensor Equation

* **Sensor definition:**
  $$Y_{\text{EEG}, s}(t) = \sum_{n=1}^{N} W^{\text{EEG}}_{sn} S_n(t)$$
* **Definition of terms:**
  * $Y_{\text{EEG}, s}(t)$: Scalp potential readout at sensor $s$ and time $t$.
  * $W^{\text{EEG}}_{sn}$: Leadfield weight from source $n$ to sensor $s$.
  * $S_n(t)$: Source current from element $n$.
  * $N$: Number of source elements.
* **Worded description:**
  Scalp potential is the linear leadfield projection of source currents onto scalp sensors.
* **Implementation location:**
  [fields.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/fields.py)

### 2. MEG-Proxy Sensor Equation

* **Sensor definition:**
  $$Y_{\text{MEG}, m}(t) = \sum_{n=1}^{N} W^{\text{MEG}}_{mn} S_n(t)$$
* **Definition of terms:**
  * $Y_{\text{MEG}, m}(t)$: Magnetic field readout at sensor $m$ and time $t$.
  * $W^{\text{MEG}}_{mn}$: Magnetic coupling weight from dipole $n$ to sensor $m$.
* **Worded description:**
  Magnetic field is the linear projection of intracellular dipole current onto magnetometer sensors.
* **Implementation location:**
  [fields.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/fields.py)

### 3. EMM-Proxy Metabolism Equation

* **Energy definition:**
  $$Y_{\text{EMM}}(t) = \frac{1}{N} \sum_{n=1}^{N} |S_n(t)|$$
* **Worded description:**
  The metabolism estimate is the mean absolute source activity cost over time.
* **Implementation location:**
  [fields.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/fields.py)

---

## Canonical Import

Every notebook script and library call standardizes to the canonical import:

```python
import jaxfne as jtfne
```

All public APIs are called through the unified `jtfne` namespace.

---

## Simulation Workflow

The tutorial walks through the standard `jaxfne` workflow:

```text
Configuration -> construct -> simulate -> separate probes -> independent figures
```

1. **Configuration:** Compose the population and register all target modalities via `jtfne.Configuration()`.
2. **Construction:** Build the model with `jtfne.construct(cfg)`.
3. **Simulation:** Run the vectorized simulation with `jtfne.simulate(model, sim)`.
4. **Separate Probing:** Compute EEG-proxy, MEG-proxy, and EMM-proxy readouts using separate operator calls.
5. **Separate Visualization:** Plot distinct panel figures for each sensor modality and export the validation receipt.
