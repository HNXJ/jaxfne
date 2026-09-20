# jaxfne Suite No. 3: Scale-Dependent Low-Frequency Structure in Proxy Field Readouts

**Population scaling, spatiotemporal density preservation, and 1/f^alpha absolute power-law structure in simulated field readouts.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb)

---

## Learning Objectives

You will learn:

1. **Noisy Asynchronous Spiking** — drive heterogeneity and randomized states establish stable asynchronous-irregular dynamics.
2. **Density Preservation** — scale size N while holding spatiotemporal density constant by expanding declared spatial extent.
3. **Absolute Power Spectrum** — estimate whole-window absolute power spectral density from proxy readouts.
4. **Log-Log Power-Law Fit** — fit 1/f^alpha slope exponents across scales in the 1-80 Hz band.
5. **Scale Curves** — check slope exponents, low-frequency absolute power, and synchrony metrics across sizes.

---

## Biological/Computational Question

**Question:** How does scaling the population size N while preserving constant spatiotemporal density alter the absolute power-law exponent in aggregate field readouts?

**Context:** 
In noisy asynchronous-irregular regimes, independent fluctuations average out under projection; low-frequency modes scale with size. Constant density keeps packaging density from confounding scale.

---

## Mathematical Glossary Flow

Foundational equations for the readout projection:

### 1. Readout Projection Equation

* **Formal definition:**
  $$Y_c(t) = \sum_{n=1}^{N} W_{cn} S_n(t)$$
* **Definition of terms:**
  * $Y_c(t)$: Simulated/proxy readout at channel/contact $c$ and time $t$.
  * $W_{cn}$: Projection weight from source element $n$ to channel/contact $c$.
  * $S_n(t)$: Native/proxy source feature from neuron/source element $n$.
  * $N$: Number of source elements, varied across scales.
  * $c$: Readout channel/contact index.
* **Worded equation:**
  The channel signal is the weighted sum of source activity across the modeled source population.
* **Implementation location:**
  `jaxfne/fields/proxy.py` (`project_laminar_sources`)
* **Scope boundary:**
  This is a proxy source-to-readout projection unless a run supplies physical geometry, calibrated source units, conductivity, boundary conditions, gauge handling, a physical field solver, and validation evidence.

### 2. Whole-Window Absolute Power Spectrum

* **Formal definition:**
  $$P(f) = \frac{|Y(f)|^2}{\text{normalization}}$$
* **Worded description:**
  Absolute power at frequency f is the squared magnitude of the windowed, detrended Fourier transform.

### 3. Log-Log Power-Law Fit

* **Formal definition:**
  $$\log_{10}(P(f)) = \beta_0 - \alpha \log_{10}(f)$$
* **Worded description:**
  The scaling exponent alpha is the negative slope of absolute power fit on log-log axes in the 1-80 Hz frequency band.

---

## Canonical Import

Every notebook script and library call uses the canonical import:

```python
import jaxfne as jtfne
```

All public APIs use the unified `jtfne` namespace.

---

## Simulation Workflow

Standard `jaxfne` workflow:

```text
Configuration -> construct -> simulate -> whole-window absolute power -> log-log polyfit -> scale curves
```

1. **Configuration:** Set scale-dependent config with `jtfne.Configuration()`.
2. **Construction:** Build the model with `jtfne.construct(cfg)`.
3. **Simulation:** Run vectorized dynamics with `jtfne.simulate(model, sim)`.
4. **Spectral Estimation:** Compute whole-window absolute power P(f) on log-log axes.
5. **Scale Curves:** Fit exponent alpha; plot slope, low-frequency absolute power, and synchrony vs scale.

## Interactive atlas (dark)

Dark-theme Plotly panels from the N=100 scale (1000 ms, dt 0.5 ms, seed 2303, with the notebook's noisy-async patch; the notebook sweeps N=10/50/100/500 at dt 0.1 ms): [index](../_static/atlas/scale_100/index.html) · [schema](../_static/atlas/scale_100/schema.html) · [3D](../_static/atlas/scale_100/network_3d.html) · [raster](../_static/atlas/scale_100/raster.html) · [LFP](../_static/atlas/scale_100/lfp.html) · [H](../_static/atlas/scale_100/h_dynamics.html) · [HDP](../_static/atlas/scale_100/hdp.html) · [oscillatory](../_static/atlas/scale_100/oscillatory.html).

Regenerate: `python scripts/generate_doc_page_atlases.py --slug scale_100`.
