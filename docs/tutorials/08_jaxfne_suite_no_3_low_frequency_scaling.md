# jaxfne Suite No. 3: Scale-Dependent Low-Frequency Structure in Proxy Field Readouts

**A compact tutorial demonstrating population scaling, spatiotemporal density preservation, and validation of 1/f^alpha absolute power-law structure in simulated field readouts.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **Noisy Asynchronous Spiking** — how drive heterogeneity and randomized states establish stable asynchronous-irregular dynamics.
2. **Density Preservation** — how to scale population size N while holding spatiotemporal density constant by expanding declared spatial extent.
3. **Absolute Power Spectrum** — how to estimate whole-window absolute power spectral density from proxy readouts.
4. **Log-Log Power-Law Fit** — how to fit 1/f^alpha slope exponents across population scales in the 1-80 Hz band.
5. **Scale Curves** — how to validate slope exponents, low-frequency absolute power, and synchrony metrics across sizes.

---

## Biological/Computational Question

**Question:** How does scaling the population size N while preserving constant spatiotemporal density alter the absolute power-law exponent in aggregate field readouts?

**Context:** 
In a noisy asynchronous-irregular regime, independent fluctuations average out under projection, leaving low-frequency modes to scale with population size. Ensuring constant density prevents confounding local packaging density with population scale.

---

## Mathematical Glossary Flow

Here, we outline the foundational equations defining the readout projection:

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
  [fields.py](file:///Users/hamednejat/workspace/main/jaxfne/jaxfne/fields.py)
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

Every notebook script and library call standardizes to the canonical import:

```python
import jaxfne as jtfne
```

All public APIs are called through the unified `jtfne` namespace.

---

## Simulation Workflow

The tutorial walks through the standard `jaxfne` workflow:

```text
Configuration -> construct -> simulate -> whole-window absolute power -> log-log polyfit -> scale curves
```

1. **Configuration:** Set up a scale-dependent configuration using `jtfne.Configuration()`.
2. **Construction:** Build the model with `jtfne.construct(cfg)`.
3. **Simulation:** Run the vectorized dynamics with `jtfne.simulate(model, sim)`.
4. **Spectral Estimation:** Compute whole-window absolute power spectrum P(f) on log-log axes.
5. **Scale Curves:** Fit exponent alpha and plot slope, low-frequency absolute power, and synchrony versus scale.
