# jaxfne Suite No. 3: Scale-Dependent Low-Frequency Structure in Proxy Field Readouts

**A compact tutorial demonstrating population scaling, proxy source-to-sensor projection, and validation of low-frequency structure in simulated field readouts.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_suite_no_3_low_frequency_scaling.ipynb)

---

## Learning Objectives

After completing this tutorial, you will understand:

1. **Low-Frequency Sensitivity** — why low-frequency proxy readouts are more sensitive to spatial scale and source coherence than local spike statistics.
2. **Emitter vs. Projection** — how to distinguish nonlinear emitter/network generation from linear source-to-probe projection.
3. **Scaling Experimentation** — how to run a package-native scaling experiment across multiple population sizes.
4. **Spectral Metrics** — how to compute bandpower ratios and synchrony proxies from simulated/proxy readouts.
5. **Validation Scaffolds** — how to export figures, manifest metadata, and validation summaries without making calibrated physical amplitude claims.

---

## Biological/Computational Question

**Question:** How does increasing the modeled neural population or declared spatial extent alter low-frequency structure in simulated/proxy field readouts?

**Working hypothesis:** Larger modeled support can preserve slow coherent source modes in aggregate readouts, while faster local fluctuations are more likely to average out under projection. This does not identify a biological mechanism by itself; it gives a controlled computational diagnostic for scale-sensitive readout behavior.

---

## Mathematical Glossary Flow

Here, we outline the foundational equations defining the readout projection:

### 1. Readout Projection Equation

* **Boundary definition:**
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
Configuration -> construct -> simulate -> probe/readout -> metrics/figures
```

1. **Configuration:** Set up a scale-dependent configuration using `jtfne.Configuration()`.
2. **Construction:** Build the model with `jtfne.construct(cfg)`.
3. **Simulation:** Run the vectorized dynamics with `jtfne.simulate(model, sim)`.
4. **Probing:** Extract proxy readout traces using `model.probe(signals)` or similar.
5. **Analysis:** Compute relative PSD, bandpower metrics, and synchrony proxies, then export manifest and validation reports.
