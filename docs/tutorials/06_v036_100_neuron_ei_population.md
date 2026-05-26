# v0.3.6: 100-Neuron Excitatory-Inhibitory Population

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/jaxfne_v036_100_neuron_ei_population.ipynb)

**Duration:** 15–20 minutes | **Difficulty:** Intermediate | **v0.3.5+**

## Overview

This tutorial demonstrates the **chainable Configuration API** of jaxfne v0.3.5+, introduced for streamlined workflow composition. You will build, simulate, and analyze a **100-neuron excitatory-inhibitory (E/I) population** using the new fluent configuration grammar.

### What You'll Learn

- Use the chainable `Configuration` API (`runtime()`, `column()`, `cell_types()`, `connectivity()`, `set_emitter()`, `probes()`)
- Configure a biologically-inspired E/I population (75% excitatory, 25% inhibitory)
- Simulate with the Izhikevich neuron model (cortical parameters)
- Extract multimodal readouts: spikes, voltage, source, and field proxies
- Inspect scope metadata to understand computational claims and limitations
- Generate population-level metrics and visualizations

### Key Concepts

**Balanced Networks:** Cortical populations are typically ~80% excitatory and ~20% inhibitory. This balance allows excitation and inhibition to approximately match, a principle central to cortical computation.

**Chainable API:** The new `Configuration` class uses method chaining to compose models fluently:

```python
cfg = (jtfne.Configuration()
    .runtime(seed=42, duration_ms=1000.0, dt_ms=0.1)
    .column(name="L2/3_column", layers=["L2/3"], n=100)
    .cell_types({"E": 0.75, "I": 0.25})
    .connectivity()
    .set_emitter(family="izhikevich", preset="cortical_eig")
    .probes(["SPK", "Vm", "source", "LFP-proxy", "CSD-proxy"]))
```

**Proxy-Scale Readouts:** Field outputs (LFP, CSD) are computed via convolution-based proxies, not full PDE solutions. This enables fast visualization and relative comparison without claiming physical amplitude.

---

## Tutorial Workflow

The notebook follows a 13-section structure:

1. **Learning Objectives** — Expected outcomes
2. **Biological Question** — The E/I coupling problem
3. **Mathematical Glossary** — Izhikevich equations and metrics
4. **Canonical Import** — `import jaxfne as jtfne`
5. **Configuration Block** — Chainable API setup
6. **Simulation Block** — Construct model and run
7. **Probe/Readout Block** — Compute population metrics
8. **Manifest and Scope Metadata** — JSON output with scope claims
9. **Figures** — Five visualizations (raster, rates, voltages, source, summary)
10. **Interpretation** — Results and dynamics
11. **Failure Modes** — Debugging silent or unstable networks
12. **Exercises** — Extensions (vary E/I ratio, try presets, etc.)
13. **Scope Boundaries** — What this tutorial is and is not

---

## Quick Reference

### Configuration

```python
cfg = jtfne.Configuration()

# Set runtime parameters
cfg = cfg.runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)

# Define cortical column: 100 neurons
cfg = cfg.column(name="L2/3_column", layers=["L2/3"], n=100)

# Excitatory-inhibitory composition
cfg = cfg.cell_types({"E": 0.75, "I": 0.25})

# All-to-all connectivity
cfg = cfg.connectivity()

# Izhikevich neuron with cortical parameters
cfg = cfg.set_emitter(family="izhikevich", preset="cortical_eig")

# Multimodal probes
cfg = cfg.probes(["SPK", "Vm", "source", "LFP-proxy", "CSD-proxy"])
```

### Simulation and Readouts

```python
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=42)

# Compute population metrics
readout_specs = [
    jtfne.readout_spec("population_rate", "spike_rate_hz"),
    jtfne.readout_spec("mean_voltage", "mean_V_m"),
]
readouts = model.compute_readout(signals, readout_specs)
```

### Expected Results

| Metric | Typical Value | Valid Range |
|--------|---------------|-------------|
| Excitatory rate | 8–15 Hz | 2–25 Hz |
| Inhibitory rate | 8–15 Hz | 2–25 Hz |
| Population mean rate | 8–15 Hz | 2–25 Hz |
| Voltage min | ~-80 mV | -100 to -60 mV |
| Voltage max | ~30 mV | 20–40 mV |
| All outputs finite | True | — |

---

## Scope and Limitations

### ✓ What This Tutorial Is

- A **computational scaffold** for learning E/I coupling principles
- A demonstration of the **new chainable Configuration API**
- A **proxy-scale** visualization and relative-comparison tool
- **JAX-based, deterministic, and reproducible**

### ✗ What This Tutorial Is NOT

- A **biophysically realistic** neuron model (Izhikevich is phenomenological)
- A **validated field solver** (LFP/CSD are proxy approximations, not PDE-solved)
- A tool for **direct comparison with real recordings** (uncalibrated)
- A framework for **source localization** (no inverse problem)
- A **complete cortical circuit** (missing plasticity, neuromodulation)

### Scope Metadata

All outputs include manifest JSON with explicit scope claims:

```json
{
  "scope_status": "computational_scaffold",
  "calibration_status": "uncalibrated_phenomenological",
  "readout_status": "proxy_scale",
  "field_mode": "proxy_convolution_no_pde",
  "physical_amplitude_claim_allowed": false
}
```

**Key principle:** Proxy-scale outputs enable comparison and visualization without claiming physical validity.

---

## Running the Notebook

### In Colab (Recommended)

Click the **Open in Colab** badge at the top. The notebook runs CPU-safe in ~2–3 minutes.

### Locally

```bash
# Download and run
jupyter notebook tutorials/jaxfne_v036_100_neuron_ei_population.ipynb

# Or execute with nbconvert
jupyter nbconvert --execute tutorials/jaxfne_v036_100_neuron_ei_population.ipynb
```

### Validation

```bash
# Run tests
JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1 pytest tests/test_v036_100_neuron_ei_population_tutorial.py -v
```

---

## Next Steps

After this tutorial:

- **Modify E/I ratios** — Explore how balance affects stability
- **Try emitter presets** — "tonic_spiking", "phasic_spiking", "fast_spiking"
- **Extend to multiple layers** — Add L4 and L5 with inter-layer connections
- **Calibrate to real data** — Use guided exercises in [Calibration guide](../guides/calibration.md)
- **Explore field relationships** — See [Tensor-Field Workflows](../guides/tensor_field_workflows.md)

---

## References

- **Izhikevich neuron model:** Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Trans. Neural Netw., 14(6), 1569–1572.
- **Balanced networks:** van Vreeswijk, C., & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitatory and inhibitory activity. Science, 274(5293), 1724–1726.
- **jaxfne API:** See [Configuration](../api/core.md), [Runtime](../api/runtime.md), [Emitters](../api/emitters.md), [Probes](../api/probes.md)

---

## Troubleshooting

**Silent network (< 1 Hz)?**  
Try a different `seed`, switch emitter preset to `"tonic_spiking"`, or increase external input current.

**Runaway network (> 50 Hz)?**  
Use a preset with stronger adaptation (e.g., `"chattering"`) or increase inhibitory coupling strength.

**NaN/Inf in output?**  
Reduce timestep to 0.01 ms, use float64 precision, or check Izhikevich parameter bounds.

For more help, see [FAQ](../faq.md) or [Guides](../guides/index.md).
