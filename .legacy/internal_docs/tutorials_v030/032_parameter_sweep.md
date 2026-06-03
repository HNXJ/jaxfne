# jaxfne v0.3.2: Single-Neuron Parameter Sweep Tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/docs/tutorials_v030/032_parameter_sweep.md)

**How native drive and parameters shape single-neuron firing rate and dynamics.**

## Learning Objectives

1. Parameter sensitivity — how drive and Izhikevich parameters change firing rate.
2. Sweep design — construct small, deterministic, CPU-safe parameter grids.
3. Sweep interpretation — read firing-rate curves and recognize spiking regimes.
4. Table and figure summaries — visualize results as grids, heatmaps, and curves.
5. Finite-output validation — detect and interpret parameter sensitivity edge cases.

## Biological/Computational Question

**How do native drive and reduced-emitter parameters shape single-neuron firing rate, voltage trajectory, and proxy readouts?**

We address this by systematically varying native drive and recording firing rate, spike count, and output finiteness.

## Mathematical Glossary Flow

### Drive Sweep Grid

**Formal equation:**
$$I_{\mathrm{native}}^{(k)} = I_{\mathrm{base}} + \Delta I \cdot k$$

**Worded equation:** Each sweep condition applies a fixed drive increment. The neuron is simulated with each value.

**Scope boundary:** Drive is a tutorial parameter, not empirically calibrated.

### Firing-Rate Response Function

**Formal equation:**
$$r^{(k)} = \frac{N_{\mathrm{spikes}}^{(k)}}{T_{\mathrm{seconds}}}$$

**Worded equation:** For each drive, count spikes and divide by total time.

**Scope boundary:** Within-run summary; not compared to biology.

## Configuration Block

```python
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
cfg = cfg.cell_types({"E": 1.0})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["MUA-proxy", "source-proxy"])
```

Sweep over native drive (or other parameters if the public API supports it).

## Interpretation

### What the Sweep Shows

1. **Monotonic or stepwise increase** in firing rate with drive.
2. **Threshold effects:** Some drive values produce spiking; others do not.
3. **Finite outputs:** All conditions should be finite.
4. **Target range:** At least one condition in 2–25 Hz.

This demonstrates parameter sensitivity without optimization.

## Scope Boundaries

### Covers

- Single-neuron parameter sweep design.
- Firing-rate response to drive variation.
- Table and figure summaries.
- Finite-output validation.

### Does NOT Cover

- Two-neuron coupling (v0.3.3).
- Optimization loops (future).
- Biological calibration.
- Conductance-based models.

All results labeled: `truth_safe_unverified / computational_scaffold / uncalibrated`

