# Suite No. 1: Computational Biophysics Tutorial

This tutorial teaches the standard jaxfne workflow:

```python
import jaxfne as jtfne

cfg = jtfne.Configuration()
cfg = cfg.runtime(...)
cfg = cfg.column(...)
cfg = cfg.cell_types(...)
cfg = cfg.connectivity(...)
cfg = cfg.set_emitter(...)
cfg = cfg.probes(...)

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, ...)
readouts = model.probe(signals, ...)
jtfne.vis.spectrolaminar(signals)
```

The notebook is organized as a four-part journey:

1. configured single neuron;
2. configured E/PV population;
3. configured laminar column and package visualizer;
4. package-native objective and tuning metadata.

## Tutorial scope

This tutorial teaches the jaxfne public API: Configuration, construct, simulate, probes, and visualization. All readouts are simulated outputs for learning and validation purposes.

## Runtime metadata

Canonical tutorial settings:

- `duration_ms = 1000.0`
- `dt_ms = 0.1`
- `dtype = float32`
- deterministic seed
- strict JSON-safe metadata
- PNG figures generated from the notebook

## Scope metadata

All outputs are simulated/proxy readouts from a computational scaffold. The tutorial does not provide calibrated physical amplitudes, subject-specific sensor geometry, empirical validation, detailed compartmental morphology, or mechanistic proof of a biological circuit.

## Generated figures

The notebook writes PNG figures under:

```text
figures/suite_no1/
```

Core figures include:

- configured single-neuron voltage-like trajectory;
- configured single-neuron source proxy;
- configured population raster;
- configured population rate summary;
- constructed connectivity matrix;
- `jtfne.vis.spectrolaminar` output;
- configured laminar readout summary;
- package tuning report summary.
