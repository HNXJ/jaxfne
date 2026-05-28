# Suite No. 1: Computational Biophysics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/dev/tutorials/jaxfne_colab_tutorial_computational_biophysics.ipynb)

This notebook teaches the public jaxfne grammar.

```python
cfg = jtfne.Configuration()
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, ...)
objectives = jtfne.rate_targets(...)
optimizer = jtfne.agsdr(...)
result = model.tune(objectives=objectives, optimizer=optimizer)
```

## Learning objectives

1. Configure single-neuron Izhikevich dynamics.
2. Build an E/I population model.
3. Compute source, field, and probe proxy readouts.
4. Run multi-objective AGSDR tuning.

## Runtime settings

- `duration_ms = 1000.0`
- `dt_ms = 0.1`
- `dtype = float32`
- deterministic seed
- JSON-safe summaries
- PNG figures

## Figures

The notebook writes PNG files under:

```text
figures/suite_no1/
```

Core figures:

- voltage/state trace
- source proxy
- population raster
- population rate
- connectivity matrix
- laminar readout
- tuning summary
