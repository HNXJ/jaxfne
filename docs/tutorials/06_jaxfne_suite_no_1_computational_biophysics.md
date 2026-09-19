# Suite No. 1: Computational Biophysics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_suite_no_1_computational_biophysics.ipynb)

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
outputs/suite_no1/figures/
```

Core figures:

- voltage/state trace
- source proxy
- population raster
- population rate
- connectivity matrix
- laminar readout
- tuning summary

## Interactive atlas (dark)

Dark-theme Plotly panels from a smoke-scale 1000 ms run (dt 0.5 ms, seed 44) of the notebook's 48-neuron laminar column (the notebook runs 5000 ms at dt 0.1 ms): [index](../_static/atlas/suite1_column/index.html) · [3D](../_static/atlas/suite1_column/network_3d.html) · [connectivity](../_static/atlas/suite1_column/connectivity.html) · [raster](../_static/atlas/suite1_column/raster.html) · [traces](../_static/atlas/suite1_column/traces.html) · [spectral](../_static/atlas/suite1_column/spectral.html) · [state summary](../_static/atlas/suite1_column/state_summary.html).

Regenerate: `python scripts/generate_doc_page_atlases.py --slug suite1_column`.
