# 600-neuron Population with Laminar Readout


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_4_homeostatic_V1_column.ipynb)

A 600-neuron E/PV/SST/VIP population read out through the laminar field proxy. This
configuration builds one unlayered population with dense recurrent connectivity; the
layered V1 column (six layers, per-layer populations, laminar positions) is the canonical
track starting at [01 — Define](01_define_genome.md).

*The Colab badge above links to `jaxfne_etude_no_4_homeostatic_V1_column.ipynb`, a
1000-neuron canonical laminar column with homeostasis; it is a different model from the
600-neuron population below.*

## Configuration

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        n=600,
        cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy")
    .probe(
        name="v1_column",
        n_contacts=6,
        modes=["spikes", "V_m", "LFP", "CSD"]
    )
)

model = jtfne.construct(cfg)
```

## Multimodal readouts

```python
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=0)

# Population-level readouts (the names are labels)
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz"),
    jtfne.readout_spec("lfp", "lfp_abs_mean"),
    jtfne.readout_spec("csd", "csd_abs_mean"),
])
```

## What the readout is

- Six contacts at evenly spaced relative depths in [0, 1].
- LFP proxy: each neuron's source current projected onto the contacts by a Gaussian of
  width 0.10 in relative depth.
- CSD proxy: the second spatial difference of that LFP proxy along the contacts.

## Canonical Atlas Visualization
```python
from jaxfne.vis.atlas_suite import build_atlas

# Emits the 7 canonical panels (schema, 3D, raster, LFP, H-dynamics, HDP, oscillatory)
manifest = build_atlas(model, signals, out_dir="docs/_static/atlas/v1_column")
print("Emitted panels:", [p["panel"] for p in manifest["panels"]])
```

## Interactive atlas (dark)

Dark-theme Plotly panels from this page's circuit (1000 ms, dt 0.5 ms, seed 0; the page shows dt 0.1 ms): [index](../_static/atlas/v1_column/index.html) · [schema](../_static/atlas/v1_column/schema.html) · [3D](../_static/atlas/v1_column/network_3d.html) · [raster](../_static/atlas/v1_column/raster.html) · [LFP](../_static/atlas/v1_column/lfp.html) · [H](../_static/atlas/v1_column/h_dynamics.html) · [HDP](../_static/atlas/v1_column/hdp.html) · [oscillatory](../_static/atlas/v1_column/oscillatory.html).

Regenerate: `python scripts/generate_doc_page_atlases.py --slug v1_column`.

## Next step

Progress to [V1-PFC dual column](05_v1_pfc_dual_column.md) for multi-areal networks.
