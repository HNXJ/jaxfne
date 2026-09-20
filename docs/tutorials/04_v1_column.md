# V1 Six-layer Column


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/etudes/jaxfne_etude_no_4_homeostatic_V1_column.ipynb)

Laminar model inspired by primate V1: six layers, depth-specific readouts.

*The Colab badge above links to `jaxfne_etude_no_4_homeostatic_V1_column.ipynb`, the closest
real, runnable notebook covering a laminar V1 column (1000-neuron canonical column with
homeostasis); it is not a byte-for-byte match of the 600-neuron config shown below.*

## Configuration

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        n=600,
        layers=["L1", "L2/3", "L4", "L5", "L6"],
        cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
        connectivity="layer_structured"
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(
        domain="laminar_column",
        conductivity="proxy",
        depths=[0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
    )
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

# Layer-specific rates
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("L4_rate", "spike_rate_hz"),
    jtfne.readout_spec("LFP_L4", "lfp_abs_mean"),
    jtfne.readout_spec("CSD_L4", "csd_abs_mean"),
])
```

## Laminar features

- L4 receives thalamocortical input (configurable)
- L2/3, L5, L6 have inter-laminar projections
- LFP-proxy reflects population summed current
- CSD-proxy reflects current source densities per layer

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
