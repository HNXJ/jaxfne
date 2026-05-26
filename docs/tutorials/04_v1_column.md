# V1 Six-layer Column


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/docs/tutorials/04_v1_column.md)

A laminar model inspired by primate V1 with six layers and depth-specific readouts.

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
signals = model.simulate(...)

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

## Next step

Progress to [V1-PFC dual column](05_v1_pfc_dual_column.md) for multi-areal networks.
