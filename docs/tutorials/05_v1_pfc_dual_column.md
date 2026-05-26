# V1-PFC Dual Column


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/docs/tutorials/05_v1_pfc_dual_column.md)

Two cortical columns (V1 and PFC) with inter-areal connections. Explore cross-area dynamics.

## Configuration

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        n=1200,
        areas={"V1": 600, "PFC": 600},
        layers={"V1": ["L1", "L2/3", "L4", "L5", "L6"], "PFC": ["L1", "L2/3", "L5", "L6"]},
        inter_areal_connectivity={"V1→PFC": 0.15, "PFC→V1": 0.10}
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="dual_laminar_column")
    .probe(name="v1_pfc_dual", modes=["spikes", "LFP", "traveling_waves"])
)

model = jtfne.construct(cfg)
```

## Explore inter-areal dynamics

```python
signals = model.simulate(...)

readouts = model.compute_readout(signals, [
    jtfne.readout_spec("V1_rate", "spike_rate_hz"),
    jtfne.readout_spec("PFC_rate", "spike_rate_hz"),
    jtfne.readout_spec("cross_area_coherence", "coherence"),
])
```

## Key features

- Bi-directional V1 ↔ PFC connections
- Laminar specificity: V1 L4 → PFC L1, PFC L5 → V1 L1
- LFP-proxy shows areal-specific spectral signatures
- Traveling-wave analysis (planned) for inter-areal phase dynamics

## Applications

- Attention and gain modulation (V1 ← PFC feedback)
- Perceptual binding and temporal coordination
- Visual working memory circuits

## Next steps

See [Guides](../guides/index.md) for how-to articles on extending these models.
