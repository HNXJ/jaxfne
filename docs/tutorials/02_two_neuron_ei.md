# Two-neuron E/I

Build a minimal recurrent network: one excitatory and one inhibitory neuron.
Observe coupling, dynamics, and multimodal readouts on a small network.

## Open as Colab notebook

**Recommended:** Open the full interactive tutorial in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/artifacts/tutorials/jaxfne_v033_two_neuron_ei.ipynb)

Or run headless: `python examples/v033_two_neuron_ei_multimodal.py`

## Network configuration

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(n=2, cell_types={"E": 0.5, "PV": 0.5})
    .emitter(family="izhikevich", preset="cortical_eig")
    .field()
    .probe(name="two_neuron_ei", modes=["spikes", "V_m"])
)

model = jtfne.construct(cfg)
```

## Simulate and inspect

```python
signals = model.simulate(jtfne.simulation(duration_ms=500.0, dt_ms=0.1))

# Mean rate over both neurons; per-neuron rates in label order (E, PV)
readouts = model.compute_readout(signals, [jtfne.readout_spec("rate", "spike_rate_hz")])
rates_hz = signals.spikes.mean(axis=0) * 1000.0 / 0.1
```

## Observe recurrent dynamics

- construct connects the pair with its default weights: E→PV +0.35 and PV→E −0.35
  (0.5/√2), no self-connections.
- At the default drive (E 5.0, PV 3.0) E fires at 12 Hz and PV stays below
  threshold, so the loop never closes: E spikes at the same times as with the
  connections removed (`jtfne.simulation(..., ablation="disconnected_null")`).
- E's input moves PV's membrane potential by at most 0.04 mV over the 500 ms run.

## Interactive atlas (dark)

Dark-theme Plotly panels from this page's circuit (500 ms, dt 0.5 ms, seed 0; the page shows dt 0.1 ms): [index](../_static/atlas/two_neuron_ei/index.html) · [schema](../_static/atlas/two_neuron_ei/schema.html) · [3D](../_static/atlas/two_neuron_ei/network_3d.html) · [raster](../_static/atlas/two_neuron_ei/raster.html) · [LFP](../_static/atlas/two_neuron_ei/lfp.html) · [H](../_static/atlas/two_neuron_ei/h_dynamics.html) · [HDP](../_static/atlas/two_neuron_ei/hdp.html) · [oscillatory](../_static/atlas/two_neuron_ei/oscillatory.html).

Regenerate: `python scripts/generate_doc_page_atlases.py --slug two_neuron_ei`.

## Next step

Progress to [100-neuron E/I network](03_network_100_ei.md) for larger-scale circuits.
