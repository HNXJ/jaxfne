# Tutorials

Learn jaxfne by working through progressively detailed examples. Each tutorial builds on the previous one.

## Beginner tutorials

**[Single-neuron multimodal](01_single_neuron_multimodal.md)**

Start here. Build, simulate, and inspect a single Izhikevich neuron with spikes, voltage, and readout operators.

**[Two-neuron E/I](02_two_neuron_ei.md)**

Excitatory and inhibitory neurons connected. Observe recurrent dynamics and coupling effects.

## Intermediate tutorials

**[100-neuron E/I network](03_network_100_ei.md)**

A balanced network of excitatory and inhibitory neurons. Explore local population activity and stability.

**[V1 six-layer column](04_v1_column.md)**

A laminar model inspired by primate V1 with six layers (L1, L2/3, L4, L5, L6) and depth-specific readouts.

## Advanced tutorial

**[V1-PFC dual column](05_v1_pfc_dual_column.md)**

Two cortical columns (V1 and PFC) with inter-areal connections. Explore cross-area interaction and traveling-wave dynamics.

## Running tutorials

Tutorials are available as Jupyter notebooks in the `tutorials/` directory:

```bash
jupyter notebook tutorials/01_single_neuron_multimodal.ipynb
```

Or run directly with nbconvert:

```bash
nbconvert --execute tutorials/01_single_neuron_multimodal.ipynb
```

## Quick example: Single-neuron primer

```python
import jaxfne as jtfne

# Configure
cfg = (
    jtfne.configuration()
    .network(n=1)
    .emitter(family="izhikevich", preset="regular_spiking")
    .field(domain="point")
    .probe(name="single", modes=["spikes", "V_m"])
)

# Build and simulate
model = jtfne.construct(cfg)
signals = model.simulate(jtfne.simulation(duration_ms=100.0))

# Inspect
print(f"Spike count: {signals.spikes.sum()}")
print(f"Voltage shape: {signals.V_m.shape}")
```

## Next steps

After tutorials:

- **[Guides](../guides/index.md)** for how-to articles and workflow tips
- **[API reference](../api/index.md)** for full class/function documentation
- **[Jaxley interoperability](../jaxley_interop.md)** for using external models
