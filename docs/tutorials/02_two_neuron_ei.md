# Two-neuron E/I

Build a minimal recurrent network: one excitatory and one inhibitory neuron. Observe coupling and dynamics.

## Network configuration

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        n=2,
        cell_types={"E": 1, "I": 1},
        connectivity={"E→E": 0.1, "E→I": 0.2, "I→E": -0.3, "I→I": -0.1}
    )
    .emitter(family="izhikevich", preset="regular_spiking")
    .field(domain="point")
    .probe(name="two_neuron_ei", modes=["spikes", "V_m"])
)

model = jtfne.construct(cfg)
```

## Simulate and inspect

```python
signals = model.simulate(jtfne.simulation(duration_ms=500.0, dt_ms=0.1))

# Readouts
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("E_rate", "spike_rate_hz"),
    jtfne.readout_spec("I_rate", "spike_rate_hz"),
])
```

## Observe recurrent dynamics

- Excitatory neuron drives inhibitory neuron
- Inhibitory feedback suppresses excitatory spiking
- Network exhibits oscillatory or stable behavior depending on connection strengths

## Next step

Progress to [100-neuron E/I network](03_network_100_ei.md) for larger-scale circuits.
