# 100-neuron E/I network

Balanced excitatory/inhibitory network with 80 E and 20 I neurons. Explore population dynamics.

## Network configuration

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(
        n=100,
        cell_types={"E": 0.8, "I": 0.2},
        connectivity="dense_random"  # or specify explicitly
    )
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="point")
    .probe(name="network_100", modes=["spikes"])
)

model = jtfne.construct(cfg)
```

## Simulate

```python
signals = model.simulate(jtfne.simulation(duration_ms=1000.0, dt_ms=0.1))

# Population rate
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("population_rate", "spike_rate_hz"),
])
```

## Observations

- Population exhibits asynchronous irregular spiking in balanced regime
- LFP-proxy shows fluctuations from population activity
- Suitable for optimization and control experiments

## Next step

Progress to [V1 six-layer column](04_v1_column.md) for structured laminar networks.
