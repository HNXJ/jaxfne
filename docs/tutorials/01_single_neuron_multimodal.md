# Single-neuron multimodal

Build, simulate, and inspect a single Izhikevich neuron. Extract spikes, voltage, and readout metrics.

## Setup

```python
import jaxfne as jtfne
import matplotlib.pyplot as plt
```

## Configure the neuron

```python
cfg = (
    jtfne.configuration()
    .network(n=1)
    .emitter(family="izhikevich", preset="regular_spiking")
    .field(domain="point")
    .probe(name="single_neuron", modes=["spikes", "V_m"])
)
```

Key parameters:

- **n=1:** One neuron
- **preset="regular_spiking":** Izhikevich parameter set (RS, FS, IB, etc. available)
- **domain="point":** No spatial extent (suitable for single neuron)

## Build the model

```python
model = jtfne.construct(cfg)
```

## Simulate

```python
sim = jtfne.simulation(
    duration_ms=100.0,
    dt_ms=0.1,
    seed=0
)

signals = model.simulate(sim)
```

**signals** contains:

- `signals.spikes`: [T, N] spike matrix (binary)
- `signals.V_m`: [T, N] voltage trace (mV)

## Compute readouts

```python
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("spike_count", "spike_count"),
    jtfne.readout_spec("mean_voltage", "voltage_mean"),
])

for r in readouts:
    print(f"{r.name}: {r.value:.2f} [{r.status}]")
```

## Visualize

```python
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Voltage trace
axes[0].plot(signals.V_m[:, 0])
axes[0].set_ylabel("Voltage (mV)")
axes[0].set_title("Neuron membrane potential")

# Spike raster
spike_times = signals.spikes[:, 0].nonzero()[0]
axes[1].vlines(spike_times, 0, 1)
axes[1].set_ylabel("Spike")
axes[1].set_xlabel("Time (ms)")
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.show()
```

## Key takeaways

- Single neuron simulations are the foundation
- Output is JSON-serializable (see `model.manifest(signals, readouts)`)
- Next: move to [Two-neuron E/I](02_two_neuron_ei.md) for dynamics

## Further reading

- [Probe operators](../probe_operators.md)
- [Output bundles](../output_bundles.md)
