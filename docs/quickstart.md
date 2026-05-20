# Quickstart

## Minimal example

```python
import json
import jaxfne as jtfne

# Create configuration
cfg = (
    jtfne.configuration()
    .network(n=100)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy",
           boundary="mean_zero_neumann", gauge="mean_zero")
    .probe(name="probe", n_contacts=16)
)

# Construct model
model = jtfne.construct(cfg)

# Run simulation
sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1)
signals = model.simulate(sim)

# Compute readouts
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz"),
    jtfne.readout_spec("lfp", "lfp_abs_mean"),
])

# Print results
print(f"jaxfne {jtfne.__version__}")
print(f"Spikes shape: {signals.spikes.shape}")
print(f"Voltage shape: {signals.V_m.shape}")
for result in readouts:
    print(f"{result.name}: {result.metric} = {result.value} [{result.status}]")
```

## What happens

1. **Configuration** builds a 100-neuron network with Izhikevich emitters
2. **Model construction** wires up the source-to-field pipeline
3. **Simulation** runs 100 ms of neural dynamics at 0.1 ms timestep
4. **Readouts** compute proxy metrics (spike rate, LFP)
5. **Results** are JSON-safe with claim-status metadata

All outputs declare their scope: **proxy readouts with conservative claim-status metadata**.

## Next steps

- See [Probe Operators](probe_operators.md) for all available readout channels
- Read [Roadmap](ROADMAP.md) for v0.2.4–v0.2.21 phases
- Explore `examples/` in the repository for complete tutorials

## Validation

Run the test suite to verify your installation:

```bash
pip install -e .[dev]
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q --tb=short
```

Expected: ~61 tests pass
