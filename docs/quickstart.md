# Quickstart

## Minimal example

```python
import jaxfne as jtfne

# Configure a single-neuron simulation
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
cfg = cfg.cell_types({"E": 1.0})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])

# Construct and simulate
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

# Inspect results
print(f"jaxfne {jtfne.__version__}")
print(f"Simulation: {signals.V_m.shape[0]} timesteps, {signals.V_m.shape[1]} neuron(s)")
print(f"Voltage range: {signals.V_m.min():.1f} to {signals.V_m.max():.1f} mV")
print(f"Spike count: {signals.spikes.sum():.0f}")
print(f"Firing rate: {signals.spikes.sum() / (1000.0 / 1000.0):.1f} Hz")
```

## What happens

1. **Configuration** builds a chainable configuration object
2. **Runtime setup** specifies seed, dtype, duration, and timestep
3. **Column definition** declares neurons and their positions
4. **Cell types** assigns neuron types (E, inhibitory, etc.)
5. **Connectivity** defines recurrent connections
6. **Emitter selection** picks the spiking neuron model (Izhikevich)
7. **Probe specification** selects readout operators
8. **Model construction** wires up the neural dynamics
9. **Simulation** runs the dynamics and returns signals
10. **Results** are JSON-safe with scope metadata (proxy readouts, no biological claims)

All outputs declare their scope: **computational scaffold with proxy readouts**.

## Next steps

- See [Probe Operators](probe_operators.md) for all available readout channels
- Explore [Guides](guides/index.md) for advanced workflows and calibration
- Explore `examples/` in the repository for complete tutorials

## Validation

Run the test suite to verify your installation:

```bash
pip install -e .[dev]
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=short
```

Expected: 1062 passed, 37 skipped
