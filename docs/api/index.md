# API reference

Complete API documentation for jaxfne classes, functions, and modules.

## Modules

### Core (`jaxfne.core`)

Main classes for configuration, model construction, simulation, and readouts.

- **[Configuration](core.md)** — `jaxfne.configuration()`
- **[Model](core.md)** — `jaxfne.Model`
- **[Simulation](core.md)** — `jaxfne.simulation()`
- **[Signals](core.md)** — Output signals container
- **[Readout](core.md)** — Readout specifications and results

### Emitters (`jaxfne.emitters`)

Neuron models and emitter implementations.

- **[Izhikevich](emitters.md)** — Izhikevich neuron model
- **[Parameters](emitters.md)** — Emitter parameter sets

### Fields (`jaxfne.fields`)

Field solvers and proxy operators.

- **[Source projection](fields.md)** — Source tensor organization
- **[Field solvers](fields.md)** — LFP/CSD computation
- **[Boundary conditions](fields.md)** — Neumann, Dirichlet, mean-zero

### Probes (`jaxfne.probes`)

Probe operators and readout channels.

- **[Operators](probes.md)** — SPK, Vm, source, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy

### Objectives (`jaxfne.objectives`)

Objective functions for optimization.

- **[Specifications](objectives.md)** — Objective specs and evaluation

### Validation (`jaxfne.validation`)

Invariant checks and claim-status validation.

- **[Checks](validation.md)** — Numerical and structural invariants

## Quick reference

| Class | Purpose |
|-------|---------|
| `Configuration` | Build network and model specs |
| `Model` | Constructed jaxfne workflow |
| `Simulation` | Run parameters |
| `Signals` | Neural outputs (spikes, voltage, field) |
| `Readout` | Computed metrics |

## Example

```python
import jaxfne as jtfne

# Configure
cfg = jtfne.configuration().network(n=100).emitter(...).field(...)

# Build
model = jtfne.construct(cfg)

# Simulate
signals = model.simulate(jtfne.simulation(duration_ms=100.0))

# Readouts
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz")
])
```

## Detailed documentation

See individual module pages for method signatures and examples.
