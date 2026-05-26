# API Reference

Complete API documentation for jaxfne classes, functions, and modules.

---

## Modules Overview

### [Core API](core.md) — Configuration, Models, Simulation

Main classes for building, constructing, and running neural simulations.

**Key classes:**
- `Configuration` — Declarative model specification with chainable methods
- `Model` — Compiled workflow ready for simulation
- `Simulation` — Run parameters (duration, timestep, seed)
- `Signals` — Output container (spikes, voltage, field readouts)
- `ReadoutSpec` / `ReadoutResult` — Metric specifications and computed values

**Key functions:**
- `configuration()` — Create a new Configuration
- `construct(cfg)` — Build Model from Configuration
- `simulate(model, ...)` — Run simulation
- `readout_spec(name, metric)` — Define a readout metric

**[→ Full Core API documentation](core.md)**

---

### [Emitters API](emitters.md) — Neuron Models

Neuron dynamics and spiking behavior definitions.

**Key components:**
- `IzhikevichParams` — Phenomenological spiking neuron parameters
- `ReceptorSpec` — Synaptic receptor kinetics (AMPA, NMDA, GABA)
- `SynapseSpec` — Single synaptic connection definition
- `EIGNetwork` — Excitatory-Inhibitory-Gap network topology
- `EdgeList` — Sparse connectivity representation

**Key functions:**
- `make_eig_network(n_exc, n_inh)` — Create network structure
- `standard_receptor_specs()` — Get standard receptor types
- `simulate_eig_izhikevich(cfg, I_ext, seed)` — Run network dynamics

**[→ Full Emitters API documentation](emitters.md)**

---

### [Fields API](fields.md) — Spatial Projection & Field Solvers

Source-to-field transformations and field computation.

**Key components:**
- `LaminarSourceGeometry` — Anatomical spatial organization
- `FieldOutput` — Container for field readouts (LFP, CSD)

**Key functions:**
- `project_laminar_sources(currents, geometry)` — Map currents to space
- `project_sources_to_laminar_field(sources, geometry)` — Compute LFP/CSD
- `probe_laminar_modes(sources, basis_spec)` — Spatial decomposition
- `validate_source_field_status(field)` — Check field consistency
- `compute_conservation_proxy_diagnostics(sources, field)` — Energy metrics

**[→ Full Fields API documentation](fields.md)**

---

### [Probes API](probes.md) — Multimodal Readouts

Eight probe operators for extracting neural readouts.

**Probe operators:**
1. **SPK** — Spike detection (binary raster)
2. **Vm** — Membrane voltage (direct state)
3. **source** — Transmembrane current (spatial projection)
4. **LFP-proxy** — Local field potential
5. **CSD-proxy** — Current source density
6. **EEG-proxy** — Electroencephalogram (scalp)
7. **MEG-proxy** — Magnetoencephalogram
8. **EMM-proxy** — Energetic metabolic metric

**Available metrics:**
- `spike_rate_hz`, `burst_frequency_hz`, `max_spike_rate_hz`
- `mean_V_m`, `min_V_m`, `max_V_m`
- `mean_source`, `mean_LFP`, `mean_CSD`, `mean_EEG`, `mean_MEG`, `mean_EMM`

**[→ Full Probes API documentation](probes.md)**

---

### [Objectives API](objectives.md) — Optimization Targets

Objective specifications for parameter optimization.

**Key components:**
- `Objective` — Single optimization target
- `ObjectiveReport` — Objective evaluation results

**Key functions:**
- `objective(name, metric, target, weight)` — Define optimization target

**Use cases:**
- Single-objective optimization (e.g., achieve 10 Hz firing rate)
- Multi-objective optimization (e.g., balance rate and voltage)
- Custom loss functions (MSE, MAE, RMSE)

**[→ Full Objectives API documentation](objectives.md)**

---

### [Runtime API](runtime.md) — Execution Configuration

JAX backend, device, and numerical precision settings.

**Key components:**
- `RuntimeConfig` — Backend and device configuration

**Key settings:**
- `seed` — Random seed for reproducibility
- `dtype` — Numerical precision ('float32', 'float64')
- `device` — Compute target ('cpu', 'gpu', 'tpu')
- `enable_x64` — Enable 64-bit precision globally

**Key functions:**
- `enable_x64()` — Enable double precision
- `runtime_report()` — Print environment info

**[→ Full Runtime API documentation](runtime.md)**

---

### [Validation API](validation.md) — Checks & Invariants

Configuration validation and consistency checks.

**Key functions:**
- `validate_config(cfg)` — Check configuration validity
- `validate_source_field_status(field)` — Verify field consistency
- `validate_projection_invariants(sources, field)` — Check source-field mapping
- `compute_conservation_proxy_diagnostics(sources, field)` — Energy metrics
- `operator_status()` — Get operator status declarations
- `config_truth_boundary(cfg)` — Get claim boundaries

**[→ Full Validation API documentation](validation.md)**

---

## Quick Reference Table

| Class | Module | Purpose |
|-------|--------|---------|
| `Configuration` | core | Declarative model specification |
| `Model` | core | Compiled simulation workflow |
| `Signals` | core | Output container (spikes, voltage, fields) |
| `Simulation` | core | Run parameters |
| `RuntimeConfig` | runtime | Execution backend settings |
| `IzhikevichParams` | emitters | Neuron model parameters |
| `ReceptorSpec` | emitters | Synaptic kinetics |
| `LaminarSourceGeometry` | fields | Spatial anatomy |
| `FieldOutput` | fields | Field readouts (LFP, CSD) |
| `Objective` | objectives | Optimization target |
| `ObjectiveReport` | objectives | Optimization results |

---

## Common Workflows

### 1. Basic Simulation

```python
import jaxfne as jtfne

# Configure
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("V1", layers=["L2/3"], n=100)
cfg = cfg.cell_types({"E": 0.8, "I": 0.2})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["SPK", "Vm", "LFP-proxy"])

# Build and simulate
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

# Compute metrics
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("firing_rate", "spike_rate_hz"),
    jtfne.readout_spec("mean_voltage", "mean_V_m")
])
```

### 2. Multi-Objective Optimization

```python
objectives = [
    jtfne.objective("rate", "spike_rate_hz", target=15.0, weight=1.0),
    jtfne.objective("voltage", "mean_V_m", target=-55.0, weight=1.0),
]

# Optimize parameters
optimizer = jtfne.gsdr(model, objectives, param_name="I_injected", n_steps=50)
best_params = optimizer.optimize()
```

### 3. Validation & Diagnostics

```python
# Validate configuration
result = jtfne.validate_config(cfg)
assert result.valid

# Check output consistency
status = jtfne.validate_source_field_status(field_output)
assert status["all_finite"]

# Get conservation metrics
diag = jtfne.compute_conservation_proxy_diagnostics(sources, field)
print(f"Energy ratio: {diag['energy_ratio']:.3f}")
```

---

## Scope & Limitations

⚠️ **All outputs are computational proxies:**

- **Izhikevich model:** Phenomenological, not biophysical
- **Source projection:** Declared anatomy, not 3D solved
- **Field solvers:** Proxy convolution, not PDE solutions
- **Readouts:** Relative metrics, not validated measurements
- **Claim level:** `"computational_scaffold"` for learning and prototyping

**Use for:** Teaching, prototyping, validation workflows

**Not suitable for:** Quantitative biological modeling, empirical fitting

---

## Full Documentation

- **[Scope and Limitations](../scope_and_limitations.md)** — Detailed boundaries and claims
- **[Probe Operators](../probe_operators.md)** — Mathematical definitions of eight operators
- **[Computation Basis](../computation_basis.md)** — TFNE architecture overview
- **[Mathematical Glossary](../mathematical_glossary_flow.md)** — Equation reference

---

## Index

**By Module:**
- [Core API](core.md)
- [Emitters API](emitters.md)
- [Fields API](fields.md)
- [Probes API](probes.md)
- [Objectives API](objectives.md)
- [Runtime API](runtime.md)
- [Validation API](validation.md)

**By Category:**
- Configuration & Construction: [Core API](core.md)
- Neural Dynamics: [Emitters API](emitters.md)
- Spatial & Field: [Fields API](fields.md)
- Readout Channels: [Probes API](probes.md)
- Optimization: [Objectives API](objectives.md)
- Execution: [Runtime API](runtime.md)
- Consistency: [Validation API](validation.md)
