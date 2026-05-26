# Jaxley interoperability

[Jaxley](https://github.com/google/jaxley) and jaxfne are complementary JAX-based projects:

- **Jaxley:** Differentiable neuron and network models in JAX (compartmental, multi-compartment, conductance-based)
- **jaxfne:** Source-to-field/readout layer for organizing outputs into tensor-field workflows

## Using Jaxley with jaxfne

Jaxley builds neuron/network models; jaxfne organizes their outputs into multimodal readouts.

### Example: Jaxley neuron → jaxfne readout

```python
import jaxley as jx
import jaxfne as jtfne
import jax.numpy as jnp

# Build a Jaxley neuron
neuron = jx.Neuron(
    conductances={"leak": 0.1, ...},
    mech_params={...}
)

# Simulate with Jaxley
v_jax = neuron.simulate(duration=100.0, ...)  # [time, compartments]

# Reshape for jaxfne (e.g., [time, neurons] where neurons=1)
source_input = v_jax[:, -1:, None]  # soma voltage

# Create jaxfne workflow
cfg = (
    jtfne.configuration()
    .network(n=1)
    .emitter(family="custom", mode="external_voltage")
    .field(domain="point")
    .probe(name="jaxley_soma")
)

model = jtfne.construct(cfg)
signals = model.simulate_external(source_input, ...)
readouts = model.compute_readout(signals, [...])
```

### Array-first trace bridge

jaxfne provides a minimal array-first bridge for converting Jaxley-style voltage traces to jaxfne Signals without running a full simulation:

```python
import jaxfne as jtfne
import numpy as np

# Jaxley-style voltage trace: [time, neurons] in mV
trace = np.random.randn(1000, 16) * 10 - 70  # 1000 timesteps, 16 neurons

# Convert via bridge
spec = jtfne.JaxleyTraceSpec(dt_ms=0.1)  # 0.1 ms timestep
signals = jtfne.jaxley_trace_to_signals(trace, spec=spec)

# Result: jaxfne.core.Signals with time_ms, V_m, spikes, metadata
print(f"V_m shape: {signals.V_m.shape}")  # (1000, 16)
print(f"Claim level: {signals.metadata['claim_level']}")  # computational_scaffold
```

**Key features:**
- No Jaxley installation required (optional dependency)
- Accepts NumPy or JAX arrays
- Flexible layout support: `time_by_unit` [T,N], `unit_by_time` [N,T], `recording_by_time` [R,T]
- Spike proxy derivation via voltage threshold (default 0.0 mV, configurable)
- Conservative voltage-proxy source (ionic current mapping deferred)
- Scope declaration: `computational_scaffold`, `physical_amplitude_claim_allowed=False`

**Scope specification:**
All outputs are marked as:
- `claim_level: "computational_scaffold"` — Designed for computational workflows and tutorial scenarios
- `physical_amplitude_claim_allowed: False` — Voltage is treated as a proxy readout, not a physical claim
- `source_calibration_status: "uncalibrated_jaxley_voltage_proxy"` — Calibration to physical units pending
- `field_solver_status: "not_computed"` — Field/LFP computation is optional downstream work

**Important:** The voltage trace is treated as a proxy external readout. Voltage is treated as a computational proxy without biological mechanism claims; field computation is optional; ionic current modeling is deferred. Field computation is deferred to the jaxfne probe/field layer if needed.

**See also:**
- `examples/07_jaxley_trace_bridge.py` — Full tutorial with layout conversion and threshold variation
- `JaxleyTraceSpec` — Configuration for trace metadata, dt, spike threshold
- `jaxley_trace_to_signals()` — Main conversion function

### Bridging: array layout conventions

Jaxley outputs spike times or voltage traces. jaxfne expects:

- **Shape:** `[time, neurons]` or `[time, neurons, features]`
- **Units:** Declared (Jaxley units → jaxfne declared mapping)
- **Indexing:** Neuron/compartment ID → jaxfne neuron index

Check [Output bundles](output_bundles.md) for metadata conventions.

## Design philosophy

jaxfne provides optional Jaxley interoperability. Key features:

- Works with or without Jaxley installation (optional dependency)
- Accepts any JAX arrays shaped consistently
- Preserves Jaxley model autonomy and differentiability

Use jaxfne when you want to:

- Organize Jaxley outputs into field/probe workflows
- Add LFP-proxy, CSD-proxy, EEG-proxy readouts downstream
- Build local/global interaction summaries
- Serialize workflows with validation metadata

## Next steps

- **[Tensor-field workflows](tensor_field_workflows.md)** for pipeline overview
- **[Tutorials](../tutorials/index.md)** for end-to-end examples
- **[Jaxley documentation](https://github.com/google/jaxley)** for Jaxley-specific topics
