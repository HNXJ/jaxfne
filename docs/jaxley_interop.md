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

### Bridging: array layout conventions

Jaxley outputs spike times or voltage traces. jaxfne expects:

- **Shape:** `[time, neurons]` or `[time, neurons, features]`
- **Units:** Declared (Jaxley units → jaxfne declared mapping)
- **Indexing:** Neuron/compartment ID → jaxfne neuron index

Check [Output bundles](output_bundles.md) for metadata conventions.

## Design philosophy

No hard dependency exists. jaxfne:

- Does not require Jaxley to be installed
- Accepts any JAX arrays shaped consistently
- Preserves Jaxley model autonomy and differentiability

Use jaxfne when you want to:

- Organize Jaxley outputs into field/probe workflows
- Add LFP-proxy, CSD-proxy, EEG-proxy readouts downstream
- Build local/global interaction summaries
- Serialize workflows with validation metadata

## Next steps

- **[Tensor-field workflows](tensor_field_workflows.md)** for pipeline overview
- **[Tutorials](tutorials/index.md)** for end-to-end examples
- **[Jaxley documentation](https://github.com/google/jaxley)** for Jaxley-specific topics
