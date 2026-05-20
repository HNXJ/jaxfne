# Tensor-field workflows

jaxfne organizes neural model outputs into a source-to-field/readout pipeline:

```
Emitter/Network → Source tensors → Field operators → Probe readouts → Output bundles
```

## Pipeline overview

### 1. Emitter outputs

Start with spike or voltage outputs from:

- jaxfne-native Izhikevich emitters
- Jaxley-style neuron/network models
- Custom JAX arrays (organized as [time, neurons] or [time, neurons, features])

### 2. Source tensors

Organize emitter outputs into spatially localized currents:

- **Source projection:** Map emitter states to physical/proxy current
- **Spatial support:** Declare layer/depth assignments for field solvers
- **Metadata:** Track source origin (emitter type, calibration status)

### 3. Field operators

Compute extracellular potentials and derived quantities:

- **LFP-proxy:** Average extracellular potential at contact depths
- **CSD-proxy:** Current-source density (second spatial derivative)
- **Geometry:** Laminar contact depths or custom field geometry

### 4. Probe readouts

Extract multimodal outputs:

- **SPK:** Spike times or spike matrix
- **Vm:** Membrane voltage or emitter state
- **Source:** Source tensor (projection or native)
- **LFP-proxy, CSD-proxy:** Field quantities
- **EEG-proxy, MEG-proxy:** Linear projections (toy or declared)
- **EMM-proxy:** Normalized activity cost

### 5. Output bundles

Serialize workflows as JSON-safe manifests:

- **Data arrays:** Readouts, signals, field outputs
- **Metadata:** Operator status, units, assumptions, claim-status
- **Receipts:** Run ID, seed, execution time, validation flags
- **Validation:** JSON encoding enforces strict serialization (no NaN/Inf by default)

## Example: Single neuron multimodal

```python
import jaxfne as jtfne

# Configure
cfg = (
    jtfne.configuration()
    .network(n=1)
    .emitter(family="izhikevich")
    .field(domain="point")
    .probe(name="single_neuron", modes=["spikes", "V_m", "source"])
)

# Build
model = jtfne.construct(cfg)

# Simulate
signals = model.simulate(jtfne.simulation(duration_ms=100.0, dt_ms=0.1))

# Readouts
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("spike_count", "spike_count"),
    jtfne.readout_spec("voltage_mean", "voltage_mean"),
])

# Manifest
manifest = model.manifest(signals, readouts)
```

## Local and global summaries

For circuit-level workflows, tensor-field operations can produce:

- **Local summaries:** Per-layer or per-region spike rates, LFP power
- **Global summaries:** Whole-network activity, cross-layer synchronization
- **Traveling-wave summaries:** (Planned) Spatiotemporal dynamics across layers/regions

See tutorials for examples.

## Next steps

- **[Tutorials](tutorials/index.md)** for progressively detailed workflows
- **[Calibration](calibration.md)** for preparing outputs for empirical validation
- **[Jaxley interoperability](jaxley_interop.md)** for using Jaxley models as sources
