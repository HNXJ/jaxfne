# Tensor-field workflows

jaxfne organizes neural model outputs into a source-to-field/readout pipeline:

```
Emitter/Network → Source tensors → Field operators → Probe readouts → Output bundles
```

## Pipeline overview

### 1. Emitter outputs

Start with spike or voltage outputs from:

- jaxfne Izhikevich emitters
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
- **Source:** Source tensor (projection or base)
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

## Mathematical notation

The source-to-field-to-probe pipeline uses standardized operator notation:

### Source tensor construction

$$\mathbf{S}(t) \in \mathbb{R}^{T \times N} \quad \text{or} \quad \mathbf{S}(t,\ell) \in \mathbb{R}^{T \times N \times L}$$

Source tensor from emitter: $T$ time steps, $N$ neurons, optional $L$ layers/depths.

### Source-to-field proxy projection

$$\phi_{\mathrm{proxy}}(t,c) = \sum_{n=1}^{N} W_{cn} S_n(t)$$

Potential at contact $c$ via row-normalized kernel $W$:

$$\sum_{n=1}^{N} W_{cn} = 1 \quad \forall c$$

Declarative projection-based computation; PDE solving deferred to physical solver path.

### Row-normalized Gaussian kernel (example)

$$W_{cn} = \frac{\exp\left(-\frac{\|z_c - z_n\|^2}{2\sigma^2}\right)}{\sum_{k=1}^{N} \exp\left(-\frac{\|z_c - z_k\|^2}{2\sigma^2}\right)}$$

Kernel centered at contact depth $z_c$, localized by $\sigma$. Normalization enforces row sum = 1.

### Field-to-probe readout (generic operator form)

$$Y_j(t) = \mathcal{O}_j[\phi_{\mathrm{proxy}}, \mathbf{S}, \ldots](t)$$

Generic probe operator $\mathcal{O}_j$ applies to field potential, source, or both.

### JSON-safe report sidecar

$$\mathcal{R} = \{\mathrm{field\_solver\_status}, \mathrm{gauge}, \mathrm{boundary\_condition}, \mathrm{csd\_sign\_convention}, \ldots\}$$

Report declares solver path (proxy vs. physical), convergence status, and claim constraints.

---

## Field/proxy diagnostics (v0.2.6+)

jaxfne distinguishes between proxy readout paths and future physical solver paths using field diagnostics:

### Proxy readout path (v0.2.3–present)

Laminar proxy operators project source tensors directly to contacts without solving a PDE:

- **Source-balance:** Deferred to physical solver path (proxy mode omits PDE validation)
- **Gauge:** Declared metadata only
- **Boundary condition:** Declared metadata only
- **Physical amplitude claims:** False by default

### Physical solver path (planned v0.3+)

Future versions will support full PDE solvers with these diagnostics:

- **Source-balance:** Integrated source vs boundary flux residual
- **Gauge:** Mean-zero residual checking
- **Boundary condition:** Flux residual validation
- **Manufactured residual:** PDE solver convergence metric

### Using diagnostics

```python
from jaxfne.validation import (
    make_field_operator_status,
    make_source_balance_diagnostic,
    make_gauge_diagnostic,
)

# Declare proxy path
operator = make_field_operator_status(operator_path="proxy")
# → field_solver_status: "laminar_proxy_no_pde"

# Declare physical_candidate path (for future integration)
operator = make_field_operator_status(operator_path="physical_candidate")
# → field_solver_status: "physical_field_solver_candidate"
```

All diagnostics keep `physical_amplitude_claim_allowed: false` in v0.2.6. Physical amplitude claims require separate calibration and validation evidence.

## Local and global summaries

For circuit-level workflows, tensor-field operations can produce:

- **Local summaries:** Per-layer or per-region spike rates, LFP power
- **Global summaries:** Whole-network activity, cross-layer synchronization
- **Traveling-wave summaries:** (Planned) Spatiotemporal dynamics across layers/regions

See tutorials for examples.

## Next steps

- **[Tutorials](../tutorials/index.md)** for progressively detailed workflows
- **[Calibration](calibration.md)** for preparing outputs for empirical validation
- **[Jaxley interoperability](jaxley_interop.md)** for using Jaxley models as sources
