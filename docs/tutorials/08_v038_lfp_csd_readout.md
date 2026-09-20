# LFP/CSD Readout Tutorial

**Version:** 0.3.8  
**Difficulty:** Intermediate  
**Duration:** 15–20 minutes to read; 5–10 minutes to execute  
**Scope:** Computational scaffold, simulated proxy fields, tutorial-scale learning

---

## Overview

This tutorial covers the jaxfne **source-to-field-to-readout workflow** for laminar contact arrays: implicit source currents (emitter + config), Gaussian-kernel projection to contacts, and LFP-proxy / CSD-proxy readout.

The core concepts:

1. **Source Declaration (Implicit):** emitter type + neuron count decide available sources
2. **Spatial Projection (Gaussian Kernel):** default **`density_preserving`** mode (not row-normalized). Use `mode="row_normalize"` only when explicitly intended (not PDE-solved)
3. **LFP-proxy:** spatially-smoothed source projection for local field potential
4. **CSD-proxy:** second spatial derivative of LFP-proxy for current-source density
5. **Probe Readout:** eight multimodal operators for spikes, voltage, sources, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy
6. **Scope Clarity:** metadata gates (`amplitude_status=False`) block amplitude overstates

This is a **computational scaffold**, not a biophysically validated model.

---

## Mathematical Model

### Source Bookkeeping

$$S(t) \in \mathbb{R}^{T \times N}$$

**Worded equation:** Source activity is a time-by-neuron matrix. Entry S(t, n) is the current from neuron n at time t.

### Projection to Laminar Contacts

$$Y(t, c) = \sum_{n=1}^{N} K(c, n) \cdot S(t, n)$$

where $K \in \mathbb{R}^{C \times N}$ is the Gaussian projection kernel.

**Worded equation:** Each contact sums neural sources. Kernel K(c, n) weights neurons near contact c highest.

### Gaussian Kernel (Row-Normalized)

$$K(c, n) = \frac{\exp\left(-0.5 \left(\frac{d_c - d_n}{w}\right)^2\right)}{\sum_{n'=1}^{N} \exp\left(-0.5 \left(\frac{d_c - d_{n'}}{w}\right)^2\right)}$$

where $d_c$ is contact depth, $d_n$ is neuron depth, and $w = 0.10$ is the kernel width.

**Worded equation:** Gaussian kernel at each contact depth, width w. Row normalization gives each contact a weighted summary.

### CSD-proxy Readout (Second Spatial Derivative)

$$\text{CSD}(t, c) \approx -\frac{Y(t, c-1) - 2Y(t, c) + Y(t, c+1)}{\Delta z^2}$$

where $\Delta z = 1/(C-1)$ is the contact spacing.

**Worded equation:** CSD-proxy is local field curvature via second difference across neighbor contacts. Negative sign follows electrostatic convention.

### Probe Readout

$$R_k(t) = Q_k(S(t), V(t), \text{spike count}, \ldots)$$

**Worded equation:** Each probe (k = spikes, V_m, source, LFP-proxy, CSD-proxy) summarizes neural state differently.

---

## Configuration API & Workflow

### The Public API Contract

Sources are **not explicitly declared**. They are **inferred** from:

1. **Emitter type & preset:** decides available sources (e.g., Izhikevich → intrinsic + synaptic currents)
2. **Probe modes:** decide which sources get computed and returned

```python
import jaxfne as jtfne

# Single neuron example
cfg_single = (jtfne.Configuration()
    .runtime(seed=42, dtype='float32', duration_ms=1000, dt_ms=0.1)
    .column(name='single_neuron_lfp', layers=['L2/3'], n=1)
    .cell_types({'E': 1.0})
    .connectivity(kind='none')
    .set_emitter('izhikevich', 'cortical_eig')
    .probes(['spikes', 'V_m', 'source', 'LFP-proxy', 'CSD-proxy'], n_contacts=16))

model = jtfne.construct(cfg_single)
signals = jtfne.simulate(model, duration_ms=1000, dt_ms=0.1, seed=42)
```

### E/I Laminar Column Example

```python
cfg_laminar = (jtfne.Configuration()
    .runtime(seed=42, dtype='float32', duration_ms=1000, dt_ms=0.1)
    .column(name='laminar_lfp_csd', layers=['L2/3', 'L4', 'L5', 'L6'], n=12)
    .cell_types({'E': 0.75, 'PV': 0.15, 'SST': 0.05, 'VIP': 0.05})
    .connectivity(kind='laminar_signed_metadata', recurrent=True)
    .set_emitter('izhikevich', 'cortical_eig')
    .probes(['spikes', 'V_m', 'source', 'LFP-proxy', 'CSD-proxy'], n_contacts=16))

model = jtfne.construct(cfg_laminar)
signals = jtfne.simulate(model, duration_ms=1000, dt_ms=0.1, seed=42)
```

### Extracting Readouts

```python
# Access signals directly
spikes = signals.spikes          # [T, N]
V_m = signals.V_m                # [T, N]
sources = signals.sources        # [T, N]

# Access LFP and CSD via probe
readouts = model.probe(signals, modes=['LFP-proxy', 'CSD-proxy'])
lfp = readouts['LFP-proxy']      # [T, C]
csd = readouts['CSD-proxy']      # [T, C]
```

---

## Signals API Rules

See [Source Bookkeeping](07_v037_source_bookkeeping.md#signals-api-rules) and [Source/Field Equations](../source_field_equations.md) for `Signals` fields and `metadata` (`amplitude_status=False` gates physical claims).

---

## Example 1: Single Neuron → Contacts

A single neuron in L2/3 projects to 16 evenly-spaced laminar contacts.

**Key observations:**

- **Source shape:** [T=10000, N=1] (time × neuron)
- **LFP-proxy shape:** [T=10000, C=16] (time × contacts)
- **CSD-proxy shape:** [T=10000, C=16] (time × contacts, second derivative)
- **Single source → distributed field:** point source smoothed by Gaussian kernel gives smooth LFP-proxy across contacts
- **Nearest contacts highest:** contacts near neuron depth get strongest signal

---

## Example 2: E/I Laminar Column

A 48-neuron laminar column (12 per layer, 4 layers), mixed E/I.

**Configuration:**

- **Neurons:** 48 total = 4 layers × 12 neurons/layer
- **Cell types:** E (75%), PV (15%), SST (5%), VIP (5%)
- **Connectivity:** Recurrent laminar connectivity (within and across layers)
- **Duration:** 1000 ms with 0.1 ms timestep

**Key observations:**

- **Source shape:** [T=10000, N=48]
- **LFP-proxy shape:** [T=10000, C=16]
- **Laminar structure emerges:** layer-wise E/I gives distinct laminar profiles
- **CSD-proxy marks layer edges:** second derivative shows where sources concentrate
- **Population rate:** typically 2–25 Hz (active-state regime here)

---

## Example 3: Layer-Resolved Analysis

Extract which layers dominate the population-level field.

**Methods:**

```python
# Partition neurons by layer
layer_indices = {
    'L2/3': np.arange(0, 12),
    'L4': np.arange(12, 24),
    'L5': np.arange(24, 36),
    'L6': np.arange(36, 48)
}

# Compute layer-resolved firing rates
for layer, indices in layer_indices.items():
    layer_spikes = signals.spikes[:, indices]
    layer_rate = (layer_spikes.mean() * 1000.0 / DT_MS)
    print(f"{layer}: {layer_rate:.2f} Hz")
```

**Interpretation:**

- Which layer fires most? (Typically L4/L5 in cortical columns)
- Does deep layer (L5) dominate the LFP-proxy? (Often yes, due to larger somatic currents)
- How does layer-resolved structure vary over time?

---

## Probe Modes & Field Computation

See [Probe Operators](../guides/probe_operators.md) for the eight operators; here: `spikes`, `V_m`, `source`, `LFP-proxy`, `CSD-proxy` (see table in [Source Bookkeeping](07_v037_source_bookkeeping.md)). Proxy fields use fast convolution, not PDE-solved.

---

## Validation & JSON Safety

See [Source Bookkeeping](07_v037_source_bookkeeping.md) for the run manifest template and `json.dumps(..., allow_nan=False)` gate.

---

## Interpretation & Statement Gates

See [Source Bookkeeping](07_v037_source_bookkeeping.md) for `amplitude_status` gate and scope limits (no compartments, no calibration, proxy-scale only).

### Reserved extensions

- Custom convolution kernels via `.field_kernel()` method
- PDE-based field solvers (optional)
- Calibration to real neural recordings
- Frequency-response properties

---

## Summary & Next Steps

See [Source Bookkeeping](07_v037_source_bookkeeping.md) for workflow summary; this tutorial adds laminar CSD/LFP specifics.

| Use | Pattern |
|---|---|
| Check scope | `assert not signals.metadata["physical_amplitude_calibrated"]` |
| Compare relatively | `L5_rate / L23_rate` |
| Document | `json.dump(signals.metadata, fp, allow_nan=False)` |

---

## References

- **[Source bookkeeping](./07_v037_source_bookkeeping.md)**
- **[Configuration API population](./06_v036_100_neuron_ei_population.md)**
- **API Reference:** [API Overview](../api/index.md)
- **Guides:** [Probe Operators](../guides/probe_operators.md) | [Tensor-Field Workflows](../guides/tensor_field_workflows.md)
- **GitHub:** [jaxfne Issues](https://github.com/HNXJ/jaxfne/issues)

---

**End of tutorial**

Feedback? Open an issue: [jaxfne/issues](https://github.com/HNXJ/jaxfne/issues)

## Interactive atlas (dark)

Dark-theme Plotly panels from the 12-neuron laminar example (1000 ms, dt 0.5 ms, seed 42; the page shows dt 0.1 ms): [index](../_static/atlas/lfp_csd_12/index.html) · [schema](../_static/atlas/lfp_csd_12/schema.html) · [3D](../_static/atlas/lfp_csd_12/network_3d.html) · [raster](../_static/atlas/lfp_csd_12/raster.html) · [LFP](../_static/atlas/lfp_csd_12/lfp.html) · [H](../_static/atlas/lfp_csd_12/h_dynamics.html) · [HDP](../_static/atlas/lfp_csd_12/hdp.html) · [oscillatory](../_static/atlas/lfp_csd_12/oscillatory.html).

Regenerate: `python scripts/generate_doc_page_atlases.py --slug lfp_csd_12`.
