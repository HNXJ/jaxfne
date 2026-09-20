# Source Bookkeeping, Field Handoff & Probe Readout

**Difficulty:** Intermediate  
**Duration:** 20–30 minutes  
**Scope:** Computational scaffold, simulated proxy fields, tutorial-scale learning

---

## Overview

This tutorial covers the jaxfne **source bookkeeping API**: implicit source declaration (emitter + probes), flow to fields (convolution proxies), and metadata gating of amplitude status.

The core concepts:

1. **Source Declaration (Implicit):** emitter type (`izhikevich`, preset) + probe modes decide which fields get computed
2. **Field Handoff (Convolution-based):** fixed kernels spread sources spatially (not PDE-solved)
3. **Probe Readout (Configurable):** modes (`source`, `LFP-proxy`, `CSD-proxy`) give different field views
4. **Scope Clarity (Metadata):** manifest keys (`amplitude_status=False`) block misreads

Not a biophysical validation tutorial. A **computational scaffold** mapping neural sources to observable fields.

---

## Interactive 3D Column Visualization

Below is a **standalone Plotly HTML** laminar column with:

- **3D neuron scatter:** colored by firing rate, placed by layer and depth
- **Hover metadata:** ID, layer, cell type, depth, source index, mean rate
- **Readout panels:** source summary, population rate, LFP-proxy, laminar voltage
- **Equation notes:** source bookkeeping (S), field handoff (Y = P·S), probe readout (R = Q·Y)
- **Controls:** pan, zoom, rotate (mouse + keyboard)

### View the Interactive Column

<iframe 
  src="../../assets/interactive/v037_source_column_3d.html"
  width="100%"
  height="800"
  style="border: 1px solid #ccc; margin: 20px 0;">
  <p>
    If the visualization doesn't load, you can open it directly:
    <a href="../../assets/interactive/v037_source_column_3d.html" target="_blank">
      v037_source_column_3d.html (opens in new tab)
    </a>
  </p>
</iframe>

---

## Understanding the Visualization

### Neuron Colors (Firing Rate)

Hover shows one row per neuron:

| Field | Meaning |
|---|---|
| Neuron ID | Index in the population |
| Layer | L2/3, L4, L5, or L6 |
| Cell type | E (excitatory), PV (parvalbumin-positive), SST, or VIP |
| Depth | Distance from surface (µm) |
| Source index | Mapping to source current in simulation |
| Rate | Mean firing rate (Hz) |

**Color scale (Viridis):** Blue = low rate, Yellow = high rate

### The Equations

Three core relations annotated in the view:

$$S(t) \in \mathbb{R}^{T \times N}$$
**Source bookkeeping:** Time-series of neural source currents. T = timepoints, N = neurons.

$$Y(t) = P \cdot S(t)$$
**Field handoff:** Spatial convolution (P is the convolution kernel) maps point sources to field.

$$R_k(t) = Q_k \cdot Y(t)$$
**Probe readout:** Different probes (k = `source`, `LFP-proxy`, `CSD-proxy`) extract different aspects of the field via operators Q_k.

---

## Configuration API & Source Declaration

### The Implicit Rule

Sources are **not explicitly declared**. They are **inferred** from:

1. **Emitter type & preset:** decides available sources (e.g., Izhikevich → intrinsic currents + synaptic input)
2. **Probe modes:** decide which sources get computed and returned

```python
import jaxfne as jtfne

cfg = (jtfne.Configuration()
    .runtime(seed=42, dtype='float32', duration_ms=1000, dt_ms=0.1)
    .column(name='tutorial_column', layers=['L2/3', 'L4', 'L5', 'L6'], n=48)
    .cell_types({'E': 0.70, 'PV': 0.15, 'SST': 0.10, 'VIP': 0.05})
    .connectivity(kind='laminar_signed_metadata', recurrent=True)
    .set_emitter('izhikevich', 'cortical_eig')
    .probes(['spikes', 'V_m', 'source', 'LFP-proxy', 'CSD-proxy'], n_contacts=16))

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000, dt_ms=0.1, seed=42)
```

**Key observation:** no `.declare_source()` method. Flow:
- **Emitter** defines neuronal dynamics and available source types
- **Probes** select which sources to extract and how to compute them
- **Signals** object returns the requested readouts

---

## Signals API Rules

After `jtfne.simulate()`, `signals` holds:

```python
signals.spikes       # np.ndarray, shape (T, N), boolean spike indicator
signals.V_m          # np.ndarray, shape (T, N), membrane voltage
signals.sources      # np.ndarray, shape (T, N) or (T, S), source currents
signals.time_ms      # np.ndarray, shape (T,), time axis in milliseconds
signals.metadata     # dict, scope/readout metadata (see below)
```

### Metadata Keys (Scope Clarity)

```python
signals.metadata = {
    "scope_status": "computational_scaffold",
    "field_claim_level": "proxy_readout",
    "representation": "relative",
    "physical_amplitude_calibrated": False,
    "source_bookkeeping": {
        "physical_amplitude_calibrated": False,
        "representation": "relative",
    },
    "duration_ms": 1000.0,
    "dt_ms": 0.1,
    "dtype": "float32",
    "seed": 42,
}
```

**Key field:** `physical_amplitude_calibrated=False` marks amplitudes as Relative.

---

## Probe Modes & Field Computation

### Available Probe Modes

| Mode | Shape | Description | Biophysical? |
|------|-------|-------------|--------------|
| `spikes` | (T, N) | Spike detection (boolean) | Simplified |
| `V_m` | (T, N) | Membrane voltage | Phenomenological |
| `source` | (T, N) | Raw source currents from emitter | Phenomenological |
| `LFP-proxy` | (T, C) | Local-field-potential proxy via convolution | **Proxy only** |
| `CSD-proxy` | (T, C) | Current-source-density proxy (spatial derivative) | **Proxy only** |

### How Proxy Fields are Computed

**Not PDE-solved.** Instead:

1. Extract neuron source currents: $S(t)$
2. Apply fixed spatial kernel (e.g., Gaussian): $Y(t) = P \cdot S(t)$
3. Take spatial derivatives where needed (CSD)

Fast (no solver loop), approximate (proxy-scoped tutorial data).

---

## Source-to-Field Mechanism

### The Handoff Flow

```
Neuron Emitter (Izhikevich)
    ↓
Intrinsic Currents (I_intrinsic)
    + Synaptic Input (I_syn = W @ s)
    = Source Signal S(t)
    ↓
Convolution Kernel P
    ↓
Field Y(t) = P * S(t)
    ↓
Probe Operators (Q_LFP, Q_CSD, etc.)
    ↓
Readout R(t) = Q @ Y(t)
```

### Spatial Representation

- **Source coordinates:** point sources at each neuron location
- **Field coordinates:** expanded via convolution (spatial units arbitrary, normalized)
- **Kernel:** Gaussian (default); width set by distance in µm
- **Boundary:** zero-padding (no boundary currents)

**Spatial units:** Relative, normalized units. The view uses µm for
layer depth (anatomical reference); field amplitudes are Relative-value.

---

## Manifest & Metadata Validation

### Run Manifest Template

```python
import json

RUN_METADATA = {
    "scope_status": "computational_scaffold",
    "readout_status": "simulated_proxy",
    "field_mode": "proxy_convolution_no_pde",
    "amplitude_status": False,
    "duration_ms": 1000.0,
    "dt_ms": 0.1,
    "dtype": "float32",
    "seed": 42,
    "n_neurons": 48,
    "layers": ["L2/3", "L4", "L5", "L6"],
    "mean_population_rate_hz": 2.5,
    "voltage_range_mv_like": [-86.0, 30.0],
    "all_outputs_finite": True,
    "equations": {
        "source_bookkeeping": "S(t) ∈ ℝ^{T×N}",
        "field_handoff": "Y(t) = P·S(t)",
        "probe_readout": "R_k(t) = Q_k·Y(t)",
    },
}

# Validate JSON safety (no NaN/Inf)
json.dumps(RUN_METADATA, allow_nan=False)
```

### JSON Safety

All manifest outputs must serialize with `allow_nan=False`:

```python
# OK
json.dumps(manifest, allow_nan=False)

# Will fail if NaN/Inf present
manifest["rate"] = float('nan')
json.dumps(manifest, allow_nan=False)  # ← JSONDecodeError
```

---

## Statement Gates & Interpretation

### The Gate: `amplitude_status`

This boolean key blocks misreads:

```python
if not metadata["amplitude_status"]:
    # BLOCKED: Stating real-world amplitude
    # "The LFP-proxy amplitude is 50 µV"
    
    # ALLOWED: Relative or tutorial statements
    # "LFP-proxy increases during high firing rate"
    # "Layer 5 sources dominate the field"
```

### Tutorial scope

- No biophysical compartments (soma, dendrite, axon)
- No temperature sensitivity, frequency-dependent effects
- No subject-specific anatomy
- No experimental validation
- Kernels are fixed defaults (not tunable in this tutorial)

### Reserved extensions

- Custom convolution kernels via `.field_kernel()` method
- PDE-based field solvers (optional)
- Calibration to real neural recordings
- Frequency-response properties

---

## Summary & Next Steps

### What You've Learned

1. **Sources are implicit:** emitter + probes decide field computation
2. **Fields are proxies:** convolution-based, fast, approximate
3. **Readouts are multi-modal:** each operator gives a different field view
4. **Metadata gates statements:** `amplitude_status=False` blocks misreads

### How to Use This in Your Work

```python
# Step 1: Configure a column
cfg = jtfne.Configuration().set_emitter(...).probes([...])

# Step 2: Simulate
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, ...)

# Step 3: Check scope before interpreting
assert not signals.metadata["physical_amplitude_calibrated"]

# Step 4: Use relative comparisons, not absolute statements
layer5_rate = signals.spikes[layer5_idx].mean()
layer23_rate = signals.spikes[layer23_idx].mean()
print(f"L5 rate is {layer5_rate / layer23_rate:.1f}x L2/3 rate")  # ✓ OK

# Step 5: Document scope in your output
json.dump(signals.metadata, fp, allow_nan=False)
```

---

## References

- **[Configuration API population](./06_v036_100_neuron_ei_population.md)**
- **API Reference:** [API Overview](../api/index.md)
- **Guides:** [Probe Operators](../guides/probe_operators.md) | [Tensor-Field Workflows](../guides/tensor_field_workflows.md)
- **Interactive Visualization:** The 3D source/field/probe column is embedded above via Plotly (interactive HTML)
- **GitHub:** [jaxfne Issues](https://github.com/HNXJ/jaxfne/issues)

---

**End of tutorial**

Feedback? Open an issue: [jaxfne/issues](https://github.com/HNXJ/jaxfne/issues)

## Interactive atlas (dark)

Dark-theme Plotly panels from this page's 48-neuron column (1000 ms, dt 0.5 ms, seed 42; the page shows dt 0.1 ms), complementing the 3D view above: [index](../_static/atlas/source_column_48/index.html) · [schema](../_static/atlas/source_column_48/schema.html) · [3D](../_static/atlas/source_column_48/network_3d.html) · [raster](../_static/atlas/source_column_48/raster.html) · [LFP](../_static/atlas/source_column_48/lfp.html) · [H](../_static/atlas/source_column_48/h_dynamics.html) · [HDP](../_static/atlas/source_column_48/hdp.html) · [oscillatory](../_static/atlas/source_column_48/oscillatory.html).

Regenerate: `python scripts/generate_doc_page_atlases.py --slug source_column_48`.
