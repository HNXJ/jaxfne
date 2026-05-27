# v0.3.7: Source Bookkeeping, Field Handoff & Probe Readout

**Version:** 0.3.7  
**Difficulty:** Intermediate  
**Duration:** 20–30 minutes  
**Scope:** Computational scaffold, simulated proxy fields, tutorial-scale learning

---

## Overview

This tutorial documents the jaxfne **source bookkeeping API**: how neural sources are declared (implicitly via emitter + probes), how they flow to fields (convolution-based proxies), and how metadata gates physical amplitude claims.

The core concepts:

1. **Source Declaration (Implicit):** Emitter type (`izhikevich`, preset) + probe modes determine which fields are computed
2. **Field Handoff (Convolution-based):** Sources spread spatially via fixed convolution kernels (not PDE-solved)
3. **Probe Readout (Configurable):** Multiple readout modes (`source`, `LFP-proxy`, `CSD-proxy`) extract different field perspectives
4. **Scope Clarity (Metadata):** Manifest keys (`physical_amplitude_claim_allowed=False`) prevent misinterpretation

This is **not a biophysical validation tutorial**. It is a **computational scaffold** for understanding how neural sources map to observable fields in the jaxfne framework.

---

## Interactive 3D Column Visualization

The visualization below is a **standalone Plotly HTML** showing a laminar cortical column with:

- **3D neuron scatter:** Colored by firing rate, positioned by layer and depth
- **Hover metadata:** Neuron ID, layer, cell type, depth, source index, mean firing rate
- **Readout panels:** Source summary, population firing rate, LFP-proxy, laminar voltage profile
- **Equation annotations:** Source bookkeeping (S), field handoff (Y = P·S), probe readout (R = Q·Y)
- **Interactive controls:** Pan, zoom, rotate (mouse + keyboard)

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

Hover over any neuron to see:
- **Neuron ID:** Index in the population
- **Layer:** L2/3, L4, L5, or L6
- **Cell type:** E (excitatory), PV (parvalbumin-positive), SST, or VIP
- **Depth:** Distance from surface (µm)
- **Source index:** Mapping to source current in simulation
- **Rate:** Mean firing rate (Hz)

**Color scale (Viridis):** Blue = low rate, Yellow = high rate

### The Equations

Three core relationships are annotated in the visualization:

$$S(t) \in \mathbb{R}^{T \times N}$$
**Source bookkeeping:** Time-series of neural source currents. T = timepoints, N = neurons.

$$Y(t) = P \cdot S(t)$$
**Field handoff:** Spatial convolution (P is the convolution kernel) maps point sources to field.

$$R_k(t) = Q_k \cdot Y(t)$$
**Probe readout:** Different probes (k = `source`, `LFP-proxy`, `CSD-proxy`) extract different aspects of the field via operators Q_k.

---

## Configuration API & Source Declaration

### The Implicit Contract

Sources are **not explicitly declared**. Instead, they are **inferred** from two decisions:

1. **Emitter type & preset:** Determines what sources are available (e.g., Izhikevich → intrinsic currents + synaptic input)
2. **Probe modes:** Determines which sources are computed and returned

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

**Key observation:** There is no `.declare_source()` method. The flow is:
- **Emitter** defines neuronal dynamics and available source types
- **Probes** select which sources to extract and how to compute them
- **Signals** object returns the requested readouts

---

## Signals API Contract

After `jtfne.simulate()`, the returned `signals` object has:

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
    "readout_status": "simulated_proxy",
    "field_mode": "proxy_convolution_no_pde",
    "physical_amplitude_claim_allowed": False,
    "duration_ms": 1000.0,
    "dt_ms": 0.1,
    "dtype": "float32",
    "seed": 42,
}
```

**Critical key:** `physical_amplitude_claim_allowed=False` gates claims about real-world amplitude. This is a computational scaffold, not a calibrated model.

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

1. Extract source currents from neurons: $S(t)$
2. Apply fixed spatial convolution kernel (e.g., Gaussian): $Y(t) = P \cdot S(t)$
3. Optionally compute spatial derivatives (for CSD)

This is **fast** (no solver loop) but **approximate** (not validated against real data).

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

- **Source coordinates:** Point sources at each neuron location
- **Field coordinates:** Spatially expanded via convolution (spatial units arbitrary, normalized)
- **Kernel type:** Gaussian (default); width parameterized by distance in µm
- **Boundary handling:** Zero-padding (no boundary currents)

**Important:** Spatial units are **arbitrary normalized units**, not physical micrometers. The visualization uses µm for layer depth (anatomical reference), but field amplitudes are not calibrated.

---

## Manifest & Metadata Validation

### Run Manifest Template

```python
import json

RUN_METADATA = {
    "scope_status": "computational_scaffold",
    "readout_status": "simulated_proxy",
    "field_mode": "proxy_convolution_no_pde",
    "physical_amplitude_claim_allowed": False,
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

## Claim Gates & Interpretation

### The Gate: `physical_amplitude_claim_allowed`

This boolean key prevents misinterpretation:

```python
if not metadata["physical_amplitude_claim_allowed"]:
    # BLOCKED: Claiming real-world amplitude
    # "The LFP-proxy amplitude is 50 µV"
    
    # ALLOWED: Relative or tutorial statements
    # "LFP-proxy increases during high firing rate"
    # "Layer 5 sources dominate the field"
```

### v0.3.7 Limitations

- No biophysical compartments (soma, dendrite, axon)
- No temperature sensitivity, frequency-dependent effects
- No subject-specific anatomy
- No experimental validation
- Kernels are fixed defaults (not tunable in v0.3.7)

### Future Work (v0.3.8+)

- Custom convolution kernels via `.field_kernel()` method
- PDE-based field solvers (optional)
- Calibration to real neural recordings
- Frequency-response properties

---

## Summary & Next Steps

### What You've Learned

1. **Sources are implicit:** Emitter + probes determine field computation
2. **Fields are proxies:** Convolution-based, fast, approximate
3. **Readouts are multi-modal:** Different operators extract different field perspectives
4. **Metadata gates claims:** `physical_amplitude_claim_allowed=False` prevents misinterpretation

### How to Use This in Your Work

```python
# Step 1: Configure a column
cfg = jtfne.Configuration().set_emitter(...).probes([...])

# Step 2: Simulate
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, ...)

# Step 3: Check scope before interpreting
assert not signals.metadata["physical_amplitude_claim_allowed"]

# Step 4: Use relative comparisons, not absolute claims
layer5_rate = signals.spikes[layer5_idx].mean()
layer23_rate = signals.spikes[layer23_idx].mean()
print(f"L5 rate is {layer5_rate / layer23_rate:.1f}x L2/3 rate")  # ✓ OK

# Step 5: Document scope in your output
json.dump(signals.metadata, fp, allow_nan=False)
```

---

## References

- **v0.3.6 Tutorial:** [Configuration API & E/I Networks](./06_v036_100_neuron_ei_population.md)
- **API Reference:** [Signals Object](../api/signals.md) | [Probes & Readouts](../api/probes.md)
- **Interactive Visualization:** [3D Source/Field/Probe Column](../../assets/interactive/v037_source_column_3d.html)
- **GitHub Issue (Roadmap):** [Kernel Customization (v0.3.8)](https://github.com/HNXJ/jaxfne/issues/TODO)

---

**End of v0.3.7 Tutorial**

Feedback? Open an issue: [jaxfne/issues](https://github.com/HNXJ/jaxfne/issues)
