# v0.3.8: LFP/CSD-Like Readout Tutorial

**Version:** 0.3.8  
**Difficulty:** Intermediate  
**Duration:** 15–20 minutes to read; 5–10 minutes to execute  
**Scope:** Computational scaffold, simulated proxy fields, tutorial-scale learning

---

## Overview

This tutorial documents the jaxfne **source-to-field-to-readout workflow** for laminar contact arrays. It shows how neural source currents emerge implicitly from emitter + configuration, project spatially to laminar contacts via a Gaussian kernel, and extract LFP-proxy and CSD-proxy readouts.

The core concepts:

1. **Source Declaration (Implicit):** Emitter type + neuron count determine available sources
2. **Spatial Projection (Gaussian Kernel):** Sources spread to contacts via a fixed, row-normalized convolution kernel (not PDE-solved)
3. **LFP-proxy:** The spatially-smoothed source projection represents local field potential
4. **CSD-proxy:** The second spatial derivative of LFP-proxy approximates current-source density
5. **Probe Readout:** Eight multimodal operators extract spikes, voltage, sources, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, and EMM-proxy
6. **Scope Clarity:** Metadata gates (`physical_amplitude_claim_allowed=False`) prevent amplitude overclaims

This is a **computational scaffold**, not a biophysically validated model.

---

## Mathematical Framework

### Source Bookkeeping

$$S(t) \in \mathbb{R}^{T \times N}$$

**Worded equation:** Source activity is stored as a time-by-neuron matrix. Each entry S(t, n) represents the current produced by neuron n at time t.

### Projection to Laminar Contacts

$$Y(t, c) = \sum_{n=1}^{N} K(c, n) \cdot S(t, n)$$

where $K \in \mathbb{R}^{C \times N}$ is the Gaussian projection kernel.

**Worded equation:** Each contact receives a weighted sum of neural sources. The Gaussian kernel K(c, n) assigns higher weight to neurons near contact c and lower weight to distant neurons.

### Gaussian Kernel (Row-Normalized)

$$K(c, n) = \frac{\exp\left(-0.5 \left(\frac{d_c - d_n}{w}\right)^2\right)}{\sum_{n'=1}^{N} \exp\left(-0.5 \left(\frac{d_c - d_{n'}}{w}\right)^2\right)}$$

where $d_c$ is contact depth, $d_n$ is neuron depth, and $w = 0.10$ is the kernel width.

**Worded equation:** The kernel is a Gaussian centered at each contact's depth, with width controlled by w. Row normalization ensures each contact receives a properly weighted summary.

### CSD-like Readout (Second Spatial Derivative)

$$\text{CSD}(t, c) \approx -\frac{Y(t, c-1) - 2Y(t, c) + Y(t, c+1)}{\Delta z^2}$$

where $\Delta z = 1/(C-1)$ is the contact spacing.

**Worded equation:** CSD-proxy approximates local curvature of the field by taking the second difference across neighboring contacts. The negative sign follows electrostatic convention.

### Probe Readout

$$R_k(t) = Q_k(S(t), V(t), \text{spike count}, \ldots)$$

**Worded equation:** Each probe (k = spikes, V_m, source, LFP-proxy, CSD-proxy) extracts a different summary of the neural state.

---

## Configuration API & Workflow

### The Public API Contract

Sources are **not explicitly declared**. Instead, they are **inferred** from:

1. **Emitter type & preset:** Determines available sources (e.g., Izhikevich → intrinsic + synaptic currents)
2. **Probe modes:** Determines which sources are computed and returned

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

## Signals API Contract

After `jtfne.simulate()`, the returned `signals` object has:

```python
signals.spikes       # np.ndarray, shape (T, N), boolean spike indicator
signals.V_m          # np.ndarray, shape (T, N), membrane voltage
signals.sources      # np.ndarray, shape (T, N), source currents
signals.time_ms      # np.ndarray, shape (T,), time axis in milliseconds
signals.metadata     # dict, scope/readout metadata
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

**Critical key:** `physical_amplitude_claim_allowed=False` gates claims about real-world amplitude.

---

## Example 1: Single Neuron → Contacts

A single neuron in layer L2/3 projects to 16 evenly-spaced laminar contacts.

**Key observations:**

- **Source shape:** [T=10000, N=1] (time × neuron)
- **LFP-proxy shape:** [T=10000, C=16] (time × contacts)
- **CSD-proxy shape:** [T=10000, C=16] (time × contacts, second derivative)
- **Single source → distributed field:** The point source is smoothed by the Gaussian kernel, producing a smooth LFP-like profile across contacts
- **Nearest contacts receive highest amplitude:** Contacts near the neuron's depth receive stronger signal

---

## Example 2: E/I Laminar Column

A 48-neuron laminar column (12 neurons per layer, 4 layers) with mixed E/I composition.

**Configuration:**

- **Neurons:** 48 total = 4 layers × 12 neurons/layer
- **Cell types:** E (75%), PV (15%), SST (5%), VIP (5%)
- **Connectivity:** Recurrent laminar connectivity (within and across layers)
- **Duration:** 1000 ms with 0.1 ms timestep

**Key observations:**

- **Source shape:** [T=10000, N=48]
- **LFP-proxy shape:** [T=10000, C=16]
- **Emerges laminar structure:** Layer-wise E/I interactions produce distinct laminar profiles
- **CSD-proxy shows layer boundaries:** Second derivative reveals where sources concentrate
- **Population rate:** Typically 2–25 Hz (active-state regime for v0.3.8)

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

### Available Modes

| Mode | Shape | Description |
|------|-------|-------------|
| `spikes` | (T, N) | Spike detection (boolean) |
| `V_m` | (T, N) | Membrane voltage |
| `source` | (T, N) | Raw source currents from emitter |
| `LFP-proxy` | (T, C) | Local-field-potential proxy via Gaussian projection |
| `CSD-proxy` | (T, C) | Current-source-density proxy (second spatial derivative) |

### How Proxy Fields Are Computed

**Not PDE-solved.** Instead:

1. Extract source currents from neurons: $S(t)$ [T, N]
2. Apply fixed Gaussian kernel: $Y(t) = S @ K^T$ [T, C]
3. Optionally compute spatial derivatives (for CSD)

This is **fast** (no solver loop) but **approximate** (not validated against real data).

---

## Validation & JSON Safety

### Run Manifest Template

```python
import json

RUN_METADATA = {
    "scope_status": "computational_scaffold",
    "readout_status": "simulated_proxy",
    "field_mode": "proxy_laminar_gaussian_kernel",
    "physical_amplitude_claim_allowed": False,
    "duration_ms": 1000.0,
    "dt_ms": 0.1,
    "dtype": "float32",
    "seed": 42,
    "n_neurons": 48,
    "n_contacts": 16,
    "layers": ["L2/3", "L4", "L5", "L6"],
    "mean_population_rate_hz": 4.5,
    "source_shape": [10000, 48],
    "lfp_proxy_shape": [10000, 16],
    "csd_proxy_shape": [10000, 16],
    "finite_outputs": True,
    "equations": {
        "source_bookkeeping": "S(t) ∈ ℝ^{T×N}",
        "source_projection": "Y(t,c) = Σ_n K(c,n) · S(t,n)",
        "lfp_proxy": "lfp_proxy = Y, spatially-smoothed field",
        "csd_proxy": "csd_proxy ≈ -d²Y/dz², second spatial derivative",
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

## Interpretation & Claim Gates

### The Gate: `physical_amplitude_claim_allowed`

This boolean key prevents misinterpretation:

```python
if not metadata["physical_amplitude_claim_allowed"]:
    # BLOCKED: Claiming real-world amplitude
    # ✗ "The LFP-proxy amplitude is 50 µV"
    # ✗ "CSD-proxy indicates a sink at L5"
    
    # ALLOWED: Relative or tutorial statements
    # ✓ "LFP-proxy increases during high firing rate"
    # ✓ "Layer 5 sources dominate the field"
    # ✓ "The kernel width of 0.10 produces smoother estimates than 0.05"
```

### v0.3.8 Limitations

- No biophysical compartments (soma, dendrite, axon)
- No temperature sensitivity, frequency-dependent effects
- No subject-specific anatomy
- No experimental validation
- Kernels are fixed defaults (not tunable in v0.3.8; planned for v0.3.9)
- Amplitudes are uncalibrated (proxy-scale only)

### Future Work (v0.3.9+)

- Custom convolution kernels via `.field_kernel()` method
- PDE-based field solvers (optional)
- Calibration to real neural recordings
- Frequency-response properties

---

## Summary & Next Steps

### What You've Learned

1. **Implicit sources:** Emitter + probes determine field computation
2. **Spatial projection:** Gaussian kernels map point sources to contacts
3. **LFP/CSD computation:** Source projection + spatial derivatives
4. **Multimodal readouts:** Different operators extract different field perspectives
5. **Metadata gates:** `physical_amplitude_claim_allowed=False` prevents misinterpretation

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

- **v0.3.7 Tutorial:** [Source Bookkeeping](./07_v037_source_bookkeeping.md)
- **v0.3.6 Tutorial:** [Configuration API & E/I Networks](./06_v036_100_neuron_ei_population.md)
- **API Reference:** [API Overview](../api/index.md)
- **Guides:** [Probe Operators](../guides/probe_operators.md) | [Tensor-Field Workflows](../guides/tensor_field_workflows.md)
- **GitHub:** [jaxfne Issues](https://github.com/HNXJ/jaxfne/issues)

---

**End of v0.3.8 Tutorial**

Feedback? Open an issue: [jaxfne/issues](https://github.com/HNXJ/jaxfne/issues)
