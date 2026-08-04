# Tutorial Figures

**Status:** Stable contract, regenerated with visual confirmation; introduced in jaxfne 0.2.27/2026-05-22, still exercised by `tests/test_tutorial_figure_manifest_v028.py` (current development tree: 0.4.8).

---

## Overview

The current figure set includes complete regenerated tutorial PNG figures demonstrating the jaxfne forward-field workflow. All figures are:

- **Simulated:** Generated from `cortical_column` network with Izhikevich emitters
- **Proxy-safe:** No biological statements or solver status overstates
- **Deterministic:** Seeded (seed=0) for reproducibility
- **CPU-safe:** Generated using matplotlib Agg backend
- **JSON-validated:** Manifest is JSON-safe with no NaN/Inf

---

## Figure Gallery

### 1. Spike Raster (Behavioral)

**File:** `01_spike_raster.png`

Spike times across all 50 simulated units. Shows aggregate firing patterns over 500 ms simulation.

**Status:** Simulated proxy  
**Data source:** `signals.spikes` (50 units, 5000 time steps)

---

### 2. Voltage Traces (Membrane State)

**File:** `02_voltage_traces.png`

Izhikevich native membrane voltage for 6 representative units. Displays voltage dynamics (mV) over simulation.

**Status:** Izhikevich native (uncalibrated)  
**Data source:** `signals.V_m` (50 units, 5000 time steps)

---

### 3. Source Proxy Heatmap (Synaptic Current Model)

**File:** `03_source_proxy_heatmap.png`

Synaptic current model across all units. Represents the proxy source used for field computation.

**Status:** Synaptic current proxy (nA, uncalibrated)  
**Data source:** `signals.sources` (50 units, 5000 time steps)

---

### 4. LFP-Proxy Trace (Scalar Readout)

**File:** `04_lfp_proxy_trace.png`

Averaged laminar field potential proxy across all 16 recording contacts. Smoothed temporal dynamics.

**Status:** LFP proxy (no sensor calibration)  
**Data source:** `signals.field.lfp_proxy` (16 contacts, 5000 time steps)

---

### 5. CSD Proxy Heatmap (Spatial Derivative)

**File:** `05_csd_proxy_heatmap.png`

Current source density proxy derived from field gradient. Spatial map over contacts and time.

**Status:** Spatial proxy (no sink-source validation)  
**Data source:** `signals.field.csd_proxy` (16 contacts, 5000 time steps)

---

### 6. Extracellular Potential Proxy (φ_e Proxy)

**File:** `06_phi_e_proxy_heatmap.png`

Extracellular potential proxy (φ_e) across contacts. Laminar field solution from source.

**Status:** Field proxy (no boundary condition validation)  
**Data source:** `signals.field.phi_e_proxy` (16 contacts, 5000 time steps)

---

### 7. Source Proxy Spatial (Kernel-Weighted)

**File:** `07_source_proxy_spatial.png`

Kernel-weighted source projection into contact space. Shows how source contributes to each contact.

**Status:** Spatial projection proxy  
**Data source:** `signals.field.source_proxy` (16 contacts, 5000 time steps)

---

### 8. Conservation Proxy Diagnostics (Metrics)

**File:** `08_conservation_diagnostics.png`

Four key conservation proxy metrics:
- **L1 norm:** Sum of absolute source amplitudes
- **L2 norm:** RMS source amplitude
- **Field grad:** Field gradient L2 norm
- **Conserv. res.:** Conservation residual (absolute)

**Status:** Proxy diagnostics (no conservation guarantee)  
**Data source:** `manifest['conservation_proxy_diagnostics']`

---

### 9. Laminar Profile Depths (Geometry)

**File:** `09_laminar_profile_depths.png`

Contact depths (y-axis position proxy). Indicates laminar sampling geometry.

**Status:** Declared geometry (no anatomical calibration)  
**Data source:** `signals.field.contact_depths` (16 contacts)

---

### 10. Firing Rate Proxy (Smoothed Activity)

**File:** `10_firing_rate_raster.png`

Smoothed spike count (50-step window) across units and time. Population-level activity proxy.

**Status:** Spike-derived proxy (no metabolic interpretation)  
**Data source:** `signals.spikes` with temporal smoothing

---

### 11. Statement Gates Summary (Metadata)

**File:** `11_status_summary.png`

Text summary of all frozen status checks and status status:
- `run_status`: tutorial_scaffold
- `model_status`: computational_scaffold
- `field_solver_status`: linear_solver
- `amplitude_status`: False
- `source_calibration_status`: uncalibrated_izhikevich_native_current
- `metabolism_status`: False

**Status:** Metadata placeholder (no real data)  
**Uses real data:** False

---

### 12. Spectral Summary (Network Activity FFT)

**File:** `12_spectral_summary.png`

Power spectral density of mean network spike activity. Log-scale frequency domain representation.

**Status:** Signal processing proxy (no neural oscillation statements)  
**Data source:** `signals.spikes` with FFT

---

## Regeneration Command

To regenerate all figures:

```bash
python scripts/generate_tutorial_figures.py
```

Output directory: `docs/_static/tutorial_figures/`

Manifest file: `docs/_static/tutorial_figures/figure_manifest.json`

---

## Manifest Schema

Each figure has:

```json
{
  "filename": "01_spike_raster.png",
  "title": "Spike Raster",
  "type": "behavioral",
  "uses_real_data": true,
  "path": "docs/_static/tutorial_figures/01_spike_raster.png",
  "visually_confirmed": true,
  "visual_status": "pass",
  "readout_status": "simulated_proxy"
}
```

Global manifest fields:

```json
{
  "figure_count": 12,
  "real_data_figure_count": 11,
  "min_required": 10,
  "jaxfne_version": "0.3.4",
  "run_status": "tutorial_scaffold",
  "model_status": "computational_scaffold",
  "field_solver_status": "linear_solver",
  "amplitude_status": false,
  "metabolism_status": false,
  "source_script": "scripts/generate_tutorial_figures.py",
  "visual_confirmation_method": "manual_inspection_and_image_nonblank_check"
}
```

---

## Status Status

All figures are:
- **Exploratory:** Teaching artifacts, not biological validation
- **Proxy-only:** Proxy-based field projection; elliptic and volumetric field solvers are reserved regimes ([Limitations and future plans](limitations_and_future_plans.md))
- **Izhikevich native:** Phenomenological neuron model (uncalibrated current units)
- **Laminar proxy:** Forward-field model for demonstrating source-to-field mapping
- **Proxy framing:** figures report proxy readouts in relative units

---

## Related Documentation

- **[Index](index.md)** — Documentation index
- **[Computation Basis](computation_basis.md)** — Computation contract
- **[Conservation Proxy Diagnostics](conservation_proxy_diagnostics.md)** — Conservation diagnostic framework

---

## See Also

- **scripts/generate_tutorial_figures.py** — Figure generation script
- **tests/test_tutorial_figure_manifest_v028.py** — Manifest validation tests
- **docs/_static/tutorial_figures/figure_manifest.json** — Generated manifest
