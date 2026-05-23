# Tutorial Figures (v0.2.28)

**Status:** Regenerated with visual confirmation  
**Version:** jaxfne 0.2.27  
**Date:** 2026-05-22

---

## Overview

v0.2.28 includes a complete set of regenerated tutorial PNG figures demonstrating the jaxfne forward-field workflow. All figures are:

- **Simulated:** Generated from `cortical_column` network with Izhikevich emitters
- **Proxy-safe:** No biological claims or solver status overclaims
- **Deterministic:** Seeded (seed=0) for reproducibility
- **CPU-safe:** Generated using matplotlib Agg backend
- **JSON-validated:** Manifest is JSON-safe with no NaN/Inf

---

## Figure Gallery

### 1. Spike Raster (Behavioral)

**File:** `01_spike_raster.png`

Spike times across all 50 simulated units. Shows aggregate firing patterns over 500 ms simulation.

**Claim status:** Simulated proxy  
**Data source:** `signals.spikes` (50 units, 5000 time steps)

---

### 2. Voltage Traces (Membrane State)

**File:** `02_voltage_traces.png`

Izhikevich native membrane voltage for 6 representative units. Displays voltage dynamics (mV) over simulation.

**Claim status:** Izhikevich native (uncalibrated)  
**Data source:** `signals.V_m` (50 units, 5000 time steps)

---

### 3. Source Proxy Heatmap (Synaptic Current Model)

**File:** `03_source_proxy_heatmap.png`

Synaptic current model across all units. Represents the proxy source used for field computation.

**Claim status:** Synaptic current proxy (nA, uncalibrated)  
**Data source:** `signals.sources` (50 units, 5000 time steps)

---

### 4. LFP-Like Proxy Trace (Scalar Readout)

**File:** `04_lfp_proxy_trace.png`

Averaged laminar field potential proxy across all 16 recording contacts. Smoothed temporal dynamics.

**Claim status:** LFP proxy (no sensor calibration)  
**Data source:** `signals.field.lfp_proxy` (16 contacts, 5000 time steps)

---

### 5. CSD Proxy Heatmap (Spatial Derivative)

**File:** `05_csd_proxy_heatmap.png`

Current source density proxy derived from field gradient. Spatial map over contacts and time.

**Claim status:** Spatial proxy (no sink-source validation)  
**Data source:** `signals.field.csd_proxy` (16 contacts, 5000 time steps)

---

### 6. Extracellular Potential Proxy (φ_e Proxy)

**File:** `06_phi_e_proxy_heatmap.png`

Extracellular potential proxy (φ_e) across contacts. Laminar field solution from source.

**Claim status:** Field proxy (no boundary condition validation)  
**Data source:** `signals.field.phi_e_proxy` (16 contacts, 5000 time steps)

---

### 7. Source Proxy Spatial (Kernel-Weighted)

**File:** `07_source_proxy_spatial.png`

Kernel-weighted source projection into contact space. Shows how source contributes to each contact.

**Claim status:** Spatial projection proxy  
**Data source:** `signals.field.source_proxy` (16 contacts, 5000 time steps)

---

### 8. Conservation Proxy Diagnostics (Metrics)

**File:** `08_conservation_diagnostics.png`

Four key conservation proxy metrics:
- **L1 norm:** Sum of absolute source amplitudes
- **L2 norm:** RMS source amplitude
- **Field grad:** Field gradient L2 norm
- **Conserv. res.:** Conservation residual (absolute)

**Claim status:** Proxy diagnostics (no conservation guarantee)  
**Data source:** `manifest['conservation_proxy_diagnostics']`

---

### 9. Laminar Profile Depths (Geometry)

**File:** `09_laminar_profile_depths.png`

Contact depths (y-axis position proxy). Indicates laminar sampling geometry.

**Claim status:** Declared geometry (no anatomical calibration)  
**Data source:** `signals.field.contact_depths` (16 contacts)

---

### 10. Firing Rate Proxy (Smoothed Activity)

**File:** `10_firing_rate_raster.png`

Smoothed spike count (50-step window) across units and time. Population-level activity proxy.

**Claim status:** Spike-derived proxy (no metabolic interpretation)  
**Data source:** `signals.spikes` with temporal smoothing

---

### 11. Claim Gates Summary (Metadata)

**File:** `11_claim_gates_summary.png`

Text summary of all frozen claim gates and truth status:
- `truth_mode`: truth_safe_unverified
- `claim_level`: computational_scaffold
- `field_solver_status`: laminar_proxy_no_pde
- `physical_amplitude_claim_allowed`: False
- `source_calibration_status`: uncalibrated_izhikevich_native_current
- `biological_metabolism_claim_allowed`: False

**Claim status:** Metadata placeholder (no real data)  
**Uses real data:** False

---

### 12. Spectral Summary (Network Activity FFT)

**File:** `12_spectral_summary.png`

Power spectral density of mean network spike activity. Log-scale frequency domain representation.

**Claim status:** Signal processing proxy (no neural oscillation claims)  
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
  "claim_status": "simulated_proxy"
}
```

Global manifest fields:

```json
{
  "figure_count": 12,
  "real_data_figure_count": 11,
  "min_required": 10,
  "jaxfne_version": "0.2.27",
  "truth_mode": "truth_safe_unverified",
  "claim_level": "computational_scaffold",
  "field_solver_status": "laminar_proxy_no_pde",
  "physical_amplitude_claim_allowed": false,
  "biological_metabolism_claim_allowed": false,
  "source_script": "scripts/generate_tutorial_figures.py",
  "visual_confirmation_method": "manual_inspection_and_image_nonblank_check"
}
```

---

## Truth Status

All figures are:
- **Exploratory:** Teaching artifacts, not biological validation
- **Proxy-only:** No Poisson solver, no Maxwell solver, no field PDE solution
- **Izhikevich native:** Phenomenological neuron model (uncalibrated current units)
- **Laminar proxy:** Forward-field model for demonstrating source-to-field mapping
- **No overclaims:** No "real EEG", "validated CSD", "biological metabolism", or solver status assertions

---

## Related Documentation

- **[docs/index.md](index.md)** — Documentation index
- **[docs/computation_basis.md](computation_basis.md)** — Computation contract (v0.2.27 basis)
- **[docs/conservation_proxy_diagnostics.md](conservation_proxy_diagnostics.md)** — Conservation diagnostic framework
- **[README.md](../README.md)** — Quick start and installation

---

## See Also

- **scripts/generate_tutorial_figures.py** — Figure generation script
- **tests/test_tutorial_figure_manifest_v028.py** — Manifest validation tests
- **docs/_static/tutorial_figures/figure_manifest.json** — Generated manifest
