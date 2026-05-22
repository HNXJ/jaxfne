# Tutorial Output Contract

**Status:** v0.2.19 Specification  
**Date:** 2026-05-21  
**truth_mode:** truth_safe_unverified

---

## Overview

The 4 core jaxfne example scripts generate self-contained evidence bundles. Each tutorial produces:

1. **JSON metadata files** — full pipeline information, claim gates, validation status
2. **Static figures** — deterministic PNG visualizations generated from simulation data
3. **Hashed artifacts** — SHA256 integrity verification for all outputs

This contract ensures tutorials are reproducible, verifiable, and scientifically honest about their scope and limitations.

**Key principle:** All outputs are **computational evidence**, not biological proof. No empirical validation. No mechanism claims. Claim gates are frozen: `physical_amplitude_claim_allowed=False`.

---

## Output Schema

Each tutorial creates an output directory with 5 required JSON files plus optional `figures/` subdirectory.

### 1. manifest.json

**Purpose:** Full pipeline metadata and model configuration.

**Key fields:**
```json
{
  "claim_level": "computational_scaffold",
  "physical_amplitude_claim_allowed": false,
  "field_claim_level": "proxy_readout_only",
  "truth_mode": "truth_safe_unverified",
  "model_config": { ... },
  "simulation_config": { ... },
  "probe_operators": [ "spk", "vm", "source", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy", "emm_proxy" ]
}
```

**Constraints:**
- `physical_amplitude_claim_allowed` is **always False** in v0.2.19
- `claim_level` must be `"computational_scaffold"`
- `field_claim_level` must be `"proxy_readout_only"`
- All values must be JSON-safe (no NaN/Inf)

### 2. probe_report.json

**Purpose:** All 8 multimodal operator reports.

**Structure:**
```json
{
  "spk": { "operator_status": "ok", ... },
  "vm": { "operator_status": "ok", ... },
  "source": { "operator_status": "ok", ... },
  "lfp_proxy": { "operator_status": "ok", ... },
  "csd_proxy": { "operator_status": "ok", ... },
  "eeg_proxy": { "operator_status": "ok", ... },
  "meg_proxy": { "operator_status": "ok", ... },
  "emm_proxy": { "operator_status": "ok", ... }
}
```

**Constraints:**
- All 8 operators must be present (even if status is "unavailable")
- Each operator must have `operator_status` field
- No NaN/Inf in any numeric fields
- All proxy operators are **simulated, not empirically validated**

### 3. metrics.json

**Purpose:** Basic signal statistics from simulation.

**Example:**
```json
{
  "spike_rate_per_timestep": 0.042,
  "Vm_mean_mV": -65.3,
  "Vm_std_mV": 12.1
}
```

**Constraints:**
- All values must be finite numbers (no NaN/Inf)
- Spike rate may be 0 (silent neurons are allowed)
- Voltage statistics should reflect neuronal dynamics
- No overclaiming of biological accuracy

### 4. validation_report.json

**Purpose:** Scope and claim-status metadata audit.

**Key fields:**
```json
{
  "claim_level": "computational_scaffold",
  "field_claim_level": "proxy_readout_only",
  "physical_amplitude_claim_allowed": false,
  "empirical_validation_status": null,
  "mechanism_claim_status": null
}
```

**Constraints:**
- Must mirror claim gates from manifest.json
- `empirical_validation_status` is null (no empirical validation in v0.2.19)
- `mechanism_claim_status` is null (no biological mechanism claims)
- Explicitly declares scope limits

### 5. asset_hashes.json

**Purpose:** SHA256 integrity verification for all outputs.

**Example:**
```json
{
  "manifest.json": "74631213f7ad454d352b19b23a41654a1ea124a861115b8886a63979fcf2c1b1",
  "probe_report.json": "b23c1bc3e4ead5b616a78dd993d7284d95a9279009f0da6b7f53daae80eca081",
  "figures/raster.png": {
    "sha256": "ce926db20b38690b63518c1b656e5e530b5b91cf694e4f8923020d774dd8a3c8",
    "bytes": 13142
  }
}
```

**Constraints:**
- All JSON files must have SHA256 entries
- All figures must have SHA256 entries
- Validator recomputes hashes and compares against recorded values
- Zero-byte figures are rejected
- Hash mismatch indicates stale or corrupted artifacts

---

## Figure Generation

Each tutorial generates static PNG figures from simulation data. Figures are **deterministic** (same seed → same figure) and **hashed** (integrity verified).

### Figure Generation Pattern

```python
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Generate figure from simulation data
    fig, ax = plt.subplots(...)
    # ... plot code ...
    fig.savefig(output_dir / "figures" / "figure_name.png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    
except ImportError:
    pass  # matplotlib not available, skip figure generation
```

**Key points:**
- matplotlib is optional [viz] extra dependency
- Non-interactive Agg backend for headless/CI environments
- Figures generated from real simulation data, not manually pre-created
- Figures are hashed and integrity verified

### Tutorial-Specific Figures

#### 03_single_neuron_multimodal_probe
- **Figure:** `figures/raster.png` (~13 KB)
- **Content:** Single-neuron spike raster
- **Data source:** `signals.spikes` (shape: 1 neuron × 1000 timesteps)
- **Visualization:** vlines at spike times, y-axis shows neuron index

#### 04_two_neuron_ei_multimodal
- **Figure:** `figures/raster.png` (~14 KB)
- **Content:** Two-neuron E/I spike raster
- **Data source:** `signals.spikes` (shape: 2 neurons × 1000 timesteps)
- **Visualization:** vlines colored by neuron type (E=blue, I=red)

#### 05_network_100_ei_multimodal
- **Figure:** `figures/raster.png` (~18 KB)
- **Content:** 100-neuron E/I population spike raster
- **Data source:** `signals.spikes` (shape: 100 neurons × 1000 timesteps)
- **Visualization:** vlines colored by cell type, dashed line marks E/I boundary

#### 02_spectrolaminar_oddball_scaffold
- **Figure:** `figures/spectrolaminar_profile.png` (~29 KB)
- **Content:** 2-subplot spectrolaminar profile
- **Data source:** `metrics.json` windows (baseline, event, post, full_peri_event)
- **Visualization:** Bar charts of alpha/beta power (left) and gamma power (right)

---

## Validation Gates

The validation script enforces strict gates on all tutorial outputs. **Any gate failure causes hard stop (exit code 1) with explicit error message.**

### Hard Gates

1. **Directory exists:** Output directory must exist
2. **Contract files present:** All 5 JSON files must exist
3. **JSON structure valid:** Each JSON must parse without errors
4. **JSON is safe:** No NaN/Infinity values allowed
5. **Figures directory exists:** `figures/` subdirectory must exist
6. **Figures present:** All expected PNG files must exist
7. **Figures nonzero:** All figure files must have size > 0 bytes
8. **Hash present:** Each figure must have SHA256 entry in asset_hashes.json
9. **Hash matches:** Recomputed SHA256 must match recorded hash exactly
10. **Claim gates frozen:**
    - `physical_amplitude_claim_allowed` = False
    - `claim_level` = "computational_scaffold"
    - `field_claim_level` = "proxy_readout_only"
11. **Probe report complete:** All 8 operators present
12. **Metrics nonzero:** Signal statistics are finite and present

### Validation Example

```bash
$ python scripts/validate_tutorial_outputs.py outputs/

=== Tutorial Output Validation Summary ===
Output root: outputs
Status: valid
Tutorials validated: 4/4

✓ 03_single_neuron_multimodal_probe: files=5, figures=1
✓ 04_two_neuron_ei_multimodal: files=5, figures=1
✓ 05_network_100_ei_multimodal: files=5, figures=1
✓ 02_spectrolaminar_oddball_scaffold: files=5, figures=1

✓ All tutorial outputs validated successfully.
```

### Failure Example

```bash
$ rm outputs/v023_single_neuron_multimodal/figures/raster.png
$ python scripts/validate_tutorial_outputs.py outputs/

❌ 03_single_neuron_multimodal_probe: Missing figure: outputs/v023_single_neuron_multimodal/figures/raster.png

=== Tutorial Output Validation Summary ===
Output root: outputs
Status: invalid
Tutorials validated: 3/4
```

---

## Truth Status and Scope

**Claim Level:** `computational_scaffold`

- **What this means:** Tutorials demonstrate the jaxfne pipeline architecture
- **What this does NOT mean:**
  - No empirical validation against electrophysiology data
  - No biological mechanism claims
  - No claims of physical amplitude accuracy
  - No calibration to biological parameters
  - No whole-brain simulation capability

**Proxy-Field Language:**
- All field readouts (LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy) are **simulated** proxies
- No empirical measurement data
- Mathematical forward model only
- For validation and conceptual demonstration only

**Frozen Gates:**
- `physical_amplitude_claim_allowed = False` (hardcoded invariant)
- `empirical_validation_status = null` (no empirical validation)
- `mechanism_claim_status = null` (no biological mechanism claims)

---

## Running Tutorials Locally

### Generate All Tutorial Outputs

```bash
cd /path/to/jaxfne
python scripts/run_all_tutorials.py --out-root outputs/
```

**Flags:**
- `--out-root OUTPUT_DIR` — Output directory (default: `outputs/`)
- `--smoke` — Reduced runtime (not yet fully implemented; defer to future version)
- `--write-figures` — Enable figure generation (default: True)

**Output:**
- All 4 tutorials executed
- JSON outputs and figures generated
- JSON report printed to stdout
- Exit code 0 if all successful, 1 if any failure

### Validate Existing Outputs

```bash
python scripts/validate_tutorial_outputs.py outputs/
```

**Input:**
- Root directory containing tutorial output directories
- Example: `outputs/` (contains `v023_single_neuron_multimodal/`, `v029_two_neuron_ei_multimodal/`, etc.)

**Output:**
- Per-tutorial validation results
- JSON report printed to stdout
- Exit code 0 if all valid, 1 if any gate fails

---

## Implementation Status

**v0.2.19:**
- ✓ All 4 tutorials generate figures
- ✓ Figure hashes in asset_hashes.json
- ✓ Test coverage for output contract
- ✓ Claim gates frozen

**v0.2.20 (in development):**
- ✓ Automated runner script (scripts/run_all_tutorials.py)
- ✓ Independent validator script (scripts/validate_tutorial_outputs.py)
- ⧵ CI integration (smoke mode with realistic runtime)
- ⧵ Plotly interactive figures (deferred to v0.2.21)

---

## Related Documentation

- [output_bundles.md](output_bundles.md) — General output bundle structure
- [probe_operators.md](../probe_operators.md) — 8-operator contract and specifications
- [tutorials/index.md](index.md) — Tutorial progression and links
- [GitHub jaxfne releases](https://github.com/HNXJ/jaxfne/releases) — Version history and downloads
