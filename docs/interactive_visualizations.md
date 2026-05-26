# Interactive Tutorial Visualizations (v0.2.21)

**Status:** v0.2.21 Optional Interactive HTML Layer  
**Date:** 2026-05-21  
**truth_mode:** truth_safe_unverified  

---

## Overview

v0.2.21 adds optional interactive Plotly HTML visualizations for tutorial outputs, generated from source simulation data. Static PNG figures remain the default and unchanged. Interactive HTML is opt-in, Plotly is optional, and all original claim gates remain frozen.

**Key principle:** Interactive visualizations are derived from the same source data as static PNG figures. No separate data sources, no additional claims, no biological validation added.

---

## Installation

Interactive visualizations require Plotly (optional [viz] extra dependency):

```bash
pip install -e '.[viz]'
```

Without Plotly, all tutorials and validators run normally. Plotly import failures are gracefully handled with clear status messages.

---

## Running Tutorials with Interactive Visualization

### Generate Static PNG Only (Default)

```bash
python scripts/run_all_tutorials.py --write-figures --out-root outputs/
```

Output: manifest.json, probe_report.json, metrics.json, validation_report.json, asset_hashes.json, figures/raster.png (or spectrolaminar_profile.png).

### Generate Static PNG + Interactive HTML

```bash
python scripts/run_all_tutorials.py --write-figures --write-interactive --out-root outputs/
```

Output: All static files plus figures/raster.html or figures/spectrolaminar_profile.html.

**Note:** For routine validation (CI, local development), use `--smoke` mode to reduce runtime:

```bash
python scripts/run_all_tutorials.py --smoke --write-figures --write-interactive --out-root outputs/test_interactive
```

Full interactive generation is recommended for release validation and manual testing.

---

## Validating Tutorial Outputs

### Validate Static PNG Only (Default)

```bash
python scripts/validate_tutorial_outputs.py outputs/
```

Checks: manifest.json, probe_report.json, metrics.json, validation_report.json, asset_hashes.json, figures/*.png, source_data.json.

Claim gates verified: physical_amplitude_claim_allowed=False, claim_level="computational_scaffold", field_claim_level="proxy_readout_only".

### Validate Static PNG + Interactive HTML

```bash
python scripts/validate_tutorial_outputs.py outputs/ --require-interactive
```

Adds validation:
- figures/raster.html or figures/spectrolaminar_profile.html exists
- File size > 0 bytes
- SHA256 hash matches asset_hashes.json
- Clear error message on mismatch

---

## Tutorial-Specific Artifacts

### Tutorial 03: Single-Neuron Multimodal Probe

**Static PNG:** `figures/raster.png` (~13 KB)  
**Interactive HTML:** `figures/raster.html` (~25-30 KB)

**Data source:**
- Source data: `figures/source_data.json` containing spike events (time_ms, unit_id)
- Figure: Interactive scatter plot of spike times vs unit ID

**Metadata in source_data.json:**
```json
{
  "source_data_kind": "spike_events",
  "tutorial_id": "03_single_neuron_multimodal_probe",
  "figure_id": "raster",
  "time_ms": [...],
  "unit_id": [...],
  "units_or_status": "binary_spike_event_proxy",
  "operator_kind": "spk",
  "claim_level": "computational_scaffold",
  "physical_amplitude_claim_allowed": false
}
```

### Tutorial 04: Two-Neuron E/I Multimodal

**Static PNG:** `figures/raster.png` (~14 KB)  
**Interactive HTML:** `figures/raster.html` (~25-30 KB)

**Data source:**
- Source data: `figures/source_data.json` containing spike events (E and I neurons, time_ms, unit_id)
- Figure: Interactive scatter plot with E/I coloring preserved

### Tutorial 05: 100-Neuron E/I Multimodal

**Static PNG:** `figures/raster.png` (~18 KB)  
**Interactive HTML:** `figures/raster.html` (~30-40 KB)

**Data source:**
- Source data: `figures/source_data.json` containing all spike events
- Figure: Interactive scatter plot with E/I boundary marked

### Tutorial 02: Spectrolaminar Oddball Scaffold

**Static PNG:** `figures/spectrolaminar_profile.png` (~29 KB)  
**Interactive HTML:** `figures/spectrolaminar_profile.html` (~25-35 KB)

**Data source:**
- Source data: `figures/source_data.json` containing spectrolaminar profile arrays
- Figure: Interactive grouped bar chart (alpha/beta and gamma power across windows)

**Metadata in source_data.json:**
```json
{
  "source_data_kind": "spectrolaminar_profile",
  "tutorial_id": "02_spectrolaminar_oddball_scaffold",
  "figure_id": "spectrolaminar_profile",
  "layers_or_depths": ["baseline", "event", "post", "full_peri_event"],
  "alpha_beta_profile": [...],
  "gamma_profile": [...],
  "units_or_status": "relative_proxy_units",
  "operator_kind": "spectrolaminar_profile",
  "claim_level": "computational_scaffold",
  "physical_amplitude_claim_allowed": false
}
```

---

## Asset Hashing

Every tutorial's `asset_hashes.json` includes:

1. **Contract JSON files** (manifest, probe_report, metrics, validation_report)
   - Format: `"filename": "sha256_hash_string"`

2. **Static PNG figures**
   - Format: `"figures/raster.png": "sha256_hash_string"`

3. **Source data JSON**
   - Format: `"figures/source_data.json": "sha256_hash_string"`

4. **Interactive HTML** (if generated)
   - Format: `"figures/raster.html": "sha256_hash_string"` (or spectrolaminar_profile.html)

**Validation:** All hashes are recomputed during validation and compared against recorded values. Mismatch indicates stale, corrupted, or manually modified artifacts.

---

## Claim Discipline

All interactive visualizations respect and propagate the immutable claim gates:

- `physical_amplitude_claim_allowed`: Always **False** (no amplitude calibration)
- `claim_level`: Always **"computational_scaffold"** (no biological claims)
- `field_claim_level`: Always **"proxy_readout_only"** (simulated fields, not real)

**What interactive HTML shows:**
- Mathematical relationships between simulation variables (spike times, field power)
- Deterministic visualization of source data arrays
- No empirical validation, no mechanism claims, no biological truth

**What interactive HTML does NOT show:**
- Biological plausibility scores
- Calibration status
- Whole-brain capability
- Empirical validation results

---

## Troubleshooting

### "Plotly not installed"
```
status: skipped
reason: Plotly not installed (optional dependency)
```

**Fix:** `pip install plotly` or `pip install -e '.[viz]'`

### "Interactive HTML missing" (with --require-interactive)
```
ValueError: Interactive HTML missing: raster.html
```

**Cause:** HTML not generated (Plotly absent or generation failed)  
**Fix:** Install Plotly and re-run runner with `--write-interactive`

### "Interactive HTML hash mismatch"
```
ValueError: Interactive HTML hash mismatch for raster.html: computed=... vs recorded=...
```

**Cause:** HTML file manually edited or source data changed  
**Fix:** Regenerate with runner (`python scripts/run_all_tutorials.py`)

---

## CI Integration

**Recommended CI command (smoke mode):**

```bash
python scripts/run_all_tutorials.py --smoke --write-figures --out-root outputs/
python scripts/validate_tutorial_outputs.py outputs/
```

This validates static PNG without interactive generation overhead. Full interactive validation is manual/release-only.

**Do NOT add `--write-interactive` to default CI** to avoid long-running tutorial simulations.

---

## Related Documentation

- [tutorial_outputs.md](tutorials/tutorial_outputs.md) — Output contract, static figure patterns
- [probe_operators.md](probe_operators.md) — 8-operator specification
- [tutorials/index.md](tutorials/index.md) — Tutorial progression

---

**Truth Status:** Interactive visualizations are computational artifacts, not empirically validated evidence. All claim gates remain frozen.

**Version:** v0.2.21 (optional layer, static PNG default preserved)
