# v0.3 Plotly and Interactive Figure Policy

**Version:** v0.3.0+  
**Last updated:** 2026-05-23  
**truth_mode:** truth_safe_unverified  

---

## Overview

All v0.3 tutorials generate **required PNG figures** for reproducibility and archival. **Interactive Plotly HTML figures are optional enhancements**, provided:
1. Plotly is available (guarded import with try/except)
2. HTML is generated from source data, not PNG conversion
3. Both PNG and HTML have SHA256 hashes recorded

**Default:** PNG figures only. Plotly is encouraged but not mandatory.

---

## PNG Figures (REQUIRED)

### Generation Policy

All v0.3 tutorials must generate at least 2 PNG figures:

**Figure types:**
- Spike raster (neurons × time, binary or marker plot)
- Voltage trace (sample neurons V_m time series)
- Field proxy heatmap (space × time, CSD or LFP if applicable)
- Firing rate dynamics (sliding window or instantaneous)
- Frequency spectrum (PSD, FFT)
- Other relevant domain figures (phase plane, current traces, etc.)

### PNG Specifications

**Technical requirements:**
- Format: PNG (uncompressed or PNG-8)
- Resolution: dpi=150 (minimum; 300 dpi recommended for publication)
- Color depth: RGB or RGBA (allow transparency for overlays)
- File size: > 1 KB (indicates real content, not blank image)
- Naming: `v030_XX_description.png` where XX = scenario number (zero-padded)

**Example filenames:**
```
v030_01_izhikevich_spike_raster.png
v030_01_izhikevich_voltage_trace.png
v030_08_csd_lfp_heatmap.png
v030_15_integrated_network_benchmark.png
```

### SHA256 Integrity

Every PNG must have its SHA256 hash computed and recorded:

```python
import hashlib

def compute_file_hash(filename):
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

# Compute hash for each PNG
artifact_hashes = {}
for png_file in png_files:
    artifact_hashes[png_file] = compute_file_hash(f'outputs/v030_XX/{png_file}')

# Save to JSON for validation
import json
with open('outputs/v030_XX/artifacts.json', 'w') as f:
    json.dump({'figures': artifact_hashes}, f, indent=2)
```

**Validation workflow:**
1. Generate PNG from simulation data
2. Compute SHA256 immediately after saving
3. Record in `artifacts.json`
4. On subsequent runs, recompute hash and compare
5. Hash mismatch = figure regenerated (or corrupted)

---

## Plotly Figures (OPTIONAL)

### When to Use Plotly

**Use Plotly if:**
- Interactive exploration is valuable (hover-over details, zoom, pan)
- Multiple subplots or 3D visualization helps interpretation
- Plotly is already a dependency in the user's environment

**Do NOT use Plotly if:**
- Plotly import fails (users without plotly should not be blocked)
- PNG alone suffices for understanding
- Interactivity adds no value over static figures

### Guarded Import Pattern

```python
# Attempt to import Plotly; continue without it if unavailable
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

# Later: only generate HTML if Plotly available
if PLOTLY_AVAILABLE:
    fig = go.Figure()
    # ... build figure ...
    fig.write_html('outputs/v030_XX_interactive_raster.html')
else:
    print("✓ Plotly not available; PNG figures are sufficient")
```

### Generation from Source Data

**Critical rule:** Generate HTML from source data, NOT by converting PNG.

**Correct:**
```python
# Data-driven generation
spike_times = [...] # Extracted from simulation
neurons = [0, 1, 2, ...]

# Create Plotly figure directly from spike_times
fig = go.Figure()
for neuron_id in neurons:
    spike_times_i = spike_times[neuron_id]
    for t in spike_times_i:
        fig.add_vline(x=t, line_dash="dash", line_color="black")
fig.write_html('outputs/v030_XX_spike_raster_interactive.html')
```

**Incorrect (DO NOT DO THIS):**
```python
# ❌ Converting PNG to HTML (loses data fidelity)
from PIL import Image
png_img = Image.open('outputs/v030_XX_raster.png')
# ... some conversion logic ...
# This is discouraged; use source data instead
```

### HTML Specifications

**Technical requirements:**
- Format: HTML5 (Plotly.js embedded)
- Self-contained (no external CDN dependencies for offline use)
- File size: 100 KB–5 MB typical (depends on data size)
- Naming: `v030_XX_description_interactive.html` (same XX and description as PNG)

**Recommended Plotly figure types:**
- Raster plot with hover (neuron ID, spike time)
- Voltage trace with range slider
- CSD/LFP heatmap with colorbar and hover
- Spectral power with frequency selection
- Parameter space Pareto front (3D scatter)

### Plotly with SHA256

Compute hashes for HTML the same way as PNG:

```python
html_hash = compute_file_hash('outputs/v030_XX_description_interactive.html')
artifact_hashes['description_interactive.html'] = html_hash

# Save both PNG and HTML hashes together
{
  "figures": {
    "v030_XX_description.png": "sha256_value_1",
    "v030_XX_description_interactive.html": "sha256_value_2"
  }
}
```

---

## Artifact Metadata JSON

All figures (PNG + HTML) must be registered in an `artifacts.json` file:

```json
{
  "metadata": {
    "scenario": "v0.3.01",
    "title": "Single Neuron I — Izhikevich Phenomenology",
    "generated_at": "2026-05-23T12:34:56Z",
    "generator": "jaxfne v0.2.30 + matplotlib 3.x + plotly 5.x"
  },
  "figures": {
    "v030_01_spike_raster.png": {
      "sha256": "abc123def456...",
      "bytes": 15234,
      "type": "png",
      "description": "Spike raster for 10 neurons over 100 ms",
      "interactive": false
    },
    "v030_01_voltage_trace.png": {
      "sha256": "ghi789jkl012...",
      "bytes": 18456,
      "type": "png",
      "description": "Membrane voltage for 3 sample neurons",
      "interactive": false
    },
    "v030_01_spike_raster_interactive.html": {
      "sha256": "mno345pqr678...",
      "bytes": 234567,
      "type": "html",
      "description": "Interactive spike raster with hover details",
      "interactive": true,
      "source_data": ["spike_times", "neuron_ids"]
    }
  }
}
```

### Artifact JSON Fields

| Field | Required? | Value |
|-------|-----------|-------|
| `scenario` | Yes | v0.3.XX identifier |
| `title` | Yes | Human-readable scenario title |
| `generated_at` | Yes | ISO 8601 timestamp |
| `generator` | No | Software versions (matplotlib, plotly) |
| `figures[name].sha256` | Yes | 64-char SHA256 hex string |
| `figures[name].bytes` | Yes | File size in bytes |
| `figures[name].type` | Yes | "png" or "html" |
| `figures[name].description` | Yes | 1-2 sentence figure description |
| `figures[name].interactive` | Yes | Boolean (PNG=false, HTML=true) |
| `figures[name].source_data` | If HTML | List of data fields used to generate figure |

---

## Validation and Reproducibility

### Validation workflow

1. **During tutorial execution:**
   - Generate PNG figures with matplotlib (always)
   - Try to generate Plotly HTML (guarded, optional)
   - Compute SHA256 for each file
   - Save metadata to `artifacts.json`

2. **Post-generation checks:**
   ```python
   import json, hashlib, os
   
   # Load metadata
   with open('outputs/v030_XX/artifacts.json') as f:
       metadata = json.load(f)
   
   # Validate each figure
   for fig_name, fig_info in metadata['figures'].items():
       filepath = f"outputs/v030_XX/{fig_name}"
       assert os.path.exists(filepath), f"File {fig_name} not found"
       
       # Recompute hash
       with open(filepath, 'rb') as f:
           actual_hash = hashlib.sha256(f.read()).hexdigest()
       
       # Compare
       recorded_hash = fig_info['sha256']
       assert actual_hash == recorded_hash, \
           f"Hash mismatch for {fig_name}: {actual_hash} != {recorded_hash}"
       
       print(f"✓ {fig_name}: {recorded_hash}")
   ```

3. **CI/CD validation:**
   - Run validation script in GitHub Actions
   - Fail build if any figure hash mismatches
   - Archive PNG/HTML artifacts (optional)

### Reproducibility guarantees

- Same Python/JAX/matplotlib versions + same random seed = same PNG byte-for-byte
- Same source data + same Plotly version ≈ same HTML (Plotly may optimize JSON serialization slightly)
- SHA256 hashes lock figure content; changes trigger validation failures

---

## Best Practices

### Figure design

✓ **Do:**
- Use clear axis labels and units
- Include colorbar/legend for all heatmaps
- Add title and caption describing the insight
- Use consistent color schemes (viridis for spectral, RdBu for signed data)
- High DPI (150+) for print quality

❌ **Don't:**
- Use 3D plots in PNG (hard to interpret, fails for screen readers)
- Embed small unreadable text
- Use color schemes that don't print in grayscale
- Include redundant data (duplication vs. complementary views)

### Performance

- PNG generation: < 5 sec per figure (matplotlib is fast)
- HTML generation: 1–10 sec per figure (depends on interactivity level)
- If Plotly takes >30 sec, consider generating fewer interactive figures

### Storage

- Committed to repo: `docs/tutorials_v030/artifacts/` (read-only archive)
- Generated during tutorial: `outputs/v030_XX/` (ephemeral, not committed)
- GitHub release: PNG only (space-efficient; HTML link to generated artifacts)

---

## Deployment and Sharing

### Local and CI
- PNG and HTML stored in `outputs/v030_XX/`
- Validation happens in GitHub Actions
- Artifacts archived (optional) for reproducibility testing

### Documentation (ReadTheDocs, GitHub)
- Embed PNG images inline in markdown
- Link to interactive HTML versions (if available)
- Assume Plotly may not be present for users without installation

### Paper/Publication
- Use PNG (universally supported)
- Cite data archive with SHA256 hashes
- Supplementary materials can include interactive HTML

---

## Example: Complete Artifact Workflow

```python
# In a v0.3 tutorial notebook (Section 9)

import matplotlib.pyplot as plt
import hashlib
import json
import os

# Create outputs directory
os.makedirs('outputs/v030_01', exist_ok=True)

# STEP 1: Generate PNG figures (always)
fig, ax = plt.subplots()
ax.plot(signals.time_ms, signals.V_m[:, 0])
ax.set_xlabel('Time (ms)')
ax.set_ylabel('V_m (mV)')
ax.set_title('Figure 1: Membrane voltage (sample neuron)')
fig.savefig('outputs/v030_01/v030_01_voltage_trace.png', dpi=150)
plt.close(fig)

# STEP 2: Compute PNG hash
def compute_hash(filename):
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

png_hash = compute_hash('outputs/v030_01/v030_01_voltage_trace.png')
print(f"✓ PNG: {png_hash}")

# STEP 3: Generate HTML if Plotly available (optional)
try:
    import plotly.graph_objects as go
    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(
        x=signals.time_ms,
        y=signals.V_m[:, 0],
        mode='lines',
        name='V_m (neuron 0)',
    ))
    fig_html.update_layout(
        xaxis_title='Time (ms)',
        yaxis_title='V_m (mV)',
        title='Interactive: Membrane voltage'
    )
    fig_html.write_html('outputs/v030_01/v030_01_voltage_trace_interactive.html')
    html_hash = compute_hash('outputs/v030_01/v030_01_voltage_trace_interactive.html')
    print(f"✓ HTML: {html_hash}")
except ImportError:
    print("⊘ Plotly not available; PNG is sufficient")
    html_hash = None

# STEP 4: Record metadata
metadata = {
    "metadata": {
        "scenario": "v0.3.01",
        "title": "Single Neuron I — Izhikevich Phenomenology",
        "generated_at": "2026-05-23T12:34:56Z",
    },
    "figures": {
        "v030_01_voltage_trace.png": {
            "sha256": png_hash,
            "bytes": os.path.getsize('outputs/v030_01/v030_01_voltage_trace.png'),
            "type": "png",
            "description": "Membrane voltage trace for sample neuron",
            "interactive": False
        }
    }
}

if html_hash:
    metadata["figures"]["v030_01_voltage_trace_interactive.html"] = {
        "sha256": html_hash,
        "bytes": os.path.getsize('outputs/v030_01/v030_01_voltage_trace_interactive.html'),
        "type": "html",
        "description": "Interactive voltage trace with hover details",
        "interactive": True,
        "source_data": ["time_ms", "V_m"]
    }

# STEP 5: Save metadata
with open('outputs/v030_01/artifacts.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ All artifacts saved and hashed")
```

---

## See Also

- [Tutorial Template](template.md) — Section 9 (Figures and Artifacts)
- [Acceptance Gates](acceptance_gates.yaml) — Figure presence and hash validation gates
- [Scenario Index](scenario_index.md) — Figure types for each of 15 scenarios
