# Plotly Visualization Guide

## Overview

Plotly is **optional** — a jaxfne addition for creating interactive HTML visualizations. Install it when needed for this feature:

```bash
pip install plotly
```

This guide describes standard practices for generating reproducible, portable HTML figures from jaxfne simulations.

## Installation

Add to your environment:

```bash
pip install plotly>=5.0
```

Or include in a project `requirements.txt`:

```
jaxfne>=0.2.14
plotly>=5.0
numpy
scipy
```

## Output directory structure

Use this standard layout for all runs:

```
outputs/
  <run_name>/
    manifest.json              # Full experiment metadata
    probe_report.json          # Probe status and contracts
    metrics.json               # Numerical results summary
    validation_report.json     # NaN/Inf/admissibility checks
    asset_hashes.json          # SHA256 hashes of all figures
    figures/
      lfp_proxy_trace.html
      csd_proxy_heatmap.html
      raster_depth.html
      spectrolaminar_summary.html
```

Create directories automatically:

```python
from pathlib import Path
run_name = "myrun_20260520_1200"
figures_dir = Path(f"outputs/{run_name}/figures")
figures_dir.mkdir(parents=True, exist_ok=True)
```

## Code examples

### LFP-proxy trace (1D time series)

```python
import plotly.graph_objects as go
from pathlib import Path

# Assume: lfp_proxy shape [T, C] (time × contacts)
lfp = readout["lfp_proxy"]
T, C = lfp.shape

fig = go.Figure()
for contact in range(C):
    fig.add_trace(go.Scatter(
        y=lfp[:, contact],
        mode="lines",
        name=f"Contact {contact}",
        opacity=0.7
    ))

fig.update_layout(
    title="LFP-proxy across laminar contacts",
    xaxis_title="Time (ms)",
    yaxis_title="Potential (proxy units)",
    template="plotly_white",
    height=500,
    width=1000
)

out = Path("outputs/myrun/figures")
out.mkdir(parents=True, exist_ok=True)
fig.write_html(
    out / "lfp_proxy_trace.html",
    include_plotlyjs="cdn",  # Use CDN for optimal file size
    full_html=True
)
```

### CSD-proxy heatmap (2D space-time)

```python
import plotly.graph_objects as go
import numpy as np

# Assume: csd_proxy shape [T, C] (time × contacts)
csd = readout["csd_proxy"]

fig = go.Figure(data=go.Heatmap(
    z=csd.T,  # Transpose for space on y-axis
    x=np.arange(csd.shape[0]),
    y=np.arange(csd.shape[1]),
    colorscale="RdBu",  # Red/blue for positive/negative
    colorbar=dict(title="CSD (proxy)")
))

fig.update_layout(
    title="CSD-proxy heatmap (laminar profile)",
    xaxis_title="Time (ms)",
    yaxis_title="Contact depth",
    template="plotly_white",
    height=500,
    width=1000
)

out.mkdir(parents=True, exist_ok=True)
fig.write_html(
    out / "csd_proxy_heatmap.html",
    include_plotlyjs="cdn",
    full_html=True
)
```

### Raster plot (spikes by depth)

```python
import plotly.graph_objects as go

# Assume: spikes shape [T, N] or spike times + cell IDs
spikes = readout["spikes"]  # [T, N]
T, N = spikes.shape

# Find spike times
spike_times = []
cell_ids = []
for n in range(N):
    spike_indices = np.where(spikes[:, n] > 0)[0]
    if len(spike_indices) > 0:
        spike_times.extend(spike_indices)
        cell_ids.extend([n] * len(spike_indices))

fig = go.Figure(data=go.Scatter(
    x=spike_times,
    y=cell_ids,
    mode="markers",
    marker=dict(size=4, color="black"),
    name="Spikes"
))

fig.update_layout(
    title="Spike raster",
    xaxis_title="Time (ms)",
    yaxis_title="Neuron ID",
    template="plotly_white",
    height=600,
    width=1200
)

fig.write_html(
    out / "raster_depth.html",
    include_plotlyjs="cdn",
    full_html=True
)
```

### Spectrolaminar summary (multiple panels)

```python
from plotly.subplots import make_subplots

# Create 2x2 subplot grid
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("LFP power", "CSD profile", "Spike rate", "Source norm")
)

# Placeholder: add traces as needed
# fig.add_trace(go.Scatter(...), row=1, col=1)
# etc.

fig.update_layout(height=800, width=1200, title="Spectrolaminar summary")
fig.write_html(
    out / "spectrolaminar_summary.html",
    include_plotlyjs="cdn",
    full_html=True
)
```

## Best practices

### 1. Use CDN for optimal file size

Always use `include_plotlyjs="cdn"`:

```python
fig.write_html(
    "figure.html",
    include_plotlyjs="cdn",  # ~10 KB file size
    full_html=True
)
```

Embedded Plotly library inflates files to 3–5 MB each. CDN-linked files stay ~10–100 KB and load the library once.

### 2. File size and performance

- **Per-figure size:** 10–200 KB (with CDN)
- **Heatmaps:** Can be large if data is very high-dimensional; consider decimation
- **Raster plots:** Sparse data (many empty time points) may benefit from downsampling

### 3. Declarative metadata

Store figure provenance in `asset_hashes.json`:

```python
import hashlib
import json

def file_sha256(path):
    """Compute SHA256 hash of file."""
    hash_obj = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

# After all figures are written:
figures_dir = Path("outputs/myrun/figures")
asset_hashes = {
    f.name: file_sha256(f)
    for f in figures_dir.glob("*.html")
}

with open(Path("outputs/myrun") / "asset_hashes.json", "w") as fp:
    json.dump(asset_hashes, fp, indent=2)
```

## Artifact hygiene

### Exclude generated HTML from version control

Add to `.gitignore`:

```
outputs/
site/
```

Generated HTML figures, build artifacts, and `site/` from `mkdocs build` should never be committed.

### Do commit manifests and metadata

Keep version control of:
- `manifest.json` (experiment metadata)
- `probe_report.json` (operator contracts)
- `metrics.json` (numerical summaries)
- `validation_report.json` (NaN/Inf checks)
- `asset_hashes.json` (reproducibility checksums)

These are small (KB scale) and document the run without storing large arrays.

### Output bundle example

```python
import json
from jaxfne.io import json_safe

# After simulation and probing:
outputs = {
    "manifest": model.manifest(signals, readout),
    "probe_report": readout,  # Already JSON-safe
    "metrics": {
        "spike_rate_mean": float(spike_count / duration_ms),
        "lfp_power_mean": float(lfp.std(axis=0).mean()),
    },
    "validation": {
        "lfp_finite": bool(np.isfinite(lfp).all()),
        "csd_finite": bool(np.isfinite(csd).all()),
    }
}

# Write manifests (keep)
out_dir = Path(f"outputs/{run_name}")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "manifest.json", "w") as f:
    json.dump(json_safe(outputs["manifest"]), f, indent=2, allow_nan=False)
with open(out_dir / "metrics.json", "w") as f:
    json.dump(json_safe(outputs["metrics"]), f, indent=2, allow_nan=False)

# Write figures (don't commit)
figs_dir = out_dir / "figures"
figs_dir.mkdir(exist_ok=True)
fig.write_html(figs_dir / "lfp_proxy_trace.html", include_plotlyjs="cdn", full_html=True)
```

## Common mistakes

### 1. Embedded Plotly library

❌ **INEFFICIENT:**
```python
fig.write_html("figure.html", include_plotlyjs=True)  # ~3 MB file
```

✅ **RECOMMENDED:**
```python
fig.write_html("figure.html", include_plotlyjs="cdn")  # ~100 KB
```

### 2. Correctly labeling proxy-scale readouts

❌ **INCORRECT:**
```python
# Avoid claiming physical amplitude:
fig.update_layout(title="Real LFP recorded from V1 cortex")  # Use proxy-scale label
```

✅ **CORRECT:**
```python
fig.update_layout(title="LFP-proxy from proxy field operator (proxy-scale units)")
```

### 3. Version control for outputs/

❌ **AVOID:**
```bash
git add outputs/myrun/figures/*.html  # Exclude generated figures
git push
```

✅ **CORRECT:**
```bash
git add outputs/myrun/manifest.json outputs/myrun/asset_hashes.json  # Metadata only
git push
```

### 4. Ensure directories exist

❌ **INCOMPLETE:**
```python
fig.write_html("outputs/myrun/figures/trace.html")  # Missing directory check
```

✅ **CORRECT:**
```python
Path("outputs/myrun/figures").mkdir(parents=True, exist_ok=True)
fig.write_html("outputs/myrun/figures/trace.html")
```

## See also

- [Probe operators](probe_operators.md) — Operator descriptions with equations
- [Tensor-field workflows](tensor_field_workflows.md) — Full pipeline overview

---

**Status:** v0.2.14  
**Last updated:** 2026-05-20  
**Plotly:** Optional jaxfne addition for interactive visualizations
