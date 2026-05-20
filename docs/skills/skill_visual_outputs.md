# Visual Outputs Developer Skill

This guide teaches you how to produce and validate interactive HTML visualizations from jaxfne simulations using Plotly.

## What is visual output in jaxfne?

Visual outputs are **optional** HTML figures generated from probe readouts. They supplement JSON output bundles with interactive, browser-accessible plots.

**Plotly is not a core dependency.** Install only when creating figures:

```bash
pip install plotly
```

## When to visualize

Use Plotly HTML figures when you:

- Need interactive exploration (zoom, pan, hover, toggle traces)
- Are sharing results with collaborators (email, shared drive, web)
- Want publication-ready plots with minimal post-processing
- Need quick iteration on visualization parameters

**Do NOT use Plotly when:**

- Figures are performance-critical (use NumPy/Matplotlib for batch processing)
- Data is confidential (HTML files embed all data; use server-side access control)
- You need hardcopy-only formats (use Matplotlib → PDF/PNG instead)
- Output must be a single file with zero external dependencies (use embedded Plotly, not CDN)

## Installation and setup

### 1. Install Plotly

```bash
pip install plotly>=5.0
```

### 2. Verify import

```python
import plotly.graph_objects as go
print(f"Plotly version: {go.__version__}")
```

### 3. Create output directory

```python
from pathlib import Path

run_name = "myexperiment_20260520_1200"
out_dir = Path(f"outputs/{run_name}")
figures_dir = out_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
```

## Code examples by operator

### SPK (spike raster)

```python
import plotly.graph_objects as go
import numpy as np

def plot_spike_raster(spikes, time_ms):
    """Plot spike raster from spike matrix."""
    # spikes: [T, N] binary array
    T, N = spikes.shape
    
    spike_times = []
    neuron_ids = []
    for n in range(N):
        indices = np.where(spikes[:, n] > 0)[0]
        spike_times.extend(indices)
        neuron_ids.extend([n] * len(indices))
    
    fig = go.Figure(data=go.Scatter(
        x=spike_times * (time_ms[1] - time_ms[0]) if len(time_ms) > 1 else spike_times,
        y=neuron_ids,
        mode="markers",
        marker=dict(size=3, color="black"),
        name="Spikes"
    ))
    
    fig.update_layout(
        title="Spike Raster",
        xaxis_title="Time (ms)",
        yaxis_title="Neuron ID",
        template="plotly_white"
    )
    return fig
```

### Vm (voltage traces)

```python
def plot_voltage_traces(V_m, contacts=None, max_traces=10):
    """Plot membrane voltage across selected neurons."""
    # V_m: [T, N] voltage array
    T, N = V_m.shape
    
    # Sample neurons if too many
    if N > max_traces:
        indices = np.linspace(0, N - 1, max_traces, dtype=int)
    else:
        indices = range(N)
    
    fig = go.Figure()
    for n in indices:
        fig.add_trace(go.Scatter(
            y=V_m[:, n],
            mode="lines",
            name=f"Neuron {n}",
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Membrane Voltage",
        xaxis_title="Time step",
        yaxis_title="Voltage (mV or native units)",
        template="plotly_white"
    )
    return fig
```

### LFP-proxy (laminar depth profile)

```python
def plot_lfp_heatmap(lfp_proxy, contact_depths=None):
    """Plot LFP-proxy across depth and time."""
    # lfp_proxy: [T, C] potential array
    
    fig = go.Figure(data=go.Heatmap(
        z=lfp_proxy.T,
        colorscale="RdBu",
        colorbar=dict(title="LFP-proxy (V)")
    ))
    
    fig.update_layout(
        title="LFP-proxy Laminar Profile",
        xaxis_title="Time (ms)",
        yaxis_title="Contact index",
        template="plotly_white",
        height=500,
        width=1000
    )
    return fig
```

### CSD-proxy (current-source density)

```python
def plot_csd_heatmap(csd_proxy):
    """Plot CSD-proxy spatial profile."""
    # csd_proxy: [T, C] CSD array
    
    fig = go.Figure(data=go.Heatmap(
        z=csd_proxy.T,
        colorscale="RdBu",
        zmid=0,  # Center colorbar at zero
        colorbar=dict(title="CSD-proxy")
    ))
    
    fig.update_layout(
        title="CSD-proxy (Current Source Density)",
        xaxis_title="Time (ms)",
        yaxis_title="Contact depth",
        template="plotly_white"
    )
    return fig
```

### Combined spectrolaminar view

```python
from plotly.subplots import make_subplots

def plot_spectrolaminar(lfp_proxy, csd_proxy, spikes):
    """Create 2x2 subplot: LFP, CSD, raster, power."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("LFP-proxy", "CSD-proxy", "Spike Raster", "LFP Power"),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Row 1, Col 1: LFP heatmap
    fig.add_trace(
        go.Heatmap(z=lfp_proxy.T, colorscale="Viridis", name="LFP"),
        row=1, col=1
    )
    
    # Row 1, Col 2: CSD heatmap
    fig.add_trace(
        go.Heatmap(z=csd_proxy.T, colorscale="RdBu", zmid=0, name="CSD"),
        row=1, col=2
    )
    
    # Row 2, Col 1: Raster
    T, N = spikes.shape
    spike_times, cell_ids = [], []
    for n in range(N):
        indices = np.where(spikes[:, n] > 0)[0]
        spike_times.extend(indices)
        cell_ids.extend([n] * len(indices))
    
    fig.add_trace(
        go.Scatter(x=spike_times, y=cell_ids, mode="markers", 
                   marker=dict(size=2, color="black"), name="Spikes"),
        row=2, col=1
    )
    
    # Row 2, Col 2: LFP power spectrum
    power = np.abs(np.fft.fft(lfp_proxy, axis=0)[:lfp_proxy.shape[0]//2, :])
    freqs = np.fft.fftfreq(lfp_proxy.shape[0])[:lfp_proxy.shape[0]//2]
    mean_power = power.mean(axis=1)
    
    fig.add_trace(
        go.Scatter(x=freqs, y=mean_power, mode="lines", name="Power"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1200, title_text="Spectrolaminar Summary")
    return fig
```

## HTML naming conventions

Use consistent, descriptive filenames:

```
lfp_proxy_trace.html          # LFP voltage traces (1D)
csd_proxy_heatmap.html        # CSD 2D space-time
raster_depth.html             # Spike raster
spike_rate_by_layer.html      # Per-layer spike statistics
spectrolaminar_summary.html   # Multi-panel overview
emm_cost_over_time.html       # Normalized cost metric
eeg_proxy_channels.html       # Toy EEG scalp channels
meg_proxy_sensors.html        # Toy MEG magnetometer
```

## Validation commands

### 1. Check Plotly version

```bash
python -c "import plotly; print(f'Plotly {plotly.__version__}')"
```

### 2. Generate and validate figures

```python
# Generate
fig = plot_lfp_heatmap(readout["lfp_proxy"])
out_dir = Path("outputs/myrun/figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.write_html(out_dir / "lfp_heatmap.html", include_plotlyjs="cdn")

# Verify file exists and has content
html_file = out_dir / "lfp_heatmap.html"
assert html_file.exists(), "HTML file was not created"
assert html_file.stat().st_size > 1000, "HTML file too small (< 1 KB)"
print(f"✓ {html_file} ({html_file.stat().st_size} bytes)")
```

### 3. Check JSON output bundles

```python
import json
from jaxfne.io import json_safe

# Verify manifests are JSON-safe
manifest = model.manifest(signals, readout)
json.dumps(json_safe(manifest), allow_nan=False)
print("✓ Manifest is JSON-safe (no NaN/Inf)")
```

### 4. Asset hashing

```python
import hashlib
from pathlib import Path

def hash_files(directory):
    """Compute SHA256 hashes of all files."""
    hashes = {}
    for f in Path(directory).glob("**/*"):
        if f.is_file():
            with open(f, "rb") as fp:
                hashes[f.name] = hashlib.sha256(fp.read()).hexdigest()
    return hashes

hashes = hash_files("outputs/myrun/figures")
for name, sha256 in hashes.items():
    print(f"{name}: {sha256[:16]}...")
```

## Common mistakes

### 1. Embedded Plotly library is huge

❌ **WRONG:**
```python
fig.write_html("trace.html", include_plotlyjs=True)  # 3+ MB!
```

✅ **CORRECT:**
```python
fig.write_html("trace.html", include_plotlyjs="cdn")  # 100 KB
```

### 2. Claiming physical amplitude in proxy figures

❌ **WRONG:**
```python
fig.update_layout(title="Real LFP from visual cortex")
```

✅ **CORRECT:**
```python
fig.update_layout(title="LFP-proxy (simulated, not calibrated)")
```

### 3. Missing directory creation

❌ **WRONG:**
```python
fig.write_html("outputs/myrun/figures/trace.html")  # Crashes if directory missing
```

✅ **CORRECT:**
```python
Path("outputs/myrun/figures").mkdir(parents=True, exist_ok=True)
fig.write_html("outputs/myrun/figures/trace.html")
```

### 4. Committing generated HTML to git

❌ **WRONG:**
```bash
git add outputs/myrun/figures/*.html
git push  # Now repository is large!
```

✅ **CORRECT:**
```bash
# Add to .gitignore:
echo "outputs/" >> .gitignore
echo "site/" >> .gitignore

# Commit only metadata:
git add outputs/myrun/manifest.json outputs/myrun/asset_hashes.json
git push
```

### 5. Not validating for NaN/Inf before plotting

❌ **WRONG:**
```python
fig = plot_lfp(readout["lfp_proxy"])  # May contain NaN/Inf!
fig.write_html("lfp.html")
```

✅ **CORRECT:**
```python
import numpy as np

lfp = readout["lfp_proxy"]
assert np.isfinite(lfp).all(), "LFP contains NaN or Inf"
fig = plot_lfp(lfp)
fig.write_html("lfp.html")
```

## Relationship to JSON output bundles

Visualizations are **supplementary**, not primary outputs. Always produce:

1. **Manifest (JSON)** — Experiment metadata, operator contracts, validation flags
2. **Metrics (JSON)** — Numerical summaries (spike rate, LFP power, etc.)
3. **Figures (HTML)** — Interactive plots for exploration and sharing

```
outputs/
  myrun_20260520/
    manifest.json        ← Primary (keep in git)
    metrics.json         ← Primary (keep in git)
    asset_hashes.json    ← Primary (keep in git)
    figures/
      lfp_trace.html     ← Secondary (do NOT commit)
      csd_heatmap.html   ← Secondary (do NOT commit)
```

## See also

- [Plotly visualization guide](../plotly_visualization.md) — Code examples and best practices
- [Probe operators](../probe_operators.md) — Operator specifications with equations
- [Tensor-field workflows](../tensor_field_workflows.md) — Full pipeline overview

---

**Status:** v0.2.14  
**Last updated:** 2026-05-20  
**Plotly version:** ≥ 5.0, optional dependency
