# v0.3 Tutorial Visualization Doctrine

**Version:** v0.3.0+  
**Last updated:** 2026-05-24  
**truth_mode:** truth_safe_unverified  

---

## Core Principle

**Visualizations are first-class evidence artifacts.** They document simulation outcomes, validate computational gates, and enable exploratory learning. Every core figure must have stable provenance: PNG hash, generation code path, and clear proxy/simulation status declaration.

---

## PNG Figures (MANDATORY)

### Requirement Rule

**Every v0.3 tutorial must produce at least 2–4 PNG figures.**

- Minimum: 2 figures (spike raster + voltage trace for single-neuron tutorials)
- Recommended: 3–4 complementary views per scenario for pedagogy
- All PNG must have valid SHA256 hashes recorded in manifest
- All PNG stored in `docs/tutorials_v030/_static/figures/` for docs stability
- All PNG generated at `dpi=150` or higher (300 dpi for publication-intent)

### PNG Specifications

**Technical contract:**
- Format: PNG (uncompressed or PNG-8)
- Resolution: **150 dpi minimum** (300 dpi recommended)
- Color depth: RGB or RGBA (transparency allowed for overlays)
- File size: ≥1 KB (indicates real content, not blank)
- Naming convention: `v030_XX_YY_description.png`
  - XX = scenario number (01–15, zero-padded)
  - YY = figure index within scenario (01, 02, 03, etc., zero-padded)
  - description = short slug (e.g., `spike_raster`, `voltage_trace`, `csd_heatmap`)

**Example filenames:**
```
v030_01_01_spike_raster.png          # Scenario 1, Figure 1
v030_01_02_voltage_trace.png         # Scenario 1, Figure 2
v030_01_03_phase_plane.png           # Scenario 1, Figure 3
v030_02_01_parameter_sweep_heatmap.png  # Scenario 2, Figure 1
v030_05_02_laminar_raster.png        # Scenario 5, Figure 2
v030_08_03_csd_heatmap.png           # Scenario 8, Figure 3
```

### SHA256 Integrity

Every PNG must be hashed immediately after generation:

```python
import hashlib

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

# After saving PNG
png_path = "outputs/v030_XX/figures/v030_XX_YY_description.png"
sha256 = compute_file_hash(png_path)
print(f"✓ PNG hash: {sha256}")
```

**Validation workflow:**
1. Generate PNG from simulation data
2. Compute SHA256 immediately after save
3. Record in manifest JSON
4. On subsequent runs, recompute and compare
5. Hash mismatch triggers re-generation or corruption alert

---

## Plotly Interactive HTML (OPTIONAL BUT PREFERRED)

### When to Use Plotly

**Use Plotly HTML when:**
- Interactive exploration adds significant value (hover details, zoom, pan, range slider)
- Multiple subplots or 3D visualization improves interpretation
- Plotly is available in user environment (guarded import)
- Tutorial complexity justifies extra generation time (typically <10 sec per figure)

**Do NOT use Plotly when:**
- Plotly import fails (users without plotly should not be blocked)
- PNG alone suffices for understanding
- Interactivity has no pedagogical benefit
- Tutorial generation time would exceed budget (e.g., >60 sec total per scenario)

### Guarded Import Pattern

```python
# At top of tutorial script
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

# Later: generate HTML only if available
if PLOTLY_AVAILABLE:
    fig = go.Figure()
    # ... build figure from source data ...
    fig.write_html(filepath)
    print(f"✓ Interactive HTML: {filepath}")
else:
    print("⊘ Plotly not available; PNG figure is sufficient")
```

### Generation from Source Data (Not PNG Conversion)

**Correct approach:** Generate Plotly directly from simulation data (arrays, lists, dicts).

```python
# Data-driven generation (CORRECT)
spike_times = signals.spike_times  # List or array
neuron_ids = list(range(n_neurons))

fig = go.Figure()
for neuron_id in neuron_ids:
    spikes_i = spike_times[neuron_id]
    for t in spikes_i:
        fig.add_vline(x=t, line_dash="dash", line_color="black", opacity=0.5)

fig.update_layout(
    title="Interactive Spike Raster",
    xaxis_title="Time (ms)",
    yaxis_title="Neuron ID",
    height=400
)
fig.write_html('outputs/v030_XX/figures/v030_XX_YY_description_interactive.html')
```

**Incorrect approach:** Do NOT convert PNG to HTML (loses fidelity and violates data-source principle).

```python
# Image conversion (WRONG) ❌
from PIL import Image
img = Image.open('v030_01_raster.png')
# ... some conversion logic ...
# This violates the source-data principle
```

### Plotly HTML Specifications

**Technical contract:**
- Format: HTML5 with embedded Plotly.js
- Self-contained (no external CDN required for offline use)
- File size: 100 KB–5 MB typical (depends on data size and complexity)
- Naming: `v030_XX_YY_description_interactive.html` (same XX, YY, description as PNG)
- Compression: gzip-friendly (optional in manifest for future archival)

**Recommended Plotly figure types:**
- **Raster plots**: scatter or vlines with hover (neuron ID, spike time)
- **Voltage traces**: line plots with range slider and crosshair
- **Heatmaps**: imshow-like with colorbar and hover cell values
- **Spectrograms**: 2D color plots with frequency/time axes
- **3D scatter**: parameter sweep results, Pareto fronts
- **Surface plots**: parameter space, fitness landscape
- **Multi-subplot layouts**: side-by-side comparisons (standard vs. oddball, etc.)

---

## Standard Figure Panel Families

Recommended visualizations by scenario class. Use these as templates to ensure consistent pedagogy and coverage.

### Single-Neuron Tutorials (v0.3.1–v0.3.2)

**Required panels:**
1. **Spike raster** — Time axis, single-point marks at spike times (PNG required)
2. **Voltage trace** — Time vs. membrane potential V_m (mV) (PNG required)
3. **Optional:** Phase plane (V_m vs. recovery variable u) or recovery trace
4. **Optional:** Current input and/or native state dynamics

**Plotly recommendations:**
- Interactive voltage trace with range slider
- Hover-enabled raster with neuron/spike details
- Draggable phase-plane plot (V_m vs. u trajectory)

---

### Parameter Sweep Tutorials (v0.3.2, v0.3.13, v0.3.14)

**Required panels:**
1. **Heatmap** — Parameter grid with firing rate or fitness values, color-coded regime labels (PNG required)
2. **Regime classification** — Grid with target/out-of-target labels overlaid (PNG required)
3. **Optional:** Sample traces from baseline condition (voltage, raster)
4. **Optional:** Pareto front (multi-objective: 2D or 3D scatter)

**Plotly recommendations:**
- Interactive heatmap with hover (parameter values, firing rate)
- 3D Pareto-front scatter (multi-objective optimization)
- Searchable/filterable table of sweep results

---

### Two-Neuron and Small Network Tutorials (v0.3.4)

**Required panels:**
1. **Raster** — E and I neuron spikes stacked (PNG required)
2. **Voltage traces** — V_m_E and V_m_I overlaid or side-by-side (PNG required)
3. **Optional:** Cross-correlogram (spike timing relationship)
4. **Optional:** Coupling diagram (schematic of E→I, I→E connections)

**Plotly recommendations:**
- Interactive multi-trace voltage plot with legend toggle
- Cross-correlogram with interactive histogram bins
- Dynamic phase-locking indicator (if applicable)

---

### Laminar Column Tutorials (v0.3.5)

**Required panels:**
1. **3D circuit anatomy** (dark themed, circuit-level view if >50 neurons) OR laminar schematic diagram (PNG preferred)
2. **Laminar raster** — Neurons color-coded by layer, depth on y-axis (PNG required)
3. **Laminar-resolved rates** — Line plot of firing rate per layer vs. time (PNG required)
4. **CSD heatmap** — Layer depth (y-axis) vs. time (x-axis), color = current density (PNG required)
5. **Optional:** Synchrony/coherence per layer

**Plotly recommendations:**
- Interactive 3D circuit viewer (if plotly-3d tools available)
- Zoomable laminar raster with layer hover-labels
- Animated CSD heatmap (per-layer time-series playback)

---

### Field Readout Tutorials (v0.3.7–v0.3.8)

**Required panels:**
1. **Source/current trace** — Population-level current proxy vs. time (PNG required)
2. **Extracellular potential (φ_e)** — Single recording location trace (PNG required)
3. **Field heatmap** — Space (x, y, or depth) vs. time, color = potential or CSD (PNG required)
4. **LFP multi-channel** — 16-channel probe traces, stacked display (PNG required)
5. **Frequency spectrum** — Power spectral density per channel or averaged (PNG required)

**Plotly recommendations:**
- Interactive field heatmap with hover location/time/value
- Scrollable LFP channel viewer (click to focus, range slider for time)
- Animated CSD heatmap (real-time layer-wise activity)
- 3D field potential surface (space × field magnitude)

---

### Advanced Tutorials: Oddball/Omission (v0.3.9–v0.3.10)

**Required panels:**
1. **Stimulus raster** — Event marks (standard vs. oddball color-coded) (PNG required)
2. **Trial-averaged raster** — Standard vs. oddball response, time-locked to stimulus (PNG required)
3. **Cross-area ERP** — Area-wise averaged response latency and polarity (PNG required)
4. **Spectral comparison** — Power spectrum (standard vs. oddball) (PNG required)

**Plotly recommendations:**
- Interactive stimulus timeline with legend filtering
- Draggable trial-selection boundaries
- Animated ERP (play through trial sequence)

---

### Learning/Plasticity Tutorials (v0.3.11–v0.3.12)

**Required panels:**
1. **STDP window** — Δt (time difference) vs. ΔW (weight change) characteristic function (PNG required)
2. **Weight evolution** — Selected synapses vs. trial number, color by pre/post type (PNG required)
3. **Connectivity heatmap** — Before/after learning, weight matrix (PNG required)
4. **Firing rate dynamics** — Population rate during learning, with stability markers (PNG required)

**Plotly recommendations:**
- Interactive STDP window (hover to see rule parameters)
- Animated weight-evolution plot (play through learning)
- Comparative connectivity before/after (toggle layer)

---

### Optimization Tutorials (v0.3.13–v0.3.14)

**Required panels:**
1. **Fitness evolution** — Best/median/worst fitness per generation (PNG required)
2. **Parameter trajectories** — Selected parameters vs. generation (PNG required)
3. **Fitted model output** — Target vs. final simulated metrics (PNG required; e.g., firing rate comparison)
4. **Pareto front** (multi-objective) — 2D or 3D scatter of non-dominated solutions (PNG required)

**Plotly recommendations:**
- Interactive fitness plot (hover to see generation details)
- 3D Pareto-front scatter with rotation/zoom
- Parameter trajectory animation (generation-by-generation playback)
- Table browser (sortable by fitness or parameter)

---

### Integration Capstone (v0.3.15)

**Required panels:**
1. **Large-network raster** — Visual sample of 1000+ neuron activity (PNG required, may be subsampled)
2. **Benchmark results** — Wall-clock time vs. network size, with scaling prediction (PNG required)
3. **Integrated readout panel** — All 8 modalities side-by-side (raster, LFP, CSD, spectrum, etc.) (PNG required)
4. **Truth boundary diagram** — Flowchart or table showing what is computed vs. proxy (PNG required)

**Plotly recommendations:**
- Interactive 3D network layout visualization
- Zoomable benchmark plot with model-fit curve
- Tabbed interface for readout modality switching

---

## Figure Manifest Contract

Every figure (PNG + optional HTML) must be registered in a manifest JSON:

```json
{
  "figures": {
    "spike_raster": {
      "scenario": "v0.3.01",
      "figure_id": "v030_01_01_spike_raster",
      "title": "Single Neuron Spike Raster",
      "png_path": "docs/tutorials_v030/_static/figures/v0301_01_spike_raster.png",
      "html_path": null,
      "interactive": false,
      "backend": "matplotlib",
      "sha256_png": "abc123def456...",
      "sha256_html": null,
      "claim_status": "simulated_proxy",
      "visual_role": "primary_spike_evidence",
      "source_data": ["V_m", "spikes"],
      "visually_confirmed": true,
      "plotly_available": true,
      "plotly_html_generated": false,
      "dpi": 150,
      "notes": "Spike raster for 1 neuron, 100 ms duration, Izhikevich model"
    },
    "voltage_trace": {
      "scenario": "v0.3.01",
      "figure_id": "v030_01_02_voltage_trace",
      "title": "Membrane Voltage Dynamics",
      "png_path": "docs/tutorials_v030/_static/figures/v0301_02_voltage_trace.png",
      "html_path": "outputs/v030_01_single_izhikevich_neuron/figures/v030_01_02_voltage_trace_interactive.html",
      "interactive": true,
      "backend": "plotly",
      "sha256_png": "ghi789jkl012...",
      "sha256_html": "mno345pqr678...",
      "claim_status": "simulated_proxy",
      "visual_role": "primary_voltage_evidence",
      "source_data": ["time_ms", "V_m"],
      "visually_confirmed": true,
      "plotly_available": true,
      "plotly_html_generated": true,
      "dpi": 150,
      "notes": "V_m trace with resting, threshold, and spike reset dynamics"
    }
  }
}
```

**Manifest fields:**

| Field | Required? | Value | Notes |
|-------|-----------|-------|-------|
| `scenario` | Yes | v0.3.XX | Scenario identifier (01–15) |
| `figure_id` | Yes | v030_XX_YY_desc | Unique figure identifier |
| `title` | Yes | Human-readable string | For documentation and figure captions |
| `png_path` | Yes | docs/tutorials_v030/_static/figures/... | Stable path for docs inclusion |
| `html_path` | If HTML present | outputs/.../v030_XX_YY_desc_interactive.html | Optional interactive figure |
| `interactive` | Yes | bool | True if HTML interactive, false for PNG-only |
| `backend` | Yes | "matplotlib" \| "plotly" \| "custom" | Source rendering tool |
| `sha256_png` | Yes | 64-char hex string | SHA256 of PNG file |
| `sha256_html` | If HTML | 64-char hex string | SHA256 of HTML file |
| `claim_status` | Yes | "simulated_proxy" | All v0.3 figures are proxy/simulated |
| `visual_role` | Yes | "primary", "diagnostic", "comparative", "reference" | Figure purpose in tutorial |
| `source_data` | Yes | List of field names | Data sources used to generate figure |
| `visually_confirmed` | Yes | bool | Human verified figure is readable and correct |
| `plotly_available` | Yes | bool | Plotly library available at generation time |
| `plotly_html_generated` | Yes | bool | HTML figure was successfully created |
| `dpi` | Yes (PNG) | 150 or higher | Raster resolution for PNG |
| `notes` | No | String | Optional metadata or generation notes |

---

## Tutorial Checklist

Every v0.3 tutorial must satisfy this checklist:

- [ ] **Minimum figures:** ≥2 PNG figures generated
- [ ] **Recommended figures:** 3–4 complementary PNG panels for pedagogy
- [ ] **PNG quality:** dpi ≥150, file size ≥1 KB, valid image
- [ ] **Naming convention:** `v030_XX_YY_description.png` exactly
- [ ] **SHA256 hashes:** All PNG hashes computed and recorded
- [ ] **Docs-stable path:** All PNG copied to `docs/tutorials_v030/_static/figures/`
- [ ] **Plotly optional:** HTML figures generated if Plotly available (guarded import)
- [ ] **HTML source data:** If HTML present, generated from source arrays (not PNG conversion)
- [ ] **Manifest complete:** All figures registered in scenario manifest JSON
- [ ] **Manifest validation:** manifest.json passes schema check (all required fields present)
- [ ] **No overclaiming:** Figure titles and captions do NOT claim biological/physical truth
- [ ] **Captions explicit:** All captions state "simulated", "proxy", "computational"
- [ ] **Cross-references:** README and scenario_index.md link to figures correctly
- [ ] **Docs render:** Markdown figures render without broken links in local/GitHub preview

---

## Proxy-Safe Titles and Labels

### Do's ✓

- "Simulated spike raster for 1 Izhikevich neuron, 100 ms"
- "Proxy current source density (CSD) heatmap; laminar_proxy_no_pde mode"
- "Fictitious feedforward input to sensory area (stimulus proxy)"
- "Computational scaffold demonstration: multi-area network activity"
- "Estimated firing rate per layer (from simulated voltage)"

### Don'ts ❌

- "Actual neuronal spike pattern recorded from cortex"
- "Real current source density from tissue"
- "Physiological laminar CSD with conservation proof"
- "Validated EEG-like readout"
- "Biologically calibrated LFP signal"

---

## Visualization Absence Handling

If Plotly is unavailable or HTML generation fails:

1. **PNG generation must still succeed** (abort only if critical matplotlib error)
2. **Manifest records:** `plotly_html_generated: false`
3. **Manifest records:** `plotly_available: false`
4. **Tutorial continues:** PNG-only figures are sufficient for all v0.3 scenarios
5. **Log message:** Print "`⊘ Plotly not available; PNG figure is sufficient`"
6. **No failing gate:** Absence of HTML is NOT a validation failure

**Example guard code:**

```python
figures_manifest = {}

# Always generate PNG
png_path = "outputs/v030_XX/figures/v030_XX_YY_description.png"
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(...)
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)

png_hash = compute_file_hash(png_path)
figures_manifest["description"] = {
    "png_path": png_path,
    "sha256_png": png_hash,
    "plotly_available": PLOTLY_AVAILABLE,
    "plotly_html_generated": False
}

# Try to generate HTML if Plotly available
if PLOTLY_AVAILABLE:
    try:
        fig_html = go.Figure()
        # ... build figure ...
        html_path = "outputs/v030_XX/figures/v030_XX_YY_description_interactive.html"
        fig_html.write_html(html_path)
        html_hash = compute_file_hash(html_path)
        figures_manifest["description"]["html_path"] = html_path
        figures_manifest["description"]["sha256_html"] = html_hash
        figures_manifest["description"]["plotly_html_generated"] = True
    except Exception as e:
        print(f"⚠ HTML generation failed: {e}; PNG is sufficient")
```

---

## Storage and Archival

### During Tutorial Execution

- **Generated figures:** `outputs/v030_XX_description/figures/`
- **Temporary/ephemeral:** Not committed to repo
- **Runtime validation:** Hashes computed, anomalies logged

### Docs-Stable Archive

- **Committed to repo:** `docs/tutorials_v030/_static/figures/`
- **Read-only after commit:** Hashes tracked for integrity
- **Cross-references:** Linked from tutorial markdown and manifests

### GitHub Release

- **PNG only** (space-efficient, universally supported)
- **HTML optional:** Link to artifact server if available
- **Manifest:** JSON hashes allow reproducibility verification

---

## Best Practices

### Figure Design

✓ **Do:**
- Use clear axis labels with units
- Include colorbar/legend for all heatmaps and multi-trace plots
- Add descriptive title and 1–2 sentence caption
- Use consistent color schemes (viridis for spectral, RdBu for signed, gray for neutral)
- High DPI (150+) for print/publication quality
- Proxy-safe language in all labels and titles

❌ **Don't:**
- Use 3D plots in static PNG (hard to interpret, inaccessible)
- Embed small unreadable text
- Use non-grayscale-safe color schemes
- Include redundant or decorative data
- Make biological/physical amplitude claims
- Assume field solver accuracy (state proxy status)

### Performance

- **PNG generation:** <5 sec per figure (matplotlib is fast)
- **HTML generation:** 1–10 sec per figure (depends on interactivity)
- **Total per scenario:** Target <60 sec for all figures combined
- **Monitoring:** Print elapsed time per figure; warn if >30 sec

### Naming and Organization

```
outputs/v030_XX_tutorial_name/
├── figures/
│   ├── v030_XX_01_primary_figure.png
│   ├── v030_XX_01_primary_figure_interactive.html
│   ├── v030_XX_02_secondary_figure.png
│   └── v030_XX_03_diagnostic_figure.png
└── artifacts.json  # or manifest.json

docs/tutorials_v030/_static/figures/
├── v0301_01_spike_raster.png
├── v0301_02_voltage_trace.png
├── v0302_01_parameter_sweep_heatmap.png
└── ... (docs-stable copies)
```

---

## Example: Complete Figure Workflow

Below is a minimal working example from a v0.3 tutorial:

```python
#!/usr/bin/env python3
"""v0.3.01 Single Neuron: Figure Generation Workflow"""

import hashlib
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Guarded Plotly import
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

def compute_file_hash(filepath: str) -> str:
    """SHA256 hash of file."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    OUT = Path("outputs/v030_01_single_neuron_izhikevich")
    STATIC = Path("docs/tutorials_v030/_static/figures")
    
    # Create directories
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    
    # Simulate (placeholder)
    time_ms = np.linspace(0, 100, 1000)
    V_m = np.sin(time_ms / 20) * 30 + np.random.randn(1000) * 2  # Dummy data
    spikes_idx = np.where(V_m > 20)[0]
    
    figures_manifest = {}
    
    # =====================================================================
    # FIGURE 1: Spike Raster (PNG only)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.vlines(time_ms[spikes_idx], 0, 1, color='black', linewidth=1.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spike')
    ax.set_title('Figure 1: Simulated Spike Raster (Izhikevich Model)')
    
    png_path_raster = OUT / "figures" / "v030_01_01_spike_raster.png"
    fig.savefig(png_path_raster, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    png_hash_raster = compute_file_hash(str(png_path_raster))
    print(f"✓ PNG raster: {png_hash_raster}")
    
    figures_manifest["spike_raster"] = {
        "scenario": "v0.3.01",
        "figure_id": "v030_01_01_spike_raster",
        "png_path": str(png_path_raster),
        "sha256_png": png_hash_raster,
        "claim_status": "simulated_proxy",
        "visual_role": "primary",
        "dpi": 150
    }
    
    # =====================================================================
    # FIGURE 2: Voltage Trace (PNG + interactive HTML)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ms, V_m, color='steelblue', linewidth=1)
    ax.axhline(-65, color='gray', linestyle='--', alpha=0.5, label='Rest')
    ax.axhline(-50, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V_m (mV)')
    ax.set_title('Figure 2: Simulated Membrane Voltage (Izhikevich Model)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    png_path_voltage = OUT / "figures" / "v030_01_02_voltage_trace.png"
    fig.savefig(png_path_voltage, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    png_hash_voltage = compute_file_hash(str(png_path_voltage))
    print(f"✓ PNG voltage: {png_hash_voltage}")
    
    figures_manifest["voltage_trace"] = {
        "scenario": "v0.3.01",
        "figure_id": "v030_01_02_voltage_trace",
        "png_path": str(png_path_voltage),
        "sha256_png": png_hash_voltage,
        "claim_status": "simulated_proxy",
        "visual_role": "primary",
        "dpi": 150,
        "plotly_available": PLOTLY_AVAILABLE,
        "plotly_html_generated": False
    }
    
    # Try interactive HTML
    if PLOTLY_AVAILABLE:
        try:
            fig_html = go.Figure()
            fig_html.add_trace(go.Scatter(
                x=time_ms, y=V_m,
                mode='lines',
                name='V_m',
                line=dict(color='steelblue', width=1)
            ))
            fig_html.update_layout(
                title='Interactive: Membrane Voltage (Izhikevich)',
                xaxis_title='Time (ms)',
                yaxis_title='V_m (mV)',
                hovermode='x unified',
                height=400
            )
            
            html_path_voltage = OUT / "figures" / "v030_01_02_voltage_trace_interactive.html"
            fig_html.write_html(str(html_path_voltage))
            html_hash_voltage = compute_file_hash(str(html_path_voltage))
            print(f"✓ HTML voltage: {html_hash_voltage}")
            
            figures_manifest["voltage_trace"]["html_path"] = str(html_path_voltage)
            figures_manifest["voltage_trace"]["sha256_html"] = html_hash_voltage
            figures_manifest["voltage_trace"]["plotly_html_generated"] = True
        except Exception as e:
            print(f"⚠ HTML generation failed: {e}; PNG is sufficient")
    else:
        print("⊘ Plotly not available; PNG figure is sufficient")
    
    # =====================================================================
    # MANIFEST
    # =====================================================================
    manifest_path = OUT / "manifest.json"
    manifest = {
        "scenario": "v0.3.01",
        "title": "Single Neuron I — Izhikevich Phenomenology",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "figures": figures_manifest
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    print(f"✓ Manifest: {manifest_path}")
    
    # =====================================================================
    # DOCS-STABLE COPIES
    # =====================================================================
    import shutil
    shutil.copy2(png_path_raster, STATIC / "v0301_01_spike_raster.png")
    shutil.copy2(png_path_voltage, STATIC / "v0301_02_voltage_trace.png")
    print(f"✓ Copied to docs/_static/figures/")

if __name__ == "__main__":
    main()
```

---

## See Also

- [Plotly Policy](plotly_policy.md) — Original Plotly and PNG integration documentation
- [Template](template.md) — Section 9 (Figures and Artifacts)
- [Scenario Index](scenario_index.md) — Figure types for each of 15 scenarios
- [Acceptance Gates](acceptance_gates.yaml) — Figure presence and hash validation
- [Tutorial README](README.md) — Overview and status
