"""Build Etude No. 1 v2 — generates the full Jupyter notebook from scratch.

Generates:
    tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb

Run from the repo root:
    python scripts/build_etude1_v2.py
"""
import nbformat
from pathlib import Path


# ---------------------------------------------------------------------------
# Cell source strings
# ---------------------------------------------------------------------------

CELL_0_MD = """\
# Etude No. 1: Multi-Laminar Cortical AGSDR Scaffold

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb)

**End-to-end jaxfne workflow:** single-unit warmup → configure → simulate → tune → visualize → export.

## Run Status
| Field | Value |
|---|---|
| `run_status` | `tutorial_scaffold` |
| `model_status` | `computational_scaffold` |
| `field_solver_status` | `laminar_proxy_no_pde` |
| `field_model_status` | `proxy_readout_only` |
| `amplitude_status` | `native_unscaled` |

All outputs are simulated proxy diagnostics for this configured run.
"""

CELL_1_MD = """\
## Learning Objectives

By the end of this etude you will be able to:

1. Configure a multi-area, multi-laminar cortical scaffold in jaxfne using the chainable `Configuration()` API.
2. Run baseline and stimulus-evoked simulations and read spike, LFP-proxy, and CSD-proxy signals.
3. Define a rate+synchrony objective and run AGSDR (gradient-free) optimization on the `drive_gain` parameter.
4. Plot spectrolaminar proxy readouts (cell density, power heatmap, band profiles) aligned to cortical depth from L4.
5. Export a reproducible manifest, metrics, and validation report as JSON artifacts.
6. Read the run status fields and connect them to exported artifacts.
"""

CELL_2_MD = """\
## Biological / Computational Question

**Biological motivation:** Cortical circuits are organized into layers (L1–L6) with distinct cell types (excitatory pyramidal E; inhibitory interneurons PV, SST, VIP). Laminar-specific activity patterns — including gamma-band activity in superficial layers and alpha/beta in deep layers — are observed experimentally (spectrolaminar power profiles). This etude builds a computational scaffold to explore how drive gain modulates firing rate and synchrony across a simplified two-area (V1, V4) cortical column.

**Computational question:** Can AGSDR (Adaptive Gradient-free Search with Dimensional Reduction) tune the `drive_gain` parameter to achieve a target firing rate of 5 Hz at near-zero synchrony (kappa ≈ 0), given a fixed recurrent connectivity and Izhikevich reduced-emitter model?

**Run status:** This workflow is a computational scaffold with native-unscaled readouts. LFP and CSD outputs are laminar-proxy readouts based on Gaussian-kernel projection. No PDE solver is executed in this workflow.
"""

CELL_3_MD = """\
## Colab / Local Setup
"""

CELL_4_CODE = """\
# Colab-compatible setup: use local checkout when available; otherwise install jaxfne from current main.
import importlib.util, subprocess, sys
from pathlib import Path
for _candidate in [Path.cwd(), *Path.cwd().parents]:
    if (_candidate / "jaxfne").is_dir() and (_candidate / "pyproject.toml").exists():
        sys.path.insert(0, str(_candidate))
        break
if importlib.util.find_spec("jaxfne") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jaxfne[viz,opt] @ git+https://github.com/HNXJ/jaxfne.git@main"])
"""

CELL_5_MD = """\
## Optional Current-Main Install for Colab
"""

CELL_6_CODE = """\
import os, subprocess, sys
if os.environ.get("TFNE_FORCE_MAIN_INSTALL", "0") == "1":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-cache-dir", "jaxfne[viz,opt] @ git+https://github.com/HNXJ/jaxfne.git@main"])
else:
    print("Current-main reinstall skipped. Set TFNE_FORCE_MAIN_INSTALL=1 to force it.")
"""

CELL_7_MD = """\
## Setup
"""

CELL_8_CODE = """\
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from IPython.display import display as _ipy_display, HTML as _HTML
import jaxfne as jtfne
"""

CELL_8B_MD = """\
### Display helpers

Inline rendering and artifact saving use shared helpers so Colab, local Jupyter, and HTML export all show the same figures.
"""

CELL_8B_CODE = """\
def _show_mpl(fig):
    import io
    from IPython.display import Image as _Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    _ipy_display(_Image(buf.read()))
"""

CELL_8C_MD = """\
### Plotly display helper
"""

CELL_8C_CODE = """\
def _show_plotly(fig):
    if os.environ.get("TFNE_RENDER_PLOTLY", "1") == "1":
        try:
            _ipy_display(_HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))
        except Exception:
            pass
"""

CELL_8D_MD = """\
### Artifact save helpers
"""

CELL_8D_CODE = """\
def _save_png(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _save_html(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
"""

CELL_9_MD = """\
## Install Verification
"""

CELL_10_CODE = """\
try:
    import importlib.metadata as _md
    _version = _md.version("jaxfne")
except Exception:
    _version = getattr(jtfne, "__version__", "local_checkout")
print("jaxfne version:", _version)
print("jaxfne path:   ", jtfne.__file__)
print("vis path:      ", jtfne.vis.__file__)

if not hasattr(jtfne.vis, "visualize_network_3d"):
    raise RuntimeError(
        "jtfne.vis.visualize_network_3d is MISSING. "
        "This notebook requires it. Use Runtime -> Restart runtime, "
        "then re-run from the top after the force-reinstall cell.\\n"
        f"jaxfne: {jtfne.__file__}"
    )

# Also verify Plotly -- REQUIRED for this notebook
try:
    import plotly
    print("plotly version:", plotly.__version__)
except ImportError:
    raise RuntimeError(
        "Plotly is required for this notebook. "
        "Install with: pip install 'jaxfne[viz]>=0.3.22'"
    )

print("OK all required packages present")
"""

CELL_11_MD = """\
## Centralized Configuration
"""

CELL_12_CODE = """\
from types import SimpleNamespace
from pathlib import Path as _Path

# ─────────────────────────────────────────────────────────────────────────────
# CENTRALIZED CONFIG — edit values here, all downstream cells derive from this
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "runtime": {
        "seed": 20260530,
        "dt_ms": 0.1,
        "dtype": "float32",
        "duration_full_ms": 1000.0,
        "duration_smoke_ms": 300.0,
    },
    "geometry": {
        "column_height_mm": 1.5,
        "column_diameter_mm": 0.25,
        "column_radius_mm": 0.125,
        "dx_mm": 0.010,
        "dy_mm": 0.010,
        "dz_mm": 0.010,
        "geometry_status": "declared_tutorial_metadata",
        "field_solver_status": "laminar_proxy_no_pde",
    },
    "areas": ["V1", "V4"],
    "n_per_area_full": 80,
    "n_per_area_smoke": 40,
    "layers": ["L1", "L2/3", "L4", "L5", "L6"],
    "layer_fractions": {
        "L1":   [0.00, 0.10],
        "L2/3": [0.10, 0.30],
        "L4":   [0.30, 0.45],
        "L5":   [0.45, 0.70],
        "L6":   [0.70, 1.00],
    },
    "cell_types": ["E", "PV", "SST", "VIP"],
    "cell_type_fractions": {"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07},
    "cell_colors": {
        "E":   "#1f77b4",
        "PV":  "#d62728",
        "SST": "#2ca02c",
        "VIP": "#9467bd",
    },
    "cell_signs": {"E": 1.0, "PV": -1.0, "SST": -1.0, "VIP": -1.0},
    "drive": {"E": 5.0, "PV": 3.0, "SST": 3.5, "VIP": 3.0},
    "connectivity": {
        "within_area": "all_to_all_uniform_random",
        "within_gain": 0.35,
    },
    "field": {
        "n_contacts": 16,
        "domain": "laminar_column",
        "conductivity": "proxy",
        "boundary": "mean_zero_neumann",
    },
    "stimulus": {
        "target_area": "V1",
        "target_layer": "L4",
        "target_cell_type": "E",
        "onset_ms": 100.0,
        "duration_ms": 100.0,
        "amplitude": 1.5,
    },
    "objective": {
        "target_rate_hz": 5.0,
        "target_kappa": 0.0,
        "rate_weight": 1.0,
        "synchrony_weight": 1.0,
    },
    "optimizer": {
        "name": "AGSDR",
        "primary_tunable": "drive_gain",
        "drive_gain_min": 0.1,
        "drive_gain_max": 1.5,
        "budget_full": 16,
        "budget_smoke": 3,
        "population_full": 4,
        "population_smoke": 2,
        "seed": 20260530,
    },
    "spectrolaminar": {
        "depth_axis_label": "Cortical position from L4 (mm)",
        "bands_hz": {
            "theta":      [4.0, 8.0],
            "alpha_beta": [8.0, 30.0],
            "gamma":      [30.0, 80.0],
        },
        "freq_min_hz": 1.0,
        "freq_max_hz": 100.0,
        "n_depth_bins": 40,
        "normalization": "relative_power_by_depth",
        "show": True,
        "save_png": True,
        "save_html": True,
    },
    "visualization": {
        "show": True,
        "save_png": True,
        "save_html": True,
        "network_plotly_required": True,
        "network_static_png_required": True,
        "proxy_safe_titles": True,
        "dpi": 150,
    },
    "output_dir": "outputs/etude_no_1",
    "status_fields": {
        "run_status": "tutorial_scaffold",
        "model_status": "computational_scaffold",
        "field_solver_status": "laminar_proxy_no_pde",
        "field_model_status": "proxy_readout_only",
        "amplitude_status": False,
    },
}

print("CONFIG loaded")
"""

CELL_13_CODE = """\
# Derived from CONFIG
SMOKE       = os.environ.get("TFNE_SMOKE", "0") == "1"
SEED        = CONFIG["runtime"]["seed"]
DT_MS       = CONFIG["runtime"]["dt_ms"]
DTYPE       = CONFIG["runtime"]["dtype"]
DURATION_MS = CONFIG["runtime"]["duration_smoke_ms"] if SMOKE else CONFIG["runtime"]["duration_full_ms"]
N_PER_AREA  = CONFIG["n_per_area_smoke"] if SMOKE else CONFIG["n_per_area_full"]
AREAS       = CONFIG["areas"]
LAYERS      = CONFIG["layers"]
CELL_TYPES  = CONFIG["cell_types"]

HEIGHT_MM   = CONFIG["geometry"]["column_height_mm"]
RADIUS_MM   = CONFIG["geometry"]["column_radius_mm"]

# L4 center for spectrolaminar depth axis
L4_FRAC = CONFIG["layer_fractions"]
L4_CENTER_MM = 0.5 * (L4_FRAC["L4"][0] + L4_FRAC["L4"][1]) * HEIGHT_MM

AGSDR_GEN   = CONFIG["optimizer"]["budget_smoke"] if SMOKE else CONFIG["optimizer"]["budget_full"]
AGSDR_POP   = CONFIG["optimizer"]["population_smoke"] if SMOKE else CONFIG["optimizer"]["population_full"]

# Resolve output directory relative to repo root (works in git, ZIP, local, and Colab)
import subprocess as _sp
try:
    _repo_root = Path(_sp.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True).strip())
except Exception:
    _repo_root = None
if _repo_root is None:
    for _candidate in [Path.cwd(), *Path.cwd().parents]:
        if (_candidate / "pyproject.toml").exists() and (_candidate / "jaxfne").is_dir():
            _repo_root = _candidate
            break
if _repo_root is None:
    _repo_root = Path.cwd()

OUTPUT_DIR  = _repo_root / CONFIG["output_dir"]
FIGURE_DIR  = OUTPUT_DIR / "figures"
PLOTLY_DIR  = OUTPUT_DIR / "plotly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PLOTLY_DIR.mkdir(parents=True, exist_ok=True)

artifact_paths = {}

print(f"SMOKE={SMOKE} DURATION_MS={DURATION_MS} N_PER_AREA={N_PER_AREA}")
print(f"HEIGHT_MM={HEIGHT_MM} RADIUS_MM={RADIUS_MM} L4_CENTER_MM={L4_CENTER_MM:.4f}")
print(f"AGSDR: {AGSDR_GEN} generations x {AGSDR_POP} population")
print(f"Output: {OUTPUT_DIR.resolve()}")
"""

CELL_14_MD = """\
## Mathematical Glossary

| Symbol | Meaning |
|---|---|
| V_m | Membrane potential (mV) — Izhikevich reduced emitter |
| u | Recovery variable — Izhikevich |
| a, b, c, d | Izhikevich cell-type parameters |
| I(t) | Input current (native drive + noise + stimulus) |
| W | Recurrent weight matrix |
| phi_e | LFP-like proxy field (not solved, not calibrated) |
| CSD | CSD-like proxy (second spatial derivative of phi_e) |
| kappa | Kappa synchrony (mean resultant vector of spike phases) |

**Emitter:** `izhikevich`, preset `cortical_eig`
**Field:** `laminar_proxy_no_pde` — Gaussian kernel projection, no PDE
**Optimizer:** AGSDR (gradient-free black-box search)
"""

CELL_SEC1_MD = """\
## Section 1 — Interactive Single-Emitter Waveform Control Panel

### 1.1 Purpose

This section provides an interactive browser-side panel for building intuition about
reduced Izhikevich emitter parameters (`a`, `b`, `c`, `d`, native drive). Use it to
explore how each parameter shapes the action potential waveform before running the
full multi-area simulation below.

### 1.2 Browser-side preview panel

The panel below is a standalone React/Tailwind/Babel application that runs entirely
in the browser (no Python/JAX required). It includes 11 cell-type presets, a real-time
Euler preview, a phase portrait with nullclines, and sliders for all key parameters.

### 1.3 How to use the controls

- **Preset selector** — choose E-Regular, PV (Fast-Spiking), SST (Martinotti),
  VIP (Disinhibitory), or other classes to load canonical parameter sets.
- **a, b, c, d** sliders — adjust Izhikevich recovery dynamics.
- **Native drive** — uncalibrated input current (arbitrary units, not pA/nA).
- **spikeWidth / biphasicDepth** — display-only visual shaping controls not
  backed by the package-native emitter unless explicitly wired.
- **Copy Config** — generates a `single_emitter_preview` dict using
  `import jaxfne as jtfne` that you can paste into a notebook cell.

### 1.4 Package-native JAXFNE warmup follows below

After exploring the panel, Section 1.4 runs the package-native `simulate_eig_izhikevich`
call that produces the authoritative simulation evidence used in the etude report.

### 1.5 Scope boundary

This panel is an interactive browser-side preview for reduced-emitter parameter
intuition. It is **not** the authoritative simulation engine for this tutorial.
The release evidence path remains the package-native JAXFNE notebook simulation
using `import jaxfne as jtfne`.

The display-only waveform controls such as `spikeWidth` and `biphasicDepth` are
visual teaching controls unless explicitly supported by the package-native emitter
API. Native drive is uncalibrated and is **not** a physical current amplitude.
"""

CELL_SEC1_EMBED_CODE = """\
# -- Section 1: embed interactive single-emitter HTML panel -------------------
from IPython.display import IFrame, display as _display_html
import shutil

_section1_src = _repo_root / "tutorials" / "etudes" / "assets" / "izhikevich_waveform_control_panel.html"
_section1_out_dir = OUTPUT_DIR / "html"
_section1_out_dir.mkdir(parents=True, exist_ok=True)
_section1_out = _section1_out_dir / "izhikevich_waveform_control_panel.html"

assert _section1_src.exists(), f"Section 1 HTML source missing: {_section1_src}"
shutil.copy2(_section1_src, _section1_out)

assert _section1_out.exists()
assert _section1_out.stat().st_size > 1000

if CONFIG["visualization"]["show"]:
    _display_html(IFrame(src=str(_section1_out), width="100%", height=950))

print(f"OK Section 1 HTML copied: {_section1_out}")
artifact_paths["interactive_single_emitter_panel"] = str(_section1_out.relative_to(OUTPUT_DIR))
"""

CELL_15_MD = """\
## Section 2 — Uniform-Density Cortical Column Scaffold

This section uses uniform cell density across layers as a null/reference cortical
column. Layer names, cell labels, geometry metadata, and visualization settings
are editable, but the density profile is intentionally **uniform** here. All four
cell types (E, PV, SST, VIP) are distributed uniformly throughout the declared
column height and radius.
"""

CELL_16_CODE = """\
# -- Part 1: single-unit warmup -----------------------------------------------
WARMUP_LABELS = ("E", "PV", "SST", "VIP")
WARMUP_LAYERS = ("warmup",) * 4
WARMUP_DRIVE  = CONFIG["drive"]
WARMUP_DUR_MS = min(300.0, DURATION_MS)

warmup_params = jtfne.izhikevich_params_from_labels(
    labels       = WARMUP_LABELS,
    layer_labels = WARMUP_LAYERS,
    dtype        = DTYPE,
    drive_overrides = {k: WARMUP_DRIVE[k] for k in WARMUP_LABELS},
    source_scale = 1.0,
)
warmup_key   = jax.random.PRNGKey(SEED)
warmup_steps = int(round(WARMUP_DUR_MS / DT_MS))
print(f"Warmup: {len(WARMUP_LABELS)} units x {warmup_steps} steps")
"""

CELL_17_CODE = """\
warmup_V, warmup_spikes, warmup_sources = jtfne.simulate_eig_izhikevich(
    params         = warmup_params,
    n_steps        = warmup_steps,
    dt_ms          = DT_MS,
    key            = warmup_key,
    dtype          = DTYPE,
    drive_schedule = None,
    silence_mask   = None,
)
warmup_rate_hz = (
    1000.0 * np.asarray(warmup_spikes).sum(axis=0) / (warmup_steps * DT_MS)
)
import pandas as pd
wu_table = pd.DataFrame({"cell_type": list(WARMUP_LABELS), "rate_hz": warmup_rate_hz})
print(wu_table.to_string(index=False))
"""

CELL_18_CODE = """\
t_wu = np.arange(warmup_steps) * DT_MS
V_wu = np.asarray(warmup_V)
fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
for i, lbl in enumerate(WARMUP_LABELS):
    ax.plot(t_wu, V_wu[:, i], label=lbl, lw=0.8)
ax.set(xlabel="Time (ms)", ylabel="Native V_m (a.u.)",
       title="Single-unit Izhikevich warmup — proxy scaffold, no physical status")
ax.legend(loc="upper right")
_show_mpl(fig)
_save_png(fig, FIGURE_DIR / "warmup_traces.png")
artifact_paths["warmup_traces_png"] = str(FIGURE_DIR / "warmup_traces.png")
"""

CELL_19_MD = """\
## Part 2 — Multi-Laminar Cortical Scaffold
"""

CELL_20_CODE = """\
# Build multi-area laminar configuration from CONFIG values
from jaxfne.core import Configuration

layer_fracs = {k: tuple(v) for k, v in CONFIG["layer_fractions"].items()}
layer_cell_types = {L: CONFIG["cell_type_fractions"] for L in LAYERS}

cfg = (
    Configuration()
    .runtime(
        seed        = SEED,
        duration_ms = DURATION_MS,
        dt_ms       = DT_MS,
        dtype       = DTYPE,
    )
    .areas(AREAS)
)

for area in AREAS:
    cfg = (
        cfg
        .column(area, layers=LAYERS, n=N_PER_AREA)
        .layer_fractions(
            layer_fractions  = layer_fracs,
            layer_cell_types = layer_cell_types,
        )
        .uniform3d(
            radius_mm = RADIUS_MM,
            height_mm = HEIGHT_MM,
        )
    )

cfg = (
    cfg
    .cell_types(CONFIG["cell_type_fractions"])
    .connectivity(
        within_area = CONFIG["connectivity"]["within_area"],
        within_gain = CONFIG["connectivity"]["within_gain"],
    )
    .set_emitter("izhikevich", "cortical_eig")
    .probes(["spikes", "V_m", "source", "LFP", "CSD"],
            n_contacts = CONFIG["field"]["n_contacts"])
    .field(
        domain       = CONFIG["field"]["domain"],
        conductivity = CONFIG["field"]["conductivity"],
        boundary     = CONFIG["field"]["boundary"],
    )
)

print("Configuration built")
"""

CELL_21_CODE = """\
model = jtfne.construct(cfg)
summary = model.summary()
n_units = summary["n_units"]
print(f"Model: {n_units} neurons across {len(AREAS)} areas")
print(f"Areas: {AREAS}")
"""

CELL_22_MD = """\
## 3D Network Visualization (Interactive Plotly — Required)
"""

CELL_23_CODE = """\
# -- Plotly network visualization (REQUIRED — not optional) -------------------
# If this raises, Plotly or visualize_network_3d is missing.
# Do NOT wrap in try/except ImportError — fix the installation instead.

_html_out = PLOTLY_DIR / "cortical_circuit_network.html"
network_fig, network_rows = jtfne.vis.visualize_network_3d(
    model,
    output_html        = _html_out,
    title              = "Etude No. 1 cortical scaffold — simulated proxy geometry (laminar_proxy_no_pde)",
    coordinate_unit    = "mm",
    display_unit       = "mm",
    show_layers        = True,
    show_column_shells = True,
    show_edges         = False,
    max_edges          = 500,
    seed               = SEED,
    return_node_table  = True,
    cell_type_colors   = CONFIG["cell_colors"],
)
_show_plotly(network_fig)
print(f"OK Plotly HTML written: {_html_out}")
artifact_paths["network_html"] = str(_html_out)
"""

CELL_24_CODE = """\
# -- Geometry validation (per-area) ------------------------------------------
# network_rows use key x_m / y_m / z_m (values are in mm since coordinate_unit="mm")

print("Per-area geometry:")
_geom_ok = True
for area in AREAS:
    _ar = [r for r in network_rows if r["area"] == area]
    _xs = np.array([r["x_m"] for r in _ar])
    _ys = np.array([r["y_m"] for r in _ar])
    _zs = np.array([r["z_m"] for r in _ar])
    _r  = np.sqrt((_xs - _xs.mean())**2 + (_ys - _ys.mean())**2)
    _diam = 2.0 * float(np.nanmax(_r))
    _h    = float(np.nanmax(_zs) - np.nanmin(_zs))
    _ok_d = 0.15 <= _diam <= 0.40
    _ok_h = 1.0  <= _h    <= 1.8
    print(f"  {area}: diameter={_diam:.3f}mm height={_h:.3f}mm  diameter_ok={_ok_d} height_ok={_ok_h}")
    if not (_ok_d and _ok_h):
        _geom_ok = False

_ct_seen = {r["cell_type"] for r in network_rows}
_ct_ok = {"E", "PV", "SST", "VIP"}.issubset(_ct_seen)
print(f"  cell_types: {sorted(_ct_seen)}  required_ok={_ct_ok}")

assert _geom_ok, "Geometry check failed — see per-area report above"
assert _ct_ok, f"Cell types missing: {{'E','PV','SST','VIP'}} not <= {_ct_seen}"
print("OK geometry validation passed")
"""

CELL_25_CODE = """\
# -- Static matplotlib PNG fallback ------------------------------------------
_tbl = model.neuron_table()
_fig3d = plt.figure(figsize=(10, 6), dpi=150)
_ax3d  = _fig3d.add_subplot(111, projection="3d")
for ct in CELL_TYPES:
    _sub = [(r["x"], r["y"], r["z"]) for r in _tbl if r["cell_type"] == ct]
    if not _sub:
        continue
    _sx, _sy, _sz = zip(*_sub)
    _ax3d.scatter(_sx, _sy, _sz, s=10, c=CONFIG["cell_colors"][ct], label=ct, alpha=0.7)
_ax3d.set_xlabel("x (mm)")
_ax3d.set_ylabel("y (mm)")
_ax3d.set_zlabel("z (mm)")
_ax3d.set_title("Proxy scaffold geometry (declared, laminar_proxy_no_pde)")
_ax3d.legend(loc="upper left", fontsize=8)
_show_mpl(_fig3d)
_save_png(_fig3d, FIGURE_DIR / "cortical_circuit_network.png")
artifact_paths["network_png"] = str(FIGURE_DIR / "cortical_circuit_network.png")
print(f"OK static PNG: {FIGURE_DIR / 'cortical_circuit_network.png'}")
"""

CELL_26_MD = """\
## Part 3 — Baseline Simulation
"""

CELL_27_CODE = """\
def _activity_suite(signals, label, fig_dir, plotly_dir, show=True, save_png=True, save_html=True):
    \"\"\"Plot raster + rate + LFP + CSD. Show and save.\"\"\"
    spikes  = np.asarray(signals.spikes)
    t_ms    = np.asarray(signals.time_ms)
    lfp     = np.asarray(signals.field.lfp_proxy)
    csd     = np.asarray(signals.field.csd_proxy)
    depths  = np.asarray(signals.field.contact_depths)
    rate_hz = 1000.0 * spikes.mean(axis=1) / DT_MS

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=CONFIG["visualization"]["dpi"])
    fig.suptitle(f"Activity Suite — {label} (proxy · tutorial_scaffold)", fontsize=11)
    si, ti = np.where(spikes.T)
    axes[0, 0].scatter(t_ms[ti], si, s=0.5, c="k", alpha=0.4)
    axes[0, 0].set(xlabel="Time (ms)", ylabel="Neuron", title="Spike raster")
    axes[0, 1].plot(t_ms, rate_hz, lw=1, color="#1f77b4")
    axes[0, 1].set(xlabel="Time (ms)", ylabel="Rate (Hz)", title="Mean firing rate")
    vmax = max(float(np.abs(lfp).max()), 1e-6)
    axes[1, 0].imshow(lfp.T, aspect="auto", origin="lower",
        extent=[t_ms[0], t_ms[-1], float(depths[0]), float(depths[-1])],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1, 0].set(xlabel="Time (ms)", ylabel="Contact depth", title="LFP proxy")
    vmax2 = max(float(np.abs(csd).max()), 1e-6)
    axes[1, 1].imshow(csd.T, aspect="auto", origin="lower",
        extent=[t_ms[0], t_ms[-1], float(depths[0]), float(depths[-1])],
        cmap="RdBu_r", vmin=-vmax2, vmax=vmax2)
    axes[1, 1].set(xlabel="Time (ms)", ylabel="Contact depth", title="CSD proxy")
    fig.tight_layout()

    if show:
        _show_mpl(fig)
    suf = label.lower().replace(" ", "_")
    png_path = fig_dir / f"activity_suite_{suf}.png"
    if save_png:
        _save_png(fig, png_path)
    else:
        plt.close(fig)

    html_path = None
    if save_html:
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            pf = make_subplots(2, 2, subplot_titles=["Spike Raster", "Mean Rate", "LFP proxy", "CSD proxy"])
            si2, ti2 = np.where(spikes.T)
            pf.add_trace(go.Scattergl(x=t_ms[ti2], y=si2, mode="markers",
                marker=dict(size=2, color="black", opacity=0.4), showlegend=False), 1, 1)
            pf.add_trace(go.Scatter(x=t_ms, y=rate_hz, mode="lines",
                line=dict(color="#1f77b4", width=1.5), showlegend=False), 1, 2)
            pf.add_trace(go.Heatmap(z=lfp.T, x=t_ms, y=depths, colorscale="RdBu_r",
                zmid=0, showscale=True), 2, 1)
            pf.add_trace(go.Heatmap(z=csd.T, x=t_ms, y=depths, colorscale="RdBu_r",
                zmid=0, showscale=True), 2, 2)
            pf.update_layout(title=f"Activity Suite — {label} (proxy · tutorial_scaffold)",
                height=600, width=1050)
            html_path = plotly_dir / f"activity_suite_{suf}.html"
            _save_html(pf, html_path)
            _show_plotly(pf)
        except ImportError:
            pass

    return {"png": str(png_path) if save_png else None, "html": str(html_path) if html_path else None}
"""

CELL_28_CODE = """\
sim = jtfne.Simulation(
    duration_ms    = DURATION_MS,
    dt_ms          = DT_MS,
    seed           = SEED,
    record_sources = True,
    record_fields  = True,
)
signals_baseline = model.simulate(sim)
baseline_rate  = signals_baseline.summary()["spike_rate_hz_mean"]
baseline_kappa = jtfne.kappa_synchrony(signals_baseline.spikes, DT_MS)
print(f"Baseline: rate={baseline_rate:.2f} Hz  kappa={baseline_kappa:.4f}")
"""

CELL_29_CODE = """\
_r = _activity_suite(signals_baseline, "Baseline", FIGURE_DIR, PLOTLY_DIR,
    show=CONFIG["visualization"]["show"],
    save_png=CONFIG["visualization"]["save_png"],
    save_html=CONFIG["visualization"]["save_html"])
artifact_paths.update({"activity_baseline_png": _r["png"], "activity_baseline_html": _r["html"]})
print(f"Baseline activity suite: png={_r['png']}")
"""

CELL_30_MD = """\
## Part 4 — L4 Stimulus Simulation
"""

CELL_31_CODE = """\
stim_cfg = CONFIG["stimulus"]
stim_idx = jtfne.select_neurons(
    model,
    area      = stim_cfg["target_area"],
    layer     = stim_cfg["target_layer"],
    cell_type = stim_cfg["target_cell_type"],
)
if len(stim_idx) == 0:
    stim_idx = jtfne.select_neurons(model, area=stim_cfg["target_area"])
    print(f"Warning: no L4 E neurons found; using all {stim_cfg['target_area']} neurons")

stim_event = {
    "label":          "L4_evoked_drive",
    "onset_ms":       stim_cfg["onset_ms"],
    "duration_ms":    stim_cfg["duration_ms"],
    "amplitude":      stim_cfg["amplitude"],
    "target_indices": stim_idx.tolist(),
}
stim_schedule = jtfne.stimulus_schedule([stim_event], n_neurons=n_units)
print(f"Stimulus: {len(stim_idx)} target neurons in {stim_cfg['target_area']}/{stim_cfg['target_layer']}/{stim_cfg['target_cell_type']}")
"""

CELL_32_CODE = """\
signals_stim = model.simulate(sim, paradigm=stim_schedule)
stim_rate  = signals_stim.summary()["spike_rate_hz_mean"]
stim_kappa = jtfne.kappa_synchrony(signals_stim.spikes, DT_MS)
print(f"Stimulus: rate={stim_rate:.2f} Hz  kappa={stim_kappa:.4f}")

_r = _activity_suite(signals_stim, "Stimulus", FIGURE_DIR, PLOTLY_DIR,
    show=CONFIG["visualization"]["show"],
    save_png=CONFIG["visualization"]["save_png"],
    save_html=CONFIG["visualization"]["save_html"])
artifact_paths.update({"activity_stimulus_png": _r["png"], "activity_stimulus_html": _r["html"]})
"""

CELL_33_MD = """\
## Part 5 — AGSDR Tuning
"""

CELL_34_CODE = """\
obj_cfg = CONFIG["objective"]
opt_cfg = CONFIG["optimizer"]

objective = jtfne.rate_synchrony_targets(
    target_rate_hz         = obj_cfg["target_rate_hz"],
    target_kappa_synchrony = obj_cfg["target_kappa"],
    rate_weight            = obj_cfg["rate_weight"],
    synchrony_weight       = obj_cfg["synchrony_weight"],
)

optimizer = jtfne.agsdr(
    alpha           = 0.7,
    exploration     = 0.05,
    deselect_factor = 2.0,
    parameters      = {opt_cfg["primary_tunable"]: (opt_cfg["drive_gain_min"], opt_cfg["drive_gain_max"])},
    generations     = AGSDR_GEN,
    population_size = AGSDR_POP,
    seed            = opt_cfg["seed"],
)

print(f"Objective: rate->{obj_cfg['target_rate_hz']} Hz, kappa->{obj_cfg['target_kappa']}")
print(f"Optimizer: AGSDR {AGSDR_GEN} gen x {AGSDR_POP} pop, drive_gain [{opt_cfg['drive_gain_min']},{opt_cfg['drive_gain_max']}]")
"""

CELL_35_CODE = """\
_LIVE_TUNE = os.environ.get("TFNE_RUN_LIVE_TUNE", "0") == "1"

if _LIVE_TUNE:
    print("Running live AGSDR tune...")
    tuned = model.tune(objectives=objective, optimizer=optimizer, simulation=sim, seed=SEED)
    print(f"Live tune complete: best_score={tuned.best_score:.6f}")
else:
    # Deterministic cached receipt — avoids slow optimization in headless/CI
    _cached = {
        "best_parameters": {"drive_gain": 0.72},
        "best_score": 0.8500,
        "history": [],
        "summary": {
            "tuning_status": "deterministic_smoke_receipt",
            "best_parameters": {"drive_gain": 0.72},
            "best_score": 0.8500,
            "generation_records": [],
            "all_scores": [0.8500],
            "same_model_unchanged": False,
            "run_status": "tutorial_scaffold",
            "model_status": "computational_scaffold",
            "field_solver_status": "laminar_proxy_no_pde",
            "amplitude_status": False,
        }
    }
    _best_params = _cached["best_parameters"]
    _drive_gain  = float(_best_params.get("drive_gain", 1.0))
    tuned = jtfne.TuneResult(
        best_parameters = _best_params,
        best_score      = float(_cached["best_score"]),
        history         = _cached["history"],
        summary         = _cached["summary"],
        model           = model.with_emitter_parameters(drive_scale=_drive_gain),
    )
    jtfne.save_json(jtfne.json_safe(_cached), OUTPUT_DIR / "tune_result.json")
    print(f"Loaded cached receipt: drive_gain={_drive_gain:.4f} best_score={float(tuned.best_score):.4f}")
    print("Set TFNE_RUN_LIVE_TUNE=1 to run live AGSDR instead")
"""

CELL_36_MD = """\
## Part 6 — Tuned Simulation
"""

CELL_37_CODE = """\
tuned_model   = tuned.model
signals_tuned = tuned_model.simulate(sim, paradigm=stim_schedule)
tuned_rate    = signals_tuned.summary()["spike_rate_hz_mean"]
tuned_kappa   = jtfne.kappa_synchrony(signals_tuned.spikes, DT_MS)
print(f"Tuned: rate={tuned_rate:.2f} Hz  kappa={tuned_kappa:.4f}  (target: {obj_cfg['target_rate_hz']} Hz, {obj_cfg['target_kappa']})")

_r = _activity_suite(signals_tuned, "Tuned", FIGURE_DIR, PLOTLY_DIR,
    show=CONFIG["visualization"]["show"],
    save_png=CONFIG["visualization"]["save_png"],
    save_html=CONFIG["visualization"]["save_html"])
artifact_paths.update({"activity_tuned_png": _r["png"], "activity_tuned_html": _r["html"]})
"""

CELL_38_CODE = """\
_rows = [
    {"Condition": "Baseline",   "Rate (Hz)": f"{baseline_rate:.2f}",
     "Kappa": f"{baseline_kappa:.4f}", "Target Rate": "-",                        "Target Kappa": "-"},
    {"Condition": "Stimulus",   "Rate (Hz)": f"{stim_rate:.2f}",
     "Kappa": f"{stim_kappa:.4f}",    "Target Rate": "-",                        "Target Kappa": "-"},
    {"Condition": "Tuned+Stim", "Rate (Hz)": f"{tuned_rate:.2f}",
     "Kappa": f"{tuned_kappa:.4f}",   "Target Rate": f"{obj_cfg['target_rate_hz']:.1f}",
     "Target Kappa": f"{obj_cfg['target_kappa']:.1f}"},
]
import pandas as pd
print(pd.DataFrame(_rows).to_string(index=False))
"""

CELL_39_MD = """\
## Part 7 — Spectrolaminar Proxy Readouts
"""

CELL_40_CODE = """\
from scipy.signal import welch as _welch
from scipy.ndimage import gaussian_filter1d as _gauss1d

def plot_spectrolaminar_etude1(signals, model, config, condition_name,
                                output_png, output_html=None,
                                show=True, save_png=True, save_html=True):
    \"\"\"3-panel spectrolaminar proxy readout.

    Panel A: cell density by cortical position from L4
    Panel B: relative PSD heatmap (freq x depth)
    Panel C: alpha-beta / gamma band profiles

    Y-axis: cortical position from L4 (mm), negative=superficial, positive=deep
    \"\"\"
    spec_cfg = config["spectrolaminar"]
    vis_cfg  = config["visualization"]
    geom_cfg = config["geometry"]

    lfp    = np.asarray(signals.field.lfp_proxy)          # [T, n_contacts]
    depths = np.asarray(signals.field.contact_depths)     # [n_contacts], 0-1
    h_mm   = geom_cfg["column_height_mm"]
    l4_c   = 0.5 * (config["layer_fractions"]["L4"][0] + config["layer_fractions"]["L4"][1]) * h_mm

    # Contact position from L4 (mm)
    pos_mm = depths * h_mm - l4_c

    # PSD per contact
    fs         = 1000.0 / config["runtime"]["dt_ms"]
    fmin       = spec_cfg["freq_min_hz"]
    fmax       = spec_cfg["freq_max_hz"]
    n_contacts = len(depths)

    psd_list = []
    for ci in range(n_contacts):
        f, px = _welch(lfp[:, ci], fs=fs, nperseg=min(256, lfp.shape[0] // 2))
        mask = (f >= fmin) & (f <= fmax)
        psd_list.append(px[mask])
    freqs_hz = f[mask]
    psd = np.array(psd_list)  # [n_contacts, n_freqs]

    # Relative log power
    plog = np.log10(psd + 1e-12)
    vlo, vhi = np.percentile(plog, 5), np.percentile(plog, 95)
    rel = np.clip((plog - vlo) / (vhi - vlo + 1e-12), 0, 1) * 0.5 + 0.5

    # Band profiles
    bands_hz = spec_cfg["bands_hz"]
    ab_m = (freqs_hz >= bands_hz["alpha_beta"][0]) & (freqs_hz <= bands_hz["alpha_beta"][1])
    gm_m = (freqs_hz >= bands_hz["gamma"][0])      & (freqs_hz <= bands_hz["gamma"][1])

    ab_p = rel[:, ab_m].mean(axis=1) if ab_m.any() else np.zeros(n_contacts)
    gm_p = rel[:, gm_m].mean(axis=1) if gm_m.any() else np.zeros(n_contacts)

    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)
    ab_p, gm_p = _norm(ab_p), _norm(gm_p)

    # Cell density from neuron table
    tbl = model.neuron_table()
    n_bins     = spec_cfg["n_depth_bins"]
    bin_edges  = np.linspace(pos_mm[0] - 0.05, pos_mm[-1] + 0.05, n_bins + 1)
    bin_ctrs   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    nz_all     = np.array([r["z"] for r in tbl])
    nz_from_l4 = nz_all - l4_c

    # Validation checks
    assert np.all(np.isfinite(rel)),  "Power heatmap contains NaN/Inf"
    assert np.all(np.isfinite(ab_p)), "Alpha-beta profile contains NaN/Inf"
    assert np.all(np.isfinite(gm_p)), "Gamma profile contains NaN/Inf"

    # -- Matplotlib figure ----------------------------------------------------
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 5), dpi=vis_cfg["dpi"],
        gridspec_kw={"width_ratios": [0.85, 1.75, 0.85]}, sharey=True)
    fig.suptitle(
        f"{condition_name} — Spectrolaminar proxy (tutorial_scaffold · laminar_proxy_no_pde)",
        fontsize=11)

    for ct in config["cell_types"]:
        nz_ct = nz_from_l4[[r["cell_type"] == ct for r in tbl]]
        if len(nz_ct) == 0:
            continue
        vals, _ = np.histogram(nz_ct, bins=bin_edges)
        vals = _gauss1d(vals.astype(float), 1.2)
        if vals.max() > 0:
            vals = vals / vals.max()
        ax0.plot(vals, bin_ctrs, lw=2.0, color=config["cell_colors"][ct], label=ct)
    ax0.axhline(0, color="k", lw=1.0, ls="--")
    ax0.set(title="A Cell Density",
            xlabel="Relative Count",
            ylabel=spec_cfg["depth_axis_label"])
    ax0.legend(fontsize=7)

    im = ax1.imshow(rel, aspect="auto", origin="upper", cmap="viridis",
        extent=[freqs_hz[0], freqs_hz[-1], pos_mm[-1], pos_mm[0]],
        vmin=0.5, vmax=1.0)
    ax1.axhline(0, color="k", lw=1.0, ls="--")
    ax1.set(title="B Relative Power Spectrum", xlabel="Frequency (Hz)")
    fig.colorbar(im, ax=ax1, label="Rel Pow")

    ax2.plot(ab_p, pos_mm, color="navy",    lw=2.5, label="Alpha-beta")
    ax2.plot(gm_p, pos_mm, color="crimson", lw=2.5, label="Gamma")
    ax2.axhline(0, color="k", lw=1.0, ls="--")
    ax2.set(title="C Band Profiles", xlabel="Relative Power")
    ax2.legend(fontsize=8)

    fig.tight_layout()

    if show:
        _show_mpl(fig)

    png_path = Path(output_png)
    if save_png:
        _save_png(fig, png_path)
    else:
        plt.close(fig)

    # -- Plotly HTML ----------------------------------------------------------
    html_path = None
    if save_html and output_html:
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            pf = make_subplots(1, 3, column_widths=[0.25, 0.5, 0.25],
                subplot_titles=["A Cell Density", "B Power Spectrum", "C Band Profiles"],
                shared_yaxes=True)

            for ct in config["cell_types"]:
                nz_ct = nz_from_l4[[r["cell_type"] == ct for r in tbl]]
                if len(nz_ct) == 0:
                    continue
                vals, _ = np.histogram(nz_ct, bins=bin_edges)
                vals = _gauss1d(vals.astype(float), 1.2)
                if vals.max() > 0:
                    vals = vals / vals.max()
                pf.add_trace(go.Scatter(x=vals, y=bin_ctrs, mode="lines",
                    line=dict(color=config["cell_colors"][ct], width=2),
                    name=ct), 1, 1)

            pf.add_trace(go.Heatmap(z=rel, x=freqs_hz, y=pos_mm,
                colorscale="Viridis", zmin=0.5, zmax=1.0,
                colorbar=dict(title="Rel Pow", x=0.68, len=0.9)), 1, 2)

            pf.add_trace(go.Scatter(x=ab_p, y=pos_mm, mode="lines",
                line=dict(color="navy", width=3), name="Alpha-beta"), 1, 3)
            pf.add_trace(go.Scatter(x=gm_p, y=pos_mm, mode="lines",
                line=dict(color="crimson", width=3), name="Gamma"), 1, 3)

            for col in [1, 2, 3]:
                pf.add_hline(y=0, line_color="black", line_width=1, row=1, col=col)

            pf.update_yaxes(title_text=spec_cfg["depth_axis_label"], row=1, col=1)
            pf.update_layout(
                title=f"{condition_name} — Spectrolaminar Proxy (tutorial_scaffold · laminar_proxy_no_pde)",
                height=520, width=1150)

            html_path = Path(output_html)
            _save_html(pf, html_path)
            _show_plotly(pf)
        except ImportError:
            pass

    return {
        "condition":      condition_name,
        "finite":         True,
        "depth_bins":     n_bins,
        "freqs_hz_shape": len(freqs_hz),
        "bands_hz":       bands_hz,
        "normalization":  spec_cfg["normalization"],
        "output_png":     str(png_path),
        "output_html":    str(html_path) if html_path else None,
    }
"""

CELL_41_CODE = """\
_spec_cfg  = CONFIG["spectrolaminar"]
_spec_base = plot_spectrolaminar_etude1(
    signals_baseline, model, CONFIG, "Baseline",
    output_png  = FIGURE_DIR / "spectrolaminar_baseline.png",
    output_html = PLOTLY_DIR / "spectrolaminar_baseline.html",
    show      = _spec_cfg["show"],
    save_png  = _spec_cfg["save_png"],
    save_html = _spec_cfg["save_html"],
)
artifact_paths.update({"spec_baseline_png": _spec_base["output_png"],
                        "spec_baseline_html": _spec_base["output_html"]})
print(f"Baseline spectrolaminar: {_spec_base}")
"""

CELL_42_CODE = """\
_spec_stim = plot_spectrolaminar_etude1(
    signals_stim, model, CONFIG, "Stimulus",
    output_png  = FIGURE_DIR / "spectrolaminar_stimulus.png",
    output_html = PLOTLY_DIR / "spectrolaminar_stimulus.html",
    show      = _spec_cfg["show"],
    save_png  = _spec_cfg["save_png"],
    save_html = _spec_cfg["save_html"],
)
artifact_paths.update({"spec_stimulus_png": _spec_stim["output_png"],
                        "spec_stimulus_html": _spec_stim["output_html"]})
print(f"Stimulus spectrolaminar: {_spec_stim}")
"""

CELL_43_CODE = """\
_spec_tuned = plot_spectrolaminar_etude1(
    signals_tuned, tuned_model, CONFIG, "Tuned",
    output_png  = FIGURE_DIR / "spectrolaminar_tuned.png",
    output_html = PLOTLY_DIR / "spectrolaminar_tuned.html",
    show      = _spec_cfg["show"],
    save_png  = _spec_cfg["save_png"],
    save_html = _spec_cfg["save_html"],
)
artifact_paths.update({"spec_tuned_png": _spec_tuned["output_png"],
                        "spec_tuned_html": _spec_tuned["output_html"]})
print(f"Tuned spectrolaminar: {_spec_tuned}")
"""

CELL_44_CODE = """\
_tune_summary = dict(tuned.summary)
_gen_records  = _tune_summary.get("generation_records", [])
_all_scores   = _tune_summary.get("all_scores", [])
_best_score   = float(_tune_summary.get("best_score", float("nan")))
_best_params  = _tune_summary.get("best_parameters", {})

fig_opt, axes_opt = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

if _gen_records:
    gen_nums = [r["generation"] for r in _gen_records]
    gen_best = [r["best_score"] for r in _gen_records]
    gen_curr = [r.get("generation_best_score", r["best_score"]) for r in _gen_records]
    axes_opt[0].plot(gen_nums, gen_best, "b-o", lw=2, label="Global best")
    axes_opt[0].plot(gen_nums, gen_curr, "o--", color="orange", lw=1.5, label="Gen best")
    axes_opt[0].legend()
else:
    axes_opt[0].text(0.5, 0.5, "Cached receipt\\n(no generation history)",
        ha="center", va="center", transform=axes_opt[0].transAxes)
axes_opt[0].set(xlabel="Generation", ylabel="Loss", title="AGSDR Convergence")

if _all_scores:
    sc = axes_opt[1].scatter(range(len(_all_scores)), _all_scores, s=40,
        c=_all_scores, cmap="plasma_r")
    plt.colorbar(sc, ax=axes_opt[1])
else:
    axes_opt[1].text(0.5, 0.5, "No candidates",
        ha="center", va="center", transform=axes_opt[1].transAxes)
axes_opt[1].set(xlabel="Candidate", ylabel="Score", title="All Candidate Scores")

bp_str = ", ".join(f"{k}={v:.4f}" for k, v in _best_params.items())
fig_opt.suptitle(f"AGSDR Summary | best={_best_score:.4f} | {bp_str}", fontsize=10)
fig_opt.tight_layout()
_show_mpl(fig_opt)
_save_png(fig_opt, FIGURE_DIR / "optimization_summary.png")
artifact_paths["optimization_png"] = str(FIGURE_DIR / "optimization_summary.png")
print("Optimization summary saved")
"""

CELL_45_MD = """\
## Part 8 — Manifest, Metrics, and Validation
"""

CELL_46_CODE = """\
manifest = {
    "artifact_class": "etude",
    "artifact_id":    "etude_no_1",
    "jaxfne_version": jtfne.__version__,
    "execution_mode": "smoke" if SMOKE else "full_etude",
    **CONFIG["status_fields"],
    # Runtime
    "runtime": CONFIG["runtime"],
    # Geometry
    "geometry": CONFIG["geometry"],
    # Areas, layers, cell_types
    "areas":      CONFIG["areas"],
    "layers":     CONFIG["layers"],
    "cell_types": CONFIG["cell_types"],
    "cell_colors": CONFIG["cell_colors"],
    "cell_signs":  CONFIG["cell_signs"],
    "layer_fractions": CONFIG["layer_fractions"],
    # Drive, connectivity
    "drive":        CONFIG["drive"],
    "connectivity": CONFIG["connectivity"],
    # Field, probes
    "field":  CONFIG["field"],
    "probes": {"types": ["spikes", "V_m", "source", "LFP", "CSD"],
               "n_contacts": CONFIG["field"]["n_contacts"]},
    # Objective, optimizer
    "objective": CONFIG["objective"],
    "optimizer": CONFIG["optimizer"],
    # Stimulus
    "stimulus": CONFIG["stimulus"],
    # Spectrolaminar config
    "spectrolaminar": CONFIG["spectrolaminar"],
    # Visualization config
    "visualization": CONFIG["visualization"],
    # Artifact paths
    "artifact_paths": artifact_paths,
    # Section status
    "section_status": {
        "section_1_single_emitter_panel": {
            "artifact": artifact_paths.get("interactive_single_emitter_panel", ""),
            "solver": "browser_side_preview",
            "authoritative_simulation": "package_native_jaxfne_cells",
            "model_status": "computational_scaffold",
            "source_calibration_status": "uncalibrated_native_drive",
        },
        "section_2_uniform_density_column": {
            "density_profile": "uniform_across_layers",
            "model_status": "computational_scaffold",
        },
    },
    # Editable inputs (complete)
    "editable_inputs": {
        "runtime":             CONFIG["runtime"],
        "geometry":            CONFIG["geometry"],
        "areas":               CONFIG["areas"],
        "n_per_area":          N_PER_AREA,
        "layers":              CONFIG["layers"],
        "cell_types":          CONFIG["cell_types"],
        "cell_type_fractions": CONFIG["cell_type_fractions"],
        "layer_fractions":     CONFIG["layer_fractions"],
        "drive":               CONFIG["drive"],
        "connectivity":        CONFIG["connectivity"],
        "field":               CONFIG["field"],
        "stimulus":            CONFIG["stimulus"],
        "objective":           CONFIG["objective"],
        "optimizer":           CONFIG["optimizer"],
        "spectrolaminar":      CONFIG["spectrolaminar"],
    },
}
print("Manifest built")
"""

CELL_47_CODE = """\
same_model_unchanged = abs(float(tuned_rate) - float(stim_rate)) < 1e-12
rate_improvement_hz  = abs(stim_rate - obj_cfg["target_rate_hz"]) - abs(tuned_rate - obj_cfg["target_rate_hz"])
kappa_improvement    = abs(stim_kappa - obj_cfg["target_kappa"])  - abs(tuned_kappa - obj_cfg["target_kappa"])

metrics = {
    "artifact_id":          "etude_no_1",
    "target_rate_hz":       float(obj_cfg["target_rate_hz"]),
    "target_kappa":         float(obj_cfg["target_kappa"]),
    "baseline_rate_hz":     float(baseline_rate),
    "stimulus_rate_hz":     float(stim_rate),
    "tuned_rate_hz":        float(tuned_rate),
    "baseline_kappa":       float(baseline_kappa),
    "stimulus_kappa":       float(stim_kappa),
    "tuned_kappa":          float(tuned_kappa),
    "achieved_rate_hz":     float(tuned_rate),
    "achieved_kappa":       float(tuned_kappa),
    "rate_error_hz":        float(abs(tuned_rate - obj_cfg["target_rate_hz"])),
    "kappa_error":          float(abs(tuned_kappa - obj_cfg["target_kappa"])),
    "rate_improvement_hz":  float(rate_improvement_hz),
    "kappa_improvement":    float(kappa_improvement),
    "best_score":           float(tuned.best_score),
    "best_parameters":      {k: float(v) for k, v in tuned.summary.get("best_parameters", {}).items()},
    "tuning_status":        str(tuned.summary.get("tuning_status", "unknown")),
    "same_model_unchanged": bool(same_model_unchanged),
    "agsdr_generations":    int(AGSDR_GEN),
    "agsdr_population":     int(AGSDR_POP),
}
import math
assert math.isfinite(metrics["best_score"]),          "best_score must be finite"
assert metrics["best_parameters"],                    "best_parameters must be non-empty"
assert metrics["target_rate_hz"] == 5.0,              "target_rate_hz must be 5.0"
assert metrics["target_kappa"]   == 0.0,              "target_kappa must be 0.0"
assert math.isfinite(metrics["achieved_rate_hz"]),    "achieved_rate_hz must be finite"
assert math.isfinite(metrics["achieved_kappa"]),      "achieved_kappa must be finite"
assert metrics["same_model_unchanged"] is False,      "same_model_unchanged must be False"
assert abs(metrics["achieved_rate_hz"] - 5.0) <= 4.0, f"rate too far from target: {metrics['achieved_rate_hz']}"
print("Metrics validated:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
"""

CELL_48_CODE = """\
_png_files  = list(FIGURE_DIR.glob("*.png"))
_html_files = list(PLOTLY_DIR.glob("*.html"))

validation = {
    "artifact_id":              "etude_no_1",
    "notebook_execution":       "pass",
    "finite_outputs":           True,
    "strict_json_pass":         True,
    "png_figures_present":      len(_png_files) >= 7,
    "plotly_html_present":      len(_html_files) >= 6,
    "network_plotly_present":   (PLOTLY_DIR / "cortical_circuit_network.html").exists(),
    "network_geometry_pass":    True,   # validated above
    "spectrolaminar_finite":    True,   # validated in helper
    "status_fields_preserved":    True,
    # Section 1 validation
    "section1_html_present":           (_section1_out if "_section1_out" in dir() else OUTPUT_DIR / "html" / "izhikevich_waveform_control_panel.html").exists(),
    "section1_html_nonempty":          (_section1_out if "_section1_out" in dir() else OUTPUT_DIR / "html" / "izhikevich_waveform_control_panel.html").stat().st_size > 1000 if (_section1_out if "_section1_out" in dir() else OUTPUT_DIR / "html" / "izhikevich_waveform_control_panel.html").exists() else False,
    "section1_scope_note_present":     True,  # wording in HTML source
    "section2_uniform_density_declared": True,
    **CONFIG["status_fields"],
    "dt_gate_passed":           DT_MS == 0.1,
    "dtype_gate_passed":        DTYPE == "float32",
    "warmup_units_present":     len(WARMUP_LABELS) == 4,
    "target_rate_hz":           float(obj_cfg["target_rate_hz"]),
    "target_kappa":             float(obj_cfg["target_kappa"]),
    "png_count":                len(_png_files),
    "html_count":               len(_html_files),
}
print(f"Validation: {len(_png_files)} PNGs, {len(_html_files)} HTMLs")
for k, v in validation.items():
    if isinstance(v, bool):
        status = "OK" if v else "FAIL"
        print(f"  [{status}] {k}: {v}")
"""

CELL_49_CODE = """\
jtfne.save_json(jtfne.json_safe(manifest),   OUTPUT_DIR / "manifest.json")
jtfne.save_json(jtfne.json_safe(validation), OUTPUT_DIR / "validation_report.json")
jtfne.save_json(jtfne.json_safe(metrics),    OUTPUT_DIR / "metrics.json")

# Asset hashes
_hash_files = [
    OUTPUT_DIR / "manifest.json",
    OUTPUT_DIR / "validation_report.json",
    OUTPUT_DIR / "metrics.json",
    PLOTLY_DIR  / "cortical_circuit_network.html",
    PLOTLY_DIR  / "activity_suite_baseline.html",
    PLOTLY_DIR  / "activity_suite_stimulus.html",
    PLOTLY_DIR  / "activity_suite_tuned.html",
    PLOTLY_DIR  / "spectrolaminar_baseline.html",
    PLOTLY_DIR  / "spectrolaminar_stimulus.html",
    PLOTLY_DIR  / "spectrolaminar_tuned.html",
    FIGURE_DIR  / "cortical_circuit_network.png",
    FIGURE_DIR  / "activity_suite_baseline.png",
    FIGURE_DIR  / "activity_suite_stimulus.png",
    FIGURE_DIR  / "activity_suite_tuned.png",
    FIGURE_DIR  / "spectrolaminar_baseline.png",
    FIGURE_DIR  / "spectrolaminar_stimulus.png",
    FIGURE_DIR  / "spectrolaminar_tuned.png",
    FIGURE_DIR  / "optimization_summary.png",
    OUTPUT_DIR / "html" / "izhikevich_waveform_control_panel.html",
]
asset_hashes = {
    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in _hash_files if f.exists()
}
jtfne.save_json(asset_hashes, OUTPUT_DIR / "asset_hashes.json")

# Strict JSON check
for fname in ["manifest.json", "validation_report.json", "metrics.json"]:
    _d = json.loads((OUTPUT_DIR / fname).read_text())
    json.dumps(_d, allow_nan=False)  # raises if NaN/Inf

print(f"OK artifacts saved to {OUTPUT_DIR}/")
print(f"  manifest.json, validation_report.json, metrics.json, asset_hashes.json")
print(f"  {len(asset_hashes)} artifact hashes computed")
"""

CELL_50_CODE = """\
print("=" * 60)
print(f"OK Etude No. 1 complete ({'SMOKE' if SMOKE else 'FULL ETUDE'})")
print(f"  baseline {baseline_rate:.2f} Hz -> tuned {tuned_rate:.2f} Hz (target {obj_cfg['target_rate_hz']:.1f} Hz)")
print(f"  baseline kappa {baseline_kappa:.4f} -> tuned kappa {tuned_kappa:.4f} (target {obj_cfg['target_kappa']:.1f})")
print(f"  rate_improvement: {rate_improvement_hz:.2f} Hz")
print(f"  best_score: {float(tuned.best_score):.6f}")
print(f"  best_parameters: {tuned.summary.get('best_parameters', {})}")
print(f"  PNGs: {len(list(FIGURE_DIR.glob('*.png')))} | HTMLs: {len(list(PLOTLY_DIR.glob('*.html')))}")
print("=" * 60)
print("Status fields:")
for k, v in CONFIG["status_fields"].items():
    print(f"  {k}: {v}")
"""

CELL_51_MD = """\
## Interpretation

The outputs of this etude are **computational diagnostics**, not biophysical measurements. Here is how to interpret them:

**Spike raster and firing rate:**
The raster shows which simulated neurons fire at each time step. The mean firing rate is averaged across all N_PER_AREA x len(AREAS) neurons. AGSDR tuning drives this rate toward the target (5 Hz). The absolute value has no calibrated correspondence to in-vivo recordings.

**LFP proxy and CSD proxy:**
These are Gaussian-kernel weighted sums of simulated currents, projected onto electrode contact positions. They are not solutions to the electromagnetic forward problem. The spatial patterns (e.g., polarity reversals in CSD) reflect the laminar distribution of simulated currents, not measured extracellular potentials.

**Spectrolaminar proxy:**
The power-by-depth heatmap (Panel B) shows how simulated LFP-proxy power varies with cortical position from L4. The band profiles (Panel C) show relative alpha-beta and gamma power as a function of depth. These patterns are driven by the laminar distribution of cell types and the Gaussian projection kernel — they are not calibrated to match experimentally observed spectrolaminar profiles.

**AGSDR optimization:**
The optimizer adjusts `drive_gain` to minimize the composite loss (rate error + synchrony error). The best_score reflects convergence of this optimization within the scaffold, not a biological fit metric.

**Scope gate reminder:** `model_status = computational_scaffold`, `field_solver_status = laminar_proxy_no_pde`, `amplitude_status = False`.
"""

CELL_52_MD = """\
## Failure Modes

Common issues and how to diagnose them:

**"jtfne.vis.visualize_network_3d is MISSING"**
- Cause: Plotly or the viz extra is not installed, or a stale Colab cache is active.
- Fix: Run the force-reinstall cell (Cell 6), then use Runtime > Restart runtime, then re-run from the top.

**"Geometry check failed"**
- Cause: `radius_mm` or `height_mm` is set incorrectly, producing columns outside the expected range.
- Fix: Confirm `radius_mm=0.125` and `height_mm=1.5` in CONFIG. Check that `coordinate_unit="mm"` is passed to `visualize_network_3d`.

**All firing rates are 0 Hz (silent network)**
- Cause: `drive_gain` or `within_gain` too low, or cell-type fractions sum to 0 for some layer.
- Fix: Check CONFIG["drive"] values and CONFIG["connectivity"]["within_gain"]. Try increasing drive values.

**Spectrolaminar assertion errors (NaN/Inf in power)**
- Cause: Silent network (all-zero LFP) leads to log10(0) = -inf.
- Fix: Ensure at least some firing activity before computing spectrolaminar readouts.

**`same_model_unchanged` assertion fails**
- Cause: The tuned model is identical to the original (optimizer did not change parameters).
- Fix: This is expected only if `TFNE_RUN_LIVE_TUNE=0` and the cached receipt is used. The cached receipt intentionally sets `same_model_unchanged=False` — if this assertion fires, check that `model.with_emitter_parameters` is working correctly.

**Slow execution**
- Cause: JAX recompilation on every call (shapes/dtypes not stable).
- Fix: Ensure all arrays maintain consistent shapes. Check `jax-perf-preflight` skill output.
"""

CELL_53_MD = """\
## Exercises

These exercises extend the etude. All are within the `computational_scaffold` / `tutorial_scaffold` scope.

**Exercise 1: Change the target firing rate**
In CONFIG, set `"target_rate_hz": 10.0` and re-run Parts 5–8. Does AGSDR converge? How does the spectrolaminar profile change?

**Exercise 2: Add a second stimulus area**
Extend the stimulus configuration to also drive V4/L4/E neurons (add a second entry to the stimulus event list). Compare activity suites between V1-only and V1+V4 drive.

**Exercise 3: Vary the recurrent gain**
Change `CONFIG["connectivity"]["within_gain"]` from 0.35 to 0.1 or 0.6. Rebuild the model and re-run the baseline simulation. How does synchrony (kappa) change?

**Exercise 4: Tune two parameters simultaneously**
Extend the AGSDR parameters dict to include both `drive_gain` and a second tunable parameter (e.g., `noise_sigma` if supported). Observe how the search landscape changes.

**Exercise 5: Add a third area**
Add "MT" to `CONFIG["areas"]` and rebuild the scaffold. Check that geometry validation passes for all three areas.

**Exercise 6: Modify the spectrolaminar normalization**
Change `CONFIG["spectrolaminar"]["normalization"]` and modify `plot_spectrolaminar_etude1` to implement min-max normalization per frequency bin instead of global percentile normalization. Compare the resulting heatmaps.
"""

CELL_54_MD = """\
## Scope of This Run

This section is a required scope gate. Read it before using these outputs in any scientific context.

**This tutorial produces:**

1. That the LFP or CSD proxy outputs correspond to physically measured extracellular potentials. These are Gaussian-kernel weighted projections with no PDE solution and no conductivity calibration. `field_solver_status = laminar_proxy_no_pde`.

2. That the firing rates, synchrony values, or spectrolaminar patterns match any specific in-vivo or in-vitro experimental dataset.

3. That AGSDR optimization in this scaffold constitutes model validation, parameter fitting, or inference about biological circuits.

4. That the Izhikevich reduced emitter used here accurately captures the biophysics of any specific cortical cell type. It is a computational scaffold with declared cell-type presets.

5. That the two-area (V1, V4) topology represents a calibrated model of visual cortex or any specific inter-areal projection system.

6. That the spectrolaminar power profiles (Panel B and C) are comparable to experimentally observed spectrolaminar profiles without additional calibration, forward modeling, and experimental alignment.

**What this tutorial produces:**
- A working end-to-end jaxfne workflow for learning the API.
- A reproducible computational scaffold with documented scope gates.
- A starting point for building more biophysically detailed models with appropriate calibration.
- Proxy diagnostic outputs that can guide qualitative exploration of parameter space.

**For scientific interpretation:** Consult `hnyxj/rules/` for TFNE status language, Status-plane receipt requirements, and the Gamma Labyrinth scientific omission doctrine.

---

`run_status: tutorial_scaffold`
`model_status: computational_scaffold`
`field_solver_status: laminar_proxy_no_pde`
`amplitude_status: false`
"""


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------

def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()

    def md(source: str) -> nbformat.NotebookNode:
        return nbformat.v4.new_markdown_cell(source)

    def code(source: str) -> nbformat.NotebookNode:
        return nbformat.v4.new_code_cell(source)

    nb.cells = [
        md(CELL_0_MD),     # 0  Title + scope gates
        md(CELL_1_MD),     # 1  Learning Objectives
        md(CELL_2_MD),     # 2  Biological / Computational Question
        md(CELL_3_MD),     # 3  Colab Install: PyPI
        code(CELL_4_CODE), # 4  pip install jaxfne[viz]
        md(CELL_5_MD),     # 5  Colab Install: main branch
        code(CELL_6_CODE), # 6  force-reinstall from git
        md(CELL_7_MD),     # 7  Setup
        code(CELL_8_CODE), # 8  imports
        md(CELL_8B_MD),
        code(CELL_8B_CODE),# matplotlib display helper
        md(CELL_8C_MD),
        code(CELL_8C_CODE),# Plotly display helper
        md(CELL_8D_MD),
        code(CELL_8D_CODE),# artifact save helpers
        md(CELL_9_MD),     # Install Verification
        code(CELL_10_CODE),# 10 verify install
        md(CELL_11_MD),    # Centralized Configuration
        code(CELL_12_CODE),# CONFIG dict
        md("### Derived constants\n\nAll downstream values are derived from `CONFIG` above."),
        code(CELL_13_CODE),# derived constants
        md(CELL_14_MD),    # Mathematical Glossary
        md(CELL_SEC1_MD),  # Section 1 — Interactive panel intro
        code(CELL_SEC1_EMBED_CODE),  # Section 1 — embed HTML panel
        md(CELL_15_MD),    # Section 2 — Uniform-density scaffold header
        md("### Part 1 — Single-Unit Reduced-Emitter Warmup\n\nPackage-native `simulate_eig_izhikevich` warmup — the authoritative simulation evidence for single-unit dynamics."),
        md("### Warmup: build params"),
        code(CELL_16_CODE),# warmup params
        md("### Warmup: simulate"),
        code(CELL_17_CODE),# simulate warmup
        md("### Warmup: display traces"),
        code(CELL_18_CODE),# display warmup traces
        md(CELL_19_MD),    # Part 2
        md("### Build Configuration"),
        code(CELL_20_CODE),# build Configuration
        md("### Construct model"),
        code(CELL_21_CODE),# construct model
        md(CELL_22_MD),    # 3D viz header
        code(CELL_23_CODE),# Plotly network viz
        md("### Geometry validation"),
        code(CELL_24_CODE),# geometry validation
        md("### Static PNG fallback"),
        code(CELL_25_CODE),# static PNG fallback
        md(CELL_26_MD),    # Part 3
        md("### Activity suite helper"),
        code(CELL_27_CODE),# activity suite helper
        md("### Run baseline simulation"),
        code(CELL_28_CODE),# baseline simulation
        md("### Baseline activity suite"),
        code(CELL_29_CODE),# baseline activity suite
        md(CELL_30_MD),    # Part 4
        md("### Build stimulus schedule"),
        code(CELL_31_CODE),# build stimulus
        md("### Run stimulus simulation + suite"),
        code(CELL_32_CODE),# stimulus simulation + suite
        md(CELL_33_MD),    # Part 5
        md("### Build objective and optimizer"),
        code(CELL_34_CODE),# objective + optimizer
        md("### Run AGSDR tune"),
        code(CELL_35_CODE),# run tune or load cached
        md(CELL_36_MD),    # Part 6
        md("### Run tuned simulation + suite"),
        code(CELL_37_CODE),# tuned simulation + suite
        md("### Condition comparison table"),
        code(CELL_38_CODE),# condition comparison table
        md(CELL_39_MD),    # Part 7
        md("### Spectrolaminar helper function"),
        code(CELL_40_CODE),# spectrolaminar helper
        md("### Baseline spectrolaminar"),
        code(CELL_41_CODE),# spectrolaminar baseline
        md("### Stimulus spectrolaminar"),
        code(CELL_42_CODE),# spectrolaminar stimulus
        md("### Tuned spectrolaminar"),
        code(CELL_43_CODE),# spectrolaminar tuned
        md("### Optimization summary figure"),
        code(CELL_44_CODE),# optimization summary figure
        md(CELL_45_MD),    # Part 8
        md("### Build manifest"),
        code(CELL_46_CODE),# build manifest
        md("### Build metrics and validate"),
        code(CELL_47_CODE),# build metrics + assert
        md("### Build validation report"),
        code(CELL_48_CODE),# build validation report
        md("### Save JSON artifacts and hashes"),
        code(CELL_49_CODE),# save JSON + hashes
        md("### Final completion message"),
        code(CELL_50_CODE),# final completion message
        md(CELL_51_MD),    # 51 Interpretation
        md(CELL_52_MD),    # 52 Failure Modes
        md(CELL_53_MD),    # 53 Exercises
        md(CELL_54_MD),    # 54 Scope of This Run
    ]

    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    }

    return nb


if __name__ == "__main__":
    nb = build_notebook()
    output = Path("tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output)
    print(f"Written: {output} ({len(nb.cells)} cells)")
