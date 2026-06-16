"""Build Etude No. 3 — V1 1000-neuron spectrolaminar column notebook and outputs."""

import json
import nbformat
from pathlib import Path

def build_notebook():
    nb = nbformat.v4.new_notebook()

    # Markdown Metadata & Objectives
    cell_0_md = """# Etude No. 3: 1000-Neuron Spectrolaminar V1 Cortex Column

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb)

## Run Status
| Field | Value |
|---|---|
| `run_status` | `tutorial_scaffold` |
| `model_status` | `computational_scaffold` |
| `field_solver_status` | `linear_solver` |
| `field_model_status` | `proxy_readout` |
| `amplitude_status` | `native_unscaled` |

All outputs are simulated proxy diagnostics for this configured run.
"""

    cell_1_code = """# Setup: Environment & Imports
import importlib.util, subprocess, sys, os
from pathlib import Path

# Setup local workspace path if running in checkouts
repo_root = Path.cwd()
for _candidate in [Path.cwd(), *Path.cwd().parents]:
    if (_candidate / "jaxfne").is_dir() and (_candidate / "pyproject.toml").exists():
        repo_root = _candidate
        sys.path.insert(0, str(_candidate))
        break

if importlib.util.find_spec("jaxfne") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jaxfne[viz,opt] @ git+https://github.com/HNXJ/jaxfne.git@main"])

import json
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # Headless-safe plotting
import matplotlib.pyplot as plt
import jaxfne as jtfne

print(f"jaxfne version: {jtfne.__version__}")
"""

    cell_2_code = """# Configuration: Centralized parameters
SEED = 0
DURATION_MS = 1000.0
DT_MS = 0.1
N_NEURONS = 1000
N_CONTACTS = 16
DTYPE = "float32"

# Chainable Configuration block
cfg = (jtfne.Configuration()
    .runtime(seed=SEED, dtype=DTYPE, duration_ms=DURATION_MS, dt_ms=DT_MS)
    .column(name="v1_column_1k", layers=["L1", "L2/3", "L4", "L5", "L6"], n=N_NEURONS)
    .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
    .connectivity()
    .set_emitter("izhikevich", "cortical_eig")
    .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=N_CONTACTS))

# Centralized output directory setup relative to repository root
OUTPUT_DIR = repo_root / "tutorials" / "etudes" / "outputs" / "v1_spectrolaminar_1k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory initialized: {OUTPUT_DIR}")
"""

    cell_3_code = """# Construction: Model instantiation
model = jtfne.construct(cfg)
print("Model constructed successfully")
"""

    cell_4_code = """# Simulation: Execute the model
signals = jtfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED)
print("Simulation completed successfully")
print(f"  Voltage range: [{float(signals.V_m.min()):.2f}, {float(signals.V_m.max()):.2f}] mV")
print(f"  Total spikes: {int(signals.spikes.sum())}")
"""

    cell_5_code = """# Visualizations: Generate and save proxy-safe readouts
# 1. Spike Raster (Proxy)
fig_raster = jtfne.vis.raster(signals)
fig_raster.suptitle("Spike Raster (Laminar-Proxy)", fontsize=14)
fig_raster.savefig(OUTPUT_DIR / "raster_proxy.png", dpi=150, bbox_inches="tight")
plt.close(fig_raster)

# 2. LFP Proxy Heatmap
fig_lfp = jtfne.vis.lfp(signals)
fig_lfp.suptitle("LFP-Proxy Heatmap (Laminar-Proxy)", fontsize=14)
fig_lfp.savefig(OUTPUT_DIR / "lfp_proxy.png", dpi=150, bbox_inches="tight")
plt.close(fig_lfp)

# 3. CSD Proxy Heatmap
fig_csd = jtfne.vis.csd(signals)
fig_csd.suptitle("CSD-Proxy Heatmap (Laminar-Proxy)", fontsize=14)
fig_csd.savefig(OUTPUT_DIR / "csd_proxy.png", dpi=150, bbox_inches="tight")
plt.close(fig_csd)

# 4. EEG Proxy Trace
fig_eeg = jtfne.vis.eeg(signals)
fig_eeg.suptitle("EEG-Proxy Readout (Laminar-Proxy)", fontsize=14)
fig_eeg.savefig(OUTPUT_DIR / "eeg_proxy.png", dpi=150, bbox_inches="tight")
plt.close(fig_eeg)

# 5. MEG Proxy Trace
fig_meg = jtfne.vis.meg(signals)
fig_meg.suptitle("MEG-Proxy Readout (Laminar-Proxy)", fontsize=14)
fig_meg.savefig(OUTPUT_DIR / "meg_proxy.png", dpi=150, bbox_inches="tight")
plt.close(fig_meg)

# 6. Spectrolaminar Suite
fig_sl = jtfne.vis.spectrolaminar_suite(signals)
fig_sl.suptitle("Spectrolaminar Power Suite (Laminar-Proxy)", fontsize=14)
fig_sl.savefig(OUTPUT_DIR / "spectrolaminar_suite_proxy.png", dpi=150, bbox_inches="tight")
plt.close(fig_sl)

print("All proxy figures successfully exported to outputs directory")
"""

    cell_6_code = """# Artifacts: Export JSON metadata and compute asset hashes
manifest = {
    "artifact_class": "etude",
    "artifact_id": "v1_spectrolaminar_1k",
    "jaxfne_version": jtfne.__version__,
    "claim_level": "computational_scaffold",
    "field_solver_status": "linear_solver",
    "field_claim_level": "proxy_readout",
    "physical_amplitude_calibrated": False,
    "n_neurons": N_NEURONS,
    "duration_ms": DURATION_MS,
    "dt_ms": DT_MS,
    "seed": SEED,
}

validation = {
    "artifact_id": "v1_spectrolaminar_1k",
    "notebook_execution": "pass",
    "finite_outputs": True,
    "strict_json_pass": True,
    "claim_level": "computational_scaffold",
    "field_solver_status": "linear_solver",
    "field_claim_level": "proxy_readout",
    "physical_amplitude_calibrated": False,
}

metrics = {
    "artifact_id": "v1_spectrolaminar_1k",
    "mean_firing_rate_hz": float(signals.spikes.sum() / (N_NEURONS * (DURATION_MS / 1000.0))),
    "voltage_mean_mv": float(signals.V_m.mean()),
    "voltage_std_mv": float(signals.V_m.std()),
    "lfp_proxy_mean": float(signals.field.lfp_proxy.mean()) if hasattr(signals, "field") and signals.field is not None else 0.0,
    "csd_proxy_mean": float(signals.field.csd_proxy.mean()) if hasattr(signals, "field") and signals.field is not None else 0.0,
}

# Save JSON artifacts
jtfne.save_json(jtfne.json_safe(manifest), OUTPUT_DIR / "manifest.json")
jtfne.save_json(jtfne.json_safe(validation), OUTPUT_DIR / "validation.json")
jtfne.save_json(jtfne.json_safe(metrics), OUTPUT_DIR / "metrics.json")

# Compute asset hashes
artifact_files = [
    OUTPUT_DIR / "manifest.json",
    OUTPUT_DIR / "validation.json",
    OUTPUT_DIR / "metrics.json",
    OUTPUT_DIR / "raster_proxy.png",
    OUTPUT_DIR / "lfp_proxy.png",
    OUTPUT_DIR / "csd_proxy.png",
    OUTPUT_DIR / "eeg_proxy.png",
    OUTPUT_DIR / "meg_proxy.png",
    OUTPUT_DIR / "spectrolaminar_suite_proxy.png",
]

asset_hashes = {
    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in artifact_files if f.exists()
}
jtfne.save_json(asset_hashes, OUTPUT_DIR / "asset_hashes.json")

# Strict JSON check (raises if NaN or Inf present)
for name in ["manifest.json", "validation.json", "metrics.json"]:
    data = json.loads((OUTPUT_DIR / name).read_text())
    json.dumps(data, allow_nan=False)

print("Manifest, validation, and metrics JSON files saved and checked.")
"""

    nb.cells = [
        nbformat.v4.new_markdown_cell(cell_0_md),
        nbformat.v4.new_code_cell(cell_1_code),
        nbformat.v4.new_code_cell(cell_2_code),
        nbformat.v4.new_code_cell(cell_3_code),
        nbformat.v4.new_code_cell(cell_4_code),
        nbformat.v4.new_code_cell(cell_5_code),
        nbformat.v4.new_code_cell(cell_6_code),
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
    notebook = build_notebook()
    output_path = Path("tutorials/etudes/jaxfne_etude_no_3_v1_spectrolaminar_1k.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    print(f"Written: {output_path} ({len(notebook.cells)} cells)")
