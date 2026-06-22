# Google Colab Quick Start

**Run jaxfne examples in Google Colab without local setup.**

**Version:** jaxfne 0.4.4  
**Last updated:** 2026-06-22  
**run_status:** tutorial_scaffold, exploratory_simulated_proxy

---

## Installation in Colab

### 1. New Colab Notebook

Open a new Colab notebook: https://colab.research.google.com/

### 2. Install jaxfne (Cell 1)

```python
%pip install "jaxfne==0.4.4"
```

**Expected output:**
```
Successfully installed jaxfne-0.4.4
```

---

## Quick Single-Neuron Example (Cell 2)

```python
import jaxfne as jtfne

cfg = (jtfne.Configuration()
          .runtime(seed=0, duration_ms=500.0, dt_ms=0.1)
          .column("single_neuron", layers=["L2/3"], n=1)
          .cell_types({"E": 1.0})
          .connectivity()
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP-proxy"]))

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=500.0, dt_ms=0.1, seed=0)
print(signals.V_m.shape, signals.spikes.sum())
```

**Expected output (shape varies with seed):**
```
(5000, 1) <spike count>
```

For laminar columns, EEG/MEG proxies, and manifest export, open the
[Quickstart](quickstart.md) or any tutorial notebook via the Colab badges on
[tutorials/index.md](tutorials/index.md).

---

## Two-Neuron E/I Example (Cell 3)

Use the canonical configuration API (same pattern as Quickstart):

```python
import jaxfne as jtfne

cfg = (jtfne.Configuration()
          .runtime(seed=0, duration_ms=200.0, dt_ms=0.1)
          .column("two_neuron_ei", layers=["L2/3"], n=2)
          .cell_types({"E": 1.0, "PV": 1.0})
          .connectivity()
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m"]))

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.1, seed=0)
manifest = model.manifest(signals)
print(signals.spikes.sum(axis=0))
```

---

## Data Access and Export (Cell 4)

```python
# Access simulation outputs
print(f"Time array shape: {signals.time_ms.shape}")
print(f"Voltage array shape: {signals.V_m.shape}")
print(f"Spike array shape: {signals.spikes.shape}")

# Export to JSON-safe format
import json

# Save manifest
with open('manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print("✓ Manifest saved to manifest.json")

# Convert arrays to lists for export
output_dict = {
    "time_ms": signals.time_ms.tolist(),
    "voltage_mV": signals.V_m.tolist(),
    "spikes": signals.spikes.tolist(),
    "model_status": manifest["basis"]["model_status"],
    "run_status": "tutorial_scaffold"
}

with open('simulation_output.json', 'w') as f:
    json.dump(output_dict, f)
print("✓ Simulation output saved to simulation_output.json")
```

---

## Scope and runtime status

jaxfne is a computational scaffold. Outputs are proxy readouts unless you supply
calibration and solver evidence.

| Topic | Status | Notes |
| --- | --- | --- |
| Physical amplitude | Uncalibrated | Simulation/proxy units |
| Metabolism / EMM | Proxy index only | Not ATP or electromagnetic energy |
| Field accuracy | Laminar proxy | Leadfield + finite-difference CSD |
| Elliptic / Maxwell solvers | Reserved | See [Limitations](limitations_and_future_plans.md) |

---

## Conservation proxy diagnostics

When field outputs are enabled, manifests may include conservation-inspired
proxy scalars:

```python
import json

cpd = manifest.get("conservation_proxy_diagnostics")
if cpd:
    print(json.dumps(cpd, indent=2))
```

---

## Saving Colab Output

To download results from Colab to your local machine:

```python
# Save manifest to local file (Colab downloads it automatically)
import json

manifest_json = json.dumps(manifest, indent=2)

# In Colab, use:
from google.colab import files
with open('jaxfne_manifest.json', 'w') as f:
    f.write(manifest_json)
files.download('jaxfne_manifest.json')
```

---

## Troubleshooting

### ImportError: "No module named 'jax'"

JAX is optional. Install with full extras:
```python
%pip install "jaxfne[all]"
```

### RuntimeError: "CUDA not detected"

Colab uses CPU by default for JAX. This is fine; jaxfne runs on CPU.

### ValueError: "NaN/Inf in outputs"

Indicates a simulation issue (e.g., numerical instability). Check:
- Simulation duration and timestep
- Neuron parameters (a, b, c, d values)
- External input magnitude

---

## See Also

- [Computation Basis](computation_basis.md) — Detailed computation contract
- [Conservation Proxy Diagnostics](conservation_proxy_diagnostics.md) — Diagnostics reference
