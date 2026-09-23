# Google Colab Quick Start

**Run jaxfne examples in Google Colab without local setup.**

**Version:** published PyPI release `jaxfne==0.5.0` (tag `v0.5.0`); previous release `0.4.25`
**Last updated:** 2026-09-20  
**run_status:** tutorial_scaffold, exploratory_simulated_proxy

---

## Installation in Colab

### 1. New Colab Notebook

Open a new Colab notebook: https://colab.research.google.com/

### 2. Install jaxfne (Cell 1)

```python
%pip install jaxfne
```

**Expected output:**
```
Installing collected packages: jaxfne
Successfully installed jaxfne-0.5.0
```

---

## Quick Single-Neuron Example (Cell 2)

```python
import jaxfne as jtfne
import json

# Single Izhikevich neuron (current API)
cfg = (
    jtfne.configuration()
    .network(n=1)
    .emitter(family="izhikevich", preset="regular_spiking")
    .field(domain="point")
    .probe(name="single_neuron", modes=["spikes", "V_m"])
)
model = jtfne.construct(cfg)

# Simulate 100 ms
signals = model.simulate(jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0))

# Generate manifest with full diagnostics
manifest = model.manifest(signals)

# Print key outputs
print("=== Computation Contract ===")
print(f"Version: {jtfne.__version__}")
print(f"Claim level: {manifest['claim_level']}")
print(f"Physical amplitude calibrated: {manifest['physical_amplitude_calibrated']}")
print(f"Field solver status: {manifest['field_solver_status']}")

print("\n=== CONSERVATION PROXY DIAGNOSTICS (v0.4.8) ===")
cpd = manifest.get("conservation_proxy_diagnostics")
if cpd:
    print(f"Status: {cpd['diagnostic_status']}")
    print(f"Source norm (L1): {cpd.get('source_norm_l1')}")
    print(f"Source norm (L2): {cpd.get('source_norm_l2')}")
    print(f"Source conservation residual: {cpd.get('source_conservation_proxy_residual')}")
    print(f"Poisson solver status: {cpd['poisson_solver_status']}")
    print(f"Maxwell solver status: {cpd['maxwell_solver_status']}")
else:
    print("(No field computed; diagnostics not available)")

print("\n=== JSON-SAFE VALIDATION ===")
try:
    json.dumps(manifest, allow_nan=False)
    print("✓ Manifest is JSON-safe (no NaN/Inf)")
except Exception as e:
    print(f"✗ JSON validation failed: {e}")

print("\n=== SIMULATION SUMMARY ===")
print(f"Simulation time: {signals.time_ms[-1]:.1f} ms")
print(f"Timesteps: {len(signals.time_ms)}")
print(f"Spike rate: {signals.spikes.sum() / len(signals.time_ms):.2f} spikes/ms")
```

**Expected output:**
```
=== Computation Contract ===
Version: 0.5.0
Claim level: computational_scaffold
Physical amplitude calibrated: False
Field solver status: linear_solver

=== CONSERVATION PROXY DIAGNOSTICS ===
Status: proxy
Source norm (L1): ~0.75
Source norm (L2): ~1.74
Source conservation residual: ~0.75
Poisson solver status: not_implemented
Maxwell solver status: not_implemented

=== JSON-SAFE VALIDATION ===
✓ Manifest is JSON-safe (no NaN/Inf)

=== SIMULATION SUMMARY ===
Simulation time: 99.9 ms
Timesteps: 1000
Spike rate: 0.00 spikes/ms
```

Interactive dark-theme panels for this run: [index](_static/atlas/single_neuron/index.html) · [schema](_static/atlas/single_neuron/schema.html) · [raster](_static/atlas/single_neuron/raster.html) · [LFP](_static/atlas/single_neuron/lfp.html) · [oscillatory](_static/atlas/single_neuron/oscillatory.html).

---

## Two-Neuron E/I Example (Cell 3)

```python
import jaxfne as jtfne
import numpy as np

# Two neurons: one excitatory, one inhibitory (current API)
cfg = (
    jtfne.configuration()
    .network(
        n=2,
        cell_types={"E": 1, "PV": 1},
        connectivity={"E→E": 0.1, "E→PV": 0.2, "PV→E": -0.3, "PV→PV": -0.1},
    )
    .emitter(family="izhikevich", preset="regular_spiking")
    .field(domain="point")
    .probe(name="two_neuron_ei", modes=["spikes", "V_m"])
)
model = jtfne.construct(cfg)

# Simulate with external input
signals = model.simulate(jtfne.simulation(duration_ms=200.0, dt_ms=0.1, seed=0))

# Get manifest
manifest = model.manifest(signals)

print("=== TWO-NEURON E/I CIRCUIT ===")
print(f"Excitatory firing rate: {signals.spikes[:, 0].sum() / 200:.2f} Hz")
print(f"Inhibitory firing rate: {signals.spikes[:, 1].sum() / 200:.2f} Hz")

# Verify status checks still immutable
assert manifest["physical_amplitude_calibrated"] == False, "Status check violated!"
print("✓ Status checks immutable: physical_amplitude_calibrated = False")
```

**Expected output:**
```
=== TWO-NEURON E/I CIRCUIT ===
Excitatory firing rate: 0.01 Hz
Inhibitory firing rate: 0.01 Hz
✓ Status checks immutable: physical_amplitude_calibrated = False
```

Interactive dark-theme panels for this run: [index](_static/atlas/two_neuron_ei/index.html) · [schema](_static/atlas/two_neuron_ei/schema.html) · [raster](_static/atlas/two_neuron_ei/raster.html) · [LFP](_static/atlas/two_neuron_ei/lfp.html) · [oscillatory](_static/atlas/two_neuron_ei/oscillatory.html).

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

## Status Status and Scientific Statements

**Value status:** every output below is Relative unless you supply an explicit
calibration step. See [Scope & status](scope_and_status.md).

### Statement boundaries (v0.4.8)

| Statement | Value | Notes |
|-------|--------|-------|
| **Amplitude** | Relative | Values are in simulation units |
| **Metabolism** | Relative | Izhikevich model is phenomenological |
| **Field readouts** | Relative | CSD/LFP outputs are forward-field projections |
| **Solver** | `linear_solver` | Elliptic/volumetric solvers are a reserved future regime; current diagnostics use a Relative-value projection |

### What v0.4.8 is

- Exploratory computational neuroscience model
- Multi-scale emitter (Izhikevich, HH) to field-proxy pipeline
- Teaching tool for understanding circuit behavior
- Optimization sandbox for fitness/plasticity experiments

---

## Example: Conservation Proxy Diagnostics (v0.4.8)

If your model includes field outputs, conservation diagnostics are available:

```python
# (Requires field computation in model)
from jaxfne import compute_conservation_proxy_diagnostics

# Extract diagnostics from manifest
cpd = manifest.get("conservation_proxy_diagnostics")

if cpd:
    print("=== CONSERVATION PROXY DIAGNOSTICS ===")
    print(f"Diagnostic version: {cpd['diagnostic_version']}")
    print(f"Physical amplitude status: {cpd['amplitude_status']}")
    
    if cpd.get('source_norm_l1') is not None:
        print(f"Source norm (L1): {cpd['source_norm_l1']:.4f}")
        print(f"Source norm (L2): {cpd['source_norm_l2']:.4f}")
        print(f"Source conservation proxy residual: {cpd['source_conservation_proxy_residual']:.4f}")
    
    print(f"\nSolver status:")
    print(f"  Poisson: {cpd['poisson_solver_status']}")
    print(f"  Maxwell: {cpd['maxwell_solver_status']}")
    print(f"  Stress-energy tensor: {cpd['stress_energy_tensor_status']}")
    print(f"  J·E power: {cpd['j_dot_e_proxy']}")
    print(f"  Poynting flux: {cpd['poynting_flux_proxy']}")
else:
    print("No field outputs; conservation diagnostics not available.")
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
- Conservation Proxy Diagnostics (`docs/conservation_proxy_diagnostics.md` — repository-internal reference, excluded from the built site) — Diagnostics reference
