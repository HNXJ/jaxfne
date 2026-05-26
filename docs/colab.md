# Google Colab Quick Start (v0.2.27)

**Run jaxfne examples in Google Colab without local setup.**

**Version:** v0.2.27  
**Last updated:** 2026-05-22  
**truth_mode:** truth_safe_unverified, exploratory_not_biological_truth

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
Successfully installed jaxfne-0.2.27
```

---

## Quick Single-Neuron Example (Cell 2)

```python
from jaxfne.core import configuration, construct, simulate
from jaxfne.emitters import IzhikevichEmitter
import json

# Create configuration
config = configuration()

# Build model with single Izhikevich neuron
emitter = IzhikevichEmitter(
    n_neurons=1,
    v_init=-65.0,
    u_init=-15.0
)
model = construct(config, emitters=[emitter])

# Simulate 100 ms
signals = simulate(model, duration_ms=100, dt_ms=0.1)

# Generate manifest with full diagnostics
manifest = model.manifest(signals)

# Print key outputs
print("=== v0.2.27 BASIS (Computation Contract) ===")
basis = manifest["basis"]
print(f"Version: {basis['jaxfne_version']}")
print(f"Computational scaffold: {basis['claim_level']}")
print(f"Physical amplitude claim allowed: {basis['physical_amplitude_claim_allowed']}")
print(f"Field solver status: {basis['field_solver_status']}")

print("\n=== CONSERVATION PROXY DIAGNOSTICS (v0.2.27) ===")
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
=== v0.2.27 BASIS (Computation Contract) ===
Version: 0.2.27
Computational scaffold: computational_scaffold
Physical amplitude claim allowed: False
Field solver status: laminar_proxy_no_pde

=== CONSERVATION PROXY DIAGNOSTICS (v0.2.27) ===
Status: proxy
Source norm (L1): ~0.15
Source norm (L2): ~0.20
Source conservation residual: ~0.01
Poisson solver status: not_implemented
Maxwell solver status: not_implemented

=== JSON-SAFE VALIDATION ===
✓ Manifest is JSON-safe (no NaN/Inf)

=== SIMULATION SUMMARY ===
Simulation time: 100.0 ms
Timesteps: 1000
Spike rate: 0.05 spikes/ms
```

---

## Two-Neuron E/I Example (Cell 3)

```python
from jaxfne.core import configuration, construct, simulate
from jaxfne.emitters import IzhikevichEmitter
import numpy as np

# Create two neurons: one excitatory, one inhibitory
config = configuration()

exc_neuron = IzhikevichEmitter(
    n_neurons=1,
    v_init=-65.0,
    u_init=-15.0,
    # Regular spiking (excitatory-like)
    a=0.02, b=0.2, c=-65.0, d=8.0,
    name="E"
)

inh_neuron = IzhikevichEmitter(
    n_neurons=1,
    v_init=-65.0,
    u_init=-15.0,
    # Fast spiking (inhibitory-like)
    a=0.1, b=0.2, c=-65.0, d=2.0,
    name="I"
)

# Build model with E and I populations
model = construct(config, emitters=[exc_neuron, inh_neuron])

# Simulate with external input
signals = simulate(model, duration_ms=200, dt_ms=0.1)

# Get manifest
manifest = model.manifest(signals)

print("=== TWO-NEURON E/I CIRCUIT ===")
print(f"Excitatory firing rate: {signals.spikes[:, 0].sum() / 200:.2f} Hz")
print(f"Inhibitory firing rate: {signals.spikes[:, 1].sum() / 200:.2f} Hz")

# Verify claim gates still immutable
basis = manifest["basis"]
assert basis["physical_amplitude_claim_allowed"] == False, "Claim gate violated!"
print("✓ Claim gates immutable: physical_amplitude_claim_allowed = False")
```

**Expected output:**
```
=== TWO-NEURON E/I CIRCUIT ===
Excitatory firing rate: 0.45 Hz
Inhibitory firing rate: 0.65 Hz
✓ Claim gates immutable: physical_amplitude_claim_allowed = False
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
    "claim_level": manifest["basis"]["claim_level"],
    "truth_mode": "truth_safe_unverified"
}

with open('simulation_output.json', 'w') as f:
    json.dump(output_dict, f)
print("✓ Simulation output saved to simulation_output.json")
```

---

## Truth Status and Scientific Claims

**Important:** jaxfne is an exploratory computational framework. Do not interpret outputs as biological truth.

### Claim Boundaries (v0.2.27)

| Claim | Status | Notes |
|-------|--------|-------|
| **Physical amplitude** | Not allowed | Values are in simulation units, not validated physical currents |
| **Biological metabolism** | Not allowed | Izhikevich model is phenomenological, not biophysical |
| **Field accuracy** | Proxy only | CSD/LFP outputs are forward-field proxies, not validated against experiment |
| **Solver status** | Not implemented | Poisson/Maxwell solvers are declared future; diagnostics are proxy summaries |

### What v0.2.27 IS

- ✓ Exploratory computational neuroscience framework
- ✓ Multi-scale emitter (Izhikevich, HH) to field-proxy pipeline
- ✓ Teaching tool for understanding circuit behavior
- ✓ Optimization sandbox for fitness/plasticity experiments

### What v0.2.27 IS NOT

- ✗ Biological validation model
- ✗ Whole-brain simulator
- ✗ Empirically calibrated circuit model
- ✗ Solver for differential equations (no PDEs solved)

---

## Example: Conservation Proxy Diagnostics (v0.2.27)

If your model includes field outputs, conservation diagnostics are available:

```python
# (Requires field computation in model)
from jaxfne import compute_conservation_proxy_diagnostics

# Extract diagnostics from manifest
cpd = manifest.get("conservation_proxy_diagnostics")

if cpd:
    print("=== CONSERVATION PROXY DIAGNOSTICS ===")
    print(f"Diagnostic version: {cpd['diagnostic_version']}")
    print(f"Physical amplitude claim: {cpd['physical_amplitude_claim_allowed']}")
    
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
- [Conservation Proxy Diagnostics](conservation_proxy_diagnostics.md) — Diagnostics reference
