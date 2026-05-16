# jaxfne

**JAX Field Neural Equations** (v0.0.2): a JAX-native source-to-field neurophysiology engine for Tensor-Field Neural Equations.

```python
import jax
import jax.numpy as jnp
import jaxfne as jtfne

cfg = jtfne.configuration()
cfg = cfg.network(
    name="V1",
    kind="cortical_column",
    n=1000,
    layers=["L1", "L2/3", "L4", "L5", "L6"],
    cell_types={"E": 0.80, "PV": 0.10, "SST": 0.07, "VIP": 0.03},
)
cfg = cfg.emitter(family="izhikevich", preset="cortical_eig")
cfg = cfg.field(domain="laminar_column", conductivity="isotropic", boundary="mean_zero_neumann", gauge="mean_zero")
cfg = cfg.probe(name="laminar_probe", modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"])

# Metadata gates included by default (v0.0.2)
model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.05, plasticity=1.0, seed=0)
signals = model.simulate(sim)

# probe() is canonical TFNE method; record() is user-friendly alias
readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])

# Manifest includes truth gates
manifest = model.manifest(signals)
```

## Doctrine

`jaxfne` is not primarily a neuron model, optimizer, plotting library, or data format. It is the composition layer:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

JAX handles array compilation and batching. Jaxley can provide detailed emitters. Optax can provide differentiable optimizers. `jaxfne` handles TFNE source-to-field/readout contracts, diagnostics, and manifests.

## Status and Roadmap

**v0.0.2** (current): API hardening with metadata gates
- Signal → Signals container (plural: holds multiple arrays)
- probe() canonical readout; record() is alias
- Metadata gates baked in (truth_mode, claim_level, calibration status, etc.)
- test suite validates JSON-safe manifest and optional dependency guards
- Scientific claims remain `truth_safe_unverified`: design scaffold only

**v0.0.3** (next): First vertical slice hardening
- Source-to-field contract validation
- Calibration status enforcement
- JSON-safe reproducibility gates
- Field solver proxy/full status documentation

**v0.0.4+**: Paradigm execution, optimization, detailed emitters
- Paradigm.batch() runtime engine
- Model.tune() with Optax/GSDR
- Detailed Jaxley compartment bridge
- MEG/EEG readout modes

## Architecture

Core pipeline (Tensor-Field Neural Equations):
```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

Package structure:
```text
jaxfne/
  __init__.py        # public API surface
  core.py            # Configuration, Model, Simulation, Signal/Signals, Probe, Objective, Paradigm
  emitters.py        # Izhikevich EIG scaffold
  fields.py          # laminar field/probe (proxy, not full PDE)
  optim.py           # AGSDR placeholder + require_optax guard
  bridges.py         # Jaxley bridge scaffold + require_jaxley guard
  io.py              # manifest, JSON-safe serialization, hashing
```

## Truth Status

- `truth_mode`: "truth_safe_unverified"
- `claim_level`: "computational_scaffold"
- `source_calibration_status`: "uncalibrated_izhikevich_native_current"

This package provides a JAX-native computational scaffold for declaring TFNE models, running simulations, recording outputs, and manifesting results with truth gates. **No biological or biophysical claims** are made. Amplitude calibration, field solver validation, and paradigm execution are planned for v0.0.3+.

## Optional Dependencies

Install with optional extras for extended features:

```bash
pip install -e ".[opt]"      # Optax integration (v0.0.4+)
pip install -e ".[jaxley]"   # Jaxley compartment bridge (v0.0.4+)
pip install -e ".[dev]"      # dev tools (pytest, ruff)
pip install -e ".[all]"      # all extras
```

Optional dependencies are not required. The core package works with JAX, NumPy, and SciPy.
