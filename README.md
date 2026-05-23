# jaxfne

**JAX-native TFNE (Tensor-Field Neural Emitter) workflows for reproducible computational neurophysiology.**

[![PyPI](https://img.shields.io/pypi/v/jaxfne.svg)](https://pypi.org/project/jaxfne/) ·
[![Docs](https://readthedocs.org/projects/jaxfne/badge/?version=latest)](https://jaxfne.readthedocs.io/en/latest/) ·
[![GitHub](https://img.shields.io/badge/github-HNXJ%2Fjaxfne-blue)](https://github.com/HNXJ/jaxfne) ·
[![Issues](https://img.shields.io/github/issues/HNXJ/jaxfne)](https://github.com/HNXJ/jaxfne/issues)

---

## What is jaxfne?

jaxfne is a compact JAX-native framework for composing neural simulations from modular operators:

```
Emitter (neuron state) → Source (membrane current) → Field (proxy/solved) → Probe (readout) → Objective
```

**Primary use:** Build reproducible laminar-field proxy simulations with deterministic PRNG, JSON-safe outputs, and clear claim boundaries.

**Not a biological simulator.** jaxfne is a computational-scaffold framework for teaching, prototyping, and experimenting with neural-field source models. All outputs are proxies unless explicitly validated against empirical data.

---

## Quick Start

### Install

```bash
pip install jaxfne
```

Optional JAX acceleration:
```bash
pip install -e '.[jax]'
```

Optional development/visualization:
```bash
pip install -e '.[dev,viz]'
```

### Minimal Example

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(name="V1", kind="cortical_column", n=100)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    .probe(name="laminar_16ch", modes=["spikes", "V_m", "source", "CSD"], n_contacts=16)
)

model = jtfne.construct(cfg)
signals = model.simulate(jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0))
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz"),
    jtfne.readout_spec("csd", "csd_abs_mean"),
])

manifest = model.manifest(signals, readouts)
print(f"Simulation complete: {signals.V_m.shape[0]} timesteps, {signals.V_m.shape[1]} neurons")
print(f"Source status: {manifest['source_calibration_status']}")
print(f"Field status: {manifest['field_solver_status']}")
```

---

## The Pipeline

### 1. Emitter: Neural Dynamics

Declare neuron model (Izhikevich or custom) and recurrent connectivity:

```python
.emitter(family="izhikevich", preset="cortical_eig")
```

**Output:** State vector $z(t)$ and native membrane current $I(t)$ [time, neurons]

**Status:** Izhikevich presets are provided; no biological calibration claimed (computational scaffold)

### 2. Source: Spatial Projection

Project neural current into space (laminar probe contacts or voxels):

```python
.field(domain="laminar_column", conductivity="proxy", ...)
```

**Output:** Source density $q(x,t)$ [time, contacts]

**Status:** Proxy projection using anatomical position and Izhikevich native current; no empirical validation

### 3. Field: Field Approximation

Current default: **proxy CSD** (no PDE solve).

```
field_solver_status = "laminar_proxy_no_pde"
```

CSD and LFP are computed from source without solving the Poisson equation. Conductivity is metadata-only.

**Available (v0.2.27+):** Conservation-inspired proxy diagnostics. Physical conductivity remains gated future work (v0.3.x).

### 4. Probe: Multimodal Readouts

Extract metrics from emitter state and field:

| Operator | Output | Meaning |
|----------|--------|---------|
| **Spikes (SPK)** | Binary spike raster [T, N] | Action potentials (thresholded state) |
| **Voltage (V_m)** | Membrane voltage [T, N] | Membrane potential state |
| **Source** | Transmembrane current [T, X] | Spatial source density |
| **LFP-proxy** | Local field potential [T, X] | Proxy; not physical units |
| **CSD-proxy** | Current-source density [T, X] | Proxy; spatial divergence of source |
| **EEG-proxy** | Electroencephalogram [T, N_channels] | Proxy; not physical units |
| **MEG-proxy** | Magnetoencephalogram [T, N_channels] | Proxy; not physical units |
| **EMM-proxy** | Metabolic-like cost [T] | Relative activity intensity (NOT biological metabolism) |

All readouts are proxies unless explicitly solved and validated.

### 5. Objective & Optimization

Declare optimization targets and run GSDR/AGSDR (custom optimizers; Optax optional):

```python
objectives = [
    jtfne.objective(name="spike_rate", target=10.0, metric="spike_rate_hz"),
    jtfne.objective(name="mean_voltage", target=-50.0, metric="mean_V_m"),
]
```

---

## Validation

### Fast validation (every commit, ~1 minute)

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q --tb=line
```

**Results:** 806 tests passed, 5 skipped. Examples 00–06 run. Build validation passes.

### Extended validation (release, ~5–10 minutes)

```bash
python scripts/run_all_tutorials.py --smoke --write-figures
python scripts/validate_tutorial_outputs.py outputs/
```

Runs large tutorials (examples 02–07) with deterministic figures and asset hashing. See [CI policy](docs/ci_policy.md).

---

## Documentation Map

| Topic | Document | Purpose |
|-------|----------|---------|
| **Equations & Math** | [Mathematical Glossary Flow](docs/mathematical_glossary_flow.md) | TFNE equations (emitter, source, field, probe) with term glossaries and claim boundaries |
| **Source Detail** | [Source/Field Equations](docs/source_field_equations.md) | Source modes, forbidden double-counting pattern, field metadata, code examples |
| **Architecture** | [Computation Basis](docs/computation_basis.md) | TFNE as collapsible tensor-field scaffold; extensibility doctrine |
| **Probe Operators** | [Probe Operators](docs/probe_operators.md) | Eight multimodal operators, claim boundaries per operator |
| **I/O & Manifests** | [Output Bundles](docs/output_bundles.md) | Signals, Manifest, ReadoutResult schema and JSON-safe contracts |
| **Bridges & Interop** | [Jaxley Interop](docs/jaxley_interop.md) | Convert Jaxley voltage traces to jaxfne Signals |
| **Scope & Limits** | [Scope and Limitations](docs/scope_and_limitations.md) | What jaxfne claims and does not claim |
| **Full Docs** | [jaxfne.readthedocs.io](https://jaxfne.readthedocs.io/) | API reference, tutorials, changelog |

---

## Roadmap

| Version | Phase | Content | Status |
|---------|-------|---------|--------|
| **v0.2.24** | Foundation Audit | Audited contracts, verified solver status, updated language | ✓ Released |
| **v0.2.25** | Docs-First | Mathematical glossary, source/field doctrine, computation basis | ✓ Released |
| **v0.2.26** | Extensibility | Documented future bases, multi-area scaffolds, BasisSpec | ✓ Released |
| **v0.2.27** | Diagnostics | Conservation-inspired proxy diagnostics, source norms, field-gradient proxy | ✓ Released |
| **v0.2.28** | Tutorial Figures | Canonical tutorial figure manifest, static PNGs, Jaxley bridge hardening | ✓ Released |
| **v0.2.29** | Tensor-Network Ancestry | Pellionisz/Llinás context, basis-transform doctrine | ✓ Released |
| **v0.2.30** | Performance Hardening | Benchmark receipts, JSON safety validation, CI policy | ✓ Released |
| **v0.3.x** | Tutorial-Scenario Line | 32-phase tutorial spine on stable v0.2.30 toolbox; no automatic package bumps | 🔄 In Progress |

**Current phase:** v0.3 tutorial-scenario line (built on `jaxfne==0.2.30`). The v0.3 line is primarily docs, notebooks, equations, and figures. The v0.3 line uses `import jaxfne as jtfne` on the stable v0.2.30 toolbox unless a package bug requires a patch release.

**v0.3.0 tutorial atlas scaffold** now available in [`docs/tutorials_v030/`](docs/tutorials_v030/) with full audit infrastructure:
- **15-scenario learning spine** (single neurons → optimization)
- **13-section notebook template** (learning objectives, mathematics, claims, figures) with LaTeX equation display policy
- **Hard acceptance gates** (firing rate 2–25 Hz, finite values, JSON-safe, JAX-native)
- **PNG + Plotly artifact system** (reproducible figures with SHA256 integrity)
- **Canonical imports** (`import jaxfne as jtfne` enforced)
- **Docs audit policy** (link validation, Colab links, LaTeX equations, term glossaries)
- **Environment setup** ([`requirements-v030-tutorials.txt`](requirements-v030-tutorials.txt), [`docs/tutorials_v030/environment.md`](docs/tutorials_v030/environment.md))
- **Automated audit script** ([`scripts/audit_v030_docs_links.py`](scripts/audit_v030_docs_links.py))

---

## Claim Status

**truth_mode:** `truth_safe_unverified`  
**claim_level:** `computational_scaffold`  
**physical_amplitude_claim_allowed:** `False`  

jaxfne is **not a biological simulator.** All outputs are computational proxies:

- **Izhikevich native current** is a mathematical dynamics model, not empirically calibrated membrane current
- **Source projection** uses declared anatomy but is not validated against measured sources
- **Field (proxy)** is NOT a solved Poisson equation; CSD/LFP are kernel-based approximations
- **Readout proxies** (LFP, CSD, EEG, MEG, EMM) are relative metrics, not physical units
- **Optimization** is mathematical fitness; success ≠ biological plausibility

**When to use jaxfne:**
- Teaching neural-field concepts
- Prototyping source-field models
- Benchmarking optimization strategies
- Validating model consistency (future: conservation diagnostics)

**When NOT to use jaxfne:**
- Making biological claims without separate empirical validation
- Publishing simulation results as if they are real neural data
- Claiming physical conductivity without calibration
- Interpreting metabolic cost (EMM-proxy) as biological metabolism

---

## License

MIT License.

---

## Contributing

Issues, feature requests, and pull requests welcome. See [CONTRIBUTING](docs/contributing.md).

