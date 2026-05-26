# jaxfne

**JAX-native TFNE (Tensor-Field Neural Equations) workflows for reproducible computational neurophysiology.**

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

**Primary use:** Build reproducible laminar-field proxy simulations with deterministic PRNG, JSON-safe outputs, and clear scope boundaries.

**Tutorial-scale computational scaffold.** jaxfne is a framework for teaching, prototyping, and experimenting with neural-field source models. All outputs are proxy readouts and simulated dynamics, not validated against empirical data.

---

## Quick Start

### Install

Install the latest release (v0.3.4):

```bash
pip install -U "jaxfne>=0.3.4"
```

Optional visualization and optimizer extras:
```bash
pip install "jaxfne[viz]"     # includes matplotlib and plotly
pip install "jaxfne[opt]"     # includes optax optimization
pip install "jaxfne[dev,viz,opt]"  # full development suite
```

### v0.3.4 Chainable Grammar Example

```python
import jaxfne as jtfne

# Configure a single-neuron simulation using chainable API
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
cfg = cfg.cell_types({"E": 1.0})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])

# Construct and simulate
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)

# Inspect results
print(f"Simulation complete: {signals.V_m.shape[0]} timesteps, {signals.V_m.shape[1]} neuron(s)")
print(f"Voltage range: {signals.V_m.min():.1f} to {signals.V_m.max():.1f} mV")
print(f"Spike count: {signals.spikes.sum():.0f}")
```

**Note:** All outputs are computational scaffolds and proxy readouts. This is not a validated biological simulator.

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

## Validation & Release Status

### v0.3.4 Release Receipt

```
PyPI package:  jaxfne==0.3.4 published
Local pytest:  1062 passed, 37 skipped
compileall:    PASS
Core grammar:  PASS (without optional matplotlib)
Notebooks:     Suite No. 1, Suite No. 2, v0.3.1, v0.3.2, v0.3.3 (all executed)
```

Install and verify:

```bash
pip install -U "jaxfne>=0.3.4"
python -c "import jaxfne; print(f'jaxfne version: {jaxfne.__version__}')"
```

### Local validation (every commit, ~1–2 minutes)

```bash
python -m compileall -q jaxfne tests examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=short
```

Report test counts and platform info (Python version, OS, git SHA) for reproducibility.

### Optional: Extended validation (~5 minutes)

```bash
# Run tutorials with figure generation (requires matplotlib)
pip install "jaxfne[viz]"
```

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
| **v0.2.24–v0.2.30** | Foundation & Hardening | Audited contracts, solver status, mathematical glossary, diagnostics, tutorials, performance validation | ✓ Released |
| **v0.3.0–v0.3.4** | Tutorial-Scenario Spine | Chainable Configuration grammar; v0.3.1 single-neuron, v0.3.2 parameter-sweep, v0.3.3 two-neuron E/I tutorials; validated execution receipts | ✓ Released |

**Current release (v0.3.4):** Chainable Configuration grammar with core tutorials validated. Stable public API for v0.3.x tutorial expansion.
- **v0.3.1:** Single-neuron Izhikevich dynamics
- **v0.3.2:** Parameter sweep exploration
- **v0.3.3:** Two-neuron excitatory-inhibitory coupling

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

## Scope Boundaries

**Scope metadata:**
- **truth_mode:** `truth_safe_unverified`
- **computational_level:** `scaffold`
- **physical_amplitude_validation:** `Not performed`

jaxfne is a **computational-scaffold framework**, not a validated biological simulator. All outputs are proxy readouts:

- **Izhikevich model:** Phenomenological spiking model (not empirically calibrated)
- **Source projection:** Declared anatomy + native current (not validated against data)
- **Field approximation:** Proxy CSD/LFP via spatial convolution (not solving Poisson equation)
- **Readout operators:** Relative-scale proxy metrics (not physical units)
- **Optimization results:** Mathematical fitness (success ≠ biological plausibility)

**Appropriate use cases:**
- Teaching neural-field and source-field concepts
- Prototyping computational neuroscience models
- Benchmarking optimization and fitting strategies
- Validating model mathematical consistency

**Do NOT use jaxfne for:**
- Biological validation without separate empirical comparison
- Publishing simulation results as experimental data
- Making physical conductivity claims without calibration
- Interpreting metabolic cost (EMM-proxy) as biological metabolism

---

## License

MIT License.

---

## Contributing

Issues, feature requests, and pull requests welcome. See [CONTRIBUTING](docs/contributing.md).

