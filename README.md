<p align="center">
  <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/jaxfne-itxt.png" alt="jaxfne" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/v/jaxfne?color=brightgreen" alt="PyPI package"></a>
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/pyversions/jaxfne" alt="Python versions"></a>
  <a href="https://jaxfne.readthedocs.io/"><img src="https://img.shields.io/badge/docs-jaxfne.readthedocs.io-blue" alt="Documentation"></a>
  <a href="https://github.com/HNXJ/jaxfne/actions/workflows/release_ci.yml"><img src="https://github.com/HNXJ/jaxfne/actions/workflows/release_ci.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://jaxfne.readthedocs.io/contributing/"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="contributions welcome"></a>
  <a href="https://github.com/HNXJ/jaxfne/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

# jaxfne

`jaxfne` is a compact JAX package for Tensor-Field Neural Equations (TFNE): a typed computational chain from neural emitters to source tensors, field-proxy operators, probe readouts, objective reports, optimizers, and run manifests.

## Install

```bash
pip install jaxfne
```

Visualization extras:

```bash
pip install "jaxfne[viz]"
```

Development checkout:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e ".[dev,viz]"
```

Canonical import:

```python
import jaxfne as jtfne
```

## Object grammar

Every jaxfne program is the same linear chain. Each step returns the input to the
next, so the whole pipeline reads as one fluent sequence:

```text
setup  ->  config  ->  construct  ->  simulate  ->  visualize  ->  tune/objective  ->  optimize  ->  export
enable_x64   Configuration   Model       Signals     vis.*           Objective         Model.tune   manifest / save_*
```

The full typed chain underneath that sequence:

```text
Config
  -> Runtime(seed, dtype, backend, jit, vmap, duration_ms, dt_ms)
  -> Identity(area, layer, cell_type, unit_id, position)
  -> Emitter(theta_e, state_0, drive, noise, key)
  -> SourceMap(source_mode, source_calibration_status, support, normalization)
  -> FieldProxy(kernel, geometry_metadata, field_solver_status, field_claim_level)
  -> Probe(kind, selector, channel_geometry, units_status, method)
  -> Signals(spk, vm, source, lfp_proxy, csd_proxy, eeg_proxy, meg_proxy, spectrolaminar_proxy, emm_proxy)
  -> Objective(metrics, targets, gates, nulls, rejection_reasons)
  -> Optimizer(search_space, budget, key, constraints)
  -> Manifest(run_id, version, repo_sha, runtime_report, artifact_paths, asset_hashes, truth_gates)
  -> Validation(finite_outputs, strict_json, png_assets, notebook_receipts, optional_dependency_laziness)
```

## Minimal workflow

```python
import jaxfne as jtfne

jtfne.enable_x64()                                   # setup: x64 before arrays

cfg = jtfne.build_laminar_column()                   # config: V1, n=1000, flat E:I (legacy default)
cfg = (cfg.set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))

model   = jtfne.construct(cfg)                        # construct: Configuration -> Model
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)  # simulate -> Signals
spk = signals.get("spk")                             # (n_steps, n_neurons) spike raster
vm_e = signals.get("vm", cell_type="E")              # membrane voltage for E cells
```

## Canonical cortex (default prior)

The verified ground-truth laminar prior is a first-class API surface — no source
editing required. Excitatory fraction rises with depth (L6 ≈90% E), inhibition
peaks superficially (L1 50% I, no PV), PV concentrates at L4; overall ≈77E:23I.

```python
# Canonical laminar cortex: real per-layer composition + laminar placement.
cfg = jtfne.build_laminar_column(n=1000, ei_profile="canonical")

# Multi-area hierarchy with the canonical prior in each area:
cfg = jtfne.build_multi_area_columns(["V1", "V4", "PFC"], ei_profile="canonical")
```

`ei_profile="flat"` (the default) preserves the legacy depth-invariant
composition and `uniform3d` placement **unchanged**; `ei_profile="canonical"`
auto-routes to laminar placement so each neuron keeps its layer label and the
per-layer E:I gradient is expressed. The exported constants
`jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS`, `jtfne.CANONICAL_Z_BANDS`, and
`jtfne.DEFAULT_LAYERS` document the prior directly.

## Tune toward a target

```python
obj = jtfne.rate_synchrony_targets()                 # defaults: 10 Hz, kappa 0 (async-irregular)
result = model.tune(obj, optimizer="AGSDR", steps=50)
tuned = result.model

manifest = jtfne.manifest(cfg, signals=signals)      # export: strict JSON-safe run manifest
```

## Checkout validation

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests -q --tb=short
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
PYTHONPATH=. python scripts/audit_notebook_grammar.py --check
mkdocs build --strict
```
