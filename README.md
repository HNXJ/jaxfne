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

## Objective grammar form

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

cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=1000.0, dt_ms=0.1)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=0)
vm_e = signals.get("vm", cell_type="E")
spk = signals.get("spk")
```

## Checkout validation

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests -q --tb=short
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
PYTHONPATH=. python scripts/audit_notebook_grammar.py --check
mkdocs build --strict
```
