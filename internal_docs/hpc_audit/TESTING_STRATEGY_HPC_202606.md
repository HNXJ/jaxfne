# jaxfne HPC/JAX Testing Strategy — Patch 202606

## Test axes

```text
backend: cpu, cuda, tpu/optional, apple/optional
dtype: float32 default, float64 only when x64 enabled at process start
shape: smoke, tutorial, stress
execution: eager, jit, vmap, pmap/shmap optional
layout: time_node, node_time, trial_time_node, node_trial_time
storage: dense opt-in, edge-list default, artifact_ref weights
```

## Required tests

### JAX transform tests

```python
def test_emitters_jit_scan_smoke(): ...
def test_simulate_flat_jit_smoke(): ...
def test_vmap_over_seed_batch(): ...
def test_vmap_over_candidate_population(): ...
def test_pmap_candidate_smoke_when_multi_device(): ...
```

### Pytree tests

```python
def test_flatnet_tree_roundtrip(): ...
def test_signaltensor_tree_roundtrip(): ...
def test_fieldtensor_tree_roundtrip(): ...
```

### No-hidden-fallback tests

```python
def test_safe_jit_raises_in_strict_mode(): ...
def test_safe_vmap_raises_in_strict_mode(): ...
```

### Numerical equivalence tests

```python
def test_flat_and_net_spikes_match_exact_seed(): ...
def test_flat_and_net_voltage_close(): ...
def test_jax_psd_and_numpy_psd_smoke_close(): ...
```

Recommended tolerances:

```text
spikes: exact equality
voltages/source: rtol=1e-5, atol=1e-5 for float32
lfp/csd proxy: rtol=1e-5, atol=1e-6
PSD/bandpower: rtol=1e-4, atol=1e-6
training score: bounded metric comparison, not byte-exact history
```

### Hardware/CI markers

```python
import pytest

cuda = pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="GPU unavailable")
multidevice = pytest.mark.skipif(len(jax.devices()) < 2, reason="multi-device unavailable")
```

## Release gate commands

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
PYTHONPATH=. TFNE_SMOKE=1 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_smoke.ipynb
PYTHONPATH=. TFNE_SMOKE=0 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_full.ipynb
python -m json.tool outputs/jaxfne_etude_no_1/manifest.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/validation_report.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/metrics.json >/dev/null
python -m build
python -m twine check dist/*
```
