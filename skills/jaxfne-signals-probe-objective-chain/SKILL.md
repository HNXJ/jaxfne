---
name: jaxfne-signals-probe-objective-chain
description: >-
  Verified Signals → probe/readout → objective → vis path for jaxfne. USE when
  analyzing simulation output. Replaces legacy draft that documented nonexistent
  signals.rate(), signals.psd(), signals.probe() methods.
---

# jaxfne Signals → Probe → Objective Chain

USE FIRST: `catalog-glossary-jaxfne`, `jaxfne-objective-grammar`.

## Signals container (what exists)

After `sig = jtfne.simulate(model, ...)`:

| Access | Pattern |
|--------|---------|
| Named arrays | `sig.get("vm")`, `sig.get("spk")`, `sig.get("lfp")`, `sig.get("csd")` |
| Selector filter | `sig.get("vm", layer="L4", cell_type="E")` |
| Field bundle | `sig.field` → `FieldOutput` with `lfp_proxy`, `csd_proxy`, … |
| Metadata | `sig.metadata`, `sig.summary()` |
| Trial axis | **`trial=` not supported** on `get()` — use `run_trials` or tutorial_utils |

Attributes on dataclass: `sig.time_ms`, `sig.V_m`, `sig.spikes`, `sig.sources`, `sig.field`.

## Analysis — use package helpers, not Signals methods

```python
# Rates / synchrony
import jax.numpy as jnp
dt = float(sig.time_ms[1] - sig.time_ms[0])
rate_hz = float(jnp.mean(sig.spikes) * (1000.0 / dt))
kappa = jtfne.kappa_synchrony(sig.spikes, dt_ms=dt)

# Visualization (proxy-safe)
jtfne.vis.raster(sig)
jtfne.vis.rate(sig)
jtfne.vis.lfp(sig)
jtfne.vis.psd(sig)
jtfne.vis.spectrolaminar_suite(sig)

# Tutorial multi-trial pipeline
from jaxfne.tutorial_utils import simulate_laminar_trials, spectrolaminar_from_trials
```

**Do not call:** `sig.rate()`, `sig.psd()`, `sig.bandpower()`, `sig.coherence()`,
`sig.cv_isi()`, `sig.probe(...)`.

## Probe / readout (Model method)

```python
readout = model.probe(sig, modes=["lfp_proxy", "csd_proxy", "source_native"])
# alias: model.record(sig, modes)
```

For EEG/MEG proxies, lead-field transforms are separate (`eeg_proxy_transform`, …)
— not auto-computed on every simulate.

## Objective evaluation

```python
obj = jtfne.rate_targets(groups={...}, targets_hz={...}, weights={...})
report = model.evaluate(sig, obj)
combined = obj.compose(other_obj)
tune_result = model.tune(obj, optimizer=jtfne.agsdr(...))
```

## Multi-trial / batch

```python
batch = jtfne.trial_batch(...)
result = model.run_trials(batch, sim)
# or tutorial_utils.simulate_laminar_trials(model_dict, cfg, n_trials=8)
```

## Manifest

```python
jtfne.manifest(cfg, signals=sig, objective=obj, ...)
model.run_receipt(sig)
```

## Frictions

- `Signals.get(..., trial=)` → `NotImplementedError` (F-009 in `FRICTIONS_STACK.md`)
- Prefer `neuron_table()` over table helpers on raw `Configuration` for counts
