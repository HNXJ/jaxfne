---
name: jaxfne-neural-network
description: >-
  Verified Model/construct/simulate/Signals/probe/objective/tune runtime chain
  — the compiled, jitted, simulatable "network" object. USE when constructing
  a Model, running a simulation, reading Signals, probing/evaluating output,
  or tuning. Merged from jaxfne-signals-probe-objective-chain + the
  Model/Signals levels of jaxfne-objective-grammar (2026-06-30). Model has NO
  __call__ (not a direct H(t+dt)=Net(X(t),H(t)) callable at the public level
  — that exact scan pattern exists internally in jaxfne/_pipeline.py,
  HDP-edge-list-only, see jaxfne-neural-tensor) and is NOT a registered JAX
  pytree at the class level; jitting happens inside simulate()/construct(),
  not by calling Model(x, h) directly.
---

# jaxfne Neural Network

USE FIRST: `catalog-glossary-jaxfne`, `jaxfne-config` or `jaxfne-neural-tensor`
(whichever built your input).

## What `Model` actually is (verified via source)

```python
Model(cfg: Configuration, params: dict, static: dict)
```

Immutable. `params` = dynamic pytree (tunable/traced arrays). `static` = JIT-static
metadata (non-array). Also exported as alias `Net`. **No `__call__`** — you do
not call `model(x_t, h_t)` directly; the dynamic-system step happens inside
`jtfne.simulate(model, ...)`.

## Level 2 — Model

```python
model = jtfne.construct(cfg)                 # Configuration path
model = jtfne.construct(tensor, runtime)      # NeuronalTensor path (jaxfne-neural-tensor)
nt = model.neuron_table()                     # list[dict] — authoritative per-neuron counts
idx = model.select(layer="L4", cell_type="E")
model = model.with_emitter_parameters(drive_per_neuron=arr)
```

## Level 3 — Simulation → Signals

**Canonical:** top-level `simulate` (not notebook-local kernels):

```python
sig = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
# or
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.5, seed=0)
sig = jtfne.simulate(model, sim=sim)
```

`Model.simulate(...)` and `Model.run_trials(...)` exist for batch/condition paths.

### Reading `Signals` — no invented methods

| Access | Pattern |
|--------|---------|
| Named arrays | `sig.get("vm")`, `sig.get("spk")`, `sig.get("lfp")`, `sig.get("csd")` |
| Selector filter | `sig.get("vm", layer="L4", cell_type="E")` |
| Field bundle | `sig.field` → `FieldOutput` with `lfp_proxy`, `csd_proxy`, … |
| Metadata | `sig.metadata`, `sig.summary()` |
| Trial axis | **`trial=` not supported** on `get()` → raises `NotImplementedError`; use `run_trials` or `tutorial_utils` |

Attributes on the dataclass: `sig.time_ms`, `sig.V_m`, `sig.spikes`, `sig.sources`, `sig.field`.

**Do not call** (not on `Signals`, will `AttributeError`): `.rate()`, `.psd()`,
`.bandpower()`, `.coherence()`, `.cv_isi()`, `.probe(...)`. Use:

```python
import jax.numpy as jnp
dt = float(sig.time_ms[1] - sig.time_ms[0])
rate_hz = float(jnp.mean(sig.spikes) * (1000.0 / dt))
kappa = jtfne.kappa_synchrony(sig.spikes, dt_ms=dt)
jtfne.vis.raster(sig); jtfne.vis.rate(sig); jtfne.vis.lfp(sig); jtfne.vis.psd(sig)
```

## Level 4 — Probe / readout

```python
readout = model.probe(sig, modes=["lfp_proxy", "csd_proxy", "source_native"])
# alias: model.record(sig, modes)
```
For EEG/MEG proxies, lead-field transforms are separate (`eeg_proxy_transform`,
`meg_proxy_transform`) — not auto-computed on every simulate.

## Level 5 — Objective

```python
obj = jtfne.rate_targets(groups={"L4": idx}, targets_hz={...}, weights={...})
obj = obj.compose(other_obj)          # Objective.compose exists
report = model.evaluate(sig, obj)     # Model.evaluate(signals, objective)
```

**Not public:** `jtfne.band_power`, `jtfne.phase_locking`, objective `+`/`*` algebra.

## Level 6 — Tune / optimize

```python
result = model.tune(obj, optimizer=jtfne.agsdr(...), ...)
# or suite helpers: jtfne.suite2_tune_noise_agsdr_adam(...)
```

**Not public:** `jtfne.optimize(...)`, `optimizer.optimize(...)`, `result.apply(model)`.

**Caveat (verified 2026-06-30):** `Model.tune()`'s differentiable-optimizer
branch (`optax_guarded_path_no_loop_v0.0.8`) is currently a metadata-only guard
that returns `REVISE`/`ACCEPT_CANDIDATE` without ever entering a real gradient
loop — `jaxfne.optim.gsgd.step_gsgd_transform`/`step_sdr_transform`/
`step_gsdr_transform`/`step_agsdr_transform` are NOT invoked by production
tuning despite their docstrings/`OptimizerSpec` names suggesting they are. The
real, wired blackbox path uses `propose_blackbox_candidates` +
`optim/core.py`'s `sdr_transform`/`gsdr_transform`/`agsdr_transform` Optax
factories.

## Multi-trial / batch

```python
batch = jtfne.trial_batch(...)
result = model.run_trials(batch, sim)
# or tutorial_utils.simulate_laminar_trials(model_dict, cfg, n_trials=8)
```

## Level 7 — Manifest / receipt

```python
m = jtfne.manifest(cfg, signals=sig, objective=obj, ...)
jtfne.save_json(m, path)
receipt = model.run_receipt(sig, tags={...})
jtfne.save_receipt(receipt, path, overwrite=False)  # write-once
```

## Full chain

```text
Configuration/NeuronalTensor → construct → simulate → Signals → (vis | probe | objective | manifest)
                                                              ↘ Model.tune → TuneResult
```

## Violations (rewrite if you see these)

- Hand-rolled PSD/raster when `jtfne.vis.*` or `tutorial_utils` pipeline exists
- `signals.rate()` / `signals.probe()` invented methods
- Global `cell_types=` for laminar E:I gradient (use per-layer table, `jaxfne-config`)
- Skipping manifest/receipt on release-facing runs without explicit reason
- Assuming `Model.tune(optimizer="GSGD"/"SDR"/"GSDR"/"AGSDR")` runs a real
  differentiable gradient loop without checking which path is actually wired

## Related skills

- `jaxfne-config` / `jaxfne-neural-tensor` — the two ways to build the `Configuration`/`NeuronalTensor` this chain consumes
- `jaxfne-vis-modules` — the `jtfne.vis.*` readout/plotting surface
