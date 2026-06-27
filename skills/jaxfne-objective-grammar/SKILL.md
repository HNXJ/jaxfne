---
name: jaxfne-objective-grammar
description: >-
  Mandatory object-transform grammar for jaxfne workflows — verified public chain
  from Configuration through simulate, probe/readout, tune, and manifest. USE
  when structuring scripts, notebooks, or new APIs. Do NOT use invented
  signals.rate() / jtfne.optimize() patterns from legacy drafts.
---

# jaxfne Objective Grammar

USE FIRST: `catalog-glossary-jaxfne`, `10_objective_grammar.md`, `jaxfne-modeling-optimization-schema`.

## Mandatory chain (verified public surface)

```text
Configuration → construct → simulate → Signals → (vis | probe | objective | manifest)
                                    ↘ Model.tune → TuneResult
```

Legacy aliases still import: `Configuration`/`Config`, `Model`/`Net`.

## Level 0 — entry

```python
import jaxfne as jtfne
jtfne.enable_x64()  # before array construction if x64 needed
```

## Level 1 — Configuration

Build via documented builders (prefer over empty `Configuration()` unless extending core):

```python
cfg = jtfne.laminar_cortex_config(...)  # or build_laminar_column, suite2_*, load_config
cfg = cfg.layer_fractions(...).area_layer_cell_types(...).runtime(...).homeostasis(...)
model = jtfne.construct(cfg)
```

Fluent methods on `Configuration` that **exist** (each returns a new `Configuration`):
`runtime`, `field`, `probe`, `column`, `layer_fractions`, `area_layer_cell_types`,
`cell_types` (global E/PV/SST/VIP fractions only), `connectivity`, `set_emitter`,
`set_probes`, `plasticity`, `homeostasis`, `hdp`, `cell_params`, `areas`.

Per-layer composition uses **`.area_layer_cell_types(area, {...})`**, not
`.cell_types({per-layer dict})`.

## Level 2 — Model

```python
model = jtfne.construct(cfg)
nt = model.neuron_table()       # list[dict] rows
idx = model.select(layer="L4", cell_type="E")
model = model.with_emitter_parameters(drive_per_neuron=arr)
```

## Level 3 — Simulation → Signals

**Canonical:** top-level simulate (not notebook-local kernels):

```python
sig = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
# or
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.5, seed=0)
sig = jtfne.simulate(model, sim=sim)
```

`Model.simulate(...)` and `Model.run_trials(...)` exist for batch/condition paths.

Read signals:

```python
vm = sig.get("vm", layer="L4", cell_type="E")
spk = sig.get("spk")
lfp = sig.get("lfp")  # proxy readout when field computed
summary = sig.summary()
```

**Not on `Signals`:** `.rate()`, `.psd()`, `.bandpower()`, `.probe()` — use
`jtfne.vis.rate(sig)`, `jtfne.vis.psd(sig)`, `jtfne.kappa_synchrony(...)`, or
`model.probe(sig, modes=[...])`.

## Level 4 — Probe / readout

```python
readout = model.probe(sig, modes=["lfp_proxy", "csd_proxy", "source_native"])
# or package vis (proxy-safe figures)
fig = jtfne.vis.lfp(sig)
fig = jtfne.vis.spectrolaminar_suite(sig)
```

## Level 5 — Objective

Construct from verified helpers:

```python
obj = jtfne.rate_targets(groups={"L4": idx}, targets_hz={...}, weights={...})
obj = obj.compose(other_obj)  # Objective.compose exists
report = model.evaluate(sig, obj)  # Model.evaluate(signals, objective)
```

**Not public top-level:** `jtfne.band_power`, `jtfne.phase_locking`, objective `+`/`*` algebra.

## Level 6 — Tune / optimize

```python
result = model.tune(obj, optimizer=jtfne.agsdr(...), ...)
# or suite helpers: jtfne.suite2_tune_noise_agsdr_adam(...)
```

**Not public:** `jtfne.optimize(...)`, `optimizer.optimize(...)`, `result.apply(model)`.

## Level 7 — Manifest / receipt

```python
m = jtfne.manifest(cfg, signals=sig, objective=obj, ...)
jtfne.save_json(m, path)
receipt = model.run_receipt(sig, tags={...})
jtfne.save_receipt(receipt, path, overwrite=False)
```

## Violations (rewrite if you see these)

- Hand-rolled PSD/raster when `jtfne.vis.*` or `tutorial_utils` pipeline exists
- `signals.rate()` / `signals.probe()` invented methods
- Global `cell_types=` for laminar E:I gradient (use per-layer table)
- Skipping manifest/receipt on release-facing runs without explicit reason

## Related

- Multi-trial spectrolaminar: `tutorial_utils` path (`catalog-glossary-jaxfne` §2)
- Tensor build: `NeuronalTensor` path (`catalog-glossary-jaxfne` §1b)
- Open contradictions: `skills/FRICTIONS_STACK.md`
