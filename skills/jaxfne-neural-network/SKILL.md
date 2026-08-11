---
name: jaxfne-neural-network
description: >-
  Use the current jaxfne runtime chain: construct, simulate, Signals, probes,
  objectives, tuning, and manifests. Use when running or reading a Model.
---

# jaxfne runtime procedure

Read `catalog-glossary-jaxfne`, then the configuration or tensor procedure that
produced the input. Mathematical stage meaning belongs to the project source
documents.

## Model

```python
model = jtfne.construct(specification)
table = model.neuron_table()
signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.5, seed=0)
```

`Model` is the runnable compiled object and `Net` is a compatibility alias.
There is no general public `Model.__call__`; use the simulation entrypoints.
Construction may be expensive; reuse the model when structure is unchanged.

## Signals

Use the current `Signals` contract:

```python
signals.get("vm")
signals.get("spk")
signals.get("lfp")
signals.get("csd")
signals.summary()
```

Verify exact keys and shapes with `signals.metadata` and tests. Trial selection
through `Signals.get(trial=...)` is not a general contract; use the supported
trial APIs when a trial axis is required.

## Probes and objectives

Keep stages separate:

```python
readout = model.probe(signals, modes=["lfp_proxy", "csd_proxy"])
objective = jtfne.rate_targets(groups, targets_hz)
report = model.evaluate(signals, objective)
```

Verify objective signatures before use. Do not invent `Signals.rate()`,
`Signals.psd()`, `jtfne.optimize()`, or an optimizer object method that the
live API does not expose.

## Tuning

`Model.tune(...)` is the tuning entrypoint and returns a `TuneResult`.
Verify the selected optimizer path and its evidence before describing it as a
gradient or differentiable result. Do not infer numerical behavior from class
names or docstrings alone.

## Multi-trial paths

- Typed model path: `TrialBatch`/`Model.run_trials(...)`.
- Tutorial path: `tutorial_utils.simulate_laminar_trials(...)` with its model
  dictionary.

Do not mix those input contracts.

## Manifest and receipts

```python
manifest = jtfne.manifest(cfg, signals=signals)
jtfne.save_json(manifest, path)
```

Use write-once receipts for run-facing artifacts. Require finite JSON and
preserve current status metadata.

## Validation

Test through the public path that users will call. For changed behavior report:

```text
API delta:
Mathematical delta:
Numerical delta:
Claim/evidence delta:
Documentation delta:
Compatibility delta:
```

Current implementation truth is in `jaxfne/_model*.py`, `jaxfne/_signals.py`,
`jaxfne/_construct*.py`, `jaxfne/io.py`, and tests.

## Continuation verification

Treat `with_hdp_initial_state(H0=..., w0=...)` as partial H-state/plastic
initialization. Exact recurrent continuation requires the selected kernel's
complete dynamic carry plus deterministic stochastic sequencing. Keep the
H-state leaf shape-preserving and opaque; the current scalar HDP coordinate is
only the implemented special case.
Recover the live step function before naming the state; classify evidence as
`SPECIFIED`, `IMPLEMENTED`, `TESTED`, or `OBSERVED`, and report the command
receipt for each executable claim. Prefer the existing `DynamicState`/
`ContinuationState` path and compare uninterrupted versus segmented voltage,
spikes, source, H-state, weight, and exposed synaptic-state outputs. Treat
internal coordinates as relative until an explicit calibration boundary; a
changed seed is a negative control, not a continuation strategy.
