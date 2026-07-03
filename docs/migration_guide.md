# Migration Guide: Configuration → NeuronalTensor

As of 0.4.7, **`NeuronalTensor` + `RuntimeConfiguration` is the preferred way
to define and run a circuit.** `Configuration` (the original fluent builder)
remains fully supported — nothing built on it breaks — but it is no longer
the primary teaching path in the README, quickstart, or new tutorials.

This page is deliberately short: the two paths converge on the same `Model`
via `construct()`, so there is no data migration, no breaking change, and no
deprecation timer. This is a recommendation about which path to *start* with,
not a requirement to rewrite anything that already works.

## Preferred: `NeuronalTensor` + `RuntimeConfiguration`

```python
import jaxfne as jtfne

tensor  = jtfne.load("circuit.json")                 # or jtfne.load_canonical_neuronal_tensor(name)
runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5)
model   = jtfne.construct(tensor, runtime)
signals = jtfne.simulate(model)
```

Use this when:

- You want an explicit, declarative `Areas x Layers x NeuronTypes` circuit
  definition that round-trips through JSON (`save_neuronal_tensor`/`load_neuronal_tensor`).
- You want one of the packaged canonical circuits
  (`jtfne.list_canonical_neuronal_tensors()` / `jtfne.load_canonical_neuronal_tensor(name)`).
- You need **HDP homeostatic plasticity** (synaptic + per-cell-type H-factor
  adaptation) — see the [HDP guide](guides/hdp.md) § "Tensor-first".
- You are writing new code with no existing `Configuration` investment.

Full reference: [NeuronalTensor API](api/neuronal_tensor.md) ·
[Configuration Grammar guide](guides/configuration_grammar.md) § "NeuronalTensor: a second on-ramp" ·
runnable example: [`examples/08_neuronal_tensor_first.py`](https://github.com/HNXJ/jaxfne/blob/main/examples/08_neuronal_tensor_first.py) /
[`tutorials/jaxfne_neuronal_tensor_first.ipynb`](https://github.com/HNXJ/jaxfne/blob/main/tutorials/jaxfne_neuronal_tensor_first.ipynb).

## Supported: `Configuration`

```python
import jaxfne as jtfne

cfg = (jtfne.build_laminar_column(n=1000, ei_profile="canonical")
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16))
model   = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

`Configuration` keeps working exactly as before — no wrapper, no shim, it is
the same code path it always was. Use it when:

- You have existing `Configuration`-based code or notebooks; there is no need
  to rewrite working code to adopt this guide.
- You're doing multi-trial spectrolaminar sweeps via `jaxfne.tutorial_utils`
  (`make_laminar_column_config` → `build_laminar_column` → `simulate_laminar_trials`),
  which is built on `Configuration` and returns a plain `dict`, not `Model`/`Signals`.
- You prefer the fluent method-chaining style for incremental, exploratory
  circuit construction.

Full reference: [Configuration Grammar guide](guides/configuration_grammar.md).

## What does *not* change either way

Both paths converge on the same `Model`, so everything downstream is
identical regardless of which one built it:

```text
Model -> simulate() -> Signals -> probe -> Objective -> Optimizer -> Manifest
```

`Model.tune()`, `with_emitter_parameters()`, `jaxfne.vis.*`, `manifest()`, and
every probe/readout/objective function are methods on `Model`/`Signals` — they
don't know or care whether `Model` came from a `Configuration` or a
`NeuronalTensor`. There is nothing to migrate in your tuning, visualization,
or export code either way.

## Quick comparison

| | `NeuronalTensor` (preferred) | `Configuration` (supported) |
|---|---|---|
| Style | Declarative data model (`Area`, `Layer`, `NeuronType`) | Fluent method chain (`.runtime().column().cell_types()...`) |
| Runtime object | `RuntimeConfiguration` (no HDP field) | `RuntimeConfig` (has `enable_hdp`/`hdp_params`) |
| JSON round-trip | `save_neuronal_tensor`/`load_neuronal_tensor`, or `jtfne.load`/canonical loaders | No dedicated method — verified generic path: `json.loads(json.dumps(dataclasses.asdict(cfg)))` then `Configuration(**loaded)` reconstructs an equal object |
| HDP homeostatic plasticity | Native (pass `runtime=RuntimeConfig(enable_hdp=True, ...)` override to `simulate()`) | Same override pattern works identically |
| Multi-trial spectrolaminar sweeps | Not yet a dedicated tensor-first helper — use `Configuration` via `tutorial_utils` | `make_laminar_column_config` → `build_laminar_column` → `simulate_laminar_trials` |
| Result type | `Model` / `Signals` | `Model` / `Signals` (or plain `dict` via `tutorial_utils`) |

If your task doesn't clearly favor one side of this table, default to
`NeuronalTensor` for new code.
