# Objective Grammar

This guide walks through a **complete script** — from circuit definition to
simulated signals, objectives, tuning, and export — using the real top-level
calls in order.

```text
Configuration → construct() → Paradigm → simulate() → probe() → Objective → tune() → validate → export
```

Related views (unchanged):

- [Configuration Grammar](configuration_grammar.md) — builder methods on `Configuration`
  (`.runtime()`, `.column()`, `.connectivity()`, …).
- [TFNE Operator Doctrine](../operator_doctrine.md) — internal tensor-operator contracts.

Every call below was run against the installed package before being written
here — none of it is illustrative pseudocode.

## The chain

### 1. Configuration

```python
cfg = (
    jtfne.Configuration()
    .runtime(seed=0, duration_ms=200.0, dt_ms=0.5)
    .column(name="V1", layers=["L1", "L4", "L6"], n=60)
    .cell_types({"E": 0.8, "PV": 0.2})
    .set_emitter("izhikevich", "cortical_eig")
    .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
)
```

See [Configuration Grammar](configuration_grammar.md) for the full builder-method
reference.

### 2. construct()

```python
model = jtfne.construct(cfg)
# construct(cfg: Configuration, *, geometry: LaminarSourceGeometry | None = None) -> Model
```

`construct()` is the slow step (tens of seconds at 10k neurons) — build once,
reuse the `Model` across seeds/trials/sweeps.

### 3. Paradigm (optional)

```python
paradigm = jtfne.omission_oddball_paradigm(
    standard_onset_ms=50.0, standard_duration_ms=20.0,
)
```

Other paradigm constructors: `jtfne.paradigm(...)`,
`jtfne.general_sequential_oddball_paradigm(...)` (the backbone for
omission/global/local/DMS/explicit-event-list designs — see the
`jaxfne-paradigm-design` skill). A paradigm is optional; `simulate()` runs a
plain drive-only trial without one.

### 4. simulate()

```python
signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=1, paradigm=paradigm)
# simulate(model: Model, sim: Simulation | None = None, paradigm: Any | None = None, **kwargs) -> Signals
```

`signals` is a `Signals` object; read individual channels with
`signals.get(key)` (keys: `"V_m"`/`"vm"`, `"spikes"`/`"spk"`, `"lfp_contacts"`,
`"csd_contacts"`, `"source"`, ...) or get everything with `signals.summary()`.

### 5. probe()

```python
readout = model.probe(signals, ["spikes", "V_m"])
# Model.probe(self, signals: Signals, modes: Sequence[str] | None = None) -> dict[str, Any]
```

`probe()` lives on `Model`, not on `Signals` — `Signals` only exposes `.get()`
and `.summary()`.

### 6. Objective

```python
nt = model.neuron_table()
groups = {"E": [i for i, row in enumerate(nt) if row["cell_type"] == "E"]}
objective = jtfne.rate_targets(groups, {"E": 10.0})
# rate_targets(groups: dict, targets_hz: dict[str, float], weights: dict | None = None) -> Objective
```

`jtfne.rate_targets(...)` always takes both `groups` *and* `targets_hz` —
`targets_hz` is required, not optional. `jtfne.rate_synchrony_targets(...)` adds
a synchrony (`kappa`) term to the same kind of objective.

### 7. tune()

```python
result = model.tune(objective=objective, optimizer=jtfne.AGSDR(), steps=2, seed=0)
# Model.tune(self, objective=None, optimizer=None, steps=0, seed=0, scope=None,
#            strict=False, simulation=None, parameter=None, bounds=None,
#            parameters=None, generations=None, population_size=None,
#            objectives=None) -> TuneResult
```

Tuning is a method on `Model` (`model.tune(...)`), not a free
`jtfne.optimize(...)` function — there is no top-level `optimize()`.
`jtfne.AGSDR(alpha=0.7, exploration=0.05, deselect_factor=2.0)` is a state spec,
not a stateful optimizer object with its own `.optimize()` method; `model.tune()`
is the entry point that runs it.

### 8. Validate

```python
status = jtfne.validate_configuration(cfg)
# validate_configuration(cfg: Configuration, strict: bool = True) -> dict[str, Any]
# -> {"status": "PASS", "issues": [], "truth_gates": {...}}
```

Other validation entry points: `jtfne.validate_config(cfg)` for the legacy
`JaxFNEConfig` path, `jtfne.validate_source_field_status(...)` for field-stage
truth gates, `jtfne.validation_report(config_valid, issues, metadata)` for a
freeform validation record.

### 9. Export

```python
man = jtfne.manifest(cfg, signals=signals)
# manifest(cfg, signals=None, readout=None, runtime_config=None, paradigm=None,
#          objective=None, evaluation=None, tuning=None, dataset=None) -> dict[str, Any]
jtfne.save_json(man, "run_manifest.json")
```

`jtfne.manifest(...)` returns a plain JSON-safe `dict` — there is no
`Manifest` object with `.save()`/`.validate()` methods. Persist it with
`jtfne.save_json(obj, path)` (`allow_nan=False`), or use
`jtfne.export_tutorial_artifacts(output_dir, manifest=..., metrics=..., validation=...)`
to write a full artifact bundle at once.

## Why this page exists separately

`Configuration` exposes ~30 builder methods; the operator doctrine describes
7 tensor stages with domain/codomain contracts. Neither answers "what do I
actually type, in order, to run something." This page is that answer, kept
honest by running every example against the installed package rather than
transcribing it from a proposal or an older skill file.

## See also

- [Configuration Grammar](configuration_grammar.md) — the builder-method reference this chain's step 1 draws from.
- [TFNE Operator Doctrine](../operator_doctrine.md) — the tensor contract each stage satisfies internally.
- [Tensor Operator Registry](../api/tensor_operators.md) — per-operator inventory.
- [Operator Inventory (generated)](../_generated/operator_inventory.md) — the full live export surface, regenerated from code.
