# JAXFNE Long-Term Plan

## Thesis

`jaxfne` should stay compact: a JAX-native bridge from emitters to source/field/probe readouts, objectives, optimization, visualizations, and evidence reports.

## 0.3.28-0.3.34 release ladder

| Version | Theme | Required gate |
|---:|---|---|
| 0.3.28 | Config owns all circuit declarations | No essential notebook-local circuit state. |
| 0.3.29 | Stable identity/selectors | Every node has area-id-layer-type quartet. |
| 0.3.30 | Connection rules | Typed pre-selector -> post-selector rules compile to finite edges. |
| 0.3.31 | Weld | Configs merge with deterministic renaming and rewritten selectors. |
| 0.3.32 | Construct/reconstruct/clone | Config -> Net -> Config roundtrip works; minimal Paradigm exists. |
| 0.3.33 | FlatNet | JAX/JIT/pmap-ready arrays with 1-to-1 tracking maps. |
| 0.3.34 | Integration gate | 35 conditions pass. |

## Background work scheduled across ladder

```text
0.3.29: Signals.get basic layouts and model selectors
0.3.30: connection introspection and weight maps
0.3.31: AGSDRTrainer.from_config skeleton
0.3.32: TrainingResult save/load and constant-DC Paradigm
0.3.33: unified vis aliases and package-level relative-power modes
0.3.34: final 35-condition gate
```

## Version-line roles

| Line | Role |
|---|---|
| v0.3.x | Config-first circuits, tutorial atlas, proxy readouts, trainers, FlatNet. |
| v0.4.x | Experimental field solvers with residual, boundary, gauge, convergence evidence. |
| v0.5.x+ | External comparisons, calibration workflows, uncertainty, inverse modeling. |

## Evidence ladder

```text
mathematical consistency
-> electromagnetic admissibility
-> numerical convergence
-> external-tool or empirical comparison
-> mechanism support through perturbation/model comparison
```

## Solver entry criteria

Open v0.4 solver implementation only after v0.3.x has:

```text
stable Config schema and schema_version
stable source schema
stable field metadata schema
boundary/gauge doctrine
source conservation tests
manifest validators
proxy-vs-solver API separation
FlatNet identity maps
notebook evidence receipts
```

## Refactor direction

Before moving package code:

```text
inventory every function/class
map public exports
find duplicate helpers
add tests for touched public behavior
keep wrappers for moved public functions
run full release gates
```

Likely consolidation areas:

```text
metric registry shared by objectives and tutorials
JSON-safe export helpers
status metadata validation helpers
visualization input coercion
optimizer/trainer report schemas
connectivity rule compiler
signal layout conversion
```
