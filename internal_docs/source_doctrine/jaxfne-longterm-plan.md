# JAXFNE Long-Term Plan

## Thesis

`jaxfne` should stay compact: a JAX-native bridge from emitters to source/field/probe readouts, objectives, optimization, visualizations, and evidence reports.

## Version-line roles

| Line | Role |
|---|---|
| v0.3.x | Tutorial atlas, Etudes, proxy readout hardening, package API cleanup. |
| v0.4.x | Experimental physical field solvers with residual, boundary, gauge, and convergence evidence. |
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
stable source schema
field metadata schema
boundary/gauge doctrine
source conservation tests
manifest validators
proxy-vs-solver API separation
notebook evidence receipts
```

## Refactor direction

Before shortening package code:

```text
inventory every function/class
map public API exports
find duplicate helper families
add tests for touched public behavior
keep wrappers for moved public functions
run full release gates
```

Likely consolidation areas:

```text
metric registry shared by objectives and tutorials
JSON-safe export helpers
truth-gate validation helpers
visualization input coercion
optimizer report schemas
```

## Ecosystem position

- Jaxley: detailed differentiable cell/compartment modeling.
- LFPy and EEG/MEG tools: detailed external forward modeling.
- `jaxfne`: compact composition, tutorial evidence, source/field/probe scaffolding, and optimizer workflows.