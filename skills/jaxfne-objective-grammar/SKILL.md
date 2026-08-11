---
name: jaxfne-objective-grammar
description: >-
  Route jaxfne software stages from a supported CircuitSpec through
  construction, simulation, Signals, readouts, objectives, tuning, and
  manifests. Use when structuring a script, notebook, pipeline, or API change.
---

# jaxfne objective-stage routing

This is a procedure. Objective, null, optimizer, and operator meaning belongs
to the project mathematical source documents and the live implementation.

## Software chain

```text
Configuration/NeuronalTensor -> construct -> Model -> simulate -> Signals
                                                     -> readout/objective/manifest
                                                     -> Model.tune -> TuneResult
```

Use `import jaxfne as jtfne`. Verify the exact public symbol before calling it.

## Route by stage

- configuration builder → `jaxfne-config`
- tensor or HDP path → `jaxfne-neural-tensor`
- model, simulation, Signals, probe, objective, or tuning → `jaxfne-neural-network`
- selector/connectivity/schema → `jaxfne-modeling-optimization-schema`
- visualization → `jaxfne-vis-modules`
- notebook, documentation, or artifact validation → `jaxfne-notebook-release-gate`

## Procedure

1. Identify the specification tier already held by the caller.
2. Build through `construct()` rather than a local numerical engine.
3. Keep simulation output in `Signals`.
4. Keep readout and objective evaluation separate from simulation.
5. Keep optimization and manifest/export separate from plotting.
6. Reuse the model for repeated runs when structure is unchanged.
7. Add a targeted test at the public entry point for changed behavior.

## Common invalid assumptions

- `Signals` does not gain invented `.rate()`, `.psd()`, or `.probe()` methods.
- There is no generic top-level `jtfne.optimize()` contract.
- Do not infer a typed `Configuration.circuit`, `.objective`, or `.optimizer`
  sub-spec from conceptual vocabulary.
- Do not hand-roll PSD, raster, field, or manifest logic when a package API
  already owns that computation.

## Evidence

Use `docs/guides/objective_grammar.md` for executable examples and
`docs/operator_doctrine.md` for stage contracts. Code and tests decide what is
currently implemented.
