# jaxfne Source Doctrine

This folder is the active compact doctrine for jaxfne work.

## Active architecture

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

## Files

| File | Purpose |
|---|---|
| `CLAUDE.md` | Worker operating contract and stop conditions. |
| `jaxfne-core-doctrine.md` | Package identity, module boundaries, truth/status posture. |
| `jaxfne-equations-runtime-validation.md` | Tensor equations, JAX/JIT rules, schemas, validation gates. |
| `jaxfne-tutorial-etude-atlas.md` | Tutorial/Etude rules, notebook gates, artifacts, visualization doctrine. |
| `jaxfne-longterm-plan.md` | 0.3.28+ release ladder and 0.4 solver entry criteria. |

## Canonical names

```text
Config, Net, Paradigm, Objective, Trainer, Signals, FlatNet
```

Compatibility aliases:

```text
Configuration -> Config
Model -> Net
FlatModel -> FlatNet
```

## Default status

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

## Immediate rules

- Canonical import: `import jaxfne as jtfne`.
- Config is the source of declarative circuit/task/training state.
- Net is the compiled implementation.
- Paradigm owns task/trial/stimulus schedules.
- Objective owns measures and gates.
- Trainer owns optimization loops.
- Signals owns tensor outputs and layout queries.
- Vis owns plotting only.
- Core stays small: facade and tensor-field contracts only.
- Plotting glue may exist in notebooks; simulation/readout/objective logic belongs in package code.
