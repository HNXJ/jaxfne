# JAXFNE Core Doctrine

## Package identity

`jaxfne` is a compact JAX-native TFNE source-to-field/readout scaffold. It is a bridge and evidence-generation layer, proxy-scoped biological simulator or full EEG/MEG forward solver.

## Canonical architecture

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

| Object | Owns | Does not own |
|---|---|---|
| Config | declarative specs, schema version, JSON-safe config | compiled arrays, running state |
| Net | compiled circuit, state init, simulation, introspection | task semantics, optimizer loops |
| Paradigm | trial schedule, events, stimuli, target mapping | circuit structure |
| Objective | metrics, gates, score composition | optimizer search |
| Trainer | candidate generation, tuning loop, save/load | plotting, field solving |
| Signals | tensor outputs, layouts, selectors | plotting style |
| Vis | raster/traces/PSD/spectrogram/suites | simulation or optimizer logic |
| FlatNet | JAX/JIT/pmap arrays and maps | public prose or artifacts |

## Module boundaries

```text
jaxfne.core          facade and minimal public contracts
jaxfne.config        Config and sub-specs
jaxfne.net           Net and compiled circuit behavior
jaxfne.paradigm      task/trial/stimulus scheduling
jaxfne.objective     objective outputs and score reports
jaxfne.optim         optimizers and trainers
jaxfne.signals       Signals and layout conversion
jaxfne.connectivity  selectors, mechanisms, rules, compiler
jaxfne.fields        source/field/probe tensor operators
jaxfne.flatten       FlatNet and tracking maps
jaxfne.vis           visualization only
```

## Truth/status gates

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

Use status/evidence wording. Reserve stronger interpretation for runs with solver, calibration, geometry, boundary, gauge, residual, units, and validation evidence.

## Source bookkeeping

Use one source mode per run:

```text
Mode A: total membrane-current source
Mode B: decomposed electrical-source mode
```

Avoid double-counting synaptic current. Native reduced-emitter current is not amperes unless calibrated. Export source calibration status.

## Probe/readout operators

Required readout kinds:

```text
spk, vm, source, lfp_like, csd_like, eeg_like, meg_like, emm_proxy
```

Each readout report includes shape, units/status, method, assumptions, source calibration, field status, finite-output status, and artifact paths.

## Public API compatibility

- Preserve public names unless a breaking cleanup is explicitly requested.
- Prefer compatibility wrappers for moved helpers.
- Keep optional dependencies lazy.
- Core import must not require visualization extras.
- Reusable visualization belongs in `jaxfne.vis`, not notebooks.
