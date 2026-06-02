# Claude Code Context - jaxfne

Last updated: 2026-06-02
Scope: active worker context for repo edits, notebooks, releases, and API migration.

## Fast trigger terms

Use this context whenever a task mentions:

```text
jaxfne, cfg, Config, Configuration, Net, Model, FlatNet, JAX, JIT, pmap, vmap,
AGSDR, trainer, objective, signal, LFP, CSD, EEG, MEG, spectrolaminar,
connectivity, weld, clone, release, PyPI, notebook, Colab
```

## Current architecture target

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

Canonical names:

```text
Config, Net, Paradigm, Objective, Trainer, Signals, FlatNet
```

Compatibility aliases during migration:

```text
Configuration -> Config
Model -> Net
FlatModel -> FlatNet
```

## Required behavior

- Inspect current exports before using or inventing APIs.
- Preserve public APIs with wrappers for at least one release line.
- Keep scientific logic in package modules, not notebooks.
- Keep tutorials package-native: `import jaxfne as jtfne`.
- Future placeholders must raise `NotImplementedError(">TBI-not-ready")`.

## Truth/status posture

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Use status/evidence wording. Do not describe proxy LFP/CSD/EEG/MEG as calibrated, solved, biological, empirical, or physical amplitude readouts.

## Module boundaries

```text
core.py          facade only; no notebook workflows or plotting
config.py        Config and typed sub-specs
net.py           Net, simulate, introspection, to_config
def paradigm.py  trial/task/stimulus schedules
objective.py     metrics, gates, objective outputs
optim/           AGSDR, trainers, training results
signals.py       Signals.get, layout conversion, get_signal
connectivity.py  selectors, mechanisms, rules, compiler
fields/          source/field/probe tensor operators
flatten.py       FlatNet, JIT/pmap maps, flatten/unflatten
vis/             raster/LFP/CSD/EEG/MEG/PSD/spectrogram/suites only
tutorial_utils.py notebook glue and compatibility aliases
```

## JAX rules

- Numerical hot paths use JAX arrays.
- Use explicit PRNG keys.
- Use `lax.scan` for time loops.
- Use `vmap`/`pmap`/`pjit` only around array-native code.
- Keep plotting, JSON, markdown, file I/O, pandas, and Python object mutation outside JIT.
- Split JIT inputs into dynamic arrays and static runtime metadata.

## Notebook/release hard gates

```text
full duration_ms >= 1000
full dt_ms = 0.1
full dtype = float32
deterministic seed
finite outputs
strict JSON
PNG figures required
Plotly HTML optional
SMOKE and FULL notebook receipts for release notebooks
```

## Common failure modes

```text
work in stale branch or wrong repo
invent API without grep/import smoke
put optimizer logic in vis
put plotting code in core
average time-domain trials before PSD
confuse (freq, contact) and (contact, freq)
double-count synaptic/source current
store raw arrays in Config JSON
claim proxy as solved field
upload PyPI with stale version in pyproject.toml
```

## Required report

```text
repo / branch / SHA
changed files
commands run
exact results
runtime facts
status/evidence level
blockers
next safe action
```
