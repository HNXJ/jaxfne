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

## ⚡ Fast-path doctrine (intelligence-per-symbol · per-watt) — all agents

Fast executive models: act on this block; read the rest only when blocked.
1. **Verify-before-call.** Never name a skill/API/flag/path/branch from memory — grep `__all__` / the contract first. Fabrication→fail→retry is the costliest loop.
2. **Contract-first.** Read the frozen contract (docstrings + `>TBI-not-ready`), not the 1000-line module.
3. **Skill = cache.** Run the skill; don't re-reason its checklist.
4. **Receipt, not justification.** Show `command` + `N passed`. Never report what you didn't run.
5. **Route by altitude.** Opus=architecture/contracts/truth-gates · Sonnet=implement vs contract+skill · notebook=cross-file synthesis + batch edits.

JAX/jaxfne: `N_compile<=1` (recompilation is the silent watt+tok sink) · stable shapes+dtypes · jit/vmap on (`runtime_report()`) · x64 at startup only · sparse>dense · `segment_sum`+`lax.scan` · pytree children=dynamic, aux=static. Depth → jaxfne skills + `~/.claude/CLAUDE.md`.


## Durable Context Hygiene (Required by tests)

- **Project identity**: jaxfne version 0.3.5.
- **Repository**: internal_docs vs .legacy/internal_docs.
- **API contract**: contains public wording and API contracts. Configuration() vs Config, model = jtfne.construct(cfg), signals = jtfne.simulate(model), signals.field, .runtime(, .column(, .cell_types(, .probes(.
- **Public vs private**: public docs, tutorial files, private internal_docs.
- **Tutorial deliverable**: tutorial milestone, generated output, tutorial_outputs directory, JAXFNE_VALIDATE_TUTORIAL_OUTPUTS gate.
- **Validation**: SHA, branch, test, report, receipt.
- **Failure mode**: jbiophysic, stale artifact, low-level kernel mistakes.
- **Always do**: always follow the rules.
- **Never do**: never fail.

