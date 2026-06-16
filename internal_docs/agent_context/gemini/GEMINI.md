# notebook Context - jaxfne

Use this file for notebook Flash/CLI workers on jaxfne repo tasks.

## Invocation summary

jaxfne is a JAX-native computational scaffold moving to a config-first architecture:

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

Trigger on these terms:

```text
cfg, Config, Net, Model, Paradigm, Objective, Trainer, Signals, FlatNet,
connectivity, selector, quartet, weld, clone, JAX, JIT, vmap, pmap,
LFP, CSD, EEG, MEG, spectrolaminar, raster, PSD, release, notebook, PyPI
```

## Decision rules

1. Route the task to the correct module before editing.
2. Inspect existing code and tests before creating APIs.
3. Use canonical import: `import jaxfne as jtfne`.
4. Preserve wrappers for moved public APIs.
5. Make unimplemented future APIs fail with `NotImplementedError(">TBI-not-ready")`.
6. Report exact commands, outputs, branch, and SHA.

## Active names

Canonical:

```text
Config, Net, FlatNet
```

Aliases during migration:

```text
Configuration, Model, FlatModel
```

## Keep out of public claims

Do not claim:

```text
real EEG/MEG
calibrated amplitude
solved Maxwell/Poisson/PDE
validated biological mechanism
physical source units
```

Default status:

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
physical_amplitude_calibrated: false
```

## Validation minimum

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
```

## ⚡ Fast-path doctrine (intelligence-per-token · per-watt) — all agents

Fast executive models: act on this block; read the rest only when blocked.
1. **Verify-before-call.** Never name a skill/API/flag/path/branch from memory — grep `__all__` / the contract first. Fabrication→fail→retry is the costliest loop.
2. **Contract-first.** Read the frozen contract (docstrings + `>TBI-not-ready`), not the 1000-line module.
3. **Skill = cache.** Run the skill; don't re-reason its checklist.
4. **Receipt, not justification.** Show `command` + `N passed`. Never report what you didn't run.
5. **Route by altitude.** Opus=architecture/contracts/truth-gates · Sonnet=implement vs contract+skill · notebook=cross-file synthesis + batch edits.

JAX/jaxfne: `N_compile<=1` (recompilation is the silent watt+token sink) · stable shapes+dtypes · jit/vmap on (`runtime_report()`) · x64 at startup only · sparse>dense · `segment_sum`+`lax.scan` · pytree children=dynamic, aux=static. Depth → jaxfne skills + `~/.claude/CLAUDE.md`.
