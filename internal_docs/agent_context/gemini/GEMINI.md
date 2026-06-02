# Gemini Context - jaxfne

Use this file for Gemini Flash/CLI workers on jaxfne repo tasks.

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
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
```

## Validation minimum

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
```
