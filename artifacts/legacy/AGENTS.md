# jaxfne Agent Coordination

Protocol version: 3.0
active line: 0.3.28+
status: config-first tensor-field scaffold

## Mission

Move jaxfne from notebook-local wiring to package-level objects:

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

Canonical import remains:

```python
import jaxfne as jtfne
```

## Required start checks

Run before changing code or reports:

```bash
git fetch origin --tags
git status --short
git branch --show-current
git rev-parse HEAD
python - <<'PY'
import jaxfne as jtfne
print(getattr(jtfne, "__version__", "unknown"))
PY
```

## Branch and release flow

```text
feature branch -> dev review -> main after receipts -> tag -> package upload
```

Remote mutation requires explicit user authorization:

```text
git push origin <branch>
git tag / tag delete / tag push
GitHub Release create/edit/publish
TestPyPI or PyPI upload
force-push to any protected branch
```

Do not force-push `main`.
Use immutable SHA URLs for final public audits.

## Object vocabulary

Canonical 0.3.28+ names:

```text
Config      declarative circuit/task/training spec
Net         compiled biophysical circuit
Paradigm    task/trial/stimulus schedule
Objective   metric/gate computation
Trainer     optimization/tuning loop
Signals     tensor outputs and query API
FlatNet     JAX/JIT/pmap-ready array form
```

Compatibility aliases for one release line:

```text
Configuration -> Config
Model -> Net
FlatModel -> FlatNet
```

## Module ownership

```text
jaxfne.core          facade, small public contracts only
jaxfne.config        Config and typed sub-specs
jaxfne.net           Net construction/simulation/introspection
jaxfne.paradigm      tasks, trials, stimuli, schedules
jaxfne.objective     objective outputs and scores
jaxfne.optim         trainers and optimizers
jaxfne.signals       signal containers, layout conversion, get_signal
jaxfne.connectivity  selectors, mechanisms, connection compiler
jaxfne.fields        source/field/probe tensor operators
jaxfne.flatten       FlatNet and tracking maps
jaxfne.vis           visualization only
tutorial_utils       notebook glue and compatibility wrappers only
```

## Truth/status gates

Keep these unless a run has solver, calibration, boundary, gauge, units, residual, convergence, and validation evidence:

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

## Release gates

Before tag/package upload:

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
PYTHONPATH=. TFNE_SMOKE=1 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_smoke.ipynb
PYTHONPATH=. TFNE_SMOKE=0 jupyter nbconvert --to notebook --execute tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb --output /tmp/etude1_full.ipynb
python -m build
python -m twine check dist/*
```

JSON checks after notebook runs:

```bash
python -m json.tool outputs/jaxfne_etude_no_1/manifest.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/validation_report.json >/dev/null
python -m json.tool outputs/jaxfne_etude_no_1/metrics.json >/dev/null
```

## Worker report format

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

Treat test counts as unverified until exact commands, branch/SHA, and receipts are shown.

## Stop conditions

Stop and report when any appear:

```text
invented public API
hidden local simulator/source/readout/objective engine
NaN/Inf export
proxy path described as solved field
uncalibrated source described as physical amplitude
silent placeholder success
test changed before failure provenance is known
non-JSON-safe array stored directly in Config
plotting/file I/O/JSON inside JIT path
release tag or upload with stale pyproject version
```
