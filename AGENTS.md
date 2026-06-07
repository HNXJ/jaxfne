# AGENTS.md — jaxfne

## Primary agent anchor

Read first: [`internal_docs/loop_context/JAXFNE_BIOPHYSICS_GLOSSARY.md`](internal_docs/loop_context/JAXFNE_BIOPHYSICS_GLOSSARY.md)

Single consolidated context for operator grammar, truth gates, publication scoreboard (25 factors), ED ladder, JAX discipline, Jaxley/PyNWB rules, stop rules, and Cursor prompts.

Publication snapshot: [`internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md`](internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md)

## Project identity

jaxfne is a compact JAX-native TFNE scaffold.

Canonical import:

```python
import jaxfne as jtfne
```

Core flow:

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

## Branch policy

- Publication work lands on `cur`.
- Permanent branches: `main`, `dev`, `agy`, `cur`.
- Do not mutate `main`, `dev`, or `agy` without explicit approval.
- Do not force-push, tag, release, publish, or build distribution artifacts unless explicitly approved.
- **After merge to a permanent branch, delete the source feature branch locally and on `origin` unless explicitly retained.**

## Scientific gates

```text
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

## Validation

```bash
python3 scripts/publication_inventory.py
python3 -m compileall -q scripts/publication jaxfne tests
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py -q --tb=line
```

## Report format

Status, repo state, changed files, commands run, exact results, evidence/truth status, blockers, next safe action.
