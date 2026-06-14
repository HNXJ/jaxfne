# AGENTS.md — jaxfne

## Read first

[`internal_docs/loop_context/AGENT_QUICKREF.md`](internal_docs/loop_context/AGENT_QUICKREF.md)

Evidence snapshot (refresh SHA each session): [`internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md`](internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md)

Deep reference (on demand): [`internal_docs/loop_context/JAXFNE_BIOPHYSICS_GLOSSARY.md`](internal_docs/loop_context/JAXFNE_BIOPHYSICS_GLOSSARY.md)

## Identity

jaxfne is a compact JAX-native TFNE scaffold.

```python
import jaxfne as jtfne
```

Public flow: `Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export` (package computes; notebooks configure/plot/export).

## Gates

```text
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

## Branch policy

Evidence on `cur`. Permanent branches: `main`, `dev`, `agy`, `cur`. Do not mutate `main`/`dev`/`agy` without approval. No force-push, tag, release, or publish without approval.

## Validation

```bash
python3 scripts/evidence_inventory.py
# scripts/evidence_figures_inventory.py is kept as a compatibility wrapper.
python3 -m compileall -q scripts/evidence_figures jaxfne tests
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py tests/test_signals_get_v0329.py -q --tb=short
```

## Report format

Status, repo state, changed files, commands run, exact results, evidence/truth status, blockers, next safe action.

## API catalog (read before writing helpers)

Before writing any jaxfne helper or hand-rolling PSD/raster/LFP-proxy/CSD-proxy/
EEG-proxy/MEG-proxy/spectrolaminar/AGSDR/manifest logic, consult the curated
lookup table: [`internal_docs/JAXFNE_AGENT_API_CATALOG.md`](internal_docs/JAXFNE_AGENT_API_CATALOG.md).
It lists the package-native functions (incl. the exact spectrolaminar pipeline)
so existing APIs are reused, not rediscovered. Canonical import: `import jaxfne as jtfne`.
