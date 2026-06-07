# AGENTS.md — jaxfne

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

## Current publication state

Publication branch: `cur`

Expected synced SHA:

```text
5c41cd248caf98a4bb5e986eef364d38bc63c79a
```

Inventory:

- 8/8 main figures
- 4/10 Extended Data

Next planned work: ED5 manifest hashes on branch `pub/ed05-manifest-hashes` from `cur`.

See `internal_docs/loop_context/CURRENT_PUBLICATION_STATE.md` for artifact list and sync commands.

## Branch policy

- Publication work lands on `cur`.
- Do not mutate `main`, `dev`, or `agy` without explicit approval.
- Do not force-push.
- Do not tag, release, publish, or build distribution artifacts unless explicitly approved.

## Scientific gates

Preserve:

```text
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Never claim real EEG/MEG, calibrated physical amplitude, solved PDEs, mechanism proof, or biological validation unless the run includes solver, calibration, units, geometry, boundary, gauge, residual, and validation evidence.

## Validation

Common gate:

```bash
python3 -m compileall -q scripts/publication jaxfne tests
python3 scripts/publication_inventory.py
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/test_api_smoke.py tests/test_root_import_lightweight.py -q --tb=line
```

For each publication artifact, run its generator and strict JSON checks for its manifest/receipt.

## Report format

Return:

1. Status
2. Repo state
3. Changed files
4. Commands run
5. Exact results
6. Evidence/truth status
7. Blockers
8. Next safe action
