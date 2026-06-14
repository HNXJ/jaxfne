# jaxfne agent quickref

**Read first.** Deep reference: [`JAXFNE_BIOPHYSICS_GLOSSARY.md`](JAXFNE_BIOPHYSICS_GLOSSARY.md). Live publication snapshot: [`CURRENT_PUBLICATION_STATE.md`](CURRENT_PUBLICATION_STATE.md).

## Freeze (every session)

```bash
git fetch --all --prune
git branch --show-current
git status --short
git rev-parse HEAD
python3 scripts/evidence_inventory.py
# scripts/evidence_figures_inventory.py is kept as a compatibility wrapper.
```

Publication track: branch `cur`. Re-freeze SHA and inventory before citing counts or technical report receipts.

## Import and public path

```python
import jaxfne as jtfne

cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
readouts = model.probe(signals, ...)
```

**Two grammars (same package, different layers):**

| Layer | Grammar |
|-------|---------|
| Public OO API | `Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export` |
| Scientific operators | `Emitter -> Source -> Field -> Probe -> Objective -> Optimizer` |

Tutorials configure, plot, and export. The package is the engine — no notebook-local solvers, readout engines, objective engines, or optimizer engines.

**Obsolete in public examples:** `jtfne.Model(cfg)`, `jtfne.Config.four_celltype`, bare `n_e`/`duration`/`dt` config fields without verifying live API.

## Truth gates (preserve in manifests)

```text
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Never claim real EEG/MEG, calibrated amplitude, biological metabolism, mechanism proof, or solved PDE/Maxwell/Poisson unless the run includes solver, geometry, boundary, gauge, residual, units, calibration, and validation evidence.

Mechanism support requires nulls, ablations, repeated seeds, and empirical comparison — not objective success alone.

## Branch policy

- Publication work: `cur`.
- Permanent branches: `main`, `dev`, `agy`, `cur`.
- Do not mutate `main`, `dev`, or `agy` without explicit approval.
- No force-push, tags, releases, wheel publish, or archive/DOI without explicit approval.
- After merge to a permanent branch, delete the source feature branch locally and on `origin` unless retained.

## Publication posture (verify live)

Expected on `cur` after `evidence_inventory.py`:

```text
main figures: 8/8
extended data: 10/10
```

ED1–ED10 complete. Release/tag/publish/archive remain approval-gated (ED10 is evidence receipt only).

## Smoke validation

```bash
python3 -m compileall -q scripts/evidence_figures jaxfne tests
python3 -m mkdocs build --strict
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py tests/test_signals_get_v0329.py \
  -q --tb=short
```

Full suite and build ladder: [`06_VALIDATION_LADDER.md`](06_VALIDATION_LADDER.md). Re-freeze pytest counts; do not cite doc tables without a live run.

Optional-dep laziness: run root-import tests in a **clean** venv (preloaded pandas invalidates the gate).

## API discipline

- Preserve public APIs; add wrappers instead of removals.
- Optional dependencies stay lazy (Jaxley, PyNWB, etc.).
- Verify `jtfne.*` with `hasattr` before recommending symbols.
- No test edits until failure provenance is known (command + traceback).

## Stop rules

Stop and report if:

- wrong branch for publication work
- invented public API or notebook-local scientific engine
- NaN/Inf in JSON exports
- proxy readout described as solved field or physical amplitude
- mechanism claim from objective alone
- optional dependency eagerly imported at root
- tag/release/publish/archive without explicit approval

## Report format

Status, repo state (branch, SHA, dirty/clean), changed files, commands run, exact results, evidence/truth status, blockers, next safe action.

## On-demand skills

`internal_docs/skills/README.md` — api-truth, test-runner, jax-lint, evidence-validator, release-mutation-guard.
