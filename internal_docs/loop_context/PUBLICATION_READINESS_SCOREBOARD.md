# Publication readiness scoreboard — jaxfne

Last updated: after ED5 on `pub/ed05-manifest-hashes`.

## Current state summary

| Field | Value |
|---|---|
| Branch | `pub/ed05-manifest-hashes` (publication work from `cur`) |
| SHA | report `git rev-parse HEAD` at run time |
| Inventory | 8/8 main figures; 5/10 Extended Data after ED5 |
| Main figures | complete (fig01–fig08) |
| Extended Data | ED1–ED5 complete; ED6–ED10 pending |
| Next planned ED | ED7 probe/readout contract matrix (`ed07_probe_operator_contracts`) |

Sync:

```bash
git fetch --all --prune
git switch cur
git pull --ff-only origin cur
python3 scripts/publication_inventory.py
```

## Core thesis

jaxfne makes emitter → source → field/probe assumptions explicit, executable, auditable, and hashable.

It is a JAX-native computational scaffold, not a validated EEG/MEG solver.

Physical amplitude and mechanism claims require calibration, geometry, boundary/gauge, solver residuals, nulls, ablations, repeated seeds, and empirical comparison.

## Scoreboard

| # | Factor | score_now | target_score | evidence_needed | next_action | stop_condition |
|---|---|---:|---:|---|---|---|
| 1 | branch/release hygiene | 90 | 100 | clean `cur`, pinned SHA before final, no stray force-push | keep main/dev/agy/cur stable; tags approval-gated | tagged release only after ED10 + approval |
| 2 | main figure stack | 88 | 100 | 8/8 scripts, PNGs, manifests, inventory | rerun all generators before final bundle | all fig01–08 SHA/manifest rows ok |
| 3 | Extended Data stack | 50 | 100 | 10/10 ED panels + receipts | ED6–ED10 | inventory 10/10 ED |
| 4 | manifest/hash closure | 70 | 100 | ED5 receipt + per-artifact SHA table | extend ED5 after each new ED | ED5 audit all rows `ok` |
| 5 | notebook execution evidence | 45 | 95 | smoke/full receipts, paths, cell counts | expand ED3 or controlled atlas job | ED3 + ED8 coverage documented |
| 6 | JSON/schema validation | 65 | 95 | strict JSON, no NaN/Inf, schema fixtures | broaden manifest/probe/objective schemas | ED2 + strict JSON on all ED manifests |
| 7 | API stability / import surface | 65 | 95 | `__all__` snapshot, root import smoke | ED1 + final release snapshot | ED1 receipt + import smoke pass |
| 8 | optional dependency laziness | 70 | 95 | subprocess import receipt | keep root import light; ED4 regression | ED4 fixtures all pass |
| 9 | JAX numerical discipline | 65 | 95 | pure kernels, PRNG keys, scan/vmap, no I/O in jit | targeted lint/receipt or tests | hot-path audit documented |
| 10 | source bookkeeping | 55 | 95 | one source mode, calibration status, no double-count | ED/source-accounting panel or tests | source report contract tests pass |
| 11 | probe/readout contracts | 70 | 95 | eight families, finite outputs, status labels | ED7 probe contract matrix | all probe rows finite + gates false |
| 12 | electromagnetic admissibility | 35 | 90 | boundary/gauge, CSD sign, continuity/passivity | solver-ladder doc/tests; keep proxy status | no PDE claim without solver evidence |
| 13 | physical amplitude discipline | 80 | 100 | `physical_amplitude_claim_allowed: false` everywhere | preserve gates in all manifests | zero true amplitude flags in ED audit |
| 14 | mechanism-claim discipline | 65 | 95 | nulls, ablations, repeated seeds | ED9 failure modes/null controls | mechanism language gated in manuscript |
| 15 | benchmark evidence | 35 | 85 | local CPU receipts, hardware/env, timed phases | ED6 benchmark table from Fig5 | no speedup claims |
| 16 | adjacent-tool positioning | 70 | 95 | capability comparison only | Fig8 + careful citations | no superiority wording |
| 17 | tutorial-to-package discipline | 45 | 90 | helpers in package; notebooks configure only | dedupe pass after ED stack | notebook audit clean |
| 18 | release archive readiness | 15 | 100 | tag, wheel hash, install log, optional DOI | ED10 approval-gated bundle | explicit approval before tag/PyPI |
| 19 | manuscript alignment | 50 | 95 | exact commands, SHA, outputs in text | update after ED10 | TBDs replaced with receipts |
| 20 | empirical validation readiness | 10 | 70 | fixed datasets, observed-data nulls | later phase | not required for scaffold paper |

**Global estimate (scaffold/software-methods):** ~72/100 after ED5.

**Global estimate (physically validated biophysics):** ~45/100 — correct posture for current proxy scaffold.

## Final publication targets

| Target | Meaning | Current posture |
|---|---|---|
| Publication scaffold | executable, documented, hashable software-methods paper | close |
| Computational biophysics | source/probe contracts, calibration ladders, validation metadata | mid-stage |
| Physical field solver | calibrated PDE/forward solve with boundary/gauge/residual/units | later |
| Empirical mechanism | observed data, nulls, ablations, held-out tests | later |

Do not collapse these targets.

## Validation commands

```bash
python3 scripts/publication_inventory.py
python3 -m mkdocs build --strict
python3 scripts/publication/ed05_manifest_hashes.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py -q --tb=line
```
