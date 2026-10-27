# CI plan: each failure names one class

Audit 2026-10-27 (Hamm: a red check should mean something specific). Sources: `.github/workflows/*.yml`,
`scripts/run_test_gate.py`, run timings from the Actions API, local gate logs.

## Measured

| Stage | Trigger | Wall | Holds |
|---|---|---|---|
| CI Fast `fast-test` | every push to dev/main, PR to dev | 4 min per leg, 3.11 and 3.14 | 15 steps in one job: compileall, ruff x2, 4 audits, atlas tests, mkdocs strict, orphans, doc-code, tree size, dev gate (~100 s), examples (~70 s) |
| CI Fast `fast-build` | after fast-test | <1 min | build, twine, wheel smoke |
| Release CI `release-test` | push/PR to main, nightly, dispatch | 45 min (3.11), 52 min (3.14) | unfiltered pytest over `tests/` (44-50 min), examples |
| Dashboard `figures` | dev pushes touching vis/docs paths | 12 min | figure site build |
| Local broad gate | before each code push | 27 min | report audits, pytest `not slow` with xdist (9.7 min), 7 paper lanes serial without xdist (17 min: pcl 9.9, caminos 3.9, potjans 2.3) |

## Findings

1. One red `fast-test (3.11)` mixes 15 unrelated checks; the interpreter-independent ones (lint, audits, docs,
   tree size) run twice and fail twice together.
2. No fast end-to-end check: a broken core path surfaces as dozens of unit failures at once.
3. State-only commits (stack markdown) rerun CI Fast and, by habit, the 27-min local broad gate.
4. The nightly Release CI reruns `main` when its tree has not changed since the last green run.
5. Paper lanes (`artifacts/etudes/papers/*/test_*.py`) run only in the local broad and rc gates: no CI job
   runs them, and they run serially without xdist (63 percent of the local broad wall time).
6. The rc gate reruns the whole suite locally after Release CI passed on the same tree; it already fetches
   that run's JUnit for parity and records `HEAD^{tree}` in its attestation, but nothing reuses it.

## Plan

| Step | Change | Red means | Call |
|---|---|---|---|
| A | Split `ci.yml` into jobs: `static` (once: compileall, ruff, audits, record integrity, tree size), `docs` (once: mkdocs strict, orphans, doc-code), `canary`, `tests` (matrix: dev gate), `examples`, `build` | the job name is the class | do |
| B | Canary chains, one per data path, seeded, outputs hashed into a frozen register (jnwb pattern): tensor -> construct -> simulate (lif edge), izhikevich, HDP run, event-scan replay (H update, spike log, weights), field/LFP/CSD probe, objective and one tune step, jdna develop. 1-2 min, first stage; unit tests then localize | one data path changed its output | do |
| C | `paper-lanes` CI job on dev pushes that touch `artifacts/etudes/papers/**` or `jaxfne/**`, xdist per lane; the local broad gate runs lanes with xdist too | a named lane broke | do |
| D | Code-tree stamp: a passing broad gate writes a stamp keyed by the hash of `jaxfne/ tests/ scripts/ examples/ pyproject.toml` plus the interpreter; a state-only commit reuses it and runs the report audits only | (no rerun on unchanged code) | do |
| E | Nightly Release CI first step: exit green when the last successful run has the same tree | (no rerun on an unchanged tree) | do |
| F | rc reuses Release CI JUnit for the same tree and runs locally only the machine-bound families (environment parity, build, isolated wheel smoke, attestation) | changes the release-evidence contract | Hamm approved 2026-10-27 |
| G | Profile `pytest --durations=50` on the broad sweep and the pcl and caminos lanes before cutting tests | numbers for the next cut | do |

Order: G and A first (measure, then the cheapest split), then B, C, D, E. Each lands with a test or a
dispatched run showing the job fails on a planted defect of its class and passes on dev.

## Status

- G done 2026-10-27. Broad pytest 12.1 min under load. Slowest files: `emitters_hh` 198 s,
  `tutorial_figure_contract_v0219` 181 s (three fixture setups of ~60 s), notebook smoke 141 s,
  equivalence gate 104 s. Lanes serial 970 s: pcl 542, caminos 247, potjans2014 118, the other four
  under 25 each. With xdist, pcl 121 s and caminos 65 s, the same tests passing.
- A landed (c07b7df1): `static`, `docs`, `fast-test`, `examples`, `fast-build`. The ids `fast-test` and
  `fast-build` stay because the dev and main rulesets require them; `fast-build` always runs and fails
  when a needed job did not pass. Planted-defect run pending.
- C, local half: pcl and caminos run under xdist in the broad gate (`XDIST_LANES`); lanes 16 min to 6 min.
- E landed (2f21b125): a nightly on a tree that already passed cancels itself (a skipped run would
  conclude success without artifacts and shadow the real run in the rc and publish lookups). First
  nightly after landing is the observation.
- Docs gate gap closed (b4133826): the docs gate and the dev gate run `test_public_docs_hygiene.py`.
