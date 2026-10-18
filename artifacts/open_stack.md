# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or problem, or is dropped.

## In flight
- O-37 Read the Docs fails on every build since 2026-10-14 (P-035): `build_docs_assets.py --build` stops at the `hdp_controllability_reachability` condition-B rate pin (10.5611 against 10.6212) | keeper | since 2026-10-18 | closes when an RTD build on dev or main succeeds
- O-39 Core API defects (P-036): `Simulation.plasticity` stored, never consumed; tune fall-through returns ACCEPT_CANDIDATE with no parameters; `or` defaults swallow 0; unavailable backend silently falls back | keeper | since 2026-10-18 | closes when each is refused or consumed, with a failing-first test
- O-40 Vacuous and ungated tests (P-037): conftest x64 reset voids float64 tests; skips on gitignored inputs; exact float pin; cwd-dependent paths; 11 lane test files in no gate | keeper | since 2026-10-18 | closes when each test runs and can fail in a gate
- O-41 Stale pointers and ignored flags (P-038): evidence checklist path in 3 scripts, mcc3 manifest names 3 absent files, 4 ignored flags, `simulate(return_state=True)` doc, non-strict RTD | keeper | since 2026-10-18 | closes when each resolves on disk or is refused
- O-42 GSDR fits record `"mcdp": True`, the core `gsdr` has no MCDP (P-039): the 350 and 500 generation fits ran without it | keeper | since 2026-10-18 | closes when manifests say what ran and MCDP is in the core, or Hamm rules it out of scope
- O-43 REPLICATIONS records (P-042): CAMINOS release row 0.5.6 against the todo's 0.5.7; scored-off GSDR and Destexhe results missing (B2) | keeper | since 2026-10-18 | closes when the table matches the lanes

## Waiting on others
- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls
- O-14 T-28 `tfne._RESERVED_LOWER` {x, y, h, s} defined, never read: wire into name checks or delete (score about 60 either way) | Hamm | since 2026-10-15 | closes when Hamm picks one
- O-16 Branch `_temp_never-sweep` on origin, unmerged, 2026-10-12, owner unknown | Hamm | since 2026-10-15 | closes when its owner merges or deletes it
- O-24 PCL Fig. 3 is scored on the numba oracle (`papers/pcl/port/run_tuning.py:85-94`, CORE_GAP), against checklist E1 and S2: wait for T-22 and rescore on the core, or grant an exception. Since 2026-10-18 the same-step operator is not the blocker (the LIF core has PCL's cascade as `event_order`); the full-network mapping of feed, complex pooling and lateral/top-down paths is | Hamm | since 2026-10-17 | closes when Hamm picks one
- O-25 GSDR `PREREG_P3.md` is a draft, unsigned, while CALIBRATION steps score against it (checklist P1) | Hamm | since 2026-10-17 | closes when Hamm signs it or rules the steps unscored
- O-26 Izhikevich 2003 v8.2 "Fig. 2 passes 8 of 8" is post hoc (`RESULT_P3.md:69-71`): wording and whether release notes may cite it | Hamm | since 2026-10-17 | closes when Hamm rules
- O-27 Tree-degree rule is tested for `artifacts/` only; root, `tests/` (493 flat) and `scripts/` (124) exceed 7 | Hamm | since 2026-10-17 | closes when Hamm sets the scope
- O-29 CAMINOS P6 status disagrees: goal G-2 "equivalence test (not yet written)", `RESULT_P6.md:3` PASS, `REPLICATIONS.md` "mixed" (paper verdict), dashboard reproduced. Proposed G-2 edit: "Held by: P6 v2 PASS (`papers/caminos/RESULT_P6.md`)" | Hamm | since 2026-10-17 | closes when G-2 is edited or ruled
- O-44 Destexhe 2009 scores E1-E4 against an unsigned draft prereg, and its header (OFF) and final section (E1 passes) disagree (P-040) | Hamm | since 2026-10-18 | closes when Hamm signs it or rules the steps unscored
- O-45 PCL: docs page says Fig. 2 and the central claim are reproduced while its own rows FAIL; local STDP/LIF and `pcl_h` rules outside the core; v5 T4 passes on a criterion changed after v4 (P-041) | Hamm | since 2026-10-18 | closes when Hamm rules the wording and the T4 criterion; the core migration then goes to todo
- O-46 Tooling to keep or delete: 16 scripts referenced by nothing, 8 audit scripts in no gate (P-042), and the ignored `--check` in `assert_release_freeze.py` and `sync_release_metadata.py` (was O-15, T-20, score about 65 to change) | Hamm | since 2026-10-18 | closes when each is kept with a reason or deleted
