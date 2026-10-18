# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or problem, or is dropped.

## In flight

- O-30 GSDR fit on the reference morphology, 500 generations (`F:\warehouse\gsdr\gsdr_fit500_reference_2026-10-17\fit.log`, code 6124c5e5, GPU lock held about 17 h) | keeper | since 2026-10-17 | closes when `manifest.json` exists and CALIBRATION records the regime at its g*

## Waiting on others

- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls

- O-14 T-28 `tfne._RESERVED_LOWER` {x, y, h, s} defined, never read: wire into name checks or delete (score about 60 either way) | Hamm | since 2026-10-15 | closes when Hamm picks one
- O-15 T-20 `--check` accepted and ignored in `assert_release_freeze.py`, `sync_release_metadata.py` (score about 65 to change) | Hamm | since 2026-10-15 | closes when kept with a note or removed
- O-16 Branch `_temp_never-sweep` on origin, unmerged, 2026-10-12, owner unknown | Hamm | since 2026-10-15 | closes when its owner merges or deletes it
- O-24 PCL Fig. 3 is scored on the numba oracle (`papers/pcl/port/run_tuning.py:85-94`, CORE_GAP), against checklist E1 and S2: wait for T-22 and rescore on the core, or grant an exception | Hamm | since 2026-10-17 | closes when Hamm picks one
- O-25 GSDR `PREREG_P3.md` is a draft, unsigned, while CALIBRATION steps score against it (checklist P1) | Hamm | since 2026-10-17 | closes when Hamm signs it or rules the steps unscored
- O-26 Izhikevich 2003 v8.2 "Fig. 2 passes 8 of 8" is post hoc (`RESULT_P3.md:69-71`): wording and whether release notes may cite it | Hamm | since 2026-10-17 | closes when Hamm rules
- O-27 Tree-degree rule is tested for `artifacts/` only; root, `tests/` (493 flat) and `scripts/` (124) exceed 7 | Hamm | since 2026-10-17 | closes when Hamm sets the scope
- O-28 `artifacts/release/current_release_authorities.json` (frozen) names 0.5.5 as target; 0.5.6 is published | Hamm | since 2026-10-17 | closes when Hamm authorizes the update
- O-35 T-22 semantics, built with the keeper's defaults (keeps choices open): (1) the declaration sits on the Configuration connection for the AdEx route (CAMINOS builds through Configuration), PCL keeps `NeuronalTensor` `local_inhibition`; (2) `event_scan` stays out of the operator (fixed rounds are not event-exact); (3) fixed R, no iterate-to-quiescence; (4) round rule: OR-monotone re-step from the pre-step state | Hamm | since 2026-10-18 | closes when Hamm confirms or changes each
- O-29 CAMINOS P6 status disagrees: goal G-2 "equivalence test (not yet written)", `RESULT_P6.md:3` PASS, `REPLICATIONS.md` "mixed" (paper verdict), dashboard reproduced. Proposed G-2 edit: "Held by: P6 v2 PASS (`papers/caminos/RESULT_P6.md`)" | Hamm | since 2026-10-17 | closes when G-2 is edited or ruled

## Known gaps (recorded, no action yet)

- O-22 Over-7 lane folders excepted in `tests/test_artifacts_tree_degree.py` while their lanes are open: `etudes/papers/caminos/` (19 files), `etudes/papers/izhikevich2003/` (13) | keeper | since 2026-10-16 | closes when each is split by kind or its exception gets a path-addressing reason
- O-19 Scratch clones (58 `cl-*`) and finished Muse sessions | keeper | since 2026-10-15 | closes when deleted after O-3 to O-5 land
