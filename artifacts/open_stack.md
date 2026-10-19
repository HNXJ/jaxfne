# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or problem, or is dropped.

## In flight
- O-40 Vacuous and ungated tests (P-037): conftest x64 reset voids float64 tests; skips on gitignored inputs; exact float pin; cwd-dependent paths; 11 lane test files in no gate | keeper | since 2026-10-18 | closes when each test runs and can fail in a gate
- O-41 Stale pointers and ignored flags (P-038): evidence checklist path in 3 scripts, mcc3 manifest names 3 absent files, 4 ignored flags, `simulate(return_state=True)` doc, non-strict RTD | keeper | since 2026-10-18 | closes when each resolves on disk or is refused
- O-42 GSDR fits record `"mcdp": True`, the core `gsdr` has no MCDP (P-039): the 350 and 500 generation fits ran without it | keeper | since 2026-10-18 | closes when manifests say what ran and MCDP is in the core
- O-43 REPLICATIONS records (P-042): CAMINOS release row 0.5.6 against the todo's 0.5.7; scored-off GSDR and Destexhe results missing (B2) | keeper | since 2026-10-18 | closes when the table matches the lanes
- O-48 Carry out Hamm's rulings of 2026-10-18 (todo T-33): records (GSDR and Destexhe steps unscored, PCL page and v5 T4 label, Izhikevich 6 of 8 under v7 with v8.2 labelled, dashboard CAMINOS note), API refusals (non-zero `Simulation.plasticity`, absent gpu/tpu backend), tree-degree exemptions | keeper | since 2026-10-18 | closes when each lands on dev with its gate
- O-49 T-32 reproduction lanes: R1 lane kit and R2 `simulate_hh_batch` built in clones (cl-r1, cl-r2), review and broad gate pending | keeper | since 2026-10-18 | closes when both are on dev

## Waiting on others
- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls
- none open (open-decisions review 2026-10-18/19: rulings in todo T-32 and T-33)
