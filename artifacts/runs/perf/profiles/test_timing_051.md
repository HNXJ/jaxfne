# Test/gate wall-time receipt 0.5.1 (ENGINE item 4a–4c, W2 lane)

HEAD: `fd46e1c5912a160f187fbd97038903783339446d` (branch `w/w2`).
Platform: Windows, py3.14.3, jax 0.10.1 (CPU). Quick timing runs only;
full broad (~37 min per `test_profile_050.md`) is NOT rerun per step.
Method: wall = `Get-Date` delta around the command; test-time = pytest's
reported time. Serial runs unless stated. Frozen `test_profile_050.md`
untouched.

## 4a — baseline re-measurement (this HEAD)

| Gate / file | Wall | Test-time | Result | Notes |
|---|---|---|---|---|
| dev gate (`scripts/run_test_gate.py dev`) | 186 s | — | PASS | profile 0.5.0: ~153 s |
| dev pytest targets (serial, `-m "not slow and not release"`) | 193 s | 170 s | 282 passed, 1 skipped, 2 deselected | before-number for proposal (4) |
| broad collect-only (`-m "not slow"`) | 19 s | — | 4177 collected, 37 slow deselected | full broad run not repeated |
| `tests/test_equivalence_gate_v20260815.py` (full file) | 103 s | 98 s | 6 passed, 1 skipped | before-number for proposal (1) |
| `tests/test_fig06_hwd_evidence.py` (full file) | 49 s | 44 s | 3 passed | before-number for proposal (3) |

Broad full-run reference stays `test_profile_050.md`: ~2216 s / 4160 tests.

## Proposal (1) — narrow equivalence gate to representative probes

Defect-table row: Numerical drift → cheapest gate = equivalence-gate
track probes. `test_equivalence_gate_v20260815.py` ran the full 7-figure
generator TWICE (`reproducible_7_of_7` and `tracked_report_matches_fresh_run`;
~50 s each; the second skips post-run on non-freeze platforms but still
regenerates). Change: keep `reproducible_7_of_7` (one full track) in broad;
mark `tracked_report_matches_fresh_run` as `slow` so it leaves dev/broad and
still runs in the slow sweep, which release and rc both execute
(`gate_release` = broad + slow + notebook + examples). No marker-expression
change, so the `(slow, notebook)` exhaustiveness proof in
`test_release_gate_hierarchy.py` is unaffected.

| Run | Before | After |
|---|---|---|
| file, broad selection (`-m "not slow"`) | 103 s wall (both generators) | 62 s wall / 54 s test-time (6 passed, 1 deselected) |
| file, slow selection (`-m "slow and not notebook"`) | — (test ran in broad) | 57 s wall (generator reruns, then platform skip as before) |
| defect-table coverage | full | full — verified 2026-09-24 |

P1 verification: `test_release_gate_hierarchy.py` 36 passed (marker algebra
untouched); broad collect keeps `reproducible_7_of_7` + cheap schema test;
moved test executes under slow sweep (release/rc); TFNE dev targets intact
(144 tests collect under dev expr); cheapest detectors for all 7
defect-table rows present (`test_docs_version_alignment.py`,
`audit_doc_code_integrity.py`, `test_tune_mixed_args_warn.py`, dev TFNE
targets, equivalence full track, audit battery, gate0+authority tests).

## Proposal (3) — cache immutable generated evidence (fig06 pattern)

`test_fig06_generator` unconditionally reran
`scripts/publication_figures/fig06_hwd_evidence.py` (~44 s) over a FROZEN
spec, then validated spec/audit/receipt + PNG. Change: skip the subprocess
when spec FROZEN + audit PASSED + receipt CLOSED + PNG present (all three
validators pass); every post-condition is still asserted, and a
missing/invalid cache regenerates exactly as before. Valid because the
generator is deterministic over frozen tracked inputs (clean-tree
regeneration is byte-identical — no tracked file dirtied by the before-run).

| Run | Before | After |
|---|---|---|
| file wall / test-time | 49 s / 44 s (3 passed) | 9 s / 0.3 s (3 passed, cache hit) |
| cache-miss branch | (always regenerated) | verified: helper False on missing PNG → regenerates; True restored |
| defect-table coverage | full | full — no defect-table detector touched; dev TFNE/JDNA untouched |

## Proposal (4) — parallelize independent families with xdist

xdist 3.8.0 already a dev dependency. Change (`scripts/run_test_gate.py`
only): new `PYTEST_XDIST_ARGS = ["-p", "xdist", "-n", "auto"]` (`-p` needed
because `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`), applied to the dev, broad and
slow pytest sweeps. Notebook sweep stays serial (scoped paths exist for
Windows kernel/zmq stability under load). No marker-expression change, so
the `(slow, notebook)` exhaustiveness proof is unaffected.

| Run | Before (serial) | After (xdist, `-n auto`, 24 CPUs) |
|---|---|---|
| dev gate (`run_test_gate.py dev`) | 186 s wall, PASS | 69 s wall, PASS |
| dev pytest sweep test-time | 170 s (282 passed, 1 skipped, 2 deselected) | 45 s (282 passed, 1 skipped; same 283 nodes, deselect count not echoed by xdist summary) |
| broad-only sample (`test_closure_hp_reconciliation.py`, xdist) | — | 21 passed, 1 skipped, no ordering failures |
| defect-table coverage | full | full — no test file touched; marker algebra intact (`test_release_gate_hierarchy.py` 36 passed); dev TFNE targets still run in dev gate |

Invariants per change: (2) adversarial TFNE/JDNA tests still in dev gate;
(5) every defect-table row keeps a detector (checked against
`test_profile_050.md` table).
