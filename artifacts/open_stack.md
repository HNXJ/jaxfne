# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Blocked integration

O-MAC10 `dev` integration of `_temp_swarm_integ_20261115` blocked | agent | since 2026-11-15 | closes when `python scripts/run_test_gate.py broad` reads 0 failed on a matching host (and CI) at the integrating SHA, with outside review. The sandboxed Mac gate (arm64, JAX 0.10.1) ended 10 failed; the same tests fail at base 24a95963 (non-regression only, not acceptance). Attributed 2026-11-15 (`artifacts/assurance/mac_gate_attribution.md`): 2 sandbox artifacts (pass unsandboxed), 5 float32 XLA-CPU bit-identity tests (platform numerics), 2 `izh2003` known-answer tests (numeric drift, not isolated), 1 stale freeze (O-FIG05), 2 `score_port_parity_checks` tests (Jaxley 0.13.0 here vs 0.14.0 in the CI lock; swapping to 0.14.0 on this host passes them, so version drift is demonstrated here only). Measured unsandboxed full run at 1373272e: 11 failed = these 10 minus the 2 sandbox ones, plus `test_p3_folders_live_at_new_layout` (my note had made `etudes/meta` 8 entries; moved, fixed) and the 2 Jaxley tests; all but the layout one fail at base 24a95963. At 939009cf: 10 failed with Jaxley 0.13.0, 8 failed with Jaxley 0.14.0 (the two parity tests pass; only Jaxley differs between the two Pythons). Hamm 2026-11-15: a documented host limitation for `_temp_*` work, not a waiver of the `dev` gate; no test weakening.
O-FIG05 equivalence-gate fig05 frozen PNG stale after a disclosed noise-stream change (865e74bb, 2026-09-24; inferred from the C3 estimator bisect); CI Linux runs the gate informationally and never sees it | Hamm (re-freeze is frozen-evidence change control) | since 2026-11-15 | closes when `tests/test_equivalence_gate_v20260815.py` passes on darwin after an owner-authorized fig05 re-freeze, or panel C renders from frozen V_m. Todo row T-FIG05.
O-NUM bit-exact assertions across differently compiled or shaped XLA graphs (#1 to #5 of the attribution) are not guaranteed; fixing means a tolerance or an O0 compile, which weakens or restructures tests | Hamm (ruling) | since 2026-11-15 | closes when Hamm rules the remedy and the tests are changed and re-run. Todo row T-NUM.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

No items. O-R5SIZE closed 2026-11-15: Hamm ruled option 1 (raise `budget_bytes` by 400 KB, 96400000 to 96800000,
`budget_bytes` only; per-file cap, file budget and record integrity unchanged; not a general relaxation). Both
`check_tree_size.py --check` and `check_record_integrity.py --check` exit 0 on the integrating SHA.
