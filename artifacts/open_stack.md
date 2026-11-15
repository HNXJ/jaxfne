# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Blocked integration

O-MAC10 `dev` integration of `_temp_swarm_integ_20261115` blocked | agent | since 2026-11-15 | closes when `python scripts/run_test_gate.py broad` reads 0 failed on a matching host (and CI) at the integrating SHA, with outside review. The Mac gate (arm64, JAX 0.10.1) ends 10 failed; the same tests fail at base 24a95963 (non-regression only, not acceptance). Hamm 2026-11-15: a documented host limitation for `_temp_*` work, not a waiver of the `dev` gate; no test weakening; each failure needs a demonstrated cause. Attribution: todo row T-ATTR.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

No items. O-R5SIZE closed 2026-11-15: Hamm ruled option 1 (raise `budget_bytes` by 400 KB, 96400000 to 96800000,
`budget_bytes` only; per-file cap, file budget and record integrity unchanged; not a general relaxation). Both
`check_tree_size.py --check` and `check_record_integrity.py --check` exit 0 on the integrating SHA.
