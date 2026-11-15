# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

No items. O-R5SIZE closed 2026-11-15: Hamm ruled option 1 (raise `budget_bytes` by 400 KB, 96400000 to 96800000,
`budget_bytes` only; per-file cap, file budget and record integrity unchanged; not a general relaxation). Both
`check_tree_size.py --check` and `check_record_integrity.py --check` exit 0 on the integrating SHA.
