# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

O-R5SIZE <two gates conflict on one file> | Hamm | since 2026-11-12 | closes when a route is ruled: the full
3168-row armcheck list (413 KB) is a pinned prereg record whose bytes `check_record_integrity.py` refuses to edit,
and it puts the tracked tree 283 KB over `check_tree_size.py`'s budget. Options: (1) raise `budget_bytes` by
400 KB in `scripts/tree_size_allowlist.json` (recommended: the record is evidence and the budget is a growth
ratchet, not a correctness bound); (2) move the row list to a new declared record and leave the old bytes in
place; (3) let the armcheck hold counts only and cite the untracked `F:/warehouse/pcl/r5/approx_diff.json`,
which needs a dated ruling because it trims a pinned record. Fix item: T-93 in `todo_stack.md`.
