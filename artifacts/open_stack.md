# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Hamm's calls (0.5.10 review, asked one at a time in this order)
- O-89 Stale clones (Hamm said delete, 2026-11-08): the 34 `E:/repos/_wt/jaxfne/*` lanes and 7 logs are deleted; the 27 top-level `E:/cl-*` targets of `artifacts/_temp/delete_stale_clones_2026-11-07.ps1` remain because the agent tool blocks top-level E:/ removal (rescue copies in `F:/warehouse/jaxfne_clone_rescue/`; no surviving clone borrows from them) (todo 14) | Hamm | since 2026-11-07 | closes when Hamm runs the script
