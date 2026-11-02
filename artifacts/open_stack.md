# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Waiting on others
- O-10 Harness proposals, jchat conference #29 (T-O10) | jynx keeper | since 2026-10-25 (opened 2026-10-15) | closes when each is taken or refused. Status 2026-10-30: T-O10 row present in `todo_stack.md`; jchat conference #29 not read in this run; over 3 days, owner decides.

## In flight
- O-69 0.5.9 lanes: next T-45, T-46; PCL R2 on real files (T-3) | orchestrator, Haiku workers | since 2026-10-26 | closes when each lands on dev or is recorded in the todo stack. Status 2026-10-30: rows T-45, T-46 and T-3 are present in `todo_stack.md`, so the closure text is met on its face; over 3 days, owner decides keep or drop.

## Hamm's calls
- O-72 Apply Hamm's 53 rulings of 2026-10-28 (T-R28) | orchestrator, Haiku workers | since 2026-10-28 | closes when T-R28 is empty
- O-68 Local copies held after the 2026-10-25 cleanup: `cl-reader-cursor` (`reader_exp/` unreviewed), `jaxfne-wt-p022`, four WSL copies, non-git leftovers (`jaxfne_rc057`, `jaxfne_receipts`, `jaxfne_reader_exp`, `jaxfne_pclr1_audit`, `F:/warehouse` leftovers, keeper worktrees with broken gitfiles); ruled 2026-10-28: rescue cited files, list each deletion, delete (T-R28) | orchestrator | since 2026-10-25 | closes when each copy is rescued and deleted. Status 2026-10-30: `cl-reader-cursor` and `jaxfne-wt-p022` present under `E:/repos/`; `jaxfne_rc057`, `jaxfne_receipts`, `jaxfne_reader_exp` and `jaxfne_pclr1_audit` absent under `E:/repos/` (other locations not searched).
