# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Waiting on others
- O-10 Harness proposals, jchat conference #29 (T-O10) | jynx keeper | since 2026-10-25 (opened 2026-10-15) | closes when each is taken or refused. Status 2026-10-30: T-O10 row present in `todo_stack.md`; jchat conference #29 not read in this run; over 3 days, owner decides.

## In flight
- O-69 0.5.9 lanes: next T-45, T-46; PCL R2 on real files (T-3) | orchestrator, Haiku workers | since 2026-10-26 | closes when each lands on dev or is recorded in the todo stack. Status 2026-10-30: rows T-45, T-46 and T-3 are present in `todo_stack.md`, so the closure text is met on its face; over 3 days, owner decides keep or drop.
- O-70 H3 signatures: P3 v2 drafted for Potjans and Vogels (`prereg/P3_v2.md` in each lane); B8 pilots not run (T-O70) | orchestrator drafts, Hamm signs | since 2026-10-26 | closes when Hamm signs or defers each. Status 2026-10-30: `prereg/P3_v2.md` exists for Potjans (`artifacts/etudes/papers/potjans2014/prereg/P3_v2.md`) and Vogels (`artifacts/etudes/papers/vogels2011/prereg/P3_v2.md`); B8 pilot status is as T-O70 states, not re-run.

## Hamm's calls
- O-72 Apply Hamm's 53 rulings of 2026-10-28 (T-R28) | orchestrator, Haiku workers | since 2026-10-28 | closes when T-R28 is empty
- O-68 Local copies held after the 2026-10-25 cleanup: `cl-reader-cursor` (`reader_exp/` unreviewed), `jaxfne-wt-p022`, four WSL copies, non-git leftovers (`jaxfne_rc057`, `jaxfne_receipts`, `jaxfne_reader_exp`, `jaxfne_pclr1_audit`, `F:/warehouse` leftovers, keeper worktrees with broken gitfiles); ruled 2026-10-28: rescue cited files, list each deletion, delete (T-R28) | orchestrator | since 2026-10-25 | closes when each copy is rescued and deleted. Status 2026-10-30: `cl-reader-cursor` and `jaxfne-wt-p022` present under `E:/repos/`; `jaxfne_rc057`, `jaxfne_receipts`, `jaxfne_reader_exp` and `jaxfne_pclr1_audit` absent under `E:/repos/` (other locations not searched).
- O-73 Cell Reports figure S1 drive: the S1 runner is unwritten; the design under review holds the drive fixed (`ext_sources="fixed"`) while p varies. `SOURCE_P1.md` line 18 and commit `b8b38aea` give p·NE Poisson sources at 4 Hz, so the open point is private vs shared sources (G3) | Hamm (decision) | since 2026-10-30 | closes when Hamm rules the S1 drive and the runner follows it; todo: none yet

## Selfcheck (jarness `harness/selfcheck.py`, 2026-10-30)

Selfcheck names the six ids below. Their successors appear only in the global `~/.claude/rules/worker-tier.md` (one `worker` with effort per call; `checker`; `reviewer`), not in the repo; the records stay unedited.
- O-74 Retired name `opencode-delegate` in `artifacts/runs/benchmark/pilot_055.md` and `results_055.json` (benchmark records) | orchestrator | since 2026-10-30 | closes when Hamm rules the records history or selfcheck stops naming the id; todo: none yet
- O-75 Retired name `worker-med` in `artifacts/control/harness/haiku55_evaluation_2026-10-09.md` | orchestrator | since 2026-10-30 | closes when Hamm rules the record history or selfcheck stops naming the id; todo: none yet
- O-76 Retired name `worker-xhigh` in `artifacts/control/harness/haiku55_evaluation_2026-10-09.md` | orchestrator | since 2026-10-30 | closes when Hamm rules the record history or selfcheck stops naming the id; todo: none yet
- O-77 Retired name `worker-max` in `artifacts/control/harness/haiku55_evaluation_2026-10-09.md` | orchestrator | since 2026-10-30 | closes when Hamm rules the record history or selfcheck stops naming the id; todo: none yet
- O-78 Retired name `second-reviewer` in `artifacts/control/harness/haiku55_evaluation_2026-10-09.md`, `artifacts/etudes/theory/theory_f1/README.md` and `theory_f4/README.md` (`etudes/` is outside this run's paths) | orchestrator | since 2026-10-30 | closes when Hamm rules the records history or selfcheck stops naming the id; todo: none yet
- O-79 Retired name `match-checker` in `artifacts/control/harness/haiku55_evaluation_2026-10-09.md` | orchestrator | since 2026-10-30 | closes when Hamm rules the record history or selfcheck stops naming the id; todo: none yet
