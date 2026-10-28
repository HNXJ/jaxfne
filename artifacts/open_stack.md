# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Waiting on others
- O-10 Harness proposals, jchat conference #29 (T-O10) | jynx keeper | since 2026-10-25 (opened 2026-10-15) | closes when each is taken or refused

## In flight
- O-69 0.5.9 lanes: next T-38 leftovers, T-44 F3, T-42 B canary, T-55 (11); PCL R2 on real files (T-3) | orchestrator, Haiku workers | since 2026-10-26 | closes when each lands on dev or is recorded in the todo stack
- O-70 H3 signatures: P3 v2 drafted for Potjans and Vogels (`prereg/P3_v2.md` in each lane); B8 pilots not run (T-O70) | orchestrator drafts, Hamm signs | since 2026-10-26 | closes when Hamm signs or defers each

## Hamm's calls
- O-71 Science, protocol, paper and repo calls batched in T-50 to T-53 (P-052 slope, P6 seeds, C10 and C1, the 0.5.9 go among them) | Hamm | since 2026-10-27 | closes when each is ruled
- O-58 WSL DNS: `/etc/resolv.conf` is empty, so pip and isolated builds fail inside WSL (T-O58) | Hamm (system setting) | since 2026-10-25 (opened 2026-10-21) | closes when `pip download jaxfne==0.5.7 --no-deps` works in WSL
- O-68 Local copies held after the 2026-10-25 cleanup: `cl-reader-cursor` (`reader_exp/` unreviewed), `jaxfne-wt-p022`, four WSL copies, non-git leftovers (`jaxfne_rc057`, `jaxfne_receipts`, `jaxfne_reader_exp`, `jaxfne_pclr1_audit`, `F:/warehouse` leftovers, keeper worktrees with broken gitfiles) (T-O68) | Hamm | since 2026-10-25 | closes when each copy is kept at a rule-conforming path or deleted
