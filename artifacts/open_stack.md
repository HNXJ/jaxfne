# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Rebuilt 2026-10-25 from four read-only audits, each finding re-opened by the dispatcher; every line
has its row `T-O<n>` in the todo stack (section 2) with the detail.

## Waiting on others
- O-10 Harness proposals, jchat conference #29 | jynx keeper | since 2026-10-25 (opened 2026-10-15) | closes when each is taken or refused

## In flight
- O-69 0.5.9 lanes (release plan 9A-9G): LIF poisson_drive list, Izhikevich E1 library-path pilot, Brunel S1 runs, 9F figure lock (E1 core, citations by symbol, S1 option, 9B checks, 9F evidence writers landed 2026-10-26) | orchestrator, Haiku workers | since 2026-10-26 | closes when each lands on dev or is recorded in the todo stack
- O-70 H3 signatures: P3 v2 drafted for Potjans and Vogels (`prereg/P3_v2.md` in each lane; scale and bands wait on Hamm); pilots not run | orchestrator drafts, Hamm signs | since 2026-10-26 | closes when Hamm signs or defers each

## Hamm's calls
- O-58 WSL DNS: `/etc/resolv.conf` is empty, so pip and isolated builds fail inside WSL | Hamm (system setting) | since 2026-10-25 (opened 2026-10-21) | closes when `pip download jaxfne==0.5.7 --no-deps` works in WSL
- O-68 Local copies held after the 2026-10-25 cleanup: `cl-reader-cursor` (`reader_exp/` unreviewed), `jaxfne-wt-p022`, four WSL copies, non-git leftovers (`jaxfne_rc057`, `jaxfne_receipts`, `jaxfne_reader_exp`, `jaxfne_pclr1_audit`, `F:/warehouse` leftovers, keeper worktrees with broken gitfiles), and the layout rule | Hamm | since 2026-10-25 | closes when each copy is kept at a rule-conforming path or deleted
