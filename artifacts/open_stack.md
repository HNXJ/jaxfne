# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Rebuilt 2026-10-25 from four read-only audits, each finding re-opened by the dispatcher; every line
has its row `T-O<n>` in the todo stack (section 2) with the detail.

## In flight
- O-41 Read the Docs publishes étude stand-ins (P-038 remainder) | keeper | since 2026-10-25 (opened 2026-10-18) | closes when RTD builds the étude pages or they leave the RTD nav
- O-59 HDP étude tune differs on GitHub's runner; the site uses the frozen theta_hat | keeper | since 2026-10-25 (opened 2026-10-21) | closes when the sensitivity is a recorded known limit or bounded by a test
- O-61 Verdict drift across receipt, registry, dashboard and docs (CAMINOS lag, "P6 not attempted", CAMINOS library and methods cells empty in the docs table, Destexhe and GSDR summaries, PCL-R1 receipt unlisted, stage labels) | orchestrator | since 2026-10-25 | closes when a test asserts one verdict per paper across the four sources and passes
- O-62 Brunel S2/S3 outputs exist unscored while `RESULT_P3.md:39` says "has not run" | keeper | since 2026-10-25 | closes when scored into `RESULT_P3.md` with tracked receipts, or recorded as spent
- O-65 57 of 208 `jaxfne.__all__` symbols are absent from `docs/reference` | keeper | since 2026-10-25 | closes when each is documented or a test names its exemption
- O-66 Tree budget: 7.9 KB bytes and 0 files of headroom after the third increment (new tests need a `budget_files` raise); `publication/atlas` is 24.6 MB (three `lfp.html` at 3.5 MB) | keeper | since 2026-10-25 | closes when their citations are checked and headroom is at least 1 MB, or the amendment is signed
- O-67 Lane tests in no gate; skips on gitignored inputs (P-037 remainder) | keeper | since 2026-10-25 (opened 2026-10-18) | closes when one process per lane runs them in a gate and each skip names its input

## Waiting on others
- O-10 Harness proposals, jchat conference #29 | jynx keeper | since 2026-10-25 (opened 2026-10-15) | closes when each is taken or refused

## Hamm's calls
- O-58 WSL DNS: `/etc/resolv.conf` is empty, so pip and isolated builds fail inside WSL | Hamm (system setting) | since 2026-10-25 (opened 2026-10-21) | closes when `pip download jaxfne==0.5.7 --no-deps` works in WSL
- O-68 Local copies held after the 2026-10-25 triage (jaxfne-theory, dash, dash_gen with 12 unmatched gh-pages commits, non-git leftovers) and the layout rule | Hamm | since 2026-10-25 | closes when each copy is kept at a rule-conforming path or deleted
