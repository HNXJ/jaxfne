# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Crazy Sailor exploration (2026-11-19)

- O-2 §13.7 critic round (Agent C): VERDICT REVISE, 9 findings; the F1 falsifier names no metric | agent; gates Hamm | since 2026-11-19 | closes when §13.7a records each finding resolved or rejected and a fresh reviewer re-reads it (EX-22)
- O-3 Fixture defect: −x motor has no distance drive, so F1 pushes +x at every distance and blocks corrective F1/F2/F4/F6 | Hamm | since 2026-11-19 | closes when Hamm keeps the fixture or approves a labelled separate scenario (EX-11)
- O-4 Primary metric is K-dependent for F1 (tail 526 at K = 600, 4114 at K = 3000) | Hamm | since 2026-11-19 | closes when a K-independent metric is frozen (EX-20)
- O-6 Plasticity Boat: eight-strategy list and primary metric not frozen | Hamm | since 2026-11-19 | closes when jomission `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/protocol_v0.md` §3 to §5 are frozen (EX-26)
- O-7 Owner's 15-stage manuscript workflow not in the repo or jarness; stage numbering is provisional; Hamm chose to proceed with provisional numbers 2026-11-19 | Hamm | since 2026-11-19 | closes when the list is supplied (EX-27)
- O-8 Plasticity Boat dev-100: R4 fails at all seven criteria-v2 sensory gains g_s tested (toward-group motor neurons at g_d = 24); R2 passes only at g_s ≤ 4.5. Diagnostic g_d = 4 at g_s = 3 passes R2 to R4 (not selection-eligible). Owner chooses: criteria v3 with a G_D grid in (4, 6), keep v2 and stop at C3, or drop the dev-100 track. Packet: jomission `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/O8_decision_packet.md` | Hamm | since 2026-11-19 | closes when Hamm decides (EX-25)
- O-9 Plasticity Boat dev-100: decoder group count is the sum over each 5-neuron motor group (about 5× the mean's force gain); frozen v1 says only "counts" | Hamm | since 2026-11-19 | closes when Hamm confirms sum or mean (EX-25)
- O-10 Record-integrity FAIL: `artifacts/release/budget_amendment_2026-10-23.md` is pinned sha256 `c52e830b…` in `scripts/record_hashes.json`; live is `cdc17e41…` after `46293fcc` (Option B section; tree-size allowlist 2797 to 2798) and `03e24a16` (text). Neither commit repinned. Option B cites a 2026-11-19 owner chat that the repo does not hold. Owner ruling: no waiver, no automatic repin to `cdc17e41…` | Hamm | since 2026-11-19 | closes when Hamm gives a SHA-specific go (repin to `cdc17e41…`, or restore `c52e830b…` and re-examine the 2798 allowlist) and `python3 scripts/check_record_integrity.py --check` exits 0 (EX-30)