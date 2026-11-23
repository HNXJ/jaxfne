# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

- O-FIG05 equivalence-gate fig05 frozen PNG stale after a disclosed noise-stream change (865e74bb, 2026-09-24; inferred from C3 estimator bisect); CI Linux runs the gate informationally; re-freeze under change control ruled 2026-11-23 (plan §10.3 item 10) | agent; blocked on Mac access | since 2026-11-15 | closes when `tests/test_equivalence_gate_v20260815.py` passes on darwin after the re-freeze (T-FIG05, T-MAC)
- O-73 `population_rates` takes `dt` in seconds positionally (`jaxfne/analysis/metrics.py:1139`); positional ms gives 1000x wrong rates with only a DeprecationWarning; Hamm 2026-11-23: remove `dt`/`discard_s` in 0.5.10 (plan §10.4 option a) | agent | since 2026-11-14 | closes when `population_rates(c, 0.1, sizes)` raises TypeError naming `dt_ms`, red on the old code (plan R07)

## Crazy Sailor exploration (2026-11-19)

- O-1 Stage-1 title and provisional abstract ("Fixation Boat") not frozen: mismatches implemented 10-neuron LIF model and dense −Δe reward | Hamm | since 2026-11-19 | closes when Hamm picks wording (EX-21)
- O-2 §13.7 critic round (Agent C): VERDICT REVISE, 9 findings; open: F2b and F4 rows of §13.3, the dev-start draw call, the decoder sweep (EX-19) | agent; gates Hamm | since 2026-11-19 | closes when §13.7a resolves each finding and fresh reviewer re-reads (EX-22)
- O-3 Fixture defect: −x motor has no distance drive, so F1 pushes +x at every distance | Hamm | since 2026-11-19 | closes when Hamm keeps fixture or approves labelled separate scenario (EX-11)
- O-4 Primary metric is K-dependent for F1 (tail 526 at K=600, 4114 at K=3000) | Hamm | since 2026-11-19 | closes when a K-independent metric is frozen (EX-20)
- O-6 Plasticity Boat: eight-strategy list and primary metric not frozen | Hamm | since 2026-11-19 | closes when jomission `protocol_v0.md` §3–§5 are frozen (EX-26)
- O-8 Plasticity Boat dev-100: R4 fails at all 7 criteria-v2 g_s tested; R2 passes only at g_s ≤ 4.5 (jomission `O8_decision_packet.md`) | Hamm | since 2026-11-19 | closes when Hamm decides criteria v3 / stop / drop (EX-25)
- O-9 Plasticity Boat dev-100: decoder group count is sum vs mean over each 5-neuron motor group | Hamm | since 2026-11-19 | closes when Hamm confirms sum or mean (EX-25)
