# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

- O-FIG05 equivalence-gate fig05 frozen PNG stale after a disclosed noise-stream change (865e74bb, 2026-09-24; inferred from C3 estimator bisect); CI Linux runs the gate informationally | Hamm (frozen-evidence change control) | since 2026-11-15 | closes when `tests/test_equivalence_gate_v20260815.py` passes on darwin after owner-authorized fig05 re-freeze or panel C renders from frozen V_m (T-FIG05)
- O-NUM bit-exact float32 assertions across differently compiled or shaped XLA graphs (#1–#5 of `artifacts/assurance/mac_gate_attribution.md`, incl. former O-97 `test_composition_identity_054.py` 2-ulp drift at n=8, 12) | Hamm | since 2026-11-15 | closes when Hamm rules the remedy and tests pass (T-NUM)
- O-73 `population_rates` takes `dt` in seconds positionally (`jaxfne/analysis/metrics.py:1139`); positional ms gives 1000x wrong rates with only a DeprecationWarning | Hamm (release) | since 2026-11-14 | closes when positional-ms test fails on old form and fix or hard error lands (plan R07, S04)
- O-74 `_bin_counts`, `fleiss_kappa_binary`, `fano_factor`, `pairwise_cc_binned_disjoint` raise ValueError on non-whole `bin_ms/dt_ms` (0.5.8 rounded) with no deprecation path (`jaxfne/analysis/metrics.py`); refusal in `docs/reference/changelog.md` (0.5.9 Fixed), unpinned | Hamm (release) | since 2026-11-14 | closes when deprecation path or changelog entry is pinned by a test (plan R08, S04)
- O-75 `fleiss_kappa_binary` changed definition at same name/signature (`jaxfne/analysis/metrics.py:324`: textbook Fleiss; verified 2026-11-21 against tag v0.5.8, whose P_o was the fraction of bins where all neurons agree; verification and changelog record done, changelog 0.5.9 Fixed) | Hamm | since 2026-11-14 | closes when Hamm's S04 ruling keeps the textbook form or reverts it (plan R09)
- O-76 Six bitwise tests are `xfail(strict=False)` and cannot fail (`tests/test_adex_route.py:1060,1086,1125`; `tests/test_lif_tensor_route.py:705,731,776`) | agent | since 2026-11-14 | closes when each is a strict xfail or passing test (T-NUM); waits on the O-NUM remedy: the reasons say the bitwise value is platform-dependent, so a strict xfail fails where it passes and a passing test needs a tolerance, which is the O-NUM criteria (2026-11-22)

## Crazy Sailor exploration (2026-11-19)

- O-1 Stage-1 title and provisional abstract ("Fixation Boat") not frozen: mismatches implemented 10-neuron LIF model and dense −Δe reward | Hamm | since 2026-11-19 | closes when Hamm picks wording (EX-21)
- O-2 §13.7 critic round (Agent C): VERDICT REVISE, 9 findings; open: F2b and F4 rows of §13.3, the dev-start draw call, the decoder sweep (EX-19) | agent; gates Hamm | since 2026-11-19 | closes when §13.7a resolves each finding and fresh reviewer re-reads (EX-22)
- O-3 Fixture defect: −x motor has no distance drive, so F1 pushes +x at every distance | Hamm | since 2026-11-19 | closes when Hamm keeps fixture or approves labelled separate scenario (EX-11)
- O-4 Primary metric is K-dependent for F1 (tail 526 at K=600, 4114 at K=3000) | Hamm | since 2026-11-19 | closes when a K-independent metric is frozen (EX-20)
- O-6 Plasticity Boat: eight-strategy list and primary metric not frozen | Hamm | since 2026-11-19 | closes when jomission `protocol_v0.md` §3–§5 are frozen (EX-26)
- O-8 Plasticity Boat dev-100: R4 fails at all 7 criteria-v2 g_s tested; R2 passes only at g_s ≤ 4.5 (jomission `O8_decision_packet.md`) | Hamm | since 2026-11-19 | closes when Hamm decides criteria v3 / stop / drop (EX-25)
- O-9 Plasticity Boat dev-100: decoder group count is sum vs mean over each 5-neuron motor group | Hamm | since 2026-11-19 | closes when Hamm confirms sum or mean (EX-25)
