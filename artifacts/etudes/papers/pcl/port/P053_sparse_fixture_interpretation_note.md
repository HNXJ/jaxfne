# P-053 sparse fixture: interpretation note (2026-11-15; reporting only, declared before the Amendment 1 rerun result)

Criteria file `P053_sparse_fixture_criteria.md` is unchanged by this note. Run 1 (original criteria, event count 170, verdict INVALID: negative control detected nothing) and the rerun under Amendment 1 are two linked attempts. Amendment 1 was informed by run 1, so the rerun is a diagnostic validation under a new protocol, not an independent confirmation under the original criteria. Both attempts, their criteria, commits and outputs are preserved. N=170 was chosen by probing and is exploratory.

Owner ruling (Hamm via chat, 2026-11-15): continue the rerun; no further criteria change after the outcome.

## Reported, non-gating additions

- Pathway engagement: besides V0 vs V3 (gating P5, combined pathways), report per seed the oracle outputs for `LAT`=0 with `TD` intact and for `TD`=0 with `LAT` intact against V0 (spike times, `w_s`, `wi_s`). If only the combined intervention engages, conclusions are restricted to combined-pathway engagement.
- Control detection: the `w_s` control counts as detecting only if its measured maximum relative `w_s` difference is at least 1e-6 (the run-1 noise floor was about 1e-15). Report the measured value.

## Inference limits

- NOT_REPRODUCED means: under Amendment 1, on the declared variants, seeds and event count, the arms matched within rtol 1e-12, with the engagement and control preconditions met. It is not a refutation of P-053 and does not cover other fixtures, weights, event counts or learning modes.
- A variant supports an inference about a pathway only if engagement was demonstrated for that pathway at that variant's observable.
- If controls fail again the verdict stays INVALID and the fixture mechanism is investigated in a separate packet.
