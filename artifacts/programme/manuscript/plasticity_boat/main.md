# Plasticity Boat: manuscript plan (draft 0, 2026-11-19)

Status: plan only. No LaTeX, no author or submission metadata (deferred to stage 13). The owner's 15-stage list is not in this repo or in jarness; stage numbers below are provisional until it is supplied.

## Venue note

- Internal target: 8 pages total, computational neuroscience short paper (owner decision 2026-11-19).
- AAAI-27 reference, read 2026-11-19: main track, 7 content pages plus up to 2 reference-only pages; anonymous review; reproducibility checklist mandatory (text in the Author Kit, not yet read); full-paper deadline 2026-07-28, passed. Source: https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/ and https://aaai.org/conference/aaai/aaai-27/submission-instructions/. Not binding for this draft.
- Author Kit: https://aaai.org/authorkit27/ (not downloaded; download needs your yes).

## Message (one sentence)

A distance-only sensor cannot express direction, so a 10-neuron LIF controller pushes the boat away from its target; directional observability is a precondition for measuring any plasticity rule.

## Claims

| id | claim | evidence | label | status |
|----|-------|----------|-------|--------|
| C1 | Decoder and plant sign are correct | contract §13.7 implementation findings; R-02 sign probe (scratch, not tracked) | descriptive | observed; needs tracked evidence |
| C2 | The fixture's −x motor has no distance drive; F1 pushes +x | R-09 (8 angles), R-11 (3 angles), R-12 (20 dev starts) | descriptive | observed |
| C3 | Starts with p_x < 0 pass the target transiently; p_x ≥ 0 do not | R-12 (8 of 8 with p_x < 0 and 0 of 12 with p_x ≥ 0, all 20 dev starts) | descriptive, post hoc split | observed; needs confirmation |
| C4 | F1 runs away at constant speed after a transient | R-10 (K = 3000, one start) | descriptive | observed, single start |
| C5 | Zero-drive control holds the boat at R | R-09 (F0 final 5.000, 8 angles) | descriptive | observed |
| C6 | H-gain (F2, F2b) does not reverse the bias | Lane A (K = 600, one start, 20 seeds identical) | descriptive | observed, degenerate across seeds |
| C7 | F3 saturates; reward ≤ 0 in every interval; no correction | R-08 (laneB2, 60 intervals) | descriptive | observed, one start |
| C8 | F6 does not improve on F1 at K = 600 | Lane C (530.54 vs 526.32, one start) | descriptive | observed; re-run pending (EX-17) |
| G1 | Learning produces fixation | none | confirmatory | GATE NOT MET |
| G2 | 100-neuron plasticity competition ranks rules | none | confirmatory | GATE NOT MET (circuit not built) |

Rivals to exclude before any claim beyond C1 to C5: decoder sign error (excluded by C1); finite-time transient (excluded for F1 by C4); initial-condition artefact (excluded by the 20 dev starts, C2).

## Figures (three main figures)

1. **Task and 10-neuron circuit.** Schematic: distance encoder, four motor populations, decoder, boat. Source: contract §12 and §13.3. Not data.
2. **Directional bias.** Distance versus interval for the 20 dev starts, coloured by sign of p_x. Source: `artifacts/_temp/crazy_sailor_explore/falsifier/dev/*.json` (must move to a tracked path, see reproducibility below). Shows C2, C3, C4.
3. **Strategy comparison at K = 600.** Pinned-window mean distance for F0, F1, F2, F2b, F3, F5, F6; single values, no CI (seed-invariant). Shows C5 to C8. Source: Lanes A, B, C JSON.

## Stage status (provisional)

| Stage | Content | Status |
|-------|---------|--------|
| 1 | Venue and reader | done (above) |
| 2 | Plan and claims | this file |
| 3 | Evidence audit | in progress (C1 needs tracked evidence) |
| 4 | Figure candidates | blocked on evidence move |
| 5 | Protocol lock | proposed (`plasticity_boat/protocol_v0.md`), not frozen |
| 6 | Draft | not started |
| 7 | Build gate | not started (AAAI style file not obtained) |
| 8 | Self-check | not started |
| 9 | Critic round 1 | not started |
| 10 to 12 | Fix, repeat, critic ACCEPT | not started |
| 13 | Author and submission metadata | deferred |
| 14 to 15 | Packaging and release | not authorised |

## Reproducibility (draft; AAAI checklist text to be read)

- Code: `jaxfne` repo, branch `_temp_crazy_sailor_m0m1_b`, HEAD 584a56ad. Not merged; not tagged.
- Evidence: `artifacts/_temp/` is gitignored. Every figure source must move to a tracked path before it is cited. Open item.
- Seeds: S1 has no stochastic input; seed enters only `sailor_reset`. Dev starts from `default_rng(2025)`.
- Hardware and runtime: recorded per run (wall_s in JSON). Full hardware record missing.
- Tests: 40 targeted M2 tests pass (`-k "sailor or m2"`); CS-10 reports 13 unrelated baseline failures (not run here).
