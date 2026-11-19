# Plasticity Boat: dev-100 gate record (frozen criteria v1)

Scope: development starts only (`default_rng(2025).uniform(0, 2π, 20)`, R = 5, K = 600, seed 0). Held-out starts unused. Criteria: `criteria_v1_frozen.md` (sha256 d57fd981…037ee, matches provenance).
Code: `code/dev100_model.py` (sha256 233a263f…84a80), `code/dev100_closed_loop.py` (sha256 700e67b6…cb0fb3). Git HEAD at run: 584a56ad (uncommitted).
Raw: `artifacts/_temp/plasticity_boat/dev100/` (`P1_pos_vel/`, `N1_pos_only/`, `P1_tau05/`, `P1_N1_summary.json`, `R1_R4_resting_responsiveness.json`).

## Positive control and negative controls (closed loop)

| Condition | Sensing | Gate | Starts pass | Occupancy range | Final distance range |
|---|---|---|---|---|---|
| P1 (primary) | z = −(p + 0.1 v) | PASS iff all 20 ≥ 0.9 | 20 / 20 | 1.0 (all) | 0.004 to 0.021 |
| N1 (negative control) | z = −p | not a gate; expected to fail | 0 / 20 | 0.000 to 0.042 | 2.27 to 15.87 |
| T05 (sensitivity) | z = −(p + 0.5 v) | not a gate | 20 / 20 | 1.0 (all) | 0.015 to 0.148 |

Window: errors[480:600], radius 1.0. Largest window error across P1 starts 0.034 (start 10).

### Verification (read-only, 2026-11-19)

- P1 recomputed from the 20 `start_NN.json` files: 20/20 PASS. Three alternative window conventions give 20/20 PASS, no flip. Theta and p0 match the stated draw exactly.
- Mechanism: each stored force equals the decode of the previous interval's group counts (exact). Single-update residual at float32 precision for all stored components. Float64 residual 6.5e-7 to 1.18e-6; starts 14 and 16 exceed a literal 1e-6 tolerance. Spikes occur in the window for every start (100 to 895 E spikes, 0 PV).
- Chatter: bounded, window peak radius 0.018 to 0.034; 21 to 60 axis sign changes per window. Period not measured (boundedness and centring only).
- Starts 00 and 17: zero ±x group counts in the window. The x offset is held by damping, not active control (inferred).

## Resting and responsiveness (R1 to R5, `R1_R4_resting_responsiveness.json`)

| Criterion | Result | Status |
|---|---|---|
| R1 intrinsic silence | E 0.0 Hz, PV 0.0 Hz | PASS |
| R2 stationary task input, E in [5, 100] Hz | E mean 250 Hz; non-motor E 325 Hz | FAIL |
| R3 directional response ≥ 5 Hz at four positions | PASS | PASS |
| R4 no neuron above 100 Hz in R2 and R3 | R2: non-motor E max 325 Hz; R3: toward-group motor E max 324.5 Hz (at every position; directional drive g_d = 24) | FAIL |

Correction (2026-11-19, after the v2 sweep): the R4 failure in R3 is in the motor neurons of the group that points toward the target, not in non-motor E. The toward-group motor neurons are driven by the unchanged directional gain g_d = 24 (current 20). The earlier wording "saturation in non-motor E" applied only to R2.
| R5 build check (drive 0.0; 10-neuron builder unchanged) | PASS | PASS |

## Outcome under the frozen rule

- P1 PASS; R2 and R4 FAIL. The rule requires P1 and R1 to R5 all to pass before the dev model may proceed to protocol freeze. **Dev-100 does not proceed.** Failures are reported unchanged; no tuning.
- The eight-strategy competition is not run.
- Any change to R2 or R4 needs a new labelled criteria version approved by the owner. The v1 file is not edited.

## Open decisions for the owner

1. Motor-group count definition: the runner sums spikes over each 5-neuron group (docstring: "the decoder is unchanged"). Frozen v1 says only "counts". Sum gives about 5× the force gain of the mean. P1 is reported under the sum. Confirm or change (a change needs a new criteria version).
2. R2 and R4 failure: revise the input level or gain under a new labelled criteria version, or keep v1 and stop the dev-100 track.
