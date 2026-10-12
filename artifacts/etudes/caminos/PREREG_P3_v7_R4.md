# L2-P3 amendment v7: CAMINOS R4 with external drive p·NE

Status: **draft**, for Hamm's signature. It amends R4 of `PREREG_P3.md` (signed v6) only.
R1–R3 and their v6 verdicts stand. No scored v7 run before the signature.

## Why

v6 R4 failed in reverse: every run at p = 0.02–0.10 oscillates, and peak excess is largest
at low p (`RESULT_P3.md`). v6 fixed the external drive at 560 sources per E and PV cell at
every p. The Methods give the count as p·NE, which is 560 only at p = 0.07. At p = 0.02 the
v6 sweep gave each cell 560 sources where p·NE gives 160.

## Change

- External drive: round(p·NE) private 4 Hz Poisson sources per E and PV cell, at each p.
  At p = 0.07 this is 560, the v6 value, so the p = 0.07 network is unchanged.
- Assumption: Fig. 1E varies the drive with p through p·NE. The paper does not state this
  for the sweep; v7 tests the literal reading.
- Nothing else changes: weights, sizes, Bernoulli connectivity, dt, T, the oscillation
  rule (peak excess threshold 4) and the analysis code.

## Test

| Id | Estimand | Pass |
|---|---|---|
| R4v7 | Oscillation at p = 0.02, 0.03, …, 0.10, seeds 1–3, drive p·NE | no seed oscillates at p = 0.02, and all seeds oscillate at p = 0.07; the smallest p at which all seeds oscillate is reported |

- Completeness: all 9 values of p × seeds 1–3, T = 12 s, dt = 0.1 ms, one code SHA. Anything
  else scores as "incomplete" (fail).
- The p = 0.07 runs must reproduce the v6 counts bit for bit (same seed, same drive). A
  mismatch voids the run.
- Runs: WSL2 CUDA, float32, one at a time after a jchat GPU claim (B5).
- The code SHA (driver option `--ext-sources pNE`, and the scorer accepting it as the v7
  reference) is fixed here before signing.

## Outcome language

- R4v7 passes: "With external drive p·NE, the port shows the critical-p onset of Fig. 1E."
  The overall CAMINOS Fig. 1 verdict is then R1–R3 (v6) plus R4v7.
- R4v7 fails: reported with its numbers. The drive reading is then excluded as the cause,
  and S3–S5 are the next candidates.
