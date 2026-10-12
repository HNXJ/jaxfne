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
| R4v7 | Oscillation at p = 0.02, 0.03, …, 0.10, seeds 1–3, drive p·NE | no seed oscillates at p = 0.02 while E still fires (mean E rate ≥ 0.1 Hz in every p = 0.02 seed), and all seeds oscillate at p = 0.07; the smallest p at which all seeds oscillate is reported |

- The E rate floor separates "asynchronous" from "silent": a silent network at p = 0.02
  meets "does not oscillate" vacuously (v6 rule), so it does not pass. The scorer's
  `r4.pass` does not apply the floor; R4v7 passes only when `r4.pass` is true and
  `per_run[].rates_Hz.E ≥ 0.1` for each p = 0.02 seed in the same `score.json`. E, PV and
  SOM rates are reported at every p.
- Completeness: all 9 values of p × seeds 1–3, T = 12 s, dt = 0.1 ms, one code SHA. Anything
  else scores as "incomplete" (fail). Scoring with `--ext-sources pNE` also prints R1–R3;
  they read "incomplete" (no seeds 4–5) and are not part of v7.
- Identity check before scoring: for seeds 1–3, the p = 0.07 count arrays
  (`counts_seed<s>_p0.07_float32_dt1e-04_src-pNE.npy`) must equal the v6 arrays
  (`F:\warehouse\caminos\scored\caminos_scored\counts_seed<s>_p0.07_float32_dt1e-04.npy`)
  element for element (`numpy.array_equal`). Both come from the same WSL2 CUDA environment
  (`~/venv_jaxfne`). A mismatch stops scoring and is reported; the code and the environment
  are then checked before any rerun.
- Runs: WSL2 CUDA, float32, one at a time after a jchat GPU claim (B5).
- Code: the runs check out the commit that carries this signed text. Its caminos code files
  (`caminos_jax.py`, `run_caminos.py`, `caminos_analysis.py`) must equal `l2-caminos`
  baca68e7 (`git diff --quiet baca68e7 -- <files>`); every run records that one commit SHA.
  Runs use `run_caminos.py --seeds 1 2 3 --p <p> --T 12.0 --ext-sources pNE --signed`;
  scoring uses `caminos_analysis.py <dir> --ext-sources pNE`, which scores only
  default-variant runs with the p·NE drive.

## Outcome language

- R4v7 passes: "With external drive p·NE, the port shows the critical-p onset of Fig. 1E."
  The overall CAMINOS Fig. 1 verdict is then R1–R3 (v6) plus R4v7.
- R4v7 fails: reported with its numbers. Private p·NE sources are then excluded as the
  cause; shared trains at equal mean drive (NE trains at probability p; the v6 S4 run cut
  the drive about 14-fold and fell silent) remains a candidate. S3 and S5 left R1–R3 inside
  their bands at p = 0.07.
