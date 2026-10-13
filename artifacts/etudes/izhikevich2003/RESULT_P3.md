# Izhikevich 2003 P3 result (prereg v7, signed 2026-10-13)

Verdict: **FAIL**. Fig. 2 fails on C4 and C8; Fig. 3 fails on N2. Scored runs: code `423a2f88`
(one SHA for Fig. 2 and Fig. 3), WSL CPU, float32, θ = 4, alpha 8–13 Hz from
`calibration/calibration_v7_15f36087.json`. Receipts: `F:\warehouse\izh2003\scored_v7\`
(`score.json`, `fig2/`, `fig3/`). Second review of the runner and scorer: PASS WITH NOTES
(2026-10-13), notes fixed before the run.

## Fig. 2 (single cells)

| Id | Cell | Verdict | Numbers |
|---|---|---|---|
| C1 | RS | pass | 8 spikes, 26.7 Hz, ISI ratio 2.53 |
| C2 | IB | pass | initial burst, 8 tail spikes |
| C3 | CH | pass | 5 bursts, onset 16.1 Hz |
| C4 | FS | **fail** | 133 Hz, ISI ratio 1.857 (pass ≤ 1.2) |
| C5 | LTS | pass | 83.3 Hz against RS 26.7 Hz |
| C6 | TC tonic | pass | I+ = 1: 7 spikes, no burst, CV 0.25 |
| C7 | TC rebound | pass | burst 5.7 ms after release |
| C8 | RZ | **fail** | 1 spike (pass: none); 3 maxima, decreasing, 29.8 Hz |

The reviewer's independent float64 Euler integration of the prereg's settings gave the kernel's
spike rows in all seven checked runs: C4 and C8 fail on the kernel's dynamics, not the runner.

## Fig. 3 (1000-cell network, seeds 1–5, 10.2 s, first 200 ms discarded)

| Id | Verdict | Per seed |
|---|---|---|
| N1 rate 4–16 Hz | pass | 8.48–8.83 Hz |
| N2 median ISI CV 0.6–1.5 | **fail** | 0.400–0.410 (966–975 cells qualify) |
| N3 alpha PE ≥ 4 | pass | f_pk 9.0–9.5 Hz, PE 5.25–12.80 |
| N4 gamma PE ≥ 4 | pass | f_pk 36.0–42.0 Hz, PE 4.67–7.85 |

## Sensitivity S1 (reported only)

Reference arm `net_reference.py` (float64 NumPy transcription of `net.m`, code `e67f3b86`), seeds
1–5. Mean extra u per spike from the first half-step overshoot 3.04–3.09.

Correction (2026-10-13): `cmd_s1` scored the reference arm's 1 ms rows as 0.5 ms rows, which
doubled its rate and halved its time axis. The first report, `sensitivity_v7\s1_report.json`
(`scores/p3_v7_s1_report.json`), is kept as a receipt. The table below is from
`scores/p3_v7_s1_report_dtfix.json`: same runs, each arm scored on its own step (fix and test
`test_p3_s1_scores_each_arm_on_its_own_step`, 2026-10-13).

| | N1 Hz | N2 CV | N3 (f_pk, PE) | N4 (f_pk, PE) |
|---|---|---|---|---|
| reference | 7.05–7.23 | 0.387–0.399 | seed 1: 11.0 Hz, 0.96; seeds 2–5: 8.0–8.5 Hz, 6.65–11.34, inside the 1 Hz edge: fails all seeds | 33.0–40.0 Hz, 4.25–7.93: passes all seeds |
| kernel | 8.48–8.83 | 0.400–0.410 | passes all seeds | passes all seeds |

N2 fails on both arms: the low CV belongs to the network as specified, not to the kernel. S1 moves
N3 across its band edge (the reference alpha peak sits at 8.0–8.5 Hz), so the prereg calls for a
kernel variant arm (`PREREG_P3b.md`, draft).

## Sensitivity S2–S4 (reported only)

Code `4a506210`, WSL CPU, seed 1; `sensitivity_v7\S{2,3,4}*`, merged in
`scores/p3_v7_sens_S2_S4.json`. S3 passes noise_scale = paper σ·√2, which the kernel's
√(0.5/dt) scaling realizes as 2σ (10 E, 4 I) at dt 0.25, the per-ms variance of the scored run.

| Run | N1 Hz | N2 CV | N3 (f_pk, PE) | N4 (f_pk, PE) |
|---|---|---|---|---|
| scored, seed 1 | 8.59 | 0.403 fail | 9.0, 6.21 | 39.5, 4.95 |
| S2 float64 | 8.65 | 0.412 fail | 9.0, 5.22 | 38.0, 5.72 |
| S3 dt 0.25, W = 4S | 8.77 | 0.399 fail | 9.5, 6.26 | 43.5, 4.72 |

S4 (C7 with I = −10 during the step): pass, burst 8.4 ms after release, 3 spikes (scored: 5.7 ms,
7 spikes). No verdict changes with precision, step or the C7 hold current.

## Amendment v8.2 (post hoc, signed 2026-10-13): Fig. 2 passes 8 of 8

`PREREG_P3_v8.md`. Post hoc: both changes were chosen after the v7 FAIL and the diagnostics, so this
describes the kernel and is not a prospective test. The v7 verdict above stands.
Score `scores/p3_v8_score.json` (`F:\warehouse\izh2003\v8\`); v7 runs at `423a2f88`, the v8 C8 run
at `9f93d76e` (WSL CPU, float32).

| Criterion | v8 rule | Result |
|---|---|---|
| C4 | rate ≥ 50 Hz, ISI ratio below both C1 and C5 | 133 Hz, 1.857 against RS 2.534 and LTS 4.387: pass |
| C8 | pulse I = 2 for 1 ms; v7's rule | 0 spikes, 3 maxima decreasing, 25.2 Hz: pass |
| C1–C3, C5–C7 | v7 | pass (copied) |

Fig. 3 keeps the v7 verdicts: N1, N3 and N4 pass; N2 fails as a paper claim (`net.m` itself gives
CV 0.39).

## P3b (net.m update order): STOPPED at the equivalence gate

`PREREG_P3b.md` signed v4, gate amendment v5 (Hamm, 2026-10-13). No scored run took place. P3
(signed v7) keeps its verdict.

### v4 gate: FAIL (`scores/p3b_gate_v4.json`, code `9ff85e7f` plus the uncommitted arm)

| Step | First divergent row | Counts (ref / kernel) | Bar |
|---|---|---|---|
| 1 (50 cells, thal 6) | 270 | 954 / 958 | rounding-flip bar fails |
| 2 (full net) | 615 | 7421 / 7445 | extra u at rtol 1e-9 before row 615 fails |

The reference fails the same bar against itself when thal moves by one ulp: step 1 diverges at row
310 (counts 954 / 965), and v differs by more than 1e-9 from row 71. That is the basis of v5.

### v5 gate: FAIL (`scores/p3b_gate_v5.json`, code `9b6b4a77` in the work clone; integrated as `87353ec9`)

| Step | R | Rasters before R | Extra u before R (rtol 1e-6) | Counts within 1 % | Verdict |
|---|---|---|---|---|---|
| 1 | 157 | identical | agrees | 954 / 958 | PASS |
| 2 | 446 | identical | fails | 7421 / 7445 | FAIL |

Per the amendment, a v5 failure stops P3b.

### Diagnostic (unscored)

Step 2, rows before R, 1803 spikes with extra u:
- 1-ulp control against the reference: max relative difference 1.4e-8, none over 1e-6.
- Kernel against the reference: max 3.0e-5, 5 over 1e-6, the worst at row 445, cell 800.

The JAX arm drifts from the NumPy reference faster than a one-ulp input change does. A likely source
is the summation order of `S @ fired` over up to 1000 terms (inferred, untested). The S1 reference
arm result (`RESULT_P3.md`) stands as the answer to the update-order question: N3 fails on every seed
under `net.m`'s order in the NumPy transcription.
