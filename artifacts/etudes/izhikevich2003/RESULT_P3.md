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
1–5; `sensitivity_v7\s1_report.json`. Mean extra u per spike from the first half-step overshoot
3.04–3.09.

| | N1 Hz | N2 CV | N3 (f_pk, PE) | N4 (f_pk, PE) |
|---|---|---|---|---|
| reference | 14.12–14.46 | 0.388–0.400 | 9.5–13.0 Hz, 0.32–1.95: fails all seeds | 42.0–48.5 Hz, 3.27–6.13: fails seeds 2, 5 |
| kernel | 8.48–8.83 | 0.400–0.410 | passes all seeds | passes all seeds |

N2 fails on both arms: the low CV belongs to the network as specified, not to the kernel. S1 moves
N3 and N4 across their band edge, which by the prereg calls for a separately preregistered kernel
variant arm that reproduces the overshoot (Hamm's decision).

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
