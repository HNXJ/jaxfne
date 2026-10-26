# CAMINOS P3 result (scored under signed v6 and amendment v7)

Verdict: **PASS under v7.** R1–R3 pass (v6); R4 fails under v6 and R4v7 passes. With the
external drive at p·NE sources, the port shows the critical-p onset of Fig. 1E: no seed
oscillates at p = 0.02 while E fires at 0.61–0.62 Hz, and all seeds oscillate from p = 0.04.
This rests on the v7 assumption that Fig. 1E scales the drive with p; the paper does not
state it for the sweep. Nothing was tuned after scoring.

The P3 criterion R2 (lag positive, mean 6–11 ms) is met. Against the paper's 8.5 ms (PLoS CB, primary)
the lag is off (e 0.15–0.26) and is recorded as a finding that needs X (`CALIBRATION.md` §10, Hamm
2026-10-14); the P3 PASS on R2 is not a reproduction of the 8.5 ms lag.

## R4v7 (amendment v7, signed 2026-10-12)

- Runs: WSL2 CUDA, float32, dt 0.1 ms, T 12 s, seeds 1–3 at p = 0.02–0.10, one run at a time,
  2026-10-12T13:12Z to 14:04Z. All 27 runs carry git SHA 6d0fd783; its caminos code equals
  baca68e7.
- Identity check: the p = 0.07 count arrays equal the v6 arrays for seeds 1–3
  (`numpy.array_equal`, PASS). Extended to seeds 4–5 at the same commit (Hamm 2026-10-12;
  2026-10-12T14:12Z–14:16Z, `F:\warehouse\caminos\scored\v7_id45_20261012\`): PASS. R1–R3
  (v6) and R4v7 therefore score one realization at p = 0.07 for all five reference seeds.
- Scoring: `caminos_analysis.py <dir> --ext-sources pNE`; `r4.pass` true, complete, 27 runs.
  E-rate floor (≥ 0.1 Hz in each p = 0.02 seed): 0.619, 0.606, 0.606 Hz, met.
- Outputs: `F:\warehouse\caminos\scored\v7_20261012\`. Score copied to `scores/score_v7.json`.

| p | f_pk (Hz) | peak excess | oscillates | E (Hz) | PV (Hz) | SOM (Hz) | lag (ms) |
|---|---|---|---|---|---|---|---|
| 0.02 | 20–21 | 2.8–3.1 | 0 of 3 | 0.61–0.62 | 6.3–6.4 | 0.7–0.8 | 14.0 |
| 0.03 | 22–23 | 22.3–38.0 | 1 of 3 | 0.95–0.97 | 8.9–9.0 | 2.1 | 11.8–12.4 |
| 0.04 | 27–28 | 34.1–47.2 | 3 of 3 | 1.06–1.08 | 9.9–10.0 | 3.3 | 10.5–10.8 |
| 0.05 | 30 | 32.4–35.7 | 3 of 3 | 1.09–1.10 | 10.4–10.6 | 4.5–4.6 | 9.0–9.2 |
| 0.06 | 32 | 15.9–21.1 | 3 of 3 | 1.11–1.12 | 10.9–11.1 | 5.8–6.0 | 7.6–8.0 |
| 0.07 | 33–34 | 13.8–16.3 | 3 of 3 | 1.10–1.13 | 11.4 | 7.0–7.3 | 6.9–7.1 |
| 0.08 | 35–36 | 11.6–14.2 | 3 of 3 | 1.06–1.10 | 11.5–11.6 | 7.9–8.1 | 6.5–6.6 |
| 0.09 | 37–38 | 8.3–12.5 | 3 of 3 | 1.02–1.06 | 11.6–11.7 | 8.6–9.1 | 6.0–6.3 |
| 0.10 | 38–39 | 8.4–9.1 | 3 of 3 | 0.99–1.00 | 11.6–11.8 | 9.4–9.7 | 5.8–5.9 |

- The smallest p at which all seeds oscillate is 0.04.
- At p = 0.02 the peak excess (2.8–3.1) is below the threshold of 4.
- At p = 0.03, seeds 2 and 3 fail only the band-edge rule: f_pk is 22 Hz, within 2 Hz of
  20 Hz, while their peak excess is 22–26. The onset between 0.02 and 0.04 therefore
  depends partly on the band edge.
- f_pk rises with p, from 20–21 Hz to 38–39 Hz, and the PV→SOM lag falls from 14.0 to 5.9 ms.

The v6 section below is kept as scored (B2).

## v6 result

v6 verdict: **FAIL.** R1–R3 pass; R4 fails. The reference model reproduces gamma, the PV→SOM
lag and the rate order at p = 0.07, but not the critical-p onset of Fig. 1E.

- Prereg: `PREREG_P3.md`, signed v6 (Hamm, 2026-10-11), code at `l2-caminos` e2501025.
- Runs: WSL2 CUDA, float32, dt 0.1 ms, T 12 s, one run at a time, 2026-10-11T23:11Z to
  2026-10-12T00:27Z. Every scored run carries git SHA e2501025.
- Outputs: `F:\warehouse\caminos\scored\` (counts, manifests, `score.json`, run log).
  Scores copied to `scores/`.
- Scoring: `caminos_analysis.py <run_dir>`; 29 scored runs, none excluded.

## Scored tests

| Test | Criterion | Value | Result |
|---|---|---|---|
| R1 | every seed oscillates, f_pk 30–40 Hz (seeds 1–5, p = 0.07) | all oscillate; f_pk 33–34 Hz, mean 33.8 | PASS |
| R2 | lag positive in every seed, mean 6–11 ms | 6.9–7.3 ms, mean 7.04 | PASS |
| R3 | PV > SOM > E in every seed, mean E 1–3 Hz | PV 11.3–11.4, SOM 6.6–7.3, E 1.05–1.14 Hz; mean E 1.11 | PASS |
| R4 | no seed oscillates at p = 0.02; all seeds oscillate at p = 0.07 | all 27 runs oscillate, p = 0.02 included; smallest p at which all seeds oscillate: 0.02 | FAIL |

## R4 in detail (seeds 1–3)

| p | f_pk (Hz) | peak excess | E (Hz) | PV (Hz) | lag (ms) |
|---|---|---|---|---|---|
| 0.02 | 28–29 | 55–98 | 3.4–3.6 | 31.2–31.5 | 10.4–10.6 |
| 0.03 | 32–33 | 72–96 | 2.4–2.5 | 22.7–22.9 | 9.6 |
| 0.04 | 34 | 62–63 | 1.9 | 17.8–18.0 | 8.7–9.0 |
| 0.05 | 34 | 28–34 | 1.5 | 14.8–15.0 | 7.9–8.1 |
| 0.06 | 34–35 | 15–21 | 1.3 | 12.8–12.9 | 7.4–7.5 |
| 0.07 | 33–34 | 14–16 | 1.1 | 11.4 | 6.9–7.1 |
| 0.08 | 33–34 | 10–13 | 1.0 | 10.2–10.3 | 6.6–6.8 |
| 0.09 | 32–33 | 10–13 | 0.85–0.88 | 9.4–9.5 | 6.3–6.5 |
| 0.10 | 33–34 | 7–9 | 0.76–0.77 | 8.6–8.7 | 6.1–6.3 |

The trend runs against Fig. 1E. Peak excess falls overall as p rises, from seed means
of 81–83 at p = 0.02–0.03 to 8 at p = 0.10, so the sparsest networks oscillate most
strongly. The threshold is 4; the lowest run is 7.1. Lower p also raises E and PV rates, so the network at low p sits in a
higher-drive regime. Which spec element produces this is not tested here; S3–S5 need driver options first.

Candidate (inferred, untested): the external drive does not follow p. The Methods give
the source count as p·NE, and the prereg fixed it at 560 (p·NE at p = 0.07) for every p
(`caminos_jax.py:289`, `ext_sources`). If Fig. 1E scales the drive with p, the sweep gave
p = 0.02 3.5× its drive (560 against 160 sources), which matches the rate rise at low p.
Testing this changes the spec, so it needs its own preregistered run.

## Sensitivity (reported, not scored)

Seed 1, p = 0.07, against the scored seed-1 run (f_pk 33, E 1.135, PV 11.42, SOM 7.29,
lag 6.9) and the seeds 1–5 range (f_pk 33–34, E 1.05–1.14, PV 11.26–11.42, SOM 6.60–7.32,
lag 6.9–7.3):

| Run | f_pk | E | PV | SOM | lag | oscillates |
|---|---|---|---|---|---|---|
| S1 float64 | 34 | 1.121 | 11.42 | 7.26 | 7.0 | yes |
| S2 dt 0.05 ms | 33 | 1.129 | 11.33 | 7.22 | 7.05 | yes |

S3–S5 ran later at `l2-caminos` 6ff832a1 (code = baca68e7, second review PASS):

| Run | f_pk | E | PV | SOM | lag | oscillates |
|---|---|---|---|---|---|---|
| S3 fixed in-degree | 35 | 1.040 | 11.09 | 6.69 | 6.8 | yes |
| S4 shared drive | — | 0.000 | 0.0004 | 0.000 | — | no (E silent: 0 spikes; PV 3) |
| S5 Brian2 step order | 33 | 1.146 | 11.47 | 7.47 | 7.0 | yes |

- S3 and S5 stay inside the R1–R3 bands. Their shifts are at or just past the seed range
  (S3: f_pk +1 Hz, E, PV and lag slightly low; S5: E, PV and SOM slightly high).
- S4 as preregistered is not a test of shared input alone. With 560 shared trains, each
  reaching a cell with probability p, a cell sees about p·560 ≈ 39 sources instead of 560,
  so its mean drive falls about 14-fold and the network goes silent. A shared-drive test at
  equal mean drive needs NE = 8000 trains at probability p (p·NE = 560 per cell); that is
  a new run.

Every S1/S2 shift is below the between-seed range except S1's f_pk. That shift is 1 Hz,
equal to the seed range and to the 1 Hz spectral grid. `score()` lists these runs as not
scored (they are not the reference configuration); the values come from `analyze_run` on
the same code.

## Ledger

Rows are in `LEDGER.md` (B2).
