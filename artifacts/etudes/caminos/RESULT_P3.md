# CAMINOS P3 result (scored under signed v6)

Verdict: **FAIL.** R1–R3 pass; R4 fails. The reference model reproduces gamma, the PV→SOM
lag and the rate order at p = 0.07, but not the critical-p onset of Fig. 1E. Nothing was
tuned after scoring.

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
higher-drive regime. Which spec element produces this (the external drive, weight scaling
with p, or the step order) is not tested here; S3–S5 need driver options first.

## Sensitivity (reported, not scored)

Seed 1, p = 0.07, against the scored seed-1 run (f_pk 33, E 1.135, PV 11.42, SOM 7.29,
lag 6.9) and the seeds 1–5 range (f_pk 33–34, E 1.05–1.14, PV 11.26–11.42, SOM 6.60–7.32,
lag 6.9–7.3):

| Run | f_pk | E | PV | SOM | lag | oscillates |
|---|---|---|---|---|---|---|
| S1 float64 | 34 | 1.121 | 11.42 | 7.26 | 7.0 | yes |
| S2 dt 0.05 ms | 33 | 1.129 | 11.33 | 7.22 | 7.05 | yes |

Every S1/S2 shift is below the between-seed range except S1's f_pk. That shift is 1 Hz,
equal to the seed range and to the 1 Hz spectral grid. `score()` lists these runs as not
scored (they are not the reference configuration); the values come from `analyze_run` on
the same code.

## Ledger

Rows are in `LEDGER.md` (B2).
