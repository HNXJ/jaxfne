# brunel2000 P3 result under signed v4

Verdict: **close**. Every estimand lies within its band and Q1–Q3 pass on all five held-out
seeds (`PREREG_P3_v4.md`, signed by Hamm 2026-10-19).

Runs: Fig. 8 points A–D at N=12500, seeds 4–8, dt 0.1 ms, 2500 ms with the first 500 ms dropped,
float32 on the CPU backend, code SHA 4969d01bca6649e95d4fe3c69efee7f3f3763140 (the commit that
carries v4), jax 0.10.1. Receipts in `C:/workspace/jaxfne_receipts/brunel2000_p3v4/scored/`
(manifest sha256 4526d456736b5e25...); 20 runs, each exit 0. The seed sets the connectivity draw
and the Poisson drive; every estimand differs across seeds (values below).

| Id | Quantity | Seeds 4–8 | z (mean ± sd) | Paper | e | Band | Pass |
|---|---|---|---|---|---|---|---|
| E1 | B cell rate, Hz | 59.53, 59.62, 59.18, 59.19, 59.14 | 59.33 ± 0.22 | 60.7 | 0.023 | 0.08 | yes |
| E2 | B global frequency, Hz | 181, 182, 187, 176, 186 | 182.4 ± 4.4 | ~180 | 0.013 | 0.06 | yes |
| E3 | C cell rate, Hz | 37.95, 37.94, 37.82, 37.84, 37.78 | 37.87 ± 0.07 | 37.7 | 0.0044 | 0.01 | yes |
| E4 | D cell rate, Hz | 5.69, 5.70, 5.59, 5.58, 5.59 | 5.63 ± 0.06 | 5.5 | 0.024 | 0.18 | yes |
| E5 | D global frequency, Hz | 22, 19, 18, 25, 19 | 20.6 ± 2.9 | 22 | 0.064 | 0.32 | yes |

| Id | Point | Criterion | Seeds 4–8 | Pass |
|---|---|---|---|---|
| Q1 | A, SR | label `saturated` or `oscillatory`, κ ≥ 0.5, CV ≤ 0.1 | `saturated`, κ 1.00, CV 0.0004–0.0006 | yes |
| Q2 | B, SI | label `oscillatory`, CV > 0.1 | `oscillatory`, CV 0.78–0.83, κ 0.30 | yes |
| Q2 | D, SI | label `oscillatory`, CV > 0.1 | `oscillatory`, CV 0.63–0.64, κ 0.083–0.096 | yes |
| Q3 | C, AI | spectral excess < 5, CV > 0.1 | excess 2.95–3.27, CV 0.40–0.41 | yes |

Unscored beside Q3: the classifier label `asynchronous` (reading (i), declined) fails at C on every
seed (label `irregular`, CV 0.40–0.41 below its 0.7 bound, κ 0.20 above its 0.1 bound). C keeps
the paper's rate and a stationary global signal while its cells fire more regularly than the
classifier's AI bounds; the pilot showed the same at dt 0.5 and 0.1.

Under the declined template band e ≤ 0.05, E1–E4 would pass and E5 (0.064) would not.

Float64 check (checklist P3, seed 4 at point C; code cc64eb73, which adds only the `dtype`
option; receipts `.../brunel2000_p3v4/unscored_f64/`, manifest sha256 cf233ec057371888...):
rate 37.955 Hz against 37.950 in float32 (0.012%), CV 0.406 against 0.405, κ 0.204 against 0.204,
spectral excess 2.88 against 2.95. Q3 and E3 hold in both; float32 is adequate here.

Sensitivity set (declared in v4; never enters the score, checklist P6). S2 and S3 ran on seeds 4-5 at
points A-D (code `cc64eb73`, CPU float32; receipts `.../brunel2000_p3v4/unscored_S2/` and `unscored_S3/`,
manifest sha256 `5d32bdcc3d9a21a650d969b3e711666c052d53f48042020fe1e5037749742300` and
`1adea95b6f832a8f73fc094e7be8547e4037548fd25b9747e348dea6ddec1a49`). S2 (5000 ms runs, last 4500 ms
scored): every point keeps its regime label and its rate within 1.5% of the scored run (A 333.3 Hz,
B 59.2-59.4, C 37.9, D 5.5-5.6). S3 (dt 0.05 ms): labels hold except C seed 4, which reads oscillatory
(spectral excess 5.46 against 2.95 scored; seed 5 stays irregular at 4.54), and B's CV rises from 0.78-0.83
to 0.93-0.97. Point C's irregular label is therefore close to the classifier's boundary under a halved
step; no verdict changes (the conditional rests on the E5 band, stated above). S1 (finite size: N=6250
with the N=12500 in-degrees held) ran on seeds 4-8 at points A-D (code `3d7a5c9b`, CPU float32, jax
0.10.1; receipts `.../brunel2000_p3v4/unscored_S1/`, manifest sha256
`e9982b5fbdd5d25843d5a1c7b0f47fa91cecab4785854046d75643301fc097ae`). A, B and C keep their labels, with
rates within +0.07% (A), +0.8 to +2.3% (B) and +1.5 to +2.3% (C) of the scored runs; B's CV rises from
0.78-0.83 to 0.90-0.93. At D the rate moves by -7.6 to +4.0% and the label falls from oscillatory to
irregular on all five seeds: spectral excess 3.4-4.0 against 5.1-7.9 scored, under the classifier's
oscillation floor of 5 (`jaxfne.analysis.metrics`). Q2 at D does not hold at half size; D's slow global
oscillation is weaker there. No verdict changes (the sensitivity set never enters the score).
