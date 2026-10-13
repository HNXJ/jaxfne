# CAMINOS P6 result (prereg v2, signed 2026-10-13, margins C)

Verdict: **PASS**. CAMINOS Fig. 1 (R1–R3, R4v7) is reproduced through the jaxfne public API within
the declared margins (T-jaxfne). Runs: code `a5539d34` (B0 route (c), ADVANCED kernel
`simulate_adex(record_v=False)`), WSL2 CUDA, float32, T = 12 s, 29 runs plus C1. Receipts:
`F:\warehouse\caminos\p6_scored_v2\` and `…\p6_scored_v2_c1\`. Score:
`scores/p6_v2_score.json` (`score_p6.py`, analysis `caminos_analysis.py` unchanged).

| Id | P6 | P5 | Margin | Verdict |
|---|---|---|---|---|
| Q1 f_pk, mean seeds 1–5 | 33.80 Hz | 33.8 Hz | 1 Hz | pass |
| Q2 PV→SOM lag | 7.22 ms | 7.04 ms | 0.34 ms | pass |
| Q3 E rate | 1.079 Hz (−2.5 %) | 1.106 Hz | 6.1 % | pass |
| Q3 PV rate | 11.25 Hz (−1.1 %) | 11.37 Hz | 1.2 % | pass |
| Q3 SOM rate | 6.78 Hz (−3.8 %) | 7.05 Hz | 8.2 % | pass |
| Q4 R1, R2, R3, R4v7 | pass, pass, pass, pass | same | equal | pass |
| Q5 onset p (seeds 1–3 all oscillate) | 0.04 | 0.04 | 0.01 | pass |
| C1 fixed 560-source drive, p = 0.02 | oscillates, f_pk 29 Hz, PE 81.3 | oscillates, 28–29 Hz | — | pass |

Sweep (drive p·NE), seeds oscillating out of 3: p = 0.02: 0; 0.03: 1; 0.04–0.10: 3. E rate at
p = 0.02 is 0.60 Hz, above the 0.1 Hz floor.

PV sits 1.1 % below P5 against a 1.2 % margin, the closest call; the PV rate's P5 seed SD is
0.6 %.
