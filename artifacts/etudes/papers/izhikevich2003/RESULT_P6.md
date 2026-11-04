# Izhikevich 2003 P6 result: Fig. 3 through the jaxfne public API

Prereg: `core/PREREG_P6_v2_signed.md` (signed v2, Hamm 2026-10-31; sha256 `fe73f816…`). Receipt:
`core/p6_v2_score.json`. Command, from a clean clone at 2ca0e38b:
`run_izh_core.py scored --seeds 2 3 4 5 6 --t-s 10.2 --signed`. Gates 1 to 3 held: realization checks pass in all
five seeds; the scorer is `run_izh.score_network_seed` with θ 4 and the 8–13 Hz band
(`calibration_v7_15f36087.json`); one code SHA, five seeds at 10.2 s.

**Verdict: PASS (Q1–Q5).** Fig. 3 (N1–N4) is reproduced through the jaxfne public API within the declared margins
(T-jaxfne); N2 stays a failed paper claim.

| Id | Estimand, mean over seeds 2–6 | P6 | P5 (seeds 1–5) | Margin | Verdict |
|---|---|---|---|---|---|
| Q1 | N1 mean rate (Hz) | 8.702 | 8.677 | 0.29 | pass |
| Q2 | N2 median ISI CV | 0.4037 | 0.4028 | 0.0085 | pass |
| Q3 | N3 f_pk (Hz) | 9.2 | 9.10 | 1.0 | pass |
| Q4 | N4 f_pk (Hz) | 37.1 | 38.1 | 5.2 | pass |
| Q5 | arm verdicts N1 pass, N2 fail (as P5); N3 PE, N4 PE | 6.83, 5.67 | 8.01, 5.83 | 5.8, 2.5 | pass |

Per-seed band calls (reported, not scored under option B): N3 PE 3.53, 7.92, 6.27, 6.93, 9.53 (seed 2 under θ);
N4 PE 5.21, 4.96, 4.80, 6.73, 6.67 (all at or above θ). N2 median CV 0.402–0.406 in every seed, 970–976
qualifying cells.
