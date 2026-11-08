# Jansen-Rit 1995 P3 v1 result (Fig. 3)

Verdict: **PASS**, 5 of 5 scoring seeds (rule: at least 4 of 5). Prereg `prereg/P3_v1_signed.md` (signed
2026-11-08 under Hamm's standing go, pinned). Frozen clone at 7ef8ff99, driver `run_jr_fig3.py --seeds 10 11 12 13
14` at its defaults; receipts `F:/warehouse/jansen_rit1995/p3_v1_scored_7ef8ff99/fig3_seed{10..14}.json` (each
records `git_sha` 7ef8ff99). Score `score.json` there (sha256 ca0c9d75…), from `score_p3_v1.py`, which applies the
prereg's fixed classes and criteria; `test_score_p3_v1.py` breaks each criterion once and requires a fail. The scorer
was written after the driver had printed the scored rows; it adds no threshold or choice beyond the prereg.

| Seed | classes at C 68, 128, 135, 270, 675, 1350 | E1 | E2 | E3 | E4 |
|---|---|---|---|---|---|
| 10-14 | noise, alpha, alpha, spike-wave, spike-wave, noise | pass | pass | pass | pass |

Per seed (C 128 / 135 `osc`; f at 270 / 675 Hz; mean EEG at 135 mV): 10: 0.65 / 0.87, 5.0 / 2.75, 7.59;
11: 0.68 / 0.87, 5.25 / 2.75, 7.60; 12: 0.68 / 0.84, 5.25 / 3.0, 7.60; 13: 0.81 / 0.89, 5.25 / 3.0, 7.60;
14: 0.80 / 0.90, 5.25 / 2.75, 7.60.
Smallest margins: noise `osc` up to 0.26 against the 0.4 threshold; spike-wave SD 11.8 mV against 5 mV.

Scope: regime class, dominant frequency and EEG level of the standard set under the declared 2 ms input hold and
RK4 at 0.1 ms; the paper's waveform, hold and integrator are not scored (`SOURCE_P1.md`).
