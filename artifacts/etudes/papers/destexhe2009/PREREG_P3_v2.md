# Destexhe 2009 preregistration P3 v2 (draft, UNSIGNED — no scored run may use it)

Supersedes the unsigned `PREREG_P3.md` (pinned as written): its held-out seeds 4–8 were spent at the
step-6 baseline (CALIBRATION.md), and its kick (300 Hz, 6 nS to RS) is replaced by the authors' code
value. Paper Fig. 6B, cortical self-sustained AI state; z is the jaxfne AdEx route at θ₀ (no fit).

## Fixed protocol (θ₀)

Driver `run_destexhe.py fig6b_cont --seed S --tag fig6b_cont --out <dir> --kick-fraction 0.25
--kick-rate-hz 286` (the code-value kick: 0.25 of RS at 286 Hz for 50 ms, CALIBRATION.md row 11).
N = 2000 (1600 RSweak + 400 FS), p = 0.02 with `sampling="capped_bernoulli"`, ge = 6 nS, gi = 67 nS,
τE/τI = 5/10 ms, reversals 0/−80 mV, dt = 0.1 ms, authors' detection `Vpeak_mV = -50.0` (Hamm ruling
2026-10-14), 10 s continuous free run, float32 on CPU. One frozen clone; its SHA goes in the result.

## Estimands (unchanged from v1)

| Id | y (Fig. 6B) | z | criterion |
|---|---|---|---|
| E1 alive | self-sustained; 6A control dies | spikes in the last 1 s; control `--strong-adaptation` dead by 10 s | z alive and control dead |
| E2 CV_ISI | 2.47 | `cv_isi` over the free run | e = \|z−y\|/\|y\| ≤ 0.05 |
| E3 CC | 0.005 | `pairwise_cc`, 5 ms bins, 500 disjoint pairs | \|z−y\| ≤ 0.01 |
| E4 rates | PY 1–3 Hz order (inset) | `mean_rate_hz` per carrier | RS < FS and RS in 1–10 Hz |

## Seeds and verdict

Scoring seeds 9–13 (unused by any run). Control C1 (Fig. 6A) on seeds 9–10. A seed passes when every
gated estimand closes. Proposed verdict: PASS when at least 4 of 5 scoring seeds pass and C1 holds;
otherwise FAIL, recorded with its per-seed table.

## Pilot evidence (unscored, seeds 0–3 and 6A seeds 0–1, code 29583485)

Receipts `F:/warehouse/destexhe2009/P3_pilot_29583485/score_seed*_6{a,b}.json`:

| seed | alive | CV (e) | CC (abs) | rate Hz |
|---|---|---|---|---|
| 0 | yes | 2.575 (0.042) | 0.0063 (0.0013) | 48.7 |
| 1 | no | 2.426 (0.018) | 0.111 (0.106) | 13.1 |
| 2 | yes | 2.626 (0.063) | 0.0044 (0.0006) | 49.0 |
| 3 | yes | 2.636 (0.067) | 0.0026 (0.0024) | 48.2 |

6A control dead on seeds 0 and 1 (rates 0.39, 0.57 Hz). E1–E3 close jointly on seed 0 only; E4 is off on
every alive seed (whole-network rate 48–49 Hz). With E4 gated the expected verdict is FAIL; without E4,
the pilots and CALIBRATION step 6 (joint close on 1 of 4 and 1 of 8 seeds) also predict FAIL.

## Owner decisions before signing

1. E4: gated as in v1, or reported only (the paper's small-network text notes about 40 Hz; v1 row E4).
2. The 4-of-5 aggregation.
3. Sign and run, or defer Destexhe to 0.5.10 (it is outside the 0.5.9 checklist, S16).
