# GSDR lane P3 fit preregistration (signed)

Status: **signed** (Hamm, 2026-11-01T01:08:55Z, as drafted in `PREREG_P3_fit.md`, pinned at ca9f2c83; only this
title and status changed).

The baseline prereg `PREREG_P3.md` (pinned) covers the unfitted circuit. This one covers the GSDR-MCDP fit
of the Fig. 6 circuit on the jaxfne core (`fit/fit_gsdr_500.py --mcdp`, antithetic-ES gradient, Hamm ruling
2026-10-31) and scores the fitted g* against the paper's Fig. 6 claim: the fitted network desynchronizes
during the stimulus (SOURCE_P1.md section 3, Fig. 6).

## Fixed protocol

- Fit: `fit_gsdr_500.py --out <dir>/seed<s> --generations 30 --population 6 --seed <s> --mcdp`, fit trial
  seeds 1–3, morphology `single`, loss weights psd 990 / rate 9 / kappa 100, CPU float32, one frozen
  clone (SHA in the result). Budget: 30 generations (the authors ran 500 epochs; CALIBRATION 6.3 moved the
  loss −9 % in 350 generations without MCDP).
- Optimizer seeds s = 9–13, one fit each (unused by any run).
- Evaluation of each g*_s: `diag/regime_at_gstar.py --gstar <dir>/seed<s>/g_star_uS_float32.npy --stim 0.6
  --seed <t> --out <json>` on held-out trial seeds t = 14–18 (unused by any fit or run).

## Estimands

| Id | y | z | criterion |
|---|---|---|---|
| G1 loss | fit reduces the loss | best loss at generation 30 vs generation 0 | z30 < z0 (reported, not gated) |
| G2 desynchronized | Fig. 6 stim kappa 0.0156 (data label) | mean `kappa_500_1000` over t = 14–18 | z < 0.1 (the asynchronous threshold of `activity_regime`) |
| G3 rate | 5–50 Hz penalty bounds | mean rate over t = 14–18 | 5 ≤ z ≤ 50 Hz |

A seed passes when G2 and G3 hold. Proposed verdict: PASS when at least 4 of 5 optimizer seeds pass;
otherwise FAIL, recorded with its per-seed table in the S figure (9E: each failure in its S figure).

## Pilot evidence (unscored, optimizer seeds 0–3, code ff12072c)

Receipts `F:/warehouse/gsdr/P3_pilot_mcdp/seed{0..3}/manifest.json` (30 generations, about 31 min each):
best loss 4493.8 → 4331.5 / 4352.5 / 4336.8 / 4267.0 (G1 holds on 4 of 4). The best candidate's fit
trials: active fraction 1.00, kappa 0.90–1.00, rate 19–22 Hz on every seed (G3 holds, G2 fails on
in-sample trials). The fit stays in the synchronous regime that CALIBRATION 6.3–6.13 records. Expected
verdict: FAIL.

## Owner decisions before signing

1. G2 threshold: 0.1 (regime boundary, proposed) or e ≤ 0.05 against 0.0156.
2. Budget: 30 generations (pilot pace) or more.
3. The 4-of-5 aggregation; sign, or defer GSDR to 0.5.10.
