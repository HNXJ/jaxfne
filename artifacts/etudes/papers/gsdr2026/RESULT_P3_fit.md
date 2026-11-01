# GSDR P3 fit result (scored, 2026-11-01)

Protocol: `PREREG_P3_fit_signed.md` (Hamm 2026-11-01), code 8516a005 in the frozen WSL clone `~/jaxfne_gsdrP3`,
CPU float32. Fits: `fit_gsdr_500.py --generations 30 --population 6 --seed s --mcdp`, s = 9–13. Evaluation:
`diag/regime_at_gstar.py --gstar <g*> --stim 0.6 --seed t` on held-out trials t = 14–18, as signed (no
`--morphology` flag, so the script default `reference` applied; the fits ran `single`). Receipts
`F:/warehouse/gsdr/P3_fit_8516a005`; per-file sha256 in `run/p3_fit_score.json`
(sha256 e77216f9…, written by `run/score_p3_fit.py`).

**Verdict: FAIL** (0 of 5 seeds pass G2 and G3; PASS needs 4).

| seed | best loss g0 → g30 (G1) | kappa_500_1000 mean [min, max] (G2 < 0.1) | rate Hz (G3 5–50) | G2 | G3 |
|---|---|---|---|---|---|
| 9 | 4493.76 → 4338.09 | 0.596 [0.590, 0.605] | 69.4 | no | no |
| 10 | 4493.76 → 4233.09 | 0.597 [0.589, 0.605] | 69.5 | no | no |
| 11 | 4493.76 → 4247.96 | 0.596 [0.590, 0.605] | 69.4 | no | no |
| 12 | 4493.76 → 4274.72 | 0.595 [0.587, 0.605] | 69.5 | no | no |
| 13 | 4493.76 → 4397.60 | 0.595 [0.588, 0.604] | 69.5 | no | no |

G1 (reported): the loss falls on 5 of 5 seeds (2.1–5.8 %). G3 uses `mean_rate_hz`; the E stimulus-window rate
`e_stim_window_mean_hz` is 119.7–119.8 Hz, also outside 5–50 Hz. The ES step clipped 158 728–171 704 candidate
coordinates per fit to [0, 10] (`es_clipped_coords`).

Run notes: the first launch refused `--resume` on empty folders before any fit ran; the shared 24 GB memory
scope was OOM-killed at generation 17–18, and every fit resumed from its generation-10 checkpoint (the resume
guards checked seed, population, es_sigma and the clip total). Logs record both.

Sensitivity (unscored): the same g* evaluated with `--morphology single`, trial 14, gives kappa 0.991 (seed 9)
and 0.927 (seed 13) at 23.7 and 24.1 Hz (`unscored_single/`): G2 fails there too, matching the in-sample fit
trials (kappa 0.90–1.00) and CALIBRATION 6.3–6.13. The near-identical reference-morphology values across seeds
are unexplained (inferred: the reference circuit is insensitive to the fitted conductances at stim 0.6); the
verdict does not depend on it. The next prereg names `--morphology` explicitly.
