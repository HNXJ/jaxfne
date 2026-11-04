# destexhe2009 P3 result under signed v3

Verdict: **FAIL**. 0 of 5 scoring seeds close E1-E3 (PASS needs 4); the Fig. 6A control holds
(`prereg/P3_v3_signed.md`, signed by Hamm 2026-11-02: E4 reported, the 4-of-5 aggregation kept). The pilots and
CALIBRATION step 6 predicted FAIL.

Runs: `run_destexhe.py fig6b_cont --seed S --tag fig6b_cont --kick-fraction 0.25 --kick-rate-hz 286 --signed` on
seeds 9-13, and the same with `--strong-adaptation --tag fig6a_cont` on seeds 9-10; N = 2000, 10 s continuous free
run, dt 0.1 ms, float32 on the CPU (`jax_devices` cpu:0), code SHA 2d17488b94d5e030f0301a420fee95000a3ff857 from a
frozen clone; 32-42 s per run, each exit 0. Receipts `F:/warehouse/destexhe2009/P3v3_signed_2d17488b/`
(`scored/manifest.json` sha256 cbd71c87…, per-run `score_seed<S>_6{a,b}.json`, verdict `p3_v3_verdict.json` sha256
9a17e59e…). The manifest carries no signed flag; the runs wrote under a `scored/` path, which `lane_kit` refuses to
unsigned runs (inferred: signed).

| seed | E1 alive | E2 CV_ISI (e) | E3 CC (\|z−y\|) | E4 rate, whole network, Hz | pass |
|---|---|---|---|---|---|
| 9 | no | 2.362 (0.044) | 0.1136 (0.1086) | 9.4 | no |
| 10 | yes | 2.606 (0.055) | 0.0069 (0.0019) | 46.2 | no |
| 11 | yes | 2.604 (0.054) | 0.0085 (0.0035) | 46.0 | no |
| 12 | no | 2.323 (0.060) | 0.0796 (0.0746) | 16.8 | no |
| 13 | no | 2.322 (0.060) | 0.1287 (0.1237) | 4.8 | no |

Criteria: E1 spikes in the last 1 s; E2 e = |z − 2.47| / 2.47 ≤ 0.05; E3 |z − 0.005| ≤ 0.01. C1 (Fig. 6A control dead
by 10 s): seeds 9 and 10 dead, 0.40 and 0.50 Hz; holds. E4 (RS < FS, RS in 1-10 Hz) is reported only; the raster stores
no RS/FS split, so the table gives the whole-network rate.

Three seeds die before 10 s, against one of four pilots. The two alive seeds close E3 and miss E2 by 0.004-0.005 in e,
at 46 Hz.
