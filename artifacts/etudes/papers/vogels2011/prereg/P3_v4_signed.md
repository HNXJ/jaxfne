# vogels2011 preregistration P3 v4 (signed)

Status: **signed** (Hamm, 2026-11-02T02:22:28Z, as drafted in `P3_v4.md`, pinned at d007b8bc; only this title and
status changed).

P3 v3 (`P3_v3.md`, pinned 2026-11-02) with its first owner decision applied: Hamm 2026-11-02 chose the slope test for
CONV. Everything not named below is v3 as written. Paper: Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W
(2011), Science 334(6062):1569-1573, doi:10.1126/science.1211095. Target: Fig. 4, recurrent network.

## Change from v3

CONV (Hamm 2026-11-02): per seed, fit the least-squares slope b of the per-second E rate over 130-150 s
(`e_rate_per_s_hz`, 20 bins) with its standard error se (residual sd over n - 2 degrees of freedom). The seed is
"not converged" only when both hold: |b| >= 0.01 Hz/s, and b differs from 0 by a two-sided t test,
|b| / se > t(0.975, 18) = 2.101. Otherwise it is converged. A non-converged scored seed is recorded with b and se,
and R-E is off for the lane (v3 item 3). On the pilots: base b -0.0019 (se 0.0079) converged; rho0 6 Hz b -0.0157
(se 0.0097), |b|/se 1.62, converged under v4 (not converged under v3); eta 0 b +0.0066 (se 0.0103) converged.

## Estimands and criteria

| Id | z (jaxfne) | criterion |
|---|---|---|
| CONV | slope b of the per-second E rate, 130-150 s, with se | every scored seed converged (rule above) |
| R-E | E mean rate, 149-150 s | e = \|z - 3 Hz\| / 3 Hz <= 0.20 (proposal, v2) |
| O-DROP | E mean rate at 0.8-1.0 s over R-E's z | ratio >= 2 (proposal, v2) |
| W-IE, AI (secondary) | as v3 | reported, never scored |

The lane closes when CONV, R-E and O-DROP hold; any one off makes the result off, recorded with its value. Seeds,
split, sensitivity set, pilot record: v3.

## Owner decision before signature

Sign as written, or defer. Signed as written (above).
