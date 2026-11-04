# vogels2011 P3 result under signed v4

Verdict: **off (R-E)**. CONV and O-DROP hold; R-E does not: the E rate settles at 4.13 Hz against the paper's
3 Hz, e 0.375 over the 0.20 band (`prereg/P3_v4_signed.md`, signed by Hamm 2026-11-02). The B8 pilot predicted
this row off (`CALIBRATION.md`, B8 T 150 s pilots: base converged at 4.11 Hz, e 0.41).

Runs: `vogels2011_api.py score --seed S --signed`, seeds 4–8, n 10 000, 1 s static then 150 s plastic in 5 s
segments, default rho0 and eta, float32 on the GPU, code SHA b23c9b491f7b4877759fcd6611e9fd34e5140307 from a
frozen clone; 14–34 min and 5.4 GB peak RSS per seed, each exit 0. The manifests at this SHA label the backend
`cpu`, a fixed string fixed at 9d75412e; the runs used the GPU (the runner sets `JAX_PLATFORMS=cuda`, and the GPU
held 15.6 GB at 100 % while this queue and the Potjans queue ran; inferred). Receipts
`F:/warehouse/vogels2011/scored_P3v4_b23c9b49/` (`scored/seed<S>/score_receipt.json`, prereg sha256 bb771943…
recorded in each; `score_p3.json` sha256 a427a6122a0fe610…). The seed sets the connectivity draw and the drive;
every value differs across seeds (below).

| Id | Seeds 4–8 | z | Criterion | Pass |
|---|---|---|---|---|
| CONV | b (Hz/s) −0.0028, −0.0016, −0.0142, −0.0033, +0.0011; \|b\|/se 0.32, 0.37, 1.90, 0.74, 0.21 | all converged | not converged only when \|b\| ≥ 0.01 and \|b\|/se > 2.101 | yes |
| R-E | 4.190, 4.180, 4.041, 4.139, 4.080 Hz | 4.126 Hz, e 0.375 | e ≤ 0.20 | no |
| O-DROP | E rate 0.8–1.0 s: 160.29, 160.32, 160.44, 160.61, 160.84 Hz | 160.50 / 4.126 = 38.9 | ratio ≥ 2 | yes |
| W-IE (secondary) | 1.79, 1.87, 1.82, 1.83, 1.90 nS | 1.84 nS | reported | not scored |
| AI (secondary) | CV 0.65–0.67, κ 0.022–0.024; labels irregular (4, 7, 8), oscillatory (5, 6) | — | reported | not scored |

Seed 6 is the one seed with |b| ≥ 0.01 Hz/s; its slope does not differ from 0 (|b|/se 1.90 < 2.101), so it is
converged under v4 and would be not converged under v3's magnitude-only rule.
