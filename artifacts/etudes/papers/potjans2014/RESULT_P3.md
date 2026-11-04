# potjans2014 P3 result under signed v2

Verdict: **off**. R-L4E, R-L6E and O-ORD hold; R-L23E, R-L5E, O-IE (layer 5) and CV (L6E, L6PV) do not
(`prereg/P3_v2_signed.md`, signed by Hamm 2026-11-02). The B8 pilot predicted the same four rows off
(`CALIBRATION.md` section 0b).

Runs: `potjans2014_api.py score --seed S --signed`, seeds 4–8, scale 0.1 (7 717 cells, in-degree held),
T 10 s at dt 0.1 ms in 1 s segments, first 0.5 s dropped, float32 on the GPU, code SHA
b23c9b491f7b4877759fcd6611e9fd34e5140307 from a frozen clone; 23–25 min and 12.5 GB peak RSS per seed, each exit 0.
Receipts `F:/warehouse/potjans2014/scored_P3v2_b23c9b49/` (`scored/seed<S>/manifest.json`, prereg sha256
a18d698a… recorded in each; `score_p3.json` sha256 bf82fc66af567d39…, inputs sha256 f825c6d9…). The seed sets
the connectivity draw and the Poisson drive; every rate differs across seeds (values below).

| Id | y (paper) | Seeds 4–8, Hz | z (seed mean), Hz | e | Criterion | Pass |
|---|---|---|---|---|---|---|
| R-L23E | 0.86 | 1.487, 1.521, 1.610, 1.501, 1.423 | 1.508 | 0.754 | e ≤ 0.20 | no |
| R-L4E | 4.45 | 4.439, 4.492, 4.396, 4.498, 4.511 | 4.467 | 0.004 | e ≤ 0.20 | yes |
| R-L5E | 7.59 | 9.592, 10.200, 9.199, 10.082, 10.428 | 9.900 | 0.304 | e ≤ 0.20 | no |
| R-L6E | 1.09 | 1.122, 1.180, 1.063, 1.139, 1.187 | 1.138 | 0.044 | e ≤ 0.20 | yes |

| Id | Criterion | z (seed means) | Pass |
|---|---|---|---|
| O-IE | I rate above E rate in every layer | L2/3 3.92 > 1.51, L4 6.29 > 4.47, L5 9.66 < 9.90, L6 8.47 > 1.14 | no (L5) |
| O-ORD | L5e highest E rate; L2/3e and L6e below L4e | 9.90 highest; 1.51 and 1.14 below 4.47 | yes |
| CV | mean CV of ISIs above 0.8 in all 8 populations | L6E 0.797, L6PV 0.797; the other six 0.814–0.882 | no |
| SYN (secondary, reported) | synchrony highest in L5e, lowest in L6 | Fano (3 ms): L5E 0.970 lowest, L6E 0.996 highest | not scored |

O-IE at layer 5 fails on the seed means (PV 9.66 against E 9.90 Hz); per seed, PV is below E on seeds 4 (by 0.004 Hz),
5, 7 and 8 and above it on seed 6. CV at L6 sits 0.003 under its bound on the seed mean and crosses it on
seeds 5–8 in at least one L6 population.
