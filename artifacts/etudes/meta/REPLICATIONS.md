# Replication index

One row per scored preregistration (release plan 0.5.6–0.5.10, §5). Verdicts stay in,
pass or fail (B2). Ledgers and results live on the lane branch until it merges to dev.

| Release | Paper, figure | Tier | Stage | Verdict | Ledger and result | Receipts | Gallery |
|---|---|---|---|---|---|---|---|
| 0.5.6 | N'dri et al. 2025, PCL Fig. 3 | T-port | P5, prereg v5 | not reproduced: v5 T2 (sf curve vs digitized Fig. 3) fails both sets, T1, T3, T4 pass; v4 FAIL stands | `pcl-port` 8f75acd0: `artifacts/etudes/pcl/LEDGER.md`, `prereg/P3_fig3_v5_result.md`, `prereg/P3_fig3_result.md`, `prereg/P3_fig3_diagnostic.md` | `F:\warehouse\pcl\scores\` | none yet (P7) |
| 0.5.6 | Tahvili, Vinck, di Volo 2026, CAMINOS Fig. 1 | T-methods | P5, prereg v6 + amendment v7 (R4); P6 v2 library path through the public API | mixed (Hamm 2026-10-14; primary 2 ms model, `CALIBRATION.md` §10): reproduced f_pk 33.80 Hz (e 0.034), E rate inside 1–3 Hz, PV/SOM/E rate order, R4 onset p = 0.04 after the v7 fidelity fix; finding that needs X: PV→SOM lag 6.47 ms on seeds 1–3 (e 0.239), 6.32 ms on seeds 4–8, P3 estimator 7.22 ms, all off against 8.5 ms; recorded sensitivity: 0.1 ms delay gives the preprint ≈ 7 ms (e ≤ 0.014) with f_pk 43–46 Hz | `l2-caminos` 1e768f9d: `artifacts/etudes/caminos/LEDGER.md`, `RESULT_P3.md`, `CALIBRATION.md` | `F:\warehouse\caminos\scored\`, `…\v7_20261012\`, `F:\warehouse\caminos\p6_scored_v2\` | none yet (P7) |
| 0.5.7 | Izhikevich 2003, Fig. 2 and 3 | T-port | P5, prereg v7 (signed 2026-10-13) | not reproduced: Fig. 2 C4 (FS ISI ratio 1.86) and C8 (RZ spikes once) fail, 6 of 8 pass; Fig. 3 N1, N3, N4 pass, N2 fails (median ISI CV 0.40, band 0.6–1.5; 0.39 on the S1 reference arm too). Amendment v8.2 (post hoc, signed 2026-10-13): Fig. 2 passes 8 of 8 (C8 pulse I = 2; C4 as the least-adapting type, FS 1.86 against RS 2.53 and LTS 4.39); N2 stays a failed paper claim (`net.m` gives 0.39). P3b (`net.m` update order) stopped at its equivalence gate | `izh2003` 34c38273: `artifacts/etudes/izhikevich2003/RESULT_P3.md`, `scores/p3_v7_score.json`, `scores/p3_v7_s1_report.json`; `scores/p3_v8_score.json` | `F:\warehouse\izh2003\scored_v7\`, `F:\warehouse\izh2003\v8\` | none yet (P7) |

| 0.5.7 | Destexhe 2009, adaptive exponential integrate-and-fire model | T-port | S2 queued; migration and fidelity done in the lane clone (lane tests 28 passed) | in progress, ships in 0.5.7 (no verdict yet) | lane 5d (unmerged) | — | — |

Decisions (Hamm 2026-10-13; CAMINOS ruling 2026-10-14):

- PCL Fig. 3 stays in progress (Hamm 2026-10-13: it must replicate). The v4 FAIL stands as
  recorded; v5 (signed 2026-10-13) also fails, on T2. The next attempt is a new,
  separately declared protocol, and the P6 library path follows (B1).
- CAMINOS final (Hamm 2026-10-14, `CALIBRATION.md` §10): PLoS CB is primary; f_pk,
  rates and R4 reproduced; the PV→SOM lag miss is a finding that needs X; no residual
  fit ran; the 0.1 ms preprint sensitivity is recorded.
- Izhikevich 2003 scored in 0.5.7: not reproduced (Fig. 2 C4/C8, Fig. 3 N2; post hoc
  amendment v8.2; P3b stopped at its equivalence gate).
- GSDR at P1 (source record) and Destexhe 2009 with S2 queued: both in progress,
  ship in 0.5.7.

B1: library-path attempts: CAMINOS P6 v2 ran through the public API (mixed verdict per §10); PCL has none yet.

## Queue

| Paper | Tier | Stage | Note |
|---|---|---|---|
| Nejat, Sherfey, Bastos 2026, PLoS One 21(8): e0354021, "A genetic algorithm for self-supervised models of oscillatory neurodynamics" (GSDR) | T-jaxfne from the start (Hamm 2026-10-13: "just do it with jaxfne") | P1 | 50-cell HH E/PV/SST circuit, GSDR (stochastic delta rule plus MCDP plus genetic revert) on recurrent conductances; Figs. 2–4 and 6 are model claims, Fig. 5 is macaque MT/MST data. Code public (`HNXJ/GSDR`, DynaSim `devDL`), read as data only. P1 source record: `artifacts/etudes/gsdr2026/SOURCE_P1.md`. In progress, ships in 0.5.7 |
