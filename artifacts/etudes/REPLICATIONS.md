# Replication index

One row per scored preregistration (release plan 0.5.6–0.5.10, §5). Verdicts stay in,
pass or fail (B2). Ledgers and results live on the lane branch until it merges to dev.

| Release | Paper, figure | Tier | Stage | Verdict | Ledger and result | Receipts | Gallery |
|---|---|---|---|---|---|---|---|
| 0.5.6 | N'dri et al. 2025, PCL Fig. 3 | T-port | P5, prereg v5 | not reproduced: v5 T2 (sf curve vs digitized Fig. 3) fails both sets, T1, T3, T4 pass; v4 FAIL stands | `pcl-port` 8f75acd0: `artifacts/etudes/pcl/LEDGER.md`, `prereg/P3_fig3_v5_result.md`, `prereg/P3_fig3_result.md`, `prereg/P3_fig3_diagnostic.md` | `F:\warehouse\pcl\scores\` | none yet (P7) |
| 0.5.6 | Tahvili, Vinck, di Volo 2026, CAMINOS Fig. 1 | T-methods | P5, prereg v6 + amendment v7 (R4) | reproduced under v7: R1–R3 pass (v6); R4 fails under v6 (every p from 0.02 oscillates) and passes under v7 (drive p·NE, onset at p = 0.04; p = 0.07 counts equal v6 in all five seeds). Library path P6 v2 (0.5.7, `caminos-p6` 5ed67635, `RESULT_P6.md`): PASS, Q1–Q5 and C1 (f_pk 33.8 Hz, lag 7.22 ms, rates within margins C, onset p = 0.04) | `l2-caminos` 1e768f9d: `artifacts/etudes/caminos/LEDGER.md`, `RESULT_P3.md` | `F:\warehouse\caminos\scored\`, `…\v7_20261012\` | none yet (P7) |
| 0.5.7 | Izhikevich 2003, Fig. 2 and 3 | T-port | P5, prereg v7 (signed 2026-10-13) | not reproduced: Fig. 2 C4 (FS ISI ratio 1.86) and C8 (RZ spikes once) fail, 6 of 8 pass; Fig. 3 N1, N3, N4 pass, N2 fails (median ISI CV 0.40, band 0.6–1.5; 0.39 on the S1 reference arm too). Amendment v8.2 (post hoc, signed 2026-10-13): Fig. 2 passes 8 of 8 (C8 pulse I = 2; C4 as the least-adapting type, FS 1.86 against RS 2.53 and LTS 4.39); N2 stays a failed paper claim (`net.m` gives 0.39). P3b (`net.m` update order) stopped at its equivalence gate | `izh2003` 34c38273: `artifacts/etudes/izhikevich2003/RESULT_P3.md`, `scores/p3_v7_score.json`, `scores/p3_v7_s1_report.json`; `scores/p3_v8_score.json` | `F:\warehouse\izh2003\scored_v7\`, `F:\warehouse\izh2003\v8\` | none yet (P7) |

Decisions (Hamm 2026-10-13):

- PCL Fig. 3 stays in progress (Hamm 2026-10-13: it must replicate). The v4 FAIL stands as
  recorded; the next attempt is a new, separately declared protocol, and the P6 library path
  follows (B1). v5 (signed 2026-10-13) also fails, on T2.
- Izhikevich 2003 ships unscored in 0.5.6: the P3 prereg is drafted and calibrated, not signed.
  Signature, S1 reference arm and scored runs move to 0.5.7.
- CAMINOS P6 (library path) moves to 0.5.7.
- Later the same day: CAMINOS `PREREG_P6.md` v2 signed (margins C, B0 route (c)) and Izhikevich
  2003 `PREREG_P3.md` v7 signed; both run in opencode lanes under WSL, scored results land in
  0.5.7.

B1: CAMINOS's library-path attempt is P6 v2 (PASS, 2026-10-13); PCL has none yet.

## Queue

| Paper | Tier | Stage | Note |
|---|---|---|---|
| Nejat, Sherfey, Bastos 2026, PLoS One 21(8): e0354021, "A genetic algorithm for self-supervised models of oscillatory neurodynamics" (GSDR) | T-jaxfne from the start (Hamm 2026-10-13: "just do it with jaxfne") | P0 | 50-cell HH E/PV/SST circuit, GSDR (stochastic delta rule plus MCDP plus genetic revert) on recurrent conductances; Figs. 2–4 and 6 are model claims, Fig. 5 is macaque MT/MST data. Code public (`HNXJ/GSDR`, DynaSim `devDL`), read as data only |
